"""순수 수치 분석: prediction 수집, attention 지표, bootstrap 구간, 계산 비용 추정.

Evaluation 모듈이 이 함수들을 조합한다. Config를 모르는 numpy/torch leaf 모듈이다.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from .metrics import finite_mean


# -----------------------------------------------------------------------------
# Prediction 수집
# -----------------------------------------------------------------------------


@torch.no_grad()
def collect_prediction_arrays(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    permute_attention: bool = False,
) -> dict[str, np.ndarray]:
    """Dataset 순서를 유지하며 logit, label, pair metadata를 NumPy array로 수집한다."""
    model.eval()
    collected: dict[str, list[np.ndarray]] = {
        "logits": [],
        "labels": [],
        "sample_id": [],
        "pair_id": [],
        "label_first": [],
        "label_second": [],
        "bounding_box_overlap_ratio": [],
        "pixel_overlap_ratio": [],
        "overlap_level": [],
    }

    for batch in data_loader:
        images = batch["image"].to(device)

        if permute_attention:
            if not hasattr(model, "forward_with_permuted_attention"):
                raise TypeError(
                    "Attention permutation은 class_attention 모델만 지원합니다."
                )

            logits = model.forward_with_permuted_attention(images, shift=1)
        else:
            logits = model(images)

        collected["logits"].append(logits.cpu().numpy())
        collected["labels"].append(batch["label"].numpy())

        for field_name in (
            "sample_id",
            "pair_id",
            "label_first",
            "label_second",
            "bounding_box_overlap_ratio",
            "pixel_overlap_ratio",
        ):
            collected[field_name].append(batch[field_name].numpy())

        collected["overlap_level"].append(np.asarray(batch["overlap_level"]))

    return {
        field_name: np.concatenate(values)
        for field_name, values in collected.items()
    }


# -----------------------------------------------------------------------------
# Attention 분석
# -----------------------------------------------------------------------------


@torch.no_grad()
def collect_attention_metric_arrays(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    thresholds: Iterable[float],
    minimum_exclusive_pixels: int,
    compute_alignment: bool = True,
) -> dict[str, np.ndarray]:
    """Attention map을 저장하지 않고 sample별 AUPRC, cross-selectivity,
    `[threshold, sample]` IoU를 수집한다."""
    model.eval()
    threshold_values = np.asarray(list(thresholds), dtype=np.float64)
    collected: dict[str, list[np.ndarray]] = {
        "sample_id": [],
        "pair_id": [],
        "overlap_level": [],
        "average_precision": [],
        "cross_selectivity": [],
        "iou_by_threshold": [],
    }

    for batch in data_loader:
        images = batch["image"].to(device)
        _, attention_maps = model(images, return_attention=True)
        if attention_maps is None:
            raise TypeError("Attention 정렬 평가는 attention 모델이 필요합니다.")

        cpu_attention_maps = attention_maps.cpu()
        batch_metrics = _attention_sample_metrics(
            cpu_attention_maps,
            batch,
            threshold_values,
            minimum_exclusive_pixels,
            compute_alignment,
        )
        collected["sample_id"].append(batch["sample_id"].numpy())
        collected["pair_id"].append(batch["pair_id"].numpy())
        collected["overlap_level"].append(np.asarray(batch["overlap_level"]))
        for field_name in (
            "average_precision",
            "cross_selectivity",
            "iou_by_threshold",
        ):
            collected[field_name].append(batch_metrics[field_name])

    return {
        "sample_id": np.concatenate(collected["sample_id"]),
        "pair_id": np.concatenate(collected["pair_id"]),
        "overlap_level": np.concatenate(collected["overlap_level"]),
        "average_precision": np.concatenate(collected["average_precision"]),
        "cross_selectivity": np.concatenate(collected["cross_selectivity"]),
        "iou_by_threshold": np.concatenate(
            collected["iou_by_threshold"],
            axis=1,
        ),
        "thresholds": threshold_values,
    }


def _attention_sample_metrics(
    attention_maps: torch.Tensor,
    batch: dict[str, Any],
    thresholds: np.ndarray,
    minimum_exclusive_pixels: int,
    compute_alignment: bool,
) -> dict[str, np.ndarray]:
    """한 batch의 attention metric을 sample 순서의 NumPy array로 계산한다."""
    positive_maps, target_masks = _positive_attention_maps_and_masks(
        attention_maps,
        batch,
    )
    sample_count = attention_maps.shape[0]
    average_precision = np.full(sample_count, np.nan, dtype=np.float64)
    cross_selectivity = np.full(sample_count, np.nan, dtype=np.float64)

    if compute_alignment:
        for sample_index in range(sample_count):
            digit_scores = []
            for digit_index in range(2):
                digit_scores.append(average_precision_score(
                    positive_maps[sample_index, digit_index].numpy(),
                    target_masks[sample_index, digit_index].numpy(),
                ))
            average_precision[sample_index] = finite_mean(
                np.asarray(digit_scores, dtype=np.float64)
            )

        if attention_maps.shape[1] > 1:
            cross_selectivity = _sample_cross_selectivity(
                attention_maps,
                batch,
                minimum_exclusive_pixels,
            )

    iou_by_threshold = np.empty(
        (len(thresholds), sample_count),
        dtype=np.float64,
    )
    for threshold_index, threshold in enumerate(thresholds):
        predicted_masks = positive_maps >= float(threshold)
        intersections = torch.logical_and(
            predicted_masks,
            target_masks,
        ).sum(dim=(-2, -1))
        unions = torch.logical_or(
            predicted_masks,
            target_masks,
        ).sum(dim=(-2, -1))
        digit_iou = torch.where(
            unions > 0,
            intersections.float() / unions.clamp_min(1).float(),
            torch.zeros_like(unions, dtype=torch.float32),
        )
        iou_by_threshold[threshold_index] = digit_iou.mean(dim=1).numpy()

    return {
        "average_precision": average_precision,
        "cross_selectivity": cross_selectivity,
        "iou_by_threshold": iou_by_threshold,
    }


def _positive_attention_maps_and_masks(
    attention_maps: torch.Tensor,
    batch: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """두 정답 class의 attention map과 stroke mask를 `[batch, 2, H, W]`로 대응시킨다.

    Shared map(channel 1개)은 두 번 사용하고, class map은 정답 label channel만 gather한다.
    """
    output_size = attention_maps.shape[-2:]
    target_masks = torch.stack(
        (
            resize_masks(batch["mask_first"], output_size)[:, 0],
            resize_masks(batch["mask_second"], output_size)[:, 0],
        ),
        dim=1,
    )

    if attention_maps.shape[1] == 1:
        return attention_maps.expand(-1, 2, -1, -1), target_masks

    positive_labels = torch.stack(
        (batch["label_first"], batch["label_second"]),
        dim=1,
    )
    gather_indices = positive_labels[:, :, None, None].expand(
        -1,
        -1,
        output_size[0],
        output_size[1],
    )
    return torch.gather(attention_maps, dim=1, index=gather_indices), target_masks


def _sample_cross_selectivity(
    attention_maps: torch.Tensor,
    batch: dict[str, Any],
    minimum_exclusive_pixels: int,
) -> np.ndarray:
    """Class attention의 고유 획 영역 selectivity를 sample별로 계산한다.

    두 exclusive 영역 중 하나라도 최소 pixel 수 미만이면 해당 sample은 NaN이다.
    """
    output_size = attention_maps.shape[-2:]
    exclusive_first = resize_masks(batch["exclusive_mask_first"], output_size)
    exclusive_second = resize_masks(batch["exclusive_mask_second"], output_size)
    values = np.full(attention_maps.shape[0], np.nan, dtype=np.float64)

    for sample_index in range(attention_maps.shape[0]):
        first_region = exclusive_first[sample_index, 0]
        second_region = exclusive_second[sample_index, 0]
        regions_are_valid = (
            int(first_region.sum()) >= minimum_exclusive_pixels
            and int(second_region.sum()) >= minimum_exclusive_pixels
        )
        if not regions_are_valid:
            continue

        first_class = int(batch["label_first"][sample_index])
        second_class = int(batch["label_second"][sample_index])
        first_map = attention_maps[sample_index, first_class]
        second_map = attention_maps[sample_index, second_class]
        selectivity = 0.5 * (
            (first_map[first_region] - second_map[first_region]).mean()
            + (second_map[second_region] - first_map[second_region]).mean()
        )
        values[sample_index] = float(selectivity.item())

    return values


def average_precision_score(scores: np.ndarray, targets: np.ndarray) -> float:
    """Binary stroke mask에 대한 average precision을 계산한다. Positive가 없으면 NaN."""
    boolean_targets = targets.astype(bool).reshape(-1)
    flattened_scores = scores.reshape(-1)
    positive_count = int(boolean_targets.sum())

    if positive_count == 0:
        return float("nan")

    order = np.argsort(-flattened_scores, kind="stable")
    sorted_targets = boolean_targets[order]
    precision_at_rank = (
        np.cumsum(sorted_targets)
        / np.arange(1, len(sorted_targets) + 1)
    )
    return float(precision_at_rank[sorted_targets].sum() / positive_count)


def resize_masks(
    masks: torch.Tensor,
    output_size: tuple[int, int],
) -> torch.Tensor:
    """Binary mask를 nearest-neighbor로 attention 해상도에 맞춘다."""
    return functional.interpolate(
        masks.float(),
        size=output_size,
        mode="nearest",
    ).bool()


# -----------------------------------------------------------------------------
# Paired bootstrap
# -----------------------------------------------------------------------------


def hierarchical_bootstrap_interval(
    values_by_seed_and_pair: np.ndarray,
    iterations: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float, float]:
    """Training seed와 test pair를 함께 복원추출해 평균의 percentile 구간을 계산한다.

    Pair draw는 선택된 모든 seed에 공통 적용해 paired 설계를 유지한다.
    """
    values = np.asarray(values_by_seed_and_pair, dtype=np.float64)
    matrix_is_valid = values.ndim == 2 and all(size > 0 for size in values.shape)
    if not matrix_is_valid:
        raise ValueError("Hierarchical bootstrap 입력은 비어 있지 않은 2차원 행렬이어야 합니다.")
    if not np.isfinite(values).all():
        raise ValueError("Hierarchical bootstrap 입력에는 유한한 값만 사용할 수 있습니다.")

    seed_count, pair_count = values.shape
    random_generator = np.random.default_rng(seed)
    bootstrap_means = np.empty(iterations, dtype=np.float64)

    for iteration in range(iterations):
        sampled_seed_indices = random_generator.integers(
            0,
            seed_count,
            size=seed_count,
        )
        sampled_pair_indices = random_generator.integers(
            0,
            pair_count,
            size=pair_count,
        )
        # Seed resampling을 먼저 평균하면 동일한 2단계 표본을 훨씬 작은 array로 계산한다.
        seed_resampled_pair_means = values[sampled_seed_indices].mean(axis=0)
        bootstrap_means[iteration] = seed_resampled_pair_means[
            sampled_pair_indices
        ].mean()

    tail_probability = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(
        bootstrap_means,
        [tail_probability, 1.0 - tail_probability],
    )
    return float(values.mean()), float(lower), float(upper)


def paired_bootstrap_interval(
    differences: np.ndarray,
    pair_ids: np.ndarray,
    iterations: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float, float]:
    """Pair별 평균을 만든 뒤 pair 단위 bootstrap으로 평균 차이 구간을 계산한다."""
    unique_pair_ids = np.unique(pair_ids)
    pair_means = np.asarray([
        differences[pair_ids == pair_id].mean()
        for pair_id in unique_pair_ids
    ])
    random_generator = np.random.default_rng(seed)

    # 5,000×10,000 index 행렬을 한 번에 만들지 않도록 resampling을 나눈다.
    bootstrap_means = np.empty(iterations, dtype=np.float64)
    chunk_size = min(100, iterations)

    for start in range(0, iterations, chunk_size):
        stop = min(start + chunk_size, iterations)
        sampled_indices = random_generator.integers(
            0,
            len(pair_means),
            size=(stop - start, len(pair_means)),
        )
        bootstrap_means[start:stop] = pair_means[sampled_indices].mean(axis=1)

    tail_probability = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(
        bootstrap_means,
        [tail_probability, 1.0 - tail_probability],
    )
    return float(pair_means.mean()), float(lower), float(upper)


def paired_level_difference(
    correctness: np.ndarray,
    pair_ids: np.ndarray,
    overlap_levels: np.ndarray,
    first_level: str,
    second_level: str,
) -> tuple[np.ndarray, np.ndarray]:
    """같은 pair의 두 overlap level correctness 차이(first − second)를 계산한다."""
    first_mask = overlap_levels == first_level
    second_mask = overlap_levels == second_level
    first_pair_ids = pair_ids[first_mask]
    second_pair_ids = pair_ids[second_mask]
    first_values = correctness[first_mask]
    second_values = correctness[second_mask]

    common_pair_ids, first_indices, second_indices = np.intersect1d(
        first_pair_ids,
        second_pair_ids,
        assume_unique=True,
        return_indices=True,
    )
    differences = first_values[first_indices] - second_values[second_indices]
    return differences.astype(np.float64), common_pair_ids.astype(np.int64)


def difference_in_differences(
    first_correctness: np.ndarray,
    second_correctness: np.ndarray,
    pair_ids: np.ndarray,
    overlap_levels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """두 모델 간 개선 폭의 High−Low 차이를 pair별로 계산한다."""
    model_difference = first_correctness - second_correctness
    return paired_level_difference(
        correctness=model_difference,
        pair_ids=pair_ids,
        overlap_levels=overlap_levels,
        first_level="high",
        second_level="low",
    )


def create_seed_effect_rows(
    correctness_by_run: dict[tuple[str, int], np.ndarray],
    pair_ids: np.ndarray,
    overlap_levels: np.ndarray,
    seeds: list[int],
    iterations: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    """Seed별 모델 비교 추정값과 test-pair bootstrap 구간(`seed_effects.csv` row)을 만든다."""
    rows = []
    high_mask = overlap_levels == "high"

    for seed in seeds:
        lenet_correctness = correctness_by_run[("lenet", seed)]
        shared_correctness = correctness_by_run[("shared_attention", seed)]
        class_correctness = correctness_by_run[("class_attention", seed)]
        comparison_values = []

        lenet_drop, lenet_drop_pair_ids = paired_level_difference(
            correctness=lenet_correctness,
            pair_ids=pair_ids,
            overlap_levels=overlap_levels,
            first_level="low",
            second_level="high",
        )
        comparison_values.append(
            ("lenet_low_minus_high", lenet_drop, lenet_drop_pair_ids)
        )
        comparison_values.extend(
            (
                (
                    "class_attention_minus_lenet_high",
                    class_correctness[high_mask] - lenet_correctness[high_mask],
                    pair_ids[high_mask],
                ),
                (
                    "class_attention_minus_shared_high",
                    class_correctness[high_mask] - shared_correctness[high_mask],
                    pair_ids[high_mask],
                ),
            )
        )
        relative_improvement, relative_pair_ids = difference_in_differences(
            class_correctness,
            lenet_correctness,
            pair_ids,
            overlap_levels,
        )
        comparison_values.append(
            (
                "class_attention_vs_lenet_high_low_difference",
                relative_improvement,
                relative_pair_ids,
            )
        )

        for comparison_index, (
            comparison_name,
            differences,
            comparison_pair_ids,
        ) in enumerate(comparison_values):
            estimate, lower, upper = paired_bootstrap_interval(
                differences=differences,
                pair_ids=comparison_pair_ids,
                iterations=iterations,
                confidence_level=confidence_level,
                seed=bootstrap_seed + seed * 10 + comparison_index,
            )
            rows.append(
                {
                    "comparison": comparison_name,
                    "seed": seed,
                    "estimate": estimate,
                    "confidence_lower": lower,
                    "confidence_upper": upper,
                }
            )

    return rows


# -----------------------------------------------------------------------------
# 모델 계산 비용
# -----------------------------------------------------------------------------


def count_parameters(model: nn.Module) -> int:
    """모델의 trainable parameter 수를 계산한다."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def estimate_multiply_accumulate_operations(
    model: nn.Module,
    input_shape: tuple[int, int, int, int],
) -> int:
    """Forward hook으로 Conv2d·Linear output shape를 관찰해 sample당 MAC을 추정한다."""
    operation_counts: list[int] = []
    hooks = []

    def count_convolution(
        module: nn.Conv2d,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        """Conv2d output element 수 × kernel 연산 수를 누적한다."""
        del inputs
        batch_size, output_channels, output_height, output_width = output.shape
        kernel_operations = (
            module.kernel_size[0]
            * module.kernel_size[1]
            * module.in_channels
            // module.groups
        )
        operation_counts.append(
            batch_size
            * output_channels
            * output_height
            * output_width
            * kernel_operations
        )

    def count_linear(
        module: nn.Linear,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        """적용 vector 수 × input/output feature 수를 누적한다."""
        del output
        applied_vector_count = inputs[0].numel() // module.in_features
        operation_counts.append(
            applied_vector_count
            * module.in_features
            * module.out_features
        )

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(count_convolution))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(count_linear))

    model.eval()

    with torch.no_grad():
        model(torch.zeros(input_shape))

    for hook in hooks:
        hook.remove()

    total_operations = sum(operation_counts)

    # Class attention의 scalar output head는 functional 호출이라 hook 대상이 아니다.
    if hasattr(model, "class_count") and hasattr(model, "output_head"):
        total_operations += model.output_head.weight.numel() * input_shape[0]

    return int(total_operations / input_shape[0])
