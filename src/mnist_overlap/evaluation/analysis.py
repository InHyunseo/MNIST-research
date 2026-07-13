"""Prediction 수집과 attention·통계·계산 비용 분석을 수행한다.

입력:
    학습된 model, DataLoader, source mask, prediction array, bootstrap 설정

출력:
    Prediction array, attention 지표, confidence interval, parameter와 MAC 수

연결:
    Evaluation runner가 이 모듈의 분석 함수를 조합하고 metrics 모듈이 분류 지표를
    별도로 계산한다.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn


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
    """Dataset 순서를 유지하며 logit, label, pair metadata를 수집한다.

    입력:
        학습된 모델, DataLoader, device, class attention permutation 여부

    처리:
        Batch forward 결과와 metadata를 CPU NumPy array로 이어 붙인다.

    출력:
        Prediction과 manifest identifier를 담은 array dictionary
    """
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
def evaluate_attention_loader(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    thresholds: Iterable[float],
    minimum_exclusive_pixels: int,
    compute_alignment: bool = True,
) -> dict[str, Any]:
    """DataLoader 한 번의 순회로 attention 정렬 지표와 IoU를 계산한다.

    입력:
        Attention 모델, mask 포함 DataLoader, threshold와 exclusive pixel 기준

    처리:
        Positive class map의 AUPRC, cross-selectivity, threshold별 IoU를 집계한다.

    출력:
        Sample metric array, 유효 비율, IoU sum/count dictionary
    """
    model.eval()
    average_precision_values: list[float] = []
    cross_selectivity_values: list[float] = []
    combined_iou_counts = {
        float(threshold): (0.0, 0)
        for threshold in thresholds
    }
    sample_count = 0

    for batch in data_loader:
        images = batch["image"].to(device)
        _, attention_maps = model(images, return_attention=True)

        if attention_maps is None:
            raise TypeError("Attention 정렬 평가는 attention 모델이 필요합니다.")

        cpu_attention_maps = attention_maps.cpu()

        if compute_alignment:
            alignment = attention_alignment_values(
                cpu_attention_maps,
                batch,
                minimum_exclusive_pixels,
            )
            average_precision_values.extend(alignment["average_precision"])
            cross_selectivity_values.extend(alignment["cross_selectivity"])

        sample_count += images.shape[0]
        batch_counts = attention_iou_counts(
            cpu_attention_maps,
            batch,
            thresholds,
        )

        for threshold, (iou_sum, count) in batch_counts.items():
            current_sum, current_count = combined_iou_counts[threshold]
            combined_iou_counts[threshold] = (
                current_sum + iou_sum,
                current_count + count,
            )

    return {
        "average_precision": np.asarray(
            average_precision_values,
            dtype=np.float64,
        ),
        "cross_selectivity": np.asarray(
            cross_selectivity_values,
            dtype=np.float64,
        ),
        "selectivity_valid_fraction": (
            len(cross_selectivity_values) / sample_count
            if sample_count
            else 0.0
        ),
        "iou_counts": combined_iou_counts,
    }


def average_precision_score(scores: np.ndarray, targets: np.ndarray) -> float:
    """Binary stroke mask에 대한 average precision을 계산한다.

    입력:
        연속 attention score와 같은 shape의 binary stroke target

    처리:
        Score 내림차순에서 positive가 나타난 rank의 precision을 평균한다.

    출력:
        Positive가 없으면 NaN, 그렇지 않으면 `[0, 1]` average precision
    """
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
    """입력 공간의 binary mask를 attention 해상도로 변경한다.

    입력:
        `[batch, 1, height, width]` mask와 목표 `(height, width)`

    처리:
        Nearest-neighbor interpolation으로 binary 경계를 보존한다.

    출력:
        목표 해상도의 Boolean mask tensor
    """
    return functional.interpolate(
        masks.float(),
        size=output_size,
        mode="nearest",
    ).bool()


def attention_alignment_values(
    attention_maps: torch.Tensor,
    batch: dict[str, Any],
    minimum_exclusive_pixels: int,
) -> dict[str, list[float]]:
    """Positive class attention의 AUPRC와 cross-map selectivity를 계산한다.

    입력:
        Attention map, source/exclusive mask batch, 최소 exclusive pixel 수

    처리:
        두 정답 class map을 선택하고 전체 획 및 고유 획 기준으로 비교한다.

    출력:
        Sample별 average precision과 유효 cross-selectivity 목록
    """
    map_height, map_width = attention_maps.shape[-2:]
    output_size = (map_height, map_width)
    mask_first = resize_masks(batch["mask_first"], output_size)
    mask_second = resize_masks(batch["mask_second"], output_size)
    exclusive_first = resize_masks(batch["exclusive_mask_first"], output_size)
    exclusive_second = resize_masks(batch["exclusive_mask_second"], output_size)
    results: dict[str, list[float]] = {
        "average_precision": [],
        "cross_selectivity": [],
    }

    for sample_index in range(attention_maps.shape[0]):
        first_class = int(batch["label_first"][sample_index])
        second_class = int(batch["label_second"][sample_index])
        first_map = _select_attention_map(
            attention_maps,
            sample_index,
            first_class,
        )
        second_map = _select_attention_map(
            attention_maps,
            sample_index,
            second_class,
        )
        results["average_precision"].extend(
            (
                average_precision_score(
                    first_map.detach().cpu().numpy(),
                    mask_first[sample_index, 0].cpu().numpy(),
                ),
                average_precision_score(
                    second_map.detach().cpu().numpy(),
                    mask_second[sample_index, 0].cpu().numpy(),
                ),
            )
        )

        # Shared map은 class별 channel이 없으므로 cross-map selectivity를 정의하지 않는다.
        if attention_maps.shape[1] == 1:
            continue

        first_region = exclusive_first[sample_index, 0]
        second_region = exclusive_second[sample_index, 0]

        if int(first_region.sum()) < minimum_exclusive_pixels:
            continue

        if int(second_region.sum()) < minimum_exclusive_pixels:
            continue

        selectivity = 0.5 * (
            (first_map[first_region] - second_map[first_region]).mean()
            + (second_map[second_region] - first_map[second_region]).mean()
        )
        results["cross_selectivity"].append(float(selectivity.item()))

    return results


def attention_iou_counts(
    attention_maps: torch.Tensor,
    batch: dict[str, Any],
    thresholds: Iterable[float],
) -> dict[float, tuple[float, int]]:
    """두 positive class map의 threshold별 IoU 합과 개수를 계산한다.

    입력:
        Attention map, source mask batch, threshold iterable

    처리:
        Positive map을 batch 단위로 모아 threshold별 intersection/union을 계산한다.

    출력:
        Threshold를 key로 갖는 `(IoU 합, map 수)` dictionary
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
        positive_maps = attention_maps.expand(-1, 2, -1, -1)
    else:
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
        positive_maps = torch.gather(
            attention_maps,
            dim=1,
            index=gather_indices,
        )

    results = {
        float(threshold): (0.0, 0)
        for threshold in thresholds
    }

    for threshold in results:
        predicted_masks = positive_maps >= threshold
        intersections = torch.logical_and(
            predicted_masks,
            target_masks,
        ).sum(dim=(-2, -1))
        unions = torch.logical_or(
            predicted_masks,
            target_masks,
        ).sum(dim=(-2, -1))
        iou_values = torch.where(
            unions > 0,
            intersections.float() / unions.clamp_min(1).float(),
            torch.zeros_like(unions, dtype=torch.float32),
        )
        results[threshold] = (
            float(iou_values.sum().item()),
            int(iou_values.numel()),
        )

    return results


def select_best_iou_threshold(
    iou_counts: dict[float, tuple[float, int]],
) -> float:
    """Validation 평균 IoU가 가장 큰 threshold를 선택한다.

    입력:
        Threshold별 IoU 합과 count dictionary

    처리:
        각 평균을 비교하며 동률이면 정렬상 뒤 threshold를 선택한다.

    출력:
        선택된 float threshold
    """
    if not iou_counts:
        raise ValueError("iou_counts는 비어 있을 수 없습니다.")

    return max(
        sorted(iou_counts),
        key=lambda threshold: (
            iou_counts[threshold][0]
            / max(iou_counts[threshold][1], 1)
        ),
    )


def _select_attention_map(
    attention_maps: torch.Tensor,
    sample_index: int,
    class_index: int,
) -> torch.Tensor:
    """Shared 또는 class attention에서 한 sample의 positive map을 선택한다.

    입력:
        Attention tensor, sample index, class index

    처리:
        Channel이 하나면 shared map을, 여러 개면 class index map을 사용한다.

    출력:
        `[height, width]` attention map
    """
    map_index = 0 if attention_maps.shape[1] == 1 else class_index
    return attention_maps[sample_index, map_index]


# -----------------------------------------------------------------------------
# Paired bootstrap
# -----------------------------------------------------------------------------


def paired_bootstrap_interval(
    differences: np.ndarray,
    pair_ids: np.ndarray,
    iterations: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float, float]:
    """Pair 내부 row를 유지한 채 평균 차이의 bootstrap interval을 계산한다.

    입력:
        Sample 차이, pair ID, 반복 수, confidence level, seed

    처리:
        Pair별 평균을 만든 뒤 pair 단위로 chunked resampling을 수행한다.

    출력:
        원본 평균, confidence 하한, confidence 상한 tuple
    """
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
    """같은 pair의 두 overlap level correctness 차이를 계산한다.

    입력:
        Sample correctness, pair ID, overlap level, 차감 순서를 정하는 두 level 이름

    처리:
        Pair마다 first level 값에서 second level 값을 빼고 완전한 pair만 유지한다.

    출력:
        Pair별 correctness 차이와 대응 pair ID array
    """
    differences = []
    retained_pair_ids = []

    for pair_id in np.unique(pair_ids):
        pair_mask = pair_ids == pair_id
        first_values = correctness[pair_mask & (overlap_levels == first_level)]
        second_values = correctness[pair_mask & (overlap_levels == second_level)]

        if len(first_values) == 1 and len(second_values) == 1:
            differences.append(float(first_values[0] - second_values[0]))
            retained_pair_ids.append(int(pair_id))

    return np.asarray(differences), np.asarray(retained_pair_ids)


def difference_in_differences(
    first_correctness: np.ndarray,
    second_correctness: np.ndarray,
    pair_ids: np.ndarray,
    overlap_levels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """두 모델 간 개선 폭의 High–Low 차이를 pair별로 계산한다.

    입력:
        두 모델 correctness, pair ID, overlap level array

    처리:
        Pair별 모델 차이에서 High 값과 Low 값의 차이를 구한다.

    출력:
        Difference-in-differences 값과 대응 pair ID array
    """
    model_difference = first_correctness - second_correctness
    values = []
    retained_pair_ids = []

    for pair_id in np.unique(pair_ids):
        pair_mask = pair_ids == pair_id
        low_values = model_difference[pair_mask & (overlap_levels == "low")]
        high_values = model_difference[pair_mask & (overlap_levels == "high")]

        if len(low_values) == 1 and len(high_values) == 1:
            values.append(float(high_values[0] - low_values[0]))
            retained_pair_ids.append(int(pair_id))

    return np.asarray(values), np.asarray(retained_pair_ids)


def finite_mean(values: np.ndarray) -> float:
    """NaN과 무한대를 제외한 array 평균을 계산한다.

    입력:
        임의 shape의 숫자 array

    처리:
        유한 값만 Boolean mask로 선택해 평균한다.

    출력:
        유효 값이 없으면 NaN, 그렇지 않으면 float 평균
    """
    finite_values = values[np.isfinite(values)]
    return float(finite_values.mean()) if len(finite_values) else float("nan")


# -----------------------------------------------------------------------------
# 모델 계산 비용
# -----------------------------------------------------------------------------


def count_parameters(model: nn.Module) -> int:
    """모델에 등록된 전체 trainable parameter 수를 계산한다.

    입력:
        PyTorch model

    처리:
        Gradient를 학습하는 모든 parameter tensor의 element 수를 더한다.

    출력:
        정수 trainable parameter 수
    """
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def estimate_multiply_accumulate_operations(
    model: nn.Module,
    input_shape: tuple[int, int, int, int],
) -> int:
    """한 sample forward의 Conv2d 및 Linear MAC 수를 추정한다.

    입력:
        PyTorch model과 `[batch, channel, height, width]` input shape

    처리:
        Forward hook으로 layer output shape를 관찰해 multiply-accumulate를 합산한다.

    출력:
        Sample당 정수 MAC 추정값
    """
    operation_counts: list[int] = []
    hooks = []

    def count_convolution(
        module: nn.Conv2d,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        """Conv2d output shape에서 현재 layer의 MAC을 누적한다.

        입력:
            Conv2d module, 입력 tuple, 출력 tensor

        처리:
            Kernel 연산 수와 output element 수를 곱한다.

        출력:
            반환값은 없으며 외부 operation count가 변경된다.
        """
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
        """Linear input shape에서 현재 layer의 MAC을 누적한다.

        입력:
            Linear module, 입력 tuple, 사용하지 않는 출력 tensor

        처리:
            적용 vector 수와 input/output feature 수를 곱한다.

        출력:
            반환값은 없으며 외부 operation count가 변경된다.
        """
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
