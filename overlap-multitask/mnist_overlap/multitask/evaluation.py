"""Baseline과 multitask의 분류·pair·복원 성능을 paired 방식으로 비교한다."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import CLASS_COUNT, COMPOSITION_MODE, OVERLAP_LEVELS
from ..data import ControlledOverlapMnistDataset
from ..metrics import (
    class_pair_accuracy,
    classification_metrics,
    exact_match_per_sample,
    hierarchical_bootstrap_interval,
    sample_deviation,
)
from .config import CHECKPOINT_DIR, METRICS_JSON_PATH, MultitaskConfig
from .losses import (
    foreground_dice_per_sample,
    source_reconstruction_loss,
)
from .model import MultitaskMnistONet
from .training import load_checkpoint


EVALUATION_BATCH_SIZE = 256
CONFIDENCE_LEVEL = 0.95
METADATA_FIELD_NAMES = (
    "sample_id",
    "pair_id",
    "label_first",
    "label_second",
    "bounding_box_overlap_ratio",
    "pixel_overlap_ratio",
)
RECONSTRUCTION_METRIC_NAMES = (
    "foreground_dice",
    "balanced_bce",
    "l1",
    "mse",
    "psnr",
)


class MultitaskPredictor:
    """Seed별 multitask 분류 prediction과 sample별 복원 지표를 수집한다."""

    def __init__(
        self,
        config: MultitaskConfig,
        reconstruction_loss_weight: float,
        device: torch.device,
    ) -> None:
        self.config = config
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.device = device

    def collect_all_seeds(
        self,
        test_dataset: ControlledOverlapMnistDataset,
        training_seeds: list[int],
    ) -> dict[int, dict[str, np.ndarray]]:
        """모든 완료 checkpoint로 test 분류와 복원 지표를 dataset 순서대로 수집한다."""
        test_loader = DataLoader(
            test_dataset,
            batch_size=EVALUATION_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )
        predictions_by_seed = {}
        for training_seed in training_seeds:
            model = self.load_model(training_seed)
            predictions_by_seed[training_seed] = self._run_model(model, test_loader)
            print(f"  multitask seed={training_seed} 추론 완료")
        return predictions_by_seed

    def load_model(self, training_seed: int) -> MultitaskMnistONet:
        """한 seed의 완료 checkpoint를 평가 device에 복원한다."""
        model = MultitaskMnistONet()
        load_checkpoint(
            model,
            CHECKPOINT_DIR / f"seed_{training_seed}.pt",
            self.device,
            self.config,
            self.reconstruction_loss_weight,
        )
        model.eval()
        return model

    @torch.no_grad()
    def _run_model(
        self,
        model: MultitaskMnistONet,
        test_loader: DataLoader,
    ) -> dict[str, np.ndarray]:
        """한 모델의 분류 logit·metadata와 class별 복원 오차를 array로 모은다."""
        collected_columns: dict[str, list[np.ndarray]] = {
            "logits": [],
            "labels": [],
            "overlap_level": [],
            **{name: [] for name in RECONSTRUCTION_METRIC_NAMES},
        }
        for field_name in METADATA_FIELD_NAMES:
            collected_columns[field_name] = []

        for batch in test_loader:
            images = batch["image"].to(self.device)
            source_images = batch["source_images"].to(self.device)
            reconstruction_targets = batch["reconstruction_targets"].to(self.device)
            source_offsets = batch["source_offsets"].to(self.device)
            reconstruction_classes = torch.stack((
                batch["label_first"],
                batch["label_second"],
            ), dim=1).to(self.device)
            output = model(images, reconstruction_classes)
            reconstruction_result = source_reconstruction_loss(
                output.reconstruction_logits,
                reconstruction_targets,
            )
            source_reconstructions = torch.sigmoid(
                output.reconstruction_logits
            )
            cropped_reconstructions = crop_source_images(
                source_reconstructions,
                source_offsets,
                source_images.shape[-1],
            )
            absolute_error = torch.abs(cropped_reconstructions - source_images)
            squared_error = torch.square(cropped_reconstructions - source_images)
            l1_per_sample = absolute_error.mean(dim=(1, 2, 3))
            mse_per_sample = squared_error.mean(dim=(1, 2, 3))
            psnr_per_sample = -10.0 * torch.log10(mse_per_sample.clamp_min(1e-12))
            dice_per_sample = foreground_dice_per_sample(
                source_reconstructions,
                reconstruction_targets,
            )

            collected_columns["logits"].append(output.logits.cpu().numpy())
            collected_columns["labels"].append(batch["label"].numpy())
            collected_columns["overlap_level"].append(
                np.asarray(batch["overlap_level"])
            )
            collected_columns["balanced_bce"].append(
                reconstruction_result.balanced_bce_per_sample.cpu().numpy()
            )
            collected_columns["foreground_dice"].append(
                dice_per_sample.cpu().numpy()
            )
            collected_columns["l1"].append(l1_per_sample.cpu().numpy())
            collected_columns["mse"].append(mse_per_sample.cpu().numpy())
            collected_columns["psnr"].append(psnr_per_sample.cpu().numpy())

            for field_name in METADATA_FIELD_NAMES:
                collected_columns[field_name].append(batch[field_name].numpy())

        return {
            field_name: np.concatenate(columns)
            for field_name, columns in collected_columns.items()
        }


class ComparisonEvaluator:
    """공통 10 seeds의 baseline 대비 multitask 차이를 계산·보고·저장한다."""

    def __init__(self, config: MultitaskConfig) -> None:
        self.config = config

    def evaluate(
        self,
        baseline_predictions_by_seed: dict[int, dict[str, np.ndarray]],
        multitask_predictions_by_seed: dict[int, dict[str, np.ndarray]],
        training_seeds: list[int],
        reconstruction_loss_weight: float,
    ) -> dict[str, Any]:
        """분류 paired 차이, pair 분석과 복원 지표를 하나의 dictionary로 계산한다."""
        self._validate_paired_predictions(
            baseline_predictions_by_seed,
            multitask_predictions_by_seed,
            training_seeds,
        )
        baseline_performance = self._classification_performance(
            baseline_predictions_by_seed, training_seeds
        )
        multitask_performance = self._classification_performance(
            multitask_predictions_by_seed, training_seeds
        )
        paired_differences = self._paired_classification_differences(
            baseline_predictions_by_seed,
            multitask_predictions_by_seed,
            training_seeds,
        )
        pair_accuracy = self._pair_accuracy_comparison(
            baseline_predictions_by_seed,
            multitask_predictions_by_seed,
            training_seeds,
        )
        pair_confusion_high = self._pair_confusion_comparison(
            baseline_predictions_by_seed,
            multitask_predictions_by_seed,
            training_seeds,
        )
        reconstruction_performance = self._reconstruction_performance(
            multitask_predictions_by_seed, training_seeds
        )

        return {
            "models": {
                "baseline": "MnistONet",
                "multitask": "MnistONet+CompactLatentDecoder",
            },
            "composition_mode": COMPOSITION_MODE,
            "training_seeds": training_seeds,
            "reconstruction_loss_weight": reconstruction_loss_weight,
            "bootstrap_iterations": self.config.baseline.evaluation.bootstrap_iterations,
            "confidence_level": CONFIDENCE_LEVEL,
            "classification_performance": {
                "baseline": baseline_performance,
                "multitask": multitask_performance,
            },
            "multitask_minus_baseline": paired_differences,
            "pair_accuracy": pair_accuracy,
            "pair_confusion_high": pair_confusion_high,
            "reconstruction_performance": reconstruction_performance,
        }

    def report(self, metrics: dict[str, Any]) -> None:
        """비교의 핵심 분류 차이와 복원 성능을 stdout에 요약한다."""
        print("\n===== Baseline vs Multitask 결과 요약 =====")
        for level in ("all", *OVERLAP_LEVELS):
            baseline = metrics["classification_performance"]["baseline"][level]
            multitask = metrics["classification_performance"]["multitask"][level]
            difference = metrics["multitask_minus_baseline"][level]
            print(
                f"  {level:8} baseline={baseline['exact_match_mean'] * 100:6.2f}% "
                f"multitask={multitask['exact_match_mean'] * 100:6.2f}% "
                f"delta={difference['estimate'] * 100:+.2f}%p "
                f"95% CI [{difference['confidence_lower'] * 100:+.2f}, "
                f"{difference['confidence_upper'] * 100:+.2f}]"
            )

        high_reconstruction = metrics["reconstruction_performance"]["high"]
        print(
            "\n[Multitask reconstruction — High] "
            f"Dice={high_reconstruction['foreground_dice_mean']:.4f}, "
            f"crop-L1={high_reconstruction['l1_mean']:.4f}, "
            f"PSNR={high_reconstruction['psnr_mean']:.2f} dB"
        )

    def save(self, metrics: dict[str, Any]) -> None:
        """비교 결과를 UTF-8 JSON으로 원자 저장한다."""
        METRICS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = METRICS_JSON_PATH.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(METRICS_JSON_PATH)
        print(f"\n비교 수치를 저장했습니다: {METRICS_JSON_PATH}")

    def _classification_performance(
        self,
        predictions_by_seed: dict[int, dict[str, np.ndarray]],
        training_seeds: list[int],
    ) -> dict[str, dict[str, float]]:
        """전체·overlap별 exact-match와 macro-F1의 seed 평균·표본 std를 계산한다."""
        performance = {}
        for level in ("all", *OVERLAP_LEVELS):
            exact_values = []
            f1_values = []
            for seed in training_seeds:
                predictions = predictions_by_seed[seed]
                level_mask = self._level_mask(predictions, level)
                classification = classification_metrics(
                    predictions["logits"][level_mask],
                    predictions["labels"][level_mask],
                )
                exact_values.append(classification["exact_match"])
                f1_values.append(classification["macro_f1"])
            exact_array = np.asarray(exact_values)
            f1_array = np.asarray(f1_values)
            performance[level] = {
                "exact_match_mean": float(exact_array.mean()),
                "exact_match_standard_deviation": sample_deviation(exact_array),
                "macro_f1_mean": float(f1_array.mean()),
                "macro_f1_standard_deviation": sample_deviation(f1_array),
            }
        return performance

    def _paired_classification_differences(
        self,
        baseline_by_seed: dict[int, dict[str, np.ndarray]],
        multitask_by_seed: dict[int, dict[str, np.ndarray]],
        training_seeds: list[int],
    ) -> dict[str, dict[str, float | int]]:
        """각 level의 sample 차이를 pair로 묶어 seed×pair hierarchical CI를 계산한다."""
        results = {}
        iterations = self.config.baseline.evaluation.bootstrap_iterations
        for level_index, level in enumerate(("all", *OVERLAP_LEVELS)):
            values_by_seed = []
            for seed in training_seeds:
                baseline = baseline_by_seed[seed]
                multitask = multitask_by_seed[seed]
                baseline_correct = self._correct_per_sample(baseline)
                multitask_correct = self._correct_per_sample(multitask)
                level_mask = self._level_mask(baseline, level)
                sample_differences = (
                    multitask_correct[level_mask] - baseline_correct[level_mask]
                )
                pair_values = self._mean_by_pair(
                    sample_differences, baseline["pair_id"][level_mask]
                )
                values_by_seed.append(pair_values)
            difference_matrix = np.stack(values_by_seed)
            estimate, lower, upper = hierarchical_bootstrap_interval(
                difference_matrix,
                iterations,
                CONFIDENCE_LEVEL,
                self.config.baseline.data.seed + level_index,
            )
            results[level] = {
                "estimate": estimate,
                "confidence_lower": lower,
                "confidence_upper": upper,
                "seed_count": len(training_seeds),
                "pair_count": int(difference_matrix.shape[1]),
            }
        return results

    def _pair_accuracy_comparison(
        self,
        baseline_by_seed: dict[int, dict[str, np.ndarray]],
        multitask_by_seed: dict[int, dict[str, np.ndarray]],
        training_seeds: list[int],
    ) -> dict[str, dict[str, list[list[float | None]]]]:
        """각 overlap level에서 모델별 pair accuracy와 multitask delta를 계산한다."""
        result = {}
        reference = baseline_by_seed[training_seeds[0]]
        for level in OVERLAP_LEVELS:
            level_mask = reference["overlap_level"] == level
            baseline_matrices = []
            multitask_matrices = []
            for seed in training_seeds:
                baseline_matrices.append(class_pair_accuracy(
                    self._correct_per_sample(baseline_by_seed[seed])[level_mask],
                    reference["label_first"][level_mask],
                    reference["label_second"][level_mask],
                    CLASS_COUNT,
                ))
                multitask_matrices.append(class_pair_accuracy(
                    self._correct_per_sample(multitask_by_seed[seed])[level_mask],
                    reference["label_first"][level_mask],
                    reference["label_second"][level_mask],
                    CLASS_COUNT,
                ))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                baseline_mean = np.nanmean(np.stack(baseline_matrices), axis=0)
                multitask_mean = np.nanmean(np.stack(multitask_matrices), axis=0)
            result[level] = {
                "baseline": self._matrix_to_json(baseline_mean),
                "multitask": self._matrix_to_json(multitask_mean),
                "difference": self._matrix_to_json(multitask_mean - baseline_mean),
            }
        return result

    def _pair_confusion_comparison(
        self,
        baseline_by_seed: dict[int, dict[str, np.ndarray]],
        multitask_by_seed: dict[int, dict[str, np.ndarray]],
        training_seeds: list[int],
    ) -> dict[str, Any]:
        """High overlap의 45 unordered pair row-normalized confusion을 seed 평균한다."""
        reference = baseline_by_seed[training_seeds[0]]
        high_mask = reference["overlap_level"] == "high"
        pair_labels = [
            f"{first}+{second}"
            for first in range(CLASS_COUNT)
            for second in range(first + 1, CLASS_COUNT)
        ]
        pair_lookup = {
            tuple(int(value) for value in label.split("+")): index
            for index, label in enumerate(pair_labels)
        }

        def mean_confusion(predictions_by_seed: dict[int, dict[str, np.ndarray]]) -> np.ndarray:
            matrices = []
            for seed in training_seeds:
                predictions = predictions_by_seed[seed]
                true_pairs = np.sort(np.stack((
                    predictions["label_first"][high_mask],
                    predictions["label_second"][high_mask],
                ), axis=1), axis=1)
                predicted_pairs = np.sort(
                    np.argpartition(predictions["logits"][high_mask], kth=-2, axis=1)[:, -2:],
                    axis=1,
                )
                matrix = np.zeros((len(pair_labels), len(pair_labels)), dtype=np.float64)
                for true_pair, predicted_pair in zip(true_pairs, predicted_pairs):
                    true_index = pair_lookup[tuple(true_pair)]
                    predicted_index = pair_lookup[tuple(predicted_pair)]
                    matrix[true_index][predicted_index] += 1.0
                row_sums = matrix.sum(axis=1, keepdims=True)
                np.divide(matrix, row_sums, out=matrix, where=row_sums != 0)
                matrices.append(matrix)
            return np.stack(matrices).mean(axis=0)

        return {
            "pair_labels": pair_labels,
            "baseline": mean_confusion(baseline_by_seed).tolist(),
            "multitask": mean_confusion(multitask_by_seed).tolist(),
        }

    def _reconstruction_performance(
        self,
        predictions_by_seed: dict[int, dict[str, np.ndarray]],
        training_seeds: list[int],
    ) -> dict[str, dict[str, float]]:
        """전체·overlap별 다섯 복원 지표의 seed 평균·표본 std를 계산한다."""
        performance = {}
        for level in ("all", *OVERLAP_LEVELS):
            entry = {}
            for metric_name in RECONSTRUCTION_METRIC_NAMES:
                seed_values = []
                for seed in training_seeds:
                    predictions = predictions_by_seed[seed]
                    level_mask = self._level_mask(predictions, level)
                    seed_values.append(float(predictions[metric_name][level_mask].mean()))
                values = np.asarray(seed_values)
                entry[f"{metric_name}_mean"] = float(values.mean())
                entry[f"{metric_name}_standard_deviation"] = sample_deviation(values)
            performance[level] = entry
        return performance

    def _validate_paired_predictions(
        self,
        baseline_by_seed: dict[int, dict[str, np.ndarray]],
        multitask_by_seed: dict[int, dict[str, np.ndarray]],
        training_seeds: list[int],
    ) -> None:
        """두 모델의 seed와 test sample 순서·정답·metadata가 완전히 같은지 검사한다."""
        if set(baseline_by_seed) != set(training_seeds):
            raise ValueError("Baseline prediction seed가 비교 seed와 일치하지 않습니다.")
        if set(multitask_by_seed) != set(training_seeds):
            raise ValueError("Multitask prediction seed가 비교 seed와 일치하지 않습니다.")
        for seed in training_seeds:
            baseline = baseline_by_seed[seed]
            multitask = multitask_by_seed[seed]
            for field_name in ("labels", "overlap_level", *METADATA_FIELD_NAMES):
                if not np.array_equal(baseline[field_name], multitask[field_name]):
                    raise ValueError(
                        f"seed={seed}의 {field_name} 순서가 모델 사이에서 다릅니다."
                    )

    @staticmethod
    def _correct_per_sample(predictions: dict[str, np.ndarray]) -> np.ndarray:
        correctness = exact_match_per_sample(
            torch.from_numpy(predictions["logits"]),
            torch.from_numpy(predictions["labels"]),
        )
        return correctness.numpy().astype(np.float64)

    @staticmethod
    def _level_mask(predictions: dict[str, np.ndarray], level: str) -> np.ndarray:
        return (
            np.ones(len(predictions["labels"]), dtype=bool)
            if level == "all"
            else predictions["overlap_level"] == level
        )

    @staticmethod
    def _mean_by_pair(values: np.ndarray, pair_ids: np.ndarray) -> np.ndarray:
        unique_pair_ids, inverse_indices = np.unique(pair_ids, return_inverse=True)
        sums = np.bincount(inverse_indices, weights=values, minlength=len(unique_pair_ids))
        counts = np.bincount(inverse_indices, minlength=len(unique_pair_ids))
        return sums / counts

    @staticmethod
    def _matrix_to_json(matrix: np.ndarray) -> list[list[float | None]]:
        return [
            [None if not np.isfinite(value) else float(value) for value in row]
            for row in matrix
        ]


def crop_source_images(
    source_reconstructions: torch.Tensor,
    source_offsets: torch.Tensor,
    source_size: int,
) -> torch.Tensor:
    """각 source의 target 좌표에서 `source_size×source_size` patch를 모은다."""
    if source_reconstructions.ndim != 4 or source_reconstructions.shape[1] != 2:
        raise ValueError("복원 결과는 `[batch,2,height,width]` 형태여야 합니다.")
    expected_offset_shape = (source_reconstructions.shape[0], 2, 2)
    if tuple(source_offsets.shape) != expected_offset_shape:
        raise ValueError("Source offset은 `[batch,2,2]` 형태여야 합니다.")

    maximum_y = source_reconstructions.shape[-2] - source_size
    maximum_x = source_reconstructions.shape[-1] - source_size
    offset_x = source_offsets[..., 0]
    offset_y = source_offsets[..., 1]
    if (
        torch.any(offset_x < 0)
        or torch.any(offset_x > maximum_x)
        or torch.any(offset_y < 0)
        or torch.any(offset_y > maximum_y)
    ):
        raise ValueError("Source crop이 reconstruction 범위를 벗어납니다.")

    sliding_patches = source_reconstructions.unfold(2, source_size, 1).unfold(
        3, source_size, 1
    )
    batch_indices = torch.arange(
        source_reconstructions.shape[0],
        device=source_reconstructions.device,
    ).unsqueeze(1)
    source_indices = torch.arange(2, device=source_reconstructions.device).unsqueeze(0)
    return sliding_patches[
        batch_indices,
        source_indices,
        offset_y,
        offset_x,
    ]
