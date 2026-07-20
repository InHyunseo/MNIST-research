"""학습 곡선·baseline 비교·pair 분석·복원 예시 그림 다섯 장을 생성한다."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import PercentFormatter

from ..config import CLASS_COUNT, OVERLAP_LEVELS
from ..data import ControlledOverlapMnistDataset
from ..training_plot import draw_training_curves
from .config import FIGURE_DIR, TRAINING_LOG_DIR
from .evaluation import crop_source_images
from .model import MultitaskMnistONet


FIGURE_DPI = 150
BASELINE_COLOR = "#4C78A8"
MULTITASK_COLOR = "#F58518"
MULTITASK_MEAN_COLOR = "#9C3B00"
EXAMPLE_CLASS_PAIR = (3, 8)


class ComparisonVisualizer:
    """저장된 비교 수치와 seed 0 모델로 발표용 그림을 생성한다."""

    def make_all(
        self,
        metrics: dict[str, Any],
        test_dataset: ControlledOverlapMnistDataset,
        example_model: MultitaskMnistONet,
        device: torch.device,
    ) -> None:
        """학습·overlap·pair·confusion·복원 그림 다섯 장을 저장한다."""
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        self.draw_training_curves(metrics["training_seeds"])
        self.draw_overlap_comparison(metrics)
        self.draw_pair_accuracy_difference(metrics)
        self.draw_pair_confusion_high(metrics)
        self.draw_reconstruction_examples(test_dataset, example_model, device)

    def draw_training_curves(self, training_seeds: list[int]) -> None:
        """열 seed의 validation accuracy·total loss와 epoch별 평균을 그린다."""
        draw_training_curves(
            [TRAINING_LOG_DIR / f"seed_{seed}.csv" for seed in training_seeds],
            accuracy_column="validation_exact_match",
            loss_column="validation_total_loss",
            line_color=MULTITASK_COLOR,
            mean_color=MULTITASK_MEAN_COLOR,
            output_path=FIGURE_DIR / "training_curves.png",
            figure_dpi=FIGURE_DPI,
        )

    def draw_overlap_comparison(self, metrics: dict[str, Any]) -> None:
        """Low·Middle·High에서 baseline과 multitask exact-match를 나란히 비교한다."""
        positions = np.arange(len(OVERLAP_LEVELS), dtype=float)
        width = 0.34
        baseline_entries = metrics["classification_performance"]["baseline"]
        multitask_entries = metrics["classification_performance"]["multitask"]
        baseline_means = [baseline_entries[level]["exact_match_mean"] for level in OVERLAP_LEVELS]
        baseline_errors = [
            baseline_entries[level]["exact_match_standard_deviation"]
            for level in OVERLAP_LEVELS
        ]
        multitask_means = [
            multitask_entries[level]["exact_match_mean"] for level in OVERLAP_LEVELS
        ]
        multitask_errors = [
            multitask_entries[level]["exact_match_standard_deviation"]
            for level in OVERLAP_LEVELS
        ]

        figure, axis = plt.subplots(figsize=(7.0, 4.6))
        axis.bar(
            positions - width / 2,
            baseline_means,
            width,
            yerr=baseline_errors,
            capsize=4,
            color=BASELINE_COLOR,
            label="Baseline",
        )
        axis.bar(
            positions + width / 2,
            multitask_means,
            width,
            yerr=multitask_errors,
            capsize=4,
            color=MULTITASK_COLOR,
            label="Multitask",
        )
        for position, baseline_mean, multitask_mean in zip(
            positions, baseline_means, multitask_means
        ):
            difference = (multitask_mean - baseline_mean) * 100.0
            axis.text(
                position,
                max(baseline_mean, multitask_mean) + 0.025,
                f"{difference:+.1f}%p",
                ha="center",
                fontsize=9,
            )

        axis.set_xticks(positions, [level.title() for level in OVERLAP_LEVELS])
        axis.set_ylim(0.0, 1.0)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axis.set_ylabel("Accuracy")
        axis.set_title("Overlap Accuracy")
        axis.grid(axis="y", alpha=0.2)
        axis.legend()
        figure.tight_layout()
        figure.savefig(
            FIGURE_DIR / "overlap_comparison.png",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )
        plt.close(figure)

    def draw_pair_accuracy_difference(self, metrics: dict[str, Any]) -> None:
        """세 overlap level의 pair별 multitask−baseline 정확도 차이를 그린다."""
        matrices = [
            np.asarray(metrics["pair_accuracy"][level]["difference"], dtype=np.float64)
            for level in OVERLAP_LEVELS
        ]
        maximum_absolute_difference = max(
            float(np.nanmax(np.abs(matrix))) for matrix in matrices
        )
        color_limit = max(maximum_absolute_difference, 0.01)
        upper_triangle = np.triu(np.ones((CLASS_COUNT, CLASS_COUNT), dtype=bool), k=1)
        figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)
        image = None

        for axis, level, matrix in zip(axes, OVERLAP_LEVELS, matrices):
            masked_matrix = np.ma.masked_where(~upper_triangle, matrix)
            image = axis.imshow(
                masked_matrix,
                cmap="RdBu",
                vmin=-color_limit,
                vmax=color_limit,
            )
            axis.set_xticks(range(CLASS_COUNT))
            axis.set_yticks(range(CLASS_COUNT))
            axis.set_title(level.title())

        assert image is not None
        colorbar = figure.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
        colorbar.set_label("Accuracy Δ")
        figure.suptitle("Pair Difference")
        figure.savefig(
            FIGURE_DIR / "pair_accuracy_difference.png",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )
        plt.close(figure)

    def draw_pair_confusion_high(self, metrics: dict[str, Any]) -> None:
        """High overlap의 45×45 unordered-pair confusion을 두 모델에 대해 그린다."""
        pair_confusion = metrics["pair_confusion_high"]
        labels = pair_confusion["pair_labels"]
        baseline = np.asarray(pair_confusion["baseline"], dtype=np.float64)
        multitask = np.asarray(pair_confusion["multitask"], dtype=np.float64)
        color_limit = max(float(baseline.max()), float(multitask.max()))
        figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.3), constrained_layout=True)
        tick_indices = np.arange(0, len(labels), 5)
        image = None

        for axis, title, matrix in (
            (axes[0], "Baseline", baseline),
            (axes[1], "Multitask", multitask),
        ):
            image = axis.imshow(matrix, cmap="magma", vmin=0.0, vmax=color_limit)
            axis.set_xticks(tick_indices, [labels[index] for index in tick_indices], rotation=90)
            axis.set_yticks(tick_indices, [labels[index] for index in tick_indices])
            axis.set_xlabel("Predicted")
            axis.set_ylabel("True")
            axis.set_title(title)

        assert image is not None
        colorbar = figure.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
        figure.suptitle("Pair Confusion")
        figure.savefig(
            FIGURE_DIR / "pair_confusion_high.png",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )
        plt.close(figure)

    @torch.no_grad()
    def draw_reconstruction_examples(
        self,
        test_dataset: ControlledOverlapMnistDataset,
        model: MultitaskMnistONet,
        device: torch.device,
    ) -> None:
        """같은 3+8 pair의 Low·Middle·High class-specific 복원을 비교한다."""
        example_indices = self._select_paired_examples(test_dataset, EXAMPLE_CLASS_PAIR)
        model.eval()
        figure, axes = plt.subplots(
            len(OVERLAP_LEVELS),
            5,
            figsize=(10.0, 6.2),
            constrained_layout=True,
            squeeze=False,
        )
        first_sample = test_dataset[example_indices[0]]
        first_class = int(first_sample["label_first"])
        second_class = int(first_sample["label_second"])
        column_titles = (
            "Mixed",
            f"GT {first_class}",
            f"GT {second_class}",
            f"Recon {first_class}",
            f"Recon {second_class}",
        )

        for row_index, (level, dataset_index) in enumerate(
            zip(OVERLAP_LEVELS, example_indices)
        ):
            sample = test_dataset[dataset_index]
            images = sample["image"].unsqueeze(0).to(device)
            reconstruction_classes = torch.tensor(
                [[sample["label_first"], sample["label_second"]]],
                device=device,
            )
            output = model(images, reconstruction_classes)
            source_reconstructions = torch.sigmoid(output.reconstruction_logits)
            cropped_reconstructions = crop_source_images(
                source_reconstructions,
                sample["source_offsets"].unsqueeze(0).to(device),
                sample["source_images"].shape[-1],
            )[0].cpu()
            displayed_images = (
                sample["image"][0],
                sample["source_images"][0],
                sample["source_images"][1],
                cropped_reconstructions[0],
                cropped_reconstructions[1],
            )

            for column_index, displayed_image in enumerate(displayed_images):
                axis = axes[row_index, column_index]
                axis.imshow(displayed_image.numpy(), cmap="gray", vmin=0.0, vmax=1.0)
                axis.set_xticks([])
                axis.set_yticks([])
                if row_index == 0:
                    axis.set_title(column_titles[column_index])
                if column_index == 0:
                    axis.set_ylabel(level.title())

        figure.suptitle("Reconstructions")
        figure.savefig(
            FIGURE_DIR / "reconstruction_examples.png",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )
        plt.close(figure)

    @staticmethod
    def _select_paired_examples(
        test_dataset: ControlledOverlapMnistDataset,
        class_pair: tuple[int, int],
    ) -> list[int]:
        """지정 class pair에서 같은 원본을 공유하는 Low·Middle·High index를 찾는다."""
        label_first = test_dataset.manifest["label_first"]
        label_second = test_dataset.manifest["label_second"]
        pair_ids = test_dataset.manifest["pair_id"]
        overlap_levels = test_dataset.manifest["overlap_level"]
        first_class, second_class = class_pair
        class_mask = (label_first == first_class) & (label_second == second_class)
        if not np.any(class_mask):
            class_mask = (label_first == second_class) & (label_second == first_class)

        for pair_id in dict.fromkeys(int(value) for value in pair_ids[class_mask]):
            selected_indices = []
            for level in OVERLAP_LEVELS:
                indices = np.flatnonzero(
                    (pair_ids == pair_id) & (overlap_levels == level)
                )
                if len(indices) != 1:
                    break
                selected_indices.append(int(indices[0]))
            if len(selected_indices) == len(OVERLAP_LEVELS):
                return selected_indices
        raise ValueError(f"숫자 조합 {class_pair}의 완전한 paired sample이 없습니다.")
