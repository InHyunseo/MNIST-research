"""
시각화 단계(`Visualizer`)를 정의한다. 학습 이력, test 추론 결과와 dataset에서 발표용
그림 네 장을 그린다. `main`의 `--plot` 인자에서만 호출되며 자체 진입점은 아니다.
각 그림 메서드의 docstring은 그림이 무엇을 어떻게 보여주는지 캡션 수준으로 설명한다.
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import PercentFormatter

from ..config import CLASS_COUNT, FIGURE_DIR, TRAINING_LOG_DIR
from ..data import ControlledOverlapMnistDataset
from ..metrics import (
    class_pair_accuracy,
    classification_metrics,
    exact_match_per_sample,
    sample_deviation,
)
from ..training_plot import draw_training_curves

OVERLAP_LEVELS = ("low", "middle", "high")
EXAMPLE_CLASS_PAIRS = ((1, 0), (4, 7), (3, 8))
BAR_COLOR = "#4C78A8"
MEAN_COLOR = "#173F67"
FIGURE_DPI = 150


class Visualizer:
    """
    입력: test dataset, predictions_by_seed, training_seeds
    출력: results/baseline/figures/ 아래 PNG 네 장

    발표용 그림(입력 예시, overlap별 정확도, High pair heatmap)을 그리는 단계이다.
    """

    def make_all(
        self,
        test_dataset: ControlledOverlapMnistDataset,
        predictions_by_seed: dict[int, dict[str, np.ndarray]],
        training_seeds: list[int],
    ) -> None:
        """
        입력: test_dataset — 입력 예시를 뽑을 test dataset
              predictions_by_seed — seed별 test prediction
              training_seeds — 학습 seed 목록
        출력: 없음 (PNG 네 장 저장)

        학습 곡선과 발표용 평가 그림을 순서대로 저장한다.
        """
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)

        self.draw_training_curves(training_seeds)
        self.draw_overlap_examples(test_dataset)
        self.draw_overlap_accuracy(predictions_by_seed, training_seeds)
        self.draw_pair_accuracy_high(predictions_by_seed, training_seeds)

    def draw_training_curves(self, training_seeds: list[int]) -> None:
        """열 seed의 validation accuracy·loss와 epoch별 평균을 그린다."""
        draw_training_curves(
            [TRAINING_LOG_DIR / f"seed_{seed}.csv" for seed in training_seeds],
            accuracy_column="validation_exact_match",
            loss_column="validation_loss",
            line_color=BAR_COLOR,
            mean_color=MEAN_COLOR,
            output_path=FIGURE_DIR / "training_curves.png",
            figure_dpi=FIGURE_DPI,
        )

    def draw_overlap_examples(self, test_dataset: ControlledOverlapMnistDataset) -> None:
        """
        입력: test_dataset — 합성 입력을 뽑을 test dataset
        출력: 없음 (results/baseline/figures/overlap_examples.png 저장)

        Controlled overlap 입력 예시 그림(3×3). 각 행은 한 숫자 조합(1+0, 4+7, 3+8),
        각 열은 overlap level(Low/Middle/High)이다. 한 행 안에서는 동일한 MNIST 원본
        두 장·pair 중심·이동 방향을 유지하고 변위만 바꿔, 열 간 차이가 순수하게 겹침
        강도의 변화임을 보여준다. 각 칸의 밝기는 두 숫자의 clipped-sum 합성 결과다.
        """
        figure, axes = plt.subplots(
            len(EXAMPLE_CLASS_PAIRS),
            len(OVERLAP_LEVELS),
            figsize=(7.5, 7.5),
            constrained_layout=True,
            squeeze=False,
        )

        for row_index, class_pair in enumerate(EXAMPLE_CLASS_PAIRS):
            dataset_indices = self._select_class_pair_examples(test_dataset, class_pair)

            for column_index, (level, dataset_index) in enumerate(zip(OVERLAP_LEVELS, dataset_indices)):
                sample = test_dataset[dataset_index]

                axis = axes[row_index, column_index]
                axis.imshow(sample["image"][0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
                axis.set_xticks([])
                axis.set_yticks([])

                if row_index == 0:
                    axis.set_title(level.title())
                if column_index == 0:
                    axis.set_ylabel(f"{class_pair[0]} + {class_pair[1]}", fontsize=11)

        figure.savefig(FIGURE_DIR / "overlap_examples.png", dpi=FIGURE_DPI)
        plt.close(figure)

    def draw_overlap_accuracy(
        self,
        predictions_by_seed: dict[int, dict[str, np.ndarray]],
        training_seeds: list[int],
    ) -> None:
        """
        입력: predictions_by_seed — seed별 test prediction
              training_seeds — 학습 seed 목록
        출력: 없음 (results/baseline/figures/overlap_accuracy.png 저장)

        overlap 강도별 Top-2 exact-match 정확도 막대 그림. 막대는 Low/Middle/High 각각에서
        전체 학습 seed의 평균 정확도, error bar는 seed 간 표본 표준편차다. 막대 위 숫자는
        평균 정확도(%)다. 겹침이 강해질수록 두 숫자를 모두 맞히기 어려워지는 경향을 보인다.
        """
        positions = np.arange(len(OVERLAP_LEVELS), dtype=float)
        means = []
        standard_deviations = []

        for level in OVERLAP_LEVELS:
            seed_values = []

            for training_seed in training_seeds:
                predictions = predictions_by_seed[training_seed]
                level_mask = predictions["overlap_level"] == level

                seed_values.append(
                    classification_metrics(
                        predictions["logits"][level_mask], predictions["labels"][level_mask]
                    )["exact_match"]
                )

            seed_values = np.asarray(seed_values)
            means.append(float(seed_values.mean()))
            standard_deviations.append(sample_deviation(seed_values))

        figure, axis = plt.subplots(figsize=(6.0, 4.4))
        axis.bar(positions, means, yerr=standard_deviations, width=0.55, color=BAR_COLOR, capsize=4)

        for position, mean in zip(positions, means):
            axis.text(position, mean + 0.01, f"{mean * 100:.1f}%", ha="center", fontsize=10)

        axis.set_xticks(positions, [level.title() for level in OVERLAP_LEVELS])
        axis.set_ylim(0.0, 1.0)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axis.set_ylabel("Accuracy")
        axis.set_title("Overlap Accuracy")
        axis.grid(axis="y", alpha=0.2)

        figure.tight_layout()
        figure.savefig(FIGURE_DIR / "overlap_accuracy.png", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

    def draw_pair_accuracy_high(
        self,
        predictions_by_seed: dict[int, dict[str, np.ndarray]],
        training_seeds: list[int],
    ) -> None:
        """
        입력: predictions_by_seed — seed별 test prediction
              training_seeds — 학습 seed 목록
        출력: 없음 (results/baseline/figures/pair_accuracy_high.png 저장)

        High overlap에서 숫자 조합별 정확도 heatmap. 각 cell (i, j)는 정답이 숫자 i와 j인
        High-overlap sample의 평균 Top-2 exact-match를 전체 seed에 대해 평균한 값이다.
        대칭 행렬이며 대각선(i=j)은 같은 숫자 두 개를 데이터에 넣지 않아 비어 있다(NaN).
        밝을수록 정확히 맞히는 조합, 어두울수록 자주 틀리는 조합이다.
        """
        reference = predictions_by_seed[training_seeds[0]]
        high_mask = reference["overlap_level"] == "high"
        label_first_high = reference["label_first"][high_mask]
        label_second_high = reference["label_second"][high_mask]

        accuracy_matrices = []

        for training_seed in training_seeds:
            predictions = predictions_by_seed[training_seed]
            correctness = exact_match_per_sample(
                torch.from_numpy(predictions["logits"]),
                torch.from_numpy(predictions["labels"]),
            ).numpy().astype(np.float64)

            accuracy_matrices.append(
                class_pair_accuracy(
                    correctness[high_mask], label_first_high, label_second_high, CLASS_COUNT
                )
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_accuracy_matrix = np.nanmean(np.stack(accuracy_matrices), axis=0)

        figure, axis = plt.subplots(figsize=(6.4, 5.4))
        image = axis.imshow(
            mean_accuracy_matrix,
            cmap="viridis",
            vmin=np.nanmin(mean_accuracy_matrix),
            vmax=np.nanmax(mean_accuracy_matrix),
        )

        axis.set_xticks(range(CLASS_COUNT))
        axis.set_yticks(range(CLASS_COUNT))
        axis.set_title("Pair Accuracy")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

        figure.tight_layout()
        figure.savefig(FIGURE_DIR / "pair_accuracy_high.png", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

    def _select_class_pair_examples(
        self,
        test_dataset: ControlledOverlapMnistDataset,
        class_pair: tuple[int, int],
    ) -> list[int]:
        """
        입력: test_dataset — 검색 대상 test dataset
              class_pair — 찾을 unordered 숫자 조합 (예: (3, 8))
        출력: 같은 pair의 Low/Middle/High 세 sample의 dataset index 목록

        지정한 숫자 조합의 첫 완전한 paired test sample 세 개(Low/Middle/High)를 찾는다.
        """
        label_first = test_dataset.manifest["label_first"]
        label_second = test_dataset.manifest["label_second"]
        pair_ids = test_dataset.manifest["pair_id"]
        overlap_levels = test_dataset.manifest["overlap_level"]

        first_class, second_class = class_pair
        matches = (
            ((label_first == first_class) & (label_second == second_class))
            | ((label_first == second_class) & (label_second == first_class))
        )

        for pair_id in dict.fromkeys(int(value) for value in pair_ids[matches]):
            selected_indices = []

            for level in OVERLAP_LEVELS:
                indices = np.flatnonzero((pair_ids == pair_id) & (overlap_levels == level))
                if len(indices) != 1:
                    break
                selected_indices.append(int(indices[0]))

            if len(selected_indices) == len(OVERLAP_LEVELS):
                return selected_indices

        raise ValueError(f"숫자 조합 {class_pair}의 완전한 Low/Middle/High sample이 없습니다.")
