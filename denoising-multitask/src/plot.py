"""
저장된 n-MNIST 최종 결과와 history에서 필수 figure를 생성한다.

입력:
    - outputs/results.csv
    - outputs/histories의 최종 run별 CSV

출력:
    - Accuracy comparison, paired delta, run별 loss·accuracy PNG

주요 기능:
    1. Seed별 결과 집계
    2. Baseline과 multitask accuracy 비교
    3. 저장된 학습 history 시각화
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dataset import NOISE_TYPES, PROJECT_DIRECTORY
from src.experiment import HISTORY_DIRECTORY, RESULT_COLUMNS, RESULTS_PATH


FIGURE_DIRECTORY = PROJECT_DIRECTORY / "outputs" / "figures"
CONDITIONS = ("classification_only", "multitask")
NOISE_LABELS = {
    "awgn": "AWGN",
    "motion_blur": "Motion blur",
    "reduced_contrast_awgn": "Reduced contrast + AWGN",
}
CONDITION_LABELS = {
    "classification_only": "Classification only",
    "multitask": "Multitask",
}


def create_plots() -> None:
    """저장된 최종 결과만 사용해 모든 필수 figure를 생성한다."""
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"최종 결과가 없습니다: {RESULTS_PATH}\n먼저 학습을 완료하세요."
        )
    results = pd.read_csv(RESULTS_PATH)
    if tuple(results.columns) != RESULT_COLUMNS:
        raise ValueError(f"results.csv schema가 올바르지 않습니다: {RESULTS_PATH}")
    _validate_complete_conditions(results)
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _plot_accuracy_comparison(results)
    _plot_accuracy_delta(results)
    _plot_training_histories()
    print(f"Figure를 생성했습니다: {FIGURE_DIRECTORY}")


def _validate_complete_conditions(results: pd.DataFrame) -> None:
    """각 noise에서 baseline과 multitask 결과가 모두 존재하는지 확인한다."""
    for noise_type in NOISE_TYPES:
        available_conditions = set(
            results.loc[results["noise_type"] == noise_type, "condition"]
        )
        missing_conditions = set(CONDITIONS) - available_conditions
        if missing_conditions:
            missing_text = ", ".join(sorted(missing_conditions))
            raise RuntimeError(
                f"{noise_type} 결과에 필요한 condition이 없습니다: {missing_text}"
            )


def _plot_accuracy_comparison(results: pd.DataFrame) -> None:
    """Noise별 두 조건의 평균 test accuracy와 표준편차를 grouped bar로 그린다."""
    x_positions = np.arange(len(NOISE_TYPES), dtype=np.float64)
    bar_width = 0.36
    figure, axis = plt.subplots(figsize=(9, 5))
    for condition_index, condition in enumerate(CONDITIONS):
        means = []
        standard_deviations = []
        for noise_type in NOISE_TYPES:
            values = results.loc[
                (results["noise_type"] == noise_type)
                & (results["condition"] == condition),
                "test_accuracy",
            ].astype(float)
            means.append(float(values.mean()))
            standard_deviations.append(
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        offset = (condition_index - 0.5) * bar_width
        axis.bar(
            x_positions + offset,
            means,
            bar_width,
            yerr=standard_deviations,
            capsize=4,
            label=CONDITION_LABELS[condition],
        )
    axis.set_xticks(x_positions, [NOISE_LABELS[name] for name in NOISE_TYPES])
    axis.set_ylabel("Test classification accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Classification accuracy by noise type")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURE_DIRECTORY / "accuracy_comparison.png", dpi=160)
    plt.close(figure)


def _plot_accuracy_delta(results: pd.DataFrame) -> None:
    """같은 seed의 multitask−baseline test accuracy 차이를 percentage point로 그린다."""
    mean_deltas = []
    standard_deviations = []
    for noise_type in NOISE_TYPES:
        noise_results = results.loc[results["noise_type"] == noise_type]
        baseline = noise_results.loc[
            noise_results["condition"] == "classification_only",
            ["random_seed", "test_accuracy"],
        ].rename(columns={"test_accuracy": "baseline_accuracy"})
        multitask = noise_results.loc[
            noise_results["condition"] == "multitask",
            ["random_seed", "test_accuracy"],
        ].rename(columns={"test_accuracy": "multitask_accuracy"})
        paired = baseline.merge(multitask, on="random_seed", validate="one_to_one")
        if paired.empty:
            raise RuntimeError(f"Paired seed 결과가 없습니다: {noise_type}")
        deltas = (
            paired["multitask_accuracy"] - paired["baseline_accuracy"]
        ) * 100.0
        mean_deltas.append(float(deltas.mean()))
        standard_deviations.append(
            float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
        )

    figure, axis = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(NOISE_TYPES))
    axis.bar(
        positions,
        mean_deltas,
        yerr=standard_deviations,
        capsize=4,
        color="#4C78A8",
    )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xticks(positions, [NOISE_LABELS[name] for name in NOISE_TYPES])
    axis.set_ylabel("Accuracy delta (percentage points)")
    axis.set_title("Multitask − classification-only accuracy")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURE_DIRECTORY / "accuracy_delta.png", dpi=160)
    plt.close(figure)


def _plot_training_histories() -> None:
    """각 최종 run의 loss와 accuracy history를 서로 다른 figure로 저장한다."""
    history_paths = sorted(HISTORY_DIRECTORY.glob("*_seed_*.csv"))
    if not history_paths:
        raise FileNotFoundError(f"최종 학습 history가 없습니다: {HISTORY_DIRECTORY}")
    for history_path in history_paths:
        history = pd.read_csv(history_path)
        _plot_loss_history(history, history_path.stem)
        _plot_accuracy_history(history, history_path.stem)


def _plot_loss_history(history: pd.DataFrame, run_name: str) -> None:
    """한 final run의 training·validation loss를 저장한다."""
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(history["epoch"], history["training_total_loss"], label="Training total")
    axis.plot(
        history["epoch"], history["validation_total_loss"], label="Validation total"
    )
    if history["training_reconstruction_loss"].notna().any():
        axis.plot(
            history["epoch"],
            history["training_reconstruction_loss"],
            linestyle="--",
            label="Training reconstruction",
        )
        axis.plot(
            history["epoch"],
            history["validation_reconstruction_loss"],
            linestyle="--",
            label="Validation reconstruction",
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title(run_name)
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURE_DIRECTORY / f"{run_name}_loss.png", dpi=160)
    plt.close(figure)


def _plot_accuracy_history(history: pd.DataFrame, run_name: str) -> None:
    """한 final run의 training·validation classification accuracy를 저장한다."""
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(history["epoch"], history["training_accuracy"], label="Training")
    axis.plot(history["epoch"], history["validation_accuracy"], label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Classification accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.set_title(run_name)
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURE_DIRECTORY / f"{run_name}_accuracy.png", dpi=160)
    plt.close(figure)
