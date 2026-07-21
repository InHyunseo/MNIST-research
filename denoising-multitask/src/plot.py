"""
저장된 n-MNIST 최종 결과와 history에서 필수 figure를 생성한다.

입력:
    - outputs/results.csv
    - outputs/histories의 최종 run별 CSV

출력:
    - Accuracy comparison, paired delta, seed-overlaid loss·accuracy PNG

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
from src.experiment import (
    HISTORY_DIRECTORY,
    RANDOM_SEEDS,
    RESULT_COLUMNS,
    RESULTS_PATH,
)


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
    """각 noise·condition에 30개 seed 결과가 모두 존재하는지 확인한다."""
    expected_seeds = set(RANDOM_SEEDS)
    for noise_type in NOISE_TYPES:
        for condition in CONDITIONS:
            condition_results = results.loc[
                (results["noise_type"] == noise_type)
                & (results["condition"] == condition)
            ]
            available_seeds = set(condition_results["random_seed"].astype(int))
            if (
                available_seeds != expected_seeds
                or len(condition_results) != len(RANDOM_SEEDS)
            ):
                missing_seeds = sorted(expected_seeds - available_seeds)
                unexpected_seeds = sorted(available_seeds - expected_seeds)
                raise RuntimeError(
                    f"완료되지 않은 결과입니다: noise={noise_type}, "
                    f"condition={condition}, missing={missing_seeds}, "
                    f"unexpected={unexpected_seeds}"
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
    """Noise와 condition별 loss·accuracy를 공통 축 범위로 저장한다."""
    legacy_patterns = (
        "*_seed_*_loss.png",
        "*_seed_*_accuracy.png",
        "*_loss_spaghetti.png",
        "*_accuracy_spaghetti.png",
    )
    for legacy_pattern in legacy_patterns:
        for legacy_path in FIGURE_DIRECTORY.glob(legacy_pattern):
            legacy_path.unlink()

    histories_by_run = {}
    maximum_loss = 0.0
    minimum_accuracy = 1.0
    for noise_type in NOISE_TYPES:
        for condition in CONDITIONS:
            history_paths = [
                HISTORY_DIRECTORY
                / f"{noise_type}_{condition}_seed_{random_seed}.csv"
                for random_seed in RANDOM_SEEDS
            ]
            missing_history_paths = [
                history_path
                for history_path in history_paths
                if not history_path.exists()
            ]
            if missing_history_paths:
                raise FileNotFoundError(
                    "최종 학습 history가 완료되지 않았습니다: "
                    f"{missing_history_paths[0]}"
                )
            histories = []
            for history_path in history_paths:
                history = pd.read_csv(history_path)
                history["random_seed"] = int(history_path.stem.rsplit("_seed_", 1)[1])
                histories.append(history)
            combined_history = pd.concat(histories, ignore_index=True)
            histories_by_run[(noise_type, condition)] = combined_history
            run_maximum_loss = combined_history[
                ["training_total_loss", "validation_total_loss"]
            ].to_numpy(dtype=np.float64).max()
            run_minimum_accuracy = combined_history[
                ["training_accuracy", "validation_accuracy"]
            ].to_numpy(dtype=np.float64).min()
            maximum_loss = max(maximum_loss, float(run_maximum_loss))
            minimum_accuracy = min(minimum_accuracy, float(run_minimum_accuracy))

    if not np.isfinite(maximum_loss) or maximum_loss <= 0.0:
        raise ValueError("History loss가 올바른 양수가 아닙니다.")
    if not np.isfinite(minimum_accuracy) or not 0.0 <= minimum_accuracy <= 1.0:
        raise ValueError("History accuracy가 0과 1 사이의 유한한 값이 아닙니다.")
    loss_upper_limit = max(0.1, float(np.ceil(maximum_loss * 10.5) / 10.0))
    accuracy_lower_limit = max(
        0.0,
        float(np.floor((minimum_accuracy - 0.01) * 20.0) / 20.0),
    )
    for (noise_type, condition), history in histories_by_run.items():
        _plot_spaghetti_history(
            history,
            noise_type,
            condition,
            loss_upper_limit,
            accuracy_lower_limit,
        )


def _plot_spaghetti_history(
    history: pd.DataFrame,
    noise_type: str,
    condition: str,
    loss_upper_limit: float,
    accuracy_lower_limit: float,
) -> None:
    """Loss와 accuracy를 한 행에 나란히 놓고 seed 곡선을 겹쳐 그린다."""
    required_columns = {
        "epoch",
        "random_seed",
        "training_total_loss",
        "validation_total_loss",
        "training_accuracy",
        "validation_accuracy",
    }
    missing_columns = required_columns - set(history.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"History에 필요한 column이 없습니다: {missing_text}")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = {"training": "#4C78A8", "validation": "#F58518"}
    seed_count = history["random_seed"].nunique()
    individual_alpha = min(0.42, 0.95 / np.sqrt(seed_count))
    metrics = (
        (
            "Loss",
            "training_total_loss",
            "validation_total_loss",
            (0.0, loss_upper_limit),
        ),
        (
            "Accuracy",
            "training_accuracy",
            "validation_accuracy",
            (accuracy_lower_limit, 1.0),
        ),
    )
    for metric_index, (title, training_column, validation_column, limits) in enumerate(
        metrics
    ):
        axis = axes[metric_index]
        for _, seed_history in history.groupby("random_seed", sort=True):
            axis.plot(
                seed_history["epoch"],
                seed_history[training_column],
                color=colors["training"],
                linewidth=1.1,
                alpha=individual_alpha,
            )
            axis.plot(
                seed_history["epoch"],
                seed_history[validation_column],
                color=colors["validation"],
                linewidth=1.1,
                alpha=individual_alpha,
            )

        mean_history = history.groupby("epoch", as_index=False)[
            [training_column, validation_column]
        ].mean()
        axis.plot(
            mean_history["epoch"],
            mean_history[training_column],
            color=colors["training"],
            linewidth=2.0,
            label="Train",
        )
        axis.plot(
            mean_history["epoch"],
            mean_history[validation_column],
            color=colors["validation"],
            linewidth=2.0,
            label="Validation",
        )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylim(*limits)
        axis.grid(alpha=0.25)

    axes[1].legend()
    figure.suptitle(f"{NOISE_LABELS[noise_type]} — {CONDITION_LABELS[condition]}")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(
        FIGURE_DIRECTORY / f"{noise_type}_{condition}_history_spaghetti.png",
        dpi=160,
    )
    plt.close(figure)
