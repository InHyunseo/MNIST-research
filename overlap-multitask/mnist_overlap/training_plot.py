"""Seed별 CSV 학습 이력에서 validation accuracy·loss spaghetti plot을 만든다."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def draw_training_curves(
    history_paths: list[Path],
    accuracy_column: str,
    loss_column: str,
    line_color: str,
    mean_color: str,
    output_path: Path,
    figure_dpi: int,
) -> None:
    """Seed 곡선과 epoch별 평균선을 accuracy·loss 두 패널에 저장한다."""
    histories = [
        _read_history(path, accuracy_column, loss_column)
        for path in history_paths
    ]
    maximum_epoch = max(int(history["epoch"][-1]) for history in histories)
    accuracy_values = _align_histories(histories, "accuracy", maximum_epoch)
    loss_values = _align_histories(histories, "loss", maximum_epoch)
    epochs = np.arange(1, maximum_epoch + 1)

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharex=True)
    for axis, title, values in zip(
        axes,
        ("Val Accuracy", "Val Loss"),
        (accuracy_values, loss_values),
    ):
        for seed_values in values:
            axis.plot(epochs, seed_values, color=line_color, alpha=0.28, linewidth=1.1)
        mean_line, = axis.plot(
            epochs,
            np.nanmean(values, axis=0),
            color=mean_color,
            linewidth=3.0,
            label="Mean",
        )
        axis.set_title(title)
        axis.set_ylim(0.0, 1.0)
        axis.set_yticks(np.linspace(0.0, 1.0, 6))
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.18)

    axes[0].legend(handles=[mean_line], frameon=False, loc="lower right")
    figure.supxlabel("Epoch")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(figure)


def _read_history(
    path: Path,
    accuracy_column: str,
    loss_column: str,
) -> dict[str, np.ndarray]:
    """CSV에서 epoch와 지정 validation 열을 float array로 읽는다."""
    if not path.exists():
        raise FileNotFoundError(f"학습 이력이 없습니다: {path}")
    with path.open(newline="", encoding="utf-8") as history_file:
        rows = list(csv.DictReader(history_file))
    if not rows:
        raise ValueError(f"학습 이력이 비어 있습니다: {path}")
    required_columns = {"epoch", accuracy_column, loss_column}
    if not required_columns.issubset(rows[0]):
        raise ValueError(f"학습 이력 열이 부족합니다: {path}")
    return {
        "epoch": np.asarray([int(row["epoch"]) for row in rows]),
        "accuracy": np.asarray([float(row[accuracy_column]) for row in rows]),
        "loss": np.asarray([float(row[loss_column]) for row in rows]),
    }


def _align_histories(
    histories: list[dict[str, np.ndarray]],
    value_name: str,
    maximum_epoch: int,
) -> np.ndarray:
    """Early stopping 길이가 다른 seed 이력을 NaN padding한 행렬로 맞춘다."""
    aligned = np.full((len(histories), maximum_epoch), np.nan, dtype=np.float64)
    for row_index, history in enumerate(histories):
        epoch_indices = history["epoch"].astype(int) - 1
        aligned[row_index, epoch_indices] = history[value_name]
    return aligned
