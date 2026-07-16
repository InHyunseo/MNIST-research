#!/usr/bin/env python
"""LeNet 발표용 그림 3종을 생성한다 (run_lenet.py 실행 후 checkpoint 필요).

1. overlap_examples.png   — 세 class 조합의 Low/Middle/High 합성 입력 (3×3)
2. overlap_accuracy.png    — overlap level별 Top-2 exact-match (seed 평균 ± std)
3. pair_accuracy_high.png  — High overlap의 45개 class-pair 정확도 heatmap

기존 3-model reporting.py를 건드리지 않고 LeNet 단독 결과만 그린다.

실행:
    .venv/bin/python scripts/make_figures.py [--config PATH] [--device cpu|cuda]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mnist_overlap.analysis import collect_prediction_arrays  # noqa: E402
from mnist_overlap.config import (  # noqa: E402
    CHECKPOINT_DIR,
    CLASS_COUNT,
    FIGURE_DIR,
    DEFAULT_CONFIG_PATH,
    create_output_directories,
    load_config,
)
from mnist_overlap.data import ControlledOverlapMnistDataset  # noqa: E402
from mnist_overlap.metrics import (  # noqa: E402
    class_pair_accuracy,
    classification_metrics,
    exact_match_per_sample,
    sample_deviation,
)
from mnist_overlap.models import create_model  # noqa: E402
from mnist_overlap.training import load_checkpoint, select_device  # noqa: E402

import torch  # noqa: E402

MODEL_NAME = "lenet"
OVERLAP_ORDER = ("low", "middle", "high")
OVERLAP_EXAMPLE_CLASS_PAIRS = ((1, 0), (4, 7), (3, 8))
COLOR = "#4C78A8"


def main() -> None:
    parser = argparse.ArgumentParser(description="LeNet 발표용 그림을 생성합니다.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    # 파일·파이프로 리다이렉트해도 진행 로그가 실시간으로 보이게 한다.
    sys.stdout.reconfigure(line_buffering=True)

    config = load_config(args.config)
    create_output_directories()
    device = select_device(args.device)
    seeds = list(config.project.training_seeds)

    dataset = ControlledOverlapMnistDataset("test", config)
    predictions_by_seed = _collect_predictions(config, dataset, seeds, device)

    example_path = FIGURE_DIR / "overlap_examples.png"
    accuracy_path = FIGURE_DIR / "overlap_accuracy.png"
    pair_path = FIGURE_DIR / "pair_accuracy_high.png"

    _figure_overlap_examples(dataset, config.report.figure_dpi, example_path)
    _figure_overlap_accuracy(
        predictions_by_seed, seeds, config.report.figure_dpi, accuracy_path
    )
    _figure_pair_accuracy_high(
        predictions_by_seed, seeds, config.report.figure_dpi, pair_path
    )
    for path in (example_path, accuracy_path, pair_path):
        print(f"생성: {path}")


def _collect_predictions(config, dataset, seeds, device):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.evaluation.batch_size,
        shuffle=False,
        num_workers=config.train.data_loader_workers,
    )
    result = {}
    for seed in seeds:
        model = create_model(MODEL_NAME, config)
        load_checkpoint(model, CHECKPOINT_DIR / f"{MODEL_NAME}_seed_{seed}.pt", device, config)
        result[seed] = collect_prediction_arrays(model, loader, device)
    return result


def _figure_overlap_examples(dataset, dpi, path):
    """세 class 조합의 Low/Middle/High 합성 입력을 3×3 grid로 저장한다."""
    figure, axes = plt.subplots(
        len(OVERLAP_EXAMPLE_CLASS_PAIRS),
        len(OVERLAP_ORDER),
        figsize=(7.5, 7.5),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, class_pair in enumerate(OVERLAP_EXAMPLE_CLASS_PAIRS):
        indices = _select_class_pair_examples(dataset, class_pair)
        for column_index, (level, dataset_index) in enumerate(zip(OVERLAP_ORDER, indices)):
            sample = dataset[dataset_index]
            axis = axes[row_index, column_index]
            axis.imshow(sample["image"][0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(level.title())
            if column_index == 0:
                axis.set_ylabel(f"{class_pair[0]} + {class_pair[1]}", fontsize=11)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def _figure_overlap_accuracy(predictions_by_seed, seeds, dpi, path):
    """overlap level별 exact-match 평균과 seed 표준편차를 막대+error bar로 저장한다."""
    positions = np.arange(len(OVERLAP_ORDER), dtype=float)
    means, errors = [], []
    for level in OVERLAP_ORDER:
        seed_values = []
        for seed in seeds:
            preds = predictions_by_seed[seed]
            mask = preds["overlap_level"] == level
            seed_values.append(
                classification_metrics(preds["logits"][mask], preds["labels"][mask])[
                    "exact_match"
                ]
            )
        seed_values = np.asarray(seed_values)
        means.append(float(seed_values.mean()))
        errors.append(sample_deviation(seed_values))

    figure, axis = plt.subplots(figsize=(6.0, 4.4))
    axis.bar(positions, means, yerr=errors, width=0.55, color=COLOR, capsize=4)
    for position, mean in zip(positions, means):
        axis.text(position, mean + 0.01, f"{mean * 100:.1f}%", ha="center", fontsize=10)
    axis.set_xticks(positions, [level.title() for level in OVERLAP_ORDER])
    axis.set_ylim(0.0, 1.0)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.set_ylabel("Top-2 exact-match")
    axis.set_title(f"LeNet accuracy by overlap ({len(seeds)} seeds)")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _figure_pair_accuracy_high(predictions_by_seed, seeds, dpi, path):
    """High overlap의 unordered class-pair 정확도를 seed 평균 heatmap으로 저장한다."""
    reference = predictions_by_seed[seeds[0]]
    high_mask = reference["overlap_level"] == "high"
    label_first = reference["label_first"][high_mask]
    label_second = reference["label_second"][high_mask]
    matrices = []
    for seed in seeds:
        preds = predictions_by_seed[seed]
        correct = exact_match_per_sample(
            torch.from_numpy(preds["logits"]), torch.from_numpy(preds["labels"])
        ).numpy().astype(np.float64)
        matrices.append(
            class_pair_accuracy(correct[high_mask], label_first, label_second, CLASS_COUNT)
        )
    mean_matrix = np.nanmean(np.stack(matrices), axis=0)

    figure, axis = plt.subplots(figsize=(6.4, 5.4))
    image = axis.imshow(mean_matrix, cmap="viridis", vmin=np.nanmin(mean_matrix), vmax=np.nanmax(mean_matrix))
    axis.set_xticks(range(CLASS_COUNT))
    axis.set_yticks(range(CLASS_COUNT))
    axis.set_xlabel("Digit class")
    axis.set_ylabel("Digit class")
    axis.set_title("LeNet High-overlap accuracy by digit pair")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _select_class_pair_examples(dataset, class_pair):
    """지정한 unordered class 조합의 첫 완전한 paired test index 세 개를 반환한다."""
    label_first = dataset.manifest["label_first"]
    label_second = dataset.manifest["label_second"]
    pair_ids = dataset.manifest["pair_id"]
    overlap_levels = dataset.manifest["overlap_level"]
    first_class, second_class = class_pair
    matches = (
        ((label_first == first_class) & (label_second == second_class))
        | ((label_first == second_class) & (label_second == first_class))
    )
    for pair_id in dict.fromkeys(int(value) for value in pair_ids[matches]):
        selected = []
        for level in OVERLAP_ORDER:
            indices = np.flatnonzero((pair_ids == pair_id) & (overlap_levels == level))
            if len(indices) != 1:
                break
            selected.append(int(indices[0]))
        if len(selected) == len(OVERLAP_ORDER):
            return selected
    raise ValueError(f"Class pair {class_pair}의 완전한 Low/Middle/High sample이 없습니다.")


if __name__ == "__main__":
    main()
