"""평가 log를 최종 표(`results/tables/`), figure(PNG 3종), `results/summary.md`로 변환한다."""

from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from .config import (
    FIGURE_DIR,
    METRIC_LOG_DIR,
    SUMMARY_PATH,
    TABLE_DIR,
    ExperimentConfig,
)
from .data import ControlledOverlapMnistDataset
from .evaluation import read_csv_rows, write_csv_rows
from .metrics import sample_deviation


MODEL_LABELS = {
    "lenet": "LeNet",
    "shared_attention": "Shared attention",
    "class_attention": "Class attention",
}
MODEL_COLORS = {
    "lenet": "#4C78A8",
    "shared_attention": "#F58518",
    "class_attention": "#54A24B",
}
OVERLAP_ORDER = ("low", "middle", "high")
OVERLAP_EXAMPLE_CLASS_PAIRS = ((1, 0), (4, 7), (3, 8))

FINAL_TABLE_FILES = (
    "model_comparison.csv",
    "hierarchical_intervals.csv",
    "attention_metrics.csv",
    "per_class_recall.csv",
    "seed_effects.csv",
    "training_stability.csv",
    "model_costs.csv",
)


def create_report(config: ExperimentConfig) -> list[Path]:
    """저장된 전체 실험 log에서 최종 결과물(표, figure, summary)을 다시 생성한다."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    comparison_path = TABLE_DIR / "model_comparison.csv"
    comparison_rows = create_model_comparison_table(
        METRIC_LOG_DIR / "model_metrics.csv",
        comparison_path,
    )
    generated_paths = [comparison_path]
    generated_paths.extend(create_final_analysis_tables())

    if config.report.save_png:
        generated_paths.extend(_create_figures(config))

    create_summary(config, comparison_rows, SUMMARY_PATH)
    generated_paths.append(SUMMARY_PATH)
    return generated_paths


def create_model_comparison_table(
    source_path: Path,
    destination_path: Path,
) -> list[dict[str, Any]]:
    """Overlap별 분류 성능의 seed 평균과 표본 표준편차 표를 만든다.

    Test-pair 불확실성은 이 표가 아닌 hierarchical interval 표에 둔다.
    """
    metric_rows = read_csv_rows(source_path)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in metric_rows:
        if row["overlap_level"] in OVERLAP_ORDER:
            grouped[(row["model"], row["overlap_level"])].append(row)

    comparison_rows = []
    for model_name in MODEL_LABELS:
        for overlap_level in OVERLAP_ORDER:
            rows = grouped[(model_name, overlap_level)]
            if not rows:
                raise ValueError(
                    f"Model {model_name}, overlap {overlap_level} metric이 없습니다."
                )
            exact_values = np.asarray([float(row["exact_match"]) for row in rows])
            f1_values = np.asarray([float(row["macro_f1"]) for row in rows])
            comparison_rows.append({
                "model": model_name,
                "overlap_level": overlap_level,
                "exact_match_mean": float(exact_values.mean()),
                "exact_match_standard_deviation": sample_deviation(exact_values),
                "macro_f1_mean": float(f1_values.mean()),
                "macro_f1_standard_deviation": sample_deviation(f1_values),
                "seed_count": len(rows),
            })
    write_csv_rows(destination_path, comparison_rows)
    return comparison_rows


def create_final_analysis_tables() -> list[Path]:
    """`outputs/metrics/`의 분석 CSV를 같은 이름으로 `results/tables/`에 복사한다.

    이 단계는 값을 변경하지 않아 log와 최종 표의 provenance가 유지된다.
    """
    generated_paths = []
    for file_name in FINAL_TABLE_FILES[1:]:
        source_path = METRIC_LOG_DIR / file_name
        destination_path = TABLE_DIR / file_name
        if not source_path.exists():
            raise FileNotFoundError(
                f"최종 표를 만들 metric log가 없습니다: {source_path}"
            )
        shutil.copyfile(source_path, destination_path)
        generated_paths.append(destination_path)
    return generated_paths


def _create_figures(config: ExperimentConfig) -> list[Path]:
    """최종 분류·attention·overlap setting figure 세 개를 생성한다."""
    classification_path = FIGURE_DIR / "classification_performance.png"
    attention_path = FIGURE_DIR / "attention_behavior.png"
    overlap_path = FIGURE_DIR / "overlap_examples.png"
    dpi = config.report.figure_dpi
    create_classification_performance_figure(
        read_csv_rows(METRIC_LOG_DIR / "model_metrics.csv"),
        read_csv_rows(METRIC_LOG_DIR / "hierarchical_intervals.csv"),
        read_csv_rows(METRIC_LOG_DIR / "seed_effects.csv"),
        classification_path,
        dpi,
    )
    create_attention_behavior_figure(
        read_csv_rows(METRIC_LOG_DIR / "attention_metrics.csv"),
        attention_path,
        dpi,
    )
    create_overlap_examples_figure(config, overlap_path)
    return [classification_path, attention_path, overlap_path]


# -----------------------------------------------------------------------------
# Markdown summary
# -----------------------------------------------------------------------------


def create_summary(
    config: ExperimentConfig,
    comparison_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    """설정, 핵심 결과, 통계표, figure 설명, 해석 범위를 `results/summary.md`로 저장한다."""
    hierarchical_rows = read_csv_rows(TABLE_DIR / "hierarchical_intervals.csv")
    attention_rows = read_csv_rows(TABLE_DIR / "attention_metrics.csv")
    per_class_rows = read_csv_rows(TABLE_DIR / "per_class_recall.csv")
    stability_rows = read_csv_rows(TABLE_DIR / "training_stability.csv")
    cost_rows = read_csv_rows(TABLE_DIR / "model_costs.csv")
    seed_count = len(config.project.training_seeds)
    confidence_percent = config.evaluation.confidence_level * 100.0
    lines = [
        "# MNIST Overlap Attention 결과 요약",
        "",
        "## 실험 설정",
        "",
        f"- 학습 seed: {list(config.project.training_seeds)} ({seed_count} runs/model)",
        f"- 최대 epoch: {config.train.maximum_epochs}, early-stopping patience: "
        f"{config.train.early_stopping_patience}",
        f"- Train: {config.dataset.train_samples:,} images",
        f"- Validation: {config.dataset.validation_pairs:,} pairs, "
        f"{config.dataset.validation_pairs * 3:,} images",
        f"- Test: {config.dataset.test_pairs:,} pairs, "
        f"{config.dataset.test_pairs * 3:,} images",
        "- Checkpoint 선택: validation Top-2 exact-match early stopping",
        "- Test 주 지표: 두 정답 class를 모두 맞힌 Top-2 exact-match",
        "",
    ]
    lines.extend(_key_findings(hierarchical_rows, confidence_percent))
    lines.extend(_performance_section(comparison_rows, seed_count))
    lines.extend(_hierarchical_section(hierarchical_rows, confidence_percent))
    lines.extend(_attention_section(attention_rows, seed_count))
    lines.extend(_per_class_section(per_class_rows, seed_count))
    lines.extend(_stability_and_cost_section(stability_rows, cost_rows))
    lines.extend([
        "## Figure 설명",
        "",
        "### Controlled Overlap",
        "",
        "![Controlled overlap examples](figures/overlap_examples.png)",
        "",
        "각 행은 같은 두 MNIST 원본, 중심, 이동 방향을 공유한다. 열 사이에서는 "
        "displacement만 바뀌므로 Low·Middle·High 성능 차이를 overlap 조작의 결과로 "
        "비교할 수 있다.",
        "",
        "### Classification Performance",
        "",
        "![Classification performance](figures/classification_performance.png)",
        "",
        "왼쪽의 작은 점은 개별 training seed, 굵은 점은 전체 평균, error bar는 seed와 "
        f"pair를 함께 복원추출한 {confidence_percent:.0f}% hierarchical bootstrap CI다. "
        "오른쪽은 High-overlap 모델 차이와 Class−LeNet 효과의 High−Low 변화를 같은 "
        "방식으로 표시한다.",
        "",
        "### Attention Behavior",
        "",
        "![Attention behavior](figures/attention_behavior.png)",
        "",
        "AUPRC와 IoU는 attention map과 실제 digit stroke mask의 정렬을, Selectivity는 "
        "두 class map이 서로의 고유 획 영역을 구분하는 정도를 나타낸다. Permutation "
        "Drop은 class-map 연결을 순환 교환했을 때 감소한 exact-match다. 작은 점은 "
        "seed별 값이고 error bar는 seed 표본 표준편차다.",
        "",
        "## 해석 범위",
        "",
        "결과는 서로 다른 class의 MNIST 숫자 두 개를 `76×76` canvas에 maximum으로 "
        "합성한 통제 조건에 한정된다. 모델 효과는 hierarchical 신뢰구간과 seed별 "
        "방향을 함께 해석하며, attention 정렬 지표가 낮다면 분류 성능만으로 공간적 "
        "설명 가능성을 주장하지 않는다. 실제 객체 가림이나 다른 데이터셋으로의 직접 "
        "일반화는 이 실험의 주장 범위가 아니다.",
        "",
        "## 결과 파일",
        "",
        "- [모델 비교표](tables/model_comparison.csv)",
        "- [Hierarchical interval 표](tables/hierarchical_intervals.csv)",
        "- [Attention 분석표](tables/attention_metrics.csv)",
        "- [숫자별 recall 표](tables/per_class_recall.csv)",
        "- [Seed별 paired 효과표](tables/seed_effects.csv)",
        "- [학습 안정성표](tables/training_stability.csv)",
        "- [모델 비용표](tables/model_costs.csv)",
        "- 45개 High-overlap class-pair 값은 `outputs/metrics/`에만 보관한다.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _key_findings(
    hierarchical_rows: list[dict[str, str]],
    confidence_percent: float,
) -> list[str]:
    """Primary hierarchical 비교의 estimate·CI·0 포함 여부로 핵심 결과 문장을 만든다."""
    lookup = {row["comparison"]: row for row in hierarchical_rows}
    comparisons = (
        ("lenet_low_minus_high", "LeNet Low − High"),
        ("class_attention_minus_lenet_high", "High: Class − LeNet"),
        ("class_attention_minus_shared_high", "High: Class − Shared"),
        (
            "class_attention_vs_lenet_high_low_difference",
            "(Class − LeNet): High − Low",
        ),
    )
    lines = ["## 핵심 결과", ""]
    for comparison_name, label in comparisons:
        row = lookup[comparison_name]
        lower = float(row["confidence_lower"])
        upper = float(row["confidence_upper"])
        zero_text = "0을 포함한다" if lower <= 0.0 <= upper else "0을 포함하지 않는다"
        lines.append(
            f"- {label}: {_format_interval(row)}; "
            f"{confidence_percent:.0f}% CI는 {zero_text}."
        )
    lines.append("")
    return lines


def _performance_section(
    rows: list[dict[str, Any]],
    seed_count: int,
) -> list[str]:
    """Overlap별 exact-match 평균±SD Markdown 표를 만든다."""
    lookup = {(row["model"], row["overlap_level"]): row for row in rows}
    lines = [
        "## 분류 성능",
        "",
        f"값은 {seed_count}개 training seed의 평균 ± 표본 표준편차이며 단위는 %다.",
        "",
        "| Model | Low | Middle | High |",
        "|---|---:|---:|---:|",
    ]
    for model_name, label in MODEL_LABELS.items():
        values = []
        for overlap_level in OVERLAP_ORDER:
            row = lookup[(model_name, overlap_level)]
            values.append(
                f"{float(row['exact_match_mean']) * 100:.2f} ± "
                f"{float(row['exact_match_standard_deviation']) * 100:.2f}"
            )
        lines.append(f"| {label} | {' | '.join(values)} |")
    lines.append("")
    return lines


def _hierarchical_section(
    rows: list[dict[str, str]],
    confidence_percent: float,
) -> list[str]:
    """Primary effect의 2단계 bootstrap Markdown 표를 만든다."""
    labels = {
        "lenet_low_minus_high": "LeNet: Low − High",
        "shared_attention_low_minus_high": "Shared: Low − High",
        "class_attention_low_minus_high": "Class: Low − High",
        "class_attention_minus_lenet_high": "High: Class − LeNet",
        "class_attention_minus_shared_high": "High: Class − Shared",
        "class_attention_vs_lenet_high_low_difference": (
            "(Class − LeNet): High − Low"
        ),
    }
    lookup = {row["comparison"]: row for row in rows}
    lines = [
        "## Hierarchical bootstrap",
        "",
        "Training seed와 pair ID를 두 단계로 복원추출하며 모델과 overlap 사이의 "
        f"대응은 유지한다. 반복 수는 {int(rows[0]['bootstrap_iterations']):,}회, "
        f"신뢰수준은 {confidence_percent:.0f}%다.",
        "",
        "| Comparison | Estimate (pp) | CI (pp) |",
        "|---|---:|---:|",
    ]
    for comparison_name, label in labels.items():
        row = lookup[comparison_name]
        lines.append(
            f"| {label} | {float(row['estimate']) * 100:+.2f} | "
            f"[{float(row['confidence_lower']) * 100:+.2f}, "
            f"{float(row['confidence_upper']) * 100:+.2f}] |"
        )
    lines.append("")
    return lines


def _attention_section(
    rows: list[dict[str, str]],
    seed_count: int,
) -> list[str]:
    """Attention mechanism metric의 seed 평균±SD 표를 만든다."""
    lines = [
        "## Attention 분석",
        "",
        f"값은 {seed_count}개 seed의 평균 ± 표본 표준편차다. IoU threshold는 각 "
        "checkpoint의 validation에서 선택하고 test에는 고정했다.",
        "",
        "| Model | AUPRC | IoU | Selectivity | Permutation Drop |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_name in ("shared_attention", "class_attention"):
        model_rows = [row for row in rows if row["model"] == model_name]
        values = []
        for field_name in (
            "test_average_precision",
            "test_iou",
            "test_cross_selectivity",
            "permutation_accuracy_drop",
        ):
            field_values = np.asarray([
                float(row[field_name])
                for row in model_rows
                if row.get(field_name, "") != ""
                and np.isfinite(float(row[field_name]))
            ])
            values.append(
                "—" if not len(field_values) else (
                    f"{field_values.mean():.4f} ± {sample_deviation(field_values):.4f}"
                )
            )
        lines.append(f"| {MODEL_LABELS[model_name]} | {' | '.join(values)} |")
    lines.append("")
    return lines


def _per_class_section(
    rows: list[dict[str, str]],
    seed_count: int,
) -> list[str]:
    """숫자 0–9 recall과 Class−LeNet 차이의 Appendix 표를 만든다."""
    lines = [
        "## Appendix: 숫자별 recall",
        "",
        f"전체 test set에서 숫자가 정답일 때 Top-2 예측에 포함된 비율이다. 값은 "
        f"{seed_count}개 seed의 평균 ± 표본 표준편차이며 단위는 %다.",
        "",
        "| Digit | LeNet | Shared | Class | Class − LeNet |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = []
        for model_name in ("lenet", "shared_attention", "class_attention"):
            values.append(
                f"{float(row[f'{model_name}_mean']) * 100:.2f} ± "
                f"{float(row[f'{model_name}_standard_deviation']) * 100:.2f}"
            )
        delta = (
            f"{float(row['class_attention_minus_lenet_mean']) * 100:+.2f} ± "
            f"{float(row['class_attention_minus_lenet_standard_deviation']) * 100:.2f}"
        )
        lines.append(f"| {row['class']} | {' | '.join(values)} | {delta} |")
    lines.append("")
    return lines


def _stability_and_cost_section(
    stability_rows: list[dict[str, str]],
    cost_rows: list[dict[str, str]],
) -> list[str]:
    """Early-stopping 진단 요약과 model cost 표를 만든다."""
    lines = [
        "## Appendix: 학습 안정성과 비용",
        "",
        "| Model | Epoch range | Reached maximum | Parameters | MACs |",
        "|---|---:|---:|---:|---:|",
    ]
    cost_lookup = {row["model"]: row for row in cost_rows}
    for model_name, label in MODEL_LABELS.items():
        model_stability = [row for row in stability_rows if row["model"] == model_name]
        epochs = [int(row["epochs_run"]) for row in model_stability]
        reached_count = sum(row["reached_maximum_epochs"] == "True" for row in model_stability)
        cost = cost_lookup[model_name]
        lines.append(
            f"| {label} | {min(epochs)}–{max(epochs)} | "
            f"{reached_count}/{len(epochs)} | {int(cost['parameters']):,} | "
            f"{int(cost['multiply_accumulate_operations']):,} |"
        )
    lines.append("")
    return lines


def _format_interval(row: dict[str, str]) -> str:
    """효과 추정값과 CI를 `+0.00 pp [−0.00, +0.00]` 문자열로 만든다."""
    estimate = float(row["estimate"]) * 100.0
    lower = float(row["confidence_lower"]) * 100.0
    upper = float(row["confidence_upper"]) * 100.0
    return f"{estimate:+.2f} pp [{lower:+.2f}, {upper:+.2f}]"


# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------


def create_classification_performance_figure(
    model_metric_rows: list[dict[str, str]],
    hierarchical_rows: list[dict[str, str]],
    seed_effect_rows: list[dict[str, str]],
    path: Path,
    dpi: int,
) -> None:
    """Test Accuracy와 주요 모델 효과를 한 장의 2-panel figure로 저장한다.

    점은 개별 seed, 굵은 표식은 전체 평균, error bar는 hierarchical bootstrap CI다.
    """
    interval_lookup = {
        row["comparison"]: row
        for row in hierarchical_rows
    }
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    accuracy_axis, effect_axis = axes
    overlap_positions = np.arange(len(OVERLAP_ORDER), dtype=float)
    model_offsets = np.linspace(-0.16, 0.16, len(MODEL_LABELS))

    for model_index, (model_name, model_label) in enumerate(MODEL_LABELS.items()):
        displayed_positions = overlap_positions + model_offsets[model_index]
        means = []
        lower_errors = []
        upper_errors = []

        for overlap_index, overlap_level in enumerate(OVERLAP_ORDER):
            seed_values = np.asarray([
                float(row["exact_match"])
                for row in model_metric_rows
                if row["model"] == model_name
                and row["overlap_level"] == overlap_level
            ])
            interval = interval_lookup[f"{model_name}_{overlap_level}_accuracy"]
            mean = float(interval["estimate"])
            lower = float(interval["confidence_lower"])
            upper = float(interval["confidence_upper"])
            means.append(mean)
            lower_errors.append(mean - lower)
            upper_errors.append(upper - mean)

            # Seed 점은 x방향으로만 작게 펼쳐 중복값도 개별 실행으로 보이게 한다.
            jitter = np.linspace(-0.035, 0.035, len(seed_values))
            accuracy_axis.scatter(
                np.full(len(seed_values), displayed_positions[overlap_index]) + jitter,
                seed_values,
                s=16,
                alpha=0.45,
                color=MODEL_COLORS[model_name],
                linewidths=0,
            )

        accuracy_axis.errorbar(
            displayed_positions,
            means,
            yerr=np.asarray([lower_errors, upper_errors]),
            marker="o",
            markersize=6,
            linewidth=1.6,
            capsize=3,
            color=MODEL_COLORS[model_name],
            label=model_label,
        )

    accuracy_axis.set_title("Test Accuracy")
    accuracy_axis.set_xticks(overlap_positions, [level.title() for level in OVERLAP_ORDER])
    accuracy_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    accuracy_axis.grid(axis="y", alpha=0.2)
    accuracy_axis.legend(frameon=False, fontsize=9)

    effect_names = (
        "class_attention_minus_lenet_high",
        "class_attention_minus_shared_high",
        "class_attention_vs_lenet_high_low_difference",
    )
    effect_labels = ("Class − LeNet", "Class − Shared", "High − Low")
    effect_positions = np.arange(len(effect_names), dtype=float)
    for effect_index, comparison_name in enumerate(effect_names):
        seed_values = np.asarray([
            float(row["estimate"])
            for row in seed_effect_rows
            if row["comparison"] == comparison_name
        ])
        interval = interval_lookup[comparison_name]
        estimate = float(interval["estimate"])
        lower = float(interval["confidence_lower"])
        upper = float(interval["confidence_upper"])
        jitter = np.linspace(-0.06, 0.06, len(seed_values))
        effect_axis.scatter(
            np.full(len(seed_values), effect_positions[effect_index]) + jitter,
            seed_values,
            s=18,
            alpha=0.5,
            color="#4C4C4C",
            linewidths=0,
        )
        effect_axis.errorbar(
            effect_positions[effect_index],
            estimate,
            yerr=np.asarray([[estimate - lower], [upper - estimate]]),
            marker="o",
            markersize=7,
            capsize=4,
            color="#111111",
        )

    effect_axis.axhline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    effect_axis.set_title("Effect Size")
    effect_axis.set_xticks(effect_positions, effect_labels, rotation=12, ha="right")
    effect_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    effect_axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def create_attention_behavior_figure(
    rows: list[dict[str, str]],
    path: Path,
    dpi: int,
) -> None:
    """Attention 정렬(AUPRC·IoU)과 class-map 사용 정도(Selectivity·Permutation Drop)를
    2×2 정량 panel로 저장한다."""
    panels = (
        ("AUPRC", "test_average_precision", ("shared_attention", "class_attention")),
        ("IoU", "test_iou", ("shared_attention", "class_attention")),
        ("Selectivity", "test_cross_selectivity", ("class_attention",)),
        ("Permutation Drop", "permutation_accuracy_drop", ("class_attention",)),
    )
    figure, axes = plt.subplots(2, 2, figsize=(8.4, 7.0))

    for axis, (title, field_name, model_names) in zip(axes.flat, panels):
        for position, model_name in enumerate(model_names):
            raw_values = np.asarray([
                float(row[field_name])
                for row in rows
                if row["model"] == model_name and row.get(field_name, "") != ""
            ])
            values = raw_values[np.isfinite(raw_values)]
            if not len(values):
                axis.text(
                    position,
                    0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    transform=axis.get_xaxis_transform(),
                )
                continue
            jitter = np.linspace(-0.06, 0.06, len(values))
            axis.scatter(
                np.full(len(values), position) + jitter,
                values,
                s=20,
                alpha=0.5,
                color=MODEL_COLORS[model_name],
                linewidths=0,
            )
            axis.errorbar(
                position,
                float(values.mean()),
                yerr=sample_deviation(values),
                marker="o",
                markersize=7,
                capsize=4,
                color=MODEL_COLORS[model_name],
            )

        axis.set_title(title)
        axis.set_xticks(
            range(len(model_names)),
            [MODEL_LABELS[name] for name in model_names],
        )
        axis.grid(axis="y", alpha=0.2)
        if title == "Permutation Drop":
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def create_overlap_examples_figure(
    config: ExperimentConfig,
    path: Path,
) -> None:
    """세 숫자 조합의 Low·Middle·High 합성 입력을 3×3 grid로 저장한다.

    각 행은 원본 두 장, pair center, 이동 방향이 같은 실제 paired test sample이다.
    """
    dataset = ControlledOverlapMnistDataset("test", config)
    figure, axes = plt.subplots(
        len(OVERLAP_EXAMPLE_CLASS_PAIRS),
        len(OVERLAP_ORDER),
        figsize=(7.5, 7.5),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, class_pair in enumerate(OVERLAP_EXAMPLE_CLASS_PAIRS):
        example_indices = _select_class_pair_examples(dataset, class_pair)
        for column_index, (overlap_level, dataset_index) in enumerate(zip(
            OVERLAP_ORDER,
            example_indices,
        )):
            sample = dataset[dataset_index]
            axis = axes[row_index, column_index]
            axis.imshow(sample["image"][0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(overlap_level.title())
            if column_index == 0:
                axis.set_ylabel(f"{class_pair[0]} + {class_pair[1]}", fontsize=11)
    figure.savefig(path, dpi=config.report.figure_dpi)
    plt.close(figure)


def _select_class_pair_examples(
    dataset: ControlledOverlapMnistDataset,
    class_pair: tuple[int, int],
) -> list[int]:
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
    matching_pair_ids = pair_ids[matches]

    for pair_id in dict.fromkeys(int(value) for value in matching_pair_ids):
        selected_indices = []
        for overlap_level in OVERLAP_ORDER:
            indices = np.flatnonzero(
                (pair_ids == pair_id) & (overlap_levels == overlap_level)
            )
            if len(indices) != 1:
                break
            selected_indices.append(int(indices[0]))
        if len(selected_indices) == len(OVERLAP_ORDER):
            return selected_indices

    raise ValueError(
        f"Class pair {class_pair}의 완전한 Low/Middle/High sample을 찾지 못했습니다."
    )
