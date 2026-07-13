"""평가 log를 최종 표·그림·Markdown 결과로 변환한다.

입력:
    `logs/metrics/`의 평가 CSV와 전체 report config

출력:
    `results/tables/`, `results/figures/`, `results/summary.md`

연결:
    Report 실행 함수가 호출하며 그림의 세부 표현만 plotter에 위임한다.
"""

from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..configuration import (
    FIGURE_DIR,
    METRIC_LOG_DIR,
    SUMMARY_PATH,
    TABLE_DIR,
)
from .plotter import (
    MODEL_LABELS,
    OVERLAP_ORDER,
    create_attention_examples_figure,
    create_overlap_accuracy_figure,
    create_overlap_examples_figure,
    create_pair_accuracy_figure,
)


# -----------------------------------------------------------------------------
# 전체 결과 생성
# -----------------------------------------------------------------------------


def create_report(config: dict[str, Any]) -> list[Path]:
    """저장된 평가 log에서 모든 최종 결과물을 생성한다.

    입력:
        Report option과 실험 정보가 포함된 전체 config

    처리:
        비교표를 먼저 집계하고 figure, 분석표, Markdown 요약을 순서대로 생성한다.

    출력:
        생성된 결과 file 경로 목록
    """
    generated_paths = []
    comparison_path = TABLE_DIR / "model_comparison.csv"
    comparison_rows = create_model_comparison_table(
        METRIC_LOG_DIR / "model_metrics.csv",
        comparison_path,
    )
    generated_paths.append(comparison_path)

    if config["report"]["save_png"]:
        generated_paths.extend(
            _create_figures(config, comparison_rows)
        )

    generated_paths.extend(_create_analysis_tables())
    create_summary(config, comparison_rows, SUMMARY_PATH)
    generated_paths.append(SUMMARY_PATH)
    return generated_paths


def _create_figures(
    config: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
) -> list[Path]:
    """정량 결과와 고정 sample의 네 PNG를 생성한다.

    입력:
        전체 config와 model comparison 집계 row

    처리:
        Plotter의 accuracy, pair, attention, overlap example 함수를 순서대로 호출한다.

    출력:
        생성된 PNG 경로 네 개
    """
    dpi = int(config["report"]["figure_dpi"])
    accuracy_path = FIGURE_DIR / "overlap_accuracy.png"
    pair_accuracy_path = FIGURE_DIR / "pair_accuracy.png"
    attention_path = FIGURE_DIR / "attention_examples.png"
    overlap_example_path = FIGURE_DIR / "overlap_examples.png"

    create_overlap_accuracy_figure(comparison_rows, accuracy_path, dpi)
    create_pair_accuracy_figure(
        read_csv_rows(METRIC_LOG_DIR / "class_pair_accuracy_high.csv"),
        pair_accuracy_path,
        dpi,
    )
    create_attention_examples_figure(config, attention_path)
    create_overlap_examples_figure(config, overlap_example_path)
    return [
        accuracy_path,
        pair_accuracy_path,
        attention_path,
        overlap_example_path,
    ]


def _create_analysis_tables() -> list[Path]:
    """Mechanism, 통계, 계산 비용의 세 보조 결과표를 준비한다.

    입력:
        Evaluation이 `logs/metrics/`에 기록한 세 CSV

    처리:
        각 표의 원본 존재를 확인하고 설명된 최종 파일명으로 복사한다.

    출력:
        `results/tables/`에 생성된 CSV 경로 세 개
    """
    attention_path = copy_attention_metrics_table(
        METRIC_LOG_DIR / "attention_metrics.csv",
        TABLE_DIR / "attention_metrics.csv",
    )
    bootstrap_path = copy_bootstrap_intervals_table(
        METRIC_LOG_DIR / "bootstrap_intervals.csv",
        TABLE_DIR / "bootstrap_intervals.csv",
    )
    model_cost_path = copy_model_costs_table(
        METRIC_LOG_DIR / "model_costs.csv",
        TABLE_DIR / "model_costs.csv",
    )
    return [attention_path, bootstrap_path, model_cost_path]


# -----------------------------------------------------------------------------
# 결과 표 생성
# -----------------------------------------------------------------------------


def create_model_comparison_table(
    source_path: Path,
    destination_path: Path,
) -> list[dict[str, Any]]:
    """Overlap별 모델 성능 평균과 seed 표준편차 표를 생성한다.

    표현:
        모델과 Low/Middle/High별 Top-2 exact-match, Macro-F1의 평균·표준편차와
        실제 seed 수를 한 행에 기록한다.

    계산:
        동일한 test sample을 평가한 seed 0, 1, 2의 metric을 구분해 읽고 arithmetic
        mean과 `ddof=1` 표본 표준편차를 계산한다.

    목적:
        세 모델의 주 성능과 seed 변동을 함께 제공하는 필수 비교표다. Test Accuracy
        figure의 원본 숫자이며 모델 우열을 판단할 때 가장 먼저 확인한다.

    입력:
        Seed별 `model_metrics.csv`와 최종 `model_comparison.csv` 경로

    처리:
        Overall row를 제외하고 model·overlap별로 집계해 UTF-8 CSV로 저장한다.

    출력:
        저장된 집계 row 목록과 `model_comparison.csv`
    """
    metric_rows = read_csv_rows(source_path)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)

    for row in metric_rows:
        if row["overlap_level"] != "all":
            grouped[(row["model"], row["overlap_level"])].append(row)

    comparison_rows = []

    for model_name in MODEL_LABELS:
        for overlap_level in OVERLAP_ORDER:
            group = grouped[(model_name, overlap_level)]

            if not group:
                raise ValueError(
                    f"Model {model_name}, overlap {overlap_level} metric이 없습니다."
                )

            # 각 값은 같은 10,000개 test image를 독립 학습 seed로 평가한 결과다.
            exact_values = np.asarray([
                float(row["exact_match"])
                for row in group
            ])
            f1_values = np.asarray([
                float(row["macro_f1"])
                for row in group
            ])
            comparison_rows.append(
                {
                    "model": model_name,
                    "overlap_level": overlap_level,
                    "exact_match_mean": float(exact_values.mean()),
                    "exact_match_standard_deviation": (
                        _sample_standard_deviation(exact_values)
                    ),
                    "macro_f1_mean": float(f1_values.mean()),
                    "macro_f1_standard_deviation": (
                        _sample_standard_deviation(f1_values)
                    ),
                    "seed_count": len(group),
                }
            )

    write_csv_rows(destination_path, comparison_rows)
    return comparison_rows


def copy_attention_metrics_table(
    source_path: Path,
    destination_path: Path,
) -> Path:
    """Attention 정렬과 class selectivity 결과를 최종 표로 복사한다.

    표현:
        Attention 모델·seed별 validation 선택 IoU threshold, test average precision,
        IoU, cross-selectivity, 유효 sample 비율과 map permutation 성능을 기록한다.

    계산:
        Threshold는 validation에서만 선택되고 나머지 값은 고정 threshold로 test에서
        계산된다. Class map permutation drop은 원래 class 연결을 순환 교환한 차이다.

    목적:
        Accuracy 변화가 attention 정렬이나 class별 공간 선택과 연결되는지 확인하는
        mechanism 표다. Shared attention의 class selectivity가 비어 있는 것은 정상이다.

    입력:
        Evaluation이 생성한 `attention_metrics.csv`와 최종 표 경로

    처리:
        계산값을 변경하지 않고 provenance를 유지한 채 결과 디렉터리로 복사한다.

    출력:
        복사된 `attention_metrics.csv` 경로
    """
    return _copy_existing_table(source_path, destination_path)


def copy_bootstrap_intervals_table(
    source_path: Path,
    destination_path: Path,
) -> Path:
    """Paired bootstrap 추정값과 신뢰구간을 최종 표로 복사한다.

    표현:
        LeNet Low−High 저하, High-overlap 모델 차이, High−Low
        difference-in-differences의 point estimate와 95% 신뢰구간을 기록한다.

    계산:
        동일 pair ID를 유지한 5,000회 paired bootstrap을 사용해 Low/Middle/High의
        paired 설계를 보존한다.

    목적:
        관측된 성능 차이가 sampling uncertainty보다 큰지 판단하는 필수 통계 표다.
        모델 평균 차이가 작을 때 과도한 우열 해석을 막는다.

    입력:
        Evaluation이 생성한 `bootstrap_intervals.csv`와 최종 표 경로

    처리:
        계산값을 변경하지 않고 결과 디렉터리로 복사한다.

    출력:
        복사된 `bootstrap_intervals.csv` 경로
    """
    return _copy_existing_table(source_path, destination_path)


def copy_model_costs_table(
    source_path: Path,
    destination_path: Path,
) -> Path:
    """모델 parameter와 추론 연산량 비교를 최종 표로 복사한다.

    표현:
        세 모델의 trainable parameter 수와 한 장의 `76×76` 입력에 대한 추정 MAC을
        기록한다.

    계산:
        Parameter는 trainable tensor 원소 수를 합산하고 MAC은 Conv2d와 Linear
        forward hook으로 출력 위치별 multiply-accumulate를 누적한다.

    목적:
        Accuracy 차이가 추가 capacity와 계산 비용을 동반하는지 확인하는 보조 표다.
        특히 Class attention의 shared classifier 반복 비용을 함께 해석한다.

    입력:
        Evaluation이 생성한 `model_costs.csv`와 최종 표 경로

    처리:
        계산값을 변경하지 않고 결과 디렉터리로 복사한다.

    출력:
        복사된 `model_costs.csv` 경로
    """
    return _copy_existing_table(source_path, destination_path)


# -----------------------------------------------------------------------------
# Markdown 요약 생성
# -----------------------------------------------------------------------------


def create_summary(
    config: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    """실험 설정과 모든 주요 결과를 하나의 Markdown 보고서로 저장한다.

    입력:
        전체 config, model comparison row, 출력 Markdown 경로

    처리:
        성능, bootstrap, attention, 비용, 오류 분석과 figure 설명을 순서대로 구성한다.

    출력:
        반환값은 없으며 결과 전체의 독자용 진입점인 `summary.md`가 생성된다.
    """
    performance_lookup = {
        (row["model"], row["overlap_level"]): row
        for row in comparison_rows
    }
    metric_rows = read_csv_rows(METRIC_LOG_DIR / "model_metrics.csv")

    for model_name in MODEL_LABELS:
        performance_lookup[(model_name, "all")] = _aggregate_metric_rows(
            metric_rows,
            model_name,
            "all",
        )

    bootstrap_rows = read_csv_rows(TABLE_DIR / "bootstrap_intervals.csv")
    attention_rows = read_csv_rows(TABLE_DIR / "attention_metrics.csv")
    model_cost_rows = read_csv_rows(TABLE_DIR / "model_costs.csv")
    per_class_rows = read_csv_rows(METRIC_LOG_DIR / "per_class_metrics.csv")
    pair_accuracy_rows = read_csv_rows(
        METRIC_LOG_DIR / "class_pair_accuracy_high.csv"
    )

    lines = [
        "# MNIST Overlap Attention 결과 요약",
        "",
        "## 실험 설정",
        "",
        f"- 모델: {', '.join(config['model']['model_names'])}",
        f"- 데이터 seed: {config['project']['data_seed']}",
        f"- 학습 seed: {config['project']['training_seeds']}",
        f"- Train: {config['dataset']['train_samples']:,} images",
        f"- Validation: {config['dataset']['validation_pairs']:,} pairs, "
        f"{config['dataset']['validation_pairs'] * 3:,} images",
        f"- Test: {config['dataset']['test_pairs']:,} pairs, "
        f"{config['dataset']['test_pairs'] * 3:,} images",
        f"- 학습: Adam, learning rate {config['train']['learning_rate']}, "
        f"batch size {config['train']['batch_size']}, 최대 "
        f"{config['train']['maximum_epochs']} epochs",
        "- Checkpoint 선택: validation Top-2 exact-match 기반 early stopping",
        "- Test 주 지표: 두 정답 class를 모두 맞힌 Top-2 exact-match accuracy",
        "",
    ]
    lines.extend(
        _create_key_finding_lines(bootstrap_rows)
        + _create_performance_lines(performance_lookup)
        + _create_bootstrap_lines(bootstrap_rows)
        + _create_attention_lines(attention_rows)
        + _create_model_cost_lines(model_cost_rows)
        + _create_error_analysis_lines(per_class_rows, pair_accuracy_rows)
        + _create_figure_lines()
        + [
            "## 해석 범위",
            "",
            "결과는 중앙 정렬된 서로 다른 class의 MNIST 숫자 두 개를 `76×76` "
            "canvas에서 maximum 합성한 조건에 한정된다. Class attention은 Shared "
            "attention보다 High-overlap 정확도가 높았지만 LeNet 대비 High-overlap "
            "차이와 overlap 증가에 따른 상대 개선 폭은 신뢰구간이 0을 포함했다.",
            "",
            "따라서 이 결과는 class-conditional attention이 모든 조건에서 plain CNN을 "
            "일관되게 개선한다는 근거로 해석하지 않는다. MultiMNIST의 state of the "
            "art, Capsule Network보다 우수함, 실제 객체 가림 문제로의 직접 일반화도 "
            "주장하지 않는다.",
            "",
            "## 원본 결과 파일",
            "",
            "- [모델 비교표](tables/model_comparison.csv)",
            "- [Paired bootstrap 표](tables/bootstrap_intervals.csv)",
            "- [Attention 분석표](tables/attention_metrics.csv)",
            "- [모델 비용표](tables/model_costs.csv)",
            "- 세부 class 및 pair 결과는 `logs/metrics/`의 CSV에 저장된다.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _create_key_finding_lines(
    bootstrap_rows: list[dict[str, str]],
) -> list[str]:
    """Bootstrap 결과에서 핵심 결론 문장을 생성한다.

    입력:
        네 비교의 estimate와 confidence interval row

    처리:
        비교 이름으로 row를 찾고 차이와 신뢰구간을 percentage point로 변환한다.

    출력:
        `핵심 결과` Markdown 문장 목록
    """
    lookup = {
        row["comparison"]: row
        for row in bootstrap_rows
    }
    baseline_drop = lookup["lenet_low_minus_high"]
    class_vs_lenet = lookup["class_attention_minus_lenet_high"]
    class_vs_shared = lookup["class_attention_minus_shared_high"]
    difference_in_differences = lookup[
        "class_attention_vs_lenet_high_low_difference"
    ]
    return [
        "## 핵심 결과",
        "",
        f"- LeNet 정확도는 Low에서 High로 {_format_interval(baseline_drop)} "
        "감소했다.",
        f"- High overlap에서 Class attention은 LeNet보다 "
        f"{_format_interval(class_vs_lenet)} 높았지만 신뢰구간이 0을 포함했다.",
        f"- High overlap에서 Class attention은 Shared attention보다 "
        f"{_format_interval(class_vs_shared)} 높았고 신뢰구간이 0보다 컸다.",
        f"- Class attention의 LeNet 대비 개선 폭이 Low보다 High에서 더 커졌는지를 "
        f"본 difference-in-differences는 {_format_interval(difference_in_differences)}로 "
        "0을 포함했다.",
        "- 세 평균 정확도에서는 Class attention이 모든 overlap level에서 가장 "
        "높았지만, plain LeNet 대비 일관된 개선이라고 결론 내릴 근거는 충분하지 않았다.",
        "",
    ]


def _create_performance_lines(
    performance_lookup: dict[tuple[str, str], dict[str, Any]],
) -> list[str]:
    """Overall 및 overlap별 Exact-match와 Macro-F1 표를 생성한다.

    입력:
        Model과 overlap level을 key로 갖는 seed 집계 dictionary

    처리:
        세 seed의 평균과 표본 표준편차를 독자용 문자열로 변환한다.

    출력:
        두 Markdown 성능표의 line 목록
    """
    levels = ("all", *OVERLAP_ORDER)
    lines = [
        "## 분류 성능",
        "",
        "값은 seed 0, 1, 2의 평균 ± 표본 표준편차다.",
        "",
        "### Top-2 exact-match accuracy (%)",
        "",
        "| Model | Overall | Low | Middle | High |",
        "|---|---:|---:|---:|---:|",
    ]

    for model_name, model_label in MODEL_LABELS.items():
        values = [
            _format_mean_and_deviation(
                performance_lookup[(model_name, level)],
                "exact_match",
                scale=100.0,
                digits=2,
            )
            for level in levels
        ]
        lines.append(f"| {model_label} | {' | '.join(values)} |")

    lines.extend(
        [
            "",
            "### Macro-F1",
            "",
            "| Model | Overall | Low | Middle | High |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for model_name, model_label in MODEL_LABELS.items():
        values = [
            _format_mean_and_deviation(
                performance_lookup[(model_name, level)],
                "macro_f1",
                scale=1.0,
                digits=4,
            )
            for level in levels
        ]
        lines.append(f"| {model_label} | {' | '.join(values)} |")

    lines.extend(
        [
            "",
            "![Overlap별 test accuracy](figures/overlap_accuracy.png)",
            "",
            "이 그림은 overlap level별 exact-match 평균과 seed 표준편차를 확대된 "
            "y축으로 표시한다. 확대 축은 작은 모델 차이를 읽기 위한 것이며 절대 성능 "
            "차이는 위 표의 수치로 판단한다.",
            "",
        ]
    )
    return lines


def _create_bootstrap_lines(
    bootstrap_rows: list[dict[str, str]],
) -> list[str]:
    """Paired bootstrap의 모든 비교 결과를 Markdown 표로 생성한다.

    입력:
        비교 이름, estimate, 95% confidence interval row

    처리:
        동일 pair ID를 보존한 차이를 percentage point로 표시하고 0 포함 여부를 판정한다.

    출력:
        Bootstrap 설명과 네 비교 row
    """
    comparison_labels = {
        "lenet_low_minus_high": "LeNet: Low − High",
        "class_attention_minus_lenet_high": "High: Class − LeNet",
        "class_attention_minus_shared_high": "High: Class − Shared",
        "class_attention_vs_lenet_high_low_difference": (
            "(Class − LeNet): High − Low"
        ),
    }
    lines = [
        "## Paired bootstrap",
        "",
        "Test의 동일 pair ID를 유지해 5,000회 재표집한 95% 신뢰구간이다.",
        "",
        "| 비교 | 추정 차이 (%p) | 95% CI (%p) | 판단 |",
        "|---|---:|---:|---|",
    ]

    for row in bootstrap_rows:
        lower = float(row["confidence_lower"])
        upper = float(row["confidence_upper"])
        interval_excludes_zero = lower > 0.0 or upper < 0.0
        judgement = "0 미포함" if interval_excludes_zero else "0 포함"
        lines.append(
            f"| {comparison_labels[row['comparison']]} | "
            f"{100.0 * float(row['estimate']):.2f} | "
            f"[{100.0 * lower:.2f}, {100.0 * upper:.2f}] | {judgement} |"
        )

    lines.append("")
    return lines


def _create_attention_lines(
    attention_rows: list[dict[str, str]],
) -> list[str]:
    """Attention 모델의 seed 평균 정렬·선택성 결과를 표로 생성한다.

    입력:
        Attention model과 seed별 validation threshold 및 test metric row

    처리:
        유한한 값만 모델별 평균·표본 표준편차로 집계한다.

    출력:
        Attention metric 표와 실제 출력 그림 설명
    """
    lines = [
        "## Attention 분석",
        "",
        "IoU threshold는 각 seed의 validation set에서 선택한 뒤 test에 고정했다. "
        "아래 값은 세 seed 평균 ± 표본 표준편차다.",
        "",
        "| Model | Threshold | AUPRC | IoU | Cross-selectivity |",
        "|---|---:|---:|---:|---:|",
    ]

    grouped_rows = {
        model_name: [
            row
            for row in attention_rows
            if row["model"] == model_name
        ]
        for model_name in ("shared_attention", "class_attention")
    }

    for model_name, rows in grouped_rows.items():
        lines.append(
            f"| {MODEL_LABELS[model_name]} | "
            f"{_format_row_statistics(rows, 'selected_iou_threshold')} | "
            f"{_format_row_statistics(rows, 'test_average_precision')} | "
            f"{_format_row_statistics(rows, 'test_iou')} | "
            f"{_format_row_statistics(rows, 'test_cross_selectivity')} |"
        )

    class_rows = grouped_rows["class_attention"]
    valid_fraction = _mean_numeric_field(
        class_rows,
        "selectivity_valid_fraction",
    )
    permuted_accuracy = _mean_numeric_field(
        class_rows,
        "permuted_exact_match",
    )
    permutation_drop = _mean_numeric_field(
        class_rows,
        "permutation_accuracy_drop",
    )
    lines.extend(
        [
            "",
            "AUPRC와 IoU의 절대값은 두 attention 모델 모두 낮아, attention map을 "
            "숫자 획의 정밀한 segmentation으로 해석하기는 어렵다. Class attention의 "
            "평균은 Shared attention보다 약간 높았지만 이 차이에 대한 별도의 "
            "신뢰구간은 계산하지 않았다.",
            "",
            f"Class attention의 cross-selectivity 유효 sample 비율은 "
            f"{100.0 * valid_fraction:.2f}%였다. Class map channel을 순환 교환하면 "
            f"exact-match가 평균 {100.0 * permuted_accuracy:.2f}%로 낮아졌고, 정상 "
            f"출력 대비 {100.0 * permutation_drop:.2f}%p 감소했다. 이는 학습된 class "
            "map과 output class의 연결이 prediction에 실제로 사용됐음을 보여준다.",
            "",
            "![실제 attention 출력](figures/attention_examples.png)",
            "",
            "그림은 seed 0 checkpoint가 동일한 `3+8` pair의 Low/Middle/High 입력에 "
            "출력한 실제 attention map이다. 정성 예시이므로 모델 전체의 attention "
            "정렬은 위 정량 지표를 우선해 해석한다.",
            "",
        ]
    )
    return lines


def _create_model_cost_lines(
    model_cost_rows: list[dict[str, str]],
) -> list[str]:
    """세 모델의 parameter와 sample당 MAC 비교표를 생성한다.

    입력:
        모델별 trainable parameter 및 추정 MAC row

    처리:
        정수 값과 LeNet 대비 MAC 배율을 계산한다.

    출력:
        모델 비용 Markdown 표
    """
    baseline_operations = int(model_cost_rows[0]["multiply_accumulate_operations"])
    lines = [
        "## 모델 크기와 계산 비용",
        "",
        "MAC은 한 장의 `76×76` 입력을 forward할 때 Conv2d와 Linear 연산을 "
        "누적한 추정값이다.",
        "",
        "| Model | Trainable parameters | MAC / image | LeNet 대비 MAC |",
        "|---|---:|---:|---:|",
    ]

    for row in model_cost_rows:
        operations = int(row["multiply_accumulate_operations"])
        lines.append(
            f"| {MODEL_LABELS[row['model']]} | {int(row['parameters']):,} | "
            f"{operations:,} | {operations / baseline_operations:.3f}× |"
        )

    lines.extend(
        [
            "",
            "Class attention은 parameter 증가는 작지만 class별 shared classifier를 "
            "반복하므로 LeNet보다 약 2.25배의 MAC을 사용한다.",
            "",
        ]
    )
    return lines


def _create_error_analysis_lines(
    per_class_rows: list[dict[str, str]],
    pair_accuracy_rows: list[dict[str, str]],
) -> list[str]:
    """Class recall과 High-overlap pair 난이도의 대표 결과를 요약한다.

    입력:
        Model·seed·class별 precision/recall 및 model·pair별 accuracy row

    처리:
        Seed 평균 recall의 최저 class와 pair accuracy의 최저·최고 조합을 찾는다.

    출력:
        오류 분석 Markdown 표와 heatmap 설명
    """
    lines = [
        "## Class 및 숫자 조합별 오류 분석",
        "",
        "| Model | 가장 낮은 평균 recall | 가장 어려운 High pair | 가장 쉬운 High pair |",
        "|---|---:|---:|---:|",
    ]

    for model_name, model_label in MODEL_LABELS.items():
        model_class_rows = [
            row
            for row in per_class_rows
            if row["model"] == model_name
        ]
        class_recalls = []

        for class_index in range(10):
            class_values = [
                float(row["recall"])
                for row in model_class_rows
                if int(row["class"]) == class_index
            ]
            class_recalls.append((class_index, float(np.mean(class_values))))

        weakest_class, weakest_recall = min(
            class_recalls,
            key=lambda item: item[1],
        )
        model_pair_rows = [
            row
            for row in pair_accuracy_rows
            if row["model"] == model_name
        ]
        hardest_pair = min(model_pair_rows, key=lambda row: float(row["accuracy"]))
        easiest_pair = max(model_pair_rows, key=lambda row: float(row["accuracy"]))
        lines.append(
            f"| {model_label} | class {weakest_class}: "
            f"{100.0 * weakest_recall:.2f}% | "
            f"{hardest_pair['label_first']}+{hardest_pair['label_second']}: "
            f"{100.0 * float(hardest_pair['accuracy']):.2f}% | "
            f"{easiest_pair['label_first']}+{easiest_pair['label_second']}: "
            f"{100.0 * float(easiest_pair['accuracy']):.2f}% |"
        )

    lines.extend(
        [
            "",
            "![High-overlap class-pair accuracy](figures/pair_accuracy.png)",
            "",
            "Heatmap의 각 cell은 seed별 sample correctness를 먼저 평균한 뒤 해당 "
            "unordered 숫자 조합에서 평균한 High-overlap exact-match다. 대각선은 같은 "
            "class 두 개를 데이터에 포함하지 않았기 때문에 흰색이다. 이 그림은 모델 "
            "순위보다 공통 난이도 패턴을 보는 보조 분석으로 사용한다.",
            "",
        ]
    )
    return lines


def _create_figure_lines() -> list[str]:
    """Controlled overlap 입력 예시의 설명과 그림 link를 생성한다.

    입력:
        Report 단계에서 생성된 고정 figure 경로

    처리:
        실제 manifest 예시가 나타내는 통제 변수를 설명한다.

    출력:
        입력 예시 Markdown section
    """
    return [
        "## Controlled overlap 입력 예시",
        "",
        "![Overlap 입력 예시](figures/overlap_examples.png)",
        "",
        "각 행은 `1+0`, `4+7`, `3+8` 조합이고 열은 Low, Middle, High다. 한 행 "
        "안에서는 MNIST 원본 두 장, pair 중심과 이동 방향을 유지하고 displacement만 "
        "변경했다. 따라서 level 간 성능 비교는 동일 pair의 overlap 강도 변화에 "
        "대응한다.",
        "",
    ]


def _aggregate_metric_rows(
    metric_rows: list[dict[str, str]],
    model_name: str,
    overlap_level: str,
) -> dict[str, Any]:
    """한 model·overlap의 seed별 분류 지표를 평균과 표준편차로 집계한다.

    입력:
        전체 metric row, model 이름, overlap level

    처리:
        조건에 맞는 exact-match와 Macro-F1을 NumPy array로 변환해 집계한다.

    출력:
        Comparison row와 같은 field를 갖는 dictionary
    """
    selected_rows = [
        row
        for row in metric_rows
        if row["model"] == model_name
        and row["overlap_level"] == overlap_level
    ]

    if not selected_rows:
        raise ValueError(
            f"Model {model_name}, overlap {overlap_level} metric이 없습니다."
        )

    exact_values = np.asarray([
        float(row["exact_match"])
        for row in selected_rows
    ])
    f1_values = np.asarray([
        float(row["macro_f1"])
        for row in selected_rows
    ])
    return {
        "exact_match_mean": float(exact_values.mean()),
        "exact_match_standard_deviation": _sample_standard_deviation(exact_values),
        "macro_f1_mean": float(f1_values.mean()),
        "macro_f1_standard_deviation": _sample_standard_deviation(f1_values),
    }


def _format_mean_and_deviation(
    row: dict[str, Any],
    metric_name: str,
    scale: float,
    digits: int,
) -> str:
    """집계 row의 평균과 표준편차를 동일 단위 문자열로 변환한다.

    입력:
        집계 row, metric field 접두어, 배율, 소수점 자리 수

    처리:
        Mean과 standard deviation에 같은 scale과 숫자 형식을 적용한다.

    출력:
        `평균 ± 표준편차` 문자열
    """
    mean = scale * float(row[f"{metric_name}_mean"])
    deviation = scale * float(row[f"{metric_name}_standard_deviation"])
    return f"{mean:.{digits}f} ± {deviation:.{digits}f}"


def _format_interval(row: dict[str, str]) -> str:
    """Bootstrap row를 percentage-point 추정값과 신뢰구간으로 표시한다.

    입력:
        Estimate와 confidence interval을 포함한 한 row

    처리:
        세 값을 100배해 소수점 둘째 자리로 반올림한다.

    출력:
        `추정값%p (95% CI [하한, 상한])` 문자열
    """
    estimate = 100.0 * float(row["estimate"])
    lower = 100.0 * float(row["confidence_lower"])
    upper = 100.0 * float(row["confidence_upper"])
    return f"{estimate:.2f}%p (95% CI [{lower:.2f}, {upper:.2f}])"


def _format_row_statistics(
    rows: list[dict[str, str]],
    field_name: str,
) -> str:
    """CSV field의 유한 값 평균과 표준편차를 문자열로 변환한다.

    입력:
        같은 모델의 seed별 row와 숫자 field 이름

    처리:
        빈 문자열, NaN, 무한대를 제외하고 평균과 표본 표준편차를 계산한다.

    출력:
        값이 없으면 em dash, 있으면 `평균 ± 표준편차` 문자열
    """
    values = _numeric_field_values(rows, field_name)

    if not values:
        return "—"

    value_array = np.asarray(values)
    deviation = _sample_standard_deviation(value_array)
    return f"{value_array.mean():.4f} ± {deviation:.4f}"


def _mean_numeric_field(
    rows: list[dict[str, str]],
    field_name: str,
) -> float:
    """CSV field의 유한 값만 선택해 평균한다.

    입력:
        CSV row와 숫자 field 이름

    처리:
        공통 숫자 변환 helper 결과에 NumPy mean을 적용한다.

    출력:
        유효한 값의 float 평균
    """
    values = _numeric_field_values(rows, field_name)

    if not values:
        raise ValueError(f"평균을 계산할 {field_name} 값이 없습니다.")

    return float(np.mean(values))


def _numeric_field_values(
    rows: list[dict[str, str]],
    field_name: str,
) -> list[float]:
    """CSV row에서 비어 있지 않은 유한 숫자 field를 수집한다.

    입력:
        CSV row 목록과 field 이름

    처리:
        빈 문자열을 제외하고 float 변환 후 `isfinite`를 검사한다.

    출력:
        유한 float 값 목록
    """
    values = []

    for row in rows:
        raw_value = row[field_name]

        if raw_value == "":
            continue

        value = float(raw_value)

        if np.isfinite(value):
            values.append(value)

    return values


# -----------------------------------------------------------------------------
# CSV 공통 처리
# -----------------------------------------------------------------------------


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """CSV 전체를 header 기반 dictionary row로 읽는다.

    입력:
        입력 CSV 경로

    처리:
        File 존재를 확인하고 `csv.DictReader` 결과를 list로 변환한다.

    출력:
        문자열 dictionary row 목록
    """
    if not path.exists():
        raise FileNotFoundError(f"필요한 metric file이 없습니다: {path}")

    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Dictionary row 목록을 header 포함 CSV로 저장한다.

    입력:
        출력 경로와 동일 key를 가진 row 목록

    처리:
        첫 row key를 header로 사용하고 UTF-8 CSV를 기록한다.

    출력:
        반환값은 없으며 row가 있을 때 CSV file이 생성된다.
    """
    if not rows:
        raise ValueError(f"저장할 table row가 없습니다: {path}")

    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sample_standard_deviation(values: np.ndarray) -> float:
    """반복 실험 값의 표본 표준편차를 계산한다.

    입력:
        하나 이상의 scalar metric array

    처리:
        두 값 이상이면 `ddof=1`, 하나면 변동을 추정할 수 없어 0을 반환한다.

    출력:
        Python float 표준편차
    """
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _copy_existing_table(source_path: Path, destination_path: Path) -> Path:
    """평가 단계의 CSV를 값 변경 없이 최종 결과 위치로 복사한다.

    입력:
        기존 metric CSV와 destination 경로

    처리:
        원본 존재를 확인하고 metadata 변환 없이 byte 단위로 복사한다.

    출력:
        복사된 destination 경로
    """
    if not source_path.exists():
        raise FileNotFoundError(f"필요한 metric file이 없습니다: {source_path}")

    shutil.copyfile(source_path, destination_path)
    return destination_path
