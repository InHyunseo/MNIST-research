"""평가 결과와 고정 test sample을 발표용 PNG로 변환한다.

입력:
    집계 metric row, 전체 config, test manifest, 학습 checkpoint

출력:
    `results/figures/` 아래의 accuracy, class-pair, overlap, attention PNG

연결:
    `reporting.generator`가 이 모듈의 공개 함수를 호출하며 data, models, training을
    사용해 정량 결과와 실제 model 출력을 시각화한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import PercentFormatter

from ..configuration import CHECKPOINT_DIR
from ..data import ControlledOverlapMnistDataset
from ..models import create_model
from ..training import load_checkpoint

MODEL_LABELS = {
    "lenet": "LeNet",
    "shared_attention": "Shared attention",
    "class_attention": "Class attention",
}
OVERLAP_ORDER = ("low", "middle", "high")
OVERLAP_EXAMPLE_CLASS_PAIRS = ((1, 0), (4, 7), (3, 8))
ATTENTION_EXAMPLE_CLASS_PAIR = (3, 8)


# -----------------------------------------------------------------------------
# 정량 결과 그림
# -----------------------------------------------------------------------------


def create_overlap_accuracy_figure(
    rows: list[dict[str, Any]],
    path: Path,
    dpi: int,
) -> None:
    """세 모델의 overlap level별 test accuracy를 선 그래프로 저장한다.

    표현:
        X축은 Low, Middle, High overlap이고 점은 세 seed의 Top-2 exact-match 평균이다.
        Error bar는 seed 0, 1, 2에서 얻은 표본 표준편차이며 숫자는 평균 백분율이다.

    계산:
        `generator`가 seed별 정확도를 먼저 평균·표준편차로 집계한다. 이 함수는 모델
        점을 가로로 분리하고 평균 ± 표준편차가 보이는 범위로 y축을 확대한다.

    목적:
        겹침이 강해질 때 정확도가 얼마나 감소하는지와 같은 조건에서 세 모델의 평균
        차이가 어느 정도인지를 한 그림에서 비교한다.

    입력:
        모델·overlap별 집계 metric row, 출력 PNG 경로, DPI

    처리:
        평균, error bar, 백분율 label, 확대 y축과 범례를 배치한다.

    출력:
        반환값은 없으며 `overlap_accuracy.png`가 생성된다.
    """
    # 논문 본문에 삽입했을 때도 범례와 숫자가 읽히도록 가로형 비율을 사용한다.
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    x_positions = np.arange(len(OVERLAP_ORDER))

    # 같은 overlap level의 세 점이 겹치지 않게 x좌표만 시각적으로 분리한다.
    # Offset은 accuracy 값이나 통계 계산에는 영향을 주지 않는다.
    model_offsets = np.linspace(-0.07, 0.07, len(MODEL_LABELS))
    all_means = [float(row["exact_match_mean"]) for row in rows]
    all_deviations = [
        float(row["exact_match_standard_deviation"])
        for row in rows
    ]

    for model_index, (model_name, model_label) in enumerate(MODEL_LABELS.items()):
        model_rows = [row for row in rows if row["model"] == model_name]
        means = [float(row["exact_match_mean"]) for row in model_rows]
        deviations = [
            float(row["exact_match_standard_deviation"])
            for row in model_rows
        ]
        displayed_positions = x_positions + model_offsets[model_index]
        plotted_line = axis.errorbar(
            displayed_positions,
            means,
            yerr=deviations,
            marker="o",
            linewidth=1.6,
            capsize=3,
            label=model_label,
        )
        line_color = plotted_line.lines[0].get_color()

        for x_position, mean, deviation in zip(
            displayed_positions,
            means,
            deviations,
        ):
            # Error bar 위에는 세 seed 평균을 백분율 소수점 첫째 자리까지 표시한다.
            axis.annotate(
                f"{mean * 100:.1f}",
                xy=(x_position, mean + deviation),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=line_color,
            )

    # 평균과 error bar 전체가 잘리지 않는 데이터 기반 확대 범위를 사용한다.
    axis_lower, axis_upper = _expanded_accuracy_limits(
        all_means,
        all_deviations,
    )
    axis.set_xticks(x_positions, [level.title() for level in OVERLAP_ORDER])
    axis.set_xlabel("Bounding-box overlap level")
    axis.set_ylim(axis_lower, axis_upper)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.set_title("Test Accuracy")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def create_pair_accuracy_figure(
    rows: list[dict[str, str]],
    path: Path,
    dpi: int,
) -> None:
    """High-overlap class-pair accuracy를 모델별 heatmap으로 저장한다.

    표현:
        각 cell은 해당 unordered 숫자 조합의 High-overlap exact-match accuracy다.
        세 패널은 LeNet, Shared attention, Class attention이며 동일 color scale을 쓴다.

    계산:
        Evaluation 단계에서 seed 0, 1, 2의 sample별 정답 여부를 평균한 뒤 각 class
        pair에서 다시 평균한다. 45개 값을 대칭 10×10 행렬의 양쪽에 기록한다.

    목적:
        모델 우열의 주 근거가 아니라 세 모델이 공통으로 어려워하거나 쉬워하는 숫자
        조합을 확인하는 오류 분석용 보조 그림이다.

    입력:
        세 seed에서 집계한 High-overlap class-pair row, 출력 PNG 경로, DPI

    처리:
        대칭 행렬, 공통 color scale, missing diagonal과 하나의 colorbar를 구성한다.

    출력:
        반환값은 없으며 `pair_accuracy.png`가 생성된다.
    """
    accuracy_values = [float(row["accuracy"]) for row in rows]
    color_step = 0.05

    # 세 모델의 실제 최솟값·최댓값을 바깥쪽 0.05 단위로 반올림한다.
    color_minimum = max(
        0.0,
        float(color_step * np.floor(min(accuracy_values) / color_step)),
    )
    color_maximum = min(
        1.0,
        float(color_step * np.ceil(max(accuracy_values) / color_step)),
    )

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    colormap = plt.get_cmap("viridis").copy()
    colormap.set_bad(color="white")
    image = None

    for axis, (model_name, model_label) in zip(axes, MODEL_LABELS.items()):
        # 동일-class pair는 multi-hot Top-2 label로 두 instance를 표현할 수 없어 제외한다.
        # 대각 원소는 NaN으로 유지하며 missing value 색인 흰색으로 표시한다.
        matrix = np.full((10, 10), np.nan)

        for row in rows:
            if row["model"] != model_name:
                continue

            first_class = int(row["label_first"])
            second_class = int(row["label_second"])
            value = float(row["accuracy"])

            # Pair 순서에 의미가 없으므로 3+7과 7+3 위치에 같은 값을 기록한다.
            matrix[first_class, second_class] = value
            matrix[second_class, first_class] = value

        image = axis.imshow(
            matrix,
            vmin=color_minimum,
            vmax=color_maximum,
            cmap=colormap,
        )
        axis.set_title(model_label)
        axis.set_xticks(range(10))
        axis.set_yticks(range(10))

    figure.supxlabel("Digit class")
    figure.supylabel("Digit class")

    if image is not None:
        # High-overlap 조건과 metric 정의는 논문 caption에서 설명하므로 label은 생략한다.
        figure.colorbar(image, ax=axes, shrink=0.85)

    figure.savefig(path, dpi=dpi)
    plt.close(figure)


# -----------------------------------------------------------------------------
# 실제 입력과 attention 예시 그림
# -----------------------------------------------------------------------------


def create_attention_examples_figure(
    config: dict[str, Any],
    path: Path,
) -> None:
    """동일한 `3+8` test pair의 overlap별 실제 attention 출력을 저장한다.

    표현:
        행은 Low, Middle, High이고 열은 입력, Shared map, Class 3 map, Class 8 map,
        `Class 3 − Class 8` 차이 map이다. 원본 map은 검정 0, 흰색 1로 표시한다.

    계산:
        Seed 0 best checkpoint에 같은 원본 pair의 세 overlap 입력을 전달한다. Sigmoid
        이후 `[0, 1]` attention tensor를 사용하고 두 class map 차이는 같은 pixel끼리
        뺀다. 차이 map은 모든 level에서 동일한 0 중심 대칭 color scale을 사용한다.

    목적:
        Raw attention 모양뿐 아니라 두 class map이 실제로 다른 공간을 선택하는지,
        그리고 그 차이가 overlap 증가에 따라 어떻게 변하는지를 정성적으로 확인한다.
        차이가 미세하면 정량 결론은 attention metric 표를 우선한다.

    입력:
        전체 config, seed 0 checkpoint, `(3, 8)` paired test sample, 출력 PNG 경로

    처리:
        두 attention 모델을 복원하고 실제 forward 출력과 class 차이 map을 배치한다.

    출력:
        반환값은 없으며 `attention_examples.png`가 생성된다.
    """
    seed = int(config["project"]["training_seeds"][0])
    shared_model = create_model("shared_attention", config)
    class_model = create_model("class_attention", config)
    load_checkpoint(
        shared_model,
        CHECKPOINT_DIR / f"shared_attention_seed_{seed}.pt",
        config=config,
    )
    load_checkpoint(
        class_model,
        CHECKPOINT_DIR / f"class_attention_seed_{seed}.pt",
        config=config,
    )
    shared_model.eval()
    class_model.eval()

    dataset = ControlledOverlapMnistDataset("test", config, include_masks=True)
    selected_indices = _select_class_pair_examples(
        dataset,
        ATTENTION_EXAMPLE_CLASS_PAIR,
    )
    first_class, second_class = ATTENTION_EXAMPLE_CLASS_PAIR
    displayed_rows = []

    with torch.no_grad():
        for overlap_level, dataset_index in zip(OVERLAP_ORDER, selected_indices):
            sample = dataset[dataset_index]
            image = sample["image"].unsqueeze(0)
            _, shared_maps = shared_model(image, return_attention=True)
            _, class_maps = class_model(image, return_attention=True)
            first_attention = class_maps[0, first_class].numpy()
            second_attention = class_maps[0, second_class].numpy()
            displayed_rows.append(
                {
                    "overlap_level": overlap_level,
                    "input": sample["image"][0].numpy(),
                    "shared": shared_maps[0, 0].numpy(),
                    "first": first_attention,
                    "second": second_attention,
                    "difference": first_attention - second_attention,
                }
            )

    maximum_absolute_difference = max(
        float(np.abs(row["difference"]).max())
        for row in displayed_rows
    )
    # 완전히 같은 map이어도 Matplotlib color 범위가 0이 되지 않도록 최소 폭을 둔다.
    difference_limit = max(maximum_absolute_difference, 0.01)

    figure, axes = plt.subplots(
        len(OVERLAP_ORDER),
        5,
        figsize=(11.2, 6.8),
        constrained_layout=True,
        squeeze=False,
    )
    column_titles = (
        "Input",
        "Shared attention",
        f"Class {first_class} attention",
        f"Class {second_class} attention",
        f"Class {first_class} − Class {second_class}",
    )
    difference_image = None

    for row_index, displayed_row in enumerate(displayed_rows):
        raw_images = (
            displayed_row["input"],
            displayed_row["shared"],
            displayed_row["first"],
            displayed_row["second"],
        )

        for column_index, displayed_image in enumerate(raw_images):
            axis = axes[row_index, column_index]
            axis.imshow(displayed_image, cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_xticks([])
            axis.set_yticks([])

        difference_axis = axes[row_index, 4]
        difference_image = difference_axis.imshow(
            displayed_row["difference"],
            cmap="coolwarm",
            vmin=-difference_limit,
            vmax=difference_limit,
        )
        # 빨강은 Class 3 attention이 더 큰 위치, 파랑은 Class 8이 더 큰 위치다.
        # 세 overlap level에 동일한 차이 범위를 적용해 행 사이 색을 직접 비교한다.
        difference_axis.set_xticks([])
        difference_axis.set_yticks([])

        for column_index, axis in enumerate(axes[row_index]):
            if row_index == 0:
                axis.set_title(column_titles[column_index])

            if column_index == 0:
                axis.set_ylabel(displayed_row["overlap_level"].title(), fontsize=10)

    if difference_image is not None:
        figure.colorbar(difference_image, ax=axes[:, 4], shrink=0.75)

    figure.savefig(path, dpi=int(config["report"]["figure_dpi"]))
    plt.close(figure)


def create_overlap_examples_figure(
    config: dict[str, Any],
    path: Path,
) -> None:
    """세 숫자 조합의 Low/Middle/High 합성 입력을 3×3로 저장한다.

    표현:
        행은 `(1, 0)`, `(4, 7)`, `(3, 8)`이고 열은 Low, Middle, High이다.

    계산:
        각 행에서 동일한 MNIST 원본 두 장, pair center와 이동 방향을 유지하고 manifest에
        저장된 displacement만 바뀐 실제 max-composition test image를 재구성한다.

    목적:
        성능 결과를 해석하기 전에 이 프로젝트가 overlap 난이도를 어떻게 통제했는지
        독자가 이미지 수준에서 확인하게 한다.

    입력:
        전체 config, 세 고정 class 조합의 paired test manifest, 출력 PNG 경로

    처리:
        각 조합에서 첫 번째 완전한 paired sample을 선택해 overlap 순서로 배치한다.

    출력:
        반환값은 없으며 `overlap_examples.png`가 생성된다.
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

        for column_index, (overlap_level, dataset_index) in enumerate(
            zip(OVERLAP_ORDER, example_indices)
        ):
            sample = dataset[dataset_index]
            axis = axes[row_index, column_index]
            axis.imshow(
                sample["image"][0].numpy(),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            axis.set_xticks([])
            axis.set_yticks([])

            if row_index == 0:
                axis.set_title(overlap_level.title())

            if column_index == 0:
                axis.set_ylabel(f"{class_pair[0]} + {class_pair[1]}", fontsize=11)

    figure.savefig(path, dpi=int(config["report"]["figure_dpi"]))
    plt.close(figure)


# -----------------------------------------------------------------------------
# Figure 공통 helper
# -----------------------------------------------------------------------------


def _expanded_accuracy_limits(
    means: list[float],
    deviations: list[float],
) -> tuple[float, float]:
    """Accuracy와 error bar를 포함하는 확대 축 범위를 계산한다.

    입력:
        모델·overlap별 평균 accuracy와 seed 표준편차 목록

    처리:
        Error bar 바깥에 0.01 여백을 추가하고 0.02 간격으로 경계를 정렬한다.

    출력:
        `[0, 1]` 범위의 y축 하한과 상한
    """
    if not means or len(means) != len(deviations):
        raise ValueError("Accuracy 평균과 표준편차 목록의 길이가 올바르지 않습니다.")

    lower_error_limits = [
        mean - deviation
        for mean, deviation in zip(means, deviations)
    ]
    upper_error_limits = [
        mean + deviation
        for mean, deviation in zip(means, deviations)
    ]
    tick_step = 0.02
    padding = 0.01
    lower_limit = tick_step * np.floor(
        (min(lower_error_limits) - padding) / tick_step
    )
    upper_limit = tick_step * np.ceil(
        (max(upper_error_limits) + padding) / tick_step
    )
    return max(0.0, float(lower_limit)), min(1.0, float(upper_limit))


def _select_class_pair_examples(
    dataset: ControlledOverlapMnistDataset,
    class_pair: tuple[int, int],
) -> list[int]:
    """지정한 unordered class 조합의 첫 paired test sample index를 반환한다.

    입력:
        Paired test Dataset과 순서에 의미가 없는 두 class 번호

    처리:
        두 class 순서를 모두 허용하고 첫 pair ID의 Low, Middle, High index를 찾는다.

    출력:
        Low, Middle, High 순서의 dataset index 세 개
    """
    label_first = dataset.manifest["label_first"]
    label_second = dataset.manifest["label_second"]
    pair_ids = dataset.manifest["pair_id"]
    overlap_levels = dataset.manifest["overlap_level"]
    requested_first, requested_second = class_pair
    matching_class_order = (
        (label_first == requested_first) & (label_second == requested_second)
    )
    matching_reverse_order = (
        (label_first == requested_second) & (label_second == requested_first)
    )
    matching_pair_ids = pair_ids[matching_class_order | matching_reverse_order]

    for pair_id in dict.fromkeys(int(value) for value in matching_pair_ids):
        selected_indices = []

        for overlap_level in OVERLAP_ORDER:
            matching_indices = np.flatnonzero(
                (pair_ids == pair_id) & (overlap_levels == overlap_level)
            )

            if len(matching_indices) != 1:
                break

            selected_indices.append(int(matching_indices[0]))

        if len(selected_indices) == len(OVERLAP_ORDER):
            return selected_indices

    raise ValueError(
        f"Class pair {class_pair}의 완전한 Low/Middle/High sample을 찾지 못했습니다."
    )
