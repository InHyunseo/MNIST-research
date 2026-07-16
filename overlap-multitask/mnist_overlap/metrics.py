"""
분류 지표와 bootstrap 통계를 계산하는 순수 수치 모듈이다. Top-2 exact-match·macro-F1·
class별 precision/recall·숫자 조합별 정확도와, 같은 pair의 overlap level 간 성능 차이에 대한
hierarchical·paired bootstrap 신뢰구간을 제공한다. 설정에 의존하지 않고 numpy/torch만
사용하며, 학습·평가·시각화 모듈이 이 함수들을 조합한다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


# -----------------------------------------------------------------------------
# 분류 지표
# -----------------------------------------------------------------------------


def top_two_predictions(logits: torch.Tensor) -> torch.Tensor:
    """
    입력: logits — `[sample, class]` 형태의 logit tensor
    출력: 상위 2 class만 True인 같은 크기의 Boolean tensor

    각 sample에서 가장 큰 logit 두 개를 예측으로 표시한다.
    """
    top_indices = torch.topk(logits, k=2, dim=1).indices

    predictions = torch.zeros_like(logits, dtype=torch.bool)
    predictions.scatter_(1, top_indices, True)

    return predictions


def exact_match_per_sample(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    입력: logits — `[sample, class]` logit tensor
          labels — `[sample, class]` multi-hot 정답 tensor
    출력: sample별 정답 여부 Boolean tensor

    top-2 예측 집합이 정답 multi-hot과 정확히 같은지 sample별로 판정한다.
    """
    predictions = top_two_predictions(logits)

    return torch.all(predictions == labels.bool(), dim=1)


def classification_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """
    입력: logits — `[sample, class]` logit array
          labels — `[sample, class]` multi-hot 정답 array
    출력: exact_match·macro_f1·per_class_precision·per_class_recall·
          correct_per_sample·predictions를 담은 dictionary

    한 prediction 묶음의 분류 지표를 한 번에 계산한다.
    """
    top_indices = np.argpartition(logits, kth=-2, axis=1)[:, -2:]

    predictions = np.zeros_like(labels, dtype=bool)
    rows = np.arange(len(labels))[:, None]
    predictions[rows, top_indices] = True
    boolean_labels = labels.astype(bool)

    true_positive = np.logical_and(predictions, boolean_labels).sum(axis=0)
    false_positive = np.logical_and(predictions, ~boolean_labels).sum(axis=0)
    false_negative = np.logical_and(~predictions, boolean_labels).sum(axis=0)

    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1_scores = _safe_divide(2.0 * precision * recall, precision + recall)
    exact = np.all(predictions == boolean_labels, axis=1)

    return {
        "exact_match": float(exact.mean()),
        "macro_f1": float(f1_scores.mean()),
        "per_class_precision": precision,
        "per_class_recall": recall,
        "correct_per_sample": exact,
        "predictions": predictions,
    }


def class_pair_accuracy(
    correct_per_sample: np.ndarray,
    label_first: np.ndarray,
    label_second: np.ndarray,
    class_count: int = 10,
) -> np.ndarray:
    """
    입력: correct_per_sample — sample별 정답 여부 배열
          label_first, label_second — sample별 두 정답 class
          class_count — 전체 class 수
    출력: `[class, class]` 대칭 정답률 행렬 (대각선은 NaN)

    unordered 정답 class pair별 평균 정답률을 대칭 행렬로 모은다.
    """
    accuracy_matrix = np.full((class_count, class_count), np.nan, dtype=np.float64)

    for first_class in range(class_count):
        for second_class in range(first_class + 1, class_count):
            pair_mask = (label_first == first_class) & (label_second == second_class)
            if not pair_mask.any():
                continue

            pair_accuracy = float(correct_per_sample[pair_mask].mean())
            accuracy_matrix[first_class, second_class] = pair_accuracy
            accuracy_matrix[second_class, first_class] = pair_accuracy

    return accuracy_matrix


def sample_deviation(values: np.ndarray) -> float:
    """
    입력: values — 표준편차를 구할 값 배열
    출력: 표본 표준편차 (값이 1개 이하이면 0.0)

    표본 표준편차(`ddof=1`)를 계산한다.
    """
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """
    입력: numerator, denominator — 같은 형태의 numpy 배열
    출력: 분모가 0인 위치를 0으로 유지한 나눗셈 결과

    element-wise 나눗셈에서 0 분모를 안전하게 처리한다.
    """
    result = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator != 0)

    return result


# -----------------------------------------------------------------------------
# Bootstrap 통계
# -----------------------------------------------------------------------------


def hierarchical_bootstrap_interval(
    values_by_seed_and_pair: np.ndarray,
    iterations: int,
    confidence_level: float,
    random_seed: int,
) -> tuple[float, float, float]:
    """
    입력: values_by_seed_and_pair — `[seed, pair]` correctness 차이 행렬
          iterations — bootstrap 반복 수
          confidence_level — 신뢰수준 (예: 0.95)
          random_seed — 난수 seed
    출력: (평균 추정값, 하한, 상한)

    training seed와 test pair를 함께 복원추출해 평균의 percentile 구간을 구한다.
    """
    values = np.asarray(values_by_seed_and_pair, dtype=np.float64)

    matrix_is_valid = values.ndim == 2 and all(size > 0 for size in values.shape)
    if not matrix_is_valid:
        raise ValueError("Hierarchical bootstrap 입력은 비어 있지 않은 2차원 행렬이어야 합니다.")
    if not np.isfinite(values).all():
        raise ValueError("Hierarchical bootstrap 입력에는 유한한 값만 사용할 수 있습니다.")

    seed_count, pair_count = values.shape
    random_generator = np.random.default_rng(random_seed)
    bootstrap_means = np.empty(iterations, dtype=np.float64)

    for iteration in range(iterations):
        sampled_seed_indices = random_generator.integers(0, seed_count, size=seed_count)
        sampled_pair_indices = random_generator.integers(0, pair_count, size=pair_count)

        # seed 재표집을 먼저 평균하면 동일한 2단계 표본을 더 작은 array로 계산한다.
        seed_resampled_pair_means = values[sampled_seed_indices].mean(axis=0)
        bootstrap_means[iteration] = seed_resampled_pair_means[sampled_pair_indices].mean()

    tail_probability = (1.0 - confidence_level) / 2.0
    lower_bound, upper_bound = np.quantile(
        bootstrap_means,
        [tail_probability, 1.0 - tail_probability],
    )

    return float(values.mean()), float(lower_bound), float(upper_bound)


def paired_bootstrap_interval(
    differences: np.ndarray,
    pair_ids: np.ndarray,
    iterations: int,
    confidence_level: float,
    random_seed: int,
) -> tuple[float, float, float]:
    """
    입력: differences — sample별 correctness 차이 배열
          pair_ids — 각 sample이 속한 pair id
          iterations — bootstrap 반복 수
          confidence_level — 신뢰수준
          random_seed — 난수 seed
    출력: (평균 차이 추정값, 하한, 상한)

    pair별 평균을 만든 뒤 pair 단위 bootstrap으로 평균 차이 구간을 구한다.
    """
    unique_pair_ids = np.unique(pair_ids)
    pair_means = np.asarray([
        differences[pair_ids == pair_id].mean()
        for pair_id in unique_pair_ids
    ])

    random_generator = np.random.default_rng(random_seed)
    bootstrap_means = np.empty(iterations, dtype=np.float64)
    chunk_size = min(100, iterations)

    for start in range(0, iterations, chunk_size):
        stop = min(start + chunk_size, iterations)

        sampled_indices = random_generator.integers(
            0,
            len(pair_means),
            size=(stop - start, len(pair_means)),
        )
        bootstrap_means[start:stop] = pair_means[sampled_indices].mean(axis=1)

    tail_probability = (1.0 - confidence_level) / 2.0
    lower_bound, upper_bound = np.quantile(
        bootstrap_means,
        [tail_probability, 1.0 - tail_probability],
    )

    return float(pair_means.mean()), float(lower_bound), float(upper_bound)


def paired_level_difference(
    correctness: np.ndarray,
    pair_ids: np.ndarray,
    overlap_levels: np.ndarray,
    first_level: str,
    second_level: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    입력: correctness — sample별 정답 여부 배열
          pair_ids — 각 sample의 pair id
          overlap_levels — 각 sample의 overlap level
          first_level, second_level — 비교할 두 overlap level 이름
    출력: (pair별 correctness 차이 배열, 공통 pair id 배열)

    같은 pair의 두 overlap level correctness 차이(first − second)를 계산한다.
    """
    first_mask = overlap_levels == first_level
    second_mask = overlap_levels == second_level

    common_pair_ids, first_indices, second_indices = np.intersect1d(
        pair_ids[first_mask],
        pair_ids[second_mask],
        assume_unique=True,
        return_indices=True,
    )

    first_correctness = correctness[first_mask][first_indices]
    second_correctness = correctness[second_mask][second_indices]
    differences = first_correctness - second_correctness

    return differences.astype(np.float64), common_pair_ids.astype(np.int64)
