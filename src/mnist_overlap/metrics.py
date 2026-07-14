"""Top-2 multi-label 분류 지표와 공용 통계 helper.

Training, evaluation, reporting이 공유하는 최하위 leaf 모듈이다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def top_two_predictions(logits: torch.Tensor) -> torch.Tensor:
    """각 sample의 가장 큰 logit 두 개를 True로 표시한 Boolean prediction을 반환한다."""
    top_indices = torch.topk(logits, k=2, dim=1).indices
    predictions = torch.zeros_like(logits, dtype=torch.bool)
    predictions.scatter_(1, top_indices, True)
    return predictions


def exact_match_per_sample(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sample별로 top-2 예측 class 집합이 정답 multi-hot과 정확히 같은지 반환한다."""
    predictions = top_two_predictions(logits)
    return torch.all(predictions == labels.bool(), dim=1)


def classification_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """`[sample, class]` logit과 multi-hot label에서 exact-match, macro-F1,
    class별 precision/recall, sample별 correctness를 한 번에 계산한다."""
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
    """Unordered 정답 class pair별 평균 exact-match를 대각선이 NaN인 대칭 행렬로 반환한다."""
    accuracy_matrix = np.full((class_count, class_count), np.nan, dtype=np.float64)
    for first_class in range(class_count):
        for second_class in range(first_class + 1, class_count):
            pair_mask = (label_first == first_class) & (label_second == second_class)
            if pair_mask.any():
                pair_accuracy = float(correct_per_sample[pair_mask].mean())
                accuracy_matrix[first_class, second_class] = pair_accuracy
                accuracy_matrix[second_class, first_class] = pair_accuracy
    return accuracy_matrix


def finite_mean(values: np.ndarray) -> float:
    """NaN과 무한대를 제외한 평균을 계산한다. 유효 값이 없으면 NaN을 반환한다."""
    finite_values = values[np.isfinite(values)]
    return float(finite_values.mean()) if len(finite_values) else float("nan")


def sample_deviation(values: np.ndarray) -> float:
    """표본 표준편차(`ddof=1`)를 계산한다. 값이 하나 이하이면 0을 반환한다."""
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """분모가 0인 위치를 0으로 유지하는 element-wise 나눗셈."""
    result = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result
