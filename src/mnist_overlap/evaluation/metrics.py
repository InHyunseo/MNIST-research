"""두 정답 class를 위한 Top-2 multi-label 성능 지표를 계산한다.

입력:
    `[sample, class]` logit과 multi-hot label

출력:
    Exact-match, Macro-F1, class별 precision/recall, class-pair accuracy

연결:
    Training engine, evaluation runner, evaluation 실행 함수가 공통으로 사용한다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def top_two_predictions(logits: torch.Tensor) -> torch.Tensor:
    """각 sample의 가장 큰 logit 두 개를 Boolean prediction으로 변환한다.

    입력:
        `[batch, class]` logit tensor

    처리:
        Class 축에서 top-2 index를 찾고 같은 shape의 Boolean tensor에 표시한다.

    출력:
        Sample마다 True가 두 개인 Boolean prediction tensor
    """
    top_indices = torch.topk(logits, k=2, dim=1).indices
    predictions = torch.zeros_like(logits, dtype=torch.bool)
    predictions.scatter_(1, top_indices, True)
    return predictions


def exact_match_per_sample(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """각 sample의 두 예측 class가 정답 집합과 정확히 같은지 계산한다.

    입력:
        `[batch, class]` logit과 multi-hot label tensor

    처리:
        Top-2 Boolean prediction을 label과 class 축 전체에서 비교한다.

    출력:
        `[batch]` Boolean exact-match tensor
    """
    predictions = top_two_predictions(logits)
    return torch.all(predictions == labels.bool(), dim=1)


def classification_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """NumPy prediction array에서 전체 분류 지표를 한 번에 계산한다.

    입력:
        `[sample, class]` logit과 multi-hot label array

    처리:
        Top-2 prediction, TP/FP/FN, precision, recall, F1, exact-match를 계산한다.

    출력:
        Scalar metric, class별 array, sample별 correctness를 담은 dictionary
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
    """정답 숫자 조합별 exact-match를 대칭 행렬로 구성한다.

    입력:
        Sample별 correctness, 두 정답 class array, 전체 class 수

    처리:
        각 unordered class pair의 sample을 선택해 평균 correctness를 계산한다.

    출력:
        대각선이 NaN인 `[class, class]` accuracy matrix
    """
    accuracy_matrix = np.full((class_count, class_count), np.nan, dtype=np.float64)
    for first_class in range(class_count):
        for second_class in range(first_class + 1, class_count):
            pair_mask = (label_first == first_class) & (label_second == second_class)
            if pair_mask.any():
                pair_accuracy = float(correct_per_sample[pair_mask].mean())
                accuracy_matrix[first_class, second_class] = pair_accuracy
                accuracy_matrix[second_class, first_class] = pair_accuracy
    return accuracy_matrix


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """분모가 0인 위치를 0으로 유지하며 element-wise 나눗셈을 수행한다.

    입력:
        Shape가 호환되는 numerator와 denominator array

    처리:
        분모가 0이 아닌 위치에만 NumPy divide를 적용한다.

    출력:
        Float64 quotient array
    """
    result = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result
