"""Metrics for fixed-length digit sequences."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SequenceMetrics:
    digit_accuracy: float
    exact_match: float
    sequence_count: int


class MetricAccumulator:
    def __init__(self) -> None:
        self.correct_digits = 0
        self.total_digits = 0
        self.exact_sequences = 0
        self.sequence_count = 0

    def update(self, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        if predictions.shape != labels.shape:
            raise ValueError(
                f"prediction and label shapes differ: {predictions.shape} vs {labels.shape}"
            )
        matches = predictions.eq(labels)
        self.correct_digits += int(matches.sum().item())
        self.total_digits += labels.numel()
        self.exact_sequences += int(matches.all(dim=1).sum().item())
        self.sequence_count += labels.shape[0]

    def compute(self) -> SequenceMetrics:
        if self.sequence_count == 0 or self.total_digits == 0:
            raise ValueError("cannot compute metrics without samples")
        return SequenceMetrics(
            digit_accuracy=self.correct_digits / self.total_digits,
            exact_match=self.exact_sequences / self.sequence_count,
            sequence_count=self.sequence_count,
        )


def compute_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> SequenceMetrics:
    accumulator = MetricAccumulator()
    accumulator.update(predictions, labels)
    return accumulator.compute()
