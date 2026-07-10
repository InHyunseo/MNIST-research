"""Training and evaluation loops shared by Python entrypoints."""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import MetricAccumulator, SequenceMetrics


@dataclass(frozen=True)
class EpochResult:
    loss: float
    metrics: SequenceMetrics


def sequence_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits.reshape(-1, 10), labels.reshape(-1))


def train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer
) -> EpochResult:
    model.train()
    loss_sum = 0.0
    sample_count = 0
    metrics = MetricAccumulator()
    for images, labels in loader:
        optimizer.zero_grad()
        logits = model(images)
        loss = sequence_loss(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        loss_sum += float(loss.item()) * batch_size
        sample_count += batch_size
        metrics.update(logits.detach().argmax(dim=2), labels)
    return EpochResult(loss=loss_sum / sample_count, metrics=metrics.compute())


@torch.inference_mode()
def evaluate_model(model: nn.Module, loader: DataLoader) -> EpochResult:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    metrics = MetricAccumulator()
    for images, labels in loader:
        logits = model(images)
        loss = sequence_loss(logits, labels)
        batch_size = labels.shape[0]
        loss_sum += float(loss.item()) * batch_size
        sample_count += batch_size
        metrics.update(logits.argmax(dim=2), labels)
    return EpochResult(loss=loss_sum / sample_count, metrics=metrics.compute())
