"""Class별 U-Net 출력에 사용하는 foreground-balanced BCE와 Dice loss다."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as functional

from ..config import CLASS_COUNT


LOSS_EPSILON = 1e-8
BCE_LOSS_WEIGHT = 0.5
DICE_LOSS_WEIGHT = 0.5


@dataclass(frozen=True)
class SemanticLossResult:
    """Batch 평균 reconstruction loss와 sample별 구성 항목."""

    loss: torch.Tensor
    per_sample_loss: torch.Tensor
    balanced_bce_per_sample: torch.Tensor
    dice_loss_per_sample: torch.Tensor


def semantic_reconstruction_loss(
    reconstruction_logits: torch.Tensor,
    reconstruction_targets: torch.Tensor,
) -> SemanticLossResult:
    """Class별 pixel BCE와 활성 class Dice를 같은 비중으로 결합한다.

    Foreground와 background BCE는 각각 pixel 수로 정규화한 뒤 1:1로 합친다.
    Dice는 sample마다 실제 숫자가 존재하는 두 class channel에만 계산한다.
    """
    _validate_semantic_tensors(reconstruction_logits, reconstruction_targets)
    spatial_dimensions = (1, 2, 3)
    foreground_weights = reconstruction_targets
    background_weights = 1.0 - reconstruction_targets
    foreground_bce = (
        (
            foreground_weights * functional.softplus(-reconstruction_logits)
        ).sum(dim=spatial_dimensions)
        / foreground_weights.sum(dim=spatial_dimensions).clamp_min(LOSS_EPSILON)
    )
    background_bce = (
        (
            background_weights * functional.softplus(reconstruction_logits)
        ).sum(dim=spatial_dimensions)
        / background_weights.sum(dim=spatial_dimensions).clamp_min(LOSS_EPSILON)
    )
    balanced_bce = 0.5 * (foreground_bce + background_bce)

    probabilities = torch.sigmoid(reconstruction_logits)
    foreground_dice = active_foreground_dice_per_sample(
        probabilities,
        reconstruction_targets,
    )
    dice_loss = 1.0 - foreground_dice
    per_sample_loss = (
        BCE_LOSS_WEIGHT * balanced_bce
        + DICE_LOSS_WEIGHT * dice_loss
    )
    return SemanticLossResult(
        per_sample_loss.mean(),
        per_sample_loss,
        balanced_bce,
        dice_loss,
    )


def active_foreground_dice_per_sample(
    reconstruction_probabilities: torch.Tensor,
    reconstruction_targets: torch.Tensor,
) -> torch.Tensor:
    """실제 숫자가 존재하는 두 class channel의 soft Dice를 sample별 평균한다."""
    _validate_semantic_tensors(
        reconstruction_probabilities,
        reconstruction_targets,
    )
    spatial_dimensions = (2, 3)
    intersection = (
        reconstruction_probabilities * reconstruction_targets
    ).sum(dim=spatial_dimensions)
    denominator = (
        torch.square(reconstruction_probabilities).sum(dim=spatial_dimensions)
        + torch.square(reconstruction_targets).sum(dim=spatial_dimensions)
    )
    dice_by_class = (2.0 * intersection + LOSS_EPSILON) / (
        denominator + LOSS_EPSILON
    )
    active_classes = reconstruction_targets.sum(dim=spatial_dimensions) > LOSS_EPSILON
    active_class_counts = active_classes.sum(dim=1)
    if torch.any(active_class_counts != 2):
        raise ValueError("각 sample의 semantic target에는 활성 class가 정확히 두 개여야 합니다.")
    return (dice_by_class * active_classes).sum(dim=1) / active_class_counts


def _validate_semantic_tensors(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    """Semantic prediction과 target의 공통 `[B,10,H,W]` 계약을 검사한다."""
    valid_shape = predictions.ndim == 4 and predictions.shape[1] == CLASS_COUNT
    if predictions.shape != targets.shape or not valid_shape:
        raise ValueError("Semantic 입력은 같은 `[batch,10,height,width]` 형태여야 합니다.")
    if torch.any(targets < 0.0) or torch.any(targets > 1.0):
        raise ValueError("Semantic target 값은 `[0,1]` 범위여야 합니다.")
