"""두 source 출력에 사용하는 foreground-balanced BCE와 Dice loss다."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as functional


LOSS_EPSILON = 1e-8
BCE_LOSS_WEIGHT = 0.5
DICE_LOSS_WEIGHT = 0.5
SOURCE_COUNT = 2


@dataclass(frozen=True)
class ReconstructionLossResult:
    """Batch 평균 reconstruction loss와 sample별 구성 항목."""

    loss: torch.Tensor
    per_sample_loss: torch.Tensor
    balanced_bce_per_sample: torch.Tensor
    dice_loss_per_sample: torch.Tensor


def source_reconstruction_loss(
    reconstruction_logits: torch.Tensor,
    reconstruction_targets: torch.Tensor,
) -> ReconstructionLossResult:
    """Source별 pixel BCE와 foreground Dice를 같은 비중으로 결합한다.

    Foreground와 background BCE는 각각 pixel 수로 정규화한 뒤 1:1로 합친다.
    따라서 검은 배경 pixel이 많은 것만으로 loss가 작아지지 않는다.
    """
    _validate_source_tensors(reconstruction_logits, reconstruction_targets)
    spatial_dimensions = (1, 2, 3)
    foreground_weights = reconstruction_targets
    background_weights = 1.0 - reconstruction_targets
    foreground_bce = (
        (foreground_weights * functional.softplus(-reconstruction_logits)).sum(
            dim=spatial_dimensions
        )
        / foreground_weights.sum(dim=spatial_dimensions).clamp_min(LOSS_EPSILON)
    )
    background_bce = (
        (background_weights * functional.softplus(reconstruction_logits)).sum(
            dim=spatial_dimensions
        )
        / background_weights.sum(dim=spatial_dimensions).clamp_min(LOSS_EPSILON)
    )
    balanced_bce = 0.5 * (foreground_bce + background_bce)

    probabilities = torch.sigmoid(reconstruction_logits)
    foreground_dice = foreground_dice_per_sample(
        probabilities,
        reconstruction_targets,
    )
    dice_loss = 1.0 - foreground_dice
    per_sample_loss = (
        BCE_LOSS_WEIGHT * balanced_bce
        + DICE_LOSS_WEIGHT * dice_loss
    )
    return ReconstructionLossResult(
        per_sample_loss.mean(),
        per_sample_loss,
        balanced_bce,
        dice_loss,
    )


def foreground_dice_per_sample(
    reconstruction_probabilities: torch.Tensor,
    reconstruction_targets: torch.Tensor,
) -> torch.Tensor:
    """두 source channel의 soft Dice를 각각 계산해 sample별 평균한다."""
    _validate_source_tensors(
        reconstruction_probabilities,
        reconstruction_targets,
    )
    spatial_dimensions = (2, 3)
    target_foreground = reconstruction_targets.sum(dim=spatial_dimensions)
    if torch.any(target_foreground <= LOSS_EPSILON):
        raise ValueError("각 source target에는 foreground가 존재해야 합니다.")
    intersection = (
        reconstruction_probabilities * reconstruction_targets
    ).sum(dim=spatial_dimensions)
    denominator = (
        torch.square(reconstruction_probabilities).sum(dim=spatial_dimensions)
        + torch.square(reconstruction_targets).sum(dim=spatial_dimensions)
    )
    dice_by_source = (2.0 * intersection + LOSS_EPSILON) / (
        denominator + LOSS_EPSILON
    )
    return dice_by_source.mean(dim=1)


def _validate_source_tensors(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    """Prediction과 target의 공통 `[B,2,H,W]` 계약을 검사한다."""
    valid_shape = predictions.ndim == 4 and predictions.shape[1] == SOURCE_COUNT
    if predictions.shape != targets.shape or not valid_shape:
        raise ValueError("복원 입력은 같은 `[batch,2,height,width]` 형태여야 합니다.")
    if torch.any(targets < 0.0) or torch.any(targets > 1.0):
        raise ValueError("복원 target 값은 `[0,1]` 범위여야 합니다.")
