"""밝은 획과 배경을 균형화한 permutation-invariant reconstruction loss다."""

from __future__ import annotations

from dataclasses import dataclass

import torch


LOSS_EPSILON = 1e-8


@dataclass(frozen=True)
class PitLossResult:
    """Batch 평균 loss, sample별 loss와 swapped assignment 선택 결과."""

    loss: torch.Tensor
    per_sample_loss: torch.Tensor
    swapped: torch.Tensor


def intensity_balanced_l1_distance(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """밝은 획과 배경의 L1을 각각 정규화해 sample별 동일 비중으로 합친다."""
    if predictions.shape != targets.shape or predictions.ndim != 3:
        raise ValueError("Balanced L1 입력은 같은 `[batch,height,width]` 형태여야 합니다.")

    absolute_error = torch.abs(predictions - targets)
    spatial_dimensions = (1, 2)
    foreground_weights = targets
    background_weights = 1.0 - targets

    foreground_error = (
        (foreground_weights * absolute_error).sum(dim=spatial_dimensions)
        / foreground_weights.sum(dim=spatial_dimensions).clamp_min(LOSS_EPSILON)
    )
    background_error = (
        (background_weights * absolute_error).sum(dim=spatial_dimensions)
        / background_weights.sum(dim=spatial_dimensions).clamp_min(LOSS_EPSILON)
    )
    return 0.5 * (foreground_error + background_error)


def permutation_invariant_reconstruction_loss(
    reconstructions: torch.Tensor,
    source_images: torch.Tensor,
) -> PitLossResult:
    """두 assignment의 balanced L1을 sample별로 비교해 더 작은 값을 선택한다."""
    expected_shape = (reconstructions.shape[0], 2, 28, 28)
    if reconstructions.shape != source_images.shape or reconstructions.shape != expected_shape:
        raise ValueError("PIT 입력은 같은 `[batch,2,28,28]` 형태여야 합니다.")

    direct_loss = (
        intensity_balanced_l1_distance(reconstructions[:, 0], source_images[:, 0])
        + intensity_balanced_l1_distance(reconstructions[:, 1], source_images[:, 1])
    )
    swapped_loss = (
        intensity_balanced_l1_distance(reconstructions[:, 0], source_images[:, 1])
        + intensity_balanced_l1_distance(reconstructions[:, 1], source_images[:, 0])
    )
    swapped = swapped_loss < direct_loss
    per_sample_loss = torch.where(swapped, swapped_loss, direct_loss)
    return PitLossResult(per_sample_loss.mean(), per_sample_loss, swapped)


def match_reconstructions_to_sources(
    reconstructions: torch.Tensor,
    swapped: torch.Tensor,
) -> torch.Tensor:
    """PIT 선택에 맞춰 두 reconstruction channel 순서를 source 순서로 정렬한다."""
    swapped_reconstructions = reconstructions[:, [1, 0]]
    swap_mask = swapped.reshape(-1, 1, 1, 1)
    return torch.where(swap_mask, swapped_reconstructions, reconstructions)
