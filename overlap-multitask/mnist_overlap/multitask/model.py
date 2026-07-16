"""공통 LeNet encoder에 U-Net expansive path를 결합한다."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..model import MnistONet


@dataclass(frozen=True)
class MultitaskOutput:
    """분류 logit과 순서 없는 두 원본 숫자 복원 결과."""

    logits: torch.Tensor
    reconstructions: torch.Tensor


class ReconstructionDecoder(nn.Module):
    """LeNet의 세 해상도를 연결해 두 장의 `64×64` source layer를 분리한다.

    원 U-Net의 `up-convolution → encoder feature concat → double convolution`
    순서를 두 해상도에 적용한다. LeNet 분류 구조를 보존해야 하므로 contracting
    path의 channel 수와 convolution은 기존 LeNet을 그대로 사용한다.
    """

    def __init__(self) -> None:
        super().__init__()
        self.bottleneck = DoubleConvolution(16, 32)
        self.up_middle = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.refine_middle = DoubleConvolution(32, 16)
        self.up_high = nn.ConvTranspose2d(16, 6, kernel_size=2, stride=2)
        self.refine_high = DoubleConvolution(12, 6)
        self.output = nn.Conv2d(6, 2, kernel_size=1)

    def forward(
        self,
        high_resolution: torch.Tensor,
        middle_resolution: torch.Tensor,
        bottleneck: torch.Tensor,
    ) -> torch.Tensor:
        """LeNet feature tuple을 `[batch,2,64,64]` source layer로 변환한다."""
        decoded_middle = self.up_middle(self.bottleneck(bottleneck))
        decoded_middle = self.refine_middle(torch.cat(
            (decoded_middle, middle_resolution),
            dim=1,
        ))

        decoded_high = self.up_high(decoded_middle)
        cropped_high_resolution = center_crop_like(high_resolution, decoded_high)
        decoded_high = self.refine_high(torch.cat(
            (decoded_high, cropped_high_resolution),
            dim=1,
        ))
        return torch.sigmoid(self.output(decoded_high))


class DoubleConvolution(nn.Sequential):
    """U-Net expansive path의 연속된 `3×3 convolution + ReLU` 두 회."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )


def center_crop_like(
    features: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """U-Net skip feature를 reference의 spatial shape에 맞춰 중앙 crop한다."""
    target_height, target_width = reference.shape[-2:]
    source_height, source_width = features.shape[-2:]
    if source_height < target_height or source_width < target_width:
        raise ValueError("Skip feature가 decoder feature보다 작아 중앙 crop할 수 없습니다.")
    start_y = (source_height - target_height) // 2
    start_x = (source_width - target_width) // 2
    return features[
        :,
        :,
        start_y:start_y + target_height,
        start_x:start_x + target_width,
    ]


class MultitaskMnistONet(nn.Module):
    """LeNet encoder를 분류 head와 복원 decoder가 공유하는 공동학습 모델."""

    def __init__(self) -> None:
        super().__init__()
        # 동일 seed baseline과 초기값을 맞추기 위해 classifier를 반드시 먼저 생성한다.
        self.classifier = MnistONet()
        self.decoder = ReconstructionDecoder()

    def forward(self, images: torch.Tensor) -> MultitaskOutput:
        """겹친 입력을 동시에 분류하고 두 원본 숫자를 복원한다."""
        high_resolution, middle_resolution, bottleneck = (
            self.classifier.encode_with_skips(images)
        )
        logits = self.classifier.classify_features(bottleneck)
        reconstructions = self.decoder(
            high_resolution,
            middle_resolution,
            bottleneck,
        )
        return MultitaskOutput(logits, reconstructions)
