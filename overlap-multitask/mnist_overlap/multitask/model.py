"""공통 LeNet에 원본 숫자 두 장을 복원하는 decoder를 결합한다."""

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
    """LeNet spatial feature를 두 장의 `28×28` grayscale 이미지로 복원한다."""

    def __init__(self) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Linear(256, 16 * 7 * 7),
            nn.ReLU(),
        )
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 2, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """`[batch,16,16,16]` feature를 `[batch,2,28,28]` 복원으로 변환한다."""
        projected_features = self.project(features)
        spatial_features = projected_features.reshape(features.shape[0], 16, 7, 7)
        return self.decode(spatial_features)


class MultitaskMnistONet(nn.Module):
    """LeNet encoder를 분류 head와 복원 decoder가 공유하는 공동학습 모델."""

    def __init__(self) -> None:
        super().__init__()
        # 동일 seed baseline과 초기값을 맞추기 위해 classifier를 반드시 먼저 생성한다.
        self.classifier = MnistONet()
        self.decoder = ReconstructionDecoder()

    def forward(self, images: torch.Tensor) -> MultitaskOutput:
        """겹친 입력을 동시에 분류하고 두 원본 숫자를 복원한다."""
        features = self.classifier.encode(images)
        logits = self.classifier.classify_features(features)
        reconstructions = self.decoder(features)
        return MultitaskOutput(logits, reconstructions)
