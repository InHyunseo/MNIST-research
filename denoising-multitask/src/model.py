"""
PyTorch 공식 tutorial의 LeNet을 encoder와 head로 나누고 denoising decoder를 결합한다.

입력:
    - batch 형태의 1×32×32 noisy image

출력:
    - 10-class classification logits
    - multitask 조건의 1×32×32 reconstruction

주요 기능:
    1. LeNet convolutional encoder
    2. LeNet fully-connected classification head
    3. Transposed-convolution denoising decoder
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as functional


class SharedEncoder(nn.Module):
    """32×32 입력을 16×5×5 spatial bottleneck으로 변환한다."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = functional.max_pool2d(
            functional.relu(self.conv1(images)), kernel_size=2
        )
        return functional.max_pool2d(
            functional.relu(self.conv2(features)), kernel_size=2
        )


class ClassificationHead(nn.Module):
    """Spatial bottleneck에서 10-class logits를 계산한다."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        flattened_features = torch.flatten(features, 1)
        hidden_features = functional.relu(self.fc1(flattened_features))
        hidden_features = functional.relu(self.fc2(hidden_features))
        return self.fc3(hidden_features)


class DenoisingDecoder(nn.Module):
    """16×5×5 bottleneck에서 linear 1×32×32 clean image를 복원한다."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(16, 6, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 1, kernel_size=6, stride=2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        reconstruction = self.layers(features)
        if reconstruction.shape[-2:] != (32, 32):
            raise RuntimeError(
                "Decoder 출력의 spatial shape이 32×32가 아닙니다: "
                f"{tuple(reconstruction.shape)}"
            )
        return reconstruction


class DenoisingAuxiliaryLeNet(nn.Module):
    """동일한 encoder·head와 조건별 decoder 유무를 결합한 모델이다."""

    def __init__(self, use_decoder: bool) -> None:
        super().__init__()
        # Decoder 유무와 관계없이 shared parameter 초기화 순서를 동일하게 유지한다.
        self.encoder = SharedEncoder()
        self.classification_head = ClassificationHead()
        self.decoder = DenoisingDecoder() if use_decoder else None

    def forward(
        self, noisy_images: torch.Tensor
    ) -> dict[str, torch.Tensor | None]:
        features = self.encoder(noisy_images)
        classification_logits = self.classification_head(features)
        reconstruction = self.decoder(features) if self.decoder is not None else None
        return {
            "classification_logits": classification_logits,
            "reconstruction": reconstruction,
        }
