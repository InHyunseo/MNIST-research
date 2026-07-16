"""겹친 두 숫자를 동시에 분류하는 LeNet 계열 고정 모델을 정의한다."""

from __future__ import annotations

import torch
from torch import nn


class MnistONet(nn.Module):
    """`76×76` 흑백 입력을 class별 독립 logit 10개로 변환한다."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(4096, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """`[batch, 1, 76, 76]` 이미지를 `[batch, 10]` logit으로 변환한다."""
        return self.classify_features(self.encode(images))

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """`[batch, 1, 76, 76]` 입력을 `[batch, 16, 16, 16]` feature로 변환한다."""
        features = images
        for layer_index in range(6):
            features = self.layers[layer_index](features)
        return features

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        """Encoder feature를 기존 fully-connected head의 10개 logit으로 변환한다."""
        logits = features
        for layer_index in range(6, len(self.layers)):
            logits = self.layers[layer_index](logits)
        return logits
