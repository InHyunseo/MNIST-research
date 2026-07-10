"""A multi-head CNN for fixed-length digit sequence recognition."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import canvas_shape


class CnnEncoder(nn.Module):
    """Return the spatial feature map before sequence classification."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        kernel_size = model_cfg["kernel_size"]
        pool_size = model_cfg["pool_size"]
        self.mean = float(cfg["preprocess"]["mean"])
        self.std = float(cfg["preprocess"]["std"])
        # LeNet-5의 convolution/pooling layer 수와 channel 구성을 참고한다.
        self.conv1 = nn.Conv2d(1, model_cfg["conv1_channels"], kernel_size)
        self.conv2 = nn.Conv2d(
            model_cfg["conv1_channels"], model_cfg["conv2_channels"], kernel_size
        )
        self.conv3 = nn.Conv2d(
            model_cfg["conv2_channels"], model_cfg["conv3_channels"], kernel_size
        )
        self.pool = nn.AvgPool2d(pool_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = images.to(torch.float32).div(255.0)
        features = (features - self.mean) / self.std
        features = self.pool(F.relu(self.conv1(features)))
        features = self.pool(F.relu(self.conv2(features)))
        return F.relu(self.conv3(features))


class StaticSequenceRecognizer(nn.Module):
    """Predict three position-wise digit distributions from one shared feature map."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.sequence_length = int(cfg["dataset"]["sequence_length"])
        self.encoder = CnnEncoder(cfg)
        height, width = canvas_shape(cfg)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, height, width, dtype=torch.uint8)
            flattened_features = self.encoder(dummy).numel()
        hidden_size = int(cfg["model"]["fc_hidden"])
        self.projection = nn.Linear(flattened_features, hidden_size)
        self.digit_heads = nn.ModuleList(
            nn.Linear(hidden_size, 10) for _ in range(self.sequence_length)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = torch.flatten(self.encoder(images), start_dim=1)
        embedding = F.relu(self.projection(features))
        return torch.stack([head(embedding) for head in self.digit_heads], dim=1)
