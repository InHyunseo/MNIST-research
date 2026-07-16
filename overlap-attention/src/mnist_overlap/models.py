"""LeNet 계열 비교 모델 세 종과 공통 backbone, 모델 factory를 정의한다.

- `ModernizedLeNet`: attention이 없는 기준 모델
- `SharedAttentionLeNet`: 모든 class가 하나의 spatial map을 공유
- `ClassConditionalAttentionLeNet`: class마다 독립적인 spatial map (permutation 평가 지원)
"""

from __future__ import annotations

import torch
from torch import nn

from .config import CLASS_COUNT, ExperimentConfig


class FeatureExtractor(nn.Module):
    """입력 이미지 `[batch, 1, H, W]`를 attention 적용 전 spatial feature로 변환한다."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        model_config = config.model

        self.layers = nn.Sequential(
            nn.Conv2d(1, model_config.first_conv_channels, model_config.kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(model_config.pool_size),
            nn.Conv2d(
                model_config.first_conv_channels,
                model_config.second_conv_channels,
                model_config.kernel_size,
            ),
            nn.ReLU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)


class SharedClassifier(nn.Module):
    """Spatial feature를 모든 모델이 공유하는 encoded vector로 변환한다."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        model_config = config.model
        flattened_features = calculate_flattened_feature_count(config)

        self.layers = nn.Sequential(
            nn.MaxPool2d(model_config.pool_size),
            nn.Flatten(),
            nn.Linear(flattened_features, model_config.first_hidden_features),
            nn.ReLU(),
            nn.Linear(
                model_config.first_hidden_features,
                model_config.second_hidden_features,
            ),
            nn.ReLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def calculate_flattened_feature_count(config: ExperimentConfig) -> int:
    """두 convolution과 pooling을 지난 뒤 flatten되는 feature 수를 계산한다."""
    spatial_size = config.dataset.canvas_size
    kernel_size = config.model.kernel_size
    pool_size = config.model.pool_size

    spatial_size = (spatial_size - kernel_size + 1) // pool_size
    spatial_size = (spatial_size - kernel_size + 1) // pool_size

    return config.model.second_conv_channels * spatial_size * spatial_size


class ModernizedLeNet(nn.Module):
    """Attention을 사용하지 않는 multi-label LeNet 기준 모델."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.feature_extractor = FeatureExtractor(config)
        self.shared_classifier = SharedClassifier(config)
        self.output_head = nn.Linear(
            config.model.second_hidden_features,
            CLASS_COUNT,
        )

    def forward(
        self,
        images: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, None]:
        features = self.feature_extractor(images)
        encoded_features = self.shared_classifier(features)
        logits = self.output_head(encoded_features)

        if return_attention:
            return logits, None

        return logits


class SharedAttentionLeNet(nn.Module):
    """하나의 spatial attention map을 전체 class 출력에 공유하는 모델."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        # 공통 layer를 기준 모델과 같은 순서로 먼저 생성해야
        # 동일 seed에서 공통 layer 초기화가 세 모델 간에 일치한다.
        self.feature_extractor = FeatureExtractor(config)
        self.shared_classifier = SharedClassifier(config)
        self.output_head = nn.Linear(
            config.model.second_hidden_features,
            CLASS_COUNT,
        )
        self.attention_layer = nn.Conv2d(
            config.model.second_conv_channels, 1, kernel_size=1
        )

    def forward(
        self,
        images: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(images)
        attention_maps = torch.sigmoid(self.attention_layer(features))
        attended_features = features * attention_maps
        encoded_features = self.shared_classifier(attended_features)
        logits = self.output_head(encoded_features)

        if return_attention:
            return logits, attention_maps

        return logits


class ClassConditionalAttentionLeNet(nn.Module):
    """Class마다 독립적인 spatial attention branch를 공유 backbone 위에 두는 모델."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.class_count = CLASS_COUNT

        # 공통 layer를 기준 모델과 같은 순서로 먼저 생성해야
        # 동일 seed에서 공통 layer 초기화가 세 모델 간에 일치한다.
        self.feature_extractor = FeatureExtractor(config)
        self.shared_classifier = SharedClassifier(config)
        self.output_head = nn.Linear(
            config.model.second_hidden_features,
            self.class_count,
        )
        self.attention_layer = nn.Conv2d(
            config.model.second_conv_channels,
            self.class_count,
            kernel_size=1,
        )

    def forward(
        self,
        images: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        logits, attention_maps = self._forward_with_attention_shift(images, shift=0)

        if return_attention:
            return logits, attention_maps

        return logits

    def forward_with_permuted_attention(
        self,
        images: torch.Tensor,
        shift: int = 1,
    ) -> torch.Tensor:
        """Class와 attention map의 대응을 순환 이동해 logit을 계산한다 (permutation 평가용)."""
        logits, _ = self._forward_with_attention_shift(images, shift=shift)
        return logits

    def _forward_with_attention_shift(
        self,
        images: torch.Tensor,
        shift: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """지정한 class-map 이동량으로 class branch 계산을 수행한다."""
        features = self.feature_extractor(images)  # [B, C, 32, 32]
        attention_maps = torch.sigmoid(self.attention_layer(features))  # [B, 10, 32, 32]

        if shift:
            selected_maps = torch.roll(attention_maps, shifts=shift, dims=1)
        else:
            selected_maps = attention_maps

        attended_features = features.unsqueeze(1) * selected_maps.unsqueeze(2)
        batch_size, class_count, feature_channels, height, width = attended_features.shape

        # Class branch를 batch 차원에 결합해 동일한 FC를 한 번 호출한다.
        flattened_branches = attended_features.reshape(
            batch_size * class_count,
            feature_channels,
            height,
            width,
        )
        encoded_branches = self.shared_classifier(flattened_branches)
        encoded_branches = encoded_branches.reshape(batch_size, class_count, -1)

        # 각 branch에는 같은 class index의 output weight만 대응시킨다.
        logits = torch.einsum(
            "bch,ch->bc",
            encoded_branches,
            self.output_head.weight,
        )
        logits = logits + self.output_head.bias

        return logits, attention_maps


MODEL_CLASSES = {
    "lenet": ModernizedLeNet,
    "shared_attention": SharedAttentionLeNet,
    "class_attention": ClassConditionalAttentionLeNet,
}


def create_model(model_name: str, config: ExperimentConfig) -> nn.Module:
    """모델 이름에 대응하는 실험 모델을 생성한다."""
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"지원하지 않는 모델 이름입니다: {model_name}")
    return MODEL_CLASSES[model_name](config)
