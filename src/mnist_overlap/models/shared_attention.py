"""모든 class가 하나의 spatial map을 공유하는 LeNet 모델을 정의한다.

입력:
    `[batch, 1, 76, 76]` Controlled Overlap MNIST 이미지

출력:
    `[batch, 10]` logit과 `[batch, 1, 32, 32]` attention map

연결:
    factory에서 생성되며 backbone과 attention 평가 모듈에서 사용한다.
"""

from typing import Any

import torch
from torch import nn

from .backbone import FeatureExtractor, SharedClassifier


class SharedAttentionLeNet(nn.Module):
    """하나의 spatial attention map을 전체 class 출력에 공유한다.

    입력:
        전체 config와 Controlled Overlap MNIST image batch

    처리:
        공통 feature에 sigmoid attention을 곱한 뒤 shared classifier를 적용한다.

    출력:
        `[batch, 10]` logit과 선택적인 shared attention map
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """공통 layer를 먼저 만들고 shared attention layer를 생성한다.

        입력:
            model section이 포함된 전체 config

        처리:
            동일 seed의 공통 layer가 기준 모델과 같도록 생성 순서를 유지한다.

        출력:
            초기화된 SharedAttentionLeNet instance
        """
        super().__init__()
        model_config = config["model"]
        feature_channels = int(model_config["second_conv_channels"])

        self.feature_extractor = FeatureExtractor(config)
        self.shared_classifier = SharedClassifier(config)
        self.output_head = nn.Linear(
            int(model_config["second_hidden_features"]),
            int(model_config["class_count"]),
        )
        self.attention_layer = nn.Conv2d(feature_channels, 1, kernel_size=1)

    def forward(
        self,
        images: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Shared attention을 적용해 class logit을 계산한다.

        입력:
            이미지 tensor와 attention 반환 여부

        처리:
            feature별 spatial weight를 곱하고 공통 classifier를 적용한다.

        출력:
            logit 또는 `(logit, shared_attention_maps)` tuple
        """
        features = self.feature_extractor(images)
        attention_maps = torch.sigmoid(self.attention_layer(features))
        attended_features = features * attention_maps
        encoded_features = self.shared_classifier(attended_features)
        logits = self.output_head(encoded_features)

        if return_attention:
            return logits, attention_maps

        return logits

