"""Attention을 사용하지 않는 Modernized LeNet 기준 모델을 정의한다.

입력:
    `[batch, 1, 76, 76]` Controlled Overlap MNIST 이미지

출력:
    `[batch, 10]` multi-label logit

연결:
    factory에서 생성되며 backbone의 공통 feature extractor와 classifier를 사용한다.
"""

from typing import Any

import torch
from torch import nn

from .backbone import FeatureExtractor, SharedClassifier


class ModernizedLeNet(nn.Module):
    """세 실험 모델의 기준이 되는 multi-label LeNet을 구현한다.

    입력:
        전체 config와 Controlled Overlap MNIST image batch

    처리:
        공통 backbone을 통과한 feature를 열 개 class logit으로 변환한다.

    출력:
        `[batch, 10]` logit과 요청 시 `None` attention
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """공통 backbone과 열 개 class output head를 생성한다.

        입력:
            model section이 포함된 전체 config

        처리:
            feature extractor, shared classifier, linear head를 순서대로 초기화한다.

        출력:
            초기화된 ModernizedLeNet instance
        """
        super().__init__()
        model_config = config["model"]

        self.feature_extractor = FeatureExtractor(config)
        self.shared_classifier = SharedClassifier(config)
        self.output_head = nn.Linear(
            int(model_config["second_hidden_features"]),
            int(model_config["class_count"]),
        )

    def forward(
        self,
        images: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, None]:
        """입력 이미지에서 열 개 class logit을 계산한다.

        입력:
            이미지 tensor와 attention 반환 여부

        처리:
            feature 추출, shared classification, output projection을 수행한다.

        출력:
            logit 또는 `(logit, None)` tuple
        """
        features = self.feature_extractor(images)
        encoded_features = self.shared_classifier(features)
        logits = self.output_head(encoded_features)

        if return_attention:
            return logits, None

        return logits

