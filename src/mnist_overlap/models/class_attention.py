"""Class마다 독립적인 spatial map을 사용하는 LeNet 모델을 정의한다.

입력:
    `[batch, 1, 76, 76]` Controlled Overlap MNIST 이미지

출력:
    `[batch, 10]` logit과 `[batch, 10, 32, 32]` attention map

연결:
    factory에서 생성되며 permutation 및 stroke-alignment 평가에 사용한다.
"""

from typing import Any

import torch
from torch import nn

from .backbone import FeatureExtractor, SharedClassifier


class ClassConditionalAttentionLeNet(nn.Module):
    """열 개 class별 spatial attention branch를 공유 backbone 위에 구성한다.

    입력:
        전체 config와 Controlled Overlap MNIST image batch

    처리:
        class map별 feature를 shared classifier에 전달하고 대응하는 head row를 적용한다.

    출력:
        `[batch, 10]` logit과 선택적인 class attention map
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """공통 layer를 먼저 만들고 열 개 attention channel을 생성한다.

        입력:
            model section이 포함된 전체 config

        처리:
            기준 모델과 같은 공통 초기화를 유지한 뒤 attention layer를 추가한다.

        출력:
            초기화된 ClassConditionalAttentionLeNet instance
        """
        super().__init__()
        model_config = config["model"]
        self.class_count = int(model_config["class_count"])
        feature_channels = int(model_config["second_conv_channels"])

        self.feature_extractor = FeatureExtractor(config)
        self.shared_classifier = SharedClassifier(config)
        self.output_head = nn.Linear(
            int(model_config["second_hidden_features"]),
            self.class_count,
        )
        self.attention_layer = nn.Conv2d(
            feature_channels,
            self.class_count,
            kernel_size=1,
        )

    def forward(
        self,
        images: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """정상 class-map 대응으로 logit을 계산한다.

        입력:
            이미지 tensor와 attention 반환 여부

        처리:
            class attention shift가 0인 내부 forward를 호출한다.

        출력:
            logit 또는 `(logit, class_attention_maps)` tuple
        """
        logits, attention_maps = self._forward_with_attention_shift(images, shift=0)

        if return_attention:
            return logits, attention_maps

        return logits

    def forward_with_permuted_attention(
        self,
        images: torch.Tensor,
        shift: int = 1,
    ) -> torch.Tensor:
        """Class와 attention map의 대응을 순환 이동해 logit을 계산한다.

        입력:
            이미지 tensor와 class 방향의 정수 이동량

        처리:
            attention channel을 이동해 원래와 다른 class branch에 연결한다.

        출력:
            `[batch, 10]` permuted-attention logit
        """
        logits, _ = self._forward_with_attention_shift(images, shift=shift)
        return logits

    def _forward_with_attention_shift(
        self,
        images: torch.Tensor,
        shift: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """지정된 class-map 이동량으로 공통 class branch 계산을 수행한다.

        입력:
            이미지 tensor와 attention channel 이동량

        처리:
            class feature를 `[batch × class, channel, height, width]`로 묶어 FC를 공유한다.

        출력:
            `[batch, 10]` logit과 원래 class attention map
        """
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

