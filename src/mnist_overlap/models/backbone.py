"""세 비교 모델이 공유하는 LeNet feature extractor와 classifier를 정의한다.

입력:
    모델 channel 수와 hidden feature 수가 포함된 실험 설정

출력:
    convolution feature와 84차원 encoded feature를 생성하는 PyTorch module

연결:
    lenet, shared_attention, class_attention 모델이 공통 backbone으로 사용한다.
"""

from typing import Any

import torch
from torch import nn


class FeatureExtractor(nn.Module):
    """입력 이미지를 attention 적용 전 spatial feature로 변환한다.

    입력:
        config와 `[batch, 1, 76, 76]` 이미지 tensor

    처리:
        두 convolution, ReLU, 첫 번째 max pooling을 순서대로 적용한다.

    출력:
        `[batch, 16, 32, 32]` feature tensor
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """설정에 맞는 convolution layer를 생성한다.

        입력:
            model section이 포함된 전체 config

        처리:
            channel, kernel, pooling 크기를 읽어 sequential layer를 구성한다.

        출력:
            초기화된 FeatureExtractor instance
        """
        super().__init__()
        model_config = config["model"]
        kernel_size = int(model_config["kernel_size"])
        pool_size = int(model_config["pool_size"])

        self.layers = nn.Sequential(
            nn.Conv2d(1, int(model_config["first_conv_channels"]), kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(pool_size),
            nn.Conv2d(
                int(model_config["first_conv_channels"]),
                int(model_config["second_conv_channels"]),
                kernel_size,
            ),
            nn.ReLU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """이미지 batch에서 spatial feature를 추출한다.

        입력:
            `[batch, 1, 76, 76]` float image tensor

        처리:
            구성된 convolution block을 한 번 통과시킨다.

        출력:
            `[batch, 16, 32, 32]` feature tensor
        """
        return self.layers(images)


class SharedClassifier(nn.Module):
    """공통 spatial feature를 모든 모델이 공유하는 encoded vector로 변환한다.

    입력:
        config와 `[batch, 16, 32, 32]` feature tensor

    처리:
        max pooling, flatten, 두 fully-connected layer를 적용한다.

    출력:
        `[batch, 84]` encoded feature tensor
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """canvas 크기에서 flatten 크기를 계산해 classifier를 생성한다.

        입력:
            dataset 및 model section이 포함된 전체 config

        처리:
            feature spatial 크기와 두 hidden layer 크기를 계산한다.

        출력:
            초기화된 SharedClassifier instance
        """
        super().__init__()
        model_config = config["model"]
        flattened_features = calculate_flattened_feature_count(config)

        self.layers = nn.Sequential(
            nn.MaxPool2d(int(model_config["pool_size"])),
            nn.Flatten(),
            nn.Linear(flattened_features, int(model_config["first_hidden_features"])),
            nn.ReLU(),
            nn.Linear(
                int(model_config["first_hidden_features"]),
                int(model_config["second_hidden_features"]),
            ),
            nn.ReLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Spatial feature를 공유 encoded vector로 변환한다.

        입력:
            `[batch, 16, 32, 32]` feature tensor

        처리:
            pooling과 fully-connected block을 적용한다.

        출력:
            `[batch, 84]` encoded feature tensor
        """
        return self.layers(features)


def calculate_flattened_feature_count(config: dict[str, Any]) -> int:
    """두 convolution과 pooling을 지난 flatten feature 수를 계산한다.

    입력:
        canvas, kernel, pooling, channel 설정

    처리:
        각 layer의 valid-convolution spatial 크기를 순서대로 계산한다.

    출력:
        첫 fully-connected layer의 정수 input feature 수
    """
    dataset_config = config["dataset"]
    model_config = config["model"]
    spatial_size = int(dataset_config["canvas_size"])
    kernel_size = int(model_config["kernel_size"])
    pool_size = int(model_config["pool_size"])

    spatial_size = (spatial_size - kernel_size + 1) // pool_size
    spatial_size = (spatial_size - kernel_size + 1) // pool_size

    feature_channels = int(model_config["second_conv_channels"])
    return feature_channels * spatial_size * spatial_size

