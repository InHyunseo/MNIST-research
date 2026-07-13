"""YAML의 모델 이름을 독립적인 PyTorch 모델 class로 변환한다.

입력:
    모델 이름과 전체 실험 config

출력:
    초기화된 ModernizedLeNet 또는 attention model

연결:
    training, evaluation 실행 함수를 models package의 구체 class에 연결한다.
"""

from typing import Any

from torch import nn

from .class_attention import ClassConditionalAttentionLeNet
from .lenet import ModernizedLeNet
from .shared_attention import SharedAttentionLeNet


def create_model(model_name: str, config: dict[str, Any]) -> nn.Module:
    """설정 이름에 대응하는 실험 모델을 생성한다.

    입력:
        `lenet`, `shared_attention`, `class_attention` 중 하나와 전체 config

    처리:
        이름을 model class에 대응시키고 config를 전달해 초기화한다.

    출력:
        선택한 PyTorch model instance
    """
    model_classes = {
        "lenet": ModernizedLeNet,
        "shared_attention": SharedAttentionLeNet,
        "class_attention": ClassConditionalAttentionLeNet,
    }

    if model_name not in model_classes:
        raise ValueError(f"지원하지 않는 모델 이름입니다: {model_name}")

    return model_classes[model_name](config)

