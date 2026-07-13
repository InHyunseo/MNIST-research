"""Controlled Overlap MNIST 데이터 생성과 로딩의 공개 import 경로를 제공한다.

입력:
    실험 config, split 이름, 원본 MNIST

출력:
    Dataset과 manifest 생성·검증 함수

연결:
    Data 실행 함수와 training/evaluation package가 이 공개 경로를 사용한다.
"""

from .dataset import ControlledOverlapMnistDataset
from .generation import (
    prepare_data,
    validate_saved_data,
)

__all__ = [
    "ControlledOverlapMnistDataset",
    "prepare_data",
    "validate_saved_data",
]
