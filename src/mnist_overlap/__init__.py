"""MNIST Overlap Attention의 공개 Python 인터페이스를 제공한다.

입력:
    YAML 실험 설정과 모델 생성 요청

출력:
    설정 로더와 모델 생성 함수

연결:
    외부 application이 config와 model factory를 안정된 경로로 import하게 한다.
"""

from .configuration import load_config
from .models import create_model

__all__ = ["create_model", "load_config"]
