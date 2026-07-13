"""모델 학습과 저장된 checkpoint 복원의 공개 경계를 제공한다.

입력:
    Config 경로, model·seed 선택, checkpoint 경로

출력:
    학습 결과 목록과 복원된 model parameter

연결:
    CLI와 pipeline은 `train_models`, evaluation과 reporting은 `load_checkpoint`를 쓴다.
"""

from .engine import load_checkpoint
from .runner import train_models

__all__ = ["load_checkpoint", "train_models"]
