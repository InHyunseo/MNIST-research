"""저장된 checkpoint의 최종 평가 실행 경계를 제공한다.

입력:
    Config 경로, model·seed 선택, 실행 device

출력:
    Prediction, classification·attention metric, bootstrap, model cost CSV

연결:
    CLI와 pipeline이 `evaluate_models`만 호출하고 세부 계산은 하위 모듈에 위임한다.
"""

from .runner import evaluate_models

__all__ = ["evaluate_models"]
