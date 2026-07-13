"""최종 결과 생성 기능의 공개 import 경로를 제공한다.

입력:
    평가 CSV와 전체 실험 config

출력:
    표, 성능·attention 그림, Markdown summary

연결:
    Report 실행 함수가 generator 구현을 이 package 경계로 호출한다.
"""

from .generator import create_report

__all__ = ["create_report"]
