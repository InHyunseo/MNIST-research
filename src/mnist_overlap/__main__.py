"""`python -m mnist_overlap` 실행을 단일 CLI로 전달한다.

입력:
    터미널에서 전달된 command와 option

출력:
    선택한 실행 단계의 터미널 출력과 종료 코드

연결:
    Python module 실행 방식을 mnist_overlap.cli에 연결한다.
"""

from .cli import main


if __name__ == "__main__":
    main()
