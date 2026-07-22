"""
n-MNIST denoising auxiliary 실험의 단일 실행 진입점이다.

입력:
    - data, train-baseline, train-multitask, train-alignment, plot command
    - 학습 command의 선택적 device 지정

출력:
    - 준비된 dataset, 학습 checkpoint와 history, 결과 CSV와 figure

주요 기능:
    1. command line argument 해석
    2. 학습 device 선택
    3. 각 src module의 공개 함수 호출
"""

from __future__ import annotations

import argparse

import torch

from src.dataset import prepare_data
from src.experiment import (
    run_baseline_experiments,
    run_gradient_alignment_experiments,
    run_multitask_experiments,
)
from src.plot import create_plots


def resolve_device(requested_device: str) -> torch.device:
    """요청한 학습 device를 확인해 PyTorch device로 반환한다."""
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 요청했지만 현재 환경에서 사용할 수 없습니다.")
    return torch.device(requested_device)


def parse_arguments() -> argparse.Namespace:
    """필요한 command와 학습 device만 command line에서 받는다."""
    parser = argparse.ArgumentParser(
        description="n-MNIST denoising auxiliary classification 실험"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("data", help="n-MNIST와 clean target을 준비합니다.")

    training_commands = {
        "train-baseline": "세 noise의 classification-only 모델을 학습합니다.",
        "train-multitask": "Noise별 weight pilot 후 multitask 모델을 학습합니다.",
        "train-alignment": "Multitask 재학습 중 CE·MSE gradient를 비교합니다.",
    }
    for command, help_text in training_commands.items():
        training_parser = subparsers.add_parser(command, help=help_text)
        training_parser.add_argument(
            "--device",
            choices=("auto", "cpu", "cuda"),
            default="auto",
            help="학습 device입니다. 기본값은 auto입니다.",
        )

    subparsers.add_parser("plot", help="저장된 최종 결과의 figure를 생성합니다.")
    return parser.parse_args()


def main() -> None:
    """선택한 command를 해당 module의 public function으로 전달한다."""
    arguments = parse_arguments()
    if arguments.command == "data":
        prepare_data()
        return
    if arguments.command == "plot":
        create_plots()
        return

    device = resolve_device(arguments.device)
    print(f"학습 device: {device}")
    if arguments.command == "train-baseline":
        run_baseline_experiments(device)
    elif arguments.command == "train-multitask":
        run_multitask_experiments(device)
    else:
        run_gradient_alignment_experiments(device)


if __name__ == "__main__":
    main()
