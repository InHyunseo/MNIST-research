"""Baseline과 multitask MNIST-O 실험을 선택 실행하는 저장소 통합 진입점."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mnist_overlap.baseline.main import run as run_baseline
from mnist_overlap.config import DEFAULT_CONFIG_PATH
from mnist_overlap.multitask.config import DEFAULT_MULTITASK_CONFIG_PATH
from mnist_overlap.multitask.main import run as run_multitask


def main() -> None:
    """CLI에서 선택한 모델의 전체 실행·평가·시각화 단계를 호출한다."""
    sys.stdout.reconfigure(line_buffering=True)
    arguments = _parse_arguments()

    if arguments.model in ("baseline", "all"):
        _print_pipeline_header("Baseline")
        run_baseline(
            config_path=arguments.baseline_config,
            device_name=arguments.device,
            skip_training=arguments.skip_training,
            plot_only=arguments.plot,
        )

    if arguments.model in ("multitask", "all"):
        _print_pipeline_header("Multitask")
        run_multitask(
            config_path=arguments.multitask_config,
            device_name=arguments.device,
            skip_training=arguments.skip_training,
            plot_only=arguments.plot,
        )


def _print_pipeline_header(name: str) -> None:
    """두 모델을 연속 실행할 때 현재 파이프라인을 명확히 표시한다."""
    print(f"\n{'=' * 18} {name} {'=' * 18}")


def _parse_arguments() -> argparse.Namespace:
    """모델 선택, config, device와 실행 단계를 해석한다."""
    parser = argparse.ArgumentParser(
        description="MNIST-O baseline과 reconstruction multitask 실험을 실행합니다."
    )
    parser.add_argument(
        "--model",
        choices=("baseline", "multitask", "all"),
        default="all",
        help="실행할 모델입니다. 기본값은 두 모델을 순서대로 실행하는 all입니다.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )
    parser.add_argument(
        "--multitask-config",
        type=Path,
        default=DEFAULT_MULTITASK_CONFIG_PATH,
    )
    stage_group = parser.add_mutually_exclusive_group()
    stage_group.add_argument(
        "--skip-training",
        action="store_true",
        help="완료 checkpoint를 사용해 선택 모델의 평가만 다시 수행합니다.",
    )
    stage_group.add_argument(
        "--plot",
        action="store_true",
        help="학습·평가 없이 선택 모델의 기존 결과 그림만 다시 생성합니다.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
