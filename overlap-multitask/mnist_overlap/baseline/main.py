"""
MNIST-O 실험의 단독 실행 진입점이다. 데이터 준비·학습·추론·평가·시각화 모듈을 조합해
실행한다. 인자 없이 실행하면 데이터 준비부터 학습·추론·평가까지 수행하고, `--plot`을
주면 학습을 다시 하지 않고 기존 checkpoint로 추론한 뒤 시각화만 수행한다.
`python -m mnist_overlap.baseline.main` 또는 호환 명령 `python -m mnist_overlap.main`으로
실행한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from ..config import DEFAULT_CONFIG_PATH, ExperimentConfig, create_output_directories, load_config
from ..data import ControlledOverlapMnistDataset, prepare_data
from ..runtime import select_device
from .evaluation import Evaluator
from .inference import Predictor
from .plot import Visualizer
from .training import train_all_seeds


def main() -> None:
    """
    입력: 명령행 인자
    출력: 없음 (checkpoint·metrics.json·그림 파일 생성과 stdout 요약)

    --plot이면 기존 checkpoint로 시각화만, 아니면 준비→학습→추론→평가를 실행한다.
    """
    sys.stdout.reconfigure(line_buffering=True)
    arguments = _parse_arguments()
    run(
        config_path=arguments.config,
        device_name=arguments.device,
        skip_training=arguments.skip_training,
        plot_only=arguments.plot,
    )


def run(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    device_name: str = "cpu",
    skip_training: bool = False,
    plot_only: bool = False,
) -> None:
    """통합 진입점에서도 재사용할 수 있도록 baseline 파이프라인을 실행한다."""
    config_path = Path(config_path)
    config = load_config(config_path)
    create_output_directories()

    device = select_device(device_name)
    training_seeds = list(config.training.seeds)

    if plot_only:
        _visualize_only(config, device, training_seeds)
    else:
        _run_experiment(
            config,
            device,
            training_seeds,
            config_path,
            device_name,
            skip_training,
        )


def _run_experiment(
    config: ExperimentConfig,
    device: torch.device,
    training_seeds: list[int],
    config_path: Path,
    device_name: str,
    skip_training: bool,
) -> None:
    """
    입력: config, device, training_seeds, config 경로와 실행 옵션
    출력: 없음 (checkpoint·metrics.json 생성과 stdout 요약)

    데이터 준비 → 학습 → 추론 → 평가를 수행한다.
    """
    print(f"MNIST-O baseline — seeds={training_seeds}, device={device_name}")

    print("\n[1/3] 데이터 준비")
    prepare_data(config)

    if not skip_training:
        print(f"\n[2/3] 학습 — {len(training_seeds)} seeds")
        train_all_seeds(config_path, device_name)
    else:
        print("\n[2/3] 학습 생략 (--skip-training)")

    print("\n[3/3] 추론·평가")
    test_dataset = ControlledOverlapMnistDataset("test")
    predictions_by_seed = Predictor(config, device).collect_all_seeds(test_dataset, training_seeds)

    evaluator = Evaluator(config)
    metrics = evaluator.evaluate(predictions_by_seed, training_seeds)
    evaluator.report(metrics)
    evaluator.save(metrics)


def _visualize_only(
    config: ExperimentConfig,
    device: torch.device,
    training_seeds: list[int],
) -> None:
    """
    입력: config, device, training_seeds
    출력: 없음 (results/baseline/figures/*.png 생성)

    학습·평가를 재실행하지 않고 기존 checkpoint로 추론한 뒤 시각화만 수행한다.
    """
    print("시각화만 실행 (--plot) — 기존 checkpoint 사용")

    prepare_data(config)

    test_dataset = ControlledOverlapMnistDataset("test")
    predictions_by_seed = Predictor(config, device).collect_all_seeds(test_dataset, training_seeds)

    Visualizer().make_all(test_dataset, predictions_by_seed, training_seeds)


def _parse_arguments() -> argparse.Namespace:
    """
    입력: 없음 (sys.argv 사용)
    출력: config·device·skip_training·plot 속성을 가진 namespace

    명령행 인자를 해석한다.
    """
    parser = argparse.ArgumentParser(description="MNIST-O 실험을 실행합니다.")

    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    stage_group = parser.add_mutually_exclusive_group()
    stage_group.add_argument(
        "--skip-training",
        action="store_true",
        help="기존 checkpoint를 재사용하고 학습을 건너뜁니다. (--plot 없이 사용)",
    )
    stage_group.add_argument(
        "--plot",
        action="store_true",
        help="학습·평가 없이 기존 checkpoint로 시각화만 수행합니다.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
