"""Multitask pilot·공동학습·baseline 비교 평가·시각화의 단독 진입점."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from ..baseline.inference import Predictor as BaselinePredictor
from ..config import create_output_directories
from ..data import ControlledOverlapMnistDataset, prepare_data
from ..runtime import select_device
from .config import (
    DEFAULT_MULTITASK_CONFIG_PATH,
    METRICS_JSON_PATH,
    MultitaskConfig,
    create_multitask_directories,
    load_multitask_config,
)
from .evaluation import ComparisonEvaluator, MultitaskPredictor
from .plot import ComparisonVisualizer
from .training import (
    load_selected_reconstruction_weight,
    run_pilot,
    train_all_seeds,
)


def main() -> None:
    """CLI 인자에 따라 multitask 전체 실행 또는 기존 결과 시각화를 수행한다."""
    sys.stdout.reconfigure(line_buffering=True)
    arguments = _parse_arguments()
    run(
        config_path=arguments.config,
        device_name=arguments.device,
        skip_training=arguments.skip_training,
        plot_only=arguments.plot,
    )


def run(
    config_path: str | Path = DEFAULT_MULTITASK_CONFIG_PATH,
    device_name: str = "cpu",
    skip_training: bool = False,
    plot_only: bool = False,
) -> None:
    """통합 진입점에서도 재사용할 수 있도록 multitask 파이프라인을 실행한다."""
    config = load_multitask_config(config_path)
    create_output_directories()
    create_multitask_directories()
    device = select_device(device_name)
    prepare_data(config.baseline)

    if plot_only:
        _visualize_only(config, device)
        return

    if skip_training:
        reconstruction_loss_weight = load_selected_reconstruction_weight(config)
        print("Pilot·학습 생략 — 기존 multitask checkpoint를 사용합니다.")
    else:
        print("\n[1/3] Reconstruction loss 가중치 pilot")
        reconstruction_loss_weight = run_pilot(config, device)
        print(f"\n[2/3] Multitask 학습 — lambda={reconstruction_loss_weight:g}")
        train_all_seeds(config, reconstruction_loss_weight, device)

    print("\n[3/3] Baseline paired 비교 평가")
    _evaluate(config, reconstruction_loss_weight, device)


def _evaluate(
    config: MultitaskConfig,
    reconstruction_loss_weight: float,
    device: torch.device,
) -> None:
    """같은 test dataset에서 baseline과 multitask 10-seed prediction을 비교한다."""
    test_dataset = ControlledOverlapMnistDataset(
        "test", include_source_images=True
    )
    training_seeds = list(config.baseline.training.seeds)
    baseline_predictions = BaselinePredictor(
        config.baseline, device
    ).collect_all_seeds(test_dataset, training_seeds)
    multitask_predictor = MultitaskPredictor(
        config, reconstruction_loss_weight, device
    )
    multitask_predictions = multitask_predictor.collect_all_seeds(
        test_dataset, training_seeds
    )
    evaluator = ComparisonEvaluator(config)
    metrics = evaluator.evaluate(
        baseline_predictions,
        multitask_predictions,
        training_seeds,
        reconstruction_loss_weight,
    )
    evaluator.report(metrics)
    evaluator.save(metrics)


def _visualize_only(config: MultitaskConfig, device: torch.device) -> None:
    """저장된 비교 수치와 final pilot-seed checkpoint로 그림 네 장을 생성한다."""
    if not METRICS_JSON_PATH.exists():
        raise FileNotFoundError(f"비교 수치가 없습니다: {METRICS_JSON_PATH}")
    metrics = json.loads(METRICS_JSON_PATH.read_text(encoding="utf-8"))
    reconstruction_loss_weight = load_selected_reconstruction_weight(config)
    predictor = MultitaskPredictor(config, reconstruction_loss_weight, device)
    test_dataset = ControlledOverlapMnistDataset(
        "test", include_source_images=True
    )
    _make_figures(config, metrics, test_dataset, predictor, device)


def _make_figures(
    config: MultitaskConfig,
    metrics: dict[str, Any],
    test_dataset: ControlledOverlapMnistDataset,
    predictor: MultitaskPredictor,
    device: torch.device,
) -> None:
    """Pilot seed의 final model과 비교 수치로 발표용 그림 네 장을 저장한다."""
    model = predictor.load_model(config.reconstruction.pilot_seed)
    ComparisonVisualizer().make_all(metrics, test_dataset, model, device)
    print(f"그림을 저장했습니다: {METRICS_JSON_PATH.parent / 'figures'}")


def _parse_arguments() -> argparse.Namespace:
    """Multitask config·device·학습 생략·시각화 전용 인자를 해석한다."""
    parser = argparse.ArgumentParser(
        description="MNIST-O 분류+복원 multitask 실험을 실행합니다."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_MULTITASK_CONFIG_PATH,
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    stage_group = parser.add_mutually_exclusive_group()
    stage_group.add_argument(
        "--skip-training",
        action="store_true",
        help="기존 pilot 선택과 final checkpoint를 사용하고 학습을 건너뜁니다.",
    )
    stage_group.add_argument(
        "--plot",
        action="store_true",
        help="학습·평가 없이 기존 비교 수치와 checkpoint로 그림만 생성합니다.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
