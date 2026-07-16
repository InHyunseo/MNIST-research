"""실험 전체 단계를 제공하는 한국어 CLI (`mnist-overlap`, `python -m mnist_overlap`)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import (
    DEFAULT_CONFIG_PATH,
    SUPPORTED_MODEL_NAMES,
    create_output_directories,
    load_config,
)
from .data import prepare_data, validate_saved_data
from .evaluation import analyze_saved_results, evaluate_models
from .reporting import create_report
from .training import train_models

DEVICE_CHOICES = ("cpu", "cuda")


def run_experiment(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    device_name: str = "cpu",
    overwrite: bool = False,
) -> dict[str, object]:
    """데이터 준비·검증 후 config의 전체 모델×seed를 학습하고 평가 log를 생성한다."""
    config = load_config(config_path)
    create_output_directories()

    data_paths = prepare_data(config, overwrite=overwrite)
    validation_messages = validate_saved_data(config)
    training_results = train_models(
        config_path=config_path,
        device_name=device_name,
        overwrite=overwrite,
    )
    evaluation_paths = evaluate_models(
        config_path=config_path,
        device_name=device_name,
        overwrite=overwrite,
    )

    return {
        "data": data_paths,
        "validation": validation_messages,
        "training": training_results,
        "evaluation": evaluation_paths,
    }


def run_full_pipeline(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    device_name: str = "cpu",
    overwrite: bool = False,
) -> dict[str, object]:
    """전체 실험(`run_experiment`)과 최종 report 생성을 연속 실행한다."""
    pipeline_results = run_experiment(
        config_path=config_path,
        device_name=device_name,
        overwrite=overwrite,
    )
    config = load_config(config_path)
    pipeline_results["report"] = create_report(config)
    return pipeline_results


def build_parser() -> argparse.ArgumentParser:
    """최상위 command와 여덟 subcommand의 argument parser를 구성한다."""
    parser = argparse.ArgumentParser(
        prog="mnist-overlap",
        description="겹친 MNIST 숫자 인식과 spatial attention 비교 실험을 실행합니다.",
        add_help=False,
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        title="실행 단계",
    )

    prepare_parser = subparsers.add_parser(
        "prepare-data",
        help="MNIST 원본과 합성 manifest를 준비합니다.",
        add_help=False,
    )
    _add_config_argument(prepare_parser)
    _add_overwrite_argument(prepare_parser, "기존 manifest를 새로 생성합니다.")

    validate_parser = subparsers.add_parser(
        "validate-data",
        help="Source 분리와 manifest 무결성을 검사합니다.",
        add_help=False,
    )
    _add_config_argument(validate_parser)

    train_parser = subparsers.add_parser(
        "train",
        help="선택한 모델과 seed를 학습합니다.",
        add_help=False,
    )
    _add_config_argument(train_parser)
    _add_model_argument(train_parser)
    _add_seed_argument(train_parser)
    _add_device_argument(train_parser)
    _add_overwrite_argument(train_parser, "기존 checkpoint를 다시 학습합니다.")

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Checkpoint의 test 성능과 attention 지표를 계산합니다.",
        add_help=False,
    )
    _add_config_argument(evaluate_parser)
    _add_model_argument(evaluate_parser)
    _add_seed_argument(evaluate_parser)
    _add_device_argument(evaluate_parser)
    _add_overwrite_argument(
        evaluate_parser,
        "선택한 prediction과 attention cache를 다시 계산합니다.",
    )

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="저장된 cache와 학습 log에서 전체 통계를 다시 계산합니다.",
        add_help=False,
    )
    _add_config_argument(analyze_parser)

    report_parser = subparsers.add_parser(
        "report",
        help="평가 log에서 표, 그림, Markdown 요약을 생성합니다.",
        add_help=False,
    )
    _add_config_argument(report_parser)

    experiment_parser = subparsers.add_parser(
        "experiment",
        help="전체 모델과 seed를 학습하고 평가 log를 생성합니다.",
        add_help=False,
    )
    _add_config_argument(experiment_parser)
    _add_device_argument(experiment_parser)
    _add_overwrite_argument(
        experiment_parser,
        "Manifest와 전체 학습·평가 artifact를 새로 생성합니다.",
    )

    run_all_parser = subparsers.add_parser(
        "run-all",
        help="전체 실험과 report 생성을 연속 실행합니다.",
        add_help=False,
    )
    _add_config_argument(run_all_parser)
    _add_device_argument(run_all_parser)
    _add_overwrite_argument(
        run_all_parser,
        "Manifest와 전체 학습·평가 artifact를 새로 생성합니다.",
    )

    command_parsers = (
        parser,
        prepare_parser,
        validate_parser,
        train_parser,
        evaluate_parser,
        analyze_parser,
        report_parser,
        experiment_parser,
        run_all_parser,
    )
    for command_parser in command_parsers:
        _add_help_argument(command_parser)

    return parser


def main(arguments: Sequence[str] | None = None) -> None:
    """CLI argument를 해석해 대응하는 실행 함수를 호출한다."""
    parser = build_parser()
    parsed = parser.parse_args(arguments)
    config_path = parsed.config

    if parsed.command == "prepare-data":
        config = load_config(config_path)
        create_output_directories()
        paths = prepare_data(config, overwrite=parsed.overwrite)
        _print_paths(paths)
        return

    if parsed.command == "validate-data":
        config = load_config(config_path)
        for message in validate_saved_data(config):
            print(f"통과: {message}")
        return

    if parsed.command == "train":
        results = train_models(
            config_path=config_path,
            model_name=parsed.model,
            seed=parsed.seed,
            device_name=parsed.device,
            overwrite=parsed.overwrite,
        )
        for result in results:
            print(
                f"학습 완료: best_epoch={result.best_epoch}, "
                f"epochs_run={result.epochs_run}, "
                f"validation_exact={result.best_validation_exact_match:.4f}, "
                f"checkpoint={result.checkpoint_path}"
            )
        return

    if parsed.command == "evaluate":
        paths = evaluate_models(
            config_path=config_path,
            model_name=parsed.model,
            seed=parsed.seed,
            device_name=parsed.device,
            overwrite=parsed.overwrite,
        )
        _print_paths(paths)
        return

    if parsed.command == "analyze":
        _print_paths(analyze_saved_results(config_path=config_path))
        return

    if parsed.command == "report":
        config = load_config(config_path)
        create_output_directories()
        _print_paths(create_report(config))
        return

    if parsed.command == "experiment":
        run_experiment(
            config_path=config_path,
            device_name=parsed.device,
            overwrite=parsed.overwrite,
        )
        print("전체 모델과 seed의 학습 및 평가를 완료했습니다.")
        return

    if parsed.command == "run-all":
        run_full_pipeline(
            config_path=config_path,
            device_name=parsed.device,
            overwrite=parsed.overwrite,
        )
        print("전체 pipeline을 완료했습니다.")
        return

    parser.error(f"지원하지 않는 command입니다: {parsed.command}")


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    """공통 `--config` option을 추가한다."""
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"실험 YAML 경로입니다. 기본값: {DEFAULT_CONFIG_PATH}",
    )


def _add_help_argument(parser: argparse.ArgumentParser) -> None:
    """한국어 설명을 사용하는 `-h/--help` option을 추가한다."""
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="도움말을 표시하고 종료합니다.",
    )


def _add_model_argument(parser: argparse.ArgumentParser) -> None:
    """선택적인 `--model` option을 추가한다."""
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODEL_NAMES,
        default=None,
        help="단일 모델만 실행합니다. 생략하면 config의 모든 모델을 실행합니다.",
    )


def _add_seed_argument(parser: argparse.ArgumentParser) -> None:
    """선택적인 `--seed` option을 추가한다."""
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="단일 seed만 실행합니다. 생략하면 config의 모든 seed를 실행합니다.",
    )


def _add_device_argument(parser: argparse.ArgumentParser) -> None:
    """기본값이 CPU인 `--device` option을 추가한다."""
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="cpu",
        help="모델 실행 device입니다. 기본값: cpu",
    )


def _add_overwrite_argument(
    parser: argparse.ArgumentParser,
    help_text: str,
) -> None:
    """Artifact 재생성을 제어하는 `--overwrite` flag를 추가한다."""
    parser.add_argument("--overwrite", action="store_true", help=help_text)


def _print_paths(paths: object) -> None:
    """실행 함수가 반환한 경로들을 줄 단위로 출력한다."""
    if isinstance(paths, dict):
        for name, path in paths.items():
            print(f"{name}: {path}")
        return

    for path in paths:
        print(f"생성: {path}")
