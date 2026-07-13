"""MNIST Overlap Attention의 모든 실행 단계를 하나의 한국어 CLI로 제공한다.

입력:
    `prepare-data`, `validate-data`, `train`, `evaluate`, `report`, `experiment`, `run-all`

출력:
    실행 결과, 생성 file 경로, 실패 원인

연결:
    `python -m mnist_overlap`과 shell script를 기능별 package에 연결한다.
"""

import argparse
from pathlib import Path
from typing import Sequence

from .configuration import (
    DEFAULT_CONFIG_PATH,
    create_output_directories,
    load_config,
)
from .data import prepare_data, validate_saved_data
from .evaluation import evaluate_models
from .pipeline import run_experiment, run_full_pipeline
from .reporting import create_report
from .training import train_models

MODEL_CHOICES = ("lenet", "shared_attention", "class_attention")
DEVICE_CHOICES = ("cpu", "cuda")


class KoreanArgumentParser(argparse.ArgumentParser):
    """Argparse의 기본 heading과 오류 접두어를 한국어로 표시한다.

    입력:
        일반 ArgumentParser와 동일한 parser 설정 및 command argument

    처리:
        기본 help text의 usage/options 표기와 오류 출력 형식을 변환한다.

    출력:
        한국어 heading과 도움말을 사용하는 ArgumentParser
    """

    def format_help(self) -> str:
        """기본 help text의 고정 영어 heading을 한국어로 바꾼다.

        입력:
            현재 parser에 등록된 positional 및 optional action

        처리:
            부모 class의 help 문자열에서 usage와 options heading을 치환한다.

        출력:
            사용자에게 표시할 한국어 help 문자열
        """
        help_text = super().format_help()
        help_text = help_text.replace("usage:", "사용법:")
        return help_text.replace("options:", "선택 항목:")

    def error(self, message: str) -> None:
        """Argument parsing 오류를 한국어 접두어와 함께 출력한다.

        입력:
            Argparse가 생성한 오류 상세 문자열

        처리:
            Usage를 stderr에 표시하고 종료 코드 2로 parser를 종료한다.

        출력:
            반환값은 없으며 `SystemExit(2)`가 발생한다.
        """
        self.print_usage()
        self.exit(2, f"{self.prog}: 입력 오류: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    """최상위 command와 일곱 subcommand의 argument parser를 구성한다.

    입력:
        고정 CLI command와 기본 config 경로

    처리:
        공통 option과 단계별 option을 한국어 help 문자열로 등록한다.

    출력:
        완성된 ArgumentParser instance
    """
    parser = KoreanArgumentParser(
        prog="python -m mnist_overlap",
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
        "Manifest와 checkpoint를 모두 새로 생성합니다.",
    )

    run_all_parser = subparsers.add_parser(
        "run-all",
        help="전체 실험과 report 생성을 연속 실행합니다.",
        add_help=False,
    )
    _add_config_argument(run_all_parser)
    _add_device_argument(run_all_parser)
    _add_overwrite_argument(run_all_parser, "Manifest와 checkpoint를 모두 새로 생성합니다.")

    command_parsers = (
        parser,
        prepare_parser,
        validate_parser,
        train_parser,
        evaluate_parser,
        report_parser,
        experiment_parser,
        run_all_parser,
    )
    for command_parser in command_parsers:
        _add_help_argument(command_parser)

    return parser


def main(arguments: Sequence[str] | None = None) -> None:
    """CLI argument를 해석해 대응하는 실행 함수를 호출한다.

    입력:
        선택적인 argument 문자열 sequence, 없으면 `sys.argv`

    처리:
        Subcommand별 option을 기능 package의 명시적 parameter로 변환한다.

    출력:
        반환값은 없으며 생성 경로와 주요 결과를 표준 출력에 표시한다.
    """
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
                f"학습 완료: epoch={result.best_epoch}, "
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
        )
        _print_paths(paths)
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
    """Subcommand에 공통 config 경로 option을 추가한다.

    입력:
        Option을 추가할 subcommand parser

    처리:
        기본 YAML 경로를 사용하는 `--config` argument를 등록한다.

    출력:
        반환값은 없으며 parser가 제자리에서 변경된다.
    """
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"실험 YAML 경로입니다. 기본값: {DEFAULT_CONFIG_PATH}",
    )


def _add_help_argument(parser: argparse.ArgumentParser) -> None:
    """Parser에 한국어 설명을 사용하는 help option을 추가한다.

    입력:
        `add_help=False`로 생성한 parser

    처리:
        `-h`와 `--help`를 argparse help action에 연결한다.

    출력:
        반환값은 없으며 parser가 제자리에서 변경된다.
    """
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="도움말을 표시하고 종료합니다.",
    )


def _add_model_argument(parser: argparse.ArgumentParser) -> None:
    """Train/evaluate parser에 선택적인 model option을 추가한다.

    입력:
        Option을 추가할 subcommand parser

    처리:
        세 지원 모델 중 하나를 선택하는 `--model` argument를 등록한다.

    출력:
        반환값은 없으며 parser가 제자리에서 변경된다.
    """
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default=None,
        help="단일 모델만 실행합니다. 생략하면 config의 모든 모델을 실행합니다.",
    )


def _add_seed_argument(parser: argparse.ArgumentParser) -> None:
    """Train/evaluate parser에 선택적인 seed option을 추가한다.

    입력:
        Option을 추가할 subcommand parser

    처리:
        정수 하나를 받는 `--seed` argument를 등록한다.

    출력:
        반환값은 없으며 parser가 제자리에서 변경된다.
    """
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="단일 seed만 실행합니다. 생략하면 config의 모든 seed를 실행합니다.",
    )


def _add_device_argument(parser: argparse.ArgumentParser) -> None:
    """실행 parser에 CPU/CUDA device option을 추가한다.

    입력:
        Option을 추가할 subcommand parser

    처리:
        기본값이 CPU인 `--device` argument를 등록한다.

    출력:
        반환값은 없으며 parser가 제자리에서 변경된다.
    """
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
    """Artifact 재생성을 제어하는 overwrite flag를 추가한다.

    입력:
        Option을 추가할 parser와 사용자에게 표시할 한국어 설명

    처리:
        Boolean `--overwrite` flag를 등록한다.

    출력:
        반환값은 없으며 parser가 제자리에서 변경된다.
    """
    parser.add_argument("--overwrite", action="store_true", help=help_text)


def _print_paths(paths: object) -> None:
    """실행 함수 반환 경로를 사용자가 읽기 좋은 줄 단위로 출력한다.

    입력:
        Path 목록 또는 이름과 Path의 dictionary

    처리:
        Dictionary와 iterable을 구분해 각 항목을 한 줄로 출력한다.

    출력:
        반환값은 없으며 표준 출력에 생성 경로가 표시된다.
    """
    if isinstance(paths, dict):
        for name, path in paths.items():
            print(f"{name}: {path}")
        return

    for path in paths:
        print(f"생성: {path}")
