"""전체 모델 실험과 최종 report 생성 순서를 제공한다.

입력:
    Config 경로, 실행 device, artifact 덮어쓰기 여부

출력:
    Experiment 단계의 log 경로와 최종 report 경로

연결:
    CLI와 shell script가 data, training, evaluation, reporting을 순서대로 호출한다.
"""

from pathlib import Path

from .configuration import (
    DEFAULT_CONFIG_PATH,
    create_output_directories,
    load_config,
)
from .data import prepare_data, validate_saved_data
from .evaluation import evaluate_models
from .reporting import create_report
from .training import train_models


def run_experiment(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    device_name: str = "cpu",
    overwrite: bool = False,
) -> dict[str, object]:
    """Config의 전체 모델과 seed를 학습하고 평가 log를 생성한다.

    입력:
        Config 경로, 학습/evaluation device, 기존 artifact 덮어쓰기 여부

    처리:
        데이터 준비·검증 후 config의 모든 모델과 seed를 학습하고 평가한다.

    출력:
        Data, validation, training, evaluation 단계의 반환값 dictionary
    """
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
    """전체 실험과 report 생성을 연속 실행해 최종 결과를 만든다.

    입력:
        Config 경로, 학습/evaluation device, 기존 artifact 덮어쓰기 여부

    처리:
        `run_experiment`로 log를 만든 뒤 동일 config로 표와 그림을 생성한다.

    출력:
        실험 결과와 report 경로를 함께 담은 dictionary
    """
    pipeline_results = run_experiment(
        config_path=config_path,
        device_name=device_name,
        overwrite=overwrite,
    )
    config = load_config(config_path)
    pipeline_results["report"] = create_report(config)
    return pipeline_results
