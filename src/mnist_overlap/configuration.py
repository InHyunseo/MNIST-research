"""MNIST Overlap Attention의 경로와 YAML 실험 설정을 관리한다.

입력:
    `configs/mnist_overlap.yaml` 또는 사용자가 지정한 YAML 경로

출력:
    검증된 config dictionary, runtime 경로, 설정 fingerprint

연결:
    모든 실행 함수가 실행 전에 호출하며 data와 training package가 경로 상수를 사용한다.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


# -----------------------------------------------------------------------------
# 프로젝트 경로와 상수
# -----------------------------------------------------------------------------


def _resolve_project_root() -> Path:
    """Runtime artifact와 config를 찾을 workspace root를 결정한다.

    입력:
        선택적인 `MNIST_OVERLAP_ROOT` 환경 변수, editable source 경로, 현재 directory

    처리:
        환경 변수, source-layout root, 현재 directory 순서로 유효한 workspace를 찾는다.

    출력:
        절대 경로로 정규화된 workspace root
    """
    environment_root = os.environ.get("MNIST_OVERLAP_ROOT")
    if environment_root:
        return Path(environment_root).expanduser().resolve()

    source_project_root = Path(__file__).resolve().parents[2]
    if (source_project_root / "configs" / "mnist_overlap.yaml").exists():
        return source_project_root

    return Path.cwd().resolve()


PROJECT_ROOT = _resolve_project_root()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "mnist_overlap.yaml"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MANIFEST_DIR = DATA_DIR / "manifests"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
TRAINING_LOG_DIR = PROJECT_ROOT / "logs" / "training"
PREDICTION_LOG_DIR = PROJECT_ROOT / "logs" / "predictions"
METRIC_LOG_DIR = PROJECT_ROOT / "logs" / "metrics"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
SUMMARY_PATH = PROJECT_ROOT / "results" / "summary.md"

SUPPORTED_MODEL_NAMES = {"lenet", "shared_attention", "class_attention"}
OVERLAP_LEVELS = ("low", "middle", "high")


# -----------------------------------------------------------------------------
# Config 공개 API
# -----------------------------------------------------------------------------


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """YAML 파일을 읽고 전체 실험 계약을 검증한다.

    입력:
        YAML 설정 파일 경로

    처리:
        UTF-8 YAML을 dictionary로 읽고 section별 검증 함수를 호출한다.

    출력:
        검증을 통과한 config dictionary
    """
    path = Path(config_path)

    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """모든 실행 단계가 공유하는 config 제약을 검사한다.

    입력:
        YAML에서 읽은 config dictionary

    처리:
        필수 section, 값 범위, 모델 이름, overlap 구간을 검사한다.

    출력:
        반환값은 없으며 잘못된 설정이면 한국어 ValueError를 발생시킨다.
    """
    required_sections = {
        "project",
        "dataset",
        "overlap",
        "model",
        "train",
        "evaluation",
        "report",
    }
    missing_sections = required_sections.difference(config)

    if missing_sections:
        raise ValueError(f"필수 config section이 없습니다: {sorted(missing_sections)}")

    _validate_project_config(config["project"])
    _validate_dataset_config(config["dataset"])
    _validate_overlap_config(config["overlap"])
    _validate_model_config(config["model"])
    _validate_training_config(config["train"])
    _validate_evaluation_config(config["evaluation"])
    _validate_report_config(config["report"])


def create_output_directories() -> None:
    """프로젝트에서 생성하는 runtime directory를 준비한다.

    입력:
        module에 정의된 프로젝트 경로 상수

    처리:
        data, checkpoint, log, result directory를 부모 경로와 함께 생성한다.

    출력:
        반환값은 없으며 필요한 directory가 filesystem에 생성된다.
    """
    output_directories = (
        RAW_DATA_DIR,
        MANIFEST_DIR,
        CHECKPOINT_DIR,
        TRAINING_LOG_DIR,
        PREDICTION_LOG_DIR,
        METRIC_LOG_DIR,
        FIGURE_DIR,
        TABLE_DIR,
    )

    for directory in output_directories:
        directory.mkdir(parents=True, exist_ok=True)


def config_fingerprint(config: dict[str, Any]) -> str:
    """Checkpoint 호환성을 판단할 학습 설정 fingerprint를 계산한다.

    입력:
        data, model, training section이 포함된 전체 config

    처리:
        학습 결과에 영향을 주는 section만 정렬된 JSON으로 직렬화하고 SHA-256을 계산한다.

    출력:
        64자리 hexadecimal fingerprint 문자열
    """
    training_contract = {
        "data_seed": config["project"]["data_seed"],
        "dataset": config["dataset"],
        "overlap": config["overlap"],
        "model": config["model"],
        "train": config["train"],
    }
    serialized_contract = json.dumps(
        training_contract,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized_contract.encode("utf-8")).hexdigest()


def data_config_fingerprint(config: dict[str, Any]) -> str:
    """Manifest 호환성을 판단할 데이터 설정 fingerprint를 계산한다.

    입력:
        dataset, overlap, class count, data seed가 포함된 전체 config

    처리:
        데이터 생성에 영향을 주는 값만 직렬화하고 SHA-256을 계산한다.

    출력:
        64자리 hexadecimal fingerprint 문자열
    """
    data_contract = {
        "data_seed": config["project"]["data_seed"],
        "dataset": config["dataset"],
        "overlap": config["overlap"],
        "class_count": config["model"]["class_count"],
    }
    serialized_contract = json.dumps(
        data_contract,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized_contract.encode("utf-8")).hexdigest()


# -----------------------------------------------------------------------------
# Section별 검증
# -----------------------------------------------------------------------------


def _validate_project_config(project_config: dict[str, Any]) -> None:
    """프로젝트 이름과 난수 seed 설정을 검사한다.

    입력:
        config의 project section

    처리:
        experiment name, data seed, training seed 목록의 타입과 존재 여부를 검사한다.

    출력:
        반환값은 없으며 잘못된 값이면 ValueError를 발생시킨다.
    """
    if not project_config.get("experiment_name"):
        raise ValueError("project.experiment_name은 비어 있을 수 없습니다.")

    if not isinstance(project_config.get("data_seed"), int):
        raise ValueError("project.data_seed는 정수여야 합니다.")

    training_seeds = project_config.get("training_seeds")
    seeds_are_valid = training_seeds and all(
        isinstance(seed, int) for seed in training_seeds
    )

    if not seeds_are_valid:
        raise ValueError("project.training_seeds에는 정수 seed가 필요합니다.")


def _validate_dataset_config(dataset_config: dict[str, Any]) -> None:
    """Canvas, sample 수, 합성 방식과 mask threshold를 검사한다.

    입력:
        config의 dataset section

    처리:
        양의 정수, MNIST source 총수, threshold와 composition을 검사한다.

    출력:
        반환값은 없으며 데이터 계약을 벗어나면 ValueError를 발생시킨다.
    """
    positive_integer_names = (
        "canvas_size",
        "digit_size",
        "train_samples",
        "validation_pairs",
        "test_pairs",
        "source_train_samples",
        "source_validation_samples",
    )
    _require_positive_values(
        "dataset",
        dataset_config,
        positive_integer_names,
        int,
    )

    if dataset_config["canvas_size"] < dataset_config["digit_size"]:
        raise ValueError("dataset.canvas_size는 digit_size보다 작을 수 없습니다.")

    source_sample_count = (
        dataset_config["source_train_samples"]
        + dataset_config["source_validation_samples"]
    )
    if source_sample_count != 60_000:
        raise ValueError("MNIST train source 분할의 합은 60,000이어야 합니다.")

    stroke_threshold = dataset_config.get("stroke_threshold")
    threshold_is_valid = (
        isinstance(stroke_threshold, (int, float))
        and 0.0 <= stroke_threshold <= 1.0
    )
    if not threshold_is_valid:
        raise ValueError("dataset.stroke_threshold는 [0, 1] 범위여야 합니다.")

    if dataset_config.get("composition") != "maximum":
        raise ValueError("dataset.composition은 'maximum'이어야 합니다.")


def _validate_overlap_config(overlap_config: dict[str, Any]) -> None:
    """Overlap 구간의 순서와 정수 이동 방향을 검사한다.

    입력:
        config의 overlap section

    처리:
        세 구간이 겹치지 않는지 확인하고 direction 중복 및 범위를 검사한다.

    출력:
        반환값은 없으며 잘못된 구간이나 방향이면 ValueError를 발생시킨다.
    """
    previous_upper_bound = -1.0

    for overlap_level in OVERLAP_LEVELS:
        bounds = overlap_config.get(overlap_level)
        bounds_have_two_values = isinstance(bounds, list) and len(bounds) == 2

        if not bounds_have_two_values:
            raise ValueError(f"overlap.{overlap_level}에는 [하한, 상한]이 필요합니다.")

        lower_bound, upper_bound = bounds
        bounds_are_valid = 0.0 <= lower_bound < upper_bound <= 1.0

        if not bounds_are_valid:
            raise ValueError(f"overlap.{overlap_level} 범위가 올바르지 않습니다.")

        if lower_bound <= previous_upper_bound:
            raise ValueError("Overlap 구간은 순서대로 배치되고 서로 겹치지 않아야 합니다.")

        previous_upper_bound = upper_bound

    directions = overlap_config.get("directions")
    if not isinstance(directions, list) or not directions:
        raise ValueError("overlap.directions는 비어 있을 수 없습니다.")

    normalized_directions = set()

    for direction in directions:
        has_two_components = isinstance(direction, list) and len(direction) == 2
        if not has_two_components:
            raise ValueError("각 overlap direction에는 정수 성분 두 개가 필요합니다.")

        components_are_valid = all(
            component in (-1, 0, 1) for component in direction
        )
        if not components_are_valid or direction == [0, 0]:
            raise ValueError("Direction 성분은 -1, 0, 1이며 동시에 0일 수 없습니다.")

        normalized_directions.add(tuple(direction))

    if len(normalized_directions) != len(directions):
        raise ValueError("overlap.directions에는 중복 방향을 넣을 수 없습니다.")


def _validate_model_config(model_config: dict[str, Any]) -> None:
    """모델 이름과 layer 크기를 검사한다.

    입력:
        config의 model section

    처리:
        지원 모델, 중복 이름, 양의 layer 크기, MNIST class 수를 확인한다.

    출력:
        반환값은 없으며 모델 계약이 잘못되면 ValueError를 발생시킨다.
    """
    model_names = model_config.get("model_names")
    names_are_unique = model_names and len(model_names) == len(set(model_names))

    if not names_are_unique:
        raise ValueError("model.model_names에는 중복되지 않은 이름이 필요합니다.")

    unsupported_names = set(model_names).difference(SUPPORTED_MODEL_NAMES)
    if unsupported_names:
        raise ValueError(f"지원하지 않는 모델 이름입니다: {sorted(unsupported_names)}")

    positive_integer_names = (
        "first_conv_channels",
        "second_conv_channels",
        "first_hidden_features",
        "second_hidden_features",
        "kernel_size",
        "pool_size",
        "class_count",
    )
    _require_positive_values(
        "model",
        model_config,
        positive_integer_names,
        int,
    )

    if model_config["class_count"] != 10:
        raise ValueError("MNIST 실험의 model.class_count는 10이어야 합니다.")


def _validate_training_config(training_config: dict[str, Any]) -> None:
    """Optimizer와 early-stopping 관련 학습 값을 검사한다.

    입력:
        config의 train section

    처리:
        epoch, batch, learning rate, patience와 worker 수의 범위를 확인한다.

    출력:
        반환값은 없으며 학습 값이 잘못되면 ValueError를 발생시킨다.
    """
    positive_value_names = (
        "maximum_epochs",
        "batch_size",
        "learning_rate",
        "early_stopping_patience",
        "early_stopping_minimum_delta",
    )
    _require_positive_values("train", training_config, positive_value_names)

    workers = training_config.get("data_loader_workers")
    if not isinstance(workers, int) or workers < 0:
        raise ValueError("train.data_loader_workers는 0 이상의 정수여야 합니다.")


def _validate_evaluation_config(section: dict[str, Any]) -> None:
    """Evaluation section의 분석 파라미터를 검사한다.

    입력:
        Config의 evaluation section

    처리:
        Bootstrap, batch, attention IoU threshold 관련 범위를 확인한다.

    출력:
        반환값은 없으며 잘못된 분석 값이면 ValueError를 발생시킨다.
    """
    if "bootstrap_iterations" in section:
        _require_positive_values(
            "evaluation",
            section,
            ("bootstrap_iterations",),
            int,
        )

    if "batch_size" in section:
        _require_positive_values("evaluation", section, ("batch_size",), int)

    _require_positive_values(
        "evaluation",
        section,
        ("minimum_exclusive_pixels",),
        int,
    )
    threshold_config = section.get("attention_iou_thresholds", {})
    start = threshold_config.get("start")
    stop = threshold_config.get("stop")
    step = threshold_config.get("step")
    thresholds_are_numeric = all(
        isinstance(value, (int, float)) for value in (start, stop, step)
    )

    if not thresholds_are_numeric:
        raise ValueError("Attention IoU threshold에는 숫자 start, stop, step이 필요합니다.")

    threshold_range_is_valid = 0.0 < start < stop < 1.0 and step > 0
    if not threshold_range_is_valid:
        raise ValueError("Attention IoU threshold는 0 < start < stop < 1을 만족해야 합니다.")


def _validate_report_config(report_config: dict[str, Any]) -> None:
    """Report 해상도와 PNG 저장 option을 검사한다.

    입력:
        config의 report section

    처리:
        DPI가 양의 정수인지, PNG 저장 option이 Boolean인지 확인한다.

    출력:
        반환값은 없으며 잘못된 값이면 ValueError를 발생시킨다.
    """
    _require_positive_values("report", report_config, ("figure_dpi",), int)

    if not isinstance(report_config.get("save_png"), bool):
        raise ValueError("report.save_png는 Boolean이어야 합니다.")


def _require_positive_values(
    section_name: str,
    section: dict[str, Any],
    names: tuple[str, ...],
    required_type: type | tuple[type, ...] = (int, float),
) -> None:
    """지정한 config 값이 요구 타입의 양수인지 검사한다.

    입력:
        section 이름, section dictionary, key 목록, 허용 타입

    처리:
        각 값을 순회하며 Boolean을 제외한 요구 타입의 양수인지 확인한다.

    출력:
        반환값은 없으며 조건을 위반한 첫 값에서 ValueError를 발생시킨다.
    """
    for name in names:
        value = section.get(name)
        value_is_valid = (
            isinstance(value, required_type)
            and not isinstance(value, bool)
            and value > 0
        )

        if not value_is_valid:
            raise ValueError(f"{section_name}.{name}은 양수여야 합니다: {value!r}")
