"""프로젝트 경로 상수, YAML 실험 설정(dataclass 스키마), 설정 fingerprint를 관리한다."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


# -----------------------------------------------------------------------------
# 프로젝트 경로와 상수
# -----------------------------------------------------------------------------


def _resolve_project_root() -> Path:
    """`MNIST_OVERLAP_ROOT` 환경 변수, source layout, 현재 directory 순으로 root를 정한다."""
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
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
TRAINING_LOG_DIR = OUTPUT_DIR / "training"
PREDICTION_LOG_DIR = OUTPUT_DIR / "predictions"
ATTENTION_LOG_DIR = OUTPUT_DIR / "attention"
METRIC_LOG_DIR = OUTPUT_DIR / "metrics"
EXPERIMENT_METADATA_PATH = OUTPUT_DIR / "experiment_metadata.json"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
SUMMARY_PATH = PROJECT_ROOT / "results" / "summary.md"

SUPPORTED_MODEL_NAMES = ("lenet", "shared_attention", "class_attention")
OVERLAP_LEVELS = ("low", "middle", "high")

# MNIST 고정 계약: class 10개, train split 60,000장, 두 숫자는 pixel-wise maximum 합성.
CLASS_COUNT = 10
MNIST_TRAIN_TOTAL = 60_000
COMPOSITION = "maximum"


# -----------------------------------------------------------------------------
# 설정 스키마
# -----------------------------------------------------------------------------


def _require_positive(section_name: str, **values: int | float) -> None:
    """지정한 값이 모두 양수인지 검사한다."""
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{section_name}.{name}은 양수여야 합니다: {value!r}")


@dataclass(frozen=True)
class ProjectConfig:
    experiment_name: str
    data_seed: int
    training_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.experiment_name:
            raise ValueError("project.experiment_name은 비어 있을 수 없습니다.")
        if not self.training_seeds:
            raise ValueError("project.training_seeds는 비어 있을 수 없습니다.")
        if len(self.training_seeds) != len(set(self.training_seeds)):
            raise ValueError("project.training_seeds에는 중복 seed를 넣을 수 없습니다.")


@dataclass(frozen=True)
class DatasetConfig:
    canvas_size: int
    digit_size: int
    train_samples: int
    validation_pairs: int
    test_pairs: int
    source_train_samples: int
    stroke_threshold: float

    def __post_init__(self) -> None:
        _require_positive(
            "dataset",
            canvas_size=self.canvas_size,
            digit_size=self.digit_size,
            train_samples=self.train_samples,
            validation_pairs=self.validation_pairs,
            test_pairs=self.test_pairs,
            source_train_samples=self.source_train_samples,
        )
        if self.canvas_size < self.digit_size:
            raise ValueError("dataset.canvas_size는 digit_size보다 작을 수 없습니다.")
        if self.source_train_samples >= MNIST_TRAIN_TOTAL:
            raise ValueError(
                f"dataset.source_train_samples는 {MNIST_TRAIN_TOTAL} 미만이어야 합니다."
            )
        if not 0.0 <= self.stroke_threshold <= 1.0:
            raise ValueError("dataset.stroke_threshold는 [0, 1] 범위여야 합니다.")

    @property
    def source_validation_samples(self) -> int:
        """MNIST train split에서 source train을 제외한 나머지 표본 수."""
        return MNIST_TRAIN_TOTAL - self.source_train_samples


@dataclass(frozen=True)
class OverlapConfig:
    low: tuple[float, float]
    middle: tuple[float, float]
    high: tuple[float, float]
    directions: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        previous_upper_bound = -1.0
        for overlap_level in OVERLAP_LEVELS:
            lower_bound, upper_bound = self.bounds(overlap_level)
            if not 0.0 <= lower_bound < upper_bound <= 1.0:
                raise ValueError(f"overlap.{overlap_level} 범위가 올바르지 않습니다.")
            if lower_bound <= previous_upper_bound:
                raise ValueError("Overlap 구간은 순서대로 배치되고 서로 겹치지 않아야 합니다.")
            previous_upper_bound = upper_bound

        if not self.directions:
            raise ValueError("overlap.directions는 비어 있을 수 없습니다.")
        for direction in self.directions:
            valid = all(component in (-1, 0, 1) for component in direction)
            if not valid or direction == (0, 0):
                raise ValueError("Direction 성분은 -1, 0, 1이며 동시에 0일 수 없습니다.")
        if len(set(self.directions)) != len(self.directions):
            raise ValueError("overlap.directions에는 중복 방향을 넣을 수 없습니다.")

    def bounds(self, overlap_level: str) -> tuple[float, float]:
        """Overlap level 이름으로 (하한, 상한) 구간을 반환한다."""
        return getattr(self, overlap_level)


@dataclass(frozen=True)
class ModelConfig:
    model_names: tuple[str, ...]
    first_conv_channels: int
    second_conv_channels: int
    first_hidden_features: int
    second_hidden_features: int
    kernel_size: int
    pool_size: int

    def __post_init__(self) -> None:
        if not self.model_names or len(self.model_names) != len(set(self.model_names)):
            raise ValueError("model.model_names에는 중복되지 않은 이름이 필요합니다.")
        unsupported_names = set(self.model_names).difference(SUPPORTED_MODEL_NAMES)
        if unsupported_names:
            raise ValueError(f"지원하지 않는 모델 이름입니다: {sorted(unsupported_names)}")
        _require_positive(
            "model",
            first_conv_channels=self.first_conv_channels,
            second_conv_channels=self.second_conv_channels,
            first_hidden_features=self.first_hidden_features,
            second_hidden_features=self.second_hidden_features,
            kernel_size=self.kernel_size,
            pool_size=self.pool_size,
        )


@dataclass(frozen=True)
class TrainConfig:
    maximum_epochs: int
    batch_size: int
    learning_rate: float
    early_stopping_patience: int
    early_stopping_minimum_delta: float
    data_loader_workers: int

    def __post_init__(self) -> None:
        _require_positive(
            "train",
            maximum_epochs=self.maximum_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_minimum_delta=self.early_stopping_minimum_delta,
        )
        if self.data_loader_workers < 0:
            raise ValueError("train.data_loader_workers는 0 이상의 정수여야 합니다.")


@dataclass(frozen=True)
class IouThresholdRange:
    start: float
    stop: float
    step: float

    def __post_init__(self) -> None:
        if not (0.0 < self.start < self.stop < 1.0 and self.step > 0):
            raise ValueError("Attention IoU threshold는 0 < start < stop < 1을 만족해야 합니다.")


@dataclass(frozen=True)
class EvaluationConfig:
    batch_size: int
    bootstrap_iterations: int
    confidence_level: float
    attention_iou_thresholds: IouThresholdRange
    minimum_exclusive_pixels: int

    def __post_init__(self) -> None:
        _require_positive(
            "evaluation",
            batch_size=self.batch_size,
            bootstrap_iterations=self.bootstrap_iterations,
            minimum_exclusive_pixels=self.minimum_exclusive_pixels,
        )
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("evaluation.confidence_level은 0과 1 사이여야 합니다.")


@dataclass(frozen=True)
class ReportConfig:
    figure_dpi: int
    save_png: bool

    def __post_init__(self) -> None:
        _require_positive("report", figure_dpi=self.figure_dpi)


@dataclass(frozen=True)
class ExperimentConfig:
    project: ProjectConfig
    dataset: DatasetConfig
    overlap: OverlapConfig
    model: ModelConfig
    train: TrainConfig
    evaluation: EvaluationConfig
    report: ReportConfig


# -----------------------------------------------------------------------------
# 설정 로딩
# -----------------------------------------------------------------------------


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> ExperimentConfig:
    """YAML 파일을 읽어 검증된 `ExperimentConfig`로 변환한다."""
    path = Path(config_path)
    with path.open(encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file)
    return _build_config(raw)


def _build_config(raw: Any) -> ExperimentConfig:
    """YAML dictionary를 타입 강제와 함께 dataclass 스키마로 변환한다."""
    if not isinstance(raw, dict):
        raise ValueError("Config YAML의 최상위는 mapping이어야 합니다.")

    required_sections = ("project", "dataset", "overlap", "model", "train", "evaluation", "report")
    missing_sections = [name for name in required_sections if name not in raw]
    if missing_sections:
        raise ValueError(f"필수 config section이 없습니다: {missing_sections}")

    return ExperimentConfig(
        project=_build_section(raw, "project", _build_project),
        dataset=_build_section(raw, "dataset", _build_dataset),
        overlap=_build_section(raw, "overlap", _build_overlap),
        model=_build_section(raw, "model", _build_model),
        train=_build_section(raw, "train", _build_train),
        evaluation=_build_section(raw, "evaluation", _build_evaluation),
        report=_build_section(raw, "report", _build_report),
    )


def _build_section(raw: dict[str, Any], name: str, builder: Any) -> Any:
    """Section 하나를 변환하고 실패 시 section 이름을 포함한 ValueError를 낸다."""
    section = raw[name]
    if not isinstance(section, dict):
        raise ValueError(f"config {name} section은 mapping이어야 합니다.")
    try:
        return builder(section)
    except KeyError as error:
        raise ValueError(f"config {name} section에 {error.args[0]!r} 값이 없습니다.") from error
    except (TypeError, ValueError) as error:
        raise ValueError(f"config {name} section이 잘못되었습니다: {error}") from error


def _build_project(section: dict[str, Any]) -> ProjectConfig:
    return ProjectConfig(
        experiment_name=str(section["experiment_name"]),
        data_seed=int(section["data_seed"]),
        training_seeds=tuple(int(seed) for seed in section["training_seeds"]),
    )


def _build_dataset(section: dict[str, Any]) -> DatasetConfig:
    return DatasetConfig(
        canvas_size=int(section["canvas_size"]),
        digit_size=int(section["digit_size"]),
        train_samples=int(section["train_samples"]),
        validation_pairs=int(section["validation_pairs"]),
        test_pairs=int(section["test_pairs"]),
        source_train_samples=int(section["source_train_samples"]),
        stroke_threshold=float(section["stroke_threshold"]),
    )


def _build_overlap(section: dict[str, Any]) -> OverlapConfig:
    def bounds(level: str) -> tuple[float, float]:
        lower_bound, upper_bound = section[level]
        return (float(lower_bound), float(upper_bound))

    return OverlapConfig(
        low=bounds("low"),
        middle=bounds("middle"),
        high=bounds("high"),
        directions=tuple(
            (int(row), int(column)) for row, column in section["directions"]
        ),
    )


def _build_model(section: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        model_names=tuple(str(name) for name in section["model_names"]),
        first_conv_channels=int(section["first_conv_channels"]),
        second_conv_channels=int(section["second_conv_channels"]),
        first_hidden_features=int(section["first_hidden_features"]),
        second_hidden_features=int(section["second_hidden_features"]),
        kernel_size=int(section["kernel_size"]),
        pool_size=int(section["pool_size"]),
    )


def _build_train(section: dict[str, Any]) -> TrainConfig:
    return TrainConfig(
        maximum_epochs=int(section["maximum_epochs"]),
        batch_size=int(section["batch_size"]),
        learning_rate=float(section["learning_rate"]),
        early_stopping_patience=int(section["early_stopping_patience"]),
        early_stopping_minimum_delta=float(section["early_stopping_minimum_delta"]),
        data_loader_workers=int(section["data_loader_workers"]),
    )


def _build_evaluation(section: dict[str, Any]) -> EvaluationConfig:
    thresholds = section["attention_iou_thresholds"]
    return EvaluationConfig(
        batch_size=int(section["batch_size"]),
        bootstrap_iterations=int(section["bootstrap_iterations"]),
        confidence_level=float(section["confidence_level"]),
        attention_iou_thresholds=IouThresholdRange(
            start=float(thresholds["start"]),
            stop=float(thresholds["stop"]),
            step=float(thresholds["step"]),
        ),
        minimum_exclusive_pixels=int(section["minimum_exclusive_pixels"]),
    )


def _build_report(section: dict[str, Any]) -> ReportConfig:
    return ReportConfig(
        figure_dpi=int(section["figure_dpi"]),
        save_png=bool(section["save_png"]),
    )


# -----------------------------------------------------------------------------
# Runtime directory와 fingerprint
# -----------------------------------------------------------------------------


def create_output_directories() -> None:
    """프로젝트가 생성하는 runtime directory를 준비한다."""
    output_directories = (
        RAW_DATA_DIR,
        MANIFEST_DIR,
        CHECKPOINT_DIR,
        TRAINING_LOG_DIR,
        PREDICTION_LOG_DIR,
        ATTENTION_LOG_DIR,
        METRIC_LOG_DIR,
        FIGURE_DIR,
        TABLE_DIR,
    )
    for directory in output_directories:
        directory.mkdir(parents=True, exist_ok=True)


def _contract_fingerprint(contract: dict[str, Any]) -> str:
    """계약 dictionary를 정렬된 JSON으로 직렬화해 SHA-256을 계산한다."""
    serialized = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def config_fingerprint(config: ExperimentConfig) -> str:
    """Checkpoint 호환성을 판단하는 학습 계약 fingerprint를 계산한다."""
    return _contract_fingerprint(
        {
            "data_seed": config.project.data_seed,
            "dataset": asdict(config.dataset),
            "overlap": asdict(config.overlap),
            "model": asdict(config.model),
            "train": asdict(config.train),
        }
    )


def data_config_fingerprint(config: ExperimentConfig) -> str:
    """Manifest 호환성을 판단하는 데이터 계약 fingerprint를 계산한다."""
    return _contract_fingerprint(
        {
            "data_seed": config.project.data_seed,
            "dataset": asdict(config.dataset),
            "overlap": asdict(config.overlap),
            "class_count": CLASS_COUNT,
        }
    )


def evaluation_fingerprint(config: ExperimentConfig) -> str:
    """Prediction·attention cache 호환성을 판단하는 평가 계약 fingerprint를 계산한다."""
    return _contract_fingerprint(
        {
            "training_fingerprint": config_fingerprint(config),
            "evaluation": asdict(config.evaluation),
        }
    )


# -----------------------------------------------------------------------------
# 실험 provenance
# -----------------------------------------------------------------------------


def update_experiment_metadata(config: ExperimentConfig, device_name: str) -> None:
    """실험 provenance와 완료 run 목록을 `outputs/experiment_metadata.json`에 원자적으로 갱신한다."""
    create_output_directories()
    fingerprint = config_fingerprint(config)
    metadata = {
        "experiment_name": config.project.experiment_name,
        "config_fingerprint": fingerprint,
        "evaluation_fingerprint": evaluation_fingerprint(config),
        "config": asdict(config),
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "pytorch_version": _pytorch_version(),
        "device": device_name,
        "completed_runs": _discover_completed_runs(config, fingerprint),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary_path = EXPERIMENT_METADATA_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(EXPERIMENT_METADATA_PATH)


def _discover_completed_runs(
    config: ExperimentConfig,
    fingerprint: str,
) -> list[dict[str, Any]]:
    """Checkpoint를 읽어 현재 fingerprint로 완료된 model·seed 목록을 구성한다."""
    try:
        import torch
    except ImportError:
        return []

    completed_runs = []
    for seed in config.project.training_seeds:
        for model_name in config.model.model_names:
            checkpoint_path = CHECKPOINT_DIR / f"{model_name}_seed_{seed}.pt"
            if not checkpoint_path.exists():
                continue

            try:
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )
            except (OSError, RuntimeError, ValueError):
                continue

            checkpoint_is_complete = (
                checkpoint.get("config_fingerprint") == fingerprint
                and checkpoint.get("training_complete") is True
            )
            if checkpoint_is_complete:
                completed_runs.append({"model": model_name, "seed": seed})

    return completed_runs


def _git_commit() -> str | None:
    """현재 workspace의 Git commit hash를 조회한다."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _pytorch_version() -> str | None:
    """설치된 PyTorch 버전을 반환한다."""
    try:
        torch_module = sys.modules.get("torch")
        if torch_module is None:
            import torch as torch_module
        return str(torch_module.__version__)
    except ImportError:
        return None
