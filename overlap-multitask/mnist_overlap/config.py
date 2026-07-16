"""최소 실험 설정을 읽고 runtime 경로와 재현성 지문을 관리한다."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "mnist_overlap.yaml"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MANIFEST_DIR = DATA_DIR / "manifests"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baseline"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
TRAINING_LOG_DIR = OUTPUT_DIR / "training"
RESULTS_DIR = PROJECT_ROOT / "results" / "baseline"
FIGURE_DIR = RESULTS_DIR / "figures"

CLASS_COUNT = 10
OVERLAP_LEVELS = ("low", "middle", "high")
COMPOSITION_MODE = "mean"


def _require_positive(section_name: str, **values: int | float) -> None:
    """지정한 설정값이 모두 양수인지 검사한다."""
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{section_name}.{name}은 양수여야 합니다: {value!r}")


@dataclass(frozen=True)
class DataConfig:
    """데이터 난수 seed와 split별 합성 표본 수."""

    seed: int
    train_samples: int
    validation_pairs: int
    test_pairs: int

    def __post_init__(self) -> None:
        _require_positive(
            "data",
            train_samples=self.train_samples,
            validation_pairs=self.validation_pairs,
            test_pairs=self.test_pairs,
        )


@dataclass(frozen=True)
class OverlapConfig:
    """Low·Middle·High bounding-box overlap 구간."""

    low: tuple[float, float]
    middle: tuple[float, float]
    high: tuple[float, float]

    def __post_init__(self) -> None:
        previous_upper_bound = -1.0
        for overlap_level in OVERLAP_LEVELS:
            lower_bound, upper_bound = self.bounds(overlap_level)
            if not 0.0 <= lower_bound < upper_bound <= 1.0:
                raise ValueError(f"overlap.{overlap_level} 범위가 올바르지 않습니다.")
            if lower_bound <= previous_upper_bound:
                raise ValueError("Overlap 구간은 순서대로 배치되고 서로 겹치지 않아야 합니다.")
            previous_upper_bound = upper_bound

    def bounds(self, overlap_level: str) -> tuple[float, float]:
        """Overlap level 이름에 대응하는 `(하한, 상한)`을 반환한다."""
        return getattr(self, overlap_level)


@dataclass(frozen=True)
class TrainingConfig:
    """반복 학습에 실제로 사용하는 hyperparameter."""

    seeds: tuple[int, ...]
    maximum_epochs: int
    batch_size: int
    learning_rate: float
    early_stopping_patience: int
    early_stopping_minimum_delta: float

    def __post_init__(self) -> None:
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("training.seeds에는 중복되지 않은 seed가 필요합니다.")
        _require_positive(
            "training",
            maximum_epochs=self.maximum_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_minimum_delta=self.early_stopping_minimum_delta,
        )


@dataclass(frozen=True)
class EvaluationConfig:
    """Bootstrap 반복 횟수."""

    bootstrap_iterations: int

    def __post_init__(self) -> None:
        _require_positive("evaluation", bootstrap_iterations=self.bootstrap_iterations)


@dataclass(frozen=True)
class ExperimentConfig:
    """데이터·overlap·학습·평가 설정 묶음."""

    data: DataConfig
    overlap: OverlapConfig
    training: TrainingConfig
    evaluation: EvaluationConfig


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> ExperimentConfig:
    """YAML을 읽어 필수값과 타입을 검증한 `ExperimentConfig`를 반환한다."""
    with Path(config_path).open(encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file)
    if not isinstance(raw, dict):
        raise ValueError("Config YAML의 최상위는 mapping이어야 합니다.")
    expected_sections = {"data", "overlap", "training", "evaluation"}
    if set(raw) != expected_sections:
        raise ValueError(
            "Config section이 올바르지 않습니다. "
            f"누락={sorted(expected_sections.difference(raw))}, "
            f"추가={sorted(set(raw).difference(expected_sections))}"
        )

    data = _section(raw, "data", {"seed", "train_samples", "validation_pairs", "test_pairs"})
    overlap = _section(raw, "overlap", set(OVERLAP_LEVELS))
    training = _section(
        raw,
        "training",
        {
            "seeds",
            "maximum_epochs",
            "batch_size",
            "learning_rate",
            "early_stopping_patience",
            "early_stopping_minimum_delta",
        },
    )
    evaluation = _section(raw, "evaluation", {"bootstrap_iterations"})

    try:
        return ExperimentConfig(
            data=DataConfig(
                seed=int(data["seed"]),
                train_samples=int(data["train_samples"]),
                validation_pairs=int(data["validation_pairs"]),
                test_pairs=int(data["test_pairs"]),
            ),
            overlap=OverlapConfig(**{
                level: _overlap_bounds(overlap[level]) for level in OVERLAP_LEVELS
            }),
            training=TrainingConfig(
                seeds=tuple(int(seed) for seed in training["seeds"]),
                maximum_epochs=int(training["maximum_epochs"]),
                batch_size=int(training["batch_size"]),
                learning_rate=float(training["learning_rate"]),
                early_stopping_patience=int(training["early_stopping_patience"]),
                early_stopping_minimum_delta=float(
                    training["early_stopping_minimum_delta"]
                ),
            ),
            evaluation=EvaluationConfig(
                bootstrap_iterations=int(evaluation["bootstrap_iterations"])
            ),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"Config 값이 잘못되었습니다: {error}") from error


def _section(raw: dict[str, Any], name: str, expected_fields: set[str]) -> dict[str, Any]:
    """설정 section이 mapping이며 필드가 정확히 일치하는지 확인한다."""
    section = raw.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"config {name} section은 mapping이어야 합니다.")
    missing_fields = expected_fields.difference(section)
    extra_fields = set(section).difference(expected_fields)
    if missing_fields or extra_fields:
        raise ValueError(
            f"config {name} 필드가 올바르지 않습니다. "
            f"누락={sorted(missing_fields)}, 추가={sorted(extra_fields)}"
        )
    return section


def _overlap_bounds(values: Any) -> tuple[float, float]:
    """YAML의 길이 2 overlap 범위를 float tuple로 변환한다."""
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError("Overlap 범위에는 하한과 상한 두 값이 필요합니다.")
    return float(values[0]), float(values[1])


def create_output_directories() -> None:
    """실행 중 사용하는 데이터·checkpoint·결과 directory를 준비한다."""
    for directory in (
        RAW_DATA_DIR,
        MANIFEST_DIR,
        CHECKPOINT_DIR,
        TRAINING_LOG_DIR,
        FIGURE_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def _fingerprint(contract: dict[str, Any]) -> str:
    serialized = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def config_fingerprint(config: ExperimentConfig) -> str:
    """Checkpoint와 데이터·학습 설정의 호환성을 나타내는 SHA-256 지문."""
    return _fingerprint({
        "data": asdict(config.data),
        "overlap": asdict(config.overlap),
        "training": asdict(config.training),
        "composition_mode": COMPOSITION_MODE,
    })


def data_config_fingerprint(config: ExperimentConfig) -> str:
    """Manifest와 데이터·overlap 설정의 호환성을 나타내는 SHA-256 지문."""
    return _fingerprint({
        "data": asdict(config.data),
        "overlap": asdict(config.overlap),
        "class_count": CLASS_COUNT,
        "composition_mode": COMPOSITION_MODE,
    })
