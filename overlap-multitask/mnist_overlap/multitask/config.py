"""Multitask 복원 설정과 전용 산출물 경로·재현성 지문을 관리한다."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..config import PROJECT_ROOT, ExperimentConfig, config_fingerprint, load_config


DEFAULT_MULTITASK_CONFIG_PATH = PROJECT_ROOT / "configs" / "mnist_multitask.yaml"
MULTITASK_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "multitask_unet"
PILOT_OUTPUT_DIR = MULTITASK_OUTPUT_DIR / "pilot"
PILOT_SELECTION_PATH = PILOT_OUTPUT_DIR / "selection.json"
CHECKPOINT_DIR = MULTITASK_OUTPUT_DIR / "checkpoints"
TRAINING_LOG_DIR = MULTITASK_OUTPUT_DIR / "training"
RESULTS_DIR = PROJECT_ROOT / "results" / "multitask_unet"
FIGURE_DIR = RESULTS_DIR / "figures"
METRICS_JSON_PATH = RESULTS_DIR / "metrics.json"
PILOT_SEED = 0
LOSS_WEIGHT_CANDIDATES = (0.05, 0.1, 0.2)

DECODER_CONTRACT = {
    "architecture": "lenet_encoder_unet_expansive_path_v2",
    "skip_shapes": [[6, 72, 72], [16, 32, 32]],
    "bottleneck_shape": [16, 16, 16],
    "decoder_channels": [32, 16, 6],
    "output_shape": [2, 64, 64],
    "target": "center_cropped_spatial_source_layers",
    "loss": "intensity_balanced_pit_l1",
}


@dataclass(frozen=True)
class ReconstructionConfig:
    """Pilot seed와 비교할 reconstruction loss 가중치 후보."""

    pilot_seed: int
    loss_weight_candidates: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.pilot_seed != PILOT_SEED:
            raise ValueError(f"reconstruction.pilot_seed는 {PILOT_SEED}이어야 합니다.")
        if self.loss_weight_candidates != LOSS_WEIGHT_CANDIDATES:
            raise ValueError(
                "reconstruction.loss_weight_candidates는 "
                f"{list(LOSS_WEIGHT_CANDIDATES)}이어야 합니다."
            )


@dataclass(frozen=True)
class MultitaskConfig:
    """Baseline 실험 설정과 multitask reconstruction 설정 묶음."""

    baseline_config_path: Path
    baseline: ExperimentConfig
    reconstruction: ReconstructionConfig


def load_multitask_config(
    config_path: str | Path = DEFAULT_MULTITASK_CONFIG_PATH,
) -> MultitaskConfig:
    """YAML을 엄격하게 검증하고 baseline·reconstruction 설정을 함께 반환한다."""
    with Path(config_path).open(encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file)
    if not isinstance(raw, dict) or set(raw) != {"baseline_config", "reconstruction"}:
        raise ValueError(
            "Multitask config 최상위에는 baseline_config와 reconstruction만 필요합니다."
        )

    reconstruction = raw["reconstruction"]
    expected_fields = {"pilot_seed", "loss_weight_candidates"}
    if not isinstance(reconstruction, dict) or set(reconstruction) != expected_fields:
        raise ValueError(
            "reconstruction에는 pilot_seed와 loss_weight_candidates만 필요합니다."
        )

    baseline_path = Path(raw["baseline_config"])
    if not baseline_path.is_absolute():
        baseline_path = PROJECT_ROOT / baseline_path
    baseline = load_config(baseline_path)

    try:
        reconstruction_config = ReconstructionConfig(
            pilot_seed=int(reconstruction["pilot_seed"]),
            loss_weight_candidates=tuple(
                float(weight) for weight in reconstruction["loss_weight_candidates"]
            ),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"Multitask config 값이 잘못되었습니다: {error}") from error

    if reconstruction_config.pilot_seed not in baseline.training.seeds:
        raise ValueError("Pilot seed는 baseline training.seeds에 포함되어야 합니다.")

    return MultitaskConfig(baseline_path, baseline, reconstruction_config)


def create_multitask_directories() -> None:
    """Pilot·checkpoint·학습 이력·결과 그림 directory를 준비한다."""
    for directory in (
        PILOT_OUTPUT_DIR,
        CHECKPOINT_DIR,
        TRAINING_LOG_DIR,
        FIGURE_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def multitask_config_fingerprint(
    config: MultitaskConfig,
    reconstruction_loss_weight: float | None = None,
) -> str:
    """Baseline·decoder·가중치 설정의 호환성을 나타내는 SHA-256 지문."""
    contract: dict[str, Any] = {
        "baseline": config_fingerprint(config.baseline),
        "reconstruction": {
            "pilot_seed": config.reconstruction.pilot_seed,
            "loss_weight_candidates": config.reconstruction.loss_weight_candidates,
        },
        "decoder": DECODER_CONTRACT,
    }
    if reconstruction_loss_weight is not None:
        contract["selected_reconstruction_loss_weight"] = reconstruction_loss_weight
    serialized = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def pilot_weight_directory(reconstruction_loss_weight: float) -> Path:
    """가중치 값을 안정적인 짧은 이름으로 바꾼 pilot 산출물 directory를 반환한다."""
    return PILOT_OUTPUT_DIR / f"lambda_{reconstruction_loss_weight:g}"
