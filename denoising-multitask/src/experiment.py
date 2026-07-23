"""
Baseline과 multitask의 pilot, 학습, checkpoint, 평가와 CSV 저장을 담당한다.

입력:
    - main.py에서 선택한 baseline, multitask, gradient alignment 또는
      reconstruction weight 비교 실행
    - 검증된 CPU 또는 CUDA device

출력:
    - Best checkpoint, epoch history, pilot·최종 결과 및 gradient CSV

주요 기능:
    1. 고정 실험 설정과 난수 재현
    2. Classification 및 reconstruction 학습·평가
    3. 노이즈별 reconstruction weight pilot과 최종 반복 실험
    4. 재학습 중 shared-encoder gradient alignment 측정
    5. Motion Blur에서 큰 reconstruction weight의 영향 비교
"""

from __future__ import annotations

import csv
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from scipy.stats import t as student_t
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import NOISE_TYPES, PROJECT_DIRECTORY, create_data_loaders
from src.model import DenoisingAuxiliaryLeNet


RANDOM_SEEDS = tuple(range(30))
PILOT_SEED = 0
RECONSTRUCTION_WEIGHT_CANDIDATES = (0.05, 0.1, 0.2)
SELECTED_RECONSTRUCTION_WEIGHTS = {
    "awgn": 0.05,
    "motion_blur": 0.1,
    "reduced_contrast_awgn": 0.1,
}
MAXIMUM_EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 0.001
VALIDATION_RATIO = 0.1
ALIGNMENT_PROBE_BATCHES = 8
WEIGHT_SWEEP_NOISE_TYPE = "motion_blur"
WEIGHT_SWEEP_SEEDS = tuple(range(5))
WEIGHT_SWEEP_CONTROL = 0.1
WEIGHT_SWEEP_CANDIDATES = (1.0, 3.0, 10.0)

OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "outputs"
CHECKPOINT_DIRECTORY = OUTPUT_DIRECTORY / "checkpoints"
HISTORY_DIRECTORY = OUTPUT_DIRECTORY / "histories"
RESULTS_PATH = OUTPUT_DIRECTORY / "results.csv"
PILOT_RESULTS_PATH = OUTPUT_DIRECTORY / "pilot_results.csv"
ALIGNMENT_DIRECTORY = OUTPUT_DIRECTORY / "gradient_alignment"
ALIGNMENT_MEASUREMENTS_PATH = ALIGNMENT_DIRECTORY / "measurements.csv"
ALIGNMENT_SUMMARY_PATH = ALIGNMENT_DIRECTORY / "summary.csv"
WEIGHT_SWEEP_SUMMARY_PATH = ALIGNMENT_DIRECTORY / "weight_sweep_summary.csv"

RESULT_COLUMNS = (
    "noise_type",
    "condition",
    "random_seed",
    "reconstruction_weight",
    "best_epoch",
    "best_validation_accuracy",
    "test_classification_loss",
    "test_accuracy",
)
PILOT_RESULT_COLUMNS = (
    "noise_type",
    "reconstruction_weight",
    "best_epoch",
    "best_validation_accuracy",
    "selected",
)
ALIGNMENT_MEASUREMENT_COLUMNS = (
    "noise_type",
    "random_seed",
    "epoch",
    "reconstruction_weight",
    "mean_cosine_similarity",
    "mean_weighted_norm_ratio",
    "positive_probe_fraction",
)
ALIGNMENT_SUMMARY_COLUMNS = (
    "noise_type",
    "number_of_seeds",
    "mean_cosine_similarity",
    "cosine_ci_lower",
    "cosine_ci_upper",
    "early_cosine_similarity",
    "middle_cosine_similarity",
    "late_cosine_similarity",
    "positive_probe_fraction",
    "mean_weighted_norm_ratio",
    "mean_accuracy_delta_percentage_points",
    "accuracy_delta_correlation",
    "accuracy_delta_correlation_p_value",
)
WEIGHT_SWEEP_SUMMARY_COLUMNS = (
    "reconstruction_weight",
    "number_of_seeds",
    "mean_test_accuracy",
    "standard_deviation_test_accuracy",
    "mean_best_validation_accuracy",
    "mean_accuracy_delta_vs_control_percentage_points",
    "mean_cosine_similarity",
    "mean_weighted_norm_ratio",
    "positive_probe_fraction",
)


@dataclass(frozen=True)
class ExperimentConfiguration:
    """Checkpoint 호환성과 한 번의 최종 학습을 정의하는 고정 설정이다."""

    noise_type: str
    condition: str
    random_seed: int
    reconstruction_weight: float
    maximum_epochs: int = MAXIMUM_EPOCHS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    validation_ratio: float = VALIDATION_RATIO


@dataclass(frozen=True)
class EpochMetrics:
    """한 epoch의 sample-weighted loss와 classification accuracy다."""

    total_loss: float
    classification_loss: float
    reconstruction_loss: float | None
    accuracy: float


@dataclass(frozen=True)
class TrainingResult:
    """Best validation checkpoint의 epoch와 accuracy를 기록한다."""

    best_epoch: int
    best_validation_accuracy: float


def run_baseline_experiments(device: torch.device) -> None:
    """세 noise와 30개 seed의 classification-only 실험을 실행한다."""
    _create_output_directories()
    for noise_type in NOISE_TYPES:
        for random_seed in RANDOM_SEEDS:
            configuration = ExperimentConfiguration(
                noise_type=noise_type,
                condition="classification_only",
                random_seed=random_seed,
                reconstruction_weight=0.0,
            )
            _run_final_experiment(configuration, device)


def run_multitask_experiments(device: torch.device) -> None:
    """Noise별 reconstruction weight pilot 후 30개 seed를 최종 학습한다."""
    _create_output_directories()
    selected_weights = {
        noise_type: _select_reconstruction_weight(noise_type, device)
        for noise_type in NOISE_TYPES
    }
    for noise_type in NOISE_TYPES:
        for random_seed in RANDOM_SEEDS:
            configuration = ExperimentConfiguration(
                noise_type=noise_type,
                condition="multitask",
                random_seed=random_seed,
                reconstruction_weight=selected_weights[noise_type],
            )
            _run_final_experiment(configuration, device)


def run_gradient_alignment_experiments(device: torch.device) -> None:
    """Baseline과 multitask를 학습하고 multitask gradient를 기록한다."""
    _create_output_directories()
    for noise_type in NOISE_TYPES:
        for random_seed in RANDOM_SEEDS:
            configuration = ExperimentConfiguration(
                noise_type=noise_type,
                condition="classification_only",
                random_seed=random_seed,
                reconstruction_weight=0.0,
            )
            _run_final_experiment(configuration, device)

    for noise_type in NOISE_TYPES:
        for random_seed in RANDOM_SEEDS:
            configuration = ExperimentConfiguration(
                noise_type=noise_type,
                condition="multitask",
                random_seed=random_seed,
                reconstruction_weight=SELECTED_RECONSTRUCTION_WEIGHTS[noise_type],
            )
            _run_final_experiment(configuration, device, measure_alignment=True)

    _save_gradient_alignment_summary()


def run_reconstruction_weight_experiments(device: torch.device) -> None:
    """Motion Blur에서 λ 1·3·10을 5개 seed로 학습하고 gradient를 기록한다."""
    _create_output_directories()
    for reconstruction_weight in WEIGHT_SWEEP_CANDIDATES:
        condition = f"multitask_weight_{reconstruction_weight:g}"
        for random_seed in WEIGHT_SWEEP_SEEDS:
            configuration = ExperimentConfiguration(
                noise_type=WEIGHT_SWEEP_NOISE_TYPE,
                condition=condition,
                random_seed=random_seed,
                reconstruction_weight=reconstruction_weight,
            )
            _run_final_experiment(configuration, device, measure_alignment=True)
    _save_reconstruction_weight_summary()


def _run_epoch(
    model: DenoisingAuxiliaryLeNet,
    data_loader: DataLoader,
    device: torch.device,
    reconstruction_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
    include_reconstruction: bool = False,
) -> EpochMetrics:
    """DataLoader를 한 번 순회한다. optimizer가 있으면 학습, 없으면 평가한다."""
    is_training = optimizer is not None
    model.train() if is_training else model.eval()
    classification_loss_function = nn.CrossEntropyLoss()
    reconstruction_loss_function = nn.MSELoss()
    total_loss_sum = 0.0
    classification_loss_sum = 0.0
    reconstruction_loss_sum = 0.0
    total_correct = 0
    total_samples = 0
    reconstruction_enabled = (
        is_training or include_reconstruction
    ) and model.decoder is not None

    grad_context = torch.enable_grad() if is_training else torch.no_grad()
    with grad_context:
        for batch in data_loader:
            noisy_images = batch["noisy_image"].to(device, non_blocking=True)
            clean_targets = batch["clean_target"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
            output = model(
                noisy_images,
                include_reconstruction=reconstruction_enabled,
            )
            classification_loss = classification_loss_function(
                output["classification_logits"], labels
            )
            reconstruction_loss = None
            total_loss = classification_loss
            if reconstruction_enabled:
                reconstruction = output["reconstruction"]
                if reconstruction is None:
                    raise RuntimeError(
                        "Multitask 모델이 reconstruction을 반환하지 않았습니다."
                    )
                reconstruction_loss = reconstruction_loss_function(
                    reconstruction, clean_targets
                )
                total_loss = (
                    classification_loss + reconstruction_weight * reconstruction_loss
                )
            if not torch.isfinite(total_loss):
                phase = "training" if is_training else "evaluation"
                raise FloatingPointError(
                    f"유한하지 않은 {phase} loss입니다: {float(total_loss.item())}"
                )
            if is_training:
                total_loss.backward()
                optimizer.step()

            current_batch_size = labels.shape[0]
            total_samples += current_batch_size
            total_loss_sum += float(total_loss.item()) * current_batch_size
            classification_loss_sum += (
                float(classification_loss.item()) * current_batch_size
            )
            if reconstruction_loss is not None:
                reconstruction_loss_sum += (
                    float(reconstruction_loss.item()) * current_batch_size
                )
            predictions = output["classification_logits"].argmax(dim=1)
            total_correct += int((predictions == labels).sum().item())

    return EpochMetrics(
        total_loss=total_loss_sum / total_samples,
        classification_loss=classification_loss_sum / total_samples,
        reconstruction_loss=(
            reconstruction_loss_sum / total_samples if reconstruction_enabled else None
        ),
        accuracy=total_correct / total_samples,
    )


def _measure_gradient_alignment(
    model: DenoisingAuxiliaryLeNet,
    data_loader: DataLoader,
    device: torch.device,
    configuration: ExperimentConfiguration,
    epoch: int,
) -> dict[str, Any]:
    """고정 validation probe에서 두 loss의 shared-encoder gradient를 비교한다."""
    if model.decoder is None:
        raise ValueError("Gradient alignment는 decoder가 있는 모델에서만 계산합니다.")
    model.eval()
    encoder_parameters = tuple(model.encoder.parameters())
    classification_loss_function = nn.CrossEntropyLoss()
    reconstruction_loss_function = nn.MSELoss()
    cosine_values = []
    norm_ratios = []
    with torch.enable_grad():
        for probe_batch, batch in enumerate(data_loader):
            if probe_batch >= ALIGNMENT_PROBE_BATCHES:
                break
            noisy_images = batch["noisy_image"].to(device, non_blocking=True)
            clean_targets = batch["clean_target"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            output = model(noisy_images, include_reconstruction=True)
            reconstruction = output["reconstruction"]
            if reconstruction is None:
                raise RuntimeError(
                    "Gradient probe에서 reconstruction을 얻지 못했습니다."
                )
            classification_loss = classification_loss_function(
                output["classification_logits"],
                labels,
            )
            reconstruction_loss = reconstruction_loss_function(
                reconstruction,
                clean_targets,
            )
            classification_gradients = torch.autograd.grad(
                classification_loss,
                encoder_parameters,
                retain_graph=True,
            )
            reconstruction_gradients = torch.autograd.grad(
                reconstruction_loss,
                encoder_parameters,
            )
            classification_gradient = torch.cat(
                [gradient.flatten() for gradient in classification_gradients]
            )
            reconstruction_gradient = torch.cat(
                [gradient.flatten() for gradient in reconstruction_gradients]
            )
            classification_norm = classification_gradient.norm()
            reconstruction_norm = reconstruction_gradient.norm()
            if classification_norm.item() == 0.0 or reconstruction_norm.item() == 0.0:
                raise FloatingPointError("0인 shared-encoder gradient norm입니다.")
            cosine_values.append(
                float(
                    torch.dot(classification_gradient, reconstruction_gradient).div(
                        classification_norm * reconstruction_norm
                    )
                )
            )
            norm_ratios.append(
                float(
                    configuration.reconstruction_weight
                    * reconstruction_norm
                    / classification_norm
                )
            )
    if len(cosine_values) != ALIGNMENT_PROBE_BATCHES:
        raise RuntimeError(
            f"Gradient probe batch가 부족합니다: "
            f"{len(cosine_values)}/{ALIGNMENT_PROBE_BATCHES}"
        )
    return {
        "noise_type": configuration.noise_type,
        "random_seed": configuration.random_seed,
        "epoch": epoch,
        "reconstruction_weight": configuration.reconstruction_weight,
        "mean_cosine_similarity": float(np.mean(cosine_values)),
        "mean_weighted_norm_ratio": float(np.mean(norm_ratios)),
        "positive_probe_fraction": float(np.mean(np.asarray(cosine_values) > 0.0)),
    }


def _run_final_experiment(
    configuration: ExperimentConfiguration,
    device: torch.device,
    measure_alignment: bool = False,
) -> None:
    """한 noise·condition·seed를 학습하거나 완료 checkpoint를 재사용해 평가한다."""
    checkpoint_path = CHECKPOINT_DIRECTORY / _artifact_name(configuration, ".pt")
    history_path = HISTORY_DIRECTORY / _artifact_name(configuration, ".csv")
    _set_random_seed(configuration.random_seed)
    training_loader, validation_loader, test_loader = create_data_loaders(
        configuration.noise_type,
        configuration.batch_size,
        configuration.validation_ratio,
        configuration.random_seed,
        use_pinned_memory=device.type == "cuda",
    )
    model = DenoisingAuxiliaryLeNet(
        use_decoder=configuration.condition.startswith("multitask")
    ).to(device)

    can_reuse = _checkpoint_matches(checkpoint_path, history_path, configuration)
    if measure_alignment:
        can_reuse = can_reuse and _alignment_run_is_complete(configuration)

    if can_reuse:
        checkpoint = _load_checkpoint(
            model, checkpoint_path, device, configuration, require_complete=True
        )
        training_result = TrainingResult(
            best_epoch=int(checkpoint["best_epoch"]),
            best_validation_accuracy=float(checkpoint["best_validation_accuracy"]),
        )
        print(f"완료 checkpoint를 재사용합니다: {checkpoint_path}")
    else:
        alignment_rows = [] if measure_alignment else None
        training_result = _train_model(
            model,
            training_loader,
            validation_loader,
            configuration,
            checkpoint_path,
            history_path,
            device,
            alignment_rows=alignment_rows,
        )
        if alignment_rows is not None:
            _save_alignment_measurements(configuration, alignment_rows)

    test_metrics = _run_epoch(
        model,
        test_loader,
        device,
        configuration.reconstruction_weight,
    )
    _upsert_final_result(configuration, training_result, test_metrics)
    print(
        f"완료: noise={configuration.noise_type} "
        f"condition={configuration.condition} seed={configuration.random_seed} "
        f"test_accuracy={test_metrics.accuracy:.4f}"
    )


def _train_model(
    model: DenoisingAuxiliaryLeNet,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    configuration: ExperimentConfiguration,
    checkpoint_path: Path,
    history_path: Path,
    device: torch.device,
    alignment_rows: list[dict[str, Any]] | None = None,
) -> TrainingResult:
    """고정 epoch를 학습하고 validation accuracy 기준 best checkpoint를 복원한다."""
    optimizer = torch.optim.Adam(model.parameters(), lr=configuration.learning_rate)
    if alignment_rows is not None:
        phase = "alignment"
    elif checkpoint_path.stem.startswith("pilot_"):
        phase = "pilot"
    else:
        phase = "final"
    best_validation_accuracy = -1.0
    best_epoch = 0
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.writer(history_file)
        writer.writerow(
            [
                "epoch",
                "training_total_loss",
                "training_classification_loss",
                "training_reconstruction_loss",
                "training_accuracy",
                "validation_total_loss",
                "validation_classification_loss",
                "validation_reconstruction_loss",
                "validation_accuracy",
            ]
        )
        for epoch in range(1, configuration.maximum_epochs + 1):
            training_metrics = _run_epoch(
                model,
                training_loader,
                device,
                configuration.reconstruction_weight,
                optimizer=optimizer,
            )
            validation_metrics = _run_epoch(
                model,
                validation_loader,
                device,
                configuration.reconstruction_weight,
                include_reconstruction=True,
            )
            writer.writerow(
                [
                    epoch,
                    f"{training_metrics.total_loss:.8f}",
                    f"{training_metrics.classification_loss:.8f}",
                    _format_optional_metric(training_metrics.reconstruction_loss),
                    f"{training_metrics.accuracy:.8f}",
                    f"{validation_metrics.total_loss:.8f}",
                    f"{validation_metrics.classification_loss:.8f}",
                    _format_optional_metric(validation_metrics.reconstruction_loss),
                    f"{validation_metrics.accuracy:.8f}",
                ]
            )
            history_file.flush()
            if alignment_rows is not None:
                alignment_rows.append(
                    _measure_gradient_alignment(
                        model,
                        validation_loader,
                        device,
                        configuration,
                        epoch,
                    )
                )
            print(
                f"phase={phase} noise={configuration.noise_type} "
                f"condition={configuration.condition} "
                f"reconstruction_weight={configuration.reconstruction_weight:g} "
                f"seed={configuration.random_seed} epoch={epoch} "
                f"validation_accuracy={validation_metrics.accuracy:.4f}"
            )
            if validation_metrics.accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_metrics.accuracy
                best_epoch = epoch
                _save_checkpoint(
                    {
                        "configuration": asdict(configuration),
                        "training_complete": False,
                        "best_epoch": best_epoch,
                        "best_validation_accuracy": best_validation_accuracy,
                        "model_state_dict": model.state_dict(),
                    },
                    checkpoint_path,
                )

    if best_epoch == 0:
        raise RuntimeError("학습에서 best checkpoint를 생성하지 못했습니다.")
    checkpoint = _load_checkpoint(
        model, checkpoint_path, device, configuration, require_complete=False
    )
    checkpoint["training_complete"] = True
    _save_checkpoint(checkpoint, checkpoint_path)
    return TrainingResult(best_epoch, best_validation_accuracy)


def _select_reconstruction_weight(noise_type: str, device: torch.device) -> float:
    """기존 pilot 결과를 재사용하거나 seed 0 pilot으로 noise별 weight를 고른다."""
    existing_selection = _read_pilot_selection(noise_type)
    if existing_selection is not None:
        print(
            f"기존 pilot 결과를 사용합니다: noise={noise_type} "
            f"weight={existing_selection}"
        )
        return existing_selection

    pilot_rows = []
    for reconstruction_weight in RECONSTRUCTION_WEIGHT_CANDIDATES:
        configuration = ExperimentConfiguration(
            noise_type=noise_type,
            condition="multitask",
            random_seed=PILOT_SEED,
            reconstruction_weight=reconstruction_weight,
        )
        checkpoint_path = CHECKPOINT_DIRECTORY / (
            f"pilot_{noise_type}_weight_{reconstruction_weight:g}.pt"
        )
        history_path = HISTORY_DIRECTORY / (
            f"pilot_{noise_type}_weight_{reconstruction_weight:g}.csv"
        )
        _set_random_seed(PILOT_SEED)
        training_loader, validation_loader, _ = create_data_loaders(
            noise_type,
            configuration.batch_size,
            configuration.validation_ratio,
            PILOT_SEED,
            use_pinned_memory=device.type == "cuda",
        )
        model = DenoisingAuxiliaryLeNet(use_decoder=True).to(device)
        try:
            result = _train_model(
                model,
                training_loader,
                validation_loader,
                configuration,
                checkpoint_path,
                history_path,
                device,
            )
            pilot_rows.append(
                {
                    "noise_type": noise_type,
                    "reconstruction_weight": reconstruction_weight,
                    "best_epoch": result.best_epoch,
                    "best_validation_accuracy": result.best_validation_accuracy,
                }
            )
        finally:
            checkpoint_path.unlink(missing_ok=True)
            checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp").unlink(
                missing_ok=True
            )
            history_path.unlink(missing_ok=True)

    selected_row = max(
        pilot_rows,
        key=lambda row: float(row["best_validation_accuracy"]),
    )
    selected_weight = float(selected_row["reconstruction_weight"])
    for row in pilot_rows:
        row["selected"] = float(row["reconstruction_weight"]) == selected_weight
    _write_pilot_results(noise_type, pilot_rows)
    print(f"Pilot 선택: noise={noise_type} weight={selected_weight}")
    return selected_weight


def _set_random_seed(random_seed: int) -> None:
    """Python·NumPy·PyTorch 난수와 결정론 실행을 고정한다."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.use_deterministic_algorithms(True)


def _create_output_directories() -> None:
    """학습 산출물에 필요한 최소 directory를 생성한다."""
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    HISTORY_DIRECTORY.mkdir(parents=True, exist_ok=True)


def _artifact_name(configuration: ExperimentConfiguration, suffix: str) -> str:
    """최종 run을 유일하게 식별하는 파일명을 만든다."""
    return (
        f"{configuration.noise_type}_{configuration.condition}_"
        f"seed_{configuration.random_seed}{suffix}"
    )


def _save_checkpoint(checkpoint: dict[str, Any], checkpoint_path: Path) -> None:
    """Checkpoint를 임시 파일에 저장한 후 목표 경로로 원자 교체한다."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(checkpoint_path)


def _load_checkpoint(
    model: DenoisingAuxiliaryLeNet,
    checkpoint_path: Path,
    device: torch.device,
    configuration: ExperimentConfiguration,
    require_complete: bool,
) -> dict[str, Any]:
    """현재 configuration과 완료 상태를 확인하고 model weight를 복원한다."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as error:
        raise RuntimeError(
            f"Checkpoint를 불러오지 못했습니다: {checkpoint_path}"
        ) from error
    if checkpoint.get("configuration") != asdict(configuration):
        raise ValueError(f"현재 설정과 다른 checkpoint입니다: {checkpoint_path}")
    if require_complete and checkpoint.get("training_complete") is not True:
        raise ValueError(f"정상 완료되지 않은 checkpoint입니다: {checkpoint_path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return checkpoint


def _checkpoint_matches(
    checkpoint_path: Path,
    history_path: Path,
    configuration: ExperimentConfiguration,
) -> bool:
    """Checkpoint와 history가 현재 설정으로 정상 완료됐는지 확인한다."""
    if not checkpoint_path.exists() or not history_path.exists():
        return False
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        return False
    return (
        checkpoint.get("configuration") == asdict(configuration)
        and checkpoint.get("training_complete") is True
    )


def _upsert_final_result(
    configuration: ExperimentConfiguration,
    training_result: TrainingResult,
    test_metrics: EpochMetrics,
) -> None:
    """한 final run의 결과를 복합 key 기준으로 results.csv에 원자 upsert한다."""
    row = {
        "noise_type": configuration.noise_type,
        "condition": configuration.condition,
        "random_seed": configuration.random_seed,
        "reconstruction_weight": configuration.reconstruction_weight,
        "best_epoch": training_result.best_epoch,
        "best_validation_accuracy": training_result.best_validation_accuracy,
        "test_classification_loss": test_metrics.classification_loss,
        "test_accuracy": test_metrics.accuracy,
    }
    if RESULTS_PATH.exists():
        results = pd.read_csv(RESULTS_PATH)
        if tuple(results.columns) != RESULT_COLUMNS:
            raise ValueError(f"results.csv schema가 올바르지 않습니다: {RESULTS_PATH}")
        matching = (
            (results["noise_type"] == configuration.noise_type)
            & (results["condition"] == configuration.condition)
            & (results["random_seed"] == configuration.random_seed)
        )
        results = results.loc[~matching]
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
    else:
        results = pd.DataFrame([row], columns=RESULT_COLUMNS)
    results = results.sort_values(
        ["noise_type", "condition", "random_seed"]
    ).reset_index(drop=True)
    _write_dataframe_atomically(results, RESULTS_PATH)


def _read_pilot_selection(noise_type: str) -> float | None:
    """완전한 기존 pilot 결과가 있으면 선택된 reconstruction weight를 반환한다."""
    if not PILOT_RESULTS_PATH.exists():
        return None
    results = pd.read_csv(PILOT_RESULTS_PATH)
    if tuple(results.columns) != PILOT_RESULT_COLUMNS:
        raise ValueError(
            f"pilot_results.csv schema가 올바르지 않습니다: {PILOT_RESULTS_PATH}"
        )
    noise_results = results.loc[results["noise_type"] == noise_type]
    if set(noise_results["reconstruction_weight"].astype(float)) != set(
        RECONSTRUCTION_WEIGHT_CANDIDATES
    ):
        return None
    selected = noise_results.loc[noise_results["selected"].astype(bool)]
    if len(selected) != 1:
        return None
    return float(selected.iloc[0]["reconstruction_weight"])


def _read_optional_dataframe(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    """선택적 CSV를 읽고, 없으면 같은 schema의 빈 DataFrame을 반환한다."""
    if not path.exists():
        return pd.DataFrame(columns=columns)
    dataframe = pd.read_csv(path)
    if tuple(dataframe.columns) != columns:
        raise ValueError(f"CSV schema가 올바르지 않습니다: {path}")
    return dataframe


def _weight_matches(values: pd.Series, reconstruction_weight: float) -> np.ndarray:
    """빈 CSV를 포함한 weight 열을 안전하게 실수로 비교한다."""
    numeric_values = pd.to_numeric(values, errors="coerce").to_numpy(
        dtype=np.float64
    )
    return np.isclose(numeric_values, reconstruction_weight)


def _alignment_run_is_complete(configuration: ExperimentConfiguration) -> bool:
    """현재 설정의 모든 epoch·probe batch가 저장됐는지 확인한다."""
    measurements = _read_optional_dataframe(
        ALIGNMENT_MEASUREMENTS_PATH,
        ALIGNMENT_MEASUREMENT_COLUMNS,
    )
    measurement_rows = measurements.loc[
        (measurements["noise_type"] == configuration.noise_type)
        & (measurements["random_seed"] == configuration.random_seed)
        & _weight_matches(
            measurements["reconstruction_weight"],
            configuration.reconstruction_weight,
        )
    ]
    expected_rows = configuration.maximum_epochs
    if len(measurement_rows) != expected_rows:
        return False
    return set(measurement_rows["epoch"].astype(int)) == set(
        range(1, configuration.maximum_epochs + 1)
    ) and np.allclose(
        measurement_rows["reconstruction_weight"],
        configuration.reconstruction_weight,
    )


def _save_alignment_measurements(
    configuration: ExperimentConfiguration,
    alignment_rows: list[dict[str, Any]],
) -> None:
    """완료된 한 run의 gradient 측정값을 복합 key 기준으로 교체한다."""
    replacement = pd.DataFrame(
        alignment_rows,
        columns=ALIGNMENT_MEASUREMENT_COLUMNS,
    )
    expected_rows = configuration.maximum_epochs
    if len(replacement) != expected_rows:
        raise RuntimeError(
            f"Gradient measurement가 불완전합니다: {len(replacement)}/{expected_rows}"
        )
    existing = _read_optional_dataframe(
        ALIGNMENT_MEASUREMENTS_PATH,
        ALIGNMENT_MEASUREMENT_COLUMNS,
    )
    keep = ~(
        (existing["noise_type"] == configuration.noise_type)
        & (existing["random_seed"] == configuration.random_seed)
        & _weight_matches(
            existing["reconstruction_weight"],
            configuration.reconstruction_weight,
        )
    )
    measurements = (
        replacement
        if existing.empty
        else pd.concat([existing.loc[keep], replacement], ignore_index=True)
    )
    measurements = measurements.sort_values(
        ["noise_type", "reconstruction_weight", "random_seed", "epoch"]
    ).reset_index(drop=True)
    _write_dataframe_atomically(measurements, ALIGNMENT_MEASUREMENTS_PATH)


def _save_gradient_alignment_summary() -> None:
    """Seed-level gradient 통계와 accuracy delta 상관을 CSV와 stdout에 남긴다."""
    measurements = _read_optional_dataframe(
        ALIGNMENT_MEASUREMENTS_PATH,
        ALIGNMENT_MEASUREMENT_COLUMNS,
    )
    results = pd.read_csv(RESULTS_PATH)
    if tuple(results.columns) != RESULT_COLUMNS:
        raise ValueError(f"results.csv schema가 올바르지 않습니다: {RESULTS_PATH}")

    accuracies = results.pivot(
        index=["noise_type", "random_seed"],
        columns="condition",
        values="test_accuracy",
    )
    required_conditions = {"classification_only", "multitask"}
    if not required_conditions.issubset(accuracies.columns):
        raise RuntimeError("Baseline과 multitask test 결과가 모두 필요합니다.")

    summary_rows = []
    print("\nGradient alignment summary (seed-level means)")
    print(
        "noise | cosine mean [95% CI] | early → middle → late | "
        "weighted MSE/CE norm | positive probes | corr(cosine, Δaccuracy)"
    )
    for noise_type in NOISE_TYPES:
        noise_measurements = measurements.loc[
            (measurements["noise_type"] == noise_type)
            & _weight_matches(
                measurements["reconstruction_weight"],
                SELECTED_RECONSTRUCTION_WEIGHTS[noise_type],
            )
        ]
        expected_rows = len(RANDOM_SEEDS) * MAXIMUM_EPOCHS
        if len(noise_measurements) != expected_rows:
            raise RuntimeError(
                f"Gradient measurement가 불완전합니다: "
                f"noise={noise_type} {len(noise_measurements)}/{expected_rows}"
            )
        seed_cosine = noise_measurements.groupby("random_seed")[
            "mean_cosine_similarity"
        ].mean()
        cosine = seed_cosine.to_numpy(dtype=np.float64)
        standard_error = cosine.std(ddof=1) / np.sqrt(len(cosine))
        half_width = float(student_t.ppf(0.975, df=len(cosine) - 1) * standard_error)
        noise_accuracies = accuracies.loc[noise_type]
        delta = (
            noise_accuracies["multitask"] - noise_accuracies["classification_only"]
        ).reindex(seed_cosine.index)
        if cosine.std(ddof=1) > 0.0 and delta.std(ddof=1) > 0.0:
            correlation = pearsonr(cosine, delta)
            correlation_value = float(correlation.statistic)
            correlation_p_value = float(correlation.pvalue)
            correlation_text = (
                f"r={correlation_value:+.3f}, p={correlation_p_value:.4f}"
            )
        else:
            correlation_value = np.nan
            correlation_p_value = np.nan
            correlation_text = "undefined"

        def stage_mean(first_epoch: int, last_epoch: int) -> float:
            return float(
                noise_measurements.loc[
                    noise_measurements["epoch"].between(first_epoch, last_epoch),
                    "mean_cosine_similarity",
                ].mean()
            )

        mean_cosine = float(cosine.mean())
        early_cosine = stage_mean(1, 10)
        middle_cosine = stage_mean(11, 20)
        late_cosine = stage_mean(21, 30)
        positive_fraction = float(noise_measurements["positive_probe_fraction"].mean())
        mean_norm_ratio = float(noise_measurements["mean_weighted_norm_ratio"].mean())
        summary_rows.append(
            {
                "noise_type": noise_type,
                "number_of_seeds": len(cosine),
                "mean_cosine_similarity": mean_cosine,
                "cosine_ci_lower": mean_cosine - half_width,
                "cosine_ci_upper": mean_cosine + half_width,
                "early_cosine_similarity": early_cosine,
                "middle_cosine_similarity": middle_cosine,
                "late_cosine_similarity": late_cosine,
                "positive_probe_fraction": positive_fraction,
                "mean_weighted_norm_ratio": mean_norm_ratio,
                "mean_accuracy_delta_percentage_points": float(delta.mean() * 100.0),
                "accuracy_delta_correlation": correlation_value,
                "accuracy_delta_correlation_p_value": correlation_p_value,
            }
        )
        print(
            f"{noise_type} | {mean_cosine:+.4f} "
            f"[{mean_cosine - half_width:+.4f}, "
            f"{mean_cosine + half_width:+.4f}] | "
            f"{early_cosine:+.4f} → {middle_cosine:+.4f} → "
            f"{late_cosine:+.4f} | {mean_norm_ratio:.6f} | "
            f"{positive_fraction * 100.0:.1f}% | "
            f"{correlation_text}"
        )
    summary = pd.DataFrame(summary_rows, columns=ALIGNMENT_SUMMARY_COLUMNS)
    _write_dataframe_atomically(summary, ALIGNMENT_SUMMARY_PATH)


def _save_reconstruction_weight_summary() -> None:
    """큰 λ별 5-seed 정확도와 gradient 통계를 CSV와 stdout에 남긴다."""
    results = pd.read_csv(RESULTS_PATH)
    if tuple(results.columns) != RESULT_COLUMNS:
        raise ValueError(f"results.csv schema가 올바르지 않습니다: {RESULTS_PATH}")
    measurements = _read_optional_dataframe(
        ALIGNMENT_MEASUREMENTS_PATH,
        ALIGNMENT_MEASUREMENT_COLUMNS,
    )
    control_results = results.loc[
        (results["noise_type"] == WEIGHT_SWEEP_NOISE_TYPE)
        & (results["condition"] == "multitask")
        & (results["random_seed"].isin(WEIGHT_SWEEP_SEEDS))
        & _weight_matches(
            results["reconstruction_weight"],
            WEIGHT_SWEEP_CONTROL,
        )
    ].set_index("random_seed")
    summary_rows = []
    print("\nReconstruction weight summary (Motion Blur, 5 seeds)")
    print(
        "lambda | test accuracy mean ± std | delta vs 0.1 | validation accuracy | "
        "cosine | weighted MSE/CE norm | positive probes"
    )
    for reconstruction_weight in (WEIGHT_SWEEP_CONTROL, *WEIGHT_SWEEP_CANDIDATES):
        condition = (
            "multitask"
            if reconstruction_weight == WEIGHT_SWEEP_CONTROL
            else f"multitask_weight_{reconstruction_weight:g}"
        )
        weight_results = results.loc[
            (results["noise_type"] == WEIGHT_SWEEP_NOISE_TYPE)
            & (results["condition"] == condition)
            & (results["random_seed"].isin(WEIGHT_SWEEP_SEEDS))
            & _weight_matches(
                results["reconstruction_weight"],
                reconstruction_weight,
            )
        ]
        weight_measurements = measurements.loc[
            (measurements["noise_type"] == WEIGHT_SWEEP_NOISE_TYPE)
            & (measurements["random_seed"].isin(WEIGHT_SWEEP_SEEDS))
            & _weight_matches(
                measurements["reconstruction_weight"],
                reconstruction_weight,
            )
        ]
        expected_measurements = len(WEIGHT_SWEEP_SEEDS) * MAXIMUM_EPOCHS
        if (
            reconstruction_weight == WEIGHT_SWEEP_CONTROL
            and (
                len(weight_results) != len(WEIGHT_SWEEP_SEEDS)
                or len(weight_measurements) != expected_measurements
            )
        ):
            print("0.1 | 기존 5-seed 결과 또는 gradient가 없어 제외합니다.")
            continue
        if len(weight_results) != len(WEIGHT_SWEEP_SEEDS):
            raise RuntimeError(
                f"Weight sweep 결과가 불완전합니다: λ={reconstruction_weight:g} "
                f"{len(weight_results)}/{len(WEIGHT_SWEEP_SEEDS)}"
            )
        if len(weight_measurements) != expected_measurements:
            raise RuntimeError(
                f"Weight sweep gradient가 불완전합니다: "
                f"λ={reconstruction_weight:g} "
                f"{len(weight_measurements)}/{expected_measurements}"
            )
        mean_test_accuracy = float(weight_results["test_accuracy"].mean())
        standard_deviation = float(weight_results["test_accuracy"].std(ddof=1))
        mean_validation_accuracy = float(
            weight_results["best_validation_accuracy"].mean()
        )
        if len(control_results) == len(WEIGHT_SWEEP_SEEDS):
            paired_results = weight_results.set_index("random_seed")
            mean_delta = float(
                (
                    paired_results["test_accuracy"]
                    - control_results["test_accuracy"]
                ).mean()
                * 100.0
            )
        else:
            mean_delta = np.nan
        mean_cosine = float(
            weight_measurements["mean_cosine_similarity"].mean()
        )
        mean_norm_ratio = float(
            weight_measurements["mean_weighted_norm_ratio"].mean()
        )
        positive_fraction = float(
            weight_measurements["positive_probe_fraction"].mean()
        )
        summary_rows.append(
            {
                "reconstruction_weight": reconstruction_weight,
                "number_of_seeds": len(WEIGHT_SWEEP_SEEDS),
                "mean_test_accuracy": mean_test_accuracy,
                "standard_deviation_test_accuracy": standard_deviation,
                "mean_best_validation_accuracy": mean_validation_accuracy,
                "mean_accuracy_delta_vs_control_percentage_points": mean_delta,
                "mean_cosine_similarity": mean_cosine,
                "mean_weighted_norm_ratio": mean_norm_ratio,
                "positive_probe_fraction": positive_fraction,
            }
        )
        print(
            f"{reconstruction_weight:g} | "
            f"{mean_test_accuracy * 100.0:.3f}% ± "
            f"{standard_deviation * 100.0:.3f}% | "
            f"{mean_delta:+.3f}pp | "
            f"{mean_validation_accuracy * 100.0:.3f}% | "
            f"{mean_cosine:+.4f} | {mean_norm_ratio:.6f} | "
            f"{positive_fraction * 100.0:.1f}%"
        )
    summary = pd.DataFrame(
        summary_rows,
        columns=WEIGHT_SWEEP_SUMMARY_COLUMNS,
    )
    _write_dataframe_atomically(summary, WEIGHT_SWEEP_SUMMARY_PATH)


def _write_pilot_results(noise_type: str, pilot_rows: list[dict[str, Any]]) -> None:
    """한 noise의 후보별 validation 결과와 선택 여부를 pilot_results.csv에 기록한다."""
    if PILOT_RESULTS_PATH.exists():
        results = pd.read_csv(PILOT_RESULTS_PATH)
        if tuple(results.columns) != PILOT_RESULT_COLUMNS:
            raise ValueError(
                f"pilot_results.csv schema가 올바르지 않습니다: {PILOT_RESULTS_PATH}"
            )
        results = results.loc[results["noise_type"] != noise_type]
        results = pd.concat([results, pd.DataFrame(pilot_rows)], ignore_index=True)
    else:
        results = pd.DataFrame(pilot_rows, columns=PILOT_RESULT_COLUMNS)
    results = results.sort_values(["noise_type", "reconstruction_weight"]).reset_index(
        drop=True
    )
    _write_dataframe_atomically(results, PILOT_RESULTS_PATH)


def _write_dataframe_atomically(dataframe: pd.DataFrame, path: Path) -> None:
    """DataFrame을 임시 CSV에 쓴 뒤 목표 경로로 원자 교체한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    dataframe.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def _format_optional_metric(metric: float | None) -> str:
    """계산하지 않은 reconstruction metric은 빈 CSV field로 기록한다."""
    return "" if metric is None else f"{metric:.8f}"
