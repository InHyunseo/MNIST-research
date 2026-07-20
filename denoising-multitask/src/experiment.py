"""
Baseline과 multitask의 pilot, 학습, checkpoint, 평가와 CSV 저장을 담당한다.

입력:
    - main.py에서 선택한 baseline 또는 multitask 실행
    - 검증된 CPU 또는 CUDA device

출력:
    - Best checkpoint, epoch history, pilot 및 최종 결과 CSV

주요 기능:
    1. 고정 실험 설정과 난수 재현
    2. Classification 및 reconstruction 학습·평가
    3. 노이즈별 reconstruction weight pilot과 최종 반복 실험
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
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import NOISE_TYPES, PROJECT_DIRECTORY, create_data_loaders
from src.model import DenoisingAuxiliaryLeNet


RANDOM_SEEDS = (0, 1, 2, 3, 4)
PILOT_SEED = 0
RECONSTRUCTION_WEIGHT_CANDIDATES = (0.05, 0.1, 0.2)
MAXIMUM_EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 0.001
VALIDATION_RATIO = 0.1

OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "outputs"
CHECKPOINT_DIRECTORY = OUTPUT_DIRECTORY / "checkpoints"
HISTORY_DIRECTORY = OUTPUT_DIRECTORY / "histories"
RESULTS_PATH = OUTPUT_DIRECTORY / "results.csv"
PILOT_RESULTS_PATH = OUTPUT_DIRECTORY / "pilot_results.csv"

RESULT_COLUMNS = (
    "noise_type",
    "condition",
    "random_seed",
    "reconstruction_weight",
    "best_epoch",
    "best_validation_accuracy",
    "test_classification_loss",
    "test_reconstruction_loss",
    "test_accuracy",
)
PILOT_RESULT_COLUMNS = (
    "noise_type",
    "reconstruction_weight",
    "best_epoch",
    "best_validation_accuracy",
    "selected",
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
    """세 noise와 다섯 seed의 classification-only 실험을 실행한다."""
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
    """Noise별 reconstruction weight pilot 후 다섯 seed를 최종 학습한다."""
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


def train_one_epoch(
    model: DenoisingAuxiliaryLeNet,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    reconstruction_weight: float,
) -> EpochMetrics:
    """DataLoader 전체를 한 번 순회해 CE 또는 CE+weighted MSE로 학습한다."""
    model.train()
    classification_loss_function = nn.CrossEntropyLoss()
    reconstruction_loss_function = nn.MSELoss()
    total_loss_sum = 0.0
    classification_loss_sum = 0.0
    reconstruction_loss_sum = 0.0
    total_correct = 0
    total_samples = 0
    reconstruction_enabled = model.decoder is not None

    for batch in data_loader:
        noisy_images = batch["noisy_image"].to(device, non_blocking=True)
        clean_targets = batch["clean_target"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(noisy_images)
        classification_loss = classification_loss_function(
            output["classification_logits"], labels
        )
        reconstruction_loss = None
        total_loss = classification_loss
        if reconstruction_enabled:
            reconstruction = output["reconstruction"]
            if reconstruction is None:
                raise RuntimeError("Multitask 모델이 reconstruction을 반환하지 않았습니다.")
            reconstruction_loss = reconstruction_loss_function(
                reconstruction, clean_targets
            )
            total_loss = classification_loss + reconstruction_weight * reconstruction_loss
        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"유한하지 않은 training loss입니다: {float(total_loss.item())}"
            )
        total_loss.backward()
        optimizer.step()

        current_batch_size = labels.shape[0]
        total_samples += current_batch_size
        total_loss_sum += float(total_loss.item()) * current_batch_size
        classification_loss_sum += float(classification_loss.item()) * current_batch_size
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


@torch.no_grad()
def evaluate_model(
    model: DenoisingAuxiliaryLeNet,
    data_loader: DataLoader,
    device: torch.device,
    reconstruction_weight: float,
) -> EpochMetrics:
    """Parameter를 변경하지 않고 loss와 classification accuracy를 계산한다."""
    model.eval()
    classification_loss_function = nn.CrossEntropyLoss()
    reconstruction_loss_function = nn.MSELoss()
    total_loss_sum = 0.0
    classification_loss_sum = 0.0
    reconstruction_loss_sum = 0.0
    total_correct = 0
    total_samples = 0
    reconstruction_enabled = model.decoder is not None

    for batch in data_loader:
        noisy_images = batch["noisy_image"].to(device, non_blocking=True)
        clean_targets = batch["clean_target"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        output = model(noisy_images)
        classification_loss = classification_loss_function(
            output["classification_logits"], labels
        )
        reconstruction_loss = None
        total_loss = classification_loss
        if reconstruction_enabled:
            reconstruction = output["reconstruction"]
            if reconstruction is None:
                raise RuntimeError("Multitask 모델이 reconstruction을 반환하지 않았습니다.")
            reconstruction_loss = reconstruction_loss_function(
                reconstruction, clean_targets
            )
            total_loss = classification_loss + reconstruction_weight * reconstruction_loss
        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"유한하지 않은 evaluation loss입니다: {float(total_loss.item())}"
            )

        current_batch_size = labels.shape[0]
        total_samples += current_batch_size
        total_loss_sum += float(total_loss.item()) * current_batch_size
        classification_loss_sum += float(classification_loss.item()) * current_batch_size
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


def _run_final_experiment(
    configuration: ExperimentConfiguration, device: torch.device
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
        use_decoder=configuration.condition == "multitask"
    ).to(device)

    if _checkpoint_matches(checkpoint_path, history_path, configuration):
        checkpoint = _load_checkpoint(
            model, checkpoint_path, device, configuration, require_complete=True
        )
        training_result = TrainingResult(
            best_epoch=int(checkpoint["best_epoch"]),
            best_validation_accuracy=float(checkpoint["best_validation_accuracy"]),
        )
        print(f"완료 checkpoint를 재사용합니다: {checkpoint_path}")
    else:
        training_result = _train_model(
            model,
            training_loader,
            validation_loader,
            configuration,
            checkpoint_path,
            history_path,
            device,
        )

    test_metrics = evaluate_model(
        model, test_loader, device, configuration.reconstruction_weight
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
) -> TrainingResult:
    """고정 epoch를 학습하고 validation accuracy 기준 best checkpoint를 복원한다."""
    optimizer = torch.optim.Adam(model.parameters(), lr=configuration.learning_rate)
    best_validation_accuracy = -1.0
    best_epoch = 0
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.writer(history_file)
        writer.writerow([
            "epoch",
            "training_total_loss",
            "training_classification_loss",
            "training_reconstruction_loss",
            "training_accuracy",
            "validation_total_loss",
            "validation_classification_loss",
            "validation_reconstruction_loss",
            "validation_accuracy",
        ])
        for epoch in range(1, configuration.maximum_epochs + 1):
            training_metrics = train_one_epoch(
                model,
                training_loader,
                optimizer,
                device,
                configuration.reconstruction_weight,
            )
            validation_metrics = evaluate_model(
                model,
                validation_loader,
                device,
                configuration.reconstruction_weight,
            )
            writer.writerow([
                epoch,
                f"{training_metrics.total_loss:.8f}",
                f"{training_metrics.classification_loss:.8f}",
                _format_optional_metric(training_metrics.reconstruction_loss),
                f"{training_metrics.accuracy:.8f}",
                f"{validation_metrics.total_loss:.8f}",
                f"{validation_metrics.classification_loss:.8f}",
                _format_optional_metric(validation_metrics.reconstruction_loss),
                f"{validation_metrics.accuracy:.8f}",
            ])
            history_file.flush()
            print(
                f"noise={configuration.noise_type} "
                f"condition={configuration.condition} "
                f"seed={configuration.random_seed} epoch={epoch} "
                f"validation_accuracy={validation_metrics.accuracy:.4f}"
            )
            if validation_metrics.accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_metrics.accuracy
                best_epoch = epoch
                _save_checkpoint({
                    "configuration": asdict(configuration),
                    "training_complete": False,
                    "best_epoch": best_epoch,
                    "best_validation_accuracy": best_validation_accuracy,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, checkpoint_path)

    if best_epoch == 0:
        raise RuntimeError("학습에서 best checkpoint를 생성하지 못했습니다.")
    checkpoint = _load_checkpoint(
        model, checkpoint_path, device, configuration, require_complete=False
    )
    checkpoint["training_complete"] = True
    _save_checkpoint(checkpoint, checkpoint_path)
    return TrainingResult(best_epoch, best_validation_accuracy)


def _select_reconstruction_weight(
    noise_type: str, device: torch.device
) -> float:
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
            pilot_rows.append({
                "noise_type": noise_type,
                "reconstruction_weight": reconstruction_weight,
                "best_epoch": result.best_epoch,
                "best_validation_accuracy": result.best_validation_accuracy,
            })
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
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
    except Exception as error:
        raise RuntimeError(f"Checkpoint를 불러오지 못했습니다: {checkpoint_path}") from error
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
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
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
        "test_reconstruction_loss": test_metrics.reconstruction_loss,
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


def _write_pilot_results(
    noise_type: str, pilot_rows: list[dict[str, Any]]
) -> None:
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
    results = results.sort_values(
        ["noise_type", "reconstruction_weight"]
    ).reset_index(drop=True)
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
