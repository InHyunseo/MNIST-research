"""
Baseline과 multitask의 pilot, 학습, checkpoint, 평가와 CSV 저장을 담당한다.

입력:
    - main.py에서 선택한 baseline, multitask 또는 edge multitask 실행
    - 검증된 CPU 또는 CUDA device

출력:
    - Best checkpoint, epoch history, pilot 및 최종 결과 CSV

주요 기능:
    1. 고정 실험 설정과 난수 재현
    2. Classification 및 reconstruction 학습·평가
    3. 노이즈별 loss weight pilot과 최종 반복 실험
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
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from src.dataset import NOISE_TYPES, PROJECT_DIRECTORY, create_data_loaders
from src.model import DenoisingAuxiliaryLeNet


RANDOM_SEEDS = tuple(range(30))
EDGE_RANDOM_SEEDS = tuple(range(10))
PILOT_SEED = 0
RECONSTRUCTION_WEIGHT_CANDIDATES = (0.05, 0.1, 0.2)
EDGE_WEIGHT_CANDIDATES = (0.0, 0.05, 0.1, 0.2)
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
    "edge_weight",
    "best_epoch",
    "best_validation_accuracy",
    "test_classification_loss",
    "test_accuracy",
)
PILOT_RESULT_COLUMNS = (
    "condition",
    "noise_type",
    "reconstruction_weight",
    "edge_weight",
    "best_epoch",
    "best_validation_accuracy",
    "selected",
)
LEGACY_RESULT_COLUMNS = tuple(
    column for column in RESULT_COLUMNS if column != "edge_weight"
)
LEGACY_PILOT_RESULT_COLUMNS = (
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
    edge_weight: float = 0.0
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
    edge_loss: float | None
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


def run_edge_experiments(device: torch.device) -> None:
    """Noise별 loss weight pilot 후 10개 seed의 MSE+Edge 실험을 실행한다."""
    _create_output_directories()
    selected_weights = {
        noise_type: _select_edge_weights(noise_type, device)
        for noise_type in NOISE_TYPES
    }
    for noise_type in NOISE_TYPES:
        reconstruction_weight, edge_weight = selected_weights[noise_type]
        for random_seed in EDGE_RANDOM_SEEDS:
            configuration = ExperimentConfiguration(
                noise_type=noise_type,
                condition="multitask_edge",
                random_seed=random_seed,
                reconstruction_weight=reconstruction_weight,
                edge_weight=edge_weight,
            )
            _run_final_experiment(configuration, device)


def _run_epoch(
    model: DenoisingAuxiliaryLeNet,
    data_loader: DataLoader,
    device: torch.device,
    reconstruction_weight: float,
    edge_weight: float = 0.0,
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
    edge_loss_sum = 0.0
    total_correct = 0
    total_samples = 0
    reconstruction_enabled = (
        (is_training or include_reconstruction) and model.decoder is not None
    )

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
            edge_loss = None
            total_loss = classification_loss
            if reconstruction_enabled:
                reconstruction = output["reconstruction"]
                if reconstruction is None:
                    raise RuntimeError("Multitask 모델이 reconstruction을 반환하지 않았습니다.")
                reconstruction_loss = reconstruction_loss_function(
                    reconstruction, clean_targets
                )
                auxiliary_loss = reconstruction_loss
                if edge_weight > 0.0:
                    edge_loss = _edge_loss(reconstruction, clean_targets)
                    auxiliary_loss = auxiliary_loss + edge_weight * edge_loss
                total_loss = (
                    classification_loss + reconstruction_weight * auxiliary_loss
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
            classification_loss_sum += float(classification_loss.item()) * current_batch_size
            if reconstruction_loss is not None:
                reconstruction_loss_sum += (
                    float(reconstruction_loss.item()) * current_batch_size
                )
            if edge_loss is not None:
                edge_loss_sum += float(edge_loss.item()) * current_batch_size
            predictions = output["classification_logits"].argmax(dim=1)
            total_correct += int((predictions == labels).sum().item())

    return EpochMetrics(
        total_loss=total_loss_sum / total_samples,
        classification_loss=classification_loss_sum / total_samples,
        reconstruction_loss=(
            reconstruction_loss_sum / total_samples if reconstruction_enabled else None
        ),
        edge_loss=(
            edge_loss_sum / total_samples
            if reconstruction_enabled and edge_weight > 0.0
            else None
        ),
        accuracy=total_correct / total_samples,
    )


def _edge_loss(
    reconstruction: torch.Tensor, clean_target: torch.Tensor
) -> torch.Tensor:
    """가로·세로 인접 픽셀의 1차 차분을 L1으로 비교한다."""
    horizontal_loss = functional.l1_loss(
        torch.diff(reconstruction, dim=-1),
        torch.diff(clean_target, dim=-1),
    )
    vertical_loss = functional.l1_loss(
        torch.diff(reconstruction, dim=-2),
        torch.diff(clean_target, dim=-2),
    )
    return 0.5 * (horizontal_loss + vertical_loss)


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
        use_decoder=configuration.condition in {"multitask", "multitask_edge"}
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

    test_metrics = _run_epoch(
        model,
        test_loader,
        device,
        configuration.reconstruction_weight,
        configuration.edge_weight,
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
    phase = "pilot" if checkpoint_path.stem.startswith("pilot_") else "final"
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
            "training_edge_loss",
            "training_accuracy",
            "validation_total_loss",
            "validation_classification_loss",
            "validation_reconstruction_loss",
            "validation_edge_loss",
            "validation_accuracy",
        ])
        for epoch in range(1, configuration.maximum_epochs + 1):
            training_metrics = _run_epoch(
                model,
                training_loader,
                device,
                configuration.reconstruction_weight,
                configuration.edge_weight,
                optimizer=optimizer,
            )
            validation_metrics = _run_epoch(
                model,
                validation_loader,
                device,
                configuration.reconstruction_weight,
                configuration.edge_weight,
                include_reconstruction=True,
            )
            writer.writerow([
                epoch,
                f"{training_metrics.total_loss:.8f}",
                f"{training_metrics.classification_loss:.8f}",
                _format_optional_metric(training_metrics.reconstruction_loss),
                _format_optional_metric(training_metrics.edge_loss),
                f"{training_metrics.accuracy:.8f}",
                f"{validation_metrics.total_loss:.8f}",
                f"{validation_metrics.classification_loss:.8f}",
                _format_optional_metric(validation_metrics.reconstruction_loss),
                _format_optional_metric(validation_metrics.edge_loss),
                f"{validation_metrics.accuracy:.8f}",
            ])
            history_file.flush()
            print(
                f"phase={phase} noise={configuration.noise_type} "
                f"condition={configuration.condition} "
                f"reconstruction_weight={configuration.reconstruction_weight:g} "
                f"edge_weight={configuration.edge_weight:g} "
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
                "condition": "multitask",
                "noise_type": noise_type,
                "reconstruction_weight": reconstruction_weight,
                "edge_weight": 0.0,
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


def _select_edge_weights(
    noise_type: str, device: torch.device
) -> tuple[float, float]:
    """Seed 0 joint pilot으로 noise별 reconstruction·edge weight를 고른다."""
    existing_selection = _read_edge_pilot_selection(noise_type)
    if existing_selection is not None:
        reconstruction_weight, edge_weight = existing_selection
        print(
            f"기존 Edge pilot 결과를 사용합니다: noise={noise_type} "
            f"reconstruction_weight={reconstruction_weight:g} "
            f"edge_weight={edge_weight:g}"
        )
        return existing_selection

    pilot_rows = []
    for reconstruction_weight in RECONSTRUCTION_WEIGHT_CANDIDATES:
        for edge_weight in EDGE_WEIGHT_CANDIDATES:
            reused_row = (
                _read_existing_mse_pilot_row(noise_type, reconstruction_weight)
                if edge_weight == 0.0
                else None
            )
            if reused_row is not None:
                pilot_rows.append({
                    "condition": "multitask_edge",
                    "noise_type": noise_type,
                    "reconstruction_weight": reconstruction_weight,
                    "edge_weight": edge_weight,
                    "best_epoch": int(reused_row["best_epoch"]),
                    "best_validation_accuracy": float(
                        reused_row["best_validation_accuracy"]
                    ),
                })
                continue

            configuration = ExperimentConfiguration(
                noise_type=noise_type,
                condition="multitask_edge",
                random_seed=PILOT_SEED,
                reconstruction_weight=reconstruction_weight,
                edge_weight=edge_weight,
            )
            checkpoint_path = CHECKPOINT_DIRECTORY / (
                f"pilot_edge_{noise_type}_weight_{reconstruction_weight:g}_"
                f"edge_{edge_weight:g}.pt"
            )
            history_path = HISTORY_DIRECTORY / (
                f"pilot_edge_{noise_type}_weight_{reconstruction_weight:g}_"
                f"edge_{edge_weight:g}.csv"
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
                    "condition": "multitask_edge",
                    "noise_type": noise_type,
                    "reconstruction_weight": reconstruction_weight,
                    "edge_weight": edge_weight,
                    "best_epoch": result.best_epoch,
                    "best_validation_accuracy": result.best_validation_accuracy,
                })
            finally:
                checkpoint_path.unlink(missing_ok=True)
                checkpoint_path.with_suffix(
                    checkpoint_path.suffix + ".tmp"
                ).unlink(missing_ok=True)
                history_path.unlink(missing_ok=True)

    selected_row = max(
        pilot_rows,
        key=lambda row: (
            float(row["best_validation_accuracy"]),
            -float(row["edge_weight"]),
            -float(row["reconstruction_weight"]),
        ),
    )
    selected_weights = (
        float(selected_row["reconstruction_weight"]),
        float(selected_row["edge_weight"]),
    )
    for row in pilot_rows:
        row["selected"] = (
            float(row["reconstruction_weight"]),
            float(row["edge_weight"]),
        ) == selected_weights
    _write_pilot_results(noise_type, pilot_rows)
    print(
        f"Edge pilot 선택: noise={noise_type} "
        f"reconstruction_weight={selected_weights[0]:g} "
        f"edge_weight={selected_weights[1]:g}"
    )
    return selected_weights


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
    if not _checkpoint_configuration_matches(checkpoint, configuration):
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
        _checkpoint_configuration_matches(checkpoint, configuration)
        and checkpoint.get("training_complete") is True
    )


def _checkpoint_configuration_matches(
    checkpoint: dict[str, Any], configuration: ExperimentConfiguration
) -> bool:
    """Edge 도입 전 checkpoint의 weight 0 설정까지 호환해 비교한다."""
    stored_configuration = checkpoint.get("configuration")
    if not isinstance(stored_configuration, dict):
        return False
    stored_configuration = dict(stored_configuration)
    stored_configuration.setdefault("edge_weight", 0.0)
    return stored_configuration == asdict(configuration)


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
        "edge_weight": configuration.edge_weight,
        "best_epoch": training_result.best_epoch,
        "best_validation_accuracy": training_result.best_validation_accuracy,
        "test_classification_loss": test_metrics.classification_loss,
        "test_accuracy": test_metrics.accuracy,
    }
    if RESULTS_PATH.exists():
        results = _read_results()
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
    results = _read_pilot_results()
    noise_results = results.loc[
        (results["noise_type"] == noise_type)
        & (results["condition"] == "multitask")
        & (results["edge_weight"] == 0.0)
    ]
    if set(noise_results["reconstruction_weight"].astype(float)) != set(
        RECONSTRUCTION_WEIGHT_CANDIDATES
    ):
        return None
    selected = noise_results.loc[noise_results["selected"].astype(bool)]
    if len(selected) != 1:
        return None
    return float(selected.iloc[0]["reconstruction_weight"])


def _read_edge_pilot_selection(
    noise_type: str,
) -> tuple[float, float] | None:
    """완전한 기존 joint pilot 결과가 있으면 선택된 두 weight를 반환한다."""
    if not PILOT_RESULTS_PATH.exists():
        return None
    results = _read_pilot_results()
    noise_results = results.loc[
        (results["noise_type"] == noise_type)
        & (results["condition"] == "multitask_edge")
    ]
    expected_pairs = {
        (reconstruction_weight, edge_weight)
        for reconstruction_weight in RECONSTRUCTION_WEIGHT_CANDIDATES
        for edge_weight in EDGE_WEIGHT_CANDIDATES
    }
    available_pairs = {
        (float(row.reconstruction_weight), float(row.edge_weight))
        for row in noise_results.itertuples(index=False)
    }
    if (
        available_pairs != expected_pairs
        or len(noise_results) != len(expected_pairs)
    ):
        return None
    selected = noise_results.loc[noise_results["selected"].astype(bool)]
    if len(selected) != 1:
        return None
    return (
        float(selected.iloc[0]["reconstruction_weight"]),
        float(selected.iloc[0]["edge_weight"]),
    )


def _read_existing_mse_pilot_row(
    noise_type: str, reconstruction_weight: float
) -> dict[str, Any] | None:
    """Edge weight 0과 동일한 기존 MSE pilot 결과 한 행을 반환한다."""
    if not PILOT_RESULTS_PATH.exists():
        return None
    results = _read_pilot_results()
    matching = results.loc[
        (results["noise_type"] == noise_type)
        & (results["condition"] == "multitask")
        & (results["reconstruction_weight"] == reconstruction_weight)
        & (results["edge_weight"] == 0.0)
    ]
    if len(matching) != 1:
        return None
    return matching.iloc[0].to_dict()


def _write_pilot_results(
    noise_type: str, pilot_rows: list[dict[str, Any]]
) -> None:
    """한 noise의 후보별 validation 결과와 선택 여부를 pilot_results.csv에 기록한다."""
    if not pilot_rows:
        raise ValueError("저장할 pilot 결과가 없습니다.")
    condition = str(pilot_rows[0]["condition"])
    if PILOT_RESULTS_PATH.exists():
        results = _read_pilot_results()
        results = results.loc[
            ~(
                (results["noise_type"] == noise_type)
                & (results["condition"] == condition)
            )
        ]
        results = pd.concat([results, pd.DataFrame(pilot_rows)], ignore_index=True)
    else:
        results = pd.DataFrame(pilot_rows, columns=PILOT_RESULT_COLUMNS)
    results = results.sort_values(
        ["condition", "noise_type", "reconstruction_weight", "edge_weight"]
    ).reset_index(drop=True)
    _write_dataframe_atomically(results, PILOT_RESULTS_PATH)


def _read_results() -> pd.DataFrame:
    """현재 또는 Edge 도입 전 schema의 final result를 읽는다."""
    results = pd.read_csv(RESULTS_PATH)
    if tuple(results.columns) == LEGACY_RESULT_COLUMNS:
        results.insert(4, "edge_weight", 0.0)
    if tuple(results.columns) != RESULT_COLUMNS:
        raise ValueError(f"results.csv schema가 올바르지 않습니다: {RESULTS_PATH}")
    return results


def _read_pilot_results() -> pd.DataFrame:
    """현재 또는 Edge 도입 전 schema의 pilot result를 읽는다."""
    results = pd.read_csv(PILOT_RESULTS_PATH)
    if tuple(results.columns) == LEGACY_PILOT_RESULT_COLUMNS:
        results.insert(0, "condition", "multitask")
        results.insert(3, "edge_weight", 0.0)
    if tuple(results.columns) != PILOT_RESULT_COLUMNS:
        raise ValueError(
            f"pilot_results.csv schema가 올바르지 않습니다: {PILOT_RESULTS_PATH}"
        )
    return results


def _write_dataframe_atomically(dataframe: pd.DataFrame, path: Path) -> None:
    """DataFrame을 임시 CSV에 쓴 뒤 목표 경로로 원자 교체한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    dataframe.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def _format_optional_metric(metric: float | None) -> str:
    """계산하지 않은 reconstruction metric은 빈 CSV field로 기록한다."""
    return "" if metric is None else f"{metric:.8f}"
