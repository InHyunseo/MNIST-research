"""모델 학습: epoch loop, early stopping, checkpoint 관리와 model×seed 실행 조정."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import (
    ATTENTION_LOG_DIR,
    CHECKPOINT_DIR,
    DEFAULT_CONFIG_PATH,
    PREDICTION_LOG_DIR,
    TRAINING_LOG_DIR,
    ExperimentConfig,
    TrainConfig,
    config_fingerprint,
    create_output_directories,
    load_config,
    update_experiment_metadata,
)
from .data import ControlledOverlapMnistDataset
from .metrics import exact_match_per_sample
from .models import create_model


@dataclass(frozen=True)
class EpochResult:
    """한 epoch의 sample-weighted 평균 loss와 exact-match."""

    loss: float
    exact_match: float


@dataclass(frozen=True)
class TrainingResult:
    """한 모델 학습의 best epoch와 생성 경로."""

    best_epoch: int
    epochs_run: int
    best_validation_exact_match: float
    checkpoint_path: Path
    history_path: Path


def set_random_seed(seed: int) -> None:
    """Python, NumPy, PyTorch 난수 상태를 고정하고 deterministic algorithm을 강제한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


# -----------------------------------------------------------------------------
# 실행 대상 선택 (evaluation과 공유)
# -----------------------------------------------------------------------------


def select_device(device_name: str) -> torch.device:
    """요청한 device 문자열을 검증해 `torch.device`로 변환한다."""
    if device_name not in ("cpu", "cuda"):
        raise ValueError("Device는 'cpu' 또는 'cuda'여야 합니다.")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 요청했지만 현재 환경에서 사용할 수 없습니다.")

    return torch.device(device_name)


def select_model_names(
    config: ExperimentConfig,
    model_name: str | None,
) -> list[str]:
    """단일 model option 또는 config의 전체 model 목록을 선택한다."""
    configured_names = list(config.model.model_names)

    if model_name is None:
        return configured_names

    if model_name not in configured_names:
        raise ValueError(f"Config에 없는 모델입니다: {model_name}")

    return [model_name]


def select_seeds(config: ExperimentConfig, seed: int | None) -> list[int]:
    """단일 seed option 또는 config의 전체 seed 목록을 선택한다."""
    if seed is not None:
        return [int(seed)]

    return list(config.project.training_seeds)


# -----------------------------------------------------------------------------
# 학습 engine
# -----------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> EpochResult:
    """DataLoader 전체를 한 번 순회해 모델을 학습한다."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch in data_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_function(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"유한하지 않은 training loss입니다: {loss.item()}")
        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_correct += int(exact_match_per_sample(logits.detach(), labels).sum().item())
        total_samples += batch_size
    return EpochResult(total_loss / total_samples, total_correct / total_samples)


@torch.no_grad()
def evaluate_validation(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> EpochResult:
    """Parameter를 변경하지 않고 validation loss와 exact-match를 계산한다."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch in data_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = model(images)
        loss = loss_function(logits, labels)
        batch_size = images.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_correct += int(exact_match_per_sample(logits, labels).sum().item())
        total_samples += batch_size
    return EpochResult(total_loss / total_samples, total_correct / total_samples)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: ExperimentConfig,
    model_name: str,
    seed: int,
    checkpoint_path: Path,
    history_path: Path,
    device: torch.device | None = None,
) -> TrainingResult:
    """한 모델을 early stopping까지 학습하고 best checkpoint를 저장한다.

    Checkpoint는 임시 파일을 거쳐 원자적으로 교체되며, 학습이 정상 종료될 때만
    `training_complete=True`로 표시된다.
    """
    set_random_seed(seed)
    selected_device = device or torch.device("cpu")
    model.to(selected_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)
    loss_function = nn.BCEWithLogitsLoss(reduction="mean")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    best_exact_match = -1.0
    best_epoch = 0
    epochs_run = 0
    epochs_without_improvement = 0
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.writer(history_file)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_exact_match",
            "validation_loss",
            "validation_exact_match",
        ])
        for epoch in range(1, config.train.maximum_epochs + 1):
            epochs_run = epoch
            train_result = train_one_epoch(
                model, train_loader, optimizer, loss_function, selected_device
            )
            validation_result = evaluate_validation(
                model, validation_loader, loss_function, selected_device
            )
            writer.writerow([
                epoch,
                f"{train_result.loss:.8f}",
                f"{train_result.exact_match:.8f}",
                f"{validation_result.loss:.8f}",
                f"{validation_result.exact_match:.8f}",
            ])
            history_file.flush()
            print(
                f"model={model_name} seed={seed} epoch={epoch} "
                f"train_loss={train_result.loss:.4f} "
                f"validation_exact={validation_result.exact_match:.4f}"
            )

            minimum_delta = config.train.early_stopping_minimum_delta
            if validation_result.exact_match > best_exact_match + minimum_delta:
                best_exact_match = validation_result.exact_match
                best_epoch = epoch
                epochs_without_improvement = 0
                _save_checkpoint_atomically({
                    "model_name": model_name,
                    "seed": seed,
                    "best_epoch": epoch,
                    "epochs_run": epoch,
                    "validation_exact_match": best_exact_match,
                    "config_fingerprint": config_fingerprint(config),
                    "training_complete": False,
                    "model_state_dict": model.state_dict(),
                }, checkpoint_path)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= config.train.early_stopping_patience:
                break

    if best_epoch == 0:
        raise RuntimeError("학습에서 checkpoint를 생성하지 못했습니다.")
    checkpoint = torch.load(checkpoint_path, map_location=selected_device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    checkpoint["best_epoch"] = best_epoch
    checkpoint["epochs_run"] = epochs_run
    checkpoint["training_complete"] = True
    _save_checkpoint_atomically(checkpoint, checkpoint_path)
    return TrainingResult(
        best_epoch,
        epochs_run,
        best_exact_match,
        checkpoint_path,
        history_path,
    )


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device | None = None,
    config: ExperimentConfig | None = None,
) -> dict[str, Any]:
    """저장된 checkpoint를 모델에 복원한다.

    Config를 주면 fingerprint 일치와 정상 학습 종료 여부를 함께 검사한다.
    """
    selected_device = device or torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=selected_device, weights_only=True)
    if config is not None and checkpoint.get("config_fingerprint") != config_fingerprint(config):
        raise ValueError(
            f"다른 config로 생성한 checkpoint입니다: {checkpoint_path}. "
            "--overwrite option으로 다시 학습하세요."
        )
    if checkpoint.get("training_complete") is not True:
        raise ValueError(
            f"정상 종료되지 않은 checkpoint입니다: {checkpoint_path}. "
            "학습 command를 다시 실행하세요."
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(selected_device)
    return checkpoint


def checkpoint_matches_config(checkpoint_path: Path, config: ExperimentConfig) -> bool:
    """기존 checkpoint가 현재 학습 config로 정상 완료된 것인지 확인한다."""
    if not checkpoint_path.exists():
        return False
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return (
        checkpoint.get("config_fingerprint") == config_fingerprint(config)
        and checkpoint.get("training_complete") is True
    )


def _save_checkpoint_atomically(
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
) -> None:
    """Checkpoint를 임시 파일에 쓴 뒤 목표 경로로 원자 교체한다."""
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(checkpoint_path)


# -----------------------------------------------------------------------------
# 학습 실행 조정
# -----------------------------------------------------------------------------


def train_models(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    model_name: str | None = None,
    seed: int | None = None,
    device_name: str = "cpu",
    overwrite: bool = False,
) -> list[TrainingResult]:
    """Config에 지정된 모델×seed 조합(또는 선택한 단일 조합)을 학습한다.

    완료된 호환 checkpoint는 재사용하고, 새로 학습하는 run은 이전 history와
    평가 cache를 함께 무효화한다.
    """
    config = load_config(config_path)
    create_output_directories()
    device = select_device(device_name)
    model_names = select_model_names(config, model_name)
    seeds = select_seeds(config, seed)

    if overwrite:
        _remove_selected_run_artifacts(model_names, seeds)

    update_experiment_metadata(config, device_name)

    train_dataset = ControlledOverlapMnistDataset("train", config)
    validation_dataset = ControlledOverlapMnistDataset("validation", config)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.data_loader_workers,
    )

    training_results = []

    for selected_seed in seeds:
        for selected_model_name in model_names:
            checkpoint_path = (
                CHECKPOINT_DIR / f"{selected_model_name}_seed_{selected_seed}.pt"
            )
            history_path = (
                TRAINING_LOG_DIR / f"{selected_model_name}_seed_{selected_seed}.csv"
            )

            if checkpoint_path.exists():
                if checkpoint_matches_config(checkpoint_path, config):
                    print(f"기존 checkpoint를 사용합니다: {checkpoint_path}")
                    continue

                checkpoint = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )
                same_config = (
                    checkpoint.get("config_fingerprint")
                    == config_fingerprint(config)
                )
                if not same_config:
                    raise RuntimeError(
                        f"현재 config와 다른 checkpoint입니다: {checkpoint_path}. "
                        "--overwrite option으로 다시 학습하세요."
                    )

                print(
                    "중단된 학습을 처음부터 다시 시작합니다: "
                    f"model={selected_model_name}, seed={selected_seed}"
                )

            # 새로 학습하는 run은 이전 history와 평가 cache를 함께 무효화한다.
            if not overwrite:
                _remove_run_artifacts(selected_model_name, selected_seed)
            set_random_seed(selected_seed)
            train_loader = _create_train_loader(
                train_dataset,
                config.train,
                selected_seed,
            )
            model = create_model(selected_model_name, config)
            result = train_model(
                model=model,
                train_loader=train_loader,
                validation_loader=validation_loader,
                config=config,
                model_name=selected_model_name,
                seed=selected_seed,
                checkpoint_path=checkpoint_path,
                history_path=history_path,
                device=device,
            )
            training_results.append(result)
            update_experiment_metadata(config, device_name)

    return training_results


def _remove_selected_run_artifacts(
    model_names: list[str],
    seeds: list[int],
) -> None:
    """Overwrite 대상 전체 run의 기존 artifact를 학습 시작 전에 정리한다."""
    for selected_seed in seeds:
        for selected_model_name in model_names:
            _remove_run_artifacts(selected_model_name, selected_seed)


def _remove_run_artifacts(model_name: str, seed: int) -> None:
    """한 model·seed의 checkpoint, history, 평가 cache를 제거한다."""
    artifact_paths = (
        CHECKPOINT_DIR / f"{model_name}_seed_{seed}.pt",
        CHECKPOINT_DIR / f"{model_name}_seed_{seed}.pt.tmp",
        TRAINING_LOG_DIR / f"{model_name}_seed_{seed}.csv",
        PREDICTION_LOG_DIR / f"{model_name}_seed_{seed}_test.npz",
        PREDICTION_LOG_DIR / f"{model_name}_seed_{seed}_test.npz.tmp.npz",
        ATTENTION_LOG_DIR / f"{model_name}_seed_{seed}_test.npz",
        ATTENTION_LOG_DIR / f"{model_name}_seed_{seed}_test.npz.tmp.npz",
    )

    for artifact_path in artifact_paths:
        artifact_path.unlink(missing_ok=True)


def _create_train_loader(
    train_dataset: ControlledOverlapMnistDataset,
    train_config: TrainConfig,
    seed: int,
) -> torch.utils.data.DataLoader:
    """모델 간 동일한 batch 순서를 재현하는 train DataLoader를 생성한다."""
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.data_loader_workers,
        generator=generator,
    )
