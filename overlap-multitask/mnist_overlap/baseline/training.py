"""MNIST-O 학습 루프, early stopping, checkpoint와 전체 seed 실행을 담당한다."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..config import (
    CHECKPOINT_DIR,
    DEFAULT_CONFIG_PATH,
    TRAINING_LOG_DIR,
    ExperimentConfig,
    config_fingerprint,
    create_output_directories,
    load_config,
)
from ..data import ControlledOverlapMnistDataset
from ..metrics import exact_match_per_sample
from ..model import MnistONet
from ..runtime import (
    create_train_loader,
    save_checkpoint_atomically,
    select_device,
    set_random_seed,
)


@dataclass(frozen=True)
class EpochResult:
    """한 epoch의 sample-weighted 평균 loss와 exact-match."""

    loss: float
    exact_match: float


@dataclass(frozen=True)
class TrainingResult:
    """한 seed 학습의 best epoch와 생성 경로."""

    best_epoch: int
    epochs_run: int
    best_validation_exact_match: float
    checkpoint_path: Path
    history_path: Path


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
    seed: int,
    checkpoint_path: Path,
    history_path: Path,
    device: torch.device,
) -> TrainingResult:
    """한 seed의 모델을 early stopping까지 학습하고 best checkpoint를 저장한다."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    loss_function = nn.BCEWithLogitsLoss()
    best_exact_match = -1.0
    best_epoch = 0
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
        for epoch in range(1, config.training.maximum_epochs + 1):
            train_result = train_one_epoch(
                model, train_loader, optimizer, loss_function, device
            )
            validation_result = evaluate_validation(
                model, validation_loader, loss_function, device
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
                f"seed={seed} epoch={epoch} train_loss={train_result.loss:.4f} "
                f"validation_exact={validation_result.exact_match:.4f}"
            )

            improved = (
                validation_result.exact_match
                > best_exact_match + config.training.early_stopping_minimum_delta
            )
            if improved:
                best_exact_match = validation_result.exact_match
                best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint_atomically({
                    "seed": seed,
                    "best_epoch": epoch,
                    "validation_exact_match": best_exact_match,
                    "config_fingerprint": config_fingerprint(config),
                    "training_complete": False,
                    "model_state_dict": model.state_dict(),
                }, checkpoint_path)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= config.training.early_stopping_patience:
                break

    if best_epoch == 0:
        raise RuntimeError("학습에서 checkpoint를 생성하지 못했습니다.")
    epochs_run = epoch
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    checkpoint["epochs_run"] = epochs_run
    checkpoint["training_complete"] = True
    save_checkpoint_atomically(checkpoint, checkpoint_path)
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
    device: torch.device,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """호환성과 정상 종료 여부를 확인하고 checkpoint를 모델에 복원한다."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("config_fingerprint") != config_fingerprint(config):
        raise ValueError(f"현재 config와 다른 checkpoint입니다: {checkpoint_path}")
    if checkpoint.get("training_complete") is not True:
        raise ValueError(f"정상 종료되지 않은 checkpoint입니다: {checkpoint_path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return checkpoint


def checkpoint_matches_config(checkpoint_path: Path, config: ExperimentConfig) -> bool:
    """Checkpoint가 현재 설정으로 정상 완료됐는지 확인한다."""
    if not checkpoint_path.exists():
        return False
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return (
        checkpoint.get("config_fingerprint") == config_fingerprint(config)
        and checkpoint.get("training_complete") is True
    )


def train_all_seeds(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    device_name: str = "cpu",
) -> list[TrainingResult]:
    """설정에 지정된 모든 seed를 학습하고 완료된 호환 checkpoint는 재사용한다."""
    config = load_config(config_path)
    create_output_directories()
    device = select_device(device_name)
    train_dataset = ControlledOverlapMnistDataset("train")
    validation_dataset = ControlledOverlapMnistDataset("validation")
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
    )

    results = []
    for seed in config.training.seeds:
        checkpoint_path = CHECKPOINT_DIR / f"seed_{seed}.pt"
        history_path = TRAINING_LOG_DIR / f"seed_{seed}.csv"
        if checkpoint_matches_config(checkpoint_path, config):
            print(f"기존 checkpoint를 사용합니다: {checkpoint_path}")
            continue

        _remove_run_artifacts(seed)
        set_random_seed(seed)
        model = MnistONet()
        result = train_model(
            model,
            create_train_loader(train_dataset, config.training, seed),
            validation_loader,
            config,
            seed,
            checkpoint_path,
            history_path,
            device,
        )
        results.append(result)
    return results


def _remove_run_artifacts(seed: int) -> None:
    """한 seed의 checkpoint와 학습 이력을 제거한다."""
    for artifact_path in (
        CHECKPOINT_DIR / f"seed_{seed}.pt",
        CHECKPOINT_DIR / f"seed_{seed}.pt.tmp",
        TRAINING_LOG_DIR / f"seed_{seed}.csv",
    ):
        artifact_path.unlink(missing_ok=True)
