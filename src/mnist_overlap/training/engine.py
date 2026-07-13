"""모델의 epoch 학습, validation, early stopping, checkpoint 처리를 담당한다.

입력:
    PyTorch model, DataLoader, optimizer, 전체 실험 config

출력:
    Epoch metric, best checkpoint, CSV 학습 이력

연결:
    Training 및 evaluation runner가 호출하고 evaluation metric을 사용한다.
"""

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

from ..configuration import config_fingerprint
from ..evaluation.metrics import exact_match_per_sample


@dataclass(frozen=True)
class EpochResult:
    """한 epoch에서 집계한 loss와 exact-match를 보관한다.

    입력:
        Sample-weighted 평균 loss와 exact-match

    처리:
        불변 dataclass로 두 metric을 묶는다.

    출력:
        Training 및 validation에서 공통으로 사용하는 EpochResult
    """

    loss: float
    exact_match: float


@dataclass(frozen=True)
class TrainingResult:
    """한 모델 학습의 best epoch와 생성 경로를 보관한다.

    입력:
        Best epoch, validation exact-match, checkpoint와 history 경로

    처리:
        불변 dataclass로 학습 결과를 묶는다.

    출력:
        실행 함수가 사용자에게 보고할 TrainingResult
    """

    best_epoch: int
    best_validation_exact_match: float
    checkpoint_path: Path
    history_path: Path


def set_random_seed(seed: int) -> None:
    """Python, NumPy, PyTorch의 난수 상태를 같은 seed로 고정한다.

    입력:
        정수 random seed

    처리:
        세 난수 생성기와 PyTorch deterministic algorithm option을 설정한다.

    출력:
        반환값은 없으며 이후 실행의 난수 상태가 고정된다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> EpochResult:
    """DataLoader 전체를 한 번 순회해 모델을 학습한다.

    입력:
        모델, train DataLoader, optimizer, loss 함수, 실행 device

    처리:
        Batch별 forward, backward, optimizer step과 sample-weighted metric을 수행한다.

    출력:
        평균 train loss와 exact-match를 담은 EpochResult
    """
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
    """Parameter를 변경하지 않고 validation loss와 exact-match를 계산한다.

    입력:
        모델, validation DataLoader, loss 함수, 실행 device

    처리:
        Evaluation mode와 no-gradient 상태에서 전체 sample metric을 집계한다.

    출력:
        평균 validation loss와 exact-match를 담은 EpochResult
    """
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
    config: dict[str, Any],
    model_name: str,
    seed: int,
    checkpoint_path: Path,
    history_path: Path,
    device: torch.device | None = None,
) -> TrainingResult:
    """한 모델을 early stopping까지 학습하고 best checkpoint를 저장한다.

    입력:
        모델, train/validation loader, config, 모델 이름, seed, 출력 경로와 device

    처리:
        Adam 학습, epoch CSV 기록, validation 개선 감시, best state 복원을 수행한다.

    출력:
        Best epoch와 결과 경로를 담은 TrainingResult
    """
    set_random_seed(seed)
    selected_device = device or torch.device("cpu")
    model.to(selected_device)
    training_config = config["train"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
    )
    loss_function = nn.BCEWithLogitsLoss(reduction="mean")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

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
        for epoch in range(1, int(training_config["maximum_epochs"]) + 1):
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

            minimum_delta = float(training_config["early_stopping_minimum_delta"])
            if validation_result.exact_match > best_exact_match + minimum_delta:
                best_exact_match = validation_result.exact_match
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save({
                    "model_name": model_name,
                    "seed": seed,
                    "epoch": epoch,
                    "validation_exact_match": best_exact_match,
                    "config_fingerprint": config_fingerprint(config),
                    "model_state_dict": model.state_dict(),
                }, checkpoint_path)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= int(training_config["early_stopping_patience"]):
                break

    if best_epoch == 0:
        raise RuntimeError("학습에서 checkpoint를 생성하지 못했습니다.")
    checkpoint = torch.load(checkpoint_path, map_location=selected_device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return TrainingResult(best_epoch, best_exact_match, checkpoint_path, history_path)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """저장된 checkpoint를 생성된 모델에 복원한다.

    입력:
        모델 instance, checkpoint 경로, device, 선택적인 현재 config

    처리:
        Checkpoint를 읽고 config fingerprint를 확인한 뒤 state dict를 적용한다.

    출력:
        Epoch와 seed metadata가 포함된 checkpoint dictionary
    """
    selected_device = device or torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=selected_device, weights_only=True)
    if config is not None and checkpoint.get("config_fingerprint") != config_fingerprint(config):
        raise ValueError(
            f"다른 config로 생성한 checkpoint입니다: {checkpoint_path}. "
            "--overwrite option으로 다시 학습하세요."
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(selected_device)
    return checkpoint


def checkpoint_matches_config(checkpoint_path: Path, config: dict[str, Any]) -> bool:
    """기존 checkpoint가 현재 학습 config와 호환되는지 확인한다.

    입력:
        Checkpoint 경로와 현재 config

    처리:
        File 존재 여부와 저장된 config fingerprint를 비교한다.

    출력:
        호환되면 True, 없거나 설정이 다르면 False
    """
    if not checkpoint_path.exists():
        return False
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return checkpoint.get("config_fingerprint") == config_fingerprint(config)
