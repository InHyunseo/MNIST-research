"""복원 이미지의 숫자 가독성을 평가하는 독립 MNIST 분류기를 학습한다."""

from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST

from .config import PROJECT_ROOT, RAW_DATA_DIR
from .runtime import save_checkpoint_atomically, select_device, set_random_seed


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mnist_classifier"
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.pt"
TRAINING_LOG_PATH = OUTPUT_DIR / "training.csv"
METRICS_PATH = PROJECT_ROOT / "results" / "mnist_classifier" / "metrics.json"

TRAINING_SEED = 0
TRAINING_SAMPLES = 55_000
VALIDATION_SAMPLES = 5_000
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MAXIMUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MINIMUM_DELTA = 2e-4


class MnistClassifier(nn.Module):
    """정규화된 clean MNIST `[B,1,28,28]`을 분류하는 LeNet이다."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """`[B,1,28,28]` image를 받아 `[B,10]` class logit을 반환한다."""
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
            raise ValueError("MNIST 분류기 입력은 `[batch,1,28,28]`이어야 합니다.")
        return self.layers(images)


class IndexedMnistDataset(Dataset):
    """MNIST tensor와 선택 index를 보관하고 image를 접근 시점에 정규화한다."""

    def __init__(self, mnist_dataset: MNIST, indices: torch.Tensor) -> None:
        self.images = mnist_dataset.data
        self.labels = mnist_dataset.targets
        self.indices = indices.to(torch.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        source_index = int(self.indices[index])
        image = self.images[source_index].to(torch.float32).div(255.0).unsqueeze(0)
        return image, self.labels[source_index].to(torch.int64)


def classifier_fingerprint() -> str:
    """모델·split·학습 계약과 checkpoint 호환성을 나타내는 SHA-256 지문이다."""
    contract = {
        "architecture": "lenet-1-6-16-120-84-10",
        "training_seed": TRAINING_SEED,
        "training_samples": TRAINING_SAMPLES,
        "validation_samples": VALIDATION_SAMPLES,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "maximum_epochs": MAXIMUM_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_minimum_delta": EARLY_STOPPING_MINIMUM_DELTA,
    }
    serialized = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def reconstruction_accuracy_per_sample(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """복원된 두 숫자의 unordered exact-match를 sample별로 반환한다.

    Args:
        logits: 두 복원 crop의 class logit `[B,2,10]`.
        labels: source 순서의 class label `[B,2]`.

    Returns:
        두 숫자를 모두 맞히면 1, 하나라도 틀리면 0인 정확도 `[B]`.
    """
    if logits.ndim != 3 or tuple(logits.shape[1:]) != (2, 10):
        raise ValueError("복원 분류 logit은 `[batch,2,10]`이어야 합니다.")
    if tuple(labels.shape) != tuple(logits.shape[:2]):
        raise ValueError("복원 class label은 `[batch,2]`이어야 합니다.")
    predicted_pairs = torch.sort(logits.argmax(dim=-1), dim=1).values
    target_pairs = torch.sort(labels, dim=1).values
    return predicted_pairs.eq(target_pairs).all(dim=1).to(torch.float32)


def load_mnist_classifier(
    device: torch.device,
    checkpoint_path: Path = CHECKPOINT_PATH,
) -> MnistClassifier:
    """호환되는 완료 checkpoint를 읽고 모든 parameter를 고정한다."""
    checkpoint = _load_compatible_checkpoint(checkpoint_path, device)
    model = MnistClassifier().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def run(device_name: str = "cpu", skip_training: bool = False) -> dict[str, float | int]:
    """독립 MNIST 분류기를 학습하거나 재사용하고 clean test 성능을 저장한다."""
    device = select_device(device_name)
    set_random_seed(TRAINING_SEED)
    train_dataset, validation_dataset, test_dataset = _create_datasets(download=True)

    if CHECKPOINT_PATH.exists():
        try:
            model = load_mnist_classifier(device)
            print(f"완료된 MNIST 평가기를 재사용합니다: {CHECKPOINT_PATH}")
        except (KeyError, RuntimeError, ValueError) as error:
            if skip_training:
                raise RuntimeError(
                    "MNIST 평가기 checkpoint가 현재 설정과 호환되지 않습니다."
                ) from error
            model = _train(train_dataset, validation_dataset, device)
    elif skip_training:
        raise FileNotFoundError(
            "MNIST 평가기 checkpoint가 없습니다. 먼저 "
            "`python main.py --model mnist --device cuda`를 실행하세요."
        )
    else:
        model = _train(train_dataset, validation_dataset, device)

    test_loss, test_accuracy = _evaluate(
        model,
        DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
        device,
    )
    checkpoint = _load_compatible_checkpoint(CHECKPOINT_PATH, device)
    metrics: dict[str, float | int] = {
        "training_seed": TRAINING_SEED,
        "best_epoch": int(checkpoint["best_epoch"]),
        "clean_mnist_validation_accuracy": float(checkpoint["validation_accuracy"]),
        "clean_mnist_test_loss": test_loss,
        "clean_mnist_test_accuracy": test_accuracy,
    }
    _save_metrics(metrics)
    print(
        "MNIST 평가기 — "
        f"clean validation={metrics['clean_mnist_validation_accuracy'] * 100:.2f}%, "
        f"clean test={test_accuracy * 100:.2f}%"
    )
    print(f"평가기 checkpoint: {CHECKPOINT_PATH}")
    return metrics


def _create_datasets(
    download: bool,
) -> tuple[IndexedMnistDataset, IndexedMnistDataset, IndexedMnistDataset]:
    """고정 permutation으로 clean MNIST train을 55k/5k로 분리한다."""
    training_source = MNIST(RAW_DATA_DIR, train=True, download=download)
    test_source = MNIST(RAW_DATA_DIR, train=False, download=download)
    generator = torch.Generator().manual_seed(TRAINING_SEED)
    permutation = torch.randperm(len(training_source), generator=generator)
    expected_samples = TRAINING_SAMPLES + VALIDATION_SAMPLES
    if len(permutation) != expected_samples:
        raise ValueError(
            f"MNIST train 표본 수가 예상과 다릅니다: {len(permutation)}"
        )
    training_indices = permutation[:TRAINING_SAMPLES]
    validation_indices = permutation[TRAINING_SAMPLES:]
    test_indices = torch.arange(len(test_source))
    return (
        IndexedMnistDataset(training_source, training_indices),
        IndexedMnistDataset(training_source, validation_indices),
        IndexedMnistDataset(test_source, test_indices),
    )


def _train(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    device: torch.device,
) -> MnistClassifier:
    """Validation accuracy 기준 early stopping으로 평가기를 학습한다."""
    set_random_seed(TRAINING_SEED)
    model = MnistClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(TRAINING_SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    best_accuracy = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    log_rows: list[dict[str, float | int]] = []

    for epoch in range(1, MAXIMUM_EPOCHS + 1):
        train_loss, train_accuracy = _train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        validation_loss, validation_accuracy = _evaluate(
            model, validation_loader, device
        )
        log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        })
        print(
            f"MNIST evaluator epoch={epoch:02d} "
            f"train_acc={train_accuracy:.4f} val_acc={validation_accuracy:.4f}"
        )

        if validation_accuracy > best_accuracy + EARLY_STOPPING_MINIMUM_DELTA:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("MNIST 평가기 best checkpoint를 선택하지 못했습니다.")
    model.load_state_dict(best_state)
    save_checkpoint_atomically(
        {
            "completed": True,
            "fingerprint": classifier_fingerprint(),
            "model_state_dict": best_state,
            "training_seed": TRAINING_SEED,
            "best_epoch": best_epoch,
            "validation_accuracy": best_accuracy,
        },
        CHECKPOINT_PATH,
    )
    _save_training_log(log_rows)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _train_one_epoch(
    model: MnistClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """한 epoch을 학습하고 sample 평균 loss와 accuracy를 반환한다."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = images.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_correct += int(logits.argmax(dim=1).eq(labels).sum())
        total_samples += batch_size
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def _evaluate(
    model: MnistClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Clean MNIST loader의 sample 평균 cross-entropy와 accuracy를 계산한다."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        total_loss += float(criterion(logits, labels))
        total_correct += int(logits.argmax(dim=1).eq(labels).sum())
        total_samples += images.shape[0]
    return total_loss / total_samples, total_correct / total_samples


def _load_compatible_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """완료 여부와 설정 지문을 검증해 checkpoint dictionary를 반환한다."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"MNIST 평가기 checkpoint가 없습니다: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not checkpoint.get("completed", False):
        raise ValueError("MNIST 평가기 checkpoint가 완료 상태가 아닙니다.")
    if checkpoint.get("fingerprint") != classifier_fingerprint():
        raise ValueError("MNIST 평가기 checkpoint 설정 지문이 다릅니다.")
    return checkpoint


def _save_training_log(rows: list[dict[str, float | int]]) -> None:
    """Epoch별 학습 기록을 CSV로 원자 저장한다."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = TRAINING_LOG_PATH.with_suffix(".csv.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(TRAINING_LOG_PATH)


def _save_metrics(metrics: dict[str, float | int]) -> None:
    """Clean MNIST test 결과를 JSON으로 원자 저장한다."""
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = METRICS_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(METRICS_PATH)
