"""Baseline과 multitask 실험이 공유하는 난수·device·DataLoader runtime 도구다."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import TrainingConfig


def set_random_seed(seed: int) -> None:
    """Python·NumPy·PyTorch 난수를 고정하고 결정론 알고리즘을 강제한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def select_device(device_name: str) -> torch.device:
    """요청한 CPU 또는 CUDA device를 검증해 반환한다."""
    if device_name not in ("cpu", "cuda"):
        raise ValueError("Device는 'cpu' 또는 'cuda'여야 합니다.")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 요청했지만 현재 환경에서 사용할 수 없습니다.")
    return torch.device(device_name)


def create_train_loader(
    train_dataset: Dataset,
    training_config: TrainingConfig,
    seed: int,
) -> DataLoader:
    """고정 seed로 재현 가능한 shuffled training DataLoader를 만든다."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )


def save_checkpoint_atomically(checkpoint: dict[str, Any], path: Path) -> None:
    """Checkpoint를 임시 파일에 쓴 뒤 목표 경로로 원자 교체한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)
