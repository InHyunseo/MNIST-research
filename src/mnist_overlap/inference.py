"""Checkpoint 하나로 단일 이미지를 추론하는 경량 API.

배포(예: ROS2 wrapper node)에서는 이 모듈의 `load_model`과 `predict`만 import하면 된다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig, load_config
from .metrics import top_two_predictions
from .models import create_model


def load_model(
    checkpoint_path: str | Path,
    device: str = "cpu",
    config: ExperimentConfig | None = None,
) -> torch.nn.Module:
    """Checkpoint에 저장된 `model_name`으로 아키텍처를 만들어 weight를 복원한다."""
    if config is None:
        config = load_config()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = create_model(str(checkpoint["model_name"]), config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    image: np.ndarray | torch.Tensor,
) -> tuple[int, int]:
    """`[H, W]` float 이미지([0, 1] 범위)에서 겹친 두 숫자를 오름차순으로 반환한다."""
    tensor = torch.as_tensor(image, dtype=torch.float32)
    device = next(model.parameters()).device
    logits = model(tensor.reshape(1, 1, *tensor.shape[-2:]).to(device))
    digits = torch.nonzero(top_two_predictions(logits)[0]).flatten()
    return int(digits[0]), int(digits[1])
