"""LeNet bottleneck에 class-conditioned compact decoder를 결합한다."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as functional

from ..config import CLASS_COUNT
from ..data import RECONSTRUCTION_SIZE
from ..model import MnistONet


CLASS_LATENT_DIMENSION = 32
BOTTLENECK_FEATURE_COUNT = 16 * 16 * 16
DECODER_HIDDEN_DIMENSIONS = (512, 1024)


@dataclass(frozen=True)
class MultitaskOutput:
    """분류 logit, 두 source 복원 logit과 복원에 사용한 class index."""

    logits: torch.Tensor
    reconstruction_logits: torch.Tensor
    reconstruction_classes: torch.Tensor


class ClassLatentEncoder(nn.Module):
    """공유 LeNet bottleneck을 class별 compact latent 10개로 투영한다."""

    def __init__(self, latent_dimension: int = CLASS_LATENT_DIMENSION) -> None:
        super().__init__()
        self.latent_dimension = latent_dimension
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(BOTTLENECK_FEATURE_COUNT, CLASS_COUNT * latent_dimension),
            nn.ReLU(),
        )

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """`[B,16,16,16]` feature를 `[B,10,D]` latent로 변환한다."""
        if tuple(bottleneck.shape[1:]) != (16, 16, 16):
            raise ValueError("LeNet bottleneck은 `[batch,16,16,16]` 형태여야 합니다.")
        projected = self.projection(bottleneck)
        return projected.reshape(-1, CLASS_COUNT, self.latent_dimension)


class SharedReconstructionDecoder(nn.Module):
    """동일한 MLP로 class latent 두 개를 각각 `64×64` source로 복원한다."""

    def __init__(self, latent_dimension: int = CLASS_LATENT_DIMENSION) -> None:
        super().__init__()
        decoder_input_dimension = latent_dimension + CLASS_COUNT
        self.layers = nn.Sequential(
            nn.Linear(decoder_input_dimension, DECODER_HIDDEN_DIMENSIONS[0]),
            nn.ReLU(),
            nn.Linear(
                DECODER_HIDDEN_DIMENSIONS[0],
                DECODER_HIDDEN_DIMENSIONS[1],
            ),
            nn.ReLU(),
            nn.Linear(
                DECODER_HIDDEN_DIMENSIONS[1],
                RECONSTRUCTION_SIZE * RECONSTRUCTION_SIZE,
            ),
        )

    def forward(
        self,
        selected_latents: torch.Tensor,
        class_indices: torch.Tensor,
    ) -> torch.Tensor:
        """`[B,2,D]` latent와 class identity로 `[B,2,64,64]` logit을 만든다."""
        batch_size, source_count, latent_dimension = selected_latents.shape
        if source_count != 2 or tuple(class_indices.shape) != (batch_size, 2):
            raise ValueError("선택 latent와 class index는 `[batch,2,...]` 형태여야 합니다.")
        class_conditions = functional.one_hot(
            class_indices,
            num_classes=CLASS_COUNT,
        ).to(dtype=selected_latents.dtype)
        decoder_inputs = torch.cat((selected_latents, class_conditions), dim=-1)
        flat_logits = self.layers(
            decoder_inputs.reshape(batch_size * source_count, latent_dimension + CLASS_COUNT)
        )
        return flat_logits.reshape(
            batch_size,
            source_count,
            RECONSTRUCTION_SIZE,
            RECONSTRUCTION_SIZE,
        )


class ClassConditionedReconstructionHead(nn.Module):
    """Class별 latent를 고른 뒤 하나의 decoder로 두 source를 복원한다."""

    def __init__(self) -> None:
        super().__init__()
        self.latent_encoder = ClassLatentEncoder()
        self.decoder = SharedReconstructionDecoder()

    def forward(
        self,
        bottleneck: torch.Tensor,
        class_indices: torch.Tensor,
    ) -> torch.Tensor:
        """요청된 두 class의 latent만 선택해 source 순서대로 복원한다."""
        class_latents = self.latent_encoder(bottleneck)
        _validate_class_indices(class_indices, bottleneck.shape[0])
        batch_indices = torch.arange(
            bottleneck.shape[0],
            device=bottleneck.device,
        ).unsqueeze(1)
        selected_latents = class_latents[batch_indices, class_indices]
        return self.decoder(selected_latents, class_indices)


class MultitaskMnistONet(nn.Module):
    """LeNet encoder를 분류 head와 compact reconstruction head가 공유한다."""

    def __init__(self) -> None:
        super().__init__()
        # 동일 seed baseline과 초기값을 맞추기 위해 classifier를 반드시 먼저 생성한다.
        self.classifier = MnistONet()
        self.reconstruction_head = ClassConditionedReconstructionHead()

    def forward(
        self,
        images: torch.Tensor,
        reconstruction_classes: torch.Tensor | None = None,
    ) -> MultitaskOutput:
        """겹친 입력을 분류하고 지정된 두 class의 원본을 복원한다.

        학습·복원 평가에서는 source 정답 class를 전달한다. 생략하면 classifier의
        Top-2 prediction을 사용하므로 label이 없는 추론에서도 동작한다.
        """
        bottleneck = self.classifier.encode(images)
        logits = self.classifier.classify_features(bottleneck)
        if reconstruction_classes is None:
            reconstruction_classes = torch.topk(logits, k=2, dim=1).indices
        reconstruction_classes = reconstruction_classes.to(device=images.device)
        _validate_class_indices(reconstruction_classes, images.shape[0])
        reconstruction_classes = reconstruction_classes.to(dtype=torch.int64)
        reconstruction_logits = self.reconstruction_head(
            bottleneck,
            reconstruction_classes,
        )
        return MultitaskOutput(
            logits,
            reconstruction_logits,
            reconstruction_classes,
        )


def _validate_class_indices(class_indices: torch.Tensor, batch_size: int) -> None:
    """복원 class가 서로 다른 두 정수 class인지 검사한다."""
    if tuple(class_indices.shape) != (batch_size, 2):
        raise ValueError("복원 class index는 `[batch,2]` 형태여야 합니다.")
    if class_indices.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise ValueError("복원 class index는 정수 tensor여야 합니다.")
    if torch.any(class_indices < 0) or torch.any(class_indices >= CLASS_COUNT):
        raise ValueError("복원 class index는 0 이상 10 미만이어야 합니다.")
    if torch.any(class_indices[:, 0] == class_indices[:, 1]):
        raise ValueError("한 sample의 두 복원 class는 서로 달라야 합니다.")
