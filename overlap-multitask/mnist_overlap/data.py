"""MNIST-O manifest를 실제 합성 image와 학습 sample로 변환한다."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST

from .config import CLASS_COUNT, RAW_DATA_DIR
from .manifest import CANVAS_SIZE, MANIFEST_PATHS, load_manifest, prepare_data

RECONSTRUCTION_SIZE = 64
RECONSTRUCTION_CROP_START = (CANVAS_SIZE - RECONSTRUCTION_SIZE) // 2

__all__ = (
    "ControlledOverlapMnistDataset",
    "RECONSTRUCTION_CROP_START",
    "RECONSTRUCTION_SIZE",
    "prepare_data",
    "render_overlap_sample",
    "render_semantic_targets",
)


def render_overlap_sample(
    source_image_first: torch.Tensor,
    source_image_second: torch.Tensor,
    offset_first: tuple[int, int],
    offset_second: tuple[int, int],
    canvas_size: int = CANVAS_SIZE,
) -> torch.Tensor:
    """두 원본 숫자를 canvas에 배치하고 pixel별 산술평균으로 합성한다."""
    canvas_first = torch.zeros((canvas_size, canvas_size), dtype=torch.float32)
    canvas_second = torch.zeros_like(canvas_first)
    _place_image(canvas_first, source_image_first, offset_first)
    _place_image(canvas_second, source_image_second, offset_second)
    return 0.5 * (canvas_first + canvas_second)


def render_semantic_targets(
    source_image_first: torch.Tensor,
    source_image_second: torch.Tensor,
    offset_first: tuple[int, int],
    offset_second: tuple[int, int],
    label_first: int,
    label_second: int,
) -> torch.Tensor:
    """두 source를 해당 class channel에 배치한 `[10,64,64]` target을 만든다."""
    if label_first == label_second:
        raise ValueError("Semantic target은 서로 다른 두 class를 전제로 합니다.")
    for offset, source_image in (
        (offset_first, source_image_first),
        (offset_second, source_image_second),
    ):
        shifted_x = offset[0] - RECONSTRUCTION_CROP_START
        shifted_y = offset[1] - RECONSTRUCTION_CROP_START
        if not (
            0 <= shifted_x <= RECONSTRUCTION_SIZE - source_image.shape[1]
            and 0 <= shifted_y <= RECONSTRUCTION_SIZE - source_image.shape[0]
        ):
            raise ValueError("Source image가 reconstruction target crop을 벗어납니다.")
    source_layers = torch.zeros((2, CANVAS_SIZE, CANVAS_SIZE), dtype=torch.float32)
    _place_image(source_layers[0], source_image_first, offset_first)
    _place_image(source_layers[1], source_image_second, offset_second)
    crop_start = RECONSTRUCTION_CROP_START
    crop_end = crop_start + RECONSTRUCTION_SIZE
    cropped_layers = source_layers[:, crop_start:crop_end, crop_start:crop_end]
    if tuple(cropped_layers.shape) != (2, RECONSTRUCTION_SIZE, RECONSTRUCTION_SIZE):
        raise RuntimeError("Reconstruction target crop shape가 올바르지 않습니다.")
    semantic_targets = torch.zeros(
        (CLASS_COUNT, RECONSTRUCTION_SIZE, RECONSTRUCTION_SIZE),
        dtype=torch.float32,
    )
    semantic_targets[label_first] = cropped_layers[0]
    semantic_targets[label_second] = cropped_layers[1]
    return semantic_targets


def _place_image(
    canvas: torch.Tensor,
    image: torch.Tensor,
    offset: tuple[int, int],
) -> None:
    """원본 image를 canvas의 지정 top-left 위치에 복사한다."""
    offset_x, offset_y = offset
    image_height, image_width = image.shape
    canvas[offset_y:offset_y + image_height, offset_x:offset_x + image_width] = image


class ControlledOverlapMnistDataset(Dataset):
    """Manifest 좌표로 동일한 합성 sample을 지연 재구성한다.

    `include_source_images=True`일 때만 원본 `[2,28,28]`, class별 semantic target
    `[10,64,64]`와 target 안의 top-left 좌표 `[2,2]`를 추가한다.
    """

    def __init__(
        self,
        split_name: str,
        manifest_path: Path | None = None,
        download: bool = False,
        include_source_images: bool = False,
    ) -> None:
        if split_name not in MANIFEST_PATHS:
            raise ValueError(f"지원하지 않는 데이터 split입니다: {split_name}")

        self.manifest = load_manifest(manifest_path or MANIFEST_PATHS[split_name])
        self.mnist_dataset = MNIST(
            RAW_DATA_DIR,
            train=split_name != "test",
            download=download,
        )
        self.include_source_images = include_source_images

    def __len__(self) -> int:
        return len(self.manifest["sample_id"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_image_first = self._source_image(index, "first")
        source_image_second = self._source_image(index, "second")
        offset_first = self._offset(index, "first")
        offset_second = self._offset(index, "second")
        label_first = int(self.manifest["label_first"][index])
        label_second = int(self.manifest["label_second"][index])
        multi_hot_label = torch.zeros(CLASS_COUNT, dtype=torch.float32)
        multi_hot_label[[label_first, label_second]] = 1.0

        sample: dict[str, Any] = {
            "image": render_overlap_sample(
                source_image_first,
                source_image_second,
                offset_first,
                offset_second,
            ).unsqueeze(0),
            "label": multi_hot_label,
            "label_first": label_first,
            "label_second": label_second,
            "sample_id": int(self.manifest["sample_id"][index]),
            "pair_id": int(self.manifest["pair_id"][index]),
            "overlap_level": str(self.manifest["overlap_level"][index]),
            "bounding_box_overlap_ratio": float(
                self.manifest["bounding_box_overlap_ratio"][index]
            ),
            "pixel_overlap_ratio": float(self.manifest["pixel_overlap_ratio"][index]),
        }
        if self.include_source_images:
            sample["source_images"] = torch.stack(
                (source_image_first, source_image_second)
            )
            sample["reconstruction_targets"] = render_semantic_targets(
                source_image_first,
                source_image_second,
                offset_first,
                offset_second,
                label_first,
                label_second,
            )
            sample["source_offsets"] = torch.tensor(
                (
                    (
                        offset_first[0] - RECONSTRUCTION_CROP_START,
                        offset_first[1] - RECONSTRUCTION_CROP_START,
                    ),
                    (
                        offset_second[0] - RECONSTRUCTION_CROP_START,
                        offset_second[1] - RECONSTRUCTION_CROP_START,
                    ),
                ),
                dtype=torch.int64,
            )
        return sample

    def _source_image(self, index: int, position: str) -> torch.Tensor:
        """Manifest의 first/second source index를 정규화한 image로 읽는다."""
        source_index = int(self.manifest[f"source_index_{position}"][index])
        return self.mnist_dataset.data[source_index].to(torch.float32).div(255.0)

    def _offset(self, index: int, position: str) -> tuple[int, int]:
        """Manifest의 first/second top-left 좌표를 정수 tuple로 읽는다."""
        return (
            int(self.manifest[f"offset_{position}_x"][index]),
            int(self.manifest[f"offset_{position}_y"][index]),
        )
