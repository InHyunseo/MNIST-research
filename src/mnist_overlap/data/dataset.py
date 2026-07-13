"""저장된 manifest에서 합성 image와 label을 지연 재구성하는 Dataset을 제공한다.

입력:
    Split manifest, 원본 MNIST image, 전체 config

출력:
    Image, multi-hot label, pair metadata와 선택적인 stroke mask sample

연결:
    Training, evaluation, reporting이 사용하며 manifest와 합성 함수는 generation에
    위임한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST

from ..configuration import RAW_DATA_DIR
from .generation import MANIFEST_PATHS, load_manifest, render_overlap_sample


class ControlledOverlapMnistDataset(Dataset):
    """Manifest 좌표를 사용해 동일한 합성 sample을 필요할 때 재구성한다.

    입력:
        Split 이름, 전체 config, mask 반환 여부, 선택적인 manifest 경로

    처리:
        원본 MNIST image 두 개를 고정 offset에 배치하고 multi-hot label을 만든다.

    출력:
        Image, label, pair metadata와 선택적인 source별 mask dictionary
    """

    def __init__(
        self,
        split_name: str,
        config: dict[str, Any],
        include_masks: bool = False,
        manifest_path: Path | None = None,
        download: bool = False,
    ) -> None:
        """Split에 대응하는 manifest와 MNIST 원본을 준비한다.

        입력:
            Split 이름, config, mask option, manifest 경로, download option

        처리:
            Manifest를 memory에 읽고 train/test에 맞는 torchvision MNIST를 연다.

        출력:
            초기화된 Dataset instance
        """
        if split_name not in MANIFEST_PATHS:
            raise ValueError(f"지원하지 않는 데이터 split입니다: {split_name}")

        self.split_name = split_name
        self.dataset_config = config["dataset"]
        self.model_config = config["model"]
        self.include_masks = include_masks
        self.manifest = load_manifest(
            manifest_path or MANIFEST_PATHS[split_name]
        )
        self.mnist_dataset = MNIST(
            RAW_DATA_DIR,
            train=split_name != "test",
            download=download,
        )

    def __len__(self) -> int:
        """현재 split의 합성 sample 수를 반환한다.

        입력:
            초기화된 Dataset의 manifest

        처리:
            `sample_id` field 길이를 확인한다.

        출력:
            정수 sample 수
        """
        return len(self.manifest["sample_id"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        """지정 index의 합성 image와 학습 metadata를 생성한다.

        입력:
            0 이상 Dataset 길이 미만의 sample index

        처리:
            두 원본 image를 읽어 max 합성하고 label 및 선택적인 mask를 구성한다.

        출력:
            Image, multi-hot label, pair metadata dictionary
        """
        source_image_first = self.mnist_dataset.data[
            int(self.manifest["source_index_first"][index])
        ].to(torch.float32).div(255.0)
        source_image_second = self.mnist_dataset.data[
            int(self.manifest["source_index_second"][index])
        ].to(torch.float32).div(255.0)
        offset_first = (
            int(self.manifest["offset_first_x"][index]),
            int(self.manifest["offset_first_y"][index]),
        )
        offset_second = (
            int(self.manifest["offset_second_x"][index]),
            int(self.manifest["offset_second_y"][index]),
        )
        rendered = render_overlap_sample(
            source_image_first,
            source_image_second,
            offset_first,
            offset_second,
            canvas_size=int(self.dataset_config["canvas_size"]),
            stroke_threshold=float(self.dataset_config["stroke_threshold"]),
        )
        label_first = int(self.manifest["label_first"][index])
        label_second = int(self.manifest["label_second"][index])
        multi_hot_label = torch.zeros(
            int(self.model_config["class_count"]),
            dtype=torch.float32,
        )
        multi_hot_label[label_first] = 1.0
        multi_hot_label[label_second] = 1.0
        sample: dict[str, Any] = {
            "image": rendered["image"].unsqueeze(0),
            "label": multi_hot_label,
            "label_first": label_first,
            "label_second": label_second,
            "sample_id": int(self.manifest["sample_id"][index]),
            "pair_id": int(self.manifest["pair_id"][index]),
            "overlap_level": str(self.manifest["overlap_level"][index]),
            "bounding_box_overlap_ratio": float(
                self.manifest["bounding_box_overlap_ratio"][index]
            ),
            "pixel_overlap_ratio": float(
                self.manifest["pixel_overlap_ratio"][index]
            ),
        }

        if self.include_masks:
            sample.update(
                {
                    name: rendered[name].unsqueeze(0)
                    for name in (
                        "mask_first",
                        "mask_second",
                        "exclusive_mask_first",
                        "exclusive_mask_second",
                    )
                }
            )

        return sample
