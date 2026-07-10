"""Deterministic fixed-slot sequences composed from the MNIST source splits."""

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST

from .config import DATA_DIR, TEST_IMAGES_PATH, TEST_LABELS_PATH, canvas_shape


class StaticSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Compose three centered MNIST digits on a fixed 32 x 96 canvas."""

    def __init__(
        self,
        source: MNIST,
        sample_count: int,
        sequence_length: int,
        slot_size: int,
        seed: int,
    ) -> None:
        if sequence_length != 3 or slot_size != 32:
            raise ValueError("StaticSequenceDataset requires three 32-pixel slots")
        self.images = source.data
        self.labels = source.targets.to(torch.int64)
        self.sample_count = sample_count
        self.sequence_length = sequence_length
        self.slot_size = slot_size

        generator = torch.Generator().manual_seed(seed)
        self.source_indices = torch.randint(
            len(source), (sample_count, sequence_length), generator=generator
        )

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        indices = self.source_indices[index]
        canvas = torch.zeros(
            (1, self.slot_size, self.slot_size * self.sequence_length), dtype=torch.uint8
        )
        offset = (self.slot_size - self.images.shape[-1]) // 2
        for position, source_index in enumerate(indices):
            x_start = position * self.slot_size + offset
            canvas[0, offset:offset + 28, x_start:x_start + 28] = self.images[source_index]
        return canvas, self.labels[indices].clone()


def create_dataset(cfg: dict[str, Any], train: bool) -> StaticSequenceDataset:
    dataset_cfg = cfg["dataset"]
    source = MNIST(DATA_DIR, train=train, download=True)
    return StaticSequenceDataset(
        source=source,
        sample_count=dataset_cfg["train_samples" if train else "test_samples"],
        sequence_length=dataset_cfg["sequence_length"],
        slot_size=dataset_cfg["slot_size"],
        seed=dataset_cfg["train_seed" if train else "test_seed"],
    )


def create_loaders(
    cfg: dict[str, Any], train_batch_size: int | None = None
) -> tuple[DataLoader, DataLoader]:
    train_cfg = cfg["train"]
    eval_cfg = cfg["evaluation"]
    shuffle_generator = torch.Generator().manual_seed(train_cfg["seed"])
    train_loader = DataLoader(
        create_dataset(cfg, train=True),
        batch_size=train_batch_size or train_cfg["batch_size"],
        shuffle=True,
        generator=shuffle_generator,
        num_workers=0,
    )
    test_loader = DataLoader(
        create_dataset(cfg, train=False),
        batch_size=eval_cfg["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    return train_loader, test_loader


def load_test_tensors(cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = create_dataset(cfg, train=False)
    loader = DataLoader(dataset, batch_size=cfg["evaluation"]["batch_size"], shuffle=False)
    images, labels = zip(*loader)
    return torch.cat(images), torch.cat(labels)


def dump_test_data(cfg: dict[str, Any]) -> tuple[torch.Size, torch.Size]:
    images, labels = load_test_tensors(cfg)
    expected_height, expected_width = canvas_shape(cfg)
    if tuple(images.shape[1:]) != (1, expected_height, expected_width):
        raise RuntimeError(f"unexpected image shape: {tuple(images.shape)}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TEST_IMAGES_PATH.write_bytes(images.contiguous().numpy().tobytes())
    TEST_LABELS_PATH.write_bytes(labels.to(torch.uint8).contiguous().numpy().tobytes())
    return images.shape, labels.shape
