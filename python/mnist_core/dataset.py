"""MNIST 로드 + raw uint8 덤프 (C++가 파이썬과 동일 바이트를 읽도록)."""
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import DATA_DIR

IMAGES_U8 = DATA_DIR / "mnist_test_images.u8"
LABELS_U8 = DATA_DIR / "mnist_test_labels.u8"


def load_test_tensors():
    ds = datasets.MNIST(DATA_DIR, train=False, download=True,
                        transform=transforms.PILToTensor())
    images = torch.stack([i for i, _ in ds])   # uint8 [M,1,28,28]
    labels = torch.tensor([y for _, y in ds])
    return images, labels


def loaders(batch_size):
    tf = transforms.PILToTensor()
    tr = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
    te = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    return (DataLoader(tr, batch_size=batch_size, shuffle=True),
            DataLoader(te, batch_size=1000))


def dump_u8():
    images, labels = load_test_tensors()
    IMAGES_U8.write_bytes(images.numpy().astype(np.uint8).tobytes())
    LABELS_U8.write_bytes(labels.numpy().astype(np.uint8).tobytes())
    return images.shape, labels.shape
