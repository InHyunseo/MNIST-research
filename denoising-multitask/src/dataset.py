"""
n-MNIST와 같은 순서의 clean MNIST를 준비하고 DataLoader를 생성한다.

입력:
    - LSU n-MNIST archive 또는 추출된 MAT 파일
    - DeepLearnToolbox clean MNIST MAT 파일

출력:
    - noisy image, clean target, class label을 반환하는 DataLoader

주요 기능:
    1. archive 최초 1회 추출과 clean dataset 다운로드
    2. MAT 배열 로딩과 32×32 padding
    3. 재현 가능한 training/validation 분리
"""

from __future__ import annotations

import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
RAW_DATA_DIRECTORY = PROJECT_DIRECTORY / "data" / "raw"
CLEAN_DATA_DIRECTORY = PROJECT_DIRECTORY / "data" / "mnist"
CLEAN_DATA_PATH = CLEAN_DATA_DIRECTORY / "mnist_uint8.mat"
CLEAN_DATA_URL = (
    "https://raw.githubusercontent.com/rasmusbergpalm/DeepLearnToolbox/"
    "5df2801f2196a2afddb7a87f800e63e153c34995/data/mnist_uint8.mat"
)

NOISE_TYPES = ("awgn", "motion_blur", "reduced_contrast_awgn")
NOISE_FILES = {
    "awgn": ("mnist-with-awgn.gz", "awgn.mat"),
    "motion_blur": ("mnist-with-motion-blur.gz", "motion_blur.mat"),
    "reduced_contrast_awgn": (
        "mnist-with-reduced-contrast-and-awgn.gz",
        "reduced_contrast_awgn.mat",
    ),
}


@dataclass(frozen=True)
class MnistArrays:
    """한 noise version의 noisy 입력, clean target과 label 배열이다."""

    training_noisy_images: np.ndarray
    training_clean_images: np.ndarray
    training_labels: np.ndarray
    test_noisy_images: np.ndarray
    test_clean_images: np.ndarray
    test_labels: np.ndarray


class NoisyMnistDataset(Dataset):
    """선택된 index의 noisy image, clean target과 class label을 반환한다."""

    def __init__(
        self,
        noisy_images: np.ndarray,
        clean_images: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.noisy_images = noisy_images
        self.clean_images = clean_images
        self.labels = labels
        self.indices = indices

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        source_index = int(self.indices[item])
        noisy_image = _image_to_tensor(self.noisy_images[source_index])
        clean_target = _image_to_tensor(self.clean_images[source_index])
        label = torch.tensor(int(self.labels[source_index]), dtype=torch.long)
        return {
            "noisy_image": noisy_image,
            "clean_target": clean_target,
            "label": label,
        }


def prepare_data() -> None:
    """n-MNIST archive를 추출하고 순서가 일치하는 clean MAT를 준비한다."""
    RAW_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    CLEAN_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for archive_name, mat_name in NOISE_FILES.values():
        archive_path = RAW_DATA_DIRECTORY / archive_name
        mat_path = RAW_DATA_DIRECTORY / mat_name
        if mat_path.exists():
            print(f"기존 n-MNIST 파일을 사용합니다: {mat_path}")
            continue
        if not archive_path.exists():
            raise FileNotFoundError(
                f"n-MNIST archive가 없습니다: {archive_path}\n"
                "다운로드한 archive를 data/raw/에 배치하세요."
            )
        _extract_mat_file(archive_path, mat_path)
        print(f"n-MNIST 파일을 추출했습니다: {mat_path}")

    if CLEAN_DATA_PATH.exists():
        print(f"기존 clean MNIST를 사용합니다: {CLEAN_DATA_PATH}")
    else:
        _download_clean_data()
        print(f"clean MNIST를 다운로드했습니다: {CLEAN_DATA_PATH}")


def create_data_loaders(
    noise_type: str,
    batch_size: int,
    validation_ratio: float,
    random_seed: int,
    use_pinned_memory: bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """지정한 noise와 seed에 대한 training, validation, test loader를 만든다."""
    arrays = load_mnist_arrays(noise_type)
    sample_count = arrays.training_labels.shape[0]
    validation_count = int(sample_count * validation_ratio)
    split_generator = torch.Generator().manual_seed(random_seed)
    shuffled_indices = torch.randperm(sample_count, generator=split_generator).numpy()
    validation_indices = shuffled_indices[:validation_count]
    training_indices = shuffled_indices[validation_count:]
    test_indices = np.arange(arrays.test_labels.shape[0], dtype=np.int64)

    training_dataset = NoisyMnistDataset(
        arrays.training_noisy_images,
        arrays.training_clean_images,
        arrays.training_labels,
        training_indices,
    )
    validation_dataset = NoisyMnistDataset(
        arrays.training_noisy_images,
        arrays.training_clean_images,
        arrays.training_labels,
        validation_indices,
    )
    test_dataset = NoisyMnistDataset(
        arrays.test_noisy_images,
        arrays.test_clean_images,
        arrays.test_labels,
        test_indices,
    )
    training_generator = torch.Generator().manual_seed(random_seed)
    common_arguments = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": use_pinned_memory,
    }
    return (
        DataLoader(
            training_dataset,
            shuffle=True,
            generator=training_generator,
            **common_arguments,
        ),
        DataLoader(validation_dataset, shuffle=False, **common_arguments),
        DataLoader(test_dataset, shuffle=False, **common_arguments),
    )


@lru_cache(maxsize=len(NOISE_TYPES))
def load_mnist_arrays(noise_type: str) -> MnistArrays:
    """한 noise version과 clean target의 MAT 배열을 메모리에 로딩한다."""
    if noise_type not in NOISE_FILES:
        raise ValueError(f"지원하지 않는 noise type입니다: {noise_type}")
    noisy_path = RAW_DATA_DIRECTORY / NOISE_FILES[noise_type][1]
    missing_paths = [path for path in (noisy_path, CLEAN_DATA_PATH) if not path.exists()]
    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            f"준비되지 않은 dataset 파일입니다: {missing_text}\n"
            "먼저 `python main.py data`를 실행하세요."
        )

    noisy_data = loadmat(noisy_path)
    clean_data = _load_clean_data()
    training_labels = np.argmax(noisy_data["train_y"], axis=1).astype(np.int64)
    test_labels = np.argmax(noisy_data["test_y"], axis=1).astype(np.int64)
    return MnistArrays(
        training_noisy_images=noisy_data["train_x"],
        training_clean_images=clean_data["train_x"],
        training_labels=training_labels,
        test_noisy_images=noisy_data["test_x"],
        test_clean_images=clean_data["test_x"],
        test_labels=test_labels,
    )


@lru_cache(maxsize=1)
def _load_clean_data() -> dict[str, np.ndarray]:
    """모든 noise version이 공유하는 clean MAT를 한 번만 로딩한다."""
    return loadmat(CLEAN_DATA_PATH)


def _image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """784-vector uint8 이미지를 정규화하고 32×32로 zero-padding한다."""
    tensor = torch.from_numpy(image.reshape(28, 28)).to(torch.float32).div_(255.0)
    return functional.pad(tensor.unsqueeze(0), (2, 2, 2, 2))


def _extract_mat_file(archive_path: Path, destination_path: Path) -> None:
    """Archive의 단일 MAT member를 임시 파일을 거쳐 안전하게 추출한다."""
    temporary_path = destination_path.with_suffix(destination_path.suffix + ".tmp")
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            mat_members = [
                member
                for member in archive.getmembers()
                if member.isfile() and Path(member.name).suffix.lower() == ".mat"
            ]
            if len(mat_members) != 1:
                raise RuntimeError(
                    "Archive에는 MAT 파일이 정확히 하나 있어야 합니다: "
                    f"{archive_path}"
                )
            source = archive.extractfile(mat_members[0])
            if source is None:
                raise RuntimeError(f"MAT member를 읽을 수 없습니다: {archive_path}")
            with source, temporary_path.open("wb") as destination:
                shutil.copyfileobj(source, destination)
        temporary_path.replace(destination_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _download_clean_data() -> None:
    """고정된 DeepLearnToolbox revision의 clean MAT를 원자적으로 다운로드한다."""
    temporary_path = CLEAN_DATA_PATH.with_suffix(CLEAN_DATA_PATH.suffix + ".tmp")
    try:
        with urllib.request.urlopen(CLEAN_DATA_URL, timeout=60) as response:
            with temporary_path.open("wb") as destination:
                shutil.copyfileobj(response, destination)
        temporary_path.replace(CLEAN_DATA_PATH)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
