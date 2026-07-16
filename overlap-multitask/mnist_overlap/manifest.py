"""Controlled MNIST-O source split과 좌표 manifest를 결정론적으로 생성한다."""

from __future__ import annotations

from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np
from torchvision.datasets import MNIST

from .config import (
    CLASS_COUNT,
    MANIFEST_DIR,
    OVERLAP_LEVELS,
    RAW_DATA_DIR,
    ExperimentConfig,
    data_config_fingerprint,
)

SOURCE_SPLIT_PATH = MANIFEST_DIR / "source_split.npz"
DATA_FINGERPRINT_PATH = MANIFEST_DIR / "config.sha256"
MANIFEST_PATHS = {
    "train": MANIFEST_DIR / "train.npz",
    "validation": MANIFEST_DIR / "validation.npz",
    "test": MANIFEST_DIR / "test.npz",
}
MANIFEST_FIELD_NAMES = (
    "sample_id",
    "pair_id",
    "source_index_first",
    "source_index_second",
    "label_first",
    "label_second",
    "offset_first_x",
    "offset_first_y",
    "offset_second_x",
    "offset_second_y",
    "displacement_x",
    "displacement_y",
    "bounding_box_overlap_ratio",
    "pixel_overlap_ratio",
    "overlap_level",
)

CANVAS_SIZE = 76
DIGIT_SIZE = 28
SOURCE_TRAIN_SAMPLE_COUNT = 50_000
STROKE_THRESHOLD = 0.5
DIRECTIONS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def prepare_data(config: ExperimentConfig) -> dict[str, Path]:
    """MNIST와 현재 fingerprint의 세 split manifest를 준비한다."""
    paths = {
        "config_fingerprint": DATA_FINGERPRINT_PATH,
        "source_split": SOURCE_SPLIT_PATH,
        **MANIFEST_PATHS,
    }
    if all(path.exists() for path in paths.values()):
        saved_fingerprint = DATA_FINGERPRINT_PATH.read_text(encoding="utf-8").strip()
        if saved_fingerprint == data_config_fingerprint(config):
            return paths
        print("  manifest fingerprint 불일치 → 재생성합니다.")

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    training_source = MNIST(RAW_DATA_DIR, train=True, download=True)
    test_source = MNIST(RAW_DATA_DIR, train=False, download=True)
    random_generator = np.random.default_rng(config.data.seed)

    train_source_indices, validation_source_indices = create_source_split(
        training_source.targets.numpy(),
        SOURCE_TRAIN_SAMPLE_COUNT,
        random_generator,
    )
    np.savez_compressed(
        SOURCE_SPLIT_PATH,
        train_source_indices=train_source_indices,
        validation_source_indices=validation_source_indices,
    )

    manifests = {
        "train": create_training_manifest(
            training_source.data.numpy(),
            training_source.targets.numpy(),
            train_source_indices,
            config,
            random_generator,
        ),
        "validation": create_paired_manifest(
            training_source.data.numpy(),
            training_source.targets.numpy(),
            validation_source_indices,
            config.data.validation_pairs,
            config,
            random_generator,
        ),
        "test": create_paired_manifest(
            test_source.data.numpy(),
            test_source.targets.numpy(),
            np.arange(len(test_source), dtype=np.int32),
            config.data.test_pairs,
            config,
            random_generator,
        ),
    }
    for split_name, manifest in manifests.items():
        save_manifest(MANIFEST_PATHS[split_name], manifest)
    DATA_FINGERPRINT_PATH.write_text(
        data_config_fingerprint(config) + "\n",
        encoding="utf-8",
    )
    return paths


def create_source_split(
    labels: np.ndarray,
    train_sample_count: int,
    random_generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """MNIST train source를 label-stratified train/validation으로 분리한다."""
    if not 0 < train_sample_count < len(labels):
        raise ValueError("train_sample_count는 0보다 크고 원본 수보다 작아야 합니다.")

    class_values, class_counts = np.unique(labels, return_counts=True)
    exact_quotas = class_counts * train_sample_count / len(labels)
    train_quotas = np.floor(exact_quotas).astype(int)
    remaining = train_sample_count - int(train_quotas.sum())
    remainder_order = np.argsort(-(exact_quotas - train_quotas))
    train_quotas[remainder_order[:remaining]] += 1

    train_indices: list[np.ndarray] = []
    validation_indices: list[np.ndarray] = []
    for class_value, train_quota in zip(class_values, train_quotas):
        class_indices = np.flatnonzero(labels == class_value)
        random_generator.shuffle(class_indices)
        train_indices.append(class_indices[:train_quota])
        validation_indices.append(class_indices[train_quota:])

    train_array = np.concatenate(train_indices).astype(np.int32)
    validation_array = np.concatenate(validation_indices).astype(np.int32)
    random_generator.shuffle(train_array)
    random_generator.shuffle(validation_array)
    return train_array, validation_array


def create_training_manifest(
    source_images: np.ndarray,
    source_labels: np.ndarray,
    allowed_source_indices: np.ndarray,
    config: ExperimentConfig,
    random_generator: np.random.Generator,
) -> dict[str, np.ndarray]:
    """45개 class pair와 세 overlap level이 균형인 train manifest를 만든다."""
    class_pairs = list(combinations(range(CLASS_COUNT), 2))
    sampling_cells = list(product(class_pairs, OVERLAP_LEVELS))
    repeated_cells = _repeat_and_shuffle(
        sampling_cells, config.data.train_samples, random_generator
    )
    indices_by_class = _group_source_indices_by_class(
        source_labels, allowed_source_indices
    )
    records = [
        _create_manifest_record(
            source_images,
            indices_by_class,
            class_pair,
            overlap_level,
            DIRECTIONS[sample_id % len(DIRECTIONS)],
            sample_id,
            sample_id,
            config,
            random_generator,
        )
        for sample_id, (class_pair, overlap_level) in enumerate(repeated_cells)
    ]
    random_generator.shuffle(records)
    return _records_to_manifest(records)


def create_paired_manifest(
    source_images: np.ndarray,
    source_labels: np.ndarray,
    allowed_source_indices: np.ndarray,
    pair_count: int,
    config: ExperimentConfig,
    random_generator: np.random.Generator,
) -> dict[str, np.ndarray]:
    """원본과 방향은 같고 overlap만 다른 Low/Middle/High pair를 만든다."""
    class_pairs = list(combinations(range(CLASS_COUNT), 2))
    repeated_pairs = _repeat_and_shuffle(class_pairs, pair_count, random_generator)
    indices_by_class = _group_source_indices_by_class(
        source_labels, allowed_source_indices
    )
    records = []
    for pair_id, class_pair in enumerate(repeated_pairs):
        fixed_sources = (
            int(random_generator.choice(indices_by_class[class_pair[0]])),
            int(random_generator.choice(indices_by_class[class_pair[1]])),
        )
        for level_index, overlap_level in enumerate(OVERLAP_LEVELS):
            records.append(_create_manifest_record(
                source_images,
                indices_by_class,
                class_pair,
                overlap_level,
                DIRECTIONS[pair_id % len(DIRECTIONS)],
                pair_id * len(OVERLAP_LEVELS) + level_index,
                pair_id,
                config,
                random_generator,
                fixed_sources,
            ))
    return _records_to_manifest(records)


def load_manifest(path: str | Path) -> dict[str, np.ndarray]:
    """압축 manifest NPZ를 memory dictionary로 읽는다."""
    with np.load(Path(path), allow_pickle=False) as archive:
        return {field_name: archive[field_name] for field_name in archive.files}


def save_manifest(path: str | Path, manifest: dict[str, np.ndarray]) -> None:
    """Manifest dictionary를 압축 NPZ로 저장한다."""
    np.savez_compressed(Path(path), **manifest)


def bounding_box_overlap_ratio(
    displacement_x: int,
    displacement_y: int,
    digit_size: int,
) -> float:
    """동일 크기 digit box 두 개의 면적 overlap 비율을 계산한다."""
    overlap_width = max(0, digit_size - abs(displacement_x))
    overlap_height = max(0, digit_size - abs(displacement_y))
    return overlap_width * overlap_height / float(digit_size * digit_size)


def _create_manifest_record(
    source_images: np.ndarray,
    source_indices_by_class: dict[int, np.ndarray],
    class_pair: tuple[int, int],
    overlap_level: str,
    direction: tuple[int, int],
    sample_id: int,
    pair_id: int,
    config: ExperimentConfig,
    random_generator: np.random.Generator,
    fixed_sources: tuple[int, int] | None = None,
) -> tuple[Any, ...]:
    """한 합성 sample을 재구성하는 source index와 좌표 record를 만든다."""
    if fixed_sources is None:
        source_index_first = int(
            random_generator.choice(source_indices_by_class[class_pair[0]])
        )
        source_index_second = int(
            random_generator.choice(source_indices_by_class[class_pair[1]])
        )
    else:
        source_index_first, source_index_second = fixed_sources

    displacement_x, displacement_y = _sample_displacement(
        direction,
        config.overlap.bounds(overlap_level),
        random_generator,
    )
    offset_first, offset_second = _centered_offsets(
        displacement_x, displacement_y
    )
    return (
        sample_id,
        pair_id,
        source_index_first,
        source_index_second,
        class_pair[0],
        class_pair[1],
        offset_first[0],
        offset_first[1],
        offset_second[0],
        offset_second[1],
        displacement_x,
        displacement_y,
        bounding_box_overlap_ratio(displacement_x, displacement_y, DIGIT_SIZE),
        _pixel_overlap_ratio(
            source_images[source_index_first],
            source_images[source_index_second],
            offset_first,
            offset_second,
        ),
        overlap_level,
    )


def _sample_displacement(
    direction: tuple[int, int],
    overlap_bounds: tuple[float, float],
    random_generator: np.random.Generator,
) -> tuple[int, int]:
    """지정 방향과 overlap 구간을 만족하는 정수 변위를 뽑는다."""
    candidates = []
    for distance in range(DIGIT_SIZE):
        displacement = (direction[0] * distance, direction[1] * distance)
        overlap_ratio = bounding_box_overlap_ratio(*displacement, DIGIT_SIZE)
        if overlap_bounds[0] <= overlap_ratio <= overlap_bounds[1]:
            candidates.append(displacement)
    if not candidates:
        raise ValueError(
            "조건을 만족하는 정수 변위가 없습니다: "
            f"direction={direction}, bounds={overlap_bounds}"
        )
    return candidates[int(random_generator.integers(len(candidates)))]


def _centered_offsets(
    displacement_x: int,
    displacement_y: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Pair center가 고정되도록 변위를 나눈 두 top-left offset을 계산한다."""
    centered_top_left = (CANVAS_SIZE - DIGIT_SIZE) // 2
    first_x = centered_top_left - displacement_x // 2
    first_y = centered_top_left - displacement_y // 2
    second_x = first_x + displacement_x
    second_y = first_y + displacement_y
    if not all(
        0 <= coordinate <= CANVAS_SIZE - DIGIT_SIZE
        for coordinate in (first_x, first_y, second_x, second_y)
    ):
        raise ValueError("중앙 offset으로 배치한 숫자가 canvas 경계를 벗어납니다.")
    return (first_x, first_y), (second_x, second_y)


def _pixel_overlap_ratio(
    image_first: np.ndarray,
    image_second: np.ndarray,
    offset_first: tuple[int, int],
    offset_second: tuple[int, int],
) -> float:
    """작은 digit stroke mask 기준 실제 pixel overlap 비율을 계산한다."""
    threshold = STROKE_THRESHOLD * 255.0
    mask_first = image_first > threshold
    mask_second = image_second > threshold
    denominator = min(int(mask_first.sum()), int(mask_second.sum()))
    if denominator == 0:
        return 0.0

    first_x, first_y = offset_first
    second_x, second_y = offset_second
    overlap_left = max(first_x, second_x)
    overlap_top = max(first_y, second_y)
    overlap_right = min(first_x + DIGIT_SIZE, second_x + DIGIT_SIZE)
    overlap_bottom = min(first_y + DIGIT_SIZE, second_y + DIGIT_SIZE)
    if overlap_left >= overlap_right or overlap_top >= overlap_bottom:
        return 0.0

    first_slice = mask_first[
        overlap_top - first_y:overlap_bottom - first_y,
        overlap_left - first_x:overlap_right - first_x,
    ]
    second_slice = mask_second[
        overlap_top - second_y:overlap_bottom - second_y,
        overlap_left - second_x:overlap_right - second_x,
    ]
    return float(np.logical_and(first_slice, second_slice).sum() / denominator)


def _group_source_indices_by_class(
    labels: np.ndarray,
    allowed_source_indices: np.ndarray,
) -> dict[int, np.ndarray]:
    """허용된 source index를 class별로 묶는다."""
    grouped = {}
    for class_value in np.unique(labels):
        class_indices = allowed_source_indices[
            labels[allowed_source_indices] == class_value
        ]
        if not len(class_indices):
            raise ValueError(f"Source split에 class {class_value} sample이 없습니다.")
        grouped[int(class_value)] = class_indices
    return grouped


def _repeat_and_shuffle(
    values: list[Any],
    requested_count: int,
    random_generator: np.random.Generator,
) -> list[Any]:
    """빈도 차이가 1 이하가 되도록 반복한 값 목록을 섞는다."""
    repeated = [values[index % len(values)] for index in range(requested_count)]
    random_generator.shuffle(repeated)
    return repeated


def _records_to_manifest(records: list[tuple[Any, ...]]) -> dict[str, np.ndarray]:
    """Record tuple 목록을 field별 typed NumPy array로 변환한다."""
    columns = zip(*records)
    integer_fields = set(MANIFEST_FIELD_NAMES[:12])
    manifest = {}
    for field_name, column in zip(MANIFEST_FIELD_NAMES, columns):
        if field_name in integer_fields:
            data_type = np.int32
        elif field_name == "overlap_level":
            data_type = "<U6"
        else:
            data_type = np.float32
        manifest[field_name] = np.asarray(column, dtype=data_type)
    return manifest
