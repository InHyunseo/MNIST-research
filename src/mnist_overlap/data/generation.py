"""Controlled Overlap MNIST manifest를 생성·검증하고 image를 합성한다.

입력:
    MNIST 원본 tensor, YAML data/overlap 설정, 저장된 manifest

출력:
    Source-disjoint manifest와 합성 image/mask 및 무결성 검사 결과

연결:
    Data 공개 API와 Dataset이 호출하며 Dataset class 자체는 dataset 모듈에 둔다.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchvision.datasets import MNIST

from ..configuration import (
    MANIFEST_DIR,
    OVERLAP_LEVELS,
    RAW_DATA_DIR,
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


# -----------------------------------------------------------------------------
# 데이터 준비
# -----------------------------------------------------------------------------


def prepare_data(config: dict[str, Any], overwrite: bool = False) -> dict[str, Path]:
    """MNIST를 내려받고 세 split의 고정 manifest를 생성한다.

    입력:
        전체 config와 기존 manifest 덮어쓰기 여부

    처리:
        Source-disjoint 분할 후 train 및 paired validation/test metadata를 만든다.

    출력:
        생성하거나 재사용한 fingerprint와 manifest 경로 dictionary
    """
    paths = {
        "config_fingerprint": DATA_FINGERPRINT_PATH,
        "source_split": SOURCE_SPLIT_PATH,
        **MANIFEST_PATHS,
    }
    if not overwrite and all(path.exists() for path in paths.values()):
        saved_fingerprint = DATA_FINGERPRINT_PATH.read_text(encoding="utf-8").strip()
        if saved_fingerprint == data_config_fingerprint(config):
            return paths
        raise ValueError(
            "저장된 manifest의 data config가 다릅니다. "
            "`python -m mnist_overlap prepare-data --overwrite`를 실행하세요."
        )

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    training_source = MNIST(RAW_DATA_DIR, train=True, download=True)
    test_source = MNIST(RAW_DATA_DIR, train=False, download=True)
    random_generator = np.random.default_rng(int(config["project"]["data_seed"]))

    # 원본 index부터 분리해 합성 sample 사이의 source 누수를 막는다.
    train_source_indices, validation_source_indices = create_source_split(
        training_source.targets.numpy(),
        int(config["dataset"]["source_train_samples"]),
        random_generator,
    )
    np.savez_compressed(
        SOURCE_SPLIT_PATH,
        train_source_indices=train_source_indices,
        validation_source_indices=validation_source_indices,
    )

    train_manifest = create_training_manifest(
        training_source.data.numpy(),
        training_source.targets.numpy(),
        train_source_indices,
        config,
        random_generator,
    )
    validation_manifest = create_paired_manifest(
        training_source.data.numpy(),
        training_source.targets.numpy(),
        validation_source_indices,
        int(config["dataset"]["validation_pairs"]),
        config,
        random_generator,
    )
    test_manifest = create_paired_manifest(
        test_source.data.numpy(),
        test_source.targets.numpy(),
        np.arange(len(test_source), dtype=np.int32),
        int(config["dataset"]["test_pairs"]),
        config,
        random_generator,
    )
    save_manifest(MANIFEST_PATHS["train"], train_manifest)
    save_manifest(MANIFEST_PATHS["validation"], validation_manifest)
    save_manifest(MANIFEST_PATHS["test"], test_manifest)
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
    """MNIST train 원본 index를 label-stratified 방식으로 분리한다.

    입력:
        원본 label, train source 수, NumPy random generator

    처리:
        Class별 비율 quota를 계산하고 각 class index를 독립적으로 섞는다.

    출력:
        서로 겹치지 않는 train 및 validation source index array
    """
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

    shuffled_train_indices = np.concatenate(train_indices).astype(np.int32)
    shuffled_validation_indices = np.concatenate(validation_indices).astype(np.int32)
    random_generator.shuffle(shuffled_train_indices)
    random_generator.shuffle(shuffled_validation_indices)
    return shuffled_train_indices, shuffled_validation_indices


# -----------------------------------------------------------------------------
# Manifest 생성
# -----------------------------------------------------------------------------


def create_training_manifest(
    source_images: np.ndarray,
    source_labels: np.ndarray,
    allowed_source_indices: np.ndarray,
    config: dict[str, Any],
    random_generator: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Class pair와 overlap level이 균형인 train manifest를 생성한다.

    입력:
        원본 image/label, 허용 source index, config, random generator

    처리:
        45개 class pair와 세 overlap level 조합을 반복해 독립 sample을 만든다.

    출력:
        Field별 NumPy array로 구성된 train manifest
    """
    sample_count = int(config["dataset"]["train_samples"])
    class_pairs = list(combinations(range(int(config["model"]["class_count"])), 2))
    sampling_cells = list(product(class_pairs, OVERLAP_LEVELS))
    repeated_cells = _repeat_and_shuffle(sampling_cells, sample_count, random_generator)
    source_indices_by_class = _group_source_indices_by_class(
        source_labels, allowed_source_indices
    )

    records = []
    directions = config["overlap"]["directions"]
    for sample_id, (class_pair, overlap_level) in enumerate(repeated_cells):
        direction = tuple(directions[sample_id % len(directions)])
        records.append(_create_manifest_record(
            source_images,
            source_indices_by_class,
            class_pair,
            overlap_level,
            direction,
            sample_id,
            sample_id,
            config,
            random_generator,
        ))
    random_generator.shuffle(records)
    return _records_to_manifest(records)


def create_paired_manifest(
    source_images: np.ndarray,
    source_labels: np.ndarray,
    allowed_source_indices: np.ndarray,
    pair_count: int,
    config: dict[str, Any],
    random_generator: np.random.Generator,
) -> dict[str, np.ndarray]:
    """원본과 방향을 공유하는 paired validation 또는 test manifest를 생성한다.

    입력:
        원본 image/label, 허용 source index, pair 수, config, random generator

    처리:
        각 pair마다 Low/Middle/High 변위만 달리한 세 sample을 만든다.

    출력:
        Field별 NumPy array로 구성된 paired manifest
    """
    class_pairs = list(combinations(range(int(config["model"]["class_count"])), 2))
    repeated_pairs = _repeat_and_shuffle(class_pairs, pair_count, random_generator)
    source_indices_by_class = _group_source_indices_by_class(
        source_labels, allowed_source_indices
    )
    directions = config["overlap"]["directions"]

    records = []
    sample_id = 0
    for pair_id, class_pair in enumerate(repeated_pairs):
        direction = tuple(directions[pair_id % len(directions)])
        source_index_first = int(
            random_generator.choice(source_indices_by_class[class_pair[0]])
        )
        source_index_second = int(
            random_generator.choice(source_indices_by_class[class_pair[1]])
        )
        # 세 overlap level에서 원본 두 개와 이동 방향을 그대로 유지한다.
        fixed_sources = (source_index_first, source_index_second)
        for overlap_level in OVERLAP_LEVELS:
            records.append(_create_manifest_record(
                source_images,
                source_indices_by_class,
                class_pair,
                overlap_level,
                direction,
                sample_id,
                pair_id,
                config,
                random_generator,
                fixed_sources=fixed_sources,
            ))
            sample_id += 1
    return _records_to_manifest(records)


# -----------------------------------------------------------------------------
# Image 합성과 manifest 저장
# -----------------------------------------------------------------------------


def render_overlap_sample(
    source_image_first: torch.Tensor,
    source_image_second: torch.Tensor,
    offset_first: tuple[int, int],
    offset_second: tuple[int, int],
    canvas_size: int,
    stroke_threshold: float,
) -> dict[str, torch.Tensor]:
    """두 원본 숫자를 canvas에 max 합성하고 source별 mask를 만든다.

    입력:
        두 `[28, 28]` image, 두 offset, canvas 크기, stroke threshold

    처리:
        각 image를 별도 canvas에 배치해 maximum과 mask 차집합을 계산한다.

    출력:
        합성 image, 두 mask, 두 exclusive mask dictionary
    """
    canvas_first = torch.zeros((canvas_size, canvas_size), dtype=torch.float32)
    canvas_second = torch.zeros_like(canvas_first)
    _place_image(canvas_first, source_image_first, offset_first)
    _place_image(canvas_second, source_image_second, offset_second)

    mask_first = canvas_first > stroke_threshold
    mask_second = canvas_second > stroke_threshold
    return {
        "image": torch.maximum(canvas_first, canvas_second),
        "mask_first": mask_first,
        "mask_second": mask_second,
        "exclusive_mask_first": mask_first & ~mask_second,
        "exclusive_mask_second": mask_second & ~mask_first,
    }


def load_manifest(path: str | Path) -> dict[str, np.ndarray]:
    """압축 manifest를 memory dictionary로 읽는다.

    입력:
        `.npz` manifest 경로

    처리:
        Archive field를 복사한 뒤 file handle을 닫는다.

    출력:
        Field 이름과 NumPy array의 dictionary
    """
    with np.load(Path(path), allow_pickle=False) as archive:
        return {field_name: archive[field_name] for field_name in archive.files}


def save_manifest(path: str | Path, manifest: dict[str, np.ndarray]) -> None:
    """Manifest dictionary를 압축 NPZ로 저장한다.

    입력:
        출력 경로와 field별 NumPy array

    처리:
        `np.savez_compressed`로 단일 archive를 기록한다.

    출력:
        반환값은 없으며 지정 경로에 manifest file이 생성된다.
    """
    np.savez_compressed(Path(path), **manifest)


# -----------------------------------------------------------------------------
# 저장 데이터 검증
# -----------------------------------------------------------------------------


def validate_saved_data(config: dict[str, Any]) -> list[str]:
    """저장된 source split과 세 manifest의 데이터 계약을 검사한다.

    입력:
        현재 전체 config

    처리:
        Fingerprint, source 교집합, field 길이, overlap, pairing을 확인한다.

    출력:
        통과한 검사를 설명하는 한국어 문자열 목록
    """
    if not SOURCE_SPLIT_PATH.exists():
        raise FileNotFoundError(
            "Source split이 없습니다. "
            "`python -m mnist_overlap prepare-data`를 먼저 실행하세요."
        )
    if not DATA_FINGERPRINT_PATH.exists():
        raise FileNotFoundError("Data config fingerprint가 없습니다. Manifest를 다시 생성하세요.")
    saved_fingerprint = DATA_FINGERPRINT_PATH.read_text(encoding="utf-8").strip()
    if saved_fingerprint != data_config_fingerprint(config):
        raise ValueError("저장된 manifest가 현재 data config와 일치하지 않습니다.")
    with np.load(SOURCE_SPLIT_PATH, allow_pickle=False) as source_split:
        train_sources = source_split["train_source_indices"]
        validation_sources = source_split["validation_source_indices"]
    if np.intersect1d(train_sources, validation_sources).size:
        raise ValueError("Train과 validation source index가 서로 겹칩니다.")

    messages = ["Train과 validation source index가 분리되어 있습니다."]
    expected_sizes = {
        "train": int(config["dataset"]["train_samples"]),
        "validation": int(config["dataset"]["validation_pairs"]) * 3,
        "test": int(config["dataset"]["test_pairs"]) * 3,
    }
    for split_name, expected_size in expected_sizes.items():
        manifest = load_manifest(MANIFEST_PATHS[split_name])
        _validate_manifest(manifest, split_name, expected_size, config)
        messages.append(
            f"{split_name} manifest의 {expected_size:,}개 sample이 유효합니다."
        )
    return messages


# -----------------------------------------------------------------------------
# 좌표와 overlap 계산
# -----------------------------------------------------------------------------


def bounding_box_overlap_ratio(
    displacement_x: int,
    displacement_y: int,
    digit_size: int,
) -> float:
    """동일 크기 digit box 두 개의 면적 overlap 비율을 계산한다.

    입력:
        X/Y 상대 변위와 digit 한 변 크기

    처리:
        두 축의 교차 길이를 곱해 하나의 box 면적으로 나눈다.

    출력:
        `[0, 1]` 범위 bounding-box overlap ratio
    """
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
    config: dict[str, Any],
    random_generator: np.random.Generator,
    fixed_sources: tuple[int, int] | None = None,
) -> tuple[Any, ...]:
    """하나의 합성 sample에 필요한 manifest record를 계산한다.

    입력:
        원본 image pool, class별 index, class pair, overlap, 방향, ID와 config

    처리:
        Source, 정수 변위, offset, box/pixel overlap을 순서대로 결정한다.

    출력:
        MANIFEST_FIELD_NAMES 순서와 일치하는 tuple record
    """
    if fixed_sources is None:
        source_index_first = int(random_generator.choice(source_indices_by_class[class_pair[0]]))
        source_index_second = int(random_generator.choice(source_indices_by_class[class_pair[1]]))
    else:
        source_index_first, source_index_second = fixed_sources

    dataset_config = config["dataset"]
    digit_size = int(dataset_config["digit_size"])
    displacement_x, displacement_y = _sample_displacement(
        direction,
        config["overlap"][overlap_level],
        digit_size,
        random_generator,
    )
    offset_first, offset_second = _centered_offsets(
        displacement_x,
        displacement_y,
        int(dataset_config["canvas_size"]),
        digit_size,
    )
    box_overlap = bounding_box_overlap_ratio(displacement_x, displacement_y, digit_size)
    pixel_overlap = _pixel_overlap_ratio(
        source_images[source_index_first],
        source_images[source_index_second],
        offset_first,
        offset_second,
        float(dataset_config["stroke_threshold"]),
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
        box_overlap,
        pixel_overlap,
        overlap_level,
    )


def _sample_displacement(
    direction: tuple[int, int],
    overlap_bounds: list[float],
    digit_size: int,
    random_generator: np.random.Generator,
) -> tuple[int, int]:
    """지정 방향과 overlap 구간을 만족하는 정수 변위를 선택한다.

    입력:
        정수 방향, overlap 하한/상한, digit 크기, random generator

    처리:
        가능한 이동 거리를 열거하고 조건을 만족하는 후보에서 하나를 뽑는다.

    출력:
        `(displacement_x, displacement_y)` 정수 tuple
    """
    candidates = []
    for distance in range(digit_size):
        displacement_x = direction[0] * distance
        displacement_y = direction[1] * distance
        overlap_ratio = bounding_box_overlap_ratio(
            displacement_x, displacement_y, digit_size
        )
        if overlap_bounds[0] <= overlap_ratio <= overlap_bounds[1]:
            candidates.append((displacement_x, displacement_y))
    if not candidates:
        raise ValueError(
            f"조건을 만족하는 정수 변위가 없습니다: "
            f"direction={direction}, bounds={overlap_bounds}"
        )
    return candidates[int(random_generator.integers(len(candidates)))]


def _centered_offsets(
    displacement_x: int,
    displacement_y: int,
    canvas_size: int,
    digit_size: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Pair center를 유지하는 두 digit의 정수 top-left offset을 계산한다.

    입력:
        X/Y 변위, canvas 크기, digit 크기

    처리:
        중앙 top-left를 기준으로 변위를 절반씩 분배하고 경계를 검사한다.

    출력:
        First 및 second digit의 `(x, y)` offset tuple
    """
    centered_top_left = (canvas_size - digit_size) // 2
    first_x = centered_top_left - displacement_x // 2
    first_y = centered_top_left - displacement_y // 2
    second_x = first_x + displacement_x
    second_y = first_y + displacement_y
    for coordinate in (first_x, first_y, second_x, second_y):
        if not 0 <= coordinate <= canvas_size - digit_size:
            raise ValueError("중앙 offset으로 배치한 숫자가 canvas 경계를 벗어납니다.")
    return (first_x, first_y), (second_x, second_y)


def _pixel_overlap_ratio(
    image_first: np.ndarray,
    image_second: np.ndarray,
    offset_first: tuple[int, int],
    offset_second: tuple[int, int],
    stroke_threshold: float,
) -> float:
    """두 digit stroke가 실제로 겹치는 pixel 비율을 계산한다.

    입력:
        두 uint8 원본 image, 두 offset, `[0, 1]` stroke threshold

    처리:
        교차 bounding 영역에서 mask intersection을 작은 mask 면적으로 나눈다.

    출력:
        `[0, 1]` 범위 pixel overlap ratio
    """
    threshold_uint8 = stroke_threshold * 255.0
    mask_first = image_first > threshold_uint8
    mask_second = image_second > threshold_uint8
    first_pixel_count = int(mask_first.sum())
    second_pixel_count = int(mask_second.sum())
    denominator = min(first_pixel_count, second_pixel_count)
    if denominator == 0:
        return 0.0

    first_x, first_y = offset_first
    second_x, second_y = offset_second
    overlap_left = max(first_x, second_x)
    overlap_top = max(first_y, second_y)
    overlap_right = min(first_x + image_first.shape[1], second_x + image_second.shape[1])
    overlap_bottom = min(first_y + image_first.shape[0], second_y + image_second.shape[0])
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


def _place_image(
    canvas: torch.Tensor,
    image: torch.Tensor,
    offset: tuple[int, int],
) -> None:
    """원본 image를 canvas의 지정 top-left 위치에 배치한다.

    입력:
        수정할 canvas, source image, `(x, y)` offset

    처리:
        Image 높이와 너비에 맞는 canvas slice에 값을 복사한다.

    출력:
        반환값은 없으며 입력 canvas가 제자리에서 변경된다.
    """
    offset_x, offset_y = offset
    image_height, image_width = image.shape
    canvas[offset_y:offset_y + image_height, offset_x:offset_x + image_width] = image


def _group_source_indices_by_class(
    labels: np.ndarray,
    allowed_source_indices: np.ndarray,
) -> dict[int, np.ndarray]:
    """허용된 MNIST source index를 class별로 묶는다.

    입력:
        전체 source label과 현재 split에서 허용된 index

    처리:
        각 class label에 해당하는 허용 index를 Boolean mask로 선택한다.

    출력:
        Class 정수를 key로 갖는 source index array dictionary
    """
    grouped = {}
    for class_value in np.unique(labels):
        class_indices = allowed_source_indices[labels[allowed_source_indices] == class_value]
        if not len(class_indices):
            raise ValueError(f"Source split에 class {class_value} sample이 없습니다.")
        grouped[int(class_value)] = class_indices
    return grouped


# -----------------------------------------------------------------------------
# Manifest 변환과 무결성 검사
# -----------------------------------------------------------------------------


def _repeat_and_shuffle(
    values: list[Any],
    requested_count: int,
    random_generator: np.random.Generator,
) -> list[Any]:
    """균형을 유지하도록 값 목록을 반복한 뒤 순서를 섞는다.

    입력:
        반복할 값 목록, 필요한 총수, random generator

    처리:
        Modulo index로 빈도 차이를 1 이하로 유지하고 전체 순서를 섞는다.

    출력:
        requested_count 길이의 shuffled list
    """
    repeated = [values[index % len(values)] for index in range(requested_count)]
    random_generator.shuffle(repeated)
    return repeated


def _records_to_manifest(records: list[tuple[Any, ...]]) -> dict[str, np.ndarray]:
    """Record tuple 목록을 field별 typed NumPy array로 변환한다.

    입력:
        동일한 field 순서를 가진 manifest record 목록

    처리:
        Column 단위로 전치하고 integer, float, 문자열 dtype을 적용한다.

    출력:
        저장 가능한 manifest dictionary
    """
    columns = list(zip(*records))
    integer_fields = set(MANIFEST_FIELD_NAMES[:12])
    manifest = {}
    for field_name, column in zip(MANIFEST_FIELD_NAMES, columns):
        if field_name in integer_fields:
            manifest[field_name] = np.asarray(column, dtype=np.int32)
        elif field_name == "overlap_level":
            manifest[field_name] = np.asarray(column, dtype="<U6")
        else:
            manifest[field_name] = np.asarray(column, dtype=np.float32)
    return manifest


def _validate_manifest(
    manifest: dict[str, np.ndarray],
    split_name: str,
    expected_size: int,
    config: dict[str, Any],
) -> None:
    """하나의 manifest가 split별 데이터 계약을 만족하는지 검사한다.

    입력:
        Manifest dictionary, split 이름, 기대 sample 수, config

    처리:
        Field, 길이, label, offset, overlap, class 균형, pairing을 확인한다.

    출력:
        반환값은 없으며 첫 계약 위반에서 ValueError를 발생시킨다.
    """
    missing_fields = set(MANIFEST_FIELD_NAMES).difference(manifest)
    if missing_fields:
        raise ValueError(
            f"{split_name} manifest에 field가 없습니다: {sorted(missing_fields)}"
        )
    if any(len(values) != expected_size for values in manifest.values()):
        raise ValueError(f"{split_name} manifest field 길이가 {expected_size}가 아닙니다.")
    if np.any(manifest["label_first"] >= manifest["label_second"]):
        raise ValueError(f"{split_name} label은 서로 다른 class를 오름차순으로 저장해야 합니다.")

    actual_displacement_x = manifest["offset_second_x"] - manifest["offset_first_x"]
    actual_displacement_y = manifest["offset_second_y"] - manifest["offset_first_y"]
    if not np.array_equal(actual_displacement_x, manifest["displacement_x"]):
        raise ValueError(f"{split_name} displacement_x와 offset 차이가 일치하지 않습니다.")
    if not np.array_equal(actual_displacement_y, manifest["displacement_y"]):
        raise ValueError(f"{split_name} displacement_y와 offset 차이가 일치하지 않습니다.")

    for overlap_level in OVERLAP_LEVELS:
        level_mask = manifest["overlap_level"] == overlap_level
        lower_bound, upper_bound = config["overlap"][overlap_level]
        ratios = manifest["bounding_box_overlap_ratio"][level_mask]
        if not len(ratios) or np.any(ratios < lower_bound) or np.any(ratios > upper_bound):
            raise ValueError(f"{split_name}의 {overlap_level} overlap ratio가 범위를 벗어납니다.")

    pair_counts = Counter(zip(manifest["label_first"], manifest["label_second"]))
    if max(pair_counts.values()) - min(pair_counts.values()) > 3:
        raise ValueError(f"{split_name} class-pair 분포가 균형을 이루지 않습니다.")

    if split_name in ("validation", "test"):
        _validate_paired_rows(manifest, split_name)


def _validate_paired_rows(manifest: dict[str, np.ndarray], split_name: str) -> None:
    """Validation/test의 세 overlap row가 같은 pair 조건을 공유하는지 검사한다.

    입력:
        Paired manifest와 오류 메시지에 사용할 split 이름

    처리:
        Pair별 세 level, source, label, 이동 방향의 동일성을 확인한다.

    출력:
        반환값은 없으며 pairing 위반에서 ValueError를 발생시킨다.
    """
    for pair_id in np.unique(manifest["pair_id"]):
        rows = np.flatnonzero(manifest["pair_id"] == pair_id)
        if len(rows) != 3 or set(manifest["overlap_level"][rows]) != set(OVERLAP_LEVELS):
            raise ValueError(
                f"{split_name} pair {pair_id}에 세 overlap level이 모두 있지 않습니다."
            )
        for field_name in (
            "source_index_first", "source_index_second", "label_first", "label_second"
        ):
            if len(np.unique(manifest[field_name][rows])) != 1:
                raise ValueError(
                    f"{split_name} pair {pair_id}에서 {field_name} 값이 변경됩니다."
                )
        directions = np.sign(np.column_stack((
            manifest["displacement_x"][rows], manifest["displacement_y"][rows]
        )))
        if len(np.unique(directions, axis=0)) != 1:
            raise ValueError(f"{split_name} pair {pair_id}에서 이동 방향이 변경됩니다.")
