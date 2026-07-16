"""Dump deterministic test images and labels for the C++ runner."""

from static_sequence_core.config import TEST_IMAGES_PATH, TEST_LABELS_PATH, load_config
from static_sequence_core.dataset import dump_test_data


def main() -> None:
    image_shape, label_shape = dump_test_data(load_config())
    print(f"images {tuple(image_shape)} -> {TEST_IMAGES_PATH}")
    print(f"labels {tuple(label_shape)} -> {TEST_LABELS_PATH}")


if __name__ == "__main__":
    main()
