"""Stage-local paths, YAML loading, and startup validation."""

import argparse
from pathlib import Path
from typing import Any

import yaml

STAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = STAGE_ROOT.parent
CONFIG_PATH = STAGE_ROOT / "configs" / "static.yaml"

DATA_DIR = STAGE_ROOT / "data"
CHECKPOINT_DIR = STAGE_ROOT / "models" / "checkpoints"
ONNX_DIR = STAGE_ROOT / "models" / "onnx"
LOGS_DIR = STAGE_ROOT / "logs"

CHECKPOINT_PATH = CHECKPOINT_DIR / "static_cnn_s0.pt"
ONNX_PATH = ONNX_DIR / "static_cnn_s0.onnx"
TEST_IMAGES_PATH = DATA_DIR / "static_test_images.u8"
TEST_LABELS_PATH = DATA_DIR / "static_test_labels.u8"


def _require_positive(section: str, values: dict[str, Any]) -> None:
    for name, value in values.items():
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{section}.{name} must be positive, got {value!r}")


def validate_config(cfg: dict[str, Any]) -> None:
    """Validate assumptions shared by training, export, and deployment."""
    required = {
        "dataset", "model", "train", "evaluation", "export", "benchmark", "preprocess"
    }
    missing = required.difference(cfg)
    if missing:
        raise ValueError(f"missing config sections: {sorted(missing)}")

    dataset = cfg["dataset"]
    if dataset["sequence_length"] != 3:
        raise ValueError("dataset.sequence_length must be 3 for the static model contract")
    if dataset["digit_size"] != 28 or dataset["slot_size"] != 32:
        raise ValueError("the static canvas contract requires digit_size=28 and slot_size=32")
    _require_positive("dataset", {
        "train_samples": dataset["train_samples"],
        "test_samples": dataset["test_samples"],
    })

    _require_positive("model", cfg["model"])
    _require_positive("train", {
        "epochs": cfg["train"]["epochs"],
        "fallback_epochs": cfg["train"]["fallback_epochs"],
        "batch_size": cfg["train"]["batch_size"],
        "learning_rate": cfg["train"]["learning_rate"],
    })
    if cfg["train"]["fallback_epochs"] < cfg["train"]["epochs"]:
        raise ValueError("train.fallback_epochs must be >= train.epochs")
    for key in ("min_digit_accuracy", "min_exact_match"):
        value = cfg["train"][key]
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"train.{key} must be in [0, 1]")

    _require_positive("evaluation", cfg["evaluation"])
    _require_positive("export", cfg["export"])
    _require_positive("benchmark", cfg["benchmark"])
    if cfg["preprocess"]["std"] <= 0:
        raise ValueError("preprocess.std must be positive")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as config_file:
        cfg = yaml.safe_load(config_file)
    validate_config(cfg)
    return cfg


def canvas_shape(cfg: dict[str, Any]) -> tuple[int, int]:
    dataset = cfg["dataset"]
    return dataset["slot_size"], dataset["slot_size"] * dataset["sequence_length"]


def config_value(cfg: dict[str, Any], dotted_key: str) -> Any:
    value: Any = cfg
    for key in dotted_key.split("."):
        value = value[key]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Print one value from static.yaml")
    parser.add_argument("key", help="Dotted key, for example benchmark.n")
    args = parser.parse_args()
    print(config_value(load_config(), args.key))


if __name__ == "__main__":
    main()
