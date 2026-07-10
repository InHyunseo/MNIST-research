"""설정 로드 + 공통 경로."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]   # base-single/
CONFIG_PATH = ROOT / "configs" / "cnn.yaml"

CKPT_DIR = ROOT / "models" / "checkpoints"
ONNX_DIR = ROOT / "models" / "onnx"
DATA_DIR = ROOT / "data"
LOGS_DIR = ROOT / "logs"


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)
