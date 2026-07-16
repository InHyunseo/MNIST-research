"""Export the static recognizer and verify PyTorch/ONNX numerical fidelity."""

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader

from static_sequence_core.config import CHECKPOINT_PATH, ONNX_DIR, ONNX_PATH, canvas_shape, load_config
from static_sequence_core.dataset import create_dataset
from static_sequence_core.model import StaticSequenceRecognizer


def export_model(cfg: dict) -> StaticSequenceRecognizer:
    model = StaticSequenceRecognizer(cfg)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True))
    model.eval()
    height, width = canvas_shape(cfg)
    dummy = torch.zeros(1, 1, height, width, dtype=torch.uint8)
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(ONNX_PATH),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=cfg["export"]["opset"],
    )
    print(f"exported -> {ONNX_PATH}")
    return model


@torch.inference_mode()
def verify_export(model: StaticSequenceRecognizer, cfg: dict) -> None:
    options = ort.SessionOptions()
    options.intra_op_num_threads = cfg["benchmark"]["threads"]
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(ONNX_PATH), sess_options=options, providers=["CPUExecutionProvider"]
    )
    loader = DataLoader(
        create_dataset(cfg, train=False),
        batch_size=cfg["evaluation"]["batch_size"],
        shuffle=False,
    )
    same_predictions = 0
    sequence_count = 0
    for images, _ in loader:
        eager_logits = model(images).numpy()
        onnx_logits = session.run(["logits"], {"image": images.numpy()})[0]
        np.testing.assert_allclose(
            eager_logits,
            onnx_logits,
            rtol=cfg["export"]["rtol"],
            atol=cfg["export"]["atol"],
        )
        eager_predictions = eager_logits.argmax(axis=2)
        onnx_predictions = onnx_logits.argmax(axis=2)
        same_predictions += int(np.all(eager_predictions == onnx_predictions, axis=1).sum())
        sequence_count += images.shape[0]
    fidelity = same_predictions / sequence_count
    if fidelity != 1.0:
        raise RuntimeError(f"decoded PyTorch/ONNX fidelity is {fidelity:.6f}, expected 1.0")
    print(f"PyTorch vs ONNX decoded fidelity = {fidelity * 100:.2f}%")


def main() -> None:
    cfg = load_config()
    model = export_model(cfg)
    verify_export(model, cfg)


if __name__ == "__main__":
    main()
