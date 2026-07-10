"""checkpoint를 ONNX로 export하고 검증한다 -> models/onnx/mnist_cnn_s{seed}.onnx.

검증: 같은 test set에서 PyTorch(eager)와 ONNX Runtime의 예측이 일치하는지 확인.
실행: python python/export_onnx.py --seed 0
"""
import argparse

import onnxruntime as ort
import torch

from mnist_core.config import load_config, CKPT_DIR, ONNX_DIR
from mnist_core.model import InferModule
from mnist_core.dataset import load_test_tensors


def export(cfg, seed):
    model = InferModule(cfg)
    ckpt = CKPT_DIR / f"mnist_cnn_s{seed}.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = ONNX_DIR / f"mnist_cnn_s{seed}.onnx"
    dummy = torch.zeros(1, 1, 28, 28, dtype=torch.uint8)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported -> {onnx_path}")
    return model, onnx_path


def verify(model, onnx_path):
    images, labels = load_test_tensors()
    with torch.no_grad():
        eager = model(images).argmax(1).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx = sess.run(None, {"image": images.numpy()})[0].argmax(1)
    y = labels.numpy()
    print(f"eager acc = {(eager == y).mean():.4f}")
    print(f"onnx  acc = {(onnx == y).mean():.4f}")
    print(f"eager vs onnx 일치율 = {(eager == onnx).mean()*100:.2f}%  (100% 기대)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config()
    model, onnx_path = export(cfg, args.seed)
    verify(model, onnx_path)
