"""Batch-1 Python ONNX Runtime latency including position-wise argmax."""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from static_sequence_core.bench import (
    benchmark_latency,
    print_report,
    write_latency_csv,
    write_prediction_csv,
)
from static_sequence_core.config import LOGS_DIR, ONNX_PATH, load_config
from static_sequence_core.dataset import load_test_tensors
from static_sequence_core.metrics import compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--logdir", type=Path, default=LOGS_DIR)
    args = parser.parse_args()

    cfg = load_config()
    benchmark_cfg = cfg["benchmark"]
    n = args.n or benchmark_cfg["n"]
    warmup = args.warmup if args.warmup is not None else benchmark_cfg["warmup"]
    threads = args.threads or benchmark_cfg["threads"]

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(ONNX_PATH), sess_options=options, providers=["CPUExecutionProvider"]
    )
    images, labels = load_test_tensors(cfg)
    images_np = images.numpy()
    predictions = session.run(["logits"], {"image": images_np})[0].argmax(axis=2)
    inputs = [images_np[index:index + 1] for index in range(len(images_np))]
    stats = benchmark_latency(
        lambda sample: session.run(["logits"], {"image": sample})[0].argmax(axis=2),
        inputs,
        warmup,
        n,
    )
    prediction_tensor = torch.from_numpy(predictions.astype(np.int64))
    metrics = compute_metrics(prediction_tensor, labels)
    print_report("py_onnx", metrics, stats)
    write_latency_csv(args.logdir / "py_onnx_latency.csv", stats["times"])
    write_prediction_csv(args.logdir / "py_onnx_predictions.csv", labels, predictions)


if __name__ == "__main__":
    main()
