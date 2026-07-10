"""Batch-1 PyTorch eager latency including position-wise argmax."""

import argparse
from pathlib import Path

import torch

from static_sequence_core.bench import (
    benchmark_latency,
    print_report,
    write_latency_csv,
    write_prediction_csv,
)
from static_sequence_core.config import CHECKPOINT_PATH, LOGS_DIR, load_config
from static_sequence_core.dataset import load_test_tensors
from static_sequence_core.metrics import compute_metrics
from static_sequence_core.model import StaticSequenceRecognizer


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
    torch.set_num_threads(threads)

    model = StaticSequenceRecognizer(cfg)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True))
    model.eval()
    images, labels = load_test_tensors(cfg)
    with torch.inference_mode():
        predictions = torch.cat([
            model(batch).argmax(dim=2)
            for batch in images.split(cfg["evaluation"]["batch_size"])
        ])
        inputs = [images[index:index + 1] for index in range(len(images))]
        stats = benchmark_latency(
            lambda sample: model(sample).argmax(dim=2), inputs, warmup, n
        )
    metrics = compute_metrics(predictions, labels)
    print_report("py_pytorch", metrics, stats)
    write_latency_csv(args.logdir / "py_pytorch_latency.csv", stats["times"])
    write_prediction_csv(args.logdir / "py_pytorch_predictions.csv", labels, predictions)


if __name__ == "__main__":
    main()
