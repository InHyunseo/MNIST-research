"""PyTorch(eager) 추론 latency 벤치마크. logs/py_pytorch_s{seed}_{latency,preds}.csv 기록.

실행: python python/benchmark_pytorch.py --seed 0 [--n 2000] [--threads 1]
"""
import argparse

import torch
from torch.utils.data import DataLoader

from mnist_core.config import load_config, CKPT_DIR, LOGS_DIR
from mnist_core.model import InferModule
from mnist_core.dataset import load_test_tensors
from mnist_core.metrics import accuracy
from mnist_core.bench import bench_latency, print_report, write_latency_csv, write_preds_csv

LABEL = "py_pytorch"


def predict_all(model, images):
    return torch.cat([model(x).argmax(1) for x in DataLoader(images, batch_size=1000)]).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    b = cfg["benchmark"]
    threads = args.threads or b["threads"]
    warmup, n = b["warmup"], args.n or b["n"]

    torch.set_num_threads(threads)
    torch.set_grad_enabled(False)

    model = InferModule(cfg)
    model.load_state_dict(torch.load(CKPT_DIR / f"mnist_cnn_s{args.seed}.pt",
                                     map_location="cpu", weights_only=True))
    model.eval()

    images, labels = load_test_tensors()
    trues = labels.numpy()
    preds = predict_all(model, images)
    acc = accuracy(preds, trues)

    inputs = [images[i:i + 1] for i in range(images.shape[0])]
    stats = bench_latency(lambda x: model(x), inputs, warmup, n)
    print_report(LABEL, threads, warmup, n, acc, stats)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    write_latency_csv(LOGS_DIR / f"{LABEL}_s{args.seed}_latency.csv", stats["times"])
    write_preds_csv(LOGS_DIR / f"{LABEL}_s{args.seed}_preds.csv", trues, preds)


if __name__ == "__main__":
    main()
