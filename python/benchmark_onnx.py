"""ONNX Runtime(Python) 추론 latency 벤치마크. logs/py_onnx_s{seed}_{latency,preds}.csv.

benchmark_pytorch.py와 동일한 측정 조건.
실행: python python/benchmark_onnx.py --seed 0 [--n 2000] [--threads 1]
"""
import argparse

import onnxruntime as ort

from mnist_core.config import load_config, ONNX_DIR, LOGS_DIR
from mnist_core.dataset import load_test_tensors
from mnist_core.metrics import accuracy
from mnist_core.bench import bench_latency, print_report, write_latency_csv, write_preds_csv

LABEL = "py_onnx"


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

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(ONNX_DIR / f"mnist_cnn_s{args.seed}.onnx"),
                                sess_options=so, providers=["CPUExecutionProvider"])

    images, labels = load_test_tensors()
    images = images.numpy()
    trues = labels.numpy()
    preds = sess.run(None, {"image": images})[0].argmax(1)
    acc = accuracy(preds, trues)

    inputs = [images[i:i + 1] for i in range(images.shape[0])]
    stats = bench_latency(lambda x: sess.run(None, {"image": x}), inputs, warmup, n)
    print_report(LABEL, threads, warmup, n, float(acc), stats)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    write_latency_csv(LOGS_DIR / f"{LABEL}_s{args.seed}_latency.csv", stats["times"])
    write_preds_csv(LOGS_DIR / f"{LABEL}_s{args.seed}_preds.csv", trues, preds)


if __name__ == "__main__":
    main()
