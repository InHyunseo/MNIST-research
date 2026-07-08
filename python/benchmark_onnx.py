"""ONNX Runtime(Python) 추론 latency 벤치마크.

기본 실행은 기존 baseline 파일명(logs/py_onnx_s{seed}_*.csv)을 유지한다.
--variant를 주면 8종 tuning ablation 로그를 생성한다.
"""
import argparse
from pathlib import Path

import onnxruntime as ort

from mnist_core.config import load_config, ONNX_DIR, LOGS_DIR
from mnist_core.dataset import load_test_tensors
from mnist_core.metrics import accuracy
from mnist_core.bench import bench_latency, print_report, write_latency_csv, write_preds_csv

LABEL = "py_onnx"
VARIANTS = {
    "none": (False, False, False),
    "graph": (True, False, False),
    "named": (False, True, False),
    "memory": (False, False, True),
    "graph_named": (True, True, False),
    "graph_memory": (True, False, True),
    "named_memory": (False, True, True),
    "all": (True, True, True),
}


def make_session(onnx_path, threads, variant):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1

    graph_opt, _, memory_reuse = VARIANTS[variant or "none"]
    so.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if graph_opt else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    so.enable_mem_pattern = memory_reuse
    so.enable_cpu_mem_arena = memory_reuse

    return ort.InferenceSession(str(onnx_path), sess_options=so,
                                providers=["CPUExecutionProvider"])


def output_names_for(variant):
    if variant is None:
        return None
    _, named_output, _ = VARIANTS[variant]
    return ["logits"] if named_output else None


def label_for(variant, threads):
    if variant is None:
        return LABEL
    return f"{LABEL}_ablation_{variant}_t{threads}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--variant", choices=VARIANTS, default=None)
    ap.add_argument("--logdir", type=Path, default=LOGS_DIR)
    args = ap.parse_args()

    cfg = load_config()
    b = cfg["benchmark"]
    threads = args.threads or b["threads"]
    warmup = args.warmup if args.warmup is not None else b["warmup"]
    n = args.n or b["n"]

    sess = make_session(ONNX_DIR / f"mnist_cnn_s{args.seed}.onnx",
                        threads, args.variant)
    output_names = output_names_for(args.variant)
    label = label_for(args.variant, threads)

    images, labels = load_test_tensors()
    images = images.numpy()
    trues = labels.numpy()
    preds = sess.run(output_names, {"image": images})[0].argmax(1)
    acc = accuracy(preds, trues)

    inputs = [images[i:i + 1] for i in range(images.shape[0])]
    stats = bench_latency(lambda x: sess.run(output_names, {"image": x}),
                          inputs, warmup, n)
    print_report(label, threads, warmup, n, float(acc), stats)

    args.logdir.mkdir(parents=True, exist_ok=True)
    write_latency_csv(args.logdir / f"{label}_s{args.seed}_latency.csv",
                      stats["times"])
    write_preds_csv(args.logdir / f"{label}_s{args.seed}_preds.csv", trues, preds)


if __name__ == "__main__":
    main()
