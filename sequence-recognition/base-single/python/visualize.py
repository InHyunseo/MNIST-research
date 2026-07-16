"""벤치마크 CSV를 읽어 결과 그래프와 표를 만든다.

기본 mode는 기존 baseline 결과(results/{figures,tables,samples})를 만든다.
추가 mode로 8종 ONNX tuning ablation과 3-backend comparison을 생성한다.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mnist_core.config import load_config, LOGS_DIR
from mnist_core.metrics import confusion, per_class_prf
from mnist_core.dataset import load_test_tensors

ROOT = LOGS_DIR.parent
BASE_BACKENDS = ["py_pytorch", "py_onnx", "cpp_onnx"]
VARIANTS = [
    "none",
    "graph",
    "named",
    "memory",
    "graph_named",
    "graph_memory",
    "named_memory",
    "all",
]
VARIANT_DISPLAY = {
    "none": "none",
    "graph": "graph opt",
    "named": "named output",
    "memory": "memory reuse",
    "graph_named": "graph + named",
    "graph_memory": "graph + memory",
    "named_memory": "named + memory",
    "all": "all",
}
BACKEND_ROWS = [
    ("py_pytorch", "PyTorch eager"),
    ("py_onnx_ablation_all", "Python ONNX"),
    ("cpp_onnx_ablation_all", "C++ ONNX"),
]


def save_fig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"figure -> {path}")


def percentile(sorted_vals, q):
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def sample_std(values):
    return values.std(ddof=1) if len(values) > 1 else 0.0


def load_latency(logdir, backend, seed):
    return np.loadtxt(logdir / f"{backend}_s{seed}_latency.csv", skiprows=1)


def load_preds(logdir, backend, seed):
    a = np.loadtxt(logdir / f"{backend}_s{seed}_preds.csv", skiprows=1,
                   delimiter=",", dtype=int)
    return a[:, 0], a[:, 1]


def run_baseline(args, cfg):
    logdir = args.logdir or LOGS_DIR
    results_root = args.results_root or ROOT / "results"
    figs_dir = results_root / "figures"
    tables_dir = results_root / "tables"
    samples_dir = results_root / "samples"
    for d in (figs_dir, tables_dir, samples_dir):
        d.mkdir(parents=True, exist_ok=True)

    seeds = args.seeds or cfg["benchmark"]["seeds"]
    data = [np.sort(np.concatenate([load_latency(logdir, b, s) for s in seeds]))
            for b in BASE_BACKENDS]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot(data, labels=BASE_BACKENDS, showfliers=False)
    ax.set_ylabel("latency (us)")
    save_fig(fig, figs_dir / "latency_boxplot.png")

    thr_mean = [np.mean([1e6 / load_latency(logdir, b, s).mean() for s in seeds])
                for b in BASE_BACKENDS]
    thr_std = [np.std([1e6 / load_latency(logdir, b, s).mean() for s in seeds])
               for b in BASE_BACKENDS]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(BASE_BACKENDS, thr_mean, yerr=thr_std, capsize=4)
    ax.set_ylabel("throughput (inf/s)")
    save_fig(fig, figs_dir / "inference_time_bar.png")

    with open(tables_dir / "benchmark_summary.csv", "w") as f:
        f.write("backend,mean_us,median_us,p95_us,throughput\n")
        for b, d in zip(BASE_BACKENDS, data):
            f.write(f"{b},{d.mean():.2f},{percentile(d, .5):.2f},"
                    f"{percentile(d, .95):.2f},{1e6 / d.mean():.0f}\n")
    print(f"table -> {tables_dir / 'benchmark_summary.csv'}")

    for seed in seeds:
        p0 = load_preds(logdir, BASE_BACKENDS[0], seed)[1]
        same = all((load_preds(logdir, b, seed)[1] == p0).all()
                   for b in BASE_BACKENDS[1:])
        print(f"seed {seed}: 3 backend 예측 동일 = {same}")

    cmat = np.zeros((10, 10), dtype=int)
    for seed in seeds:
        true, pred = load_preds(logdir, "py_pytorch", seed)
        cmat += confusion(true, pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cmat, cmap="Blues")
    ax.set_xlabel("pred")
    ax.set_ylabel("true")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    for i in range(10):
        for j in range(10):
            if cmat[i, j]:
                ax.text(j, i, cmat[i, j], ha="center", va="center", fontsize=6,
                        color="white" if i == j else "black")
    save_fig(fig, figs_dir / "confusion_matrix.png")

    accs, f1s = [], []
    for seed in seeds:
        true, pred = load_preds(logdir, "py_pytorch", seed)
        recall, _, f1 = per_class_prf(confusion(true, pred))
        accs.append(recall)
        f1s.append(f1)
    accs, f1s = np.array(accs), np.array(f1s)
    for arr, name, ylab in [(accs, "per_class_accuracy.png", "accuracy"),
                            (f1s, "per_class_f1.png", "F1")]:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(range(10), arr.mean(0), yerr=arr.std(0), capsize=3)
        ax.set_xlabel("digit")
        ax.set_ylabel(ylab)
        ax.set_xticks(range(10))
        ax.set_ylim(0.95, 1.0)
        save_fig(fig, figs_dir / name)

    curve = np.loadtxt(logdir / "train_curve.csv", skiprows=1, delimiter=",")
    for col, name, ylab in [(1, "loss_curve.png", "train loss"),
                            (2, "accuracy_curve.png", "test accuracy")]:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(curve[:, 0], curve[:, col])
        ax.set_xlabel("step")
        ax.set_ylabel(ylab)
        save_fig(fig, figs_dir / name)

    imgs, _ = load_test_tensors()
    true, pred = load_preds(logdir, "py_pytorch", seeds[0])
    for idxs, name in [(np.where(true == pred)[0][:8], "correct_examples.png"),
                       (np.where(true != pred)[0][:8], "wrong_examples.png")]:
        fig, axes = plt.subplots(1, len(idxs), figsize=(len(idxs) * 1.2, 1.6))
        axes = np.atleast_1d(axes)
        for ax, i in zip(axes, idxs):
            ax.imshow(imgs[i, 0].numpy(), cmap="gray")
            ax.axis("off")
            ax.set_title(f"{true[i]}/{pred[i]}", fontsize=7)
        fig.tight_layout()
        fig.savefig(samples_dir / name, dpi=150)
        plt.close(fig)
        print(f"sample -> {samples_dir / name}")


def ablation_latency_path(logdir, backend, variant, thread, seed):
    return logdir / f"{backend}_{variant}_t{thread}_s{seed}_latency.csv"


def ablation_pred_path(logdir, backend, variant, thread, seed):
    return logdir / f"{backend}_{variant}_t{thread}_s{seed}_preds.csv"


def load_ablation_latency(logdir, backend, variant, thread, seed):
    return np.loadtxt(ablation_latency_path(logdir, backend, variant, thread, seed),
                      skiprows=1)


def load_ablation_preds(logdir, backend, variant, thread, seed):
    a = np.loadtxt(ablation_pred_path(logdir, backend, variant, thread, seed),
                   skiprows=1, delimiter=",", dtype=int)
    return a[:, 0], a[:, 1]


def ablation_seed_stats(logdir, backend, variant, thread, seeds):
    means, medians, p95s = [], [], []
    for seed in seeds:
        v = load_ablation_latency(logdir, backend, variant, thread, seed)
        means.append(v.mean())
        medians.append(np.median(v))
        p95s.append(np.percentile(v, 95))
    return np.array(means), np.array(medians), np.array(p95s)


def write_ablation_summary(logdir, tables_dir, seeds, threads, variants):
    rows = []
    for variant in variants:
        for thread in threads:
            py_means, py_medians, py_p95s = ablation_seed_stats(
                logdir, "py_onnx_ablation", variant, thread, seeds)
            cpp_means, cpp_medians, cpp_p95s = ablation_seed_stats(
                logdir, "cpp_onnx_ablation", variant, thread, seeds)
            rows.append({
                "variant": variant,
                "threads": thread,
                "py_mean_us": py_means.mean(),
                "py_mean_std_us": sample_std(py_means),
                "cpp_mean_us": cpp_means.mean(),
                "cpp_mean_std_us": sample_std(cpp_means),
                "cpp_speedup_mean": py_means.mean() / cpp_means.mean(),
                "abs_diff_mean_us": py_means.mean() - cpp_means.mean(),
                "py_median_us": py_medians.mean(),
                "cpp_median_us": cpp_medians.mean(),
                "cpp_speedup_median": py_medians.mean() / cpp_medians.mean(),
                "py_p95_us": py_p95s.mean(),
                "cpp_p95_us": cpp_p95s.mean(),
                "cpp_speedup_p95": py_p95s.mean() / cpp_p95s.mean(),
            })

    path = tables_dir / "tuning_ablation_summary.csv"
    with open(path, "w") as f:
        f.write("variant,threads,py_mean_us,py_mean_std_us,cpp_mean_us,"
                "cpp_mean_std_us,cpp_speedup_mean,abs_diff_mean_us,"
                "py_median_us,cpp_median_us,cpp_speedup_median,"
                "py_p95_us,cpp_p95_us,cpp_speedup_p95\n")
        for r in rows:
            f.write(f"{r['variant']},{r['threads']},{r['py_mean_us']:.2f},"
                    f"{r['py_mean_std_us']:.2f},{r['cpp_mean_us']:.2f},"
                    f"{r['cpp_mean_std_us']:.2f},{r['cpp_speedup_mean']:.3f},"
                    f"{r['abs_diff_mean_us']:.2f},{r['py_median_us']:.2f},"
                    f"{r['cpp_median_us']:.2f},{r['cpp_speedup_median']:.3f},"
                    f"{r['py_p95_us']:.2f},{r['cpp_p95_us']:.2f},"
                    f"{r['cpp_speedup_p95']:.3f}\n")
    print(f"table -> {path}")
    return rows


def write_ablation_fidelity(logdir, tables_dir, seeds, threads, variants):
    path = tables_dir / "tuning_ablation_prediction_fidelity.csv"
    with open(path, "w") as f:
        f.write("variant,seed,threads,py_cpp_same,accuracy,count\n")
        for variant in variants:
            for seed in seeds:
                for thread in threads:
                    true_py, pred_py = load_ablation_preds(
                        logdir, "py_onnx_ablation", variant, thread, seed)
                    true_cpp, pred_cpp = load_ablation_preds(
                        logdir, "cpp_onnx_ablation", variant, thread, seed)
                    if not np.array_equal(true_py, true_cpp):
                        raise ValueError(
                            f"true label mismatch: {variant} seed={seed} thread={thread}")
                    same = np.array_equal(pred_py, pred_cpp)
                    acc = (pred_py == true_py).mean()
                    f.write(f"{variant},{seed},{thread},{same},{acc:.6f},{len(true_py)}\n")
    print(f"table -> {path}")


def plot_ablation_variant(rows, figs_dir, variant, threads):
    py_mean, py_std, cpp_mean, cpp_std, speedups = [], [], [], [], []
    for thread in threads:
        r = next(row for row in rows
                 if row["variant"] == variant and row["threads"] == thread)
        py_mean.append(r["py_mean_us"])
        py_std.append(r["py_mean_std_us"])
        cpp_mean.append(r["cpp_mean_us"])
        cpp_std.append(r["cpp_mean_std_us"])
        speedups.append(r["cpp_speedup_mean"])

    x = np.arange(len(threads))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, py_mean, width, yerr=py_std, capsize=3,
           label="Python ONNX")
    ax.bar(x + width / 2, cpp_mean, width, yerr=cpp_std, capsize=3,
           label="C++ ONNX")
    ymax = max(max(np.array(py_mean) + np.array(py_std)),
               max(np.array(cpp_mean) + np.array(cpp_std)))
    for i, speedup in enumerate(speedups):
        ax.text(x[i], ymax * 1.03, f"{speedup:.2f}x",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in threads])
    ax.set_xlabel("intra-op threads")
    ax.set_ylabel("mean latency (us)")
    ax.set_title(f"Tuning ablation: {VARIANT_DISPLAY[variant]}")
    ax.set_ylim(0, ymax * 1.18)
    ax.legend()
    save_fig(fig, figs_dir / f"latency_variant_{variant}.png")


def write_ablation_notes(results_root):
    path = results_root / "README.md"
    path.write_text(
        "# Tuning Ablation Notes\n\n"
        "This directory compares combinations of three ONNX Runtime settings:\n\n"
        "- Graph optimization: `ORT_ENABLE_ALL` vs `ORT_DISABLE_ALL`.\n"
        "- Named output: Python requests only `logits` instead of all outputs.\n"
        "- Memory reuse: CPU memory arena and memory pattern enabled vs disabled.\n\n"
        "The `none` variant matches the baseline ONNX Runtime policy: graph "
        "optimization and memory reuse are explicitly disabled, and Python does "
        "not request named output.\n\n"
        "C++ ONNX Runtime requires explicit output names for this single-output model, "
        "so the named-output flag is a real Python-side ablation and a recorded/no-op "
        "condition for C++.\n\n"
        "Each figure fixes one tuning variant and compares Python ONNX vs C++ ONNX "
        "across thread settings 1, 2, and 4. Bar heights are mean latency across "
        "seeds, error bars are seed standard deviation, and labels above each group "
        "show Python/C++ mean-latency speedup.\n"
    )
    print(f"note -> {path}")


def run_tuning_ablation(args, cfg):
    logdir = args.logdir or LOGS_DIR / "tuning_ablation"
    results_root = args.results_root or ROOT / "results" / "tuning_ablation"
    seeds = args.seeds or cfg["benchmark"]["seeds"]
    threads = args.threads or [1, 2, 4]
    variants = args.variants or VARIANTS

    figs_dir = results_root / "figures"
    tables_dir = results_root / "tables"
    figs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows = write_ablation_summary(logdir, tables_dir, seeds, threads, variants)
    write_ablation_fidelity(logdir, tables_dir, seeds, threads, variants)
    for variant in variants:
        plot_ablation_variant(rows, figs_dir, variant, threads)
    write_ablation_notes(results_root)


def backend_log_path(log_root, backend, thread, seed, suffix):
    if backend == "py_pytorch":
        return log_root / "backend_comparison" / f"py_pytorch_t{thread}_s{seed}_{suffix}.csv"
    return log_root / "tuning_ablation" / f"{backend}_t{thread}_s{seed}_{suffix}.csv"


def load_backend_latency(log_root, backend, thread, seed):
    return np.loadtxt(backend_log_path(log_root, backend, thread, seed, "latency"),
                      skiprows=1)


def load_backend_preds(log_root, backend, thread, seed):
    a = np.loadtxt(backend_log_path(log_root, backend, thread, seed, "preds"),
                   skiprows=1, delimiter=",", dtype=int)
    return a[:, 0], a[:, 1]


def collect_backend_rows(log_root, seeds, threads):
    rows = []
    for backend, label in BACKEND_ROWS:
        for thread in threads:
            means, medians, p95s = [], [], []
            for seed in seeds:
                v = load_backend_latency(log_root, backend, thread, seed)
                means.append(v.mean())
                medians.append(np.median(v))
                p95s.append(np.percentile(v, 95))
            means = np.array(means)
            medians = np.array(medians)
            p95s = np.array(p95s)
            rows.append({
                "backend": backend,
                "label": label,
                "threads": thread,
                "mean_us": means.mean(),
                "mean_std_us": sample_std(means),
                "median_us": medians.mean(),
                "median_std_us": sample_std(medians),
                "p95_us": p95s.mean(),
                "p95_std_us": sample_std(p95s),
                "throughput": 1e6 / means.mean(),
            })
    return rows


def collect_backend_overall(log_root, seeds, threads):
    rows = []
    for backend, label in BACKEND_ROWS:
        mean_vals, median_vals, p95_vals = [], [], []
        for thread in threads:
            for seed in seeds:
                v = load_backend_latency(log_root, backend, thread, seed)
                mean_vals.append(v.mean())
                median_vals.append(np.median(v))
                p95_vals.append(np.percentile(v, 95))
        mean_vals = np.array(mean_vals)
        median_vals = np.array(median_vals)
        p95_vals = np.array(p95_vals)
        rows.append({
            "backend": backend,
            "label": label,
            "mean_us": mean_vals.mean(),
            "mean_std_us": sample_std(mean_vals),
            "median_us": median_vals.mean(),
            "median_std_us": sample_std(median_vals),
            "p95_us": p95_vals.mean(),
            "p95_std_us": sample_std(p95_vals),
            "throughput": 1e6 / mean_vals.mean(),
        })
    return rows


def write_backend_csv(rows, path):
    with open(path, "w") as f:
        f.write("backend,label,threads,mean_us,mean_std_us,median_us,median_std_us,"
                "p95_us,p95_std_us,throughput\n")
        for r in rows:
            f.write(f"{r['backend']},{r['label']},{r['threads']},"
                    f"{r['mean_us']:.2f},{r['mean_std_us']:.2f},"
                    f"{r['median_us']:.2f},{r['median_std_us']:.2f},"
                    f"{r['p95_us']:.2f},{r['p95_std_us']:.2f},"
                    f"{r['throughput']:.0f}\n")
    print(f"table -> {path}")


def write_backend_overall_md(rows, path):
    by_backend = {r["backend"]: r for r in rows}
    py_onnx = by_backend["py_onnx_ablation_all"]
    cpp = by_backend["cpp_onnx_ablation_all"]
    pytorch = by_backend["py_pytorch"]
    with open(path, "w") as f:
        f.write("# Backend Comparison Summary\n\n")
        f.write("Values are averaged across matched seed/thread conditions. "
                "Standard deviation is computed across those matched conditions.\n\n")
        f.write("ONNX rows use the final common ONNX Runtime setting from the "
                "`all` tuning-ablation variant. PyTorch eager is not an ONNX "
                "Runtime backend, so only the same seed/thread sweep is applied.\n\n")
        f.write("| Backend | Mean latency | Median latency | p95 latency | "
                "Speedup vs PyTorch | Speedup vs Python ONNX |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for r in rows:
            speed_pt = pytorch["mean_us"] / r["mean_us"]
            speed_py_onnx = py_onnx["mean_us"] / r["mean_us"]
            f.write(f"| {r['label']} | {r['mean_us']:.2f} +/- {r['mean_std_us']:.2f} us | "
                    f"{r['median_us']:.2f} +/- {r['median_std_us']:.2f} us | "
                    f"{r['p95_us']:.2f} +/- {r['p95_std_us']:.2f} us | "
                    f"{speed_pt:.2f}x | {speed_py_onnx:.2f}x |\n")
        f.write("\n")
        f.write("## Key Takeaway\n\n")
        f.write(f"Compared with PyTorch eager, Python ONNX is "
                f"{pytorch['mean_us'] / py_onnx['mean_us']:.2f}x faster on mean "
                f"latency, and C++ ONNX is {pytorch['mean_us'] / cpp['mean_us']:.2f}x "
                f"faster. Under the same final ONNX Runtime setting, C++ ONNX is "
                f"{py_onnx['mean_us'] / cpp['mean_us']:.2f}x faster than Python ONNX "
                f"on mean latency, with an absolute mean-latency gap of "
                f"{py_onnx['mean_us'] - cpp['mean_us']:.2f} us.\n")
    print(f"table -> {path}")


def write_backend_fidelity(log_root, tables_dir, seeds, threads):
    path = tables_dir / "backend_comparison_fidelity.csv"
    with open(path, "w") as f:
        f.write("seed,threads,py_pytorch_vs_py_onnx,py_onnx_vs_cpp_onnx,accuracy,count\n")
        for seed in seeds:
            for thread in threads:
                true_pt, pred_pt = load_backend_preds(log_root, "py_pytorch", thread, seed)
                true_py, pred_py = load_backend_preds(
                    log_root, "py_onnx_ablation_all", thread, seed)
                true_cpp, pred_cpp = load_backend_preds(
                    log_root, "cpp_onnx_ablation_all", thread, seed)
                if not (np.array_equal(true_pt, true_py) and
                        np.array_equal(true_py, true_cpp)):
                    raise ValueError(f"true label mismatch: seed={seed} thread={thread}")
                pt_py_same = np.array_equal(pred_pt, pred_py)
                py_cpp_same = np.array_equal(pred_py, pred_cpp)
                acc = (pred_py == true_py).mean()
                f.write(f"{seed},{thread},{pt_py_same},{py_cpp_same},"
                        f"{acc:.6f},{len(true_py)}\n")
    print(f"table -> {path}")


def run_backend_comparison(args, cfg):
    log_root = args.log_root or LOGS_DIR
    results_root = args.results_root or ROOT / "results" / "backend_comparison"
    seeds = args.seeds or cfg["benchmark"]["seeds"]
    threads = args.threads or [1, 2, 4]
    tables_dir = results_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_backend_rows(log_root, seeds, threads)
    overall = collect_backend_overall(log_root, seeds, threads)
    write_backend_csv(rows, tables_dir / "backend_comparison_by_thread.csv")
    write_backend_overall_md(overall, tables_dir / "backend_comparison_summary.md")
    write_backend_fidelity(log_root, tables_dir, seeds, threads)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "tuning-ablation",
                                       "backend-comparison"],
                    default="baseline")
    ap.add_argument("--logdir", type=Path, default=None)
    ap.add_argument("--log-root", type=Path, default=None)
    ap.add_argument("--results-root", type=Path, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--threads", type=int, nargs="+", default=None)
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=None)
    args = ap.parse_args()

    if args.mode == "baseline":
        run_baseline(args, cfg)
    elif args.mode == "tuning-ablation":
        run_tuning_ablation(args, cfg)
    elif args.mode == "backend-comparison":
        run_backend_comparison(args, cfg)
    else:
        raise ValueError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
