"""logs/ CSV를 읽어 results/{figures,tables,samples}에 그래프와 표를 만든다.

벤치마크는 CSV만 쓰고 그리기는 여기서 한다(추론 경로에는 plotting을 넣지 않는다).
실행: python python/visualize.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mnist_core.config import load_config, LOGS_DIR
from mnist_core.metrics import confusion, per_class_prf
from mnist_core.dataset import load_test_tensors

ROOT = LOGS_DIR.parent
FIGS = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"
SAMPLES = ROOT / "results" / "samples"
for d in (FIGS, TABLES, SAMPLES):
    d.mkdir(parents=True, exist_ok=True)

BACKENDS = ["py_pytorch", "py_onnx", "cpp_onnx"]
SEEDS = load_config()["benchmark"]["seeds"]


def load_latency(backend, seed):
    return np.loadtxt(LOGS_DIR / f"{backend}_s{seed}_latency.csv", skiprows=1)


def load_preds(backend, seed):
    a = np.loadtxt(LOGS_DIR / f"{backend}_s{seed}_preds.csv", skiprows=1,
                   delimiter=",", dtype=int)
    return a[:, 0], a[:, 1]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / name, dpi=150)
    plt.close(fig)
    print(f"figure -> {name}")


def pct(sorted_v, q):
    return sorted_v[min(len(sorted_v) - 1, int(q * len(sorted_v)))]


# 1) latency box (backend별, 전 seed 합침)
data = [np.sort(np.concatenate([load_latency(b, s) for s in SEEDS])) for b in BACKENDS]
fig, ax = plt.subplots(figsize=(5, 4))
ax.boxplot(data, labels=BACKENDS, showfliers=False)
ax.set_ylabel("latency (us)")
save(fig, "latency_boxplot.png")

# 2) throughput bar (seed별 mean throughput -> mean +- std)
thr_mean = [np.mean([1e6 / load_latency(b, s).mean() for s in SEEDS]) for b in BACKENDS]
thr_std = [np.std([1e6 / load_latency(b, s).mean() for s in SEEDS]) for b in BACKENDS]
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(BACKENDS, thr_mean, yerr=thr_std, capsize=4)
ax.set_ylabel("throughput (inf/s)")
save(fig, "inference_time_bar.png")

# summary 표 + fidelity 체크
with open(TABLES / "benchmark_summary.csv", "w") as f:
    f.write("backend,mean_us,median_us,p95_us,throughput\n")
    for b, d in zip(BACKENDS, data):
        f.write(f"{b},{d.mean():.2f},{pct(d,.5):.2f},{pct(d,.95):.2f},{1e6/d.mean():.0f}\n")
print("table -> benchmark_summary.csv")
for s in SEEDS:
    p0 = load_preds(BACKENDS[0], s)[1]
    same = all((load_preds(b, s)[1] == p0).all() for b in BACKENDS[1:])
    print(f"seed {s}: 3 backend 예측 동일 = {same}")

# 3) confusion matrix (py_pytorch, 전 seed 합)
C = np.zeros((10, 10), dtype=int)
for s in SEEDS:
    t, p = load_preds("py_pytorch", s)
    C += confusion(t, p)
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(C, cmap="Blues")
ax.set_xlabel("pred")
ax.set_ylabel("true")
ax.set_xticks(range(10))
ax.set_yticks(range(10))
for i in range(10):
    for j in range(10):
        if C[i, j]:
            ax.text(j, i, C[i, j], ha="center", va="center", fontsize=6,
                    color="white" if i == j else "black")
save(fig, "confusion_matrix.png")

# 4) per-class accuracy & F1 (seed별 -> mean +- std)
accs, f1s = [], []
for s in SEEDS:
    t, p = load_preds("py_pytorch", s)
    recall, _, f1 = per_class_prf(confusion(t, p))
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
    save(fig, name)

# 5) 학습곡선 (seed 0): col1=train_loss, col2=test_acc
curve = np.loadtxt(LOGS_DIR / "train_curve.csv", skiprows=1, delimiter=",")
for col, name, ylab in [(1, "loss_curve.png", "train loss"),
                        (2, "accuracy_curve.png", "test accuracy")]:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(curve[:, 0], curve[:, col])
    ax.set_xlabel("step")
    ax.set_ylabel(ylab)
    save(fig, name)

# 6) 샘플 예측 (맞은/틀린 예시, py_pytorch seed0)
imgs, _ = load_test_tensors()
t, p = load_preds("py_pytorch", SEEDS[0])
for idxs, name in [(np.where(t == p)[0][:8], "correct_examples.png"),
                   (np.where(t != p)[0][:8], "wrong_examples.png")]:
    fig, axes = plt.subplots(1, len(idxs), figsize=(len(idxs) * 1.2, 1.6))
    for ax, i in zip(axes, idxs):
        ax.imshow(imgs[i, 0].numpy(), cmap="gray")
        ax.axis("off")
        ax.set_title(f"{t[i]}/{p[i]}", fontsize=7)
    fig.tight_layout()
    fig.savefig(SAMPLES / name, dpi=150)
    plt.close(fig)
    print(f"sample -> {name}")
