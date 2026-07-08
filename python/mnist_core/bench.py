"""추론 latency 측정과 CSV 로깅. 모든 backend가 공유한다.

bench_latency: 미리 만들어 둔 batch=1 입력들에 infer_fn을 호출하며 시간을 잰다.
처음 warmup회는 버리고 이후 n회를 time.perf_counter로 측정한다(입력 생성은 측정 구간 밖).
CSV는 측정이 끝난 뒤 기록한다.
"""
import time


def percentile(sorted_vals, q):
    idx = min(len(sorted_vals) - 1, int(q * len(sorted_vals)))
    return sorted_vals[idx]


def bench_latency(infer_fn, inputs, warmup, n):
    m = len(inputs)
    times_us = []
    for i in range(warmup + n):
        x = inputs[i % m]
        t0 = time.perf_counter()
        infer_fn(x)
        t1 = time.perf_counter()
        if i >= warmup:
            times_us.append((t1 - t0) * 1e6)
    times_us.sort()
    mean = sum(times_us) / len(times_us)
    return {
        "mean": mean,
        "median": percentile(times_us, 0.50),
        "p95": percentile(times_us, 0.95),
        "min": times_us[0],
        "throughput": 1e6 / mean,
        "times": times_us,
    }


def print_report(label, threads, warmup, n, acc, stats):
    print(f"[{label}] threads={threads} warmup={warmup} n={n}")
    print(f"  accuracy   = {acc:.4f}")
    print(f"  latency us : mean={stats['mean']:.1f} median={stats['median']:.1f} "
          f"p95={stats['p95']:.1f} min={stats['min']:.1f}")
    print(f"  throughput = {stats['throughput']:.0f} inf/s")


def write_latency_csv(path, times_us):
    with open(path, "w") as f:
        f.write("latency_us\n")
        for t in times_us:
            f.write(f"{t:.3f}\n")


def write_preds_csv(path, trues, preds):
    with open(path, "w") as f:
        f.write("true,pred\n")
        for t, p in zip(trues, preds):
            f.write(f"{int(t)},{int(p)}\n")
