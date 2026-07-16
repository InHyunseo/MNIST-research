"""Latency measurement and stage-local CSV logging."""

import csv
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from .codec import digits_to_string


def _percentile(sorted_values: list[float], quantile: float) -> float:
    index = min(len(sorted_values) - 1, int(quantile * len(sorted_values)))
    return sorted_values[index]


def benchmark_latency(
    infer: Callable[[Any], Any], inputs: Sequence[Any], warmup: int, n: int
) -> dict[str, Any]:
    times_us: list[float] = []
    for index in range(warmup + n):
        sample = inputs[index % len(inputs)]
        start = time.perf_counter()
        infer(sample)
        end = time.perf_counter()
        if index >= warmup:
            times_us.append((end - start) * 1e6)
    times_us.sort()
    mean = sum(times_us) / len(times_us)
    return {
        "mean": mean,
        "median": _percentile(times_us, 0.50),
        "p95": _percentile(times_us, 0.95),
        "min": times_us[0],
        "throughput": 1e6 / mean,
        "times": times_us,
    }


def print_report(label: str, metrics: Any, stats: dict[str, Any]) -> None:
    print(f"[{label}] digit_accuracy={metrics.digit_accuracy:.4f} "
          f"exact_match={metrics.exact_match:.4f}")
    print(f"  latency us : mean={stats['mean']:.1f} median={stats['median']:.1f} "
          f"p95={stats['p95']:.1f} min={stats['min']:.1f}")
    print(f"  throughput = {stats['throughput']:.0f} seq/s")


def write_latency_csv(path: Path, times_us: Sequence[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["latency_us"])
        writer.writerows((f"{latency:.3f}",) for latency in times_us)


def write_prediction_csv(path: Path, labels: Any, predictions: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["true", "pred", "correct"])
        for truth, prediction in zip(labels, predictions):
            true_text = digits_to_string(truth)
            pred_text = digits_to_string(prediction)
            writer.writerow([true_text, pred_text, int(true_text == pred_text)])
