"""Verify exact string equality across all three benchmark backends."""

import csv
from pathlib import Path

from static_sequence_core.config import LOGS_DIR

BACKENDS = ("py_pytorch", "py_onnx", "cpp_onnx")


def read_predictions(path: Path) -> list[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return [(row["true"], row["pred"]) for row in csv.DictReader(source)]


def main() -> None:
    baseline = read_predictions(LOGS_DIR / f"{BACKENDS[0]}_predictions.csv")
    for backend in BACKENDS[1:]:
        candidate = read_predictions(LOGS_DIR / f"{backend}_predictions.csv")
        if len(candidate) != len(baseline):
            raise SystemExit(
                f"prediction count mismatch for {backend}: {len(candidate)} vs {len(baseline)}"
            )
        if candidate != baseline:
            mismatch = next(
                index for index, pair in enumerate(zip(baseline, candidate)) if pair[0] != pair[1]
            )
            raise SystemExit(f"prediction mismatch for {backend} at row {mismatch}")
        print(f"{BACKENDS[0]} vs {backend}: 100.00% string fidelity ({len(baseline)} samples)")


if __name__ == "__main__":
    main()
