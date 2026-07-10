#!/usr/bin/env bash
set -euo pipefail
STAGE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$STAGE_ROOT/.." && pwd)"
PY="${PY:-$REPO_ROOT/.venv/bin/python}"
cd "$STAGE_ROOT"

config_value() {
  PYTHONPATH=python "$PY" -m static_sequence_core.config "$1"
}

N="${N:-$(config_value benchmark.n)}"
WARMUP="${WARMUP:-$(config_value benchmark.warmup)}"
THREADS="${THREADS:-$(config_value benchmark.threads)}"

"$PY" python/dump_data.py
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j

"$PY" python/benchmark_pytorch.py --n "$N" --warmup "$WARMUP" --threads "$THREADS"
"$PY" python/benchmark_onnx.py --n "$N" --warmup "$WARMUP" --threads "$THREADS"
cpp/build/static_sequence_bench \
  --n "$N" \
  --warmup "$WARMUP" \
  --threads "$THREADS"
"$PY" python/verify_fidelity.py
