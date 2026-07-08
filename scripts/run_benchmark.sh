#!/usr/bin/env bash
# Unified benchmark runner.
# Modes:
#   baseline            -> logs/
#   tuning-ablation     -> logs/tuning_ablation/
#   backend-comparison  -> logs/backend_comparison/
set -e
cd "$(dirname "$0")/.."

MODE="${1:-baseline}"
PY=.venv/bin/python
N="${N:-2000}"
SEEDS="${SEEDS:-0 1 2 3 4}"
THREADS="${THREADS:-1 2 4}"
VARIANTS="${VARIANTS:-none graph named memory graph_named graph_memory named_memory all}"

build_cpp() {
  cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build cpp/build -j >/dev/null
}

case "$MODE" in
  baseline)
    LOGDIR="${LOGDIR:-logs}"
    BASE_THREADS="${BASE_THREADS:-1}"
    build_cpp
    mkdir -p "$LOGDIR"
    "$PY" python/dump_data.py
    for s in $SEEDS; do
      "$PY" python/benchmark_pytorch.py --seed "$s" --threads "$BASE_THREADS" --n "$N" --logdir "$LOGDIR"
      "$PY" python/benchmark_onnx.py --seed "$s" --threads "$BASE_THREADS" --n "$N" --logdir "$LOGDIR"
      cpp/build/bench --seed "$s" --threads "$BASE_THREADS" --n "$N" --logdir "$LOGDIR"
    done
    echo "=== done. baseline logs in $LOGDIR ==="
    ;;

  tuning-ablation)
    LOGDIR="${LOGDIR:-logs/tuning_ablation}"
    build_cpp
    mkdir -p "$LOGDIR"
    "$PY" python/dump_data.py
    for variant in $VARIANTS; do
      for s in $SEEDS; do
        for t in $THREADS; do
          "$PY" python/benchmark_onnx.py \
            --variant "$variant" --seed "$s" --threads "$t" --n "$N" --logdir "$LOGDIR"
          cpp/build/bench \
            --variant "$variant" --seed "$s" --threads "$t" --n "$N" --logdir "$LOGDIR"
        done
      done
    done
    echo "=== done. ablation logs in $LOGDIR ==="
    ;;

  backend-comparison)
    LOGDIR="${LOGDIR:-logs/backend_comparison}"
    mkdir -p "$LOGDIR"
    for s in $SEEDS; do
      for t in $THREADS; do
        "$PY" python/benchmark_pytorch.py \
          --threaded-label --seed "$s" --threads "$t" --n "$N" --logdir "$LOGDIR"
      done
    done
    echo "=== done. PyTorch logs in $LOGDIR ==="
    ;;

  *)
    echo "unknown mode: $MODE" >&2
    echo "usage: bash scripts/run_benchmark.sh [baseline|tuning-ablation|backend-comparison]" >&2
    exit 2
    ;;
esac
