#!/usr/bin/env bash
# Unified visualization runner.
# Modes: baseline, tuning-ablation, backend-comparison
set -e
cd "$(dirname "$0")/.."

MODE="${1:-baseline}"
SEEDS="${SEEDS:-0 1 2 3 4}"
THREADS="${THREADS:-1 2 4}"
VARIANTS="${VARIANTS:-none graph named memory graph_named graph_memory named_memory all}"

case "$MODE" in
  baseline)
    .venv/bin/python python/visualize.py --mode baseline
    ;;
  tuning-ablation)
    .venv/bin/python python/visualize.py --mode tuning-ablation \
      --seeds $SEEDS --threads $THREADS --variants $VARIANTS
    ;;
  backend-comparison)
    .venv/bin/python python/visualize.py --mode backend-comparison \
      --seeds $SEEDS --threads $THREADS
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    echo "usage: bash scripts/run_visualize.sh [baseline|tuning-ablation|backend-comparison]" >&2
    exit 2
    ;;
esac
