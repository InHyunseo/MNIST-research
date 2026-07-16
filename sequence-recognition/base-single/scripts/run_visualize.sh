#!/usr/bin/env bash
# Unified visualization runner.
# Modes: baseline, tuning-ablation, backend-comparison
set -e
STAGE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$STAGE_ROOT/.." && pwd)"
cd "$STAGE_ROOT"

MODE="${1:-baseline}"
PY="${PY:-$REPO_ROOT/.venv/bin/python}"
SEEDS="${SEEDS:-0 1 2 3 4}"
THREADS="${THREADS:-1 2 4}"
VARIANTS="${VARIANTS:-none graph named memory graph_named graph_memory named_memory all}"

case "$MODE" in
  baseline)
    "$PY" python/visualize.py --mode baseline
    ;;
  tuning-ablation)
    "$PY" python/visualize.py --mode tuning-ablation \
      --seeds $SEEDS --threads $THREADS --variants $VARIANTS
    ;;
  backend-comparison)
    "$PY" python/visualize.py --mode backend-comparison \
      --seeds $SEEDS --threads $THREADS
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    echo "usage: bash scripts/run_visualize.sh [baseline|tuning-ablation|backend-comparison]" >&2
    exit 2
    ;;
esac
