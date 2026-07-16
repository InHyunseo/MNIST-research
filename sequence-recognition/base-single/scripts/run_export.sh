#!/usr/bin/env bash
# 5 seed checkpoint -> ONNX export + 검증.
set -e
STAGE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$STAGE_ROOT/.." && pwd)"
cd "$STAGE_ROOT"
PY="${PY:-$REPO_ROOT/.venv/bin/python}"
for s in 0 1 2 3 4; do
  "$PY" python/export_onnx.py --seed "$s"
done
