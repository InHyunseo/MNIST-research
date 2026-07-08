#!/usr/bin/env bash
# 5 seed checkpoint -> ONNX export + 검증.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for s in 0 1 2 3 4; do
  "$PY" python/export_onnx.py --seed "$s"
done
