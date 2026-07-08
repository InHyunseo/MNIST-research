#!/usr/bin/env bash
# C++ 빌드 -> u8 덤프 -> 3 backend x 5 seed latency+preds 로그(logs/).
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOGDIR=logs
N=2000

cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build cpp/build -j >/dev/null

mkdir -p "$LOGDIR"
"$PY" python/dump_data.py

for s in 0 1 2 3 4; do
  "$PY" python/benchmark_pytorch.py --seed "$s"
  "$PY" python/benchmark_onnx.py --seed "$s"
  cpp/build/bench "$s" 1 "$N" "$LOGDIR"
done
echo "=== done. logs in $LOGDIR ==="
