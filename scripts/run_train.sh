#!/usr/bin/env bash
# 5 seed 학습 (seed 0만 학습곡선 로그). 하이퍼파라미터는 configs/cnn.yaml.
set -e
cd "$(dirname "$0")/.."          # project root
PY=.venv/bin/python
SEEDS="0 1 2 3 4"
for s in $SEEDS; do
  if [ "$s" = "0" ]; then
    "$PY" python/train.py --seed "$s" --curve
  else
    "$PY" python/train.py --seed "$s"
  fi
done
