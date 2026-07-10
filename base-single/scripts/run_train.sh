#!/usr/bin/env bash
# 5 seed 학습 (seed 0만 학습곡선 로그). 하이퍼파라미터는 configs/cnn.yaml.
set -e
STAGE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$STAGE_ROOT/.." && pwd)"
cd "$STAGE_ROOT"
PY="${PY:-$REPO_ROOT/.venv/bin/python}"
SEEDS="0 1 2 3 4"
for s in $SEEDS; do
  if [ "$s" = "0" ]; then
    "$PY" python/train.py --seed "$s" --curve
  else
    "$PY" python/train.py --seed "$s"
  fi
done
