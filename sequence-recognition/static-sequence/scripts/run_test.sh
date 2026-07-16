#!/usr/bin/env bash
set -euo pipefail
STAGE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$STAGE_ROOT/.." && pwd)"
PY="${PY:-$REPO_ROOT/.venv/bin/python}"
cd "$STAGE_ROOT"

PYTHONPATH=python "$PY" -m compileall -q python tests
PYTHONPATH=python "$PY" -m unittest discover -s tests -v
