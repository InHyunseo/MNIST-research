#!/usr/bin/env bash
# logs/ -> results/{figures,tables,samples}
set -e
cd "$(dirname "$0")/.."
.venv/bin/python python/visualize.py
