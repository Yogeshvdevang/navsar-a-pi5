#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.."; pwd)"
PYTHON_BIN="python3"
if [ -x "$ROOT_DIR/venv/bin/python" ]; then
  PYTHON_BIN="$ROOT_DIR/venv/bin/python"
fi

export PYTHONPATH="$ROOT_DIR/src"
exec "$PYTHON_BIN" -m navisar.main
