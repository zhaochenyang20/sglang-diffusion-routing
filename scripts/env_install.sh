#!/usr/bin/env bash
set -euo pipefail

# Install this repo in editable mode + dev deps.
# Usage: bash scripts/env_install.sh

python -m pip install -U pip setuptools wheel
python -m pip install -e ".[dev]"

echo "[ok] installed editable + dev deps"
python -c "import sglang_d_router; print('sglang_d_router:', sglang_d_router.__file__)"
command -v sglang-d-router-demo >/dev/null && echo "[ok] CLI: sglang-d-router-demo" || echo "[warn] CLI not found"
