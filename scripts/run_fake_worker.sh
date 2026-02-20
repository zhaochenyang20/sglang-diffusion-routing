#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-10090}"
APP="${APP:-examples.fake_worker_app:app}"

echo "[fake-worker] starting ${APP} at ${HOST}:${PORT}"
exec uvicorn "$APP" --host "$HOST" --port "$PORT" --log-level info
