#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30080}"

# space-separated list:
WORKERS="${WORKERS:-http://127.0.0.1:10090 http://127.0.0.1:10091}"

HEALTH_INT="${HEALTH_INT:-1}"
FAIL_THRESH="${FAIL_THRESH:-2}"
ALGO="${ALGO:-least-request}"
TIMEOUT="${TIMEOUT:-}"
MAX_CONN="${MAX_CONN:-100}"
VERBOSE="${VERBOSE:-1}"

cmd=(sglang-d-router-demo
  --host "$HOST" --port "$PORT"
  --worker-urls $WORKERS
  --health-check-interval "$HEALTH_INT"
  --health-check-failure-threshold "$FAIL_THRESH"
  --routing-algorithm "$ALGO"
  --max-connections "$MAX_CONN"
)

if [[ -n "$TIMEOUT" ]]; then
  cmd+=(--timeout "$TIMEOUT")
fi
if [[ "$VERBOSE" == "1" ]]; then
  cmd+=(--verbose)
fi

echo "[router] ${cmd[*]}"
exec "${cmd[@]}"
