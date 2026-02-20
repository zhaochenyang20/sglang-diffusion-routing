#!/usr/bin/env bash
set -euo pipefail

# Send a demo request to the router (or directly to a worker).
#
# Usage:
#   bash scripts/send_request.sh
#   BASE=http://127.0.0.1:30080 bash scripts/send_request.sh
#   BASE=http://127.0.0.1:30080 REQ_PATH=/v1/images/generations PROMPT="a cat" bash scripts/send_request.sh
#   BASE=http://127.0.0.1:10090 REQ_PATH=/health METHOD=GET bash scripts/send_request.sh
#
# Notes:
# - Default path is /v1/images/generations (recommended).
# - If you use /generate in your router/worker, set REQ_PATH=/generate.
#
# IMPORTANT:
# - Do NOT use PATH=... because PATH is a system variable required to locate commands (bash/curl/etc).

BASE="${BASE:-http://127.0.0.1:30080}"
REQ_PATH="${REQ_PATH:-/v1/images/generations}"

PROMPT="${PROMPT:-a cat wearing sunglasses}"
N="${N:-1}"
SIZE="${SIZE:-512x512}"

# For health check convenience:
METHOD="${METHOD:-POST}"   # GET/POST

url="${BASE}${REQ_PATH}"
OUT="/tmp/send_request.out"

echo "[send] ${METHOD} ${url}"

if [[ "$METHOD" == "GET" ]]; then
  http_code="$(curl -sS -o "$OUT" -w "%{http_code}" --max-time 30 "$url" || true)"
  cat "$OUT" || true
  echo
  echo "[send] http_code=$http_code"
  [[ "$http_code" == "200" ]] || exit 1
  exit 0
fi

payload=$(cat <<JSON
{"prompt":"${PROMPT}","n":${N},"size":"${SIZE}"}
JSON
)

http_code="$(curl -sS -o "$OUT" -w "%{http_code}" --max-time 300 \
  -X POST "$url" \
  -H "Content-Type: application/json" \
  -d "$payload" || true)"

cat "$OUT" || true
echo
echo "[send] http_code=$http_code"

if [[ "$http_code" != 2* ]]; then
  echo "[send] request failed (non-2xx). Tip: check router/worker logs."
  exit 1
fi
