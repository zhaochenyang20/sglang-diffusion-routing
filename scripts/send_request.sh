#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   bash scripts/send_request.sh
#   BASE=http://127.0.0.1:30080 REQ_PATH=/generate bash scripts/send_request.sh
#   BASE=http://127.0.0.1:30080 REQ_PATH=/v1/images/generations bash scripts/send_request.sh
#   BASE=http://127.0.0.1:10090 REQ_PATH=/health METHOD=GET bash scripts/send_request.sh
#
# Knobs:
#   PROMPT="a cat" SIZE=512x512 N=1 bash scripts/send_request.sh
#   WIDTH=512 HEIGHT=512 bash scripts/send_request.sh
#   MODEL=runwayml/stable-diffusion-v1-5 bash scripts/send_request.sh
#   RESPONSE_FORMAT=b64_json bash scripts/send_request.sh

BASE="${BASE:-http://127.0.0.1:30080}"
REQ_PATH="${REQ_PATH:-/generate}"
METHOD="${METHOD:-POST}"

PROMPT="${PROMPT:-a cat wearing sunglasses}"
N="${N:-1}"

# Either use SIZE="512x512" or WIDTH/HEIGHT
SIZE="${SIZE:-512x512}"
WIDTH="${WIDTH:-}"
HEIGHT="${HEIGHT:-}"

MODEL="${MODEL:-}"
RESPONSE_FORMAT="${RESPONSE_FORMAT:-b64_json}"  # key fix: avoid "url" which requires cloud storage

OUT="/tmp/send_request.out"
url="${BASE}${REQ_PATH}"

echo "[send] ${METHOD} ${url}"

if [[ "$METHOD" == "GET" ]]; then
  http_code="$(curl -sS -o "$OUT" -w "%{http_code}" --max-time 30 "$url" || true)"
  cat "$OUT" || true
  echo
  echo "[send] http_code=$http_code"
  [[ "$http_code" == "200" ]] || exit 1
  exit 0
fi

# Build JSON payload in bash (minimal escaping).
# If your prompt includes quotes/newlines, keep it simple for now or extend escaping.
prompt_escaped="${PROMPT//\\/\\\\}"
prompt_escaped="${prompt_escaped//\"/\\\"}"

payload="{\"prompt\":\"$prompt_escaped\",\"n\":$N,\"response_format\":\"$RESPONSE_FORMAT\""

if [[ -n "$MODEL" ]]; then
  model_escaped="${MODEL//\\/\\\\}"
  model_escaped="${model_escaped//\"/\\\"}"
  payload+=",\"model\":\"$model_escaped\""
fi

if [[ -n "$WIDTH" && -n "$HEIGHT" ]]; then
  payload+=",\"width\":$WIDTH,\"height\":$HEIGHT"
else
  size_escaped="${SIZE//\\/\\\\}"
  size_escaped="${size_escaped//\"/\\\"}"
  payload+=",\"size\":\"$size_escaped\""
fi

payload+="}"

echo "[send] payload=$payload"

http_code="$(curl -sS -o "$OUT" -w "%{http_code}" --max-time 300 \
  -X POST "$url" \
  -H "Content-Type: application/json" \
  -d "$payload" || true)"

cat "$OUT" || true
echo
echo "[send] http_code=$http_code"

if [[ "$http_code" != 2* ]]; then
  echo "[send] request failed (non-2xx)."
  exit 1
fi
