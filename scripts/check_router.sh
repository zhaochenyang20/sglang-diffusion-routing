#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-http://127.0.0.1:30080}"

echo "== /health =="
curl -s "$BASE/health" || true
echo
echo "== /list_workers =="
curl -s "$BASE/list_workers" || true
echo
echo "== /health_workers =="
curl -s "$BASE/health_workers" || true
echo