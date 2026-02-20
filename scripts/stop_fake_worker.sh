#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="./tmp/sglang_d_fake_workers"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    -h|--help) sed -n '1,120p' "$0"; exit 0 ;;
    *) echo "[error] unknown arg: $1"; exit 1 ;;
  esac
done

shopt -s nullglob
pids=("$OUT_DIR"/fake_*.pid)

if [[ ${#pids[@]} -eq 0 ]]; then
  echo "[stop] no pid files under $OUT_DIR"
  exit 0
fi

for f in "${pids[@]}"; do
  pid="$(cat "$f" || true)"
  if [[ -n "$pid" ]]; then
    echo "[stop] $f pid=$pid"
    kill "$pid" 2>/dev/null || true
  fi
  rm -f "$f"
done

echo "[ok] stopped fake workers."
