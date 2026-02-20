#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-10090}"
MODEL="${MODEL:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}"
DTYPE="${DTYPE:-bf16}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export CUDA_VISIBLE_DEVICES
export HF_HOME="${HF_HOME:-/data/cache/huggingface}"

echo "[gpu-worker] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES host=$HOST port=$PORT model=$MODEL dtype=$DTYPE"
exec sglang serve \
  --host "$HOST" \
  --port "$PORT" \
  --model-path "$MODEL" \
  --dtype "$DTYPE"
