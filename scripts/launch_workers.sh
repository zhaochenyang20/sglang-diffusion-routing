#!/usr/bin/env bash
set -euo pipefail

# Launch N diffusion workers by reusing run_model_serve_1.sh.
#
# Examples:
#   bash scripts/launch_workers.sh --n 2 --base-port 10090 --gpus 0,1
#   bash scripts/launch_workers.sh --n 4 --base-port 10090 --gpu-base 0
#   MODEL=runwayml/stable-diffusion-v1-5 bash tests/scripts/launch_workers.sh --n 2
#
# Stop:
#   bash tests/scripts/stop_workers.sh

HOST="127.0.0.1"
N="2"
BASE_PORT="10090"

# Either provide --gpus "0,1,2" OR provide --gpu-base 0 (auto 0..N-1)
GPUS=""
GPU_BASE="0"

RUN_SCRIPT="scripts/run_model_serve.sh"

OUT_DIR="./tmp/sglang_workers"
MODEL="${MODEL:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}"
DTYPE="${DTYPE:-bf16}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --base-port) BASE_PORT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --gpu-base) GPU_BASE="$2"; shift 2 ;;
    --model-path) MODEL="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --run-script) RUN_SCRIPT="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    -h|--help) sed -n '1,200p' "$0"; exit 0 ;;
    *) echo "[error] unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "$OUT_DIR"

if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "[error] run script not found: $RUN_SCRIPT"
  exit 1
fi

IFS=',' read -r -a GPU_LIST <<< "${GPUS:-}"

echo "[launcher] host=$HOST n=$N base_port=$BASE_PORT model=$MODEL dtype=$DTYPE out=$OUT_DIR"
echo "[launcher] run_script=$RUN_SCRIPT"

for i in $(seq 0 $((N-1))); do
  port=$((BASE_PORT + i))

  if [[ -n "$GPUS" ]]; then
    gpu="${GPU_LIST[$i]:-}"
    if [[ -z "$gpu" ]]; then
      echo "[error] --gpus provided but missing entry for worker#$i (need at least $N gpus)"
      exit 1
    fi
  else
    gpu=$((GPU_BASE + i))
  fi

  logfile="$OUT_DIR/worker_${port}.log"
  pidfile="$OUT_DIR/worker_${port}.pid"

  echo "[launcher] starting worker#$i gpu=$gpu port=$port log=$logfile"

  # Export per-worker env vars and start the worker script in background.
  nohup env \
    HOST="$HOST" \
    PORT="$port" \
    MODEL="$MODEL" \
    DTYPE="$DTYPE" \
    CUDA_VISIBLE_DEVICES="$gpu" \
    HF_HOME="${HF_HOME:-/data/cache/huggingface}" \
    bash "$RUN_SCRIPT" \
    >"$logfile" 2>&1 &

  echo $! >"$pidfile"
done

echo "[ok] launched $N workers."
echo "[ok] pids/logs in: $OUT_DIR"
echo "[hint] check: ls -l $OUT_DIR; tail -f $OUT_DIR/worker_${BASE_PORT}.log"
echo "[hint] stop: bash tests/scripts/stop_workers.sh --out-dir $OUT_DIR"
