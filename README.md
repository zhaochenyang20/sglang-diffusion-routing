# sglang-diffusion-routing

A lightweight router for SGLang diffusion workers.

It provides worker registration, load balancing, health checking, and request proxying for diffusion generation APIs.

## Highlights

- `least-request` routing by default, with `round-robin` and `random`.
- Background health checks with quarantine after repeated failures.
- Router APIs for worker registration, health inspection, and proxy forwarding.
- `update_weights_from_disk` broadcast to all healthy workers.

## Installation

From repository root:

```bash
# Create a virtual environment
# python3 -m venv .venv
# source .venv/bin/activate
# pip install uv
git clone --recursive https://github.com/sglang/sglang-diffusion-routing.git
cd sglang-diffusion-routing
uv pip install .
```

Workers require SGLang diffusion support:

```bash
# If cloned sglang-diffusion-routing without --recursive, run:
# git submodule update --init --recursive
cd sglang
uv pip install "sglang[diffusion]" --prerelease=allow
cd ..
```

## Quick Start

### Start diffusion workers

```bash
# If connect to HuggingFace is not allowed
# You can set the environment variable SGLANG_USE_MODELSCOPE=TRUE

# worker 1
CUDA_VISIBLE_DEVICES=0 sglang serve \
    --model-path Qwen/Qwen-Image \
    --num-gpus 1 \
    --host 127.0.0.1 \
    --port 30000

# worker 2
CUDA_VISIBLE_DEVICES=1 sglang serve \
    --model-path Qwen/Qwen-Image \
    --num-gpus 1 \
    --host 127.0.0.1 \
    --port 30001
```

### Start the router

1. Script entry

```bash
sglang-d-router --port 30081 \
    --worker-urls http://localhost:30000 http://localhost:30001
```

2. Module entry

```bash
python -m sglang_diffusion_routing --port 30081 \
    --worker-urls http://localhost:30000 http://localhost:30001
```

3. Or start empty and add workers later:

```bash
sglang-d-router --port 30081
curl -X POST "http://localhost:30081/add_worker?url=http://localhost:30000"
```

### Test the router

```bash
# Check router health
curl http://localhost:30081/health

# List registered workers
curl http://localhost:30081/list_workers

# Image generation request (returns base64-encoded image)
curl -X POST http://localhost:30081/generate \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen-Image",
        "prompt": "a cute cat",
        "num_images": 1,
        "response_format": "b64_json"
    }'

# Decode and save the image locally
curl -s -X POST http://localhost:30081/generate \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen-Image",
        "prompt": "a cute cat",
        "num_images": 1,
        "response_format": "b64_json"
    }' | python3 -c "
import sys, json, base64
resp = json.load(sys.stdin)
img = base64.b64decode(resp['data'][0]['b64_json'])
with open('output.png', 'wb') as f:
    f.write(img)
print('Saved to output.png')
"


curl -X POST http://localhost:30081/update_weights_from_disk \
    -H "Content-Type: application/json" \
    -d '{"model_path": "Qwen/Qwen-Image-2512"}'
```

## Router API

- `POST /add_worker`: add worker via query (`?url=`) or JSON body.
- `GET /list_workers`: list registered workers.
- `GET /health`: aggregated router health.
- `GET /health_workers`: per-worker health and active request counts.
- `POST /generate`: forwards to worker `/v1/images/generations`.
- `POST /generate_video`: forwards to worker `/v1/videos`; rejects image-only workers (`T2I`/`I2I`/`TI2I`) with `400`.
- `POST /update_weights_from_disk`: broadcast to healthy workers.
- `GET|POST|PUT|DELETE /{path}`: catch-all proxy forwarding.
- `POST /update_weights_from_disk`: broadcast to all healthy workers.

## Benchmark Scripts

Benchmark scripts are available under `tests/benchmarks/diffusion_router/` and are intended for manual runs.
They are not part of default unit test collection (`pytest tests/unit -v`).

Single benchmark:

```bash
python tests/benchmarks/diffusion_router/bench_router.py \
    --model Qwen/Qwen-Image \
    --num-workers 2 \
    --num-prompts 20 \
    --max-concurrency 4
```

Algorithm comparison:

```bash
python tests/benchmarks/diffusion_router/bench_routing_algorithms.py \
    --model Qwen/Qwen-Image \
    --num-workers 2 \
    --num-prompts 20 \
    --max-concurrency 4
```

## Project Layout

```text
.
├── docs/
│   └── update_weights_from_disk.md
├── src/sglang_diffusion_routing/
│   ├── cli/
│   └── router/
├── tests/
│   ├── benchmarks/
│   │   └── diffusion_router/
│   │       ├── bench_router.py
│   │       └── bench_routing_algorithms.py
│   └── unit/
├── pyproject.toml
└── README.md
```

## Acknowledgment

This project is derived from [radixark/miles#544](https://github.com/radixark/miles/pull/544). Thanks to the original authors for their work.

## Notes

- Quarantined workers are intentionally not auto-reintroduced.
- Router responses are fully buffered; streaming passthrough is not implemented.
