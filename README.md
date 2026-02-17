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
python3 -m venv .venv
. .venv/bin/activate
pip install .
```

Development install:

```bash
pip install -e .
```

Run tests:

```bash
pip install pytest
pytest tests/unit -v
```

Workers require SGLang diffusion support:

```bash
pip install "sglang[diffusion]"
```

## Quick Start

### 1) Start diffusion workers

```bash
# worker 1
CUDA_VISIBLE_DEVICES=0 sglang serve \
    --model-path stabilityai/stable-diffusion-3-medium-diffusers \
    --num-gpus 1 \
    --host 127.0.0.1 \
    --port 30000

# worker 2
CUDA_VISIBLE_DEVICES=1 sglang serve \
    --model-path stabilityai/stable-diffusion-3-medium-diffusers \
    --num-gpus 1 \
    --host 127.0.0.1 \
    --port 30001
```

### 2) Start the router

Script entry:

```bash
sglang-d-router --port 30080 \
    --worker-urls http://localhost:30000 http://localhost:30001
```

Module entry:

```bash
python -m sglang_diffusion_routing --port 30080 \
    --worker-urls http://localhost:30000 http://localhost:30001
```

Or start empty and add workers later:

```bash
sglang-d-router --port 30080
curl -X POST "http://localhost:30080/add_worker?url=http://localhost:30000"
```

### 3) Test the router

```bash
# Check router health
curl http://localhost:30080/health

# List registered workers
curl http://localhost:30080/list_workers

# Image generation request (SD3)
curl -X POST http://localhost:30080/generate \
    -H "Content-Type: application/json" \
    -d '{
        "model": "stabilityai/stable-diffusion-3-medium-diffusers",
        "prompt": "a cute cat",
        "num_images": 1
    }'

# Video generation request
curl -X POST http://localhost:30080/generate_video \
    -H "Content-Type: application/json" \
    -d '{
        "model": "stabilityai/stable-video-diffusion",
        "prompt": "a flowing river"
    }'

# Check per-worker health and load
curl http://localhost:30080/health_workers
```

## Router API

- `POST /add_worker`: add worker via query (`?url=`) or JSON body.
- `GET /list_workers`: list registered workers.
- `GET /health`: aggregated router health.
- `GET /health_workers`: per-worker health and active request counts.
- `POST /generate`: forwards to worker `/v1/images/generations`.
- `POST /generate_video`: forwards to worker `/v1/videos`.
- `POST /update_weights_from_disk`: broadcast to healthy workers.
- `GET|POST|PUT|DELETE /{path}`: catch-all proxy forwarding.

## `update_weights_from_disk` behavior

Full details: [docs/update_weights_from_disk.md](docs/update_weights_from_disk.md)

- The router forwards request payloads as-is to each healthy worker.
- The router does not validate payload schema; payload semantics are worker-defined.
- Worker servers must implement `POST /update_weights_from_disk`.

Example:

```bash
curl -X POST http://localhost:30080/update_weights_from_disk \
    -H "Content-Type: application/json" \
    -d '{"model_path": "/path/to/new/weights"}'
```

Response shape:

```json
{
  "results": [
    {
      "worker_url": "http://localhost:30000",
      "status_code": 200,
      "body": {
        "ok": true
      }
    }
  ]
}
```

## Benchmark Scripts

Benchmark scripts are available under `tests/benchmarks/diffusion_router/` and are intended for manual runs.
They are not part of default unit test collection (`pytest tests/unit -v`).

Single benchmark:

```bash
python tests/benchmarks/diffusion_router/bench_router.py \
    --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --num-workers 2 \
    --num-prompts 20 \
    --max-concurrency 4
```

Algorithm comparison:

```bash
python tests/benchmarks/diffusion_router/bench_routing_algorithms.py \
    --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
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
