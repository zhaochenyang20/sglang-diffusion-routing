# SGLang Diffusion Router

A lightweight router for SGLang diffusion workers used in RL systems.

It provides worker registration, load balancing, health checking, refit weights and request proxying for diffusion generation APIs.

## API Reference

- `POST /add_worker`: add worker via query (`?url=`) or JSON body.
- `GET /list_workers`: list registered workers.
- `GET /health`: aggregated router health.
- `GET /health_workers`: per-worker health and active request counts.
- `POST /generate`: forwards to worker `/v1/images/generations`.
- `POST /generate_video`: forwards to worker `/v1/videos`; rejects image-only workers (`T2I`/`I2I`/`TI2I`) with `400`.
- `POST /update_weights_from_disk`: broadcast to all healthy workers.
- `GET|POST|PUT|DELETE /{path}`: catch-all proxy forwarding.

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
    --port 30002

sglang-d-router --port 30081 \
    --worker-urls http://localhost:30000 http://localhost:30002
```

## Demonstrative Examples


### With Python Requests

```python
import requests
import base64

ROUTER = "http://localhost:30081"

# Check router health
resp = requests.get(f"{ROUTER}/health")
print(resp.json())

# List registered workers
resp = requests.get(f"{ROUTER}/list_workers")
print(resp.json())

# Image generation request (returns base64-encoded image)
resp = requests.post(f"{ROUTER}/generate", json={
    "model": "Qwen/Qwen-Image",
    "prompt": "a cute cat",
    "num_images": 1,
    "response_format": "b64_json",
})
data = resp.json()
print(data)

# Decode and save the image locally
img = base64.b64decode(data["data"][0]["b64_json"])
with open("output.png", "wb") as f:
    f.write(img)
print("Saved to output.png")

# Video generation request
resp = requests.post(f"{ROUTER}/generate_video", json={
    "model": "Qwen/Qwen-Image",
    "prompt": "a flowing river",
})
print(resp.json())

# Check per-worker health and load
resp = requests.get(f"{ROUTER}/health_workers")
print(resp.json())

# Update weights from disk
resp = requests.post(f"{ROUTER}/update_weights_from_disk", json={
    "model_path": "Qwen/Qwen-Image-2512",
})
print(resp.json())
```

### With Curl
3. Or start empty and add workers later:

```bash
sglang-d-router --port 30081
curl -X POST "http://localhost:30081/add_worker?url=http://localhost:30000"
```

### Auto-launch workers via YAML config

Instead of starting workers manually, you can let the router spawn and manage
them through a launcher backend.

**Local subprocess launcher** (`examples/local_launcher.yaml`):

```bash
sglang-d-router --port 30081 --launcher-config examples/local_launcher.yaml
```

```yaml
launcher:
  backend: local
  model: Qwen/Qwen-Image
  num_workers: 2
  num_gpus_per_worker: 1
  worker_base_port: 10090
  wait_timeout: 600
```

Fields not set in the YAML fall back to defaults defined in each backend's
config dataclass (see `LocalLauncherConfig`).

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
SGLANG_USE_MODELSCOPE=TRUE python tests/benchmarks/diffusion_router/bench_router.py \
    --model Qwen/Qwen-Image \
    --num-workers 2 \
    --num-prompts 20 \
    --max-concurrency 4
```

Algorithm comparison:

```bash
SGLANG_USE_MODELSCOPE=TRUE python tests/benchmarks/diffusion_router/bench_routing_algorithms.py \
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
├── examples/
│   ├── local_launcher.yaml
│   └── ray_launcher.yaml
├── src/sglang_diffusion_routing/
│   ├── cli/
│   ├── launcher/          # local / ray backend implementations
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

This project is derived from [radixark/miles#544](https://github.com/radixark/miles/pull/544). Thanks to the original authors.

SGLang Diffusion RL team is responsible for the development and maintenance of this project. Our team mates in alphabetical order:

Banghua Zhu, Chengliang Qian, Chenyang Zhao, Fenglin Yu, Hao Jin, Huapeng Zhou, Jiajun Li, Kangrui Du, Kun Lin, Mao Cheng, Mengyang Liu, Qiujiang Chen, Shenggui Li, Shirui Chen, Shuwen Wang, Xi Chen, Xiaole Guo, Ying Sheng, Yueming Yuan, Yuhao Yang, Yusheng Su, Zhiheng Ye
