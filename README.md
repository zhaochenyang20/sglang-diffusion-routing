# SGLang Diffusion Router

A lightweight router for SGLang diffusion workers used in RL systems. It provides worker registration, load balancing, health checking, refit weights and request proxying for diffusion generation APIs.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Start diffusion workers](#start-diffusion-workers)
  - [Start the router](#start-the-router)
- [Router API](#router-api)
  - [Inference Endpoints](#inference-endpoints)
  - [Videos Result Query](#videos-result-query)
  - [Model Discovery and Health Checks](#model-discovery-and-health-checks)
  - [Worker Management APIs](#worker-management-apis)
  - [RL Related API](#rl-related-api)
- [Acknowledgment](#acknowledgment)


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

### Co-Launch Workers and Router

Instead of starting workers manually, you can let the router spawn and manage them via a YAML config file.

```bash
sglang-d-router --port 30081 --launcher-config examples/local_launcher.yaml
```

```yaml
launcher:
  backend: local
  model: Qwen/Qwen-Image
  num_workers: 8
  num_gpus_per_worker: 1
  worker_base_port: 10090
  wait_timeout: 600
```

### Manual Launch Workers

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

# Register a worker
resp = requests.post(f"{ROUTER}/workers", json={"url": "http://localhost:30000"})
print(resp.json())

# List registered workers (with health/load)
resp = requests.get(f"{ROUTER}/workers")
print(resp.json())
worker_id = resp.json()["workers"][0]["worker_id"]

# Get / update worker details
resp = requests.get(f"{ROUTER}/workers/{worker_id}")
print(resp.json())
resp = requests.put(
    f"{ROUTER}/workers/{worker_id}",
    json={"is_dead": False, "refresh_video_support": True},
)
print(resp.json())

# Image generation request (returns base64-encoded image)
resp = requests.post(f"{ROUTER}/v1/images/generations", json={
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
# Note that Qwen-Image does not support video generation,
# so this request will fail.

resp = requests.post(f"{ROUTER}/v1/videos", json={
    "model": "Qwen/Qwen-Image",
    "prompt": "a flowing river",
})
print(resp.json())
video_id = resp.json().get("video_id") or resp.json().get("id")
if video_id:
    print(requests.get(f"{ROUTER}/v1/videos/{video_id}").json())

# Update weights from disk
resp = requests.post(f"{ROUTER}/update_weights_from_disk", json={
    "model_path": "Qwen/Qwen-Image-2512",
})
print(resp.json())
```

### With Curl

```bash
# Check router health
curl http://localhost:30081/health

# Register a worker
curl -X POST http://localhost:30081/workers \
    -H "Content-Type: application/json" \
    -d '{"url": "http://localhost:30000"}'

# List registered workers (with health/load)
curl http://localhost:30081/workers

# Image generation request (returns base64-encoded image)
curl -X POST http://localhost:30081/v1/images/generations \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen-Image",
        "prompt": "a cute cat",
        "num_images": 1,
        "response_format": "b64_json"
    }'

# Decode and save the image locally
curl -s -X POST http://localhost:30081/v1/images/generations \
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

# Video generation request
curl -X POST http://localhost:30081/v1/videos \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen/Qwen-Image", "prompt": "a flowing river"}'

# Poll a specific video job by video_id
curl http://localhost:30081/v1/videos/{video_id}


curl -X POST http://localhost:30081/update_weights_from_disk \
    -H "Content-Type: application/json" \
    -d '{"model_path": "Qwen/Qwen-Image-2512"}'
```

## Router API

### Inference Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/images/generations` | Entrypoint for text-to-image generation |
| `POST` | `/v1/videos` | Entrypoint for text-to-video generation |

### Videos Result Query

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/videos` | List or poll video jobs |
| `GET` | `/v1/videos/{video_id}` | Get status/details of a single video job |
| `GET` | `/v1/videos/{video_id}/content` | Download generated video content |

Video query routing is stable by `video_id`: router caches `video_id -> worker` on create (`POST /v1/videos`), then forwards detail/content queries to the same worker. Unknown `video_id` returns `404`.

### Model Discovery and Health Checks

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/models` | OpenAI-style model discovery |
| `GET` | `/health` | Basic health probe |

`GET /v1/models` aggregates model lists from healthy workers and de-duplicates by model `id`.

### Worker Management APIs

| Method | Path | Description |
|---|---|---|
| `POST` | `/workers` | Register a worker |
| `GET` | `/workers` | List workers (including health/load) |
| `GET` | `/workers/{worker_id}` | Get worker details |
| `PUT` | `/workers/{worker_id}` | Update worker configuration |
| `DELETE` | `/workers/{worker_id}` | Deregister a worker |

`worker_id` is the URL-encoded worker URL (machine-oriented), and each worker payload also includes `display_id` as a human-readable ID.

`PUT /workers/{worker_id}` currently supports:
- `is_dead` (boolean): quarantine (`true`) or recover (`false`) this worker.
- `refresh_video_support` (boolean): re-probe worker `/v1/models` capability.

### RL Related API

| Method | Path | Description |
|---|---|---|
| `POST` | `/update_weights_from_disk` | Reload weights from disk on all healthy workers |
| `POST` | `/release_memory_occupation` | Broadcast sleep to all healthy workers (release GPU memory occupation) |
| `POST` | `/resume_memory_occupation` | Broadcast wake to all healthy workers (resume GPU memory occupation) |

## Acknowledgment

This project is derived from [radixark/miles#544](https://github.com/radixark/miles/pull/544). Thanks to the original authors.

SGLang Diffusion RL team is responsible for the development and maintenance of this project. Our team mates in alphabetical order:

Banghua Zhu, Chengliang Qian, Chenyang Zhao, Fenglin Yu, Hao Jin, Huapeng Zhou, Jiajun Li, Kangrui Du, Kun Lin, Mao Cheng, Mengyang Liu, Qiujiang Chen, Shenggui Li, Shirui Chen, Shuwen Wang, Xi Chen, Xiaole Guo, Ying Sheng, Yueming Yuan, Yuhao Yang, Yusheng Su, Zhiheng Ye
