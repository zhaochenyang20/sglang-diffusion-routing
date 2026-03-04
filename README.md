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
  model: stabilityai/stable-diffusion-3-medium-diffusers
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
    --model-path stabilityai/stable-diffusion-3-medium-diffusers \
    --num-gpus 1 \
    --host 127.0.0.1 \
    --port 30000

# worker 2
CUDA_VISIBLE_DEVICES=1 sglang serve \
    --model-path stabilityai/stable-diffusion-3-medium-diffusers \
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

# Image generation request (OpenAI-compatible, returns base64-encoded image)
resp = requests.post(f"{ROUTER}/v1/images/generations", json={
    "model": "stabilityai/stable-diffusion-3-medium-diffusers",
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
# Note that Stable-Diffusion-3 does not support video generation,
# so this request will fail. Use a video-capable model instead.

resp = requests.post(f"{ROUTER}/v1/videos", json={
    "model": "stabilityai/stable-diffusion-3-medium-diffusers",
    "prompt": "a flowing river",
})
print(resp.json())
video_id = resp.json().get("video_id") or resp.json().get("id")
if video_id:
    print(requests.get(f"{ROUTER}/v1/videos/{video_id}").json())

# Update weights from disk
resp = requests.post(f"{ROUTER}/update_weights_from_disk", json={
    "model_path": "/path/to/new/checkpoint",
})
print(resp.json())
```

### Native Diffusion Generate Endpoint (with Trajectory & Log-Prob)

The `/v1/diffusion/generate` endpoint exposes trajectory data (latents, timesteps)
and log-probabilities that the OpenAI-compatible endpoints intentionally omit.
This is intended for RL training pipelines that need intermediate diffusion outputs.

```python
import requests
import base64
import io
import numpy as np

ROUTER = "http://localhost:30081"

# Generate with trajectory latents and log-probs
resp = requests.post(f"{ROUTER}/v1/diffusion/generate", json={
    "prompt": "a cute cat",
    "width": 512,
    "height": 512,
    "num_inference_steps": 28,
    "guidance_scale": 7.0,
    "seed": 42,
    "get_latents": True,
    "get_log_probs": True,
})
data = resp.json()
print(f"Request ID: {data['id']}")
print(f"Inference time: {data.get('inference_time_s')}s")

# Decode the output image
if data.get("output_b64"):
    img_bytes = base64.b64decode(data["output_b64"])
    with open("output.png", "wb") as f:
        f.write(img_bytes)
    print(f"Saved image ({data.get('output_format', 'unknown')} format)")

# Decode trajectory data
trajectory = data.get("trajectory")
if trajectory:
    if trajectory.get("latents"):
        latents = np.load(io.BytesIO(base64.b64decode(trajectory["latents"])))
        print(f"Latents shape: {trajectory['latents_shape']}, dtype: {trajectory['latents_dtype']}")
        print(f"Decoded latents array shape: {latents.shape}")

    if trajectory.get("timesteps"):
        timesteps = [np.load(io.BytesIO(base64.b64decode(t))) for t in trajectory["timesteps"]]
        print(f"Timesteps count: {len(timesteps)}")

    if trajectory.get("log_probs"):
        log_probs = np.load(io.BytesIO(base64.b64decode(trajectory["log_probs"])))
        print(f"Log-probs shape: {trajectory['log_probs_shape']}")
        print(f"Decoded log-probs array shape: {log_probs.shape}")
```

### Rollout with SDE/CPS Log-Prob Computation

For RL training, you can enable rollout mode with flow-matching SDE or CPS
(Conditional Probability Score) log-probability computation. This is supported
on both the OpenAI-compatible endpoints and the native diffusion endpoint.

```python
import requests

ROUTER = "http://localhost:30081"

# Rollout with SDE log-prob (default)
resp = requests.post(f"{ROUTER}/v1/images/generations", json={
    "model": "stabilityai/stable-diffusion-3-medium-diffusers",
    "prompt": "a cute cat",
    "width": 512,
    "height": 512,
    "num_inference_steps": 28,
    "rollout": True,
    "rollout_sde_type": "sde",       # "sde" or "cps"
    "rollout_noise_level": 0.7,
    "response_format": "b64_json",
})
print(resp.json())

# Rollout with CPS log-prob
resp = requests.post(f"{ROUTER}/v1/images/generations", json={
    "model": "stabilityai/stable-diffusion-3-medium-diffusers",
    "prompt": "a cute cat",
    "width": 512,
    "height": 512,
    "num_inference_steps": 28,
    "rollout": True,
    "rollout_sde_type": "cps",
    "rollout_noise_level": 0.5,
    "response_format": "b64_json",
})
print(resp.json())

# Rollout via native diffusion endpoint (with trajectory + log-probs)
resp = requests.post(f"{ROUTER}/v1/diffusion/generate", json={
    "prompt": "a cute cat",
    "width": 512,
    "height": 512,
    "num_inference_steps": 28,
    "guidance_scale": 7.0,
    "seed": 42,
    "get_latents": True,
    "get_log_probs": True,
})
data = resp.json()
print(f"Trajectory available: {data.get('trajectory') is not None}")
```

### Sleep / Wake (Memory Occupation Control)

In RL training pipelines, the diffusion server is typically slept during the
training phase to free GPU memory, then woken up for the next rollout with
updated weights. The router broadcasts sleep/wake commands to all workers.

```python
import requests

ROUTER = "http://localhost:30081"

# --- Sleep: release GPU memory on all workers ---
resp = requests.post(f"{ROUTER}/release_memory_occupation", json={})
print(resp.json())
# Each worker result shows {"worker_url": "...", "status_code": 200, "body": {"success": true, "sleeping": true}}

# While sleeping, generation requests are rejected:
resp = requests.post(f"{ROUTER}/v1/images/generations", json={
    "model": "stabilityai/stable-diffusion-3-medium-diffusers",
    "prompt": "a cute cat",
    "response_format": "b64_json",
})
print(resp.status_code)  # 503 — workers are sleeping

# Sleep is idempotent: calling again is safe and returns 200.
resp = requests.post(f"{ROUTER}/release_memory_occupation", json={})
print(resp.json())

# --- Wake: resume GPU memory on all workers ---
resp = requests.post(f"{ROUTER}/resume_memory_occupation", json={})
print(resp.json())
# Each worker result shows {"worker_url": "...", "status_code": 200, "body": {"success": true, "sleeping": false}}

# Wake is also idempotent.
resp = requests.post(f"{ROUTER}/resume_memory_occupation", json={})
print(resp.json())

# After waking, you can optionally refit weights before the next rollout:
resp = requests.post(f"{ROUTER}/update_weights_from_disk", json={
    "model_path": "/path/to/new/checkpoint",
})
print(resp.json())

# Now generation works again:
resp = requests.post(f"{ROUTER}/v1/images/generations", json={
    "model": "stabilityai/stable-diffusion-3-medium-diffusers",
    "prompt": "a cute cat",
    "response_format": "b64_json",
})
print(resp.status_code)  # 200
```

A typical RL loop looks like:

```
1. Wake workers    → POST /resume_memory_occupation
2. Refit weights   → POST /update_weights_from_disk
3. Run rollout     → POST /v1/images/generations or /v1/diffusion/generate
4. Sleep workers   → POST /release_memory_occupation
5. Train on rollout data (GPU memory now free for training)
6. Repeat from step 1
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
        "model": "stabilityai/stable-diffusion-3-medium-diffusers",
        "prompt": "a cute cat",
        "num_images": 1,
        "response_format": "b64_json"
    }'

# Decode and save the image locally
curl -s -X POST http://localhost:30081/v1/images/generations \
    -H "Content-Type: application/json" \
    -d '{
        "model": "stabilityai/stable-diffusion-3-medium-diffusers",
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

# Native diffusion generate with trajectory
curl -X POST http://localhost:30081/v1/diffusion/generate \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "a cute cat",
        "width": 512,
        "height": 512,
        "get_latents": true,
        "get_log_probs": true
    }'

# Video generation request
curl -X POST http://localhost:30081/v1/videos \
    -H "Content-Type: application/json" \
    -d '{"model": "stabilityai/stable-diffusion-3-medium-diffusers", "prompt": "a flowing river"}'

# Poll a specific video job by video_id
curl http://localhost:30081/v1/videos/{video_id}

# Update weights from disk
curl -X POST http://localhost:30081/update_weights_from_disk \
    -H "Content-Type: application/json" \
    -d '{"model_path": "/path/to/new/checkpoint"}'

# Sleep workers (release GPU memory)
curl -X POST http://localhost:30081/release_memory_occupation \
    -H "Content-Type: application/json" \
    -d '{}'

# Wake workers (resume GPU memory)
curl -X POST http://localhost:30081/resume_memory_occupation \
    -H "Content-Type: application/json" \
    -d '{}'
```

## Router API

### Inference Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/images/generations` | OpenAI-compatible text-to-image generation |
| `POST` | `/v1/diffusion/generate` | Native SGLang-D generation with trajectory & log-prob support |
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
| `POST` | `/release_memory_occupation` | Sleep all healthy workers (release GPU memory) |
| `POST` | `/resume_memory_occupation` | Wake all sleeping workers (resume GPU memory) |

Both sleep and wake are idempotent. While sleeping, generation requests are rejected (503 from router). A typical RL loop: wake → refit weights → rollout → sleep → train → repeat.

## Acknowledgment

This project is derived from [radixark/miles#544](https://github.com/radixark/miles/pull/544). Thanks to the original authors.

SGLang Diffusion RL team is responsible for the development and maintenance of this project. Our team mates in alphabetical order:

Banghua Zhu, Chengliang Qian, Chenyang Zhao, Fenglin Yu, Hao Jin, Huapeng Zhou, Jiajun Li, Kangrui Du, Kun Lin, Mao Cheng, Mengyang Liu, Qiujiang Chen, Shenggui Li, Shirui Chen, Shuwen Wang, Xi Chen, Xiaole Guo, Ying Sheng, Yueming Yuan, Yuhao Yang, Yusheng Su, Zhiheng Ye
