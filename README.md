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

```bash
git clone --recursive https://github.com/zhaochenyang20/sglang-diffusion-routing.git
cd sglang-diffusion-routing

# If cloned without --recursive, initialize the sglang submodule:
# git submodule update --init --recursive

# Install the router package
uv pip install .

# Install SGLang diffusion from the pinned submodule (includes RL patches).
# Do NOT install sglang from PyPI — the submodule tracks a fork with
# /v1/diffusion/generate, flow-matching log-prob, and other RL features.
cd sglang/python
uv pip install ".[diffusion]" --prerelease=allow
cd ../..
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

<details>
<summary>Manual Launch Workers</summary>

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

</details>



## Demonstrative Examples

A typical RL loop looks like:

```
1. Start workers   → sglang-d-router --port 30081 --launcher-config examples/local_launcher.yaml
2. Rollout         → POST /v1/images/generations or /v1/diffusion/generate
3. Sleep workers   → POST /release_memory_occupation
4. Train on rollout data (GPU memory now free for training)
5. Wake workers    → POST /resume_memory_occupation
6. Refit weights   → POST /update_weights_from_disk
7. Repeat from step 2
```

We provide all the steps in the code examples below.

### With Python Requests

```python
import requests
import base64

ROUTER = "http://localhost:30081"

# Check router health
resp = requests.get(f"{ROUTER}/health")
print(resp.json())

# Register a worker (only needed for manual launch; co-launch handles this)
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

# FIXME:
# Video generation request
# Note that Stable-Diffusion-3 does not support video generation,
# so this request will fail. Use a video-capable model instead.

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
    "model_path": "stabilityai/stable-diffusion-3-medium-diffusers",
})
print(resp.json())

# sleep and wake up
resp = requests.post(f"{ROUTER}/release_memory_occupation", json={})
print(resp.json())


resp = requests.post(f"{ROUTER}/resume_memory_occupation", json={})
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
print(f"Inference time: {data.get('inference_time_s')}s")

# Decode the output image
img_bytes = base64.b64decode(data["output_b64"])
with open("output.png", "wb") as f:
    f.write(img_bytes)
print(f"Saved image ({data.get('output_format', 'unknown')} format)")

# Decode trajectory data
trajectory = data.get("trajectory")
latents = np.load(io.BytesIO(base64.b64decode(trajectory["latents"])))
print(f"Latents shape: {trajectory['latents_shape']}, dtype: {trajectory['latents_dtype']}")
print(f"Decoded latents array shape: {latents.shape}")

timesteps = [np.load(io.BytesIO(base64.b64decode(t))) for t in trajectory["timesteps"]]
print(f"Timesteps count: {len(timesteps)}")

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

### Sleep / Wake Workers (GPU Memory Management)

In RL pipelines, you typically alternate between rollout (inference) and training
phases. During training, you want to free GPU memory held by the diffusion workers.
The router provides `release_memory_occupation` (sleep) and
`resume_memory_occupation` (wake) endpoints for this.

Both operations are idempotent and broadcast to all workers. While sleeping,
generation requests are rejected with 503. Waking also clears any dead-worker
marks that accumulated during sleep, so workers recover cleanly.

```python
import requests

ROUTER = "http://localhost:30081"

# --- After rollout is done, free GPU memory for training ---

# Sleep all workers (release GPU memory)
resp = requests.post(f"{ROUTER}/release_memory_occupation", json={})
print(resp.json())
# Example response:
# {"results": [{"worker_url": "http://...", "status_code": 200, "body": {...}}, ...]}

# Verify workers are sleeping
resp = requests.get(f"{ROUTER}/health")
health = resp.json()
print(f"Sleeping: {health['sleeping_workers']}, Healthy: {health['healthy_workers']}")

# Generation requests are rejected while workers sleep
resp = requests.post(f"{ROUTER}/v1/images/generations", json={
    "model": "stabilityai/stable-diffusion-3-medium-diffusers",
    "prompt": "a cute cat",
})
print(resp.status_code)  # 503

# Calling sleep again is safe (idempotent)
resp = requests.post(f"{ROUTER}/release_memory_occupation", json={})
print(resp.json())
# {"message": "All workers are already sleeping", "sleeping_workers": N}

# --- Training phase happens here ---

# --- Wake workers back up before next rollout ---

# Wake all sleeping workers (resume GPU memory)
resp = requests.post(f"{ROUTER}/resume_memory_occupation", json={})
print(resp.json())
# Workers are now active again; dead-worker marks cleared automatically.

# Verify workers are back
resp = requests.get(f"{ROUTER}/health")
health = resp.json()
print(f"Sleeping: {health['sleeping_workers']}, Healthy: {health['healthy_workers']}")

# Calling wake again is safe (idempotent)
resp = requests.post(f"{ROUTER}/resume_memory_occupation", json={})
print(resp.json())
# {"message": "All workers are already active", "active_workers": N}

# Optionally reload updated weights after training
resp = requests.post(f"{ROUTER}/update_weights_from_disk", json={
    "model_path": "stabilityai/stable-diffusion-3-medium-diffusers",
})
print(resp.json())

# Now ready for next rollout
resp = requests.post(f"{ROUTER}/v1/images/generations", json={
    "model": "stabilityai/stable-diffusion-3-medium-diffusers",
    "prompt": "a cute cat",
    "response_format": "b64_json",
})
print(resp.status_code)  # 200
```

### Full RL Loop Example

Putting it all together — a minimal RL training loop using the router:

```python
import requests

ROUTER = "http://localhost:30081"
MODEL = "stabilityai/stable-diffusion-3-medium-diffusers"

prompts = ["a cute cat", "a sunset over mountains", "a robot painting"]

for epoch in range(num_epochs):
    # 1. Wake workers
    requests.post(f"{ROUTER}/resume_memory_occupation", json={})

    # 2. (Optional) Load latest weights from training
    if epoch > 0:
        requests.post(f"{ROUTER}/update_weights_from_disk", json={
            "model_path": MODEL,
        })

    # 3. Rollout — generate images with trajectory data for RL
    trajectories = []
    for prompt in prompts:
        resp = requests.post(f"{ROUTER}/v1/diffusion/generate", json={
            "prompt": prompt,
            "width": 512,
            "height": 512,
            "num_inference_steps": 28,
            "guidance_scale": 7.0,
            "get_latents": True,
            "get_log_probs": True,
        })
        trajectories.append(resp.json())

    # 4. Sleep workers — free GPU memory for training
    requests.post(f"{ROUTER}/release_memory_occupation", json={})

    # 5. Train on collected trajectories
    # train(trajectories)
    pass
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

This project is derived from [radixark/miles#544](https://github.com/radixark/miles/pull/544) and [alphabetc1/sglang](https://github.com/alphabetc1/sglang/tree/sglang/diffusion-rl-base). Thanks to the original authors.

SGLang Diffusion RL team is responsible for the development and maintenance of this project. Our team mates in alphabetical order:

Banghua Zhu, Chengliang Qian, Chenyang Zhao, Fenglin Yu, Hao Jin, Huapeng Zhou, Jiajun Li, Kangrui Du, Kun Lin, Mao Cheng, Mengyang Liu, Qiujiang Chen, Shenggui Li, Shirui Chen, Shuwen Wang, Xi Chen, Xiaole Guo, Ying Sheng, Yueming Yuan, Yuhao Yang, Yusheng Su, Zhiheng Ye
