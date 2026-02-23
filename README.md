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

## Acknowledgment

This project is derived from [radixark/miles#544](https://github.com/radixark/miles/pull/544). Thanks to the original authors.

SGLang Diffusion RL team is responsible for the development and maintenance of this project. Our team mates in alphabetical order:

Banghua Zhu, Chengliang Qian, Chenyang Zhao, Fenglin Yu, Hao Jin, Huapeng Zhou, Jiajun Li, Kangrui Du, Kun Lin, Mao Cheng, Mengyang Liu, Qiujiang Chen, Shenggui Li, Shirui Chen, Shuwen Wang, Xi Chen, Xiaole Guo, Ying Sheng, Yueming Yuan, Yuhao Yang, Yusheng Su, Zhiheng Ye
