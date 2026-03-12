# Trajectory & Log-Prob Collection

This guide covers the RL-specific generation endpoints that expose intermediate
diffusion outputs: trajectory latents, timesteps, and per-step log-probabilities.

## Native Diffusion Generate Endpoint

The `POST /v1/diffusion/generate` endpoint returns trajectory data that the
OpenAI-compatible endpoints intentionally omit. Use `get_latents` and
`get_log_probs` to request the extra metadata.

```python
import requests
import base64
import io
import numpy as np

ROUTER = "http://localhost:30081"

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

# Decode trajectory arrays (base64-encoded .npy blobs)
trajectory = data.get("trajectory")

latents = np.load(io.BytesIO(base64.b64decode(trajectory["latents"])))
print(f"Latents: shape={trajectory['latents_shape']}, dtype={trajectory['latents_dtype']}")

timesteps = [np.load(io.BytesIO(base64.b64decode(t))) for t in trajectory["timesteps"]]
print(f"Timesteps: {len(timesteps)} steps")

log_probs = np.load(io.BytesIO(base64.b64decode(trajectory["log_probs"])))
print(f"Log-probs: shape={trajectory['log_probs_shape']}")
```

## Rollout with SDE/CPS Log-Prob

For RL training you can enable rollout mode with flow-matching SDE or CPS
(Conditional Probability Score) log-probability computation. This is supported
on both the OpenAI-compatible endpoints and the native diffusion endpoint.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `rollout` | bool | `false` | Enable rollout mode |
| `rollout_sde_type` | str | `"sde"` | `"sde"` or `"cps"` |
| `rollout_noise_level` | float | `0.7` | Noise level for the SDE/CPS step |

```python
import requests

ROUTER = "http://localhost:30081"

# SDE rollout (default)
resp = requests.post(f"{ROUTER}/v1/images/generations", json={
    "model": "stabilityai/stable-diffusion-3-medium-diffusers",
    "prompt": "a cute cat",
    "width": 512,
    "height": 512,
    "num_inference_steps": 28,
    "rollout": True,
    "rollout_sde_type": "sde",
    "rollout_noise_level": 0.7,
    "response_format": "b64_json",
})

# CPS rollout
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

# Rollout via native endpoint (with full trajectory)
resp = requests.post(f"{ROUTER}/v1/diffusion/generate", json={
    "prompt": "a cute cat",
    "width": 512,
    "height": 512,
    "num_inference_steps": 28,
    "guidance_scale": 7.0,
    "get_latents": True,
    "get_log_probs": True,
})
```
