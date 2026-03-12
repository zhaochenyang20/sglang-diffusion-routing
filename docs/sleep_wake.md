# Sleep / Wake Workers (GPU Memory Management)

In RL pipelines, you typically alternate between rollout (inference) and training
phases. During training, you want to free GPU memory held by the diffusion workers.
The router provides `release_memory_occupation` (sleep) and
`resume_memory_occupation` (wake) endpoints for this.

Both operations are idempotent and broadcast to all workers. While sleeping,
generation requests are rejected with 503. Waking also clears any dead-worker
marks that accumulated during sleep, so workers recover cleanly.

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/release_memory_occupation` | Sleep all healthy workers (release GPU memory) |
| `POST` | `/resume_memory_occupation` | Wake all sleeping workers (resume GPU memory) |

## Example

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

### Curl

```bash
# Sleep workers (release GPU memory)
curl -X POST http://localhost:30081/release_memory_occupation \
    -H "Content-Type: application/json" \
    -d '{}'

# Wake workers (resume GPU memory)
curl -X POST http://localhost:30081/resume_memory_occupation \
    -H "Content-Type: application/json" \
    -d '{}'
```
