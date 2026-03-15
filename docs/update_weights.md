# Update Weights

The router provides `POST /update_weights_from_disk` to broadcast a weight-reload command to all healthy workers simultaneously. This is the "refit" step in an RL training loop — after the trainer writes new checkpoints to disk, the router tells every worker to pick them up.

## How It Works

```
Client                              Router                             Workers
  │                                   │                                  │
  │  POST /update_weights_from_disk   │                                  │
  │  { "model_path": "..." }          │                                  │
  │ ──────────────────────────────▶   │                                  │
  │                                   │  POST /update_weights_from_disk  │
  │                                   │  (broadcast to all)              │
  │                                   │ ──────────────────────────────▶  │ Worker 0
  │                                   │ ──────────────────────────────▶  │ Worker 1
  │                                   │ ──────────────────────────────▶  │ Worker N
  │                                   │                                  │
  │                                   │  ◀───── per-worker results       │
  │  ◀── { "results": [...] }         │                                  │
```

- The router forwards the request body as-is to each worker's `/update_weights_from_disk` endpoint
- Workers that are dead (quarantined) or sleeping are excluded from the broadcast
- Each worker's response (status code + body) is collected and returned in the aggregated `results` array

## Usage

### Basic

```bash
curl -X POST http://localhost:30081/update_weights_from_disk \
  -H "Content-Type: application/json" \
  -d '{"model_path": "/checkpoints/step-1000"}'
```

### Python

```python
import requests

ROUTER = "http://localhost:30081"

resp = requests.post(f"{ROUTER}/update_weights_from_disk", json={
    "model_path": "/checkpoints/step-1000",
})
print(resp.json())
```

### Response Format

Success (200):

```json
{
  "results": [
    {
      "worker_url": "http://127.0.0.1:10090",
      "status_code": 200,
      "body": { "ok": true, "model_path": "/checkpoints/step-1000" }
    },
    {
      "worker_url": "http://127.0.0.1:10092",
      "status_code": 200,
      "body": { "ok": true, "model_path": "/checkpoints/step-1000" }
    }
  ]
}
```

No healthy workers (503):

```json
{
  "error": "No healthy workers available in the pool"
}
```

Partial failure — some workers succeed, some fail:

```json
{
  "results": [
    {
      "worker_url": "http://127.0.0.1:10090",
      "status_code": 200,
      "body": { "ok": true, "model_path": "/checkpoints/step-1000" }
    },
    {
      "worker_url": "http://127.0.0.1:10092",
      "status_code": 500,
      "body": { "error": "failed to load weights" }
    }
  ]
}
```

The router always returns 200 as long as the broadcast was attempted. Check individual `status_code` in `results` to detect per-worker failures.

## RL Loop Integration

A typical RL training loop uses update_weights together with sleep/wake:

```
1. Wake workers     → POST /resume_memory_occupation
2. Refit weights    → POST /update_weights_from_disk
3. Rollout          → POST /v1/images/generations (or /v1/diffusion/generate)
4. Sleep workers    → POST /release_memory_occupation
5. Train on rollout data (GPU memory now free)
6. Repeat from step 1
```

```python
import requests

ROUTER = "http://localhost:30081"

# After training writes new checkpoint...
# 1. Wake workers
requests.post(f"{ROUTER}/resume_memory_occupation", json={})

# 2. Load new weights
resp = requests.post(f"{ROUTER}/update_weights_from_disk", json={
    "model_path": "/checkpoints/step-1000",
})
# Check all workers loaded successfully
for r in resp.json()["results"]:
    assert r["status_code"] == 200, f"Worker {r['worker_url']} failed: {r['body']}"

# 3. Rollout
resp = requests.post(f"{ROUTER}/v1/images/generations", json={
    "model": "my-model",
    "prompt": "a cute cat",
})

# 4. Sleep workers to free GPU memory for training
requests.post(f"{ROUTER}/release_memory_occupation", json={})
```

## Behavior Notes

- Dead workers (quarantined by health checks) are skipped
- Sleeping workers are skipped — wake them first with `POST /resume_memory_occupation`
- The request body is forwarded verbatim to each worker; the router does not interpret or validate `model_path`
- If a worker fails to load weights (returns non-200), it remains registered — the router does not auto-quarantine on weight update failure
