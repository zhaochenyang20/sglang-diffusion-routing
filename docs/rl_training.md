# RL Training Integration Guide

This guide shows how to integrate the SGLang Diffusion Router into an RL
training pipeline. It covers the full loop: wake workers, refit weights,
rollout with trajectory collection, sleep workers, and train.

## Prerequisites

- Router running with workers registered (see [README](../README.md#quick-start))
- Workers launched with a model that supports trajectory output (e.g. `stabilityai/stable-diffusion-3-medium-diffusers`)

## Typical RL Loop

```
1. Wake workers    → POST /resume_memory_occupation
2. Refit weights   → POST /update_weights_from_disk
3. Rollout         → POST /v1/diffusion/generate (with get_latents / get_log_probs)
4. Sleep workers   → POST /release_memory_occupation
5. Train on collected trajectories (GPU memory now free)
6. Repeat from step 1
```

## Full Example

```python
import requests
import base64
import io
import numpy as np

ROUTER = "http://localhost:30081"
MODEL = "stabilityai/stable-diffusion-3-medium-diffusers"

prompts = ["a cute cat", "a sunset over mountains", "a robot painting"]
num_epochs = 10

for epoch in range(num_epochs):
    # 1. Wake workers (idempotent — safe to call even if already awake)
    requests.post(f"{ROUTER}/resume_memory_occupation", json={})

    # 2. Load latest checkpoint (skip on first epoch)
    if epoch > 0:
        requests.post(f"{ROUTER}/update_weights_from_disk", json={
            "model_path": MODEL,
        })

    # 3. Rollout — collect trajectories with latents and log-probs
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
        data = resp.json()

        # Decode trajectory arrays
        traj = data.get("trajectory", {})
        entry = {"prompt": prompt}
        if traj.get("latents"):
            entry["latents"] = np.load(io.BytesIO(base64.b64decode(traj["latents"])))
        if traj.get("log_probs"):
            entry["log_probs"] = np.load(io.BytesIO(base64.b64decode(traj["log_probs"])))
        if traj.get("timesteps"):
            entry["timesteps"] = [
                np.load(io.BytesIO(base64.b64decode(t))) for t in traj["timesteps"]
            ]
        trajectories.append(entry)

    # 4. Sleep workers — free GPU memory for training
    requests.post(f"{ROUTER}/release_memory_occupation", json={})

    # 5. Train on collected trajectories
    # train(trajectories)
    print(f"Epoch {epoch}: collected {len(trajectories)} trajectories")
```

## Related Docs

- [Sleep / Wake (GPU Memory Management)](sleep_wake.md)
- [README — Router API Reference](../README.md#router-api)
