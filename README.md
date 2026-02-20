# sglang-diffusion-routing

`sglang-diffusion-routing` is a **demonstrative and testable setup** for running
**multiple SGLang Diffusion workers behind a lightweight DP (data-parallel) router**.

The router provides:
- **Load-balanced request routing** across diffusion workers
- **Health checking & quarantine** (automatic dead worker detection)
- Support for diffusion **generation APIs** (e.g. `/generate`, `/v1/images/generations`)
- Compatibility with recent SGLang diffusion features such as
  - advanced generation methods (e.g. SDE / CPS)
  - `update_weights_from_disk` (PR [#18306](https://github.com/sgl-project/sglang/pull/18306))

This repository is intended for:
- local / cluster experimentation
- router logic validation
- debugging and development (with both GPU and fake workers)

---

## Acknowledgment

This repository **copies and adapts code from**:

> https://github.com/radixark/miles/pull/544

All credit for the original router logic and design goes to the authors of that PR.
This repo reorganizes the code into a standalone, runnable, and testable project,
with additional scripts and documentation.

---

## Repository Structure

```

sglang-diffusion-routing/
├── src/sglang_d_router/        # Router implementation (FastAPI)
├── examples/                   # Demo apps (router demo, fake worker)
├── scripts/                    # Helper scripts (workers, router, requests)
├── tests/                      # Pytest-based router tests
├── pyproject.toml
└── README.md

````

---

## Installation

> **Note:** This project is currently intended for **local development only**.
> No PyPI package is published.

### 1. Install the router (local editable install)

From the repository root:

```bash
pip install .
````

or (editable mode):

```bash
pip install -e .
```

This installs the `sglang_d_router` Python package and exposes router entrypoints.

---

### 2. Install SGLang with diffusion support

```bash
pip install "sglang[diffusion]"
```

Verify:

```bash
sglang --help
```

---

## Core Concepts

### Router

* A FastAPI service that accepts diffusion requests
* Periodically health-checks registered workers
* Routes requests to **healthy workers only**
* Supports multiple routing strategies (e.g. least-request)

### Worker

Two types are supported:

1. **GPU Worker**

   * A real SGLang diffusion server (`sglang serve`)
   * Uses GPU(s)
   * Runs actual diffusion models

2. **Fake Worker**

   * Lightweight FastAPI app
   * Implements `/health` and generation endpoints
   * Used for router testing without GPU / model load

Both workers expose the same HTTP contract.

---

## Quick Start (GPU Workers)

### 1. Launch multiple GPU workers

The script below launches **N SGLang diffusion workers** by reusing a single
`run_model_serve_1.sh` script and automatically assigning ports / GPUs.

```bash
MODEL=Wan-AI/Wan2.2-T2V-A14B-Diffusers \
bash scripts/launch_workers.sh \
  --n 2 \
  --base-port 10090 \
  --gpus 0,1
```

This starts:

* worker #0 → `http://127.0.0.1:10090` (GPU 0)
* worker #1 → `http://127.0.0.1:10091` (GPU 1)

Logs and PIDs are written to:

```
./tmp/sglang_workers/
```

---

### 2. Launch the router

```bash
bash scripts/run_router.sh
```

By default, this runs:

```
http://127.0.0.1:30080
```

and registers the workers at ports `10090`, `10091`.

---

### 3. Check router and worker health

```bash
curl http://127.0.0.1:30080/health
curl http://127.0.0.1:30080/health_workers
```

---

### 4. Send a generation request

```bash
BASE=http://127.0.0.1:30080 \
REQ_PATH=/generate \
bash scripts/send_request.sh
```

The request uses:

* `response_format = b64_json` (no cloud storage required)
* minimal generation payload

---

## Fake Worker Mode (No GPU Required)

Fake workers are useful for:

* router logic testing
* health-check / quarantine validation
* development on CPU-only machines

### 1. Launch fake workers

```bash
bash scripts/launch_fake_workers.sh --n 2 --base-port 10090
```

---

### 2. Launch the router (same as GPU mode)

```bash
bash scripts/run_router.sh
```

---

### 3. Send requests through the router

```bash
BASE=http://127.0.0.1:30080 \
REQ_PATH=/generate \
bash scripts/send_request.sh
```

---

## Scripts Reference

All helper scripts live under `scripts/`.

### `run_model_serve.sh`

Launch **one GPU diffusion worker**.

Environment variables:

* `MODEL` – diffusion model path or HF repo
* `PORT` – service port
* `CUDA_VISIBLE_DEVICES` – GPU selection
* `DTYPE` – bf16 / fp16

Example:

```bash
CUDA_VISIBLE_DEVICES=0 PORT=10090 MODEL=runwayml/stable-diffusion-v1-5 \
bash scripts/run_model_serve_1.sh
```

---

### `launch_workers.sh`

Launch **multiple GPU workers** by wrapping `run_model_serve.sh`.

Key options:

* `--n` – number of workers
* `--base-port` – starting port
* `--gpus` – comma-separated GPU list
* `--gpu-base` – auto-assign GPUs from a base index

---

### `run_router.sh`

Launch the diffusion router with:

* worker registration
* health check loop
* load-balancing enabled

---

### `launch_fake_workers.sh`

Launch **CPU-only fake workers** for testing.

---

### `stop_workers.sh`

Stop all GPU workers launched by `launch_workers.sh`.

---

### `stop_fake_workers.sh`

Stop all fake workers.

---

### `send_request.sh`

Send a test request to:

* the router, or
* a specific worker

Supports:

* `/health` (GET)
* `/generate` or `/v1/images/generations` (POST)
* configurable prompt / size / response format

Example:

```bash
BASE=http://127.0.0.1:30080 REQ_PATH=/health METHOD=GET bash scripts/send_request.sh
```

---

## Testing

Unit and integration tests are under `tests/` and can be run with:

```bash
pytest
```

Tests cover:

* routing algorithms
* request counting
* health check & quarantine logic
* router endpoints

---

## Notes & Limitations

* This repository is **not a production system**
* Cloud storage is **not configured** (use `response_format=b64_json`)
* Intended for research, debugging, and system prototyping

---

## License

See the original Miles repository and SGLang project for licensing details.
