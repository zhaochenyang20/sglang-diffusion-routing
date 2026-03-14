# Launcher

The launcher lets the router spawn and manage SGLang diffusion worker processes automatically. Provide a YAML config instead of starting workers by hand and passing `--worker-urls`.

## Architecture

```
sglang-d-router --launcher-config config.yaml
  │
  ├─ 1. Load config ─── YAML → OmegaConf validation → LocalLauncherConfig
  │
  ├─ 2. Launch workers ─── per worker: assign GPUs → reserve ports → Popen("sglang serve ...")
  │     ┌──────────┐  ┌──────────┐       ┌──────────┐
  │     │ Worker 0 │  │ Worker 1 │  ...  │ Worker N │
  │     │ GPU 0    │  │ GPU 1    │       │ GPU N    │
  │     │ :10090   │  │ :10092   │       │ :...     │
  │     └──────────┘  └──────────┘       └──────────┘
  │
  ├─ 3. Health-check & register ─── poll /health in parallel, 200 OK → register_worker(url)
  │
  ├─ 4. Serve ─── start uvicorn, router handles traffic
  │
  └─ 5. Shutdown ─── SIGINT → wait 15s → SIGKILL process group
```

## Usage

### Minimal: single worker

```yaml
# config.yaml
launcher:
  model: Qwen/Qwen-Image
```

```bash
sglang-d-router --port 30081 --launcher-config config.yaml
```

### Multi-worker with auto GPU assignment

```yaml
launcher:
  model: Qwen/Qwen-Image
  num_workers: 8
  num_gpus_per_worker: 1
  worker_base_port: 10090
  wait_timeout: 600
```

8 workers on GPUs 0–7 (auto-detected), ports 10090, 10092, ..., 10104.

### Multi-GPU workers with explicit assignment

```yaml
launcher:
  model: Qwen/Qwen-Image
  num_workers: 2
  num_gpus_per_worker: 4
  worker_gpu_ids: ["0,1,2,3", "4,5,6,7"]
```

### Extra sglang arguments

```yaml
launcher:
  model: Qwen/Qwen-Image
  num_workers: 4
  worker_extra_args: "--dit-cpu-offload false --text-encoder-cpu-offload false"
```

`worker_extra_args` is appended to the `sglang serve` command (parsed via `shlex.split`).

### Mixing launched and external workers

```bash
sglang-d-router \
  --port 30081 \
  --launcher-config config.yaml \
  --worker-urls http://remote-host:10090
```

### Combining with router flags

```bash
sglang-d-router \
  --port 30081 \
  --launcher-config config.yaml \
  --routing-algorithm round-robin \
  --health-check-interval 5 \
  --timeout 180 \
  --verbose
```

## Configuration Reference

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | *(required)* | Model path for `sglang serve --model-path` |
| `backend` | string | `"local"` | Launcher backend (only `"local"` for now) |
| `num_workers` | int | `1` | Number of worker processes |
| `num_gpus_per_worker` | int | `1` | GPUs per worker |
| `worker_host` | string | `"127.0.0.1"` | Worker bind address |
| `worker_base_port` | int | `10090` | Starting HTTP port (worker *i* → `base + i*2`) |
| `worker_gpu_ids` | list | `null` | Explicit per-worker GPU IDs, e.g. `["0,1", "4,5"]` |
| `worker_extra_args` | string | `""` | Extra CLI args for `sglang serve` |
| `wait_timeout` | int | `600` | Per-worker health-check timeout (seconds) |
| `master_port_base` | int | `30005` | PyTorch distributed master port base |
| `scheduler_port_base` | int | `5555` | Scheduler port base |
| `internal_port_stride` | int | `1000` | Port stride between workers for internal ports |

Config is validated via OmegaConf structured configs — type mismatches, unknown keys, and missing required fields are caught at load time.

## Key Internals

### GPU assignment

Priority: `worker_gpu_ids` → `CUDA_VISIBLE_DEVICES` env → `torch.cuda.device_count()`.

- If `worker_gpu_ids` is set, used directly as each worker's `CUDA_VISIBLE_DEVICES` (length must equal `num_workers`)
- Otherwise the GPU pool is auto-detected and sliced into groups of `num_gpus_per_worker`
- If no GPUs found, falls back to CPU mode

### Port allocation

Each worker needs 4 ports: HTTP, ZMQ (HTTP+1), master, scheduler.

- HTTP port scans from `worker_base_port + i*2`; both port and port+1 must be free (ZMQ uses port+1)
- Master/scheduler ports scan from their respective base + `i * internal_port_stride`
- All claimed ports are tracked in a `used_ports` set to prevent collisions

### Extending with new backends

1. Implement `LauncherBackend` (`launch` / `wait_ready_and_register` / `shutdown`)
2. Add a `@dataclass` config class
3. Register in `config.py`:
   ```python
   SCHEMA_REGISTRY["k8s"] = K8sLauncherConfig
   BACKEND_REGISTRY["k8s"] = K8sLauncher
   ```
4. Set `backend: k8s` in YAML

## Module Map

| Module | Role |
|---|---|
| `launcher/backend.py` | Abstract `LauncherBackend` interface and data types |
| `launcher/config.py` | YAML loading, validation, backend factory |
| `launcher/local.py` | `LocalLauncher` implementation and config dataclass |
| `launcher/utils.py` | GPU assignment, port detection, health polling, process cleanup |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `'sglang' command not found` | sglang not installed | `cd sglang/python && pip install ".[diffusion]"` |
| `Not enough GPUs` | GPU pool too small | Reduce `num_workers` / `num_gpus_per_worker` or set `CUDA_VISIBLE_DEVICES` |
| `Timed out waiting for worker` | Worker crashed or slow model download | Check worker logs, increase `wait_timeout`, or run `sglang serve` manually to debug |
| Port conflicts | Port already in use | Change `worker_base_port` or kill the conflicting process |
| `worker_gpu_ids has N entries but num_workers is M` | Length mismatch | Ensure `len(worker_gpu_ids) == num_workers` |
