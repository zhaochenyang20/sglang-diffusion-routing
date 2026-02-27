# Development

## Development Install

```bash
pip install -e .
```

Run CPU only tests:

```bash
pip install pytest
# CPU-only tests (unit + integration)
pytest tests/unit tests/integration -v

# Real E2E tests (GPU required, longer runtime)
pytest tests/e2e/test_e2e_sglang.py -v -s
```

## Benchmark Scripts

Benchmark scripts are available under `tests/benchmarks/diffusion_router/` and are intended for manual runs.
They are not part of default unit test collection (`pytest tests/unit tests/integration -v`).

Single benchmark:

```bash
python tests/benchmarks/diffusion_router/bench_router.py \
    --model Qwen/Qwen-Image \
    --num-workers 8 \
    --num-prompts 20 \
    --max-concurrency 8
```

Algorithm comparison:

```bash
python tests/benchmarks/diffusion_router/bench_routing_algorithms.py \
    --model Qwen/Qwen-Image \
    --num-workers 2 \
    --num-prompts 20 \
    --max-concurrency 4
```
