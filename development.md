# Development

## Development Install

```bash
pip install -e .
```

Run tests:

```bash
pip install pytest
pytest tests/unit -v
```

## Benchmark Scripts

Benchmark scripts are available under `tests/benchmarks/diffusion_router/` and are intended for manual runs.
They are not part of default unit test collection (`pytest tests/unit -v`).

Single benchmark:

```bash
python tests/benchmarks/diffusion_router/bench_router.py \
    --model Qwen/Qwen-Image \
    --num-workers 2 \
    --num-prompts 20 \
    --max-concurrency 4
```

Algorithm comparison:

```bash
python tests/benchmarks/diffusion_router/bench_routing_algorithms.py \
    --model Qwen/Qwen-Image \
    --num-workers 2 \
    --num-prompts 20 \
    --max-concurrency 4
```
