#!/usr/bin/env python3
"""
Launch diffusion workers and router, then run a serving benchmark.

Example:
  python tests/benchmarks/diffusion_router/bench_router.py \
    --model Qwen/Qwen-Image \
    --num-workers 2 \
    --num-prompts 20 \
    --max-concurrency 2
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import httpx


def _repo_root() -> Path:
    # tests/benchmarks/diffusion_router/bench_router.py -> repo root
    return Path(__file__).resolve().parents[3]


def _require_non_empty_model(model: str) -> str:
    normalized = model.strip()
    if not normalized:
        raise ValueError(
            "--model must be a non-empty model ID/path. "
            "Detected an empty value, which often means a shell variable such as "
            "$MODEL was unset."
        )
    return normalized


def _infer_client_host(host: str) -> str:
    if host in ("0.0.0.0", "::"):
        return "127.0.0.1"
    return host


def _wait_for_health(
    url: str,
    timeout: int,
    label: str,
    proc: subprocess.Popen | None = None,
) -> None:
    start = time.time()
    last_print = 0.0
    while True:
        elapsed = time.time() - start

        # Fail fast if a managed process exits unexpectedly.
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"{label} process exited with code {proc.returncode}. "
                "Run the command directly to inspect startup errors."
            )

        try:
            resp = httpx.get(f"{url}/health", timeout=1.0)
            if resp.status_code == 200:
                print(f"  [bench] {label} is healthy ({elapsed:.0f}s)", flush=True)
                return
        except httpx.HTTPError:
            pass

        if elapsed - last_print >= 30:
            print(
                f"  [bench] Still waiting for {label}... ({elapsed:.0f}s elapsed)",
                flush=True,
            )
            last_print = elapsed

        if elapsed > timeout:
            raise TimeoutError(f"Timed out waiting for {label} at {url}.")
        time.sleep(1)


def _normalize_connect_host(host: str) -> str:
    if host in ("0.0.0.0", "::"):
        return "127.0.0.1"
    if host == "localhost":
        return "127.0.0.1"
    return host


def _is_port_available(host: str, port: int) -> bool:
    connect_host = _normalize_connect_host(host)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((connect_host, port)) != 0


def _reserve_available_port(
    host: str, preferred_port: int, used_ports: set[int]
) -> int:
    if preferred_port < 1 or preferred_port > 65535:
        raise ValueError(f"Invalid port: {preferred_port}")

    for port in range(preferred_port, 65536):
        if port in used_ports:
            continue
        if _is_port_available(host, port):
            used_ports.add(port)
            return port

    for port in range(1024, preferred_port):
        if port in used_ports:
            continue
        if _is_port_available(host, port):
            used_ports.add(port)
            return port

    raise RuntimeError(
        f"Unable to reserve a free port for host {host}. Preferred start={preferred_port}."
    )


def _parse_gpu_id_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _detect_gpu_count() -> int:
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _resolve_gpu_pool(
    args: argparse.Namespace, env: dict[str, str]
) -> list[str] | None:
    if args.worker_gpu_ids:
        return [str(x) for x in args.worker_gpu_ids]

    visible = env.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        parsed = _parse_gpu_id_list(visible)
        if parsed:
            return parsed

    gpu_count = _detect_gpu_count()
    if gpu_count > 0:
        return [str(i) for i in range(gpu_count)]
    return None


def _terminate_all(processes: Iterable[subprocess.Popen]) -> None:
    procs = list(processes)

    def _signal_group(proc: subprocess.Popen, sig: int) -> None:
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
        except Exception:
            if proc.poll() is None:
                try:
                    os.kill(proc.pid, sig)
                except ProcessLookupError:
                    pass

    for proc in procs:
        _signal_group(proc, signal.SIGTERM)

    for proc in procs:
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _signal_group(proc, signal.SIGKILL)

    for proc in procs:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _build_pythonpath_with_src(repo_root: Path, env: dict[str, str]) -> dict[str, str]:
    out = dict(env)
    src_dir = str(repo_root / "src")
    old = out.get("PYTHONPATH")
    out["PYTHONPATH"] = src_dir if not old else f"{src_dir}:{old}"
    return out


def _build_bench_command(args: argparse.Namespace, base_url: str) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "sglang.multimodal_gen.benchmarks.bench_serving",
        "--base-url",
        base_url,
        "--model",
        args.model,
        "--dataset",
        args.dataset,
        "--num-prompts",
        str(args.num_prompts),
        "--request-rate",
        str(args.request_rate),
        "--log-level",
        args.log_level,
    ]

    if args.dataset_path:
        cmd += ["--dataset-path", args.dataset_path]
    if args.max_concurrency:
        cmd += ["--max-concurrency", str(args.max_concurrency)]
    if args.task:
        cmd += ["--task", args.task]
    if args.width:
        cmd += ["--width", str(args.width)]
    if args.height:
        cmd += ["--height", str(args.height)]
    if args.num_frames:
        cmd += ["--num-frames", str(args.num_frames)]
    if args.fps:
        cmd += ["--fps", str(args.fps)]
    if args.output_file:
        cmd += ["--output-file", args.output_file]
    if args.disable_tqdm:
        cmd.append("--disable-tqdm")
    if args.bench_extra_args:
        cmd += shlex.split(args.bench_extra_args)

    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark sglang-d-router with sglang bench_serving."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Diffusion model HF ID or local path."
    )
    parser.add_argument(
        "--router-host", type=str, default="127.0.0.1", help="Router bind host."
    )
    parser.add_argument("--router-port", type=int, default=30080, help="Router port.")
    parser.add_argument(
        "--routing-algorithm",
        type=str,
        default="least-request",
        choices=["least-request", "round-robin", "random"],
        help="Load-balancing algorithm for the router.",
    )
    parser.add_argument(
        "--router-verbose", action="store_true", help="Enable router verbose logging."
    )
    parser.add_argument("--router-max-connections", type=int, default=100)
    parser.add_argument("--router-timeout", type=float, default=120.0)
    parser.add_argument("--router-health-check-interval", type=int, default=10)
    parser.add_argument("--router-health-check-failure-threshold", type=int, default=3)
    parser.add_argument(
        "--router-extra-args",
        type=str,
        default="",
        help="Extra args for the router CLI command.",
    )

    parser.add_argument(
        "--worker-host", type=str, default="127.0.0.1", help="Worker bind host."
    )
    parser.add_argument(
        "--worker-urls", nargs="*", default=[], help="Existing worker URLs to use."
    )
    parser.add_argument(
        "--num-workers", type=int, default=1, help="Number of workers to launch."
    )
    parser.add_argument(
        "--worker-base-port",
        type=int,
        default=10090,
        help="Base port for launched workers.",
    )
    parser.add_argument(
        "--worker-port-stride",
        type=int,
        default=2,
        help="Port increment between launched workers. Keep >=2 to avoid port collisions.",
    )
    parser.add_argument(
        "--worker-master-port-base",
        type=int,
        default=30005,
        help="Base torch distributed master port for launched workers.",
    )
    parser.add_argument(
        "--worker-scheduler-port-base",
        type=int,
        default=5555,
        help="Base scheduler port for launched workers.",
    )
    parser.add_argument(
        "--worker-internal-port-stride",
        type=int,
        default=1000,
        help="Stride used between workers for master/scheduler base ports.",
    )
    parser.add_argument(
        "--num-gpus-per-worker", type=int, default=1, help="GPUs per worker."
    )
    parser.add_argument(
        "--worker-gpu-ids",
        nargs="*",
        default=None,
        help=(
            "Optional GPU IDs/UUIDs for launched workers. They are consumed in order, "
            "in groups of --num-gpus-per-worker."
        ),
    )
    parser.add_argument(
        "--worker-extra-args",
        type=str,
        default="",
        help="Extra args for `sglang serve`.",
    )
    parser.add_argument(
        "--skip-workers", action="store_true", help="Do not launch workers."
    )

    parser.add_argument(
        "--dataset", type=str, default="random", choices=["vbench", "random"]
    )
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument(
        "--bench-extra-args", type=str, default="", help="Extra args for bench_serving."
    )
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=1200,
        help="Seconds to wait for services health.",
    )

    args = parser.parse_args()
    args.model = _require_non_empty_model(args.model)

    repo_root = _repo_root()
    managed_processes: list[subprocess.Popen] = []
    launched_workers: list[tuple[str, subprocess.Popen]] = []
    worker_urls = list(args.worker_urls)
    used_ports: set[int] = set()

    try:
        import sglang  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "sglang is not installed.\n"
            'Install with: uv pip install "sglang[diffusion]" --prerelease=allow'
        ) from exc

    try:
        if args.num_workers < 0:
            raise ValueError("--num-workers must be >= 0")
        if args.num_gpus_per_worker < 1:
            raise ValueError("--num-gpus-per-worker must be >= 1")

        if not args.skip_workers:
            host_for_url = _infer_client_host(args.worker_host)
            worker_env = os.environ.copy()
            gpu_pool = _resolve_gpu_pool(args, worker_env)

            needed = args.num_workers * args.num_gpus_per_worker
            if gpu_pool and len(gpu_pool) < needed:
                raise RuntimeError(
                    f"Not enough GPUs for requested workers. Need {needed}, found {len(gpu_pool)} in pool {gpu_pool}."
                )

            for i in range(args.num_workers):
                preferred_worker_port = (
                    args.worker_base_port + i * args.worker_port_stride
                )
                worker_port = _reserve_available_port(
                    args.worker_host, preferred_worker_port, used_ports
                )

                preferred_master = (
                    args.worker_master_port_base + i * args.worker_internal_port_stride
                )
                preferred_scheduler = (
                    args.worker_scheduler_port_base
                    + i * args.worker_internal_port_stride
                )
                master_port = _reserve_available_port(
                    "127.0.0.1", preferred_master, used_ports
                )
                scheduler_port = _reserve_available_port(
                    "127.0.0.1", preferred_scheduler, used_ports
                )

                cmd = [
                    "sglang",
                    "serve",
                    "--model-path",
                    args.model,
                    "--num-gpus",
                    str(args.num_gpus_per_worker),
                    "--host",
                    args.worker_host,
                    "--port",
                    str(worker_port),
                    "--master-port",
                    str(master_port),
                    "--scheduler-port",
                    str(scheduler_port),
                ]
                if args.worker_extra_args:
                    cmd += shlex.split(args.worker_extra_args)

                env = dict(worker_env)
                if gpu_pool:
                    start = i * args.num_gpus_per_worker
                    stop = start + args.num_gpus_per_worker
                    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_pool[start:stop])

                worker_url = f"http://{host_for_url}:{worker_port}"
                print(f"[run] {' '.join(shlex.quote(x) for x in cmd)}", flush=True)
                proc = subprocess.Popen(
                    cmd,
                    cwd=repo_root,
                    env=env,
                    start_new_session=True,
                )
                managed_processes.append(proc)
                launched_workers.append((worker_url, proc))
                worker_urls.append(worker_url)

        if not worker_urls:
            raise RuntimeError(
                "No workers available. Use --worker-urls or launch workers by removing --skip-workers."
            )

        for url, proc in launched_workers:
            _wait_for_health(url, args.wait_timeout, f"worker {url}", proc=proc)
        for url in worker_urls:
            if url not in {u for u, _ in launched_workers}:
                _wait_for_health(url, args.wait_timeout, f"external worker {url}")

        router_env = _build_pythonpath_with_src(repo_root, os.environ.copy())
        router_cmd = [
            sys.executable,
            "-m",
            "sglang_diffusion_routing",
            "--host",
            args.router_host,
            "--port",
            str(args.router_port),
            "--worker-urls",
            *worker_urls,
            "--routing-algorithm",
            args.routing_algorithm,
            "--max-connections",
            str(args.router_max_connections),
            "--timeout",
            str(args.router_timeout),
            "--health-check-interval",
            str(args.router_health_check_interval),
            "--health-check-failure-threshold",
            str(args.router_health_check_failure_threshold),
            "--log-level",
            args.log_level.lower(),
        ]
        if args.router_verbose:
            router_cmd.append("--verbose")
        if args.router_extra_args:
            router_cmd += shlex.split(args.router_extra_args)

        print(f"[run] {' '.join(shlex.quote(x) for x in router_cmd)}", flush=True)
        router_proc = subprocess.Popen(
            router_cmd,
            cwd=repo_root,
            env=router_env,
            start_new_session=True,
        )
        managed_processes.append(router_proc)

        router_client_host = _infer_client_host(args.router_host)
        router_url = f"http://{router_client_host}:{args.router_port}"
        _wait_for_health(
            router_url, args.wait_timeout, f"router {router_url}", proc=router_proc
        )

        bench_cmd = _build_bench_command(args, router_url)
        print(f"[run] {' '.join(shlex.quote(x) for x in bench_cmd)}", flush=True)
        rc = subprocess.call(bench_cmd, cwd=repo_root)

        if rc == 0 and args.output_file:
            out = Path(args.output_file)
            if out.exists():
                try:
                    data = json.loads(out.read_text())
                    print(
                        "[bench] result summary:"
                        f" throughput_qps={data.get('throughput_qps')},"
                        f" latency_mean={data.get('latency_mean')},"
                        f" latency_p99={data.get('latency_p99')}",
                        flush=True,
                    )
                except Exception:
                    pass
        return rc
    finally:
        if managed_processes:
            _terminate_all(managed_processes)


if __name__ == "__main__":
    raise SystemExit(main())
