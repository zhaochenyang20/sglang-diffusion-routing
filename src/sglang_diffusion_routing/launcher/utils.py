"""Shared utilities for launcher backends."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from collections.abc import Iterable

import httpx
import torch

# TODO (mengyang, shuwen, chenyang): these utils should be clean up.


def infer_connect_host(host: str) -> str:
    """Normalize bind-all addresses to loopback for client connections."""
    if host in ("0.0.0.0", "::", "localhost"):
        return "127.0.0.1"
    return host


def is_port_available(host: str, port: int) -> bool:
    """Check whether port on host is free (no listener)."""
    connect_host = infer_connect_host(host)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((connect_host, port)) != 0


def reserve_available_port(host: str, preferred_port: int, used_ports: set[int]) -> int:
    """Find and reserve a free port, scanning from preferred_port."""
    if preferred_port < 1 or preferred_port > 65535:
        raise ValueError(f"Invalid port: {preferred_port}")

    for port in range(preferred_port, 65536):
        if port in used_ports:
            continue
        if is_port_available(host, port):
            used_ports.add(port)
            return port

    for port in range(1024, preferred_port):
        if port in used_ports:
            continue
        if is_port_available(host, port):
            used_ports.add(port)
            return port

    raise RuntimeError(
        f"Unable to reserve a free port for host {host}. "
        f"Preferred start={preferred_port}."
    )


def resolve_gpu_pool(
    worker_gpu_ids: list[str] | None,
    env: dict[str, str] | None = None,
) -> list[str] | None:
    """Determine available GPU IDs from explicit list, CUDA_VISIBLE_DEVICES, or auto-detect."""
    if worker_gpu_ids:
        return [str(x) for x in worker_gpu_ids]

    if env is None:
        env = os.environ

    visible = env.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        parsed = [item.strip() for item in visible.split(",") if item.strip()]
        if parsed:
            return parsed

    gpu_count = int(torch.cuda.device_count())
    if gpu_count > 0:
        return [str(i) for i in range(gpu_count)]
    return None


def build_gpu_assignments(
    worker_gpu_ids: list[str] | None,
    num_workers: int,
    num_gpus_per_worker: int,
    env: dict[str, str] | None = None,
) -> list[str] | None:
    """Build a per-worker list of CUDA_VISIBLE_DEVICES strings.

    When worker_gpu_ids is provided, each element is used directly as the
    CUDA_VISIBLE_DEVICES value for the corresponding worker
    (e.g. ["0,1", "4,5"]).

    When worker_gpu_ids is None, GPUs are auto-detected from the
    CUDA_VISIBLE_DEVICES environment variable or torch.cuda.device_count()
    and sliced into groups of num_gpus_per_worker.

    Returns None when no GPUs can be determined (CPU-only mode).
    """
    if worker_gpu_ids:
        if len(worker_gpu_ids) != num_workers:
            raise ValueError(
                f"worker_gpu_ids has {len(worker_gpu_ids)} entries but "
                f"num_workers is {num_workers}; they must match."
            )
        return [str(x) for x in worker_gpu_ids]

    # Auto-detect available GPUs.
    if env is None:
        env = os.environ

    gpu_pool: list[str] | None = None
    visible = env.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        parsed = [item.strip() for item in visible.split(",") if item.strip()]
        if parsed:
            gpu_pool = parsed

    if gpu_pool is None:
        gpu_count = int(torch.cuda.device_count())
        if gpu_count > 0:
            gpu_pool = [str(i) for i in range(gpu_count)]

    if gpu_pool is None:
        return None

    needed = num_workers * num_gpus_per_worker
    if len(gpu_pool) < needed:
        raise RuntimeError(
            f"Not enough GPUs. Need {needed} ({num_workers} workers * "
            f"{num_gpus_per_worker} GPUs each), found {len(gpu_pool)} "
            f"in pool {gpu_pool}."
        )

    assignments: list[str] = []
    for i in range(num_workers):
        start = i * num_gpus_per_worker
        stop = start + num_gpus_per_worker
        assignments.append(",".join(gpu_pool[start:stop]))
    return assignments


def wait_for_health(
    url: str,
    timeout: int,
    label: str,
    proc: subprocess.Popen | None = None,
    log_prefix: str = "[launcher]",
) -> None:
    """Poll url/health until it returns 200 or timeout seconds elapse."""
    start = time.time()
    last_print = 0.0
    while True:
        elapsed = time.time() - start

        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"{label} process exited with code {proc.returncode}. "
                "Run the command directly to inspect startup errors."
            )

        try:
            resp = httpx.get(f"{url}/health", timeout=1.0)
            if resp.status_code == 200:
                print(
                    f"  {log_prefix} {label} is healthy ({elapsed:.0f}s)",
                    flush=True,
                )
                return
        except httpx.HTTPError:
            pass

        if elapsed - last_print >= 30:
            print(
                f"  {log_prefix} Still waiting for {label}... "
                f"({elapsed:.0f}s elapsed)",
                flush=True,
            )
            last_print = elapsed

        if elapsed > timeout:
            raise TimeoutError(f"Timed out waiting for {label} at {url}.")
        time.sleep(1)


def terminate_all(processes: Iterable[subprocess.Popen]) -> None:
    """Shut down all processes gracefully with SIGINT, then SIGKILL as fallback."""
    procs = list(processes)

    for proc in procs:
        try:
            os.kill(proc.pid, signal.SIGINT)
        except ProcessLookupError:
            pass

    for proc in procs:
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    for proc in procs:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
