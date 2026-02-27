"""
Shared helpers/fixtures for CPU-only integration tests.

These tests spawn real router and fake worker processes and validate behavior
through real HTTP requests over TCP sockets.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FAKE_WORKER_SCRIPT = Path(__file__).resolve().parent / "fake_worker.py"
PYTHON = sys.executable


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_healthy(url: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{url}/health", timeout=2.0)
            if r.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise TimeoutError(f"Service at {url} not healthy within {timeout}s")


def _wait_responding(url: str, timeout: float = 10.0) -> None:
    """Wait for any HTTP response (even 503)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            httpx.get(f"{url}/health", timeout=2.0)
            return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise TimeoutError(f"Service at {url} not responding within {timeout}s")


def _kill_proc(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait(timeout=3)


def _env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(REPO_ROOT / "src")
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src}:{old}" if old else src
    return env


def _start_worker(worker_id: str, **kw) -> tuple[subprocess.Popen, str]:
    port = _find_free_port()
    cmd = [
        PYTHON,
        str(FAKE_WORKER_SCRIPT),
        "--port",
        str(port),
        "--worker-id",
        worker_id,
    ]
    for k, v in kw.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    proc = subprocess.Popen(
        cmd,
        env=_env(),
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc, f"http://127.0.0.1:{port}"


def _start_router(worker_urls: list[str], **kw) -> tuple[subprocess.Popen, str]:
    port = _find_free_port()
    cmd = [
        PYTHON,
        "-m",
        "sglang_diffusion_routing",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--health-check-interval",
        "3600",
        "--log-level",
        "warning",
    ]
    if worker_urls:
        cmd += ["--worker-urls", *worker_urls]
    for k, v in kw.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    proc = subprocess.Popen(
        cmd,
        env=_env(),
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc, f"http://127.0.0.1:{port}"


class FakeWorker:
    def __init__(self, proc, url, worker_id):
        self.proc, self.url, self.worker_id = proc, url, worker_id

    def stats(self) -> dict:
        return httpx.get(f"{self.url}/stats", timeout=3.0).json()


@pytest.fixture(scope="module")
def fake_workers():
    workers, procs = [], []
    for i in range(2):
        proc, url = _start_worker(f"worker-{i}")
        procs.append(proc)
        workers.append(FakeWorker(proc, url, f"worker-{i}"))
    for w in workers:
        _wait_healthy(w.url)
    yield workers
    for p in procs:
        _kill_proc(p)


@pytest.fixture(scope="module")
def router_url(fake_workers):
    proc, url = _start_router(
        [w.url for w in fake_workers], routing_algorithm="round-robin"
    )
    try:
        _wait_healthy(url)
    except TimeoutError:
        _kill_proc(proc)
        raise
    yield url
    _kill_proc(proc)
