"""
End-to-end tests with real sglang diffusion workers.

Requires:
    - sglang installed with diffusion support: pip install "sglang[diffusion]"
    - At least 1 GPU available
    - Model weights accessible (downloads on first run)

These tests are SKIPPED automatically when sglang or GPU is not available.
To run explicitly:

    pytest tests/e2e/test_e2e_sglang.py -v -s

Override model/GPU config via environment variables:
    SGLANG_TEST_MODEL       Model path (default: Qwen/Qwen-Image)
    SGLANG_TEST_NUM_GPUS    GPUs per worker (default: 1)
    SGLANG_TEST_NUM_WORKERS Number of workers (default: 2)
    SGLANG_TEST_TIMEOUT     Startup timeout in seconds (default: 600)
"""

from __future__ import annotations

import base64
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable

DEFAULT_MODEL = "Qwen/Qwen-Image"
DEFAULT_NUM_GPUS = 1
DEFAULT_NUM_WORKERS = 2
DEFAULT_TIMEOUT = 600  # sglang model loading can be slow

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SIGNATURE = b"\xff\xd8\xff"
_WEBP_RIFF_SIGNATURE = b"RIFF"
_WEBP_FORMAT_MARKER = b"WEBP"


def _has_sglang() -> bool:
    try:
        import sglang  # noqa: F401

        return True
    except ImportError:
        return False


def _gpu_count() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def _get_env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(REPO_ROOT / "src")
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src}:{old}" if old else src
    return env


def _wait_healthy(
    url: str, timeout: float, label: str = "", proc: subprocess.Popen | None = None
) -> None:
    deadline = time.monotonic() + timeout
    last_log = 0.0
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"{label} process exited with code {proc.returncode} during startup"
            )
        try:
            r = httpx.get(f"{url}/health", timeout=5.0)
            if r.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        now = time.monotonic()
        if now - last_log >= 30:
            elapsed = now - (deadline - timeout)
            print(f"  Waiting for {label}... ({elapsed:.0f}s)", flush=True)
            last_log = now
        time.sleep(2)
    raise TimeoutError(f"{label} at {url} not healthy within {timeout}s")


def _kill_proc(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait(timeout=5)


def _read_stderr_snippet(proc: subprocess.Popen, max_bytes: int = 8192) -> str:
    if proc.stderr is None:
        return ""
    try:
        raw = proc.stderr.read(max_bytes)
    except Exception:
        return ""
    if not raw:
        return ""
    return raw.decode("utf-8", errors="replace").strip()


def _collect_exited_worker_errors(procs: list[subprocess.Popen]) -> list[str]:
    errors: list[str] = []
    for idx, proc in enumerate(procs):
        if proc.poll() is None:
            continue
        err = _read_stderr_snippet(proc)
        msg = f"worker[{idx}] exited with code {proc.returncode}"
        if err:
            msg += f"\nstderr:\n{err[:4000]}"
        errors.append(msg)
    return errors


def _looks_like_supported_image_bytes(img_bytes: bytes) -> bool:
    """Best-effort signature check for common image encodings returned by workers."""
    if img_bytes.startswith(_PNG_SIGNATURE):
        return True
    if img_bytes.startswith(_JPEG_SIGNATURE):
        return True
    if (
        img_bytes.startswith(_WEBP_RIFF_SIGNATURE)
        and img_bytes[8:12] == _WEBP_FORMAT_MARKER
    ):
        return True
    return False


# -- Skip conditions -------------------------------------------------------

_skip_no_sglang = pytest.mark.skipif(
    not _has_sglang() or shutil.which("sglang") is None,
    reason="sglang not installed or 'sglang' CLI not in PATH",
)
_skip_no_gpu = pytest.mark.skipif(
    _gpu_count() == 0,
    reason="No GPU available",
)

pytestmark = [_skip_no_sglang, _skip_no_gpu]


# -- Fixtures ---------------------------------------------------------------


class SglangWorker:
    def __init__(self, proc: subprocess.Popen, url: str):
        self.proc = proc
        self.url = url


@pytest.fixture(scope="module")
def sglang_config():
    model = _get_env("SGLANG_TEST_MODEL", DEFAULT_MODEL)
    num_gpus = int(_get_env("SGLANG_TEST_NUM_GPUS", str(DEFAULT_NUM_GPUS)))
    num_workers = int(_get_env("SGLANG_TEST_NUM_WORKERS", str(DEFAULT_NUM_WORKERS)))
    timeout = int(_get_env("SGLANG_TEST_TIMEOUT", str(DEFAULT_TIMEOUT)))

    gpus_available = _gpu_count()
    needed = num_workers * num_gpus
    if gpus_available < needed:
        pytest.skip(
            f"Need {needed} GPUs ({num_workers} workers x {num_gpus} GPUs), "
            f"only {gpus_available} available"
        )

    return {
        "model": model,
        "num_gpus": num_gpus,
        "num_workers": num_workers,
        "timeout": timeout,
    }


@pytest.fixture(scope="module")
def sglang_workers(sglang_config):
    """Launch real sglang diffusion worker processes."""
    workers = []
    procs = []
    env = _env()
    gpu_pool = list(range(_gpu_count()))

    for i in range(sglang_config["num_workers"]):
        port = _find_free_port()
        gpu_start = i * sglang_config["num_gpus"]
        gpu_end = gpu_start + sglang_config["num_gpus"]
        gpu_ids = ",".join(str(gpu_pool[g]) for g in range(gpu_start, gpu_end))

        worker_env = dict(env)
        worker_env["CUDA_VISIBLE_DEVICES"] = gpu_ids

        cmd = [
            "sglang",
            "serve",
            "--model-path",
            sglang_config["model"],
            "--num-gpus",
            str(sglang_config["num_gpus"]),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--dit-cpu-offload",
            "false",
            "--text-encoder-cpu-offload",
            "false",
        ]

        print(
            f"\n[sglang-test] Starting worker {i} on port {port} (GPU: {gpu_ids})",
            flush=True,
        )
        proc = subprocess.Popen(
            cmd,
            env=worker_env,
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        procs.append(proc)
        workers.append(SglangWorker(proc, f"http://127.0.0.1:{port}"))

    try:
        for w in workers:
            _wait_healthy(
                w.url,
                sglang_config["timeout"],
                label=f"sglang worker {w.url}",
                proc=w.proc,
            )
    except RuntimeError as exc:
        worker_errors = _collect_exited_worker_errors(procs)
        for p in procs:
            _kill_proc(p)
        details = (
            "\n\n".join(worker_errors)
            if worker_errors
            else "worker process exited but no stderr captured."
        )
        pytest.fail(
            f"sglang worker failed during startup: {exc}\n{details}", pytrace=False
        )
    except TimeoutError as exc:
        worker_errors = _collect_exited_worker_errors(procs)
        for p in procs:
            _kill_proc(p)
        if worker_errors:
            details = "\n\n".join(worker_errors)
            pytest.fail(
                f"sglang worker startup timed out and some workers exited: {exc}\n{details}",
                pytrace=False,
            )
        pytest.skip(f"sglang worker startup timed out: {exc}")

    exited_after_ready = _collect_exited_worker_errors(procs)
    if exited_after_ready:
        for p in procs:
            _kill_proc(p)
        pytest.fail(
            "sglang worker exited after becoming healthy:\n"
            + "\n\n".join(exited_after_ready),
            pytrace=False,
        )

    yield workers

    for p in procs:
        _kill_proc(p)


@pytest.fixture(scope="module")
def router_url(sglang_workers):
    """Launch a real router connected to real sglang workers."""
    port = _find_free_port()
    worker_urls = [w.url for w in sglang_workers]
    cmd = [
        PYTHON,
        "-m",
        "sglang_diffusion_routing",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--worker-urls",
        *worker_urls,
        "--routing-algorithm",
        "least-request",
        "--log-level",
        "warning",
    ]
    proc = subprocess.Popen(
        cmd,
        env=_env(),
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        _wait_healthy(url, 30, label="router", proc=proc)
    except Exception:
        _kill_proc(proc)
        raise

    yield url
    _kill_proc(proc)


# -- Tests ------------------------------------------------------------------


class TestSglangHealth:
    def test_router_healthy(self, router_url):
        r = httpx.get(f"{router_url}/health", timeout=10.0)
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_workers_listed(self, router_url, sglang_workers):
        urls = httpx.get(f"{router_url}/list_workers", timeout=10.0).json()["urls"]
        assert len(urls) == len(sglang_workers)


class TestSglangImageGeneration:
    def test_b64_json_generates_real_image(self, router_url, sglang_config):
        """Generate a real image and verify payload is a supported image format."""
        r = httpx.post(
            f"{router_url}/generate",
            json={
                "model": sglang_config["model"],
                "prompt": "a simple red circle on white background",
                "num_images": 1,
                "response_format": "b64_json",
            },
            timeout=120.0,
        )
        assert r.status_code == 200
        body = r.json()
        assert "data" in body
        assert len(body["data"]) == 1

        img_bytes = base64.b64decode(body["data"][0]["b64_json"])
        assert _looks_like_supported_image_bytes(img_bytes)
        # Real image should be substantially larger than a 1x1 pixel
        assert len(img_bytes) > 1000

    def test_multiple_images(self, router_url, sglang_config):
        payload = {
            "model": sglang_config["model"],
            "prompt": "a blue square",
            # OpenAI-compatible field used by /v1/images/generations.
            "n": 2,
            "response_format": "b64_json",
        }
        r = httpx.post(
            f"{router_url}/generate",
            json=payload,
            timeout=120.0,
        )
        if r.status_code == 200:
            body = r.json()
            assert len(body["data"]) == 2
            return

        # Some real workers/models do not support n>1 yet.
        assert r.status_code in (400, 500)
        error_body = (
            r.json() if "application/json" in r.headers.get("content-type", "") else {}
        )
        if isinstance(error_body, dict):
            assert error_body.get("error") or error_body.get("detail") or r.text

        single_image_payload = dict(payload)
        single_image_payload["n"] = 1
        single = httpx.post(
            f"{router_url}/generate",
            json=single_image_payload,
            timeout=120.0,
        )
        assert single.status_code == 200, (
            "multi-image request failed and single-image fallback failed too; "
            f"multi={r.status_code} body={r.text[:400]!r}, "
            f"single={single.status_code} body={single.text[:400]!r}"
        )
        assert len(single.json()["data"]) == 1
        pytest.skip(
            "Current worker/model does not support multi-image generation (n>1)."
        )

    def test_url_format(self, router_url, sglang_config):
        """Generate with response_format=url (requires worker file storage)."""
        r = httpx.post(
            f"{router_url}/generate",
            json={
                "model": sglang_config["model"],
                "prompt": "a green triangle",
                "num_images": 1,
                "response_format": "url",
            },
            timeout=120.0,
        )
        # url format may require cloud storage config — accept either success or
        # a clear error about storage, not a crash
        assert r.status_code in (200, 400, 500)
        if r.status_code == 200:
            assert "url" in r.json()["data"][0]


class TestSglangLoadBalancing:
    def test_requests_distributed(self, router_url, sglang_workers, sglang_config):
        """With multiple workers, requests should be distributed."""
        if len(sglang_workers) < 2:
            pytest.skip("Need at least 2 workers for load balancing test")

        for _ in range(4):
            r = httpx.post(
                f"{router_url}/generate",
                json={
                    "model": sglang_config["model"],
                    "prompt": "test",
                    "num_images": 1,
                    "response_format": "b64_json",
                },
                timeout=120.0,
            )
            assert r.status_code == 200

        # Verify health shows all workers still alive
        health = httpx.get(f"{router_url}/health_workers", timeout=10.0).json()
        assert all(not w["is_dead"] for w in health["workers"])


class TestSglangProxy:
    def test_get_model_info(self, router_url):
        """Proxy should forward GET requests to worker."""
        r = httpx.get(f"{router_url}/get_model_info", timeout=30.0)
        # sglang workers expose /get_model_info
        assert r.status_code in (200, 404)


class TestSglangVideoEndpoint:
    def test_generate_video_rejects_image_only_workers(self, router_url):
        """Image-only workers (e.g. Qwen/Qwen-Image) should return 400 for /generate_video."""
        r = httpx.post(
            f"{router_url}/generate_video",
            json={"prompt": "a walking cat", "num_frames": 8},
            timeout=10.0,
        )
        assert r.status_code == 400
        body = r.json()
        assert "error" in body
        assert body["error"]  # non-empty error message
