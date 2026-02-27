"""
Integration tests: CLI integration.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import httpx
import pytest

from .common import PYTHON, _env, _find_free_port, _kill_proc, _wait_healthy

pytestmark = pytest.mark.integration


class TestCLI:
    def test_cli_starts_working_router(self, fake_workers):
        port = _find_free_port()
        proc = subprocess.Popen(
            [
                PYTHON,
                "-m",
                "sglang_diffusion_routing",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--worker-urls",
                *[w.url for w in fake_workers],
                "--routing-algorithm",
                "least-request",
                "--log-level",
                "warning",
            ],
            env=_env(),
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        url = f"http://127.0.0.1:{port}"
        try:
            _wait_healthy(url)
            assert httpx.get(f"{url}/health", timeout=5.0).status_code == 200
            r = httpx.post(
                f"{url}/generate",
                json={"prompt": "cli", "response_format": "b64_json"},
                timeout=10.0,
            )
            assert r.status_code == 200
            assert "data" in r.json()
        finally:
            _kill_proc(proc)

    def test_script_entry_point(self, fake_workers):
        script = Path(PYTHON).parent / "sglang-d-router"
        if not script.exists():
            pytest.skip("sglang-d-router not installed in PATH")
        port = _find_free_port()
        proc = subprocess.Popen(
            [
                str(script),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--worker-urls",
                *[w.url for w in fake_workers],
                "--log-level",
                "warning",
            ],
            env=_env(),
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        url = f"http://127.0.0.1:{port}"
        try:
            _wait_healthy(url)
            assert httpx.get(f"{url}/health", timeout=5.0).status_code == 200
        finally:
            _kill_proc(proc)

    def test_help_flag(self):
        r = subprocess.run(
            [PYTHON, "-m", "sglang_diffusion_routing", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert r.returncode == 0
        assert "sglang-d-router" in r.stdout

    def test_verbose_flag(self, fake_workers):
        port = _find_free_port()
        proc = subprocess.Popen(
            [
                PYTHON,
                "-m",
                "sglang_diffusion_routing",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--worker-urls",
                *[w.url for w in fake_workers],
                "--verbose",
                "--log-level",
                "warning",
            ],
            env=_env(),
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        url = f"http://127.0.0.1:{port}"
        try:
            _wait_healthy(url)
            assert httpx.get(f"{url}/health", timeout=5.0).json()["status"] == "healthy"
        finally:
            _kill_proc(proc)
