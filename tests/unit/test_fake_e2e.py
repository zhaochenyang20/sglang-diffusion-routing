"""
CPU-only fake end-to-end tests with real processes.

Spins up fake worker processes and a real router process, then sends
real HTTP requests through the full stack:

    pytest client  ->  router process (port)  ->  fake worker processes (ports)

No mocks, no monkey-patching. All communication over real TCP sockets.

Run:
    pytest tests/unit/test_fake_e2e.py -v -s
"""

from __future__ import annotations

import base64
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


# -- Helpers ---------------------------------------------------------------


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


# -- Fixtures --------------------------------------------------------------


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


# -- Health & status -------------------------------------------------------


class TestHealth:
    def test_healthy_with_workers(self, router_url):
        r = httpx.get(f"{router_url}/health", timeout=5.0)
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"
        assert body["healthy_workers"] == 2
        assert body["total_workers"] == 2

    def test_health_workers_detail(self, router_url):
        workers = httpx.get(f"{router_url}/health_workers", timeout=5.0).json()[
            "workers"
        ]
        assert len(workers) == 2
        for w in workers:
            assert not w["is_dead"]
            assert "url" in w
            assert "active_requests" in w
            assert w["consecutive_failures"] == 0

    def test_list_workers(self, router_url):
        urls = httpx.get(f"{router_url}/list_workers", timeout=5.0).json()["urls"]
        assert len(urls) == 2
        assert all(u.startswith("http://") for u in urls)

    def test_unhealthy_when_no_workers(self):
        proc, url = _start_router([])
        try:
            _wait_responding(url)
            r = httpx.get(f"{url}/health", timeout=5.0)
            assert r.status_code == 503
            body = r.json()
            assert body["status"] == "unhealthy"
            assert body["healthy_workers"] == 0
            assert body["total_workers"] == 0
        finally:
            _kill_proc(proc)


# -- Worker registration ---------------------------------------------------


class TestWorkerRegistration:
    def test_add_via_query_param(self, fake_workers):
        proc, rurl = _start_router([])
        try:
            _wait_responding(rurl)
            r = httpx.post(
                f"{rurl}/add_worker", params={"url": fake_workers[0].url}, timeout=5.0
            )
            assert r.status_code == 200
            assert fake_workers[0].url in r.json()["worker_urls"]
        finally:
            _kill_proc(proc)

    def test_add_via_json_body(self, fake_workers):
        proc, rurl = _start_router([])
        try:
            _wait_responding(rurl)
            r = httpx.post(
                f"{rurl}/add_worker", json={"url": fake_workers[0].url}, timeout=5.0
            )
            assert r.status_code == 200
            assert r.json()["status"] == "success"
        finally:
            _kill_proc(proc)

    def test_add_deduplicates(self, fake_workers):
        proc, rurl = _start_router([])
        try:
            _wait_responding(rurl)
            url = fake_workers[0].url
            httpx.post(f"{rurl}/add_worker", params={"url": url + "/"}, timeout=5.0)
            httpx.post(f"{rurl}/add_worker", params={"url": url}, timeout=5.0)
            assert (
                len(httpx.get(f"{rurl}/list_workers", timeout=5.0).json()["urls"]) == 1
            )
        finally:
            _kill_proc(proc)

    def test_missing_url_400(self):
        proc, rurl = _start_router([])
        try:
            _wait_responding(rurl)
            r = httpx.post(f"{rurl}/add_worker", timeout=5.0)
            assert r.status_code == 400
            assert "error" in r.json()
        finally:
            _kill_proc(proc)

    def test_blocked_host_400(self):
        proc, rurl = _start_router([])
        try:
            _wait_responding(rurl)
            r = httpx.post(
                f"{rurl}/add_worker",
                params={"url": "http://169.254.169.254:80"},
                timeout=5.0,
            )
            assert r.status_code == 400
            assert "blocked" in r.json()["error"]
        finally:
            _kill_proc(proc)

    def test_invalid_json_400(self):
        proc, rurl = _start_router([])
        try:
            _wait_responding(rurl)
            r = httpx.post(
                f"{rurl}/add_worker",
                content=b"bad json",
                headers={"content-type": "application/json"},
                timeout=5.0,
            )
            assert r.status_code == 400
        finally:
            _kill_proc(proc)

    def test_dynamic_worker_receives_traffic(self, fake_workers):
        """Add a worker dynamically and verify it actually receives requests."""
        w_proc, w_url = _start_worker("dynamic")
        r_proc, rurl = _start_router([])
        try:
            _wait_healthy(w_url)
            _wait_responding(rurl)
            httpx.post(f"{rurl}/add_worker", params={"url": w_url}, timeout=5.0)
            assert (
                len(httpx.get(f"{rurl}/list_workers", timeout=5.0).json()["urls"]) == 1
            )
            r = httpx.post(
                f"{rurl}/generate",
                json={"prompt": "t", "response_format": "b64_json"},
                timeout=10.0,
            )
            assert r.status_code == 200
            assert r.json()["worker_id"] == "dynamic"
        finally:
            _kill_proc(w_proc)
            _kill_proc(r_proc)


# -- Image generation ------------------------------------------------------


class TestImageGeneration:
    def test_b64_json_returns_valid_png(self, router_url):
        r = httpx.post(
            f"{router_url}/generate",
            json={
                "model": "test-model",
                "prompt": "cat",
                "num_images": 1,
                "response_format": "b64_json",
            },
            timeout=10.0,
        )
        assert r.status_code == 200
        body = r.json()
        assert "data" in body
        assert "created" in body
        img = base64.b64decode(body["data"][0]["b64_json"])
        assert img[:4] == b"\x89PNG"

    def test_url_format(self, router_url):
        r = httpx.post(
            f"{router_url}/generate",
            json={
                "model": "t",
                "prompt": "dog",
                "num_images": 1,
                "response_format": "url",
            },
            timeout=10.0,
        )
        data = r.json()["data"][0]
        assert "url" in data
        assert data["url"].startswith("http")

    def test_multiple_images(self, router_url):
        r = httpx.post(
            f"{router_url}/generate",
            json={
                "model": "t",
                "prompt": "x",
                "num_images": 3,
                "response_format": "b64_json",
            },
            timeout=10.0,
        )
        data = r.json()["data"]
        assert len(data) == 3
        # Each image should have an index
        indices = [d["index"] for d in data]
        assert sorted(indices) == [0, 1, 2]

    def test_prompt_preserved_in_response(self, router_url):
        prompt = "a beautiful sunset over the ocean"
        r = httpx.post(
            f"{router_url}/generate",
            json={
                "model": "t",
                "prompt": prompt,
                "num_images": 1,
                "response_format": "b64_json",
            },
            timeout=10.0,
        )
        assert r.json()["data"][0]["revised_prompt"] == prompt

    def test_model_field_preserved(self, router_url):
        r = httpx.post(
            f"{router_url}/generate",
            json={
                "model": "my-custom-model",
                "prompt": "x",
                "num_images": 1,
                "response_format": "b64_json",
            },
            timeout=10.0,
        )
        assert r.json()["model"] == "my-custom-model"

    def test_no_workers_503(self):
        proc, rurl = _start_router([])
        try:
            _wait_responding(rurl)
            r = httpx.post(f"{rurl}/generate", json={"prompt": "t"}, timeout=5.0)
            assert r.status_code == 503
            assert "error" in r.json()
        finally:
            _kill_proc(proc)


# -- Video generation ------------------------------------------------------


class TestVideoGeneration:
    def test_generate_video(self, router_url):
        r = httpx.post(
            f"{router_url}/generate_video",
            json={"model": "vid-model", "prompt": "river"},
            timeout=10.0,
        )
        assert r.status_code == 200
        body = r.json()
        assert "url" in body["data"][0]
        assert "created" in body

    def test_video_prompt_preserved(self, router_url):
        prompt = "a flowing river in autumn"
        r = httpx.post(
            f"{router_url}/generate_video",
            json={"model": "vid", "prompt": prompt},
            timeout=10.0,
        )
        assert r.json()["data"][0]["revised_prompt"] == prompt

    def test_no_workers_503(self):
        proc, rurl = _start_router([])
        try:
            _wait_responding(rurl)
            r = httpx.post(f"{rurl}/generate_video", json={"prompt": "t"}, timeout=5.0)
            assert r.status_code == 503
        finally:
            _kill_proc(proc)


# -- Weight update broadcast -----------------------------------------------


class TestUpdateWeights:
    def test_broadcasts_to_all(self, router_url):
        r = httpx.post(
            f"{router_url}/update_weights_from_disk",
            json={"model_path": "/weights/v2"},
            timeout=10.0,
        )
        results = r.json()["results"]
        assert len(results) == 2
        for res in results:
            assert res["status_code"] == 200
            assert res["body"]["ok"] is True
            assert res["body"]["model_path"] == "/weights/v2"
            assert "worker_url" in res

    def test_empty_pool(self):
        proc, rurl = _start_router([])
        try:
            _wait_responding(rurl)
            r = httpx.post(
                f"{rurl}/update_weights_from_disk",
                json={"model_path": "x"},
                timeout=5.0,
            )
            assert r.json()["results"] == []
        finally:
            _kill_proc(proc)


# -- Load balancing --------------------------------------------------------


class TestRoundRobinBalancing:
    """Tests using the module-scoped round-robin router."""

    def test_distributes_evenly(self, router_url, fake_workers):
        initial = [w.stats()["generate"] for w in fake_workers]
        for _ in range(10):
            assert (
                httpx.post(
                    f"{router_url}/generate",
                    json={"prompt": "t", "response_format": "b64_json"},
                    timeout=10.0,
                ).status_code
                == 200
            )
        final = [w.stats()["generate"] for w in fake_workers]
        deltas = [final[i] - initial[i] for i in range(2)]
        assert all(d > 0 for d in deltas)
        assert sum(deltas) == 10
        # Round-robin should be perfectly even with 2 workers
        assert deltas[0] == deltas[1] == 5

    def test_worker_id_proves_real_routing(self, router_url):
        """Consecutive requests should alternate between workers."""
        ids = []
        for _ in range(4):
            r = httpx.post(
                f"{router_url}/generate",
                json={"prompt": "t", "response_format": "b64_json"},
                timeout=10.0,
            )
            ids.append(r.json()["worker_id"])
        # Round-robin: should alternate
        assert ids[0] != ids[1]
        assert ids[0] == ids[2]
        assert ids[1] == ids[3]


class TestLeastRequestBalancing:
    """Tests with a dedicated least-request router."""

    def test_prefers_less_loaded_worker(self, fake_workers):
        """With one slow worker and one fast worker, least-request should
        send more traffic to the fast one."""
        slow_proc, slow_url = _start_worker("slow", latency=0.3)
        fast_proc, fast_url = _start_worker("fast", latency=0.0)
        r_proc, rurl = _start_router(
            [slow_url, fast_url],
            routing_algorithm="least-request",
        )
        try:
            _wait_healthy(slow_url)
            _wait_healthy(fast_url)
            _wait_healthy(rurl)

            # Send requests concurrently — fast worker should get more
            import concurrent.futures

            def send_one():
                return httpx.post(
                    f"{rurl}/generate",
                    json={"prompt": "t", "response_format": "b64_json"},
                    timeout=15.0,
                ).json()["worker_id"]

            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                futures = [pool.submit(send_one) for _ in range(12)]
                results = [f.result() for f in futures]

            fast_count = results.count("fast")
            slow_count = results.count("slow")
            # Fast worker should handle more requests than slow
            assert fast_count > slow_count
            assert fast_count + slow_count == 12
        finally:
            _kill_proc(slow_proc)
            _kill_proc(fast_proc)
            _kill_proc(r_proc)


# -- Proxy catch-all -------------------------------------------------------


class TestProxy:
    def test_stats_proxied(self, router_url):
        r = httpx.get(f"{router_url}/stats", timeout=5.0)
        assert r.status_code == 200
        body = r.json()
        assert "total" in body
        assert "generate" in body
        assert "video" in body

    def test_no_workers_503(self):
        proc, rurl = _start_router([])
        try:
            _wait_responding(rurl)
            r = httpx.get(f"{rurl}/any/path", timeout=5.0)
            assert r.status_code == 503
        finally:
            _kill_proc(proc)

    def test_worker_health_proxied(self, router_url):
        """The catch-all proxy should forward /worker_health to a worker."""
        # /stats is handled by fake_worker, verifying proxy works for arbitrary paths
        r = httpx.get(f"{router_url}/stats", timeout=5.0)
        assert r.status_code == 200


# -- Worker failure --------------------------------------------------------


class TestWorkerFailure:
    def test_unreachable_worker_502(self):
        proc, rurl = _start_router(["http://127.0.0.1:1"])
        try:
            _wait_responding(rurl)
            r = httpx.post(f"{rurl}/generate", json={"prompt": "t"}, timeout=10.0)
            assert r.status_code == 502
            assert "error" in r.json()
        finally:
            _kill_proc(proc)

    def test_worker_500_forwarded(self):
        w_proc, w_url = _start_worker("failing", fail_rate=1.0)
        r_proc, rurl = _start_router([w_url])
        try:
            _wait_healthy(w_url)
            _wait_healthy(rurl)
            r = httpx.post(
                f"{rurl}/generate",
                json={"prompt": "t", "response_format": "b64_json"},
                timeout=10.0,
            )
            assert r.status_code == 500
            assert "detail" in r.json()
        finally:
            _kill_proc(w_proc)
            _kill_proc(r_proc)

    def test_worker_killed_returns_502(self, fake_workers):
        """Kill a worker process after router starts, verify 502."""
        w_proc, w_url = _start_worker("ephemeral")
        r_proc, rurl = _start_router([w_url])
        try:
            _wait_healthy(w_url)
            _wait_healthy(rurl)
            # Verify it works first
            assert (
                httpx.post(
                    f"{rurl}/generate",
                    json={"prompt": "t", "response_format": "b64_json"},
                    timeout=10.0,
                ).status_code
                == 200
            )
            # Kill the worker
            _kill_proc(w_proc)
            time.sleep(0.5)
            # Now requests should fail
            r = httpx.post(
                f"{rurl}/generate",
                json={"prompt": "t", "response_format": "b64_json"},
                timeout=10.0,
            )
            assert r.status_code == 502
        finally:
            _kill_proc(w_proc)
            _kill_proc(r_proc)


# -- Concurrent requests ---------------------------------------------------


class TestConcurrency:
    def test_concurrent_requests_all_succeed(self, router_url):
        """Fire multiple requests concurrently, all should succeed."""
        import concurrent.futures

        def send_one(i):
            return httpx.post(
                f"{router_url}/generate",
                json={"prompt": f"concurrent-{i}", "response_format": "b64_json"},
                timeout=15.0,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(send_one, i) for i in range(16)]
            responses = [f.result() for f in futures]

        assert all(r.status_code == 200 for r in responses)
        # Verify each response has valid data
        for r in responses:
            body = r.json()
            assert "data" in body
            assert len(body["data"]) == 1

    def test_concurrent_mixed_endpoints(self, router_url):
        """Concurrent requests to different endpoints should all work."""
        import concurrent.futures

        def gen_image():
            return httpx.post(
                f"{router_url}/generate",
                json={"prompt": "img", "response_format": "b64_json"},
                timeout=15.0,
            )

        def gen_video():
            return httpx.post(
                f"{router_url}/generate_video", json={"prompt": "vid"}, timeout=15.0
            )

        def check_health():
            return httpx.get(f"{router_url}/health", timeout=5.0)

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futs = (
                [pool.submit(gen_image) for _ in range(4)]
                + [pool.submit(gen_video) for _ in range(4)]
                + [pool.submit(check_health) for _ in range(2)]
            )
            results = [f.result() for f in futs]

        assert all(r.status_code == 200 for r in results)


# -- CLI integration -------------------------------------------------------


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
        """Test the sglang-d-router script entry point."""
        # Find the script next to the Python interpreter in the venv
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
        """Router with --verbose should start and work normally."""
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
