"""
Integration tests: runtime behavior (load balancing, failures, concurrency).
"""

from __future__ import annotations

import concurrent.futures
import time

import httpx
import pytest

from .common import (
    _kill_proc,
    _start_router,
    _start_worker,
    _wait_healthy,
    _wait_responding,
)

pytestmark = pytest.mark.integration


class TestRoundRobinBalancing:
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
        assert deltas[0] == deltas[1] == 5

    def test_worker_id_proves_real_routing(self, router_url):
        ids = []
        for _ in range(4):
            r = httpx.post(
                f"{router_url}/generate",
                json={"prompt": "t", "response_format": "b64_json"},
                timeout=10.0,
            )
            ids.append(r.json()["worker_id"])
        assert ids[0] != ids[1]
        assert ids[0] == ids[2]
        assert ids[1] == ids[3]


class TestLeastRequestBalancing:
    def test_prefers_less_loaded_worker(self, fake_workers):
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
            assert fast_count > slow_count
            assert fast_count + slow_count == 12
        finally:
            _kill_proc(slow_proc)
            _kill_proc(fast_proc)
            _kill_proc(r_proc)


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
        w_proc, w_url = _start_worker("ephemeral")
        r_proc, rurl = _start_router([w_url])
        try:
            _wait_healthy(w_url)
            _wait_healthy(rurl)
            assert (
                httpx.post(
                    f"{rurl}/generate",
                    json={"prompt": "t", "response_format": "b64_json"},
                    timeout=10.0,
                ).status_code
                == 200
            )
            _kill_proc(w_proc)
            time.sleep(0.5)
            r = httpx.post(
                f"{rurl}/generate",
                json={"prompt": "t", "response_format": "b64_json"},
                timeout=10.0,
            )
            assert r.status_code == 502
        finally:
            _kill_proc(w_proc)
            _kill_proc(r_proc)


class TestConcurrency:
    def test_concurrent_requests_all_succeed(self, router_url):
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
        for r in responses:
            body = r.json()
            assert "data" in body
            assert len(body["data"]) == 1

    def test_concurrent_mixed_endpoints(self, router_url):
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
