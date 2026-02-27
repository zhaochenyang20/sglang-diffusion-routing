"""
Integration tests: API endpoints (health, workers, image, video, weights, proxy).
"""

from __future__ import annotations

import base64

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


class TestHealth:
    def test_healthy_with_workers(self, router_url):
        r = httpx.get(f"{router_url}/health", timeout=5.0)
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"
        assert body["healthy_workers"] == 2
        assert body["total_workers"] == 2

    def test_health_workers_detail(self, router_url):
        workers = httpx.get(f"{router_url}/workers", timeout=5.0).json()["workers"]
        assert len(workers) == 2
        for w in workers:
            assert not w["is_dead"]
            assert "url" in w
            assert "active_requests" in w
            assert w["consecutive_failures"] == 0

    def test_list_workers(self, router_url):
        workers = httpx.get(f"{router_url}/workers", timeout=5.0).json()["workers"]
        urls = [worker["url"] for worker in workers]
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


class TestWorkerRegistration:
    def test_add_via_query_param(self, fake_workers):
        proc, rurl = _start_router([])
        try:
            _wait_responding(rurl)
            r = httpx.post(
                f"{rurl}/workers", params={"url": fake_workers[0].url}, timeout=5.0
            )
            assert r.status_code == 200
            assert r.json()["worker"]["url"] == fake_workers[0].url
        finally:
            _kill_proc(proc)

    def test_add_via_json_body(self, fake_workers):
        proc, rurl = _start_router([])
        try:
            _wait_responding(rurl)
            r = httpx.post(
                f"{rurl}/workers", json={"url": fake_workers[0].url}, timeout=5.0
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
            httpx.post(f"{rurl}/workers", params={"url": url + "/"}, timeout=5.0)
            httpx.post(f"{rurl}/workers", params={"url": url}, timeout=5.0)
            workers = httpx.get(f"{rurl}/workers", timeout=5.0).json()["workers"]
            assert len(workers) == 1
        finally:
            _kill_proc(proc)

    def test_missing_url_400(self):
        proc, rurl = _start_router([])
        try:
            _wait_responding(rurl)
            r = httpx.post(f"{rurl}/workers", timeout=5.0)
            assert r.status_code == 400
            assert "error" in r.json()
        finally:
            _kill_proc(proc)

    def test_blocked_host_400(self):
        proc, rurl = _start_router([])
        try:
            _wait_responding(rurl)
            r = httpx.post(
                f"{rurl}/workers",
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
                f"{rurl}/workers",
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
            httpx.post(f"{rurl}/workers", params={"url": w_url}, timeout=5.0)
            workers = httpx.get(f"{rurl}/workers", timeout=5.0).json()["workers"]
            assert len(workers) == 1
            r = httpx.post(
                f"{rurl}/v1/images/generations",
                json={"prompt": "t", "response_format": "b64_json"},
                timeout=10.0,
            )
            assert r.status_code == 200
            assert r.json()["worker_id"] == "dynamic"
        finally:
            _kill_proc(w_proc)
            _kill_proc(r_proc)


class TestImageGeneration:
    def test_b64_json_returns_valid_png(self, router_url):
        r = httpx.post(
            f"{router_url}/v1/images/generations",
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
            f"{router_url}/v1/images/generations",
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
            f"{router_url}/v1/images/generations",
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
        indices = [d["index"] for d in data]
        assert sorted(indices) == [0, 1, 2]

    def test_prompt_preserved_in_response(self, router_url):
        prompt = "a beautiful sunset over the ocean"
        r = httpx.post(
            f"{router_url}/v1/images/generations",
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
            f"{router_url}/v1/images/generations",
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
            r = httpx.post(
                f"{rurl}/v1/images/generations", json={"prompt": "t"}, timeout=5.0
            )
            assert r.status_code == 503
            assert "error" in r.json()
        finally:
            _kill_proc(proc)


class TestVideoGeneration:
    def test_generate_video(self, router_url):
        r = httpx.post(
            f"{router_url}/v1/videos",
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
            f"{router_url}/v1/videos",
            json={"model": "vid", "prompt": prompt},
            timeout=10.0,
        )
        assert r.json()["data"][0]["revised_prompt"] == prompt

    def test_no_workers_503(self):
        proc, rurl = _start_router([])
        try:
            _wait_responding(rurl)
            r = httpx.post(f"{rurl}/v1/videos", json={"prompt": "t"}, timeout=5.0)
            assert r.status_code == 400
            assert "video-capable" in r.json()["error"]
        finally:
            _kill_proc(proc)


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
            assert r.status_code == 503
            assert "No healthy workers available" in r.json()["error"]
        finally:
            _kill_proc(proc)


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
        r = httpx.get(f"{router_url}/stats", timeout=5.0)
        assert r.status_code == 200
