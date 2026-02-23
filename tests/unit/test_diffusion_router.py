"""Unit tests for DiffusionRouter core functionality.

Tests routing algorithms, count management, worker registration,
and response building. All tests call real code directly — no mocks,
no HTTP, no fake interfaces.
"""

import asyncio
import json
from argparse import Namespace

import pytest

from sglang_diffusion_routing import DiffusionRouter


def _make_args(**overrides) -> Namespace:
    defaults = dict(
        host="127.0.0.1",
        port=30080,
        max_connections=100,
        timeout=120.0,
        routing_algorithm="least-request",
    )
    defaults.update(overrides)
    return Namespace(**defaults)


@pytest.fixture
def router_factory():
    created: list[DiffusionRouter] = []

    def _create(workers=None, dead=None, **kw):
        r = DiffusionRouter(_make_args(**kw))
        if workers is not None:
            r.worker_request_counts = dict(workers)
            r.worker_failure_counts = {u: 0 for u in workers}
        if dead:
            r.dead_workers = set(dead)
        created.append(r)
        return r

    yield _create
    for r in created:
        asyncio.run(r.client.aclose())


# ── Routing algorithms ────────────────────────────────────────────────


class TestLeastRequest:
    def test_picks_min_load(self, router_factory):
        r = router_factory(
            {"http://w1:8000": 5, "http://w2:8000": 2, "http://w3:8000": 8}
        )
        assert r._select_worker_by_routing() == "http://w2:8000"

    def test_selects_min_load(self, router_factory):
        router = router_factory(
            {"http://w1:8000": 5, "http://w2:8000": 2, "http://w3:8000": 8}
        )
        selected = router._select_worker_by_routing()
        assert selected == "http://w2:8000"
        assert router.worker_request_counts["http://w2:8000"] == 3

    def test_excludes_dead_workers(self, router_factory):
        router = router_factory(
            {"http://w1:8000": 5, "http://w2:8000": 2, "http://w3:8000": 8},
            dead={"http://w2:8000"},
        )
        selected = router._select_worker_by_routing()
        assert selected == "http://w1:8000"
        assert router.worker_request_counts["http://w1:8000"] == 6


class TestRoundRobin:
    def test_cycles_in_order(self, router_factory):
        r = router_factory(
            {"http://w1:8000": 0, "http://w2:8000": 0, "http://w3:8000": 0},
            routing_algorithm="round-robin",
        )
        results = [r._select_worker_by_routing() for _ in range(6)]
        workers = list(r.worker_request_counts.keys())
        expected = [workers[i % 3] for i in range(6)]
        assert results == expected
        for url in workers:
            assert r.worker_request_counts[url] == 2

    def test_skips_dead_workers(self, router_factory):
        r = router_factory(
            {"http://w1:8000": 0, "http://w2:8000": 0, "http://w3:8000": 0},
            dead={"http://w2:8000"},
            routing_algorithm="round-robin",
        )
        results = [r._select_worker_by_routing() for _ in range(4)]
        assert "http://w2:8000" not in results
        assert all(url in ("http://w1:8000", "http://w3:8000") for url in results)


class TestRandomRouting:
    def test_covers_all_workers(self, router_factory):
        r = router_factory(
            {"http://w1:8000": 0, "http://w2:8000": 0, "http://w3:8000": 0},
            routing_algorithm="random",
        )
        seen = set()
        for _ in range(30):
            # Reset counts so they do not grow unbounded
            for url in r.worker_request_counts:
                r.worker_request_counts[url] = 0
            seen.add(r._select_worker_by_routing())
        assert seen == {"http://w1:8000", "http://w2:8000", "http://w3:8000"}

    def test_excludes_dead_workers(self, router_factory):
        r = router_factory(
            {"http://w1:8000": 0, "http://w2:8000": 0, "http://w3:8000": 0},
            dead={"http://w2:8000"},
            routing_algorithm="random",
        )
        for _ in range(20):
            url = r._select_worker_by_routing()
            assert url != "http://w2:8000"
            r.worker_request_counts[url] -= 1  # reset increment


class TestRoutingEdgeCases:
    @pytest.mark.parametrize("algo", ["least-request", "round-robin", "random"])
    def test_no_workers_raises(self, router_factory, algo):
        r = router_factory({}, routing_algorithm=algo)
        with pytest.raises(RuntimeError, match="No workers registered"):
            r._select_worker_by_routing()

    @pytest.mark.parametrize("algo", ["least-request", "round-robin", "random"])
    def test_all_dead_raises(self, router_factory, algo):
        r = router_factory(
            {"http://w1:8000": 0, "http://w2:8000": 0},
            dead={"http://w1:8000", "http://w2:8000"},
            routing_algorithm=algo,
        )
        with pytest.raises(RuntimeError, match="No healthy workers"):
            r._select_worker_by_routing()


class TestCountManagement:
    """Test that _select_worker_by_routing / _finish_url correctly track active request counts."""

    @pytest.mark.parametrize("algorithm", ["least-request", "round-robin", "random"])
    def test_increment_and_finish(self, router_factory, algorithm):
        router = router_factory({"http://w1:8000": 0}, routing_algorithm=algorithm)
        url = router._select_worker_by_routing()
        assert router.worker_request_counts[url] == 1
        router._finish_url(url)
        assert router.worker_request_counts[url] == 0


# ── Worker registration ──────────────────────────────────────────────


class TestRegisterWorker:
    def test_registers_and_deduplicates(self, router_factory):
        r = router_factory()
        r.register_worker("http://w1:8000")
        r.register_worker("http://w1:8000")
        r.register_worker("http://w2:9000")
        assert len(r.worker_request_counts) == 2

    def test_registered_worker_is_routable(self, router_factory):
        r = router_factory()
        r.register_worker("http://w1:8000")
        assert r._select_worker_by_routing() == "http://w1:8000"

    def test_rejects_blocked_host(self, router_factory):
        r = router_factory()
        with pytest.raises(ValueError, match="blocked"):
            r.register_worker("http://169.254.169.254:80")
        assert len(r.worker_request_counts) == 0

    def test_dead_then_register_new_restores_routing(self, router_factory):
        r = router_factory(
            {"http://w1:8000": 0},
            dead={"http://w1:8000"},
        )
        with pytest.raises(RuntimeError):
            r._select_worker_by_routing()
        r.register_worker("http://w2:9000")
        assert r._select_worker_by_routing() == "http://w2:9000"


# ── Response building ────────────────────────────────────────────────


class TestBuildProxyResponse:
    def test_small_json(self, router_factory):
        r = router_factory()
        content = json.dumps({"key": "value"}).encode()
        resp = r._build_proxy_response(
            content, 200, {"content-type": "application/json"}
        )
        assert json.loads(resp.body) == {"key": "value"}
        assert resp.status_code == 200

    def test_large_json_returns_raw(self, router_factory):
        r = router_factory()
        big = json.dumps({"data": "x" * (300 * 1024)}).encode()
        resp = r._build_proxy_response(big, 200, {"content-type": "application/json"})
        assert resp.body == big

    def test_preserves_status_code(self, router_factory):
        r = router_factory()
        content = json.dumps({"error": "not found"}).encode()
        resp = r._build_proxy_response(
            content, 404, {"content-type": "application/json"}
        )
        assert resp.status_code == 404


# ── Static helpers ────────────────────────────────────────────────────


class TestSanitizeResponseHeaders:
    def test_removes_hop_by_hop_and_encoding(self):
        headers = {
            "content-type": "application/json",
            "connection": "keep-alive",
            "transfer-encoding": "chunked",
            "content-length": "1234",
            "content-encoding": "gzip",
            "x-custom": "value",
        }
        result = DiffusionRouter._sanitize_response_headers(headers)
        assert set(result.keys()) == {"content-type", "x-custom"}


class TestTryDecodeJson:
    def test_valid_json(self):
        assert DiffusionRouter._try_decode_json(b'{"a": 1}') == {"a": 1}

    def test_invalid_json_returns_raw(self):
        result = DiffusionRouter._try_decode_json(b"not json")
        assert "raw" in result


# ── Constructor ───────────────────────────────────────────────────────


class TestConstructor:
    def test_default_algorithm(self):
        args = Namespace(
            host="127.0.0.1", port=30080, max_connections=100, timeout=120.0
        )
        r = DiffusionRouter(args)
        try:
            assert r.routing_algorithm == "least-request"
        finally:
            asyncio.run(r.client.aclose())

    def test_none_timeout_defaults_to_120(self):
        args = Namespace(
            host="127.0.0.1",
            port=30080,
            max_connections=100,
            timeout=None,
            routing_algorithm="least-request",
        )
        r = DiffusionRouter(args)
        try:
            assert r.client.timeout.connect == 120.0
        finally:
            asyncio.run(r.client.aclose())

    def test_initial_state_empty(self):
        args = _make_args()
        r = DiffusionRouter(args)
        try:
            assert r.worker_request_counts == {}
            assert r.dead_workers == set()
        finally:
            asyncio.run(r.client.aclose())

    def test_all_routes_registered(self):
        args = _make_args()
        r = DiffusionRouter(args)
        try:
            routes = [route.path for route in r.app.routes]
            for path in [
                "/add_worker",
                "/list_workers",
                "/health",
                "/health_workers",
                "/generate",
                "/generate_video",
                "/update_weights_from_disk",
                "/{path:path}",
            ]:
                assert path in routes
        finally:
            asyncio.run(r.client.aclose())
