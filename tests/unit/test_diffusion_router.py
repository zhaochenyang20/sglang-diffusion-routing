import asyncio
import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from sglang_diffusion_routing import DiffusionRouter


def make_router_args(**overrides) -> Namespace:
    """Create a Namespace with default DiffusionRouter args, applying overrides."""
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
    """Factory fixture that creates routers and closes their clients at teardown."""
    created_routers: list[DiffusionRouter] = []

    def _create(
        workers: dict[str, int],
        dead: set[str] | None = None,
        **arg_overrides,
    ) -> DiffusionRouter:
        router = DiffusionRouter(make_router_args(**arg_overrides))
        router.worker_request_counts = dict(workers)
        router.worker_failure_counts = {url: 0 for url in workers}
        if dead:
            router.dead_workers = set(dead)
        created_routers.append(router)
        return router

    yield _create

    for router in created_routers:
        asyncio.run(router.client.aclose())


class TestLeastRequest:
    """Test the least-request (default) load-balancing algorithm."""

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
    """Test the round-robin load-balancing algorithm."""

    def test_cycles_workers(self, router_factory):
        router = router_factory(
            {"http://w1:8000": 0, "http://w2:8000": 0, "http://w3:8000": 0},
            routing_algorithm="round-robin",
        )
        results = [router._select_worker_by_routing() for _ in range(6)]
        workers = list(router.worker_request_counts.keys())
        expected = [workers[i % 3] for i in range(6)]
        assert results == expected
        for url in workers:
            assert router.worker_request_counts[url] == 2

    def test_excludes_dead_workers(self, router_factory):
        router = router_factory(
            {"http://w1:8000": 0, "http://w2:8000": 0, "http://w3:8000": 0},
            dead={"http://w2:8000"},
            routing_algorithm="round-robin",
        )
        results = [router._select_worker_by_routing() for _ in range(4)]
        assert "http://w2:8000" not in results
        assert all(url in ("http://w1:8000", "http://w3:8000") for url in results)


class TestRandom:
    """Test the random load-balancing algorithm."""

    def test_selects_from_valid_workers(self, router_factory):
        router = router_factory(
            {"http://w1:8000": 0, "http://w2:8000": 0, "http://w3:8000": 0},
            routing_algorithm="random",
        )
        seen = set()
        for _ in range(30):
            # Reset counts so they do not grow unbounded
            for url in router.worker_request_counts:
                router.worker_request_counts[url] = 0
            seen.add(router._select_worker_by_routing())
        assert seen == {"http://w1:8000", "http://w2:8000", "http://w3:8000"}

    def test_excludes_dead_workers(self, router_factory):
        router = router_factory(
            {"http://w1:8000": 0, "http://w2:8000": 0, "http://w3:8000": 0},
            dead={"http://w2:8000"},
            routing_algorithm="random",
        )
        for _ in range(20):
            url = router._select_worker_by_routing()
            assert url != "http://w2:8000"
            router.worker_request_counts[url] -= 1  # reset increment


class TestErrorCases:
    """Test error handling across all routing algorithms."""

    @pytest.mark.parametrize("algorithm", ["least-request", "round-robin", "random"])
    def test_raises_when_no_workers(self, router_factory, algorithm):
        router = router_factory({}, routing_algorithm=algorithm)
        with pytest.raises(RuntimeError, match="No workers registered"):
            router._select_worker_by_routing()

    @pytest.mark.parametrize("algorithm", ["least-request", "round-robin", "random"])
    def test_raises_when_all_dead(self, router_factory, algorithm):
        router = router_factory(
            {"http://w1:8000": 0, "http://w2:8000": 0},
            dead={"http://w1:8000", "http://w2:8000"},
            routing_algorithm=algorithm,
        )
        with pytest.raises(RuntimeError, match="No healthy workers"):
            router._select_worker_by_routing()


class TestCountManagement:
    """Test that _select_worker_by_routing / _finish_url correctly track active request counts."""

    @pytest.mark.parametrize("algorithm", ["least-request", "round-robin", "random"])
    def test_increment_and_finish(self, router_factory, algorithm):
        router = router_factory({"http://w1:8000": 0}, routing_algorithm=algorithm)
        url = router._select_worker_by_routing()
        assert router.worker_request_counts[url] == 1
        router._finish_url(url)
        assert router.worker_request_counts[url] == 0


class TestDefaults:
    """Test default routing algorithm when the attribute is absent."""

    def test_default_algorithm_is_least_request(self):
        args = Namespace(
            host="127.0.0.1", port=30080, max_connections=100, timeout=120.0
        )
        # args has no routing_algorithm attribute
        router = DiffusionRouter(args)
        try:
            assert router.routing_algorithm == "least-request"
        finally:
            asyncio.run(router.client.aclose())


class TestRegressions:
    def test_forward_body_error_does_not_leak_request_count(self, router_factory):
        router = router_factory({"http://w1:8000": 0})

        class BrokenRequest:
            method = "POST"
            headers = {"content-type": "application/json"}
            url = SimpleNamespace(query="")

            async def body(self):
                raise RuntimeError("body read failed")

        response = asyncio.run(
            router._forward_to_worker(BrokenRequest(), "v1/images/generations")
        )
        assert response.status_code == 502
        assert router.worker_request_counts["http://w1:8000"] == 0

    def test_register_worker_normalizes_duplicate_urls(self, router_factory):
        router = router_factory({})
        router.register_worker("http://LOCALHOST:10090/")
        router.register_worker("http://localhost:10090")
        assert list(router.worker_request_counts.keys()) == ["http://localhost:10090"]

    def test_register_worker_rejects_metadata_host(self, router_factory):
        router = router_factory({})
        with pytest.raises(ValueError, match="host is blocked"):
            router.register_worker("http://169.254.169.254:80")

    def test_broadcast_to_workers_collects_per_worker_results(self, router_factory):
        router = router_factory({"http://w1:8000": 0, "http://w2:8000": 0})

        class FakeResponse:
            def __init__(self, status_code: int, body: dict):
                self.status_code = status_code
                self._body = body

            async def aread(self) -> bytes:
                return json.dumps(self._body).encode("utf-8")

        responses = {
            "http://w1:8000/update_weights_from_disk": FakeResponse(200, {"ok": True}),
            "http://w2:8000/update_weights_from_disk": FakeResponse(
                500, {"error": "bad worker"}
            ),
        }

        async def fake_post(url, content, headers):
            del content, headers
            return responses[url]

        router.client.post = fake_post  # type: ignore[assignment]
        result = asyncio.run(
            router._broadcast_to_workers("update_weights_from_disk", b"{}", {})
        )
        assert len(result) == 2
        assert {item["worker_url"] for item in result} == {
            "http://w1:8000",
            "http://w2:8000",
        }
