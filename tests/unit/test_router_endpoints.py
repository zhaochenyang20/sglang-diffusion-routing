from argparse import Namespace

from fastapi.testclient import TestClient

from sglang_diffusion_routing import DiffusionRouter


def make_router_args(**overrides) -> Namespace:
    defaults = dict(
        host="127.0.0.1",
        port=30080,
        max_connections=100,
        timeout=120.0,
        routing_algorithm="least-request",
        health_check_interval=3600,
        health_check_failure_threshold=3,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_add_worker_normalizes_and_deduplicates():
    router = DiffusionRouter(make_router_args())
    with TestClient(router.app) as client:
        first = client.post("/add_worker", params={"url": "http://LOCALHOST:10090/"})
        assert first.status_code == 200

        second = client.post("/add_worker", params={"url": "http://localhost:10090"})
        assert second.status_code == 200
        payload = second.json()
        assert payload["worker_urls"] == ["http://localhost:10090"]

        listed = client.get("/list_workers")
        assert listed.status_code == 200
        assert listed.json()["urls"] == ["http://localhost:10090"]


def test_add_worker_rejects_blocked_metadata_host():
    router = DiffusionRouter(make_router_args())
    with TestClient(router.app) as client:
        response = client.post(
            "/add_worker", params={"url": "http://169.254.169.254:80"}
        )
        assert response.status_code == 400
        assert "blocked" in response.json()["error"]


def test_update_weights_from_disk_returns_broadcast_results():
    router = DiffusionRouter(make_router_args())
    router.register_worker("http://localhost:10090")

    async def fake_broadcast(path: str, body: bytes, headers: dict):
        assert path == "update_weights_from_disk"
        assert body == b'{"model_path":"abc"}'
        assert headers.get("content-type", "").startswith("application/json")
        return [
            {
                "worker_url": "http://localhost:10090",
                "status_code": 200,
                "body": {"ok": True},
            }
        ]

    router._broadcast_to_workers = fake_broadcast  # type: ignore[assignment]
    with TestClient(router.app) as client:
        response = client.post("/update_weights_from_disk", json={"model_path": "abc"})
        assert response.status_code == 200
        assert response.json()["results"][0]["status_code"] == 200
