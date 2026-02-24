from argparse import Namespace
from urllib.parse import quote

from fastapi.responses import JSONResponse
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


def test_post_workers_normalizes_and_deduplicates():
    router = DiffusionRouter(make_router_args())

    async def fake_refresh(worker_url: str):
        router.worker_video_support[worker_url] = False

    router.refresh_worker_video_support = fake_refresh  # type: ignore[assignment]
    with TestClient(router.app) as client:
        first = client.post("/workers", params={"url": "http://LOCALHOST:10090/"})
        assert first.status_code == 200

        second = client.post("/workers", params={"url": "http://localhost:10090"})
        assert second.status_code == 200

        listed = client.get("/workers")
        assert listed.status_code == 200
        workers = listed.json()["workers"]
        assert len(workers) == 1
        assert workers[0]["url"] == "http://localhost:10090"
        assert workers[0]["worker_id"] == quote("http://localhost:10090", safe="")
        assert workers[0]["active_requests"] == 0
        assert workers[0]["is_dead"] is False
        assert workers[0]["consecutive_failures"] == 0
        assert workers[0]["video_support"] is False


def test_post_workers_rejects_blocked_metadata_host():
    router = DiffusionRouter(make_router_args())
    with TestClient(router.app) as client:
        response = client.post("/workers", params={"url": "http://169.254.169.254:80"})
        assert response.status_code == 400
        assert "blocked" in response.json()["error"]


def test_get_worker_by_id_success_and_not_found():
    router = DiffusionRouter(make_router_args())
    worker_url = "http://localhost:10090"
    worker_id = quote(worker_url, safe="")
    router.register_worker(worker_url)
    router.worker_video_support[worker_url] = True

    with TestClient(router.app) as client:
        found = client.get(f"/workers/{worker_id}")
        assert found.status_code == 200
        assert found.json()["worker"]["url"] == worker_url
        assert found.json()["worker"]["video_support"] is True

        missing = client.get(f"/workers/{quote('http://localhost:19999', safe='')}")
        assert missing.status_code == 404


def test_put_worker_updates_flags_and_refreshes_capability():
    router = DiffusionRouter(make_router_args())
    worker_url = "http://localhost:10090"
    worker_id = quote(worker_url, safe="")
    router.register_worker(worker_url)
    router.worker_video_support[worker_url] = None
    refresh_calls: list[str] = []

    async def fake_refresh(url: str):
        refresh_calls.append(url)
        router.worker_video_support[url] = True

    router.refresh_worker_video_support = fake_refresh  # type: ignore[assignment]
    with TestClient(router.app) as client:
        response = client.put(
            f"/workers/{worker_id}",
            json={"is_dead": True, "refresh_video_support": True},
        )
        assert response.status_code == 200
        assert refresh_calls == [worker_url]
        assert response.json()["worker"]["is_dead"] is True
        assert response.json()["worker"]["video_support"] is True

        revive = client.put(f"/workers/{worker_id}", json={"is_dead": False})
        assert revive.status_code == 200
        assert revive.json()["worker"]["is_dead"] is False


def test_put_worker_rejects_invalid_payload():
    router = DiffusionRouter(make_router_args())
    worker_url = "http://localhost:10090"
    worker_id = quote(worker_url, safe="")
    router.register_worker(worker_url)

    with TestClient(router.app) as client:
        unknown = client.put(f"/workers/{worker_id}", json={"unsupported": True})
        assert unknown.status_code == 400
        assert "Unsupported fields" in unknown.json()["error"]

        invalid_type = client.put(f"/workers/{worker_id}", json={"is_dead": "yes"})
        assert invalid_type.status_code == 400
        assert "must be a boolean" in invalid_type.json()["error"]

        missing_body = client.put(f"/workers/{worker_id}", json={})
        assert missing_body.status_code == 400
        assert "At least one field is required" in missing_body.json()["error"]


def test_delete_worker_removes_runtime_state():
    router = DiffusionRouter(make_router_args())
    worker_url = "http://localhost:10090"
    worker_id = quote(worker_url, safe="")
    router.register_worker(worker_url)
    router.worker_failure_counts[worker_url] = 2
    router.worker_video_support[worker_url] = True
    router.dead_workers.add(worker_url)

    with TestClient(router.app) as client:
        deleted = client.delete(f"/workers/{worker_id}")
        assert deleted.status_code == 200
        assert worker_url not in router.worker_request_counts
        assert worker_url not in router.worker_failure_counts
        assert worker_url not in router.worker_video_support
        assert worker_url not in router.dead_workers

        missing = client.delete(f"/workers/{worker_id}")
        assert missing.status_code == 404


def test_v1_images_generations_routes_to_image_path():
    router = DiffusionRouter(make_router_args())
    call_args: dict = {}

    async def fake_forward(request, path: str, worker_urls=None):
        del request
        call_args["path"] = path
        call_args["worker_urls"] = worker_urls
        return JSONResponse(status_code=200, content={"ok": True})

    router._forward_to_worker = fake_forward  # type: ignore[assignment]
    with TestClient(router.app) as client:
        response = client.post("/v1/images/generations", json={"prompt": "cat"})
        assert response.status_code == 200
        assert call_args["path"] == "v1/images/generations"
        assert call_args["worker_urls"] is None


def test_v1_videos_requires_video_capable_workers():
    router = DiffusionRouter(make_router_args())
    router.register_worker("http://localhost:10090")
    router.worker_video_support["http://localhost:10090"] = False

    with TestClient(router.app) as client:
        response = client.post("/v1/videos", json={"prompt": "river"})
        assert response.status_code == 400
        assert "No video-capable workers available" in response.json()["error"]


def test_v1_videos_routes_only_to_video_capable_workers_and_caches_video_id():
    router = DiffusionRouter(make_router_args())
    router.register_worker("http://localhost:10090")
    router.register_worker("http://localhost:10091")
    router.worker_video_support["http://localhost:10090"] = True
    router.worker_video_support["http://localhost:10091"] = False
    call_args: dict = {}

    async def fake_forward_selected_worker(request, path: str, worker_url: str):
        del request
        call_args["path"] = path
        call_args["worker_url"] = worker_url
        return JSONResponse(status_code=200, content={"id": "video_123"})

    def fake_select(worker_urls=None):
        call_args["worker_urls"] = worker_urls
        return "http://localhost:10090"

    router._select_worker_by_routing = fake_select  # type: ignore[assignment]
    router._forward_to_selected_worker = fake_forward_selected_worker  # type: ignore[assignment]
    with TestClient(router.app) as client:
        response = client.post("/v1/videos", json={"prompt": "river"})
        assert response.status_code == 200
        assert call_args["path"] == "v1/videos"
        assert call_args["worker_urls"] == ["http://localhost:10090"]
        assert call_args["worker_url"] == "http://localhost:10090"
        assert router.video_job_to_worker["video_123"] == "http://localhost:10090"


def test_get_v1_videos_by_query_video_id_routes_to_mapped_worker():
    router = DiffusionRouter(make_router_args())
    router.register_worker("http://localhost:10090")
    router.video_job_to_worker["video_123"] = "http://localhost:10090"
    call_args: dict = {}

    async def fake_forward_registered_worker(request, path: str, worker_url: str):
        del request
        call_args["path"] = path
        call_args["worker_url"] = worker_url
        return JSONResponse(
            status_code=200, content={"id": "video_123", "status": "done"}
        )

    router._forward_to_registered_worker = fake_forward_registered_worker  # type: ignore[assignment]
    with TestClient(router.app) as client:
        response = client.get("/v1/videos", params={"video_id": "video_123"})
        assert response.status_code == 200
        assert call_args["path"] == "v1/videos"
        assert call_args["worker_url"] == "http://localhost:10090"


def test_get_v1_videos_unknown_query_video_id_returns_404():
    router = DiffusionRouter(make_router_args())
    with TestClient(router.app) as client:
        response = client.get("/v1/videos", params={"video_id": "missing"})
        assert response.status_code == 404
        assert "Unknown video_id" in response.json()["error"]


def test_get_v1_video_detail_and_content_route_to_mapped_worker():
    router = DiffusionRouter(make_router_args())
    router.register_worker("http://localhost:10090")
    router.video_job_to_worker["video_123"] = "http://localhost:10090"
    paths: list[str] = []

    async def fake_forward_registered_worker(request, path: str, worker_url: str):
        del request, worker_url
        paths.append(path)
        return JSONResponse(status_code=200, content={"ok": True})

    router._forward_to_registered_worker = fake_forward_registered_worker  # type: ignore[assignment]
    with TestClient(router.app) as client:
        detail = client.get("/v1/videos/video_123")
        content = client.get("/v1/videos/video_123/content")
        assert detail.status_code == 200
        assert content.status_code == 200
        assert paths == ["v1/videos/video_123", "v1/videos/video_123/content"]


def test_get_v1_models_aggregates_and_deduplicates():
    router = DiffusionRouter(make_router_args())
    router.register_worker("http://localhost:10090")
    router.register_worker("http://localhost:10091")

    class FakeModelResponse:
        def __init__(self, status_code: int, payload: dict):
            self.status_code = status_code
            self._payload = payload

        def json(self) -> dict:
            return self._payload

    async def fake_get(url: str, headers: dict):
        del headers
        if url == "http://localhost:10090/v1/models":
            return FakeModelResponse(
                200,
                {"data": [{"id": "model-a"}, {"id": "model-b"}]},
            )
        if url == "http://localhost:10091/v1/models":
            return FakeModelResponse(
                200,
                {"data": [{"id": "model-b"}, {"id": "model-c"}]},
            )
        return FakeModelResponse(500, {"error": "unexpected"})

    router.client.get = fake_get  # type: ignore[assignment]
    with TestClient(router.app) as client:
        response = client.get("/v1/models")
        assert response.status_code == 200
        model_ids = [item["id"] for item in response.json()["data"]]
        assert model_ids == ["model-a", "model-b", "model-c"]


def test_get_v1_models_returns_502_when_all_workers_fail():
    router = DiffusionRouter(make_router_args())
    router.register_worker("http://localhost:10090")

    class FakeModelResponse:
        def __init__(self, status_code: int):
            self.status_code = status_code

        def json(self) -> dict:
            return {"error": "failed"}

    async def fake_get(url: str, headers: dict):
        del url, headers
        return FakeModelResponse(500)

    router.client.get = fake_get  # type: ignore[assignment]
    with TestClient(router.app) as client:
        response = client.get("/v1/models")
        assert response.status_code == 502
        assert "Failed to fetch models from workers" in response.json()["error"]


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


def test_update_weights_from_disk_returns_503_without_healthy_workers():
    router = DiffusionRouter(make_router_args())
    router.register_worker("http://localhost:10090")
    router.dead_workers.add("http://localhost:10090")

    with TestClient(router.app) as client:
        response = client.post("/update_weights_from_disk", json={"model_path": "abc"})
        assert response.status_code == 503
        assert "No healthy workers available" in response.json()["error"]
