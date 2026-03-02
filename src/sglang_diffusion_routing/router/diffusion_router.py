# This module is derived from radixark/miles#544.
# See README.md for full acknowledgment.

import asyncio
import ipaddress
import json
import logging
import random
from contextlib import asynccontextmanager
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

logger = logging.getLogger(__name__)

_METADATA_HOSTS = {"169.254.169.254", "metadata.google.internal"}
_IMAGE_TASK_TYPES = {"T2I", "I2I", "TI2I"}


class DiffusionRouter:

    def __init__(self, args, verbose: bool = False):
        """Initialize the router for load-balancing sglang-diffusion workers."""
        self.args = args
        self.verbose = verbose

        self.app = FastAPI(lifespan=self._lifespan)

        # URL -> active request count
        self.worker_request_counts: dict[str, int] = {}
        # URL -> consecutive health check failures
        self.worker_failure_counts: dict[str, int] = {}
        # URL -> whether worker supports video generation
        # True: supports, False: image-only, None: unknown/unprobed
        self.worker_video_support: dict[str, bool | None] = {}
        # quarantined workers excluded from routing
        self.dead_workers: set[str] = set()
        # record workers in sleeping status
        self.sleeping_workers: set[str] = set()
        # video_id -> worker URL mapping for stable query routing
        self.video_job_to_worker: dict[str, str] = {}
        self._health_task: asyncio.Task | None = None

        self.routing_algorithm = getattr(args, "routing_algorithm", "least-request")
        self._rr_index = 0

        max_connections = getattr(args, "max_connections", 100)
        timeout = getattr(args, "timeout", 120.0)
        if timeout is None:
            timeout = 120.0

        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_connections),
            timeout=httpx.Timeout(timeout),
        )

        self._setup_routes()

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        await self._start_background_health_check()
        try:
            yield
        finally:
            await self._shutdown()

    def _setup_routes(self) -> None:
        self.app.post("/workers")(self.create_worker)
        self.app.get("/workers")(self.get_workers)
        # worker_id is a URL-encoded worker URL; decoded value can contain "/".
        # Use :path converter so encoded slashes still resolve to this route.
        self.app.get("/workers/{worker_id:path}")(self.get_worker)
        self.app.put("/workers/{worker_id:path}")(self.update_worker)
        self.app.delete("/workers/{worker_id:path}")(self.delete_worker)
        self.app.post("/v1/images/generations")(self.generate)
        self.app.post("/v1/videos")(self.generate_video)
        self.app.get("/v1/videos")(self.list_or_poll_videos)
        self.app.get("/v1/videos/{video_id}")(self.get_video_job)
        self.app.get("/v1/videos/{video_id}/content")(self.get_video_content)
        self.app.get("/v1/models")(self.get_models)
        self.app.get("/health")(self.health)
        self.app.post("/update_weights_from_disk")(self.update_weights_from_disk)
        self.app.post("/release_memory_occupation")(self.release_memory_occupation)
        self.app.post("/resume_memory_occupation")(self.resume_memory_occupation)
        self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])(
            self.proxy
        )

    async def _start_background_health_check(self) -> None:
        if self._health_task is None or self._health_task.done():
            self._health_task = asyncio.create_task(self._health_check_loop())

    async def _shutdown(self) -> None:
        if self._health_task is not None:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        await self.client.aclose()

    async def _check_worker_health(self, url: str) -> tuple[str, bool]:
        try:
            response = await self.client.get(f"{url}/health", timeout=5.0)
            if response.status_code == 200:
                return url, True
            logger.debug(
                "[diffusion-router] Worker %s unhealthy (status %s)",
                url,
                response.status_code,
            )
        except Exception as exc:
            logger.debug(
                "[diffusion-router] Worker %s health check failed: %s", url, exc
            )
        return url, False

    async def _health_check_loop(self) -> None:
        """Background loop to monitor worker health and quarantine failing workers."""
        interval = getattr(self.args, "health_check_interval", 10)
        threshold = getattr(self.args, "health_check_failure_threshold", 3)

        while True:
            try:
                await asyncio.sleep(interval)

                urls = [
                    u for u in self.worker_request_counts if u not in self.dead_workers
                ]
                if not urls:
                    continue

                results = await asyncio.gather(
                    *(self._check_worker_health(url) for url in urls)
                )
                for url, is_healthy in results:
                    if not is_healthy:
                        failures = self.worker_failure_counts.get(url, 0) + 1
                        self.worker_failure_counts[url] = failures
                        if failures >= threshold:
                            logger.warning(
                                "[diffusion-router] Worker %s failed %s consecutive checks. Marking DEAD.",
                                url,
                                threshold,
                            )
                            self.dead_workers.add(url)
                    else:
                        self.worker_failure_counts[url] = 0

                healthy = len(self.worker_request_counts) - len(self.dead_workers)
                logger.debug(
                    "[diffusion-router] Health check complete. %s workers healthy.",
                    healthy,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    "[diffusion-router] Unexpected error in health check loop: %s",
                    exc,
                    exc_info=True,
                )
                await asyncio.sleep(5)

    def _select_worker_by_routing(self, worker_urls: list[str] | None = None) -> str:
        """Select a worker URL based on routing algorithm and optional candidates.

        Args:
            worker_urls: Optional list of worker URLs to consider. If provided,
            only these workers will be considered for selection. If not provided,
            all registered workers will be considered.
        """
        if not self.worker_request_counts:
            raise RuntimeError("No workers registered in the pool")

        valid_workers = [
            w
            for w in self.worker_request_counts
            if w not in self.dead_workers and w not in self.sleeping_workers
        ]
        if worker_urls is not None:
            allowed = {w for w in worker_urls if w in self.worker_request_counts}
            valid_workers = [w for w in valid_workers if w in allowed]
        if not valid_workers:
            raise RuntimeError("No healthy workers available in the pool")

        if self.routing_algorithm == "round-robin":
            url = valid_workers[self._rr_index % len(valid_workers)]
            self._rr_index = (self._rr_index + 1) % len(valid_workers)
        elif self.routing_algorithm == "random":
            url = random.choice(valid_workers)
        else:
            url = min(valid_workers, key=self.worker_request_counts.get)

        self.worker_request_counts[url] += 1
        return url

    def _finish_url(self, url: str) -> None:
        """Mark the request to the given URL as finished."""
        if url not in self.worker_request_counts:
            logger.error("[diffusion-router] URL %s not recognized in _finish_url", url)
            return

        next_count = self.worker_request_counts[url] - 1
        if next_count < 0:
            logger.error(
                "[diffusion-router] URL %s count went negative; clamping to zero", url
            )
            next_count = 0
        self.worker_request_counts[url] = next_count

    def _build_proxy_response(
        self, content: bytes, status_code: int, headers: dict
    ) -> Response:
        """
        Build an HTTP response from proxied bytes.

        If the payload is small and valid JSON, return JSONResponse.
        Otherwise, return raw bytes to avoid expensive JSON re-encoding.
        """
        content_type = headers.get("content-type", "")
        max_json_reencode_bytes = 256 * 1024
        if len(content) <= max_json_reencode_bytes:
            try:
                data = json.loads(content)
                return JSONResponse(
                    content=data, status_code=status_code, headers=headers
                )
            except Exception:
                pass

        return Response(
            content=content,
            status_code=status_code,
            headers=headers,
            media_type=content_type,
        )

    async def _forward_to_worker(
        self, request: Request, path: str, worker_urls: list[str] | None = None
    ) -> Response:
        """Forward request to a selected worker (optionally from candidate URLs)."""
        try:
            worker_url = self._select_worker_by_routing(worker_urls=worker_urls)
        except RuntimeError as exc:
            return JSONResponse(status_code=503, content={"error": str(exc)})
        return await self._forward_to_selected_worker(request, path, worker_url)

    async def _forward_to_selected_worker(
        self, request: Request, path: str, worker_url: str
    ) -> Response:
        """Forward request to a known selected worker.

        NOTE: caller must ensure request count has already been incremented.
        """
        try:
            query = request.url.query
            url = (
                f"{worker_url}/{path}" if not query else f"{worker_url}/{path}?{query}"
            )
            body = await request.body()
            headers = dict(request.headers)
            if body is not None:
                headers = {
                    k: v
                    for k, v in headers.items()
                    if k.lower() not in ("content-length", "transfer-encoding")
                }

            response = await self.client.request(
                request.method, url, content=body, headers=headers
            )
            content = await response.aread()
            resp_headers = self._sanitize_response_headers(response.headers)
            return self._build_proxy_response(
                content, response.status_code, resp_headers
            )
        except Exception as exc:
            logger.error(
                "[diffusion-router] Failed to forward request to %s: %s",
                worker_url,
                exc,
            )
            return JSONResponse(
                status_code=502, content={"error": f"Worker request failed: {exc}"}
            )
        finally:
            self._finish_url(worker_url)

    async def _forward_to_registered_worker(
        self, request: Request, path: str, worker_url: str
    ) -> Response:
        """Forward request to a specific registered and healthy worker."""
        if worker_url not in self.worker_request_counts:
            return JSONResponse(
                status_code=404, content={"error": "Mapped worker not found"}
            )
        if worker_url in self.dead_workers:
            return JSONResponse(
                status_code=503, content={"error": "Mapped worker is unavailable"}
            )
        if worker_url in self.sleeping_workers:
            return JSONResponse(
                status_code=503, content={"error": "Mapped worker is sleeping"}
            )
        self.worker_request_counts[worker_url] += 1
        return await self._forward_to_selected_worker(request, path, worker_url)

    @staticmethod
    def _extract_video_id(payload: dict | list | None) -> str | None:
        """Extract a stable video/job ID from common response shapes."""
        if isinstance(payload, dict):
            for key in ("video_id", "id", "job_id"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            nested = payload.get("data")
            return DiffusionRouter._extract_video_id(nested)  # type: ignore[arg-type]
        if isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, dict):
                return DiffusionRouter._extract_video_id(first)
        return None

    def _cache_video_job_mapping(self, response: Response, worker_url: str) -> None:
        """Cache video_id->worker mapping when create-video succeeds."""
        if response.status_code >= 400:
            return

        body = getattr(response, "body", b"")
        if not body:
            return
        if isinstance(body, str):
            body = body.encode("utf-8")
        if not isinstance(body, (bytes, bytearray)):
            return

        try:
            payload = json.loads(body)
        except Exception:
            return

        video_id = self._extract_video_id(payload)
        if video_id:
            self.video_job_to_worker[video_id] = worker_url

    async def _probe_worker_video_support(self, worker_url: str) -> bool | None:
        """Probe /v1/models and infer if this worker supports video generation."""
        try:
            response = await self.client.get(f"{worker_url}/v1/models", timeout=5.0)
            if response.status_code == 200:
                payload = response.json()
                data = payload.get("data")
                task_type = (
                    data[0].get("task_type")
                    if isinstance(data, list) and data
                    else None
                )
                if isinstance(task_type, str):
                    return task_type.upper() not in _IMAGE_TASK_TYPES
        except (httpx.RequestError, json.JSONDecodeError) as exc:
            logger.debug(
                "[diffusion-router] video support probe failed: worker=%s error=%s",
                worker_url,
                exc,
            )
            return None

    async def refresh_worker_video_support(self, worker_url: str) -> None:
        """Refresh cached video capability for a single worker."""
        self.worker_video_support[worker_url] = await self._probe_worker_video_support(
            worker_url
        )

    async def _broadcast_to_workers(
        self, path: str, body: bytes, headers: dict
    ) -> list[dict]:
        """
        Broadcast request to eligible workers.

        Rules:
        - For resume_memory_occupation (wake): target sleeping workers (even if currently marked dead).
        - For all other requests: target only active workers (exclude dead AND sleeping).
        """
        if path == "resume_memory_occupation":
            # Wake is a recovery point: allow waking workers that were marked dead during sleep.
            urls = [u for u in self.sleeping_workers if u in self.worker_request_counts]
        else:
            urls = [
                u
                for u in self.worker_request_counts
                if u not in self.dead_workers and u not in self.sleeping_workers
            ]

        if not urls:
            return []

        async def _send(worker_url: str) -> dict:
            try:
                response = await self.client.post(
                    f"{worker_url}/{path}", content=body, headers=headers
                )
                content = await response.aread()
                return {
                    "worker_url": worker_url,
                    "status_code": response.status_code,
                    "body": self._try_decode_json(content),
                }
            except Exception as exc:
                return {
                    "worker_url": worker_url,
                    "status_code": 502,
                    "body": {"error": str(exc)},
                }

        return await asyncio.gather(*(_send(u) for u in urls))

    @staticmethod
    def _try_decode_json(content: bytes):
        try:
            return json.loads(content)
        except Exception:
            return {"raw": content.decode("utf-8", errors="replace")}

    @staticmethod
    def _sanitize_response_headers(headers) -> dict:
        """Remove hop-by-hop and encoding headers that no longer match buffered content."""
        hop_by_hop = {
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailers",
            "transfer-encoding",
            "upgrade",
        }
        dropped = {"content-length", "content-encoding"}
        return {
            k: v for k, v in headers.items() if k.lower() not in hop_by_hop | dropped
        }

    @staticmethod
    def normalize_worker_url(url: str) -> str:
        if not isinstance(url, str):
            raise ValueError("worker_url must be a string")

        raw = url.strip()
        if not raw:
            raise ValueError("worker_url cannot be empty")

        parsed = urlsplit(raw)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("worker_url must start with http:// or https://")
        if not parsed.netloc:
            raise ValueError("worker_url must include host and port")
        if parsed.username or parsed.password:
            raise ValueError("worker_url must not include user credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("worker_url must not include query or fragment")
        if parsed.path not in ("", "/"):
            raise ValueError("worker_url path is not allowed")

        hostname = (parsed.hostname or "").lower()
        if hostname in _METADATA_HOSTS:
            raise ValueError("worker_url host is blocked")
        parsed_ip = None
        try:
            parsed_ip = ipaddress.ip_address(hostname)
        except ValueError:
            # Non-IP hostname
            pass
        if parsed_ip is not None and parsed_ip.is_link_local:
            raise ValueError("link-local worker_url hosts are blocked")

        if parsed.port is None:
            normalized_netloc = hostname
        elif ":" in hostname and not hostname.startswith("["):
            normalized_netloc = f"[{hostname}]:{parsed.port}"
        else:
            normalized_netloc = f"{hostname}:{parsed.port}"

        normalized = urlunsplit((parsed.scheme, normalized_netloc, "", "", ""))
        return normalized.rstrip("/")

    @staticmethod
    def encode_worker_id(worker_url: str) -> str:
        """Encode a normalized worker URL into a path-safe worker_id."""
        return quote(worker_url, safe="")

    @classmethod
    def decode_worker_id(cls, worker_id: str) -> str:
        """Decode and normalize worker_id back into worker URL."""
        decoded = unquote(worker_id).strip()
        if not decoded:
            raise ValueError("worker_id cannot be empty")
        return cls.normalize_worker_url(decoded)

    @staticmethod
    def build_worker_display_id(worker_url: str) -> str:
        """Build a human-readable worker identifier from URL."""
        parsed = urlsplit(worker_url)
        if parsed.netloc:
            return parsed.netloc
        return worker_url

    def _build_worker_payload(self, worker_url: str) -> dict:
        return {
            "worker_id": self.encode_worker_id(worker_url),
            "display_id": self.build_worker_display_id(worker_url),
            "url": worker_url,
            "active_requests": self.worker_request_counts.get(worker_url, 0),
            "is_dead": worker_url in self.dead_workers,
            "is_sleeping": worker_url in self.sleeping_workers,
            "consecutive_failures": self.worker_failure_counts.get(worker_url, 0),
            "video_support": self.worker_video_support.get(worker_url),
        }

    async def _extract_worker_url(
        self, request: Request
    ) -> tuple[str | None, Response | None]:
        worker_url = request.query_params.get("url") or request.query_params.get(
            "worker_url"
        )
        if worker_url:
            return worker_url, None

        body = await request.body()
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            return None, JSONResponse(
                status_code=400, content={"error": "Invalid JSON body"}
            )

        if not isinstance(payload, dict):
            return None, JSONResponse(
                status_code=400,
                content={"error": "Request body must be a JSON object"},
            )
        return payload.get("url") or payload.get("worker_url"), None

    async def generate(self, request: Request):
        """Route image generation requests to worker /v1/images/generations."""
        return await self._forward_to_worker(request, "v1/images/generations")

    async def generate_video(self, request: Request):
        """Route video generation requests to worker /v1/videos."""
        candidate_workers = self._video_capable_workers()
        if not candidate_workers:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "No video-capable workers available in current worker pool.",
                },
            )
        try:
            worker_url = self._select_worker_by_routing(worker_urls=candidate_workers)
        except RuntimeError as exc:
            return JSONResponse(status_code=503, content={"error": str(exc)})

        response = await self._forward_to_selected_worker(
            request, "v1/videos", worker_url
        )
        self._cache_video_job_mapping(response, worker_url)
        return response

    def _video_capable_workers(self) -> list[str]:
        return [
            worker_url
            for worker_url, support in self.worker_video_support.items()
            if support
            and worker_url in self.worker_request_counts
            and worker_url not in self.dead_workers
            and worker_url not in self.sleeping_workers
        ]

    @staticmethod
    def _extract_video_id_from_query(request: Request) -> str | None:
        for key in ("video_id", "id", "job_id"):
            value = request.query_params.get(key)
            if value and value.strip():
                return value.strip()
        return None

    async def list_or_poll_videos(self, request: Request):
        """List jobs or poll a known video job by query parameter."""
        video_id = self._extract_video_id_from_query(request)
        if video_id:
            worker_url = self.video_job_to_worker.get(video_id)
            if not worker_url:
                return JSONResponse(
                    status_code=404, content={"error": "Unknown video_id"}
                )
            return await self._forward_to_registered_worker(
                request, "v1/videos", worker_url
            )

        candidate_workers = self._video_capable_workers()
        if candidate_workers:
            return await self._forward_to_worker(
                request, "v1/videos", worker_urls=candidate_workers
            )
        return await self._forward_to_worker(request, "v1/videos")

    async def get_video_job(self, request: Request, video_id: str):
        """Get status/details for a known video job."""
        worker_url = self.video_job_to_worker.get(video_id)
        if not worker_url:
            return JSONResponse(status_code=404, content={"error": "Unknown video_id"})
        return await self._forward_to_registered_worker(
            request, f"v1/videos/{video_id}", worker_url
        )

    async def get_video_content(self, request: Request, video_id: str):
        """Download content for a known video job."""
        worker_url = self.video_job_to_worker.get(video_id)
        if not worker_url:
            return JSONResponse(status_code=404, content={"error": "Unknown video_id"})
        return await self._forward_to_registered_worker(
            request, f"v1/videos/{video_id}/content", worker_url
        )

    async def get_models(self, request: Request):
        """Aggregate /v1/models responses from healthy workers and de-duplicate."""
        worker_urls = [
            url
            for url in self.worker_request_counts.keys()
            if url not in self.dead_workers and url not in self.sleeping_workers
        ]
        if not worker_urls:
            return JSONResponse(
                status_code=503,
                content={"error": "No healthy workers available in the pool"},
            )

        query = request.url.query
        request_headers = {
            k: v
            for k, v in dict(request.headers).items()
            if k.lower() not in ("content-length", "transfer-encoding")
        }

        async def _fetch(worker_url: str) -> tuple[str, list[dict], str | None]:
            self.worker_request_counts[worker_url] += 1
            try:
                url = (
                    f"{worker_url}/v1/models"
                    if not query
                    else f"{worker_url}/v1/models?{query}"
                )
                response = await self.client.get(url, headers=request_headers)
                if response.status_code != 200:
                    return worker_url, [], f"status={response.status_code}"
                payload = response.json()
                data = payload.get("data")
                if not isinstance(data, list):
                    return worker_url, [], "invalid models payload"
                return (
                    worker_url,
                    [item for item in data if isinstance(item, dict)],
                    None,
                )
            except Exception as exc:
                return worker_url, [], str(exc)
            finally:
                self._finish_url(worker_url)

        results = await asyncio.gather(
            *(_fetch(worker_url) for worker_url in worker_urls)
        )
        merged: list[dict] = []
        seen_keys: set[str] = set()
        errors: dict[str, str] = {}

        for worker_url, models, error in results:
            if error:
                errors[worker_url] = error
                continue
            for model in models:
                model_id = model.get("id") or model.get("model")
                dedupe_key = (
                    model_id
                    if isinstance(model_id, str) and model_id
                    else json.dumps(model, sort_keys=True)
                )
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                merged.append(model)

        if not merged:
            return JSONResponse(
                status_code=502,
                content={
                    "error": "Failed to fetch models from workers",
                    "details": errors,
                },
            )
        return JSONResponse(content={"object": "list", "data": merged})

    async def health(self, request: Request):
        """Aggregated health status: healthy if at least one worker is alive."""
        total = len(self.worker_request_counts)
        dead = len(self.dead_workers)
        sleeping = len(self.sleeping_workers)
        healthy = total - dead - sleeping
        status = "healthy" if healthy > 0 else "unhealthy"
        code = 200 if healthy > 0 else 503
        return JSONResponse(
            status_code=code,
            content={
                "status": status,
                "healthy_workers": healthy,
                "total_workers": total,
                "dead_workers": dead,
                "sleeping_workers": sleeping,
            },
        )

    async def update_weights_from_disk(self, request: Request):
        """Broadcast weight reload to all healthy workers."""
        healthy_workers = [
            url for url in self.worker_request_counts if url not in self.dead_workers
        ]
        if not healthy_workers:
            return JSONResponse(
                status_code=503,
                content={"error": "No healthy workers available in the pool"},
            )

        body = await request.body()
        headers = dict(request.headers)
        results = await self._broadcast_to_workers(
            "update_weights_from_disk", body, headers
        )
        return JSONResponse(content={"results": results})

    async def _broadcast_to_pool(self, request: Request, path: str) -> tuple[int, dict]:
        if not self.worker_request_counts:
            return 503, {"error": "No workers registered in the pool"}

        body = await request.body()
        headers = dict(request.headers)

        results = await self._broadcast_to_workers(path, body, headers)
        if not results:
            return 503, {"error": "No eligible workers available in the pool"}

        return 200, {"results": results}

    async def release_memory_occupation(self, request: Request):
        status, payload = await self._broadcast_to_pool(
            request, "release_memory_occupation"
        )
        if status != 200:
            return JSONResponse(status_code=status, content=payload)

        for item in payload["results"]:
            if item.get("status_code") == 200:
                self.sleeping_workers.add(item["worker_url"])

        return JSONResponse(content=payload)

    async def resume_memory_occupation(self, request: Request):
        status, payload = await self._broadcast_to_pool(
            request, "resume_memory_occupation"
        )
        if status != 200:
            return JSONResponse(status_code=status, content=payload)

        for item in payload["results"]:
            if item.get("status_code") == 200:
                url = item["worker_url"]
                self.sleeping_workers.discard(url)
                self.worker_failure_counts[url] = 0
                self.dead_workers.discard(url)  # wake success => recover

        return JSONResponse(content=payload)

    def register_worker(self, url: str) -> None:
        """Register a worker URL if not already known."""
        normalized_url = self.normalize_worker_url(url)
        if normalized_url not in self.worker_request_counts:
            self.worker_request_counts[normalized_url] = 0
            self.worker_failure_counts[normalized_url] = 0
            self.worker_video_support[normalized_url] = None
            if self.verbose:
                print(f"[diffusion-router] Added new worker: {normalized_url}")

    def deregister_worker(self, url: str) -> None:
        normalized_url = self.normalize_worker_url(url)
        self.worker_request_counts.pop(normalized_url, None)
        self.worker_failure_counts.pop(normalized_url, None)
        self.worker_video_support.pop(normalized_url, None)
        self.dead_workers.discard(normalized_url)
        self.sleeping_workers.discard(normalized_url)
        stale_video_ids = [
            video_id
            for video_id, mapped_worker in self.video_job_to_worker.items()
            if mapped_worker == normalized_url
        ]
        for video_id in stale_video_ids:
            self.video_job_to_worker.pop(video_id, None)

    async def create_worker(self, request: Request):
        """Register a new worker via /workers."""
        worker_url, error_response = await self._extract_worker_url(request)
        if error_response is not None:
            return error_response
        if not worker_url:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "worker_url is required (use query ?url=... or JSON body)"
                },
            )

        try:
            normalized_url = self.normalize_worker_url(worker_url)
            self.register_worker(normalized_url)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})

        await self.refresh_worker_video_support(normalized_url)
        return JSONResponse(
            content={
                "status": "success",
                "worker": self._build_worker_payload(normalized_url),
            }
        )

    async def get_workers(self, request: Request):
        """List all workers with health and load details."""
        workers = [
            self._build_worker_payload(worker_url)
            for worker_url in self.worker_request_counts
        ]
        return JSONResponse(content={"workers": workers})

    async def get_worker(self, request: Request, worker_id: str):
        """Get details for a single worker."""
        try:
            worker_url = self.decode_worker_id(worker_id)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        if worker_url not in self.worker_request_counts:
            return JSONResponse(status_code=404, content={"error": "Worker not found"})
        return JSONResponse(content={"worker": self._build_worker_payload(worker_url)})

    async def update_worker(self, request: Request, worker_id: str):
        """Update a worker's lightweight runtime state."""
        try:
            worker_url = self.decode_worker_id(worker_id)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        if worker_url not in self.worker_request_counts:
            return JSONResponse(status_code=404, content={"error": "Worker not found"})

        body = await request.body()
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})
        if not isinstance(payload, dict):
            return JSONResponse(
                status_code=400,
                content={"error": "Request body must be a JSON object"},
            )

        allowed_fields = {"is_dead", "refresh_video_support"}
        unknown_fields = sorted(set(payload) - allowed_fields)
        if unknown_fields:
            return JSONResponse(
                status_code=400,
                content={"error": f"Unsupported fields: {', '.join(unknown_fields)}"},
            )
        if not payload:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "At least one field is required: is_dead, refresh_video_support"
                },
            )

        if "is_dead" in payload and not isinstance(payload["is_dead"], bool):
            return JSONResponse(
                status_code=400, content={"error": "is_dead must be a boolean"}
            )
        if "refresh_video_support" in payload and not isinstance(
            payload["refresh_video_support"], bool
        ):
            return JSONResponse(
                status_code=400,
                content={"error": "refresh_video_support must be a boolean"},
            )

        if payload.get("is_dead") is True:
            self.dead_workers.add(worker_url)
        elif payload.get("is_dead") is False:
            self.dead_workers.discard(worker_url)

        if payload.get("refresh_video_support") is True:
            await self.refresh_worker_video_support(worker_url)

        return JSONResponse(
            content={
                "status": "success",
                "worker": self._build_worker_payload(worker_url),
            }
        )

    async def delete_worker(self, request: Request, worker_id: str):
        """Remove a worker from the pool."""
        try:
            worker_url = self.decode_worker_id(worker_id)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        if worker_url not in self.worker_request_counts:
            return JSONResponse(status_code=404, content={"error": "Worker not found"})
        self.deregister_worker(worker_url)
        return JSONResponse(content={"status": "success", "worker_id": worker_id})

    async def proxy(self, request: Request, path: str):
        """Catch-all: forward unmatched requests to a selected worker."""
        return await self._forward_to_worker(request, path)
