# This module is derived from radixark/miles#544.
# See README.md for full acknowledgment.

import asyncio
import ipaddress
import json
import logging
import random
from urllib.parse import urlsplit, urlunsplit

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

        self.app = FastAPI()
        self.app.add_event_handler("startup", self._start_background_health_check)
        self.app.add_event_handler("shutdown", self._shutdown)

        # URL -> active request count
        self.worker_request_counts: dict[str, int] = {}
        # URL -> consecutive health check failures
        self.worker_failure_counts: dict[str, int] = {}
        # URL -> whether worker supports video generation
        # True: supports, False: image-only, None: unknown/unprobed
        self.worker_video_support: dict[str, bool | None] = {}
        # quarantined workers excluded from routing
        self.dead_workers: set[str] = set()
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

    def _setup_routes(self) -> None:
        self.app.post("/add_worker")(self.add_worker)
        self.app.get("/list_workers")(self.list_workers)
        self.app.get("/health")(self.health)
        self.app.get("/health_workers")(self.health_workers)
        self.app.post("/generate")(self.generate)
        self.app.post("/generate_video")(self.generate_video)
        self.app.post("/update_weights_from_disk")(self.update_weights_from_disk)
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
            w for w in self.worker_request_counts if w not in self.dead_workers
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
        except (httpx.RequestError, json.JSONDecodeError):
            return None

    async def refresh_worker_video_support(self, worker_url: str) -> None:
        """Refresh cached video capability for a single worker."""
        self.worker_video_support[worker_url] = await self._probe_worker_video_support(
            worker_url
        )

    async def _broadcast_to_workers(
        self, path: str, body: bytes, headers: dict
    ) -> list[dict]:
        """Send a request to all healthy workers and collect results."""
        urls = [u for u in self.worker_request_counts if u not in self.dead_workers]
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

    async def generate(self, request: Request):
        """Route image generation to /v1/images/generations."""
        return await self._forward_to_worker(request, "v1/images/generations")

    async def generate_video(self, request: Request):
        """Route video generation to /v1/videos."""
        candidate_workers = [
            worker_url
            for worker_url, support in self.worker_video_support.items()
            if support
        ]

        if not candidate_workers:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "No video-capable workers available in current worker pool.",
                },
            )
        return await self._forward_to_worker(
            request, "v1/videos", worker_urls=candidate_workers
        )

    async def health(self, request: Request):
        """Aggregated health status: healthy if at least one worker is alive."""
        total = len(self.worker_request_counts)
        dead = len(self.dead_workers)
        healthy = total - dead
        status = "healthy" if healthy > 0 else "unhealthy"
        code = 200 if healthy > 0 else 503
        return JSONResponse(
            status_code=code,
            content={
                "status": status,
                "healthy_workers": healthy,
                "total_workers": total,
            },
        )

    async def health_workers(self, request: Request):
        """Per-worker health and load information."""
        workers = []
        for url, count in self.worker_request_counts.items():
            workers.append(
                {
                    "url": url,
                    "active_requests": count,
                    "is_dead": url in self.dead_workers,
                    "consecutive_failures": self.worker_failure_counts.get(url, 0),
                }
            )
        return JSONResponse(content={"workers": workers})

    async def update_weights_from_disk(self, request: Request):
        """Broadcast weight reload to all healthy workers."""
        body = await request.body()
        headers = dict(request.headers)
        results = await self._broadcast_to_workers(
            "update_weights_from_disk", body, headers
        )
        return JSONResponse(content={"results": results})

    def register_worker(self, url: str) -> None:
        """Register a worker URL if not already known."""
        normalized_url = self.normalize_worker_url(url)
        if normalized_url not in self.worker_request_counts:
            self.worker_request_counts[normalized_url] = 0
            self.worker_failure_counts[normalized_url] = 0
            self.worker_video_support[normalized_url] = None
            if self.verbose:
                print(f"[diffusion-router] Added new worker: {normalized_url}")

    async def add_worker(self, request: Request):
        """Register a new diffusion worker (query: ?url=... or JSON body)."""
        worker_url = request.query_params.get("url") or request.query_params.get(
            "worker_url"
        )
        if not worker_url:
            body = await request.body()
            try:
                payload = json.loads(body) if body else {}
            except json.JSONDecodeError:
                return JSONResponse(
                    status_code=400, content={"error": "Invalid JSON body"}
                )
            worker_url = payload.get("url") or payload.get("worker_url")

        if not worker_url:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "worker_url is required (use query ?url=... or JSON body)"
                },
            )

        try:
            self.register_worker(worker_url)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        await self.refresh_worker_video_support(worker_url)
        return {
            "status": "success",
            "worker_urls": list(self.worker_request_counts.keys()),
        }

    async def list_workers(self, request: Request):
        """List all registered workers."""
        return {"urls": list(self.worker_request_counts.keys())}

    async def proxy(self, request: Request, path: str):
        """Catch-all: forward unmatched requests to a selected worker."""
        return await self._forward_to_worker(request, path)
