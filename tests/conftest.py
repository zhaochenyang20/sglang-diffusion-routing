import os
import sys
import time
from types import SimpleNamespace

import pytest
import requests

# Ensure <repo_root>/src is importable for tests (no editable install needed).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from sglang_d_router.router.diffusion_router import DiffusionRouter  # noqa: E402
from tests.fake_worker import make_fake_worker  # noqa: E402
from tests.utils_uvicorn import pick_free_port, start_uvicorn_app, stop_uvicorn  # noqa: E402


@pytest.fixture
def two_workers():
    w1_app = make_fake_worker("w1")
    w2_app = make_fake_worker("w2")

    w1 = start_uvicorn_app(w1_app, port=pick_free_port())
    w2 = start_uvicorn_app(w2_app, port=pick_free_port())

    yield [w1, w2]

    stop_uvicorn(w1)
    stop_uvicorn(w2)


@pytest.fixture
def router(two_workers):
    # Align with your demo.py / DiffusionRouter args expectation.
    args = SimpleNamespace(
        host="127.0.0.1",
        port=pick_free_port(),
        worker_urls=[],
        max_connections=100,
        timeout=None,
        health_check_interval=1,  # speed up tests
        health_check_failure_threshold=2,
        routing_algorithm="least-request",
        verbose=False,
    )

    router_obj = DiffusionRouter(args, verbose=False)

    # Register workers
    worker_urls = []
    for w in two_workers:
        url = f"http://{w.host}:{w.port}"
        worker_urls.append(url)
        router_obj.register_worker(url)

    # Some variants of the copied router may not populate worker_request_counts
    # through register_worker(). Ensure internal state exists for tests.
    if getattr(router_obj, "worker_request_counts", None) is None:
        router_obj.worker_request_counts = {}
    for url in worker_urls:
        router_obj.worker_request_counts.setdefault(url, 0)

    # Ensure failure counts/dead set exist if the router expects them
    if getattr(router_obj, "worker_failure_counts", None) is None:
        router_obj.worker_failure_counts = {url: 0 for url in router_obj.worker_request_counts}
    if getattr(router_obj, "dead_workers", None) is None:
        router_obj.dead_workers = set()

    handle = start_uvicorn_app(router_obj.app, host=args.host, port=args.port, log_level="warning")

    # Wait router health ready
    base = f"http://{handle.host}:{handle.port}"
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            r = requests.get(base + "/health", timeout=0.2)
            # some routers may reply 503 while bootstrapping; accept it as "up"
            if r.status_code in (200, 503):
                break
        except Exception:
            time.sleep(0.05)

    yield SimpleNamespace(handle=handle, base=base, obj=router_obj)

    stop_uvicorn(handle)
