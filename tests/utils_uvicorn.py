import socket
import threading
import time
from dataclasses import dataclass
from typing import Optional

import requests
import uvicorn


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@dataclass
class UvicornHandle:
    host: str
    port: int
    server: uvicorn.Server
    thread: threading.Thread


def start_uvicorn_app(app, host: str = "127.0.0.1", port: Optional[int] = None, log_level: str = "warning") -> UvicornHandle:
    if port is None:
        port = pick_free_port()

    config = uvicorn.Config(app=app, host=host, port=port, log_level=log_level, lifespan="on")
    server = uvicorn.Server(config=config)

    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    # Wait until server is ready
    base = f"http://{host}:{port}"
    deadline = time.time() + 10.0
    last_err = None
    while time.time() < deadline:
        try:
            # We don't know which routes exist, but TCP connect is enough.
            requests.get(base + "/health", timeout=0.2)
            break
        except Exception as e:
            last_err = e
            time.sleep(0.05)

    # Some apps may not have /health; do a bare connect fallback by requesting /
    if last_err is not None:
        deadline = time.time() + 10.0
        while time.time() < deadline:
            try:
                requests.get(base + "/", timeout=0.2)
                break
            except Exception:
                time.sleep(0.05)

    return UvicornHandle(host=host, port=port, server=server, thread=t)


def stop_uvicorn(handle: UvicornHandle) -> None:
    handle.server.should_exit = True
    # give it a moment to exit
    handle.thread.join(timeout=5)
