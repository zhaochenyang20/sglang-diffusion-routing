#!/usr/bin/env python3
"""
Fake sglang diffusion worker for integration testing.

Implements the same HTTP API contract as a real sglang diffusion worker,
but returns canned responses without any GPU or model dependencies.

Usage:
    python tests/integration/fake_worker.py --port 19000
    python tests/integration/fake_worker.py --port 19000 --fail-rate 0.5
    python tests/integration/fake_worker.py --port 19000 --latency 0.2
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import random
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# 1x1 red PNG pixel
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
    "2mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
)
_TINY_PNG_B64 = base64.b64encode(_TINY_PNG).decode()


def create_app(
    fail_rate: float = 0.0,
    latency: float = 0.0,
    worker_id: str = "fake-worker",
    task_type: str = "T2V",
) -> FastAPI:
    app = FastAPI()
    request_count = {"total": 0, "generate": 0, "video": 0, "weights": 0}

    @app.get("/health")
    async def health():
        return {"status": "ok", "worker_id": worker_id}

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": "fake-model",
                    "task_type": task_type,
                }
            ],
        }

    @app.post("/v1/images/generations")
    async def generate_image(request: Request):
        request_count["total"] += 1
        request_count["generate"] += 1

        if latency > 0:
            await asyncio.sleep(latency)

        if fail_rate > 0 and random.random() < fail_rate:
            return JSONResponse(
                status_code=500,
                content={"detail": "Simulated worker failure"},
            )

        body = await request.json()
        response_format = body.get("response_format", "url")
        prompt = body.get("prompt", "")
        n = body.get("n", body.get("num_images", 1))

        data = []
        for i in range(n):
            if response_format == "b64_json":
                data.append(
                    {
                        "b64_json": _TINY_PNG_B64,
                        "revised_prompt": prompt,
                        "index": i,
                    }
                )
            else:
                data.append(
                    {
                        "url": f"http://localhost/files/img_{request_count['generate']:04d}_{i}.png",
                        "revised_prompt": prompt,
                        "index": i,
                    }
                )

        return {
            "created": int(time.time()),
            "data": data,
            "model": body.get("model", "fake-model"),
            "worker_id": worker_id,
        }

    @app.post("/v1/videos")
    async def generate_video(request: Request):
        request_count["total"] += 1
        request_count["video"] += 1

        if latency > 0:
            await asyncio.sleep(latency)

        body = await request.json()
        prompt = body.get("prompt", "")

        return {
            "created": int(time.time()),
            "data": [
                {
                    "url": f"http://localhost/files/vid_{request_count['video']:04d}.mp4",
                    "revised_prompt": prompt,
                }
            ],
            "model": body.get("model", "fake-model"),
            "worker_id": worker_id,
        }

    @app.post("/update_weights_from_disk")
    async def update_weights(request: Request):
        request_count["total"] += 1
        request_count["weights"] += 1
        body = await request.json()
        return {
            "ok": True,
            "model_path": body.get("model_path", ""),
            "worker_id": worker_id,
        }

    @app.get("/stats")
    async def stats():
        """Test helper: return request counts."""
        return request_count

    return app


def main():
    parser = argparse.ArgumentParser(description="Fake sglang diffusion worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--fail-rate", type=float, default=0.0)
    parser.add_argument("--latency", type=float, default=0.0)
    parser.add_argument("--worker-id", type=str, default=None)
    parser.add_argument("--task-type", type=str, default="T2V")
    args = parser.parse_args()

    worker_id = args.worker_id or f"fake-worker-{args.port}"
    app = create_app(
        fail_rate=args.fail_rate,
        latency=args.latency,
        worker_id=worker_id,
        task_type=args.task_type,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
