from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def make_fake_worker(worker_id: str) -> FastAPI:
    app = FastAPI()
    state = {"worker_id": worker_id, "update_calls": 0}

    @app.get("/health")
    def health():
        return {"status": "ok", "worker_id": worker_id}

    @app.post("/v1/images/generations")
    async def images(req: Request):
        payload = await req.json()
        # Return a minimal OpenAI-like structure (not full fidelity, enough for router)
        return JSONResponse(
            {
                "worker_id": worker_id,
                "received": payload,
                "data": [{"b64_json": "ZmFrZV9pbWFnZV9kYXRh"}],  # "fake_image_data" base64-ish
            }
        )

    @app.post("/v1/videos")
    async def videos(req: Request):
        payload = await req.json()
        return JSONResponse({"worker_id": worker_id, "received": payload, "data": [{"b64_json": "ZmFrZV92aWRlbw=="}]})

    @app.post("/update_weights_from_disk")
    async def update_weights(req: Request):
        _ = await req.json()
        state["update_calls"] += 1
        return {"ok": True, "worker_id": worker_id, "update_calls": state["update_calls"]}

    @app.get("/__state")
    def get_state():
        return state

    return app
