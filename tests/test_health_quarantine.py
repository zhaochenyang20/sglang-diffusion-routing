import time

import requests

from tests.utils_uvicorn import stop_uvicorn


def _any_dead_or_quarantined(payload: dict) -> bool:
    """
    Tolerant parsing for /health_workers response across variants.

    Accept formats like:
      1) {"workers": [{"url":..., "is_dead": True, ...}, ...]}   (your current output)
      2) {"workers": {"url": {"is_dead": True}, ...}}
      3) {"alive": [...], "dead": [...]}  or {"healthy": [...], "unhealthy": [...]}
      4) {"some_url": {"is_dead": True}}  (flat dict)
    """
    if not isinstance(payload, dict):
        return False

    # Variant 1: list of worker dicts
    workers = payload.get("workers")
    if isinstance(workers, list):
        for w in workers:
            if isinstance(w, dict):
                if w.get("dead") is True or w.get("is_dead") is True or w.get("quarantined") is True:
                    return True
        # also accept "status" field variants
        for w in workers:
            if isinstance(w, dict) and w.get("status") in ("dead", "quarantined", "unhealthy"):
                return True

    # Variant 2: dict mapping url -> info
    if isinstance(workers, dict):
        for _, info in workers.items():
            if isinstance(info, dict):
                if info.get("dead") is True or info.get("is_dead") is True or info.get("quarantined") is True:
                    return True
                if info.get("status") in ("dead", "quarantined", "unhealthy"):
                    return True

    # Variant 3: explicit lists
    for key in ("dead", "quarantined", "unhealthy"):
        v = payload.get(key)
        if isinstance(v, list) and len(v) > 0:
            return True

    # Variant 4: flat dict
    for _, v in payload.items():
        if isinstance(v, dict):
            if v.get("dead") is True or v.get("is_dead") is True or v.get("quarantined") is True:
                return True
            if v.get("status") in ("dead", "quarantined", "unhealthy"):
                return True

    return False


def test_health_check_quarantine(router, two_workers):
    # Kill worker 2
    w2 = two_workers[1]
    stop_uvicorn(w2)

    # Wait for health-check loop to mark it dead
    deadline = time.time() + 5.0
    last_payload = None

    while time.time() < deadline:
        r = requests.get(router.base + "/health_workers", timeout=1)
        if r.status_code != 200:
            time.sleep(0.1)
            continue

        last_payload = r.json()
        if _any_dead_or_quarantined(last_payload):
            return

        time.sleep(0.2)

    raise AssertionError(
        f"Expected one worker to be marked dead/quarantined in /health_workers, got: {last_payload}"
    )
