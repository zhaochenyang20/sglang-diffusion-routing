import requests
import time


def test_update_weights_broadcast(router, two_workers):
    # Call router broadcast endpoint (ported from miles#544)
    r = requests.post(router.base + "/update_weights_from_disk", json={"flush_cache": True}, timeout=5)
    assert r.status_code == 200

    # each worker should have update_calls == 1
    time.sleep(0.2)
    for w in two_workers:
        s = requests.get(f"http://{w.host}:{w.port}/__state", timeout=1).json()
        assert s["update_calls"] == 1
