import requests


def test_proxy_images_generations(router):
    payload = {"model": "fake-model", "prompt": "a cat", "n": 1, "size": "1024x1024"}
    r = requests.post(router.base + "/v1/images/generations", json=payload, timeout=3)
    assert r.status_code == 200
    j = r.json()
    assert "worker_id" in j
    assert j["received"]["prompt"] == "a cat"
