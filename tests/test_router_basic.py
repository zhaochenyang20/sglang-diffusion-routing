import requests


def _extract_worker_urls(list_workers_json):
    """
    Be tolerant to different /list_workers response formats across router versions.
    Accept:
      - {"urls": ["url1", "url2"]}                         (your current PR variant)
      - {"workers": ["url1", "url2"]}
      - {"workers": {"url1": 0, "url2": 3}}
      - {"worker_urls": ["url1", "url2"]}
      - ["url1", "url2"]
    """
    if isinstance(list_workers_json, list):
        return list_workers_json

    if not isinstance(list_workers_json, dict):
        return []

    # PR variant: {"urls": [...]}
    if "urls" in list_workers_json and isinstance(list_workers_json["urls"], list):
        return list_workers_json["urls"]

    if "workers" in list_workers_json:
        w = list_workers_json["workers"]
        if isinstance(w, list):
            return w
        if isinstance(w, dict):
            return list(w.keys())

    if "worker_urls" in list_workers_json and isinstance(list_workers_json["worker_urls"], list):
        return list_workers_json["worker_urls"]

    return []


def test_health_and_list_workers(router):
    r = requests.get(router.base + "/health", timeout=1)
    assert r.status_code == 200

    r = requests.get(router.base + "/list_workers", timeout=1)
    assert r.status_code == 200

    urls = _extract_worker_urls(r.json())
    assert len(urls) == 2, f"/list_workers returned unexpected payload: {r.json()}"

    r = requests.get(router.base + "/health_workers", timeout=1)
    assert r.status_code == 200
