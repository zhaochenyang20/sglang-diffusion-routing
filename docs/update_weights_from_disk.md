# update_weights_from_disk

This document describes `POST /update_weights_from_disk` behavior in this repository.

## Router behavior

The router does not validate or transform payload fields.
It forwards the original request body to every healthy worker and returns per-worker results.

Payload semantics are therefore defined by the worker implementation, not by the router.

## Requirements

- Worker servers must implement `POST /update_weights_from_disk`.
- For SGLang workers, use a version that includes this endpoint.
- Weights must match your worker runtime expectations.

## Basic example

```bash
curl -X POST http://localhost:30080/update_weights_from_disk \
    -H "Content-Type: application/json" \
    -d '{"model_path": "/path/to/new/weights"}'
```

## Optional fields

Some worker versions support optional fields such as `target_modules`:

```bash
curl -X POST http://localhost:30080/update_weights_from_disk \
    -H "Content-Type: application/json" \
    -d '{"model_path": "/path/to/weights", "target_modules": ["transformer", "vae"]}'
```

If your worker version does not support extra fields, failure is returned by the worker side.

## Response shape

The router response includes one item per healthy worker:

```json
{
  "results": [
    {
      "worker_url": "http://localhost:10090",
      "status_code": 200,
      "body": {
        "ok": true
      }
    },
    {
      "worker_url": "http://localhost:10092",
      "status_code": 500,
      "body": {
        "error": "worker-side failure"
      }
    }
  ]
}
```

Notes:
- Quarantined workers are excluded from broadcast.
- Transport/runtime exceptions are surfaced as per-worker `status_code=502`.
