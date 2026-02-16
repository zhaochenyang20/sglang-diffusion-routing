import argparse

import uvicorn

from sglang_d_router.router.diffusion_router import DiffusionRouter


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SGLang Diffusion Router Demo")
    p.add_argument("--host", type=str, default="0.0.0.0", help="Router bind address")
    p.add_argument("--port", type=int, default=30080, help="Router port")
    p.add_argument("--worker-urls", nargs="*", default=[], help="Initial diffusion worker URLs")
    p.add_argument("--max-connections", type=int, default=100, help="Max concurrent connections to workers")
    p.add_argument("--timeout", type=float, default=None, help="Request timeout in seconds")
    p.add_argument("--health-check-interval", type=int, default=10, help="Seconds between health checks")
    p.add_argument("--health-check-failure-threshold", type=int, default=3, help="Failures before marking dead")
    p.add_argument(
        "--routing-algorithm",
        type=str,
        default="least-request",
        choices=["least-request", "round-robin", "random"],
        help="Load-balancing algorithm",
    )
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    router = DiffusionRouter(args, verbose=args.verbose)

    for url in args.worker_urls:
        router.register_worker(url)

    print(f"[router-demo] Starting on {args.host}:{args.port}")
    print(f"[router-demo] Workers: {args.worker_urls or '(none)'}")
    uvicorn.run(router.app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
