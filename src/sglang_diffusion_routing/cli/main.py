# This module is derived from radixark/miles#544.
# See README.md for full acknowledgment.

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from contextlib import suppress

from sglang_diffusion_routing import DiffusionRouter
from sglang_diffusion_routing.launcher import config as _lcfg


def _print_fired_up_banner(log_prefix: str) -> None:
    banner = r"""

  ____   ____ _                      ____  _  __  __           _
 / ___| / ___| |    __ _ _ __   __ _|  _ \(_)/ _|/ _|_   _ ___(_) ___  _ __
 \___ \| |  _| |   / _` | '_ \ / _` | | | | | |_| |_| | | / __| |/ _ \| '_ \
  ___) | |_| | |__| (_| | | | | (_| | |_| | |  _|  _| |_| \__ \ | (_) | | | |
 |____/ \____|_____\__,_|_| |_|\__, |____/|_|_| |_|  \__,_|___/_|\___/|_| |_|
                                |___/

  ____             _            _____                _
 |  _ \ ___  _   _| |_ ___ ____|  ___(_)_ __ ___  __| |_   _ _ __
 | |_) / _ \| | | | __/ _ \  __| |_  | | '__/ _ \/ _` | | | | '_ \
 |  _ < (_) | |_| | ||  __/ |  |  _| | | | |  __/ (_| | |_| | |_) |
 |_| \_\___/ \__,_|\__\___|_|  |_|   |_|_|  \___|\__,_\__,__| .__/
                                                            |_|

"""
    use_color = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    if not use_color:
        print(f"{log_prefix} {banner}", flush=True)
        return

    colors = [
        "\033[38;5;45m",  # cyan
        "\033[38;5;51m",  # bright cyan
        "\033[38;5;123m",  # light cyan
        "\033[38;5;159m",  # pale blue
    ]
    reset = "\033[0m"
    colored_lines = []
    for idx, line in enumerate(banner.splitlines()):
        if line.strip():
            color = colors[idx % len(colors)]
            colored_lines.append(f"{color}{line}{reset}")
        else:
            colored_lines.append(line)
    colored_banner = "\n".join(colored_lines)
    print(f"{log_prefix} {colored_banner}", flush=True)


def _run_router_server(
    args: argparse.Namespace,
    router: DiffusionRouter,
    log_prefix: str = "[router]",
) -> None:
    try:
        import uvicorn  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "uvicorn is required to run router. Install with: pip install uvicorn"
        ) from exc

    worker_urls = list(args.worker_urls or [])
    refresh_tasks = []
    for url in worker_urls:
        normalized_url = router.normalize_worker_url(url)
        router.register_worker(normalized_url)
        refresh_tasks.append(router.refresh_worker_video_support(normalized_url))

    if refresh_tasks:

        async def _refresh_all_worker_video_support() -> None:
            await asyncio.gather(*refresh_tasks)

        asyncio.run(_refresh_all_worker_video_support())

    print(f"{log_prefix} starting router on {args.host}:{args.port}", flush=True)
    print(
        f"{log_prefix} workers: {list(router.worker_request_counts.keys()) or '(none - add via POST /workers)'}",
        flush=True,
    )
    config = uvicorn.Config(
        app=router.app,
        host=args.host,
        port=args.port,
        log_level=getattr(args, "log_level", "info"),
    )
    server = uvicorn.Server(config)

    async def _serve_with_banner() -> None:
        banner_printed = False

        async def _wait_and_print_banner() -> None:
            nonlocal banner_printed
            while not server.started and not server.should_exit:
                await asyncio.sleep(0.1)
            if server.started and not banner_printed:
                banner_printed = True
                _print_fired_up_banner(log_prefix)

        watcher = asyncio.create_task(_wait_and_print_banner())
        try:
            await server.serve()
        finally:
            watcher.cancel()
            with suppress(asyncio.CancelledError):
                await watcher

    asyncio.run(_serve_with_banner())


def _add_router_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Router bind address."
    )
    parser.add_argument("--port", type=int, default=30080, help="Router port.")
    parser.add_argument(
        "--worker-urls", nargs="*", default=[], help="Initial diffusion worker URLs."
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=100,
        help="Max concurrent connections to workers.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Router-to-worker request timeout in seconds.",
    )
    parser.add_argument(
        "--health-check-interval",
        type=int,
        default=10,
        help="Seconds between health checks.",
    )
    parser.add_argument(
        "--health-check-failure-threshold",
        type=int,
        default=3,
        help="Consecutive failures before quarantine.",
    )
    parser.add_argument(
        "--routing-algorithm",
        type=str,
        default="least-request",
        choices=["least-request", "round-robin", "random"],
        help="Load-balancing algorithm.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging."
    )
    parser.add_argument(
        "--log-level", type=str, default="info", help="Uvicorn log level."
    )
    parser.add_argument(
        "--launcher-config",
        type=str,
        default=None,
        dest="launcher_config",
        help="YAML config for launching router managed workers (see examples/local_launcher.yaml).",
    )


def _handle_router(args: argparse.Namespace) -> int:
    log_prefix, backend, router = "[sglang-d-router]", None, None

    try:
        router = DiffusionRouter(args, verbose=args.verbose)
        if args.launcher_config is not None:
            launcher_cfg = _lcfg.load_launcher_config(args.launcher_config)
            wait_timeout = launcher_cfg.wait_timeout
            backend = _lcfg.create_backend(launcher_cfg)
            launched_urls = backend.launch()
            backend.wait_ready_and_register(
                register_func=router.register_worker,
                timeout=wait_timeout,
                log_prefix=log_prefix,
            )
            registered_urls = set(router.worker_request_counts.keys())
            pending_urls = [u for u in launched_urls if u not in registered_urls]
            if pending_urls:
                raise RuntimeError(
                    "managed workers failed to become healthy before router startup: "
                    + ", ".join(pending_urls)
                )

        _run_router_server(args, router=router, log_prefix=log_prefix)
        return 0
    finally:
        # TODO (mengyang, shuwen, chenyang): refactor the exit logic of router and backend.
        if router is not None:
            try:
                asyncio.run(router.client.aclose())
            except Exception as exc:
                print(
                    f"{log_prefix} warning: failed to close router client: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        if backend is not None:
            print(f"{log_prefix} shutting down managed workers...", flush=True)
            backend.shutdown()
            print(f"{log_prefix} all managed workers terminated.", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sglang-d-router",
        description="SGLang diffusion router CLI.",
    )
    _add_router_args(parser)
    parser.set_defaults(handler=_handle_router)
    return parser


def run_cli(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = args.handler
    return handler(args)


def main(argv: list[str] | None = None) -> int:
    try:
        return run_cli(argv)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[sglang-d-router] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
