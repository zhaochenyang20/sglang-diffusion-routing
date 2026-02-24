from __future__ import annotations

from unittest import mock

from sglang_diffusion_routing.cli.main import build_parser, run_cli


class TestCLIParser:
    def test_defaults(self):
        args = build_parser().parse_args([])
        assert args.host == "0.0.0.0"
        assert args.port == 30080
        assert args.worker_urls == []
        assert args.routing_algorithm == "least-request"
        assert args.timeout == 120.0
        assert args.max_connections == 100
        assert args.health_check_interval == 10
        assert args.health_check_failure_threshold == 3
        assert args.verbose is False
        assert args.log_level == "info"

    def test_parses_worker_urls(self):
        args = build_parser().parse_args(
            [
                "--host",
                "127.0.0.1",
                "--port",
                "31000",
                "--worker-urls",
                "http://localhost:10090",
                "http://localhost:10092",
                "--routing-algorithm",
                "round-robin",
                "--verbose",
                "--log-level",
                "warning",
            ]
        )
        assert args.host == "127.0.0.1"
        assert args.port == 31000
        assert args.worker_urls == ["http://localhost:10090", "http://localhost:10092"]
        assert args.routing_algorithm == "round-robin"
        assert args.verbose is True
        assert args.log_level == "warning"


def test_run_cli_calls_router_runner():
    def _mock_run_router_server(args, router, log_prefix):
        for url in args.worker_urls:
            router.register_worker(url)

    with mock.patch("sglang_diffusion_routing.cli.main._run_router_server") as mock_run:
        mock_run.side_effect = _mock_run_router_server
        code = run_cli(["--port", "30123", "--worker-urls", "http://localhost:10090"])
        assert code == 0
        mock_run.assert_called_once()
        args = mock_run.call_args.args[0]
        assert args.port == 30123
        assert args.worker_urls == ["http://localhost:10090"]
        router = mock_run.call_args.kwargs["router"]
        assert "http://localhost:10090" in router.worker_request_counts
        assert mock_run.call_args.kwargs["log_prefix"] == "[sglang-d-router]"
