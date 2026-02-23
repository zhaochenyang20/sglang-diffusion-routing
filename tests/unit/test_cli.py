"""Unit tests for CLI argument parsing.

Tests the argparse configuration directly — no mocks, no process spawning.
CLI integration (actually starting the router) is covered in e2e tests.
"""

from __future__ import annotations

import pytest

from sglang_diffusion_routing.cli.main import build_parser


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

    def test_full_args(self):
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
                "--timeout",
                "0.5",
                "--max-connections",
                "500",
                "--health-check-interval",
                "30",
                "--health-check-failure-threshold",
                "5",
                "--verbose",
                "--log-level",
                "warning",
            ]
        )
        assert args.host == "127.0.0.1"
        assert args.port == 31000
        assert args.worker_urls == ["http://localhost:10090", "http://localhost:10092"]
        assert args.routing_algorithm == "round-robin"
        assert args.timeout == 0.5

    def test_rejects_invalid_routing_algorithm(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--routing-algorithm", "invalid-algo"])

    def test_accepts_all_valid_algorithms(self):
        for algo in ("least-request", "round-robin", "random"):
            args = build_parser().parse_args(["--routing-algorithm", algo])
            assert args.routing_algorithm == algo
