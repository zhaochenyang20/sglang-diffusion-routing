#!/usr/bin/env python3
"""
Compare routing algorithms by running bench_router.py for each one and
collecting results in JSON and CSV outputs.

Example:
  python tests/benchmarks/diffusion_router/bench_routing_algorithms.py \
    --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --num-workers 2 \
    --num-prompts 10 \
    --max-concurrency 2
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ALL_ALGORITHMS = ["least-request", "round-robin", "random"]
BASELINE = "random"


def _require_non_empty_model(model: str) -> str:
    normalized = model.strip()
    if not normalized:
        raise ValueError(
            "--model must be a non-empty model ID/path. "
            "Detected an empty value, which often means a shell variable such as "
            "$MODEL was unset."
        )
    return normalized


def _pct_delta(value: float | int | str, baseline: float | int | str) -> float | str:
    if not isinstance(value, (int, float)):
        return ""
    if not isinstance(baseline, (int, float)):
        return ""
    if baseline == 0:
        return ""
    return ((value - baseline) / abs(baseline)) * 100.0


def _format_num(value: object, width: int = 10) -> str:
    if isinstance(value, (int, float)):
        return (
            f"{value:>{width}.4f}" if isinstance(value, float) else f"{value:>{width}d}"
        )
    return f"{str(value):>{width}}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare routing algorithms by running bench_router.py for each."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Diffusion model HF ID or local path."
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=ALL_ALGORITHMS,
        choices=ALL_ALGORITHMS,
        help="Algorithms to compare (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store result artifacts.",
    )

    # Pass-through arguments for bench_router.py
    parser.add_argument("--router-host", type=str, default="127.0.0.1")
    parser.add_argument("--router-port", type=int, default=30080)
    parser.add_argument("--router-verbose", action="store_true")
    parser.add_argument("--router-max-connections", type=int, default=100)
    parser.add_argument("--router-timeout", type=float, default=120.0)
    parser.add_argument("--router-health-check-interval", type=int, default=10)
    parser.add_argument("--router-health-check-failure-threshold", type=int, default=3)
    parser.add_argument("--router-extra-args", type=str, default="")
    parser.add_argument("--worker-host", type=str, default="127.0.0.1")
    parser.add_argument("--worker-urls", nargs="*", default=[])
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-base-port", type=int, default=10090)
    parser.add_argument("--worker-port-stride", type=int, default=2)
    parser.add_argument("--worker-master-port-base", type=int, default=30005)
    parser.add_argument("--worker-scheduler-port-base", type=int, default=5555)
    parser.add_argument("--worker-internal-port-stride", type=int, default=1000)
    parser.add_argument("--num-gpus-per-worker", type=int, default=1)
    parser.add_argument("--worker-gpu-ids", nargs="*", default=None)
    parser.add_argument("--worker-extra-args", type=str, default="")
    parser.add_argument("--skip-workers", action="store_true")
    parser.add_argument(
        "--dataset", type=str, default="random", choices=["vbench", "random"]
    )
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--bench-extra-args", type=str, default="")
    parser.add_argument("--wait-timeout", type=int, default=1200)

    args = parser.parse_args()
    args.model = _require_non_empty_model(args.model)

    script_dir = Path(__file__).resolve().parent
    bench_router_script = script_dir / "bench_router.py"
    if not bench_router_script.exists():
        raise RuntimeError(f"Missing benchmark script: {bench_router_script}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else script_dir
        / "outputs"
        / f"routing_algo_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    py = sys.executable

    for algo in args.algorithms:
        print(f"\n{'=' * 72}", flush=True)
        print(f"[bench] Running routing algorithm: {algo}", flush=True)
        print(f"{'=' * 72}\n", flush=True)

        out_file = output_dir / f"bench_{algo}.json"
        cmd = [
            py,
            str(bench_router_script),
            "--model",
            args.model,
            "--routing-algorithm",
            algo,
            "--router-host",
            args.router_host,
            "--router-port",
            str(args.router_port),
            "--router-max-connections",
            str(args.router_max_connections),
            "--router-timeout",
            str(args.router_timeout),
            "--router-health-check-interval",
            str(args.router_health_check_interval),
            "--router-health-check-failure-threshold",
            str(args.router_health_check_failure_threshold),
            "--worker-host",
            args.worker_host,
            "--num-workers",
            str(args.num_workers),
            "--worker-base-port",
            str(args.worker_base_port),
            "--worker-port-stride",
            str(args.worker_port_stride),
            "--worker-master-port-base",
            str(args.worker_master_port_base),
            "--worker-scheduler-port-base",
            str(args.worker_scheduler_port_base),
            "--worker-internal-port-stride",
            str(args.worker_internal_port_stride),
            "--num-gpus-per-worker",
            str(args.num_gpus_per_worker),
            "--dataset",
            args.dataset,
            "--num-prompts",
            str(args.num_prompts),
            "--max-concurrency",
            str(args.max_concurrency),
            "--request-rate",
            str(args.request_rate),
            "--wait-timeout",
            str(args.wait_timeout),
            "--log-level",
            args.log_level,
            "--output-file",
            str(out_file),
        ]

        if args.worker_urls:
            cmd += ["--worker-urls", *args.worker_urls]
        if args.worker_gpu_ids:
            cmd += ["--worker-gpu-ids", *args.worker_gpu_ids]
        if args.dataset_path:
            cmd += ["--dataset-path", args.dataset_path]
        if args.task:
            cmd += ["--task", args.task]
        if args.width:
            cmd += ["--width", str(args.width)]
        if args.height:
            cmd += ["--height", str(args.height)]
        if args.num_frames:
            cmd += ["--num-frames", str(args.num_frames)]
        if args.fps:
            cmd += ["--fps", str(args.fps)]
        if args.disable_tqdm:
            cmd.append("--disable-tqdm")
        if args.router_verbose:
            cmd.append("--router-verbose")
        if args.skip_workers:
            cmd.append("--skip-workers")
        if args.worker_extra_args:
            cmd += ["--worker-extra-args", args.worker_extra_args]
        if args.router_extra_args:
            cmd += ["--router-extra-args", args.router_extra_args]
        if args.bench_extra_args:
            cmd += ["--bench-extra-args", args.bench_extra_args]

        print("[run]", " ".join(shlex.quote(x) for x in cmd), flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            print(
                f"[warn] benchmark exited with code {rc} for algorithm '{algo}'",
                flush=True,
            )
            results[algo] = {"error": f"exit_code={rc}"}
            continue

        if not out_file.exists():
            print(f"[warn] output file missing: {out_file}", flush=True)
            results[algo] = {"error": "output_file_missing"}
            continue

        try:
            results[algo] = json.loads(out_file.read_text())
        except json.JSONDecodeError as exc:
            print(f"[warn] invalid JSON in {out_file}: {exc}", flush=True)
            results[algo] = {"error": f"json_parse_error={exc}"}

    metric_keys = ["throughput_qps", "latency_mean", "latency_median", "latency_p99"]
    baseline = results.get(BASELINE, {})
    rows: list[dict[str, object]] = []

    for algo in args.algorithms:
        data = results.get(algo, {})
        if "error" in data:
            row = {
                "algorithm": algo,
                "throughput_qps": "",
                "latency_mean": "",
                "latency_median": "",
                "latency_p99": "",
                "duration": "",
                "completed_requests": "",
                "failed_requests": "",
                "throughput_qps_delta_pct": "",
                "latency_mean_delta_pct": "",
                "latency_median_delta_pct": "",
                "latency_p99_delta_pct": "",
                "error": data["error"],
            }
            rows.append(row)
            continue

        row = {
            "algorithm": algo,
            "throughput_qps": data.get("throughput_qps", ""),
            "latency_mean": data.get("latency_mean", ""),
            "latency_median": data.get("latency_median", ""),
            "latency_p99": data.get("latency_p99", ""),
            "duration": data.get("duration", ""),
            "completed_requests": data.get("completed_requests", ""),
            "failed_requests": data.get("failed_requests", ""),
            "error": "",
        }
        for key in metric_keys:
            row[f"{key}_delta_pct"] = _pct_delta(
                row.get(key, ""), baseline.get(key, "")
            )
        rows.append(row)

    print(f"\n{'=' * 108}", flush=True)
    print(f"[summary] Routing Algorithm Comparison (baseline: {BASELINE})", flush=True)
    print(f"{'=' * 108}", flush=True)
    header = (
        f"{'Algorithm':<16} {'Throughput':>12} {'TputDelta%':>12} "
        f"{'MeanLat':>12} {'MeanDelta%':>12} "
        f"{'P99Lat':>12} {'P99Delta%':>12} "
        f"{'Done':>8} {'Fail':>8}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for row in rows:
        if row.get("error"):
            print(
                f"{row['algorithm']:<16} {'-':>12} {'-':>12} {'-':>12} {'-':>12} "
                f"{'-':>12} {'-':>12} {'-':>8} {'-':>8}  error={row['error']}",
                flush=True,
            )
            continue
        print(
            f"{row['algorithm']:<16} "
            f"{_format_num(row['throughput_qps'], 12)} {_format_num(row['throughput_qps_delta_pct'], 12)} "
            f"{_format_num(row['latency_mean'], 12)} {_format_num(row['latency_mean_delta_pct'], 12)} "
            f"{_format_num(row['latency_p99'], 12)} {_format_num(row['latency_p99_delta_pct'], 12)} "
            f"{_format_num(row['completed_requests'], 8)} {_format_num(row['failed_requests'], 8)}",
            flush=True,
        )

    csv_path = output_dir / "routing_algorithm_comparison.csv"
    fieldnames = list(rows[0].keys()) if rows else ["algorithm"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "routing_algorithm_comparison.json"
    summary_path.write_text(
        json.dumps({"results": results, "rows": rows, "baseline": BASELINE}, indent=2)
    )

    print("\n[done] Artifacts:", flush=True)
    print(f"  - {csv_path}", flush=True)
    print(f"  - {summary_path}", flush=True)

    # Return non-zero if every algorithm failed.
    failures = sum(1 for row in rows if row.get("error"))
    return 1 if rows and failures == len(rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
