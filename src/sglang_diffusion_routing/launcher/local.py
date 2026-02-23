"""Local subprocess launcher backend."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING, DictConfig

from sglang_diffusion_routing.launcher.backend import (
    LaunchedWorker,
    LauncherBackend,
    WorkerLaunchResult,
)
from sglang_diffusion_routing.launcher.utils import (
    build_gpu_assignments,
    infer_connect_host,
    reserve_available_port,
    terminate_all,
    wait_for_health,
)


@dataclass
class LocalLauncherConfig:
    """Typed configuration for the local subprocess launcher backend."""

    backend: str = "local"

    model: str = MISSING

    num_workers: int = 1
    num_gpus_per_worker: int = 1
    worker_host: str = "127.0.0.1"
    worker_base_port: int = 10090
    worker_gpu_ids: Optional[list[str]] = None
    worker_extra_args: str = ""
    log_prefix: str = "[local-launcher]"

    master_port_base: int = 30005
    scheduler_port_base: int = 5555
    internal_port_stride: int = 1000

    wait_timeout: int = 600


class LocalLauncher(LauncherBackend):
    """Launch SGLang workers as local subprocesses."""

    def __init__(self, config: DictConfig) -> None:
        self._config = config
        self._result: WorkerLaunchResult | None = None

    def launch(self) -> list[str]:
        cfg = self._config

        model = cfg.model
        if isinstance(model, str):
            model = model.strip()
        if not model:
            raise ValueError(
                "launcher config 'model' is required and must not be empty"
            )

        if shutil.which("sglang") is None:
            raise RuntimeError("'sglang' command not found on PATH. ")

        self._result = _launch_workers(
            model=model,
            num_workers=cfg.num_workers,
            num_gpus_per_worker=cfg.num_gpus_per_worker,
            worker_host=cfg.worker_host,
            worker_base_port=cfg.worker_base_port,
            worker_gpu_ids=cfg.worker_gpu_ids,
            worker_extra_args=cfg.worker_extra_args,
            master_port_base=cfg.master_port_base,
            scheduler_port_base=cfg.scheduler_port_base,
            internal_port_stride=cfg.internal_port_stride,
            log_prefix=cfg.log_prefix,
        )
        return self._result.urls

    def wait_ready_and_register(
        self,
        register_func: Callable[[str], None],
        timeout: int,
        log_prefix: str = "[launcher]",
    ) -> None:
        if self._result is None:
            return

        threads = [
            threading.Thread(
                target=self._wait_and_register_worker,
                args=(w, register_func, timeout, log_prefix),
            )
            for w in self._result.workers
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    @staticmethod
    def _wait_and_register_worker(
        worker: LaunchedWorker,
        register_func: Callable[[str], None],
        timeout: int,
        log_prefix: str,
    ) -> None:
        try:
            wait_for_health(
                url=worker.url,
                timeout=timeout,
                label=f"worker {worker.url}",
                proc=worker.process,
                log_prefix=log_prefix,
            )
            register_func(worker.url)
            print(f"{log_prefix} registered {worker.url}", flush=True)
        except Exception as exc:
            print(
                f"{log_prefix} worker {worker.url} failed health check: {exc}",
                flush=True,
            )

    def shutdown(self) -> None:
        if self._result is None:
            return
        terminate_all(self._result.all_processes)
        self._result = None


def _launch_workers(
    model: str,
    num_workers: int,
    num_gpus_per_worker: int = 1,
    worker_host: str = "127.0.0.1",
    worker_base_port: int = 10090,
    worker_gpu_ids: list[str] | None = None,
    worker_extra_args: str = "",
    master_port_base: int = 30005,
    scheduler_port_base: int = 5555,
    internal_port_stride: int = 1000,
    log_prefix: str = "[launcher]",
) -> WorkerLaunchResult:
    """Launch num_workers 'sglang serve' subprocesses."""
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    if num_gpus_per_worker < 1:
        raise ValueError("num_gpus_per_worker must be >= 1")

    worker_env = os.environ.copy()
    gpu_assignments = build_gpu_assignments(
        worker_gpu_ids, num_workers, num_gpus_per_worker, worker_env
    )

    host_for_url = infer_connect_host(worker_host)
    used_ports: set[int] = set()
    result = WorkerLaunchResult()

    for i in range(num_workers):
        worker = _launch_single_worker(
            index=i,
            model=model,
            num_gpus_per_worker=num_gpus_per_worker,
            worker_host=worker_host,
            worker_base_port=worker_base_port,
            host_for_url=host_for_url,
            worker_env=worker_env,
            cuda_devices=gpu_assignments[i] if gpu_assignments else None,
            worker_extra_args=worker_extra_args,
            used_ports=used_ports,
            master_port_base=master_port_base,
            scheduler_port_base=scheduler_port_base,
            internal_port_stride=internal_port_stride,
            log_prefix=log_prefix,
        )
        result.workers.append(worker)
        result.all_processes.append(worker.process)

    return result


def _launch_single_worker(
    *,
    index: int,
    model: str,
    num_gpus_per_worker: int,
    worker_host: str,
    worker_base_port: int,
    host_for_url: str,
    worker_env: dict[str, str],
    cuda_devices: str | None,
    worker_extra_args: str,
    used_ports: set[int],
    master_port_base: int,
    scheduler_port_base: int,
    internal_port_stride: int,
    log_prefix: str,
) -> LaunchedWorker:
    """Launch a single SGLang Diffusion worker subprocess."""
    preferred_worker_port = worker_base_port + index * 2
    worker_port = reserve_available_port(worker_host, preferred_worker_port, used_ports)

    master_port = reserve_available_port(
        "127.0.0.1",
        master_port_base + index * internal_port_stride,
        used_ports,
    )
    scheduler_port = reserve_available_port(
        "127.0.0.1",
        scheduler_port_base + index * internal_port_stride,
        used_ports,
    )

    cmd = [
        "sglang",
        "serve",
        "--model-path",
        model,
        "--num-gpus",
        str(num_gpus_per_worker),
        "--host",
        worker_host,
        "--port",
        str(worker_port),
        "--master-port",
        str(master_port),
        "--scheduler-port",
        str(scheduler_port),
    ]
    if worker_extra_args:
        cmd += shlex.split(worker_extra_args)

    env = dict(worker_env)
    if cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices

    worker_url = f"http://{host_for_url}:{worker_port}"
    print(
        f"{log_prefix} launching worker {index}: "
        f"{' '.join(shlex.quote(x) for x in cmd)}",
        flush=True,
    )

    proc = subprocess.Popen(
        cmd,
        env=env,
        start_new_session=True,
    )
    return LaunchedWorker(url=worker_url, process=proc)
