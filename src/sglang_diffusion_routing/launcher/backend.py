"""Abstract base class and shared data types for launcher backends."""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class LaunchedWorker:
    """A worker managed by a launcher backend."""

    url: str
    process: subprocess.Popen


@dataclass
class WorkerLaunchResult:
    """Aggregated result of launching worker subprocesses."""

    workers: list[LaunchedWorker] = field(default_factory=list)
    all_processes: list[subprocess.Popen] = field(default_factory=list)

    @property
    def urls(self) -> list[str]:
        return [w.url for w in self.workers]


class LauncherBackend(ABC):
    """Interface for launching and managing SGLang diffusion workers.

    Each backend exposes the same lifecycle:
    launch → wait_ready_and_register → shutdown.
    """

    @abstractmethod
    def launch(self) -> list[str]:
        """Launch workers and return their base URLs."""

    @abstractmethod
    def wait_ready_and_register(
        self,
        register_func: Callable[[str], None],
        timeout: int,
        log_prefix: str = "[launcher]",
    ) -> None:
        """Wait for workers to become healthy and register each via register_func."""

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up all managed workers."""
