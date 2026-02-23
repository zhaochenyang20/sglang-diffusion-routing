"""Launcher backends for spinning up SGLang diffusion workers.

Supported backends:
- ``local``: launch workers as local subprocesses.
"""

from sglang_diffusion_routing.launcher.backend import (
    LaunchedWorker,
    LauncherBackend,
    WorkerLaunchResult,
)
from sglang_diffusion_routing.launcher.config import create_backend, load_launcher_config
from sglang_diffusion_routing.launcher.local import LocalLauncher, LocalLauncherConfig

__all__ = [
    "LaunchedWorker",
    "LauncherBackend",
    "LocalLauncher",
    "LocalLauncherConfig",
    "WorkerLaunchResult",
    "create_backend",
    "load_launcher_config",
]
