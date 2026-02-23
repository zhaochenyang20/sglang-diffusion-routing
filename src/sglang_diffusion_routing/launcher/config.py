"""YAML configuration loading and backend factory."""

from __future__ import annotations

from pathlib import Path

import yaml
from omegaconf import DictConfig, OmegaConf

from sglang_diffusion_routing.launcher.backend import LauncherBackend
from sglang_diffusion_routing.launcher.local import LocalLauncher, LocalLauncherConfig

SCHEMA_REGISTRY: dict[str, type] = {
    "local": LocalLauncherConfig,
}

BACKEND_REGISTRY: dict[str, type[LauncherBackend]] = {
    "local": LocalLauncher,
}


def load_launcher_config(config_path: str) -> DictConfig:
    """Read a YAML config file and return a validated OmegaConf config.

    1. Parse the YAML and extract the launcher mapping.
    2. Read the backend key to select the structured schema.
    3. Merge the YAML values onto the schema defaults.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open() as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict) or "launcher" not in raw:
        raise ValueError(
            f"Config file must contain a top-level 'launcher' key: {config_path}"
        )

    launcher_raw = raw["launcher"]
    if not isinstance(launcher_raw, dict):
        raise ValueError("'launcher' must be a dictionary")

    backend_name = launcher_raw.get("backend", "local")
    schema_cls = SCHEMA_REGISTRY.get(backend_name)
    if schema_cls is None:
        available = ", ".join(sorted(SCHEMA_REGISTRY))
        raise ValueError(
            f"Unknown launcher backend: {backend_name!r}. "
            f"Available backends: {available}"
        )

    schema = OmegaConf.structured(schema_cls)
    yaml_cfg = OmegaConf.create(launcher_raw)
    merged = OmegaConf.merge(schema, yaml_cfg)
    return merged


def create_backend(config: DictConfig) -> LauncherBackend:
    """Instantiate a LauncherBackend from a validated config."""
    backend_name = config.backend
    cls = BACKEND_REGISTRY.get(backend_name)
    if cls is None:
        available = ", ".join(sorted(BACKEND_REGISTRY))
        raise ValueError(
            f"Unknown launcher backend: {backend_name!r}. "
            f"Available backends: {available}"
        )
    return cls(config)
