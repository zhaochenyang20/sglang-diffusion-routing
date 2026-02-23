"""Tests for the launcher subsystem (config loading, validation, backend factory)."""

from __future__ import annotations

import textwrap
from unittest import mock

import pytest
from omegaconf import OmegaConf

from sglang_diffusion_routing.launcher.config import (
    create_backend,
    load_launcher_config,
)
from sglang_diffusion_routing.launcher.local import LocalLauncher, LocalLauncherConfig


class TestLocalLauncherConfig:
    """OmegaConf schema validation for the local backend."""

    def test_defaults(self):
        cfg = OmegaConf.structured(LocalLauncherConfig(model="my-model"))
        assert cfg.backend == "local"
        assert cfg.model == "my-model"
        assert cfg.num_workers == 1
        assert cfg.num_gpus_per_worker == 1
        assert cfg.worker_host == "127.0.0.1"
        assert cfg.worker_base_port == 10090
        assert cfg.worker_gpu_ids is None
        assert cfg.worker_extra_args == ""
        assert cfg.wait_timeout == 600

    def test_model_is_required(self):
        """Accessing an unset MISSING field should raise."""
        cfg = OmegaConf.structured(LocalLauncherConfig)
        with pytest.raises(Exception):  # MissingMandatoryValue
            _ = cfg.model

    def test_rejects_wrong_type(self):
        """Merging a string into an int field should raise."""
        schema = OmegaConf.structured(LocalLauncherConfig)
        override = OmegaConf.create({"model": "m", "num_workers": "not-an-int"})
        with pytest.raises(Exception):  # ValidationError
            OmegaConf.merge(schema, override)

    def test_rejects_unknown_key(self):
        """Struct mode rejects keys not in the dataclass."""
        schema = OmegaConf.structured(LocalLauncherConfig)
        override = OmegaConf.create({"model": "m", "bogus_key": 123})
        with pytest.raises(Exception):  # ConfigKeyError
            OmegaConf.merge(schema, override)


class TestLoadLauncherConfig:
    """End-to-end YAML loading with OmegaConf validation."""

    def test_minimal_config(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(
            textwrap.dedent(
                """\
                launcher:
                  model: Qwen/Qwen-Image
            """
            )
        )
        cfg = load_launcher_config(str(cfg_file))
        assert cfg.model == "Qwen/Qwen-Image"
        assert cfg.backend == "local"
        assert cfg.num_workers == 1

    def test_override_defaults(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(
            textwrap.dedent(
                """\
                launcher:
                  model: my-model
                  num_workers: 4
                  num_gpus_per_worker: 2
                  worker_base_port: 20000
                  wait_timeout: 300
            """
            )
        )
        cfg = load_launcher_config(str(cfg_file))
        assert cfg.num_workers == 4
        assert cfg.num_gpus_per_worker == 2
        assert cfg.worker_base_port == 20000
        assert cfg.wait_timeout == 300

    def test_missing_model_raises(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(
            textwrap.dedent(
                """\
                launcher:
                  num_workers: 2
            """
            )
        )
        cfg = load_launcher_config(str(cfg_file))
        with pytest.raises(Exception):  # MissingMandatoryValue on access
            _ = cfg.model

    def test_missing_launcher_key_raises(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("something_else:\n  foo: bar\n")
        with pytest.raises(ValueError, match="top-level 'launcher' key"):
            load_launcher_config(str(cfg_file))

    def test_unknown_backend_raises(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(
            textwrap.dedent(
                """\
                launcher:
                  backend: kubernetes
                  model: my-model
            """
            )
        )
        with pytest.raises(ValueError, match="Unknown launcher backend"):
            load_launcher_config(str(cfg_file))

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_launcher_config("/no/such/file.yaml")

    def test_unknown_key_in_yaml_raises(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(
            textwrap.dedent(
                """\
                launcher:
                  model: my-model
                  bogus_key: 123
            """
            )
        )
        with pytest.raises(Exception):  # ConfigKeyError
            load_launcher_config(str(cfg_file))


class TestCreateBackend:
    def test_creates_local_launcher(self):
        cfg = OmegaConf.structured(LocalLauncherConfig(model="my-model"))
        backend = create_backend(cfg)
        assert isinstance(backend, LocalLauncher)

    def test_unknown_backend_raises(self):
        cfg = OmegaConf.structured(
            LocalLauncherConfig(model="m", backend="nonexistent")
        )
        with pytest.raises(ValueError, match="Unknown launcher backend"):
            create_backend(cfg)


class TestLocalLauncherLaunch:
    def test_launch_returns_worker_urls(self):
        cfg = OmegaConf.structured(LocalLauncherConfig(model="my-model", num_workers=2))
        launcher = LocalLauncher(cfg)

        fake_proc = mock.MagicMock()
        fake_proc.pid = 12345

        with (
            mock.patch("shutil.which", return_value="/usr/bin/sglang"),
            mock.patch("subprocess.Popen", return_value=fake_proc),
            mock.patch(
                "sglang_diffusion_routing.launcher.utils.is_port_available",
                return_value=True,
            ),
            mock.patch(
                "sglang_diffusion_routing.launcher.utils.build_gpu_assignments",
                return_value=None,
            ),
        ):
            urls = launcher.launch()

        assert len(urls) == 2
        assert all(u.startswith("http://") for u in urls)

    def test_launch_raises_without_sglang(self):
        cfg = OmegaConf.structured(LocalLauncherConfig(model="my-model"))
        launcher = LocalLauncher(cfg)

        with mock.patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="sglang.*not found"):
                launcher.launch()

    def test_launch_raises_on_empty_model(self):
        cfg = OmegaConf.structured(LocalLauncherConfig(model="  "))
        launcher = LocalLauncher(cfg)

        with pytest.raises(ValueError, match="model.*required"):
            launcher.launch()
