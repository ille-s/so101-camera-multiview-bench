"""Runtime settings: where user data lives.

Replaces the thesis workspace's ``ws_utils.get_workspace_root()``. A public user
has no "workspace" -- they have a data directory they choose. Everything the
package *ships* (USD assets, scene configs, eval.json) is resolved relative to
``__file__`` instead and never goes through here.

Usage::

    from so101_mvbench.settings import SETTINGS

    root = SETTINGS.require_datasets_root()      # fails loud with the env var name
    out  = SETTINGS.outputs_root / "my_run"

Override per process::

    SO101_BENCH_DATA_ROOT=/mnt/data so101-train ...

Override in a test::

    Settings.from_env(data_root=tmp_path)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__version__ = "1.0.0"

ENV_DATA_ROOT = "SO101_BENCH_DATA_ROOT"
ENV_DATASETS = "SO101_BENCH_DATASETS"
ENV_OUTPUTS = "SO101_BENCH_OUTPUTS"
ENV_LOG_FILE = "SO101_BENCH_LOG_FILE"


@dataclass(frozen=True, slots=True)
class Settings:
    """Where user data lives. Immutable; build a new one to change it."""

    data_root: Path
    datasets_root: Path
    outputs_root: Path
    log_file: Path | None

    @classmethod
    def from_env(cls, **overrides: Path | str | None) -> Settings:
        """Build from environment variables, with keyword overrides taking priority."""

        def pick(key: str, env: str, default: Path | None) -> Path | None:
            if key in overrides:
                val = overrides[key]
                return Path(val) if val is not None else None
            raw = os.environ.get(env)
            return Path(raw).expanduser() if raw else default

        data_root = pick("data_root", ENV_DATA_ROOT, Path.cwd())
        assert data_root is not None  # data_root has a non-None default
        return cls(
            data_root=data_root,
            datasets_root=pick("datasets_root", ENV_DATASETS, data_root / "datasets"),
            outputs_root=pick("outputs_root", ENV_OUTPUTS, data_root / "outputs"),
            log_file=pick("log_file", ENV_LOG_FILE, None),
        )

    def require_datasets_root(self) -> Path:
        """Return ``datasets_root``, or fail with the env var the user needs to set."""
        if not self.datasets_root.is_dir():
            raise FileNotFoundError(
                f"datasets root not found: {self.datasets_root}\n"
                f"Set {ENV_DATASETS} (or {ENV_DATA_ROOT}) to point at your data directory."
            )
        return self.datasets_root

    def require_outputs_root(self) -> Path:
        """Return ``outputs_root``, creating it if absent (training writes here)."""
        self.outputs_root.mkdir(parents=True, exist_ok=True)
        return self.outputs_root


SETTINGS: Settings = Settings.from_env()


if __name__ == "__main__":  # ponytail: inline self-check, `python -m so101_mvbench.settings`
    s = Settings.from_env(data_root="/tmp/x")
    assert s.datasets_root == Path("/tmp/x/datasets"), s
    assert s.outputs_root == Path("/tmp/x/outputs"), s
    assert s.log_file is None
    s2 = Settings.from_env(data_root="/tmp/x", datasets_root="/elsewhere")
    assert s2.datasets_root == Path("/elsewhere"), s2
    try:
        Settings.from_env(data_root="/nonexistent").require_datasets_root()
    except FileNotFoundError as exc:
        assert ENV_DATASETS in str(exc)
    else:
        raise AssertionError("require_datasets_root must fail loud on a missing dir")
    print("OK settings")
