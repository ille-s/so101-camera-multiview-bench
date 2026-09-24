"""Reproducible camera-viewpoint ablations for imitation learning on the SO-ARM101."""

# Deliberately no submodule re-exports: importing this package must stay free of
# Isaac Lab / torch so that `import so101_mvbench` works without a simulator.

# pyproject.toml is the single source of the version. Reading it back from the
# installed metadata means the two cannot drift; the fallback covers a source
# tree that was never pip-installed (PYTHONPATH=src), where there is no metadata
# to read and the literal is the best available answer.
try:
    from importlib.metadata import PackageNotFoundError, version as _version

    __version__ = _version("so101-camera-multiview-bench")
except PackageNotFoundError:  # pragma: no cover - uninstalled source tree
    __version__ = "0.1.0"

__all__ = ["__version__"]
