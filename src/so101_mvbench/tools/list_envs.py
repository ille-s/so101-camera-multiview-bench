# SPDX-License-Identifier: MIT
"""List all gym-registered Isaac Lab environments from this package.

Usage:
    list_envs

Filters to envs whose ID starts with "Lerobot-" or "LiftCube-" (this package's
namespace), so upstream Isaac Lab task IDs are not shown. Output is sorted
alphabetically by task ID for stable, reviewer-reproducible diffs.

Adapted from NVIDIA Sim-to-Real-SO-101-Workshop scripts/list_envs.py.
"""

__version__ = "1.0.0"

from isaaclab.app import AppLauncher

# Isaac Sim must be launched before importing isaaclab.envs / gym tasks.
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

import so101_mvbench.tasks  # noqa: F401,E402 — triggers gym.register


_PACKAGE_PREFIXES = ("Lerobot-", "LiftCube-")


def _collect_rows() -> list[tuple[int, str, str]]:
    """Sorted list of (idx, task_id, env_cfg_entry_point) for our package's envs."""
    specs = sorted(gym.registry.values(), key=lambda s: s.id)
    rows: list[tuple[int, str, str]] = []
    for task_spec in specs:
        if any(task_spec.id.startswith(p) for p in _PACKAGE_PREFIXES):
            cfg = task_spec.kwargs.get("env_cfg_entry_point", "")
            rows.append((len(rows) + 1, task_spec.id, str(cfg)))
    return rows


def main() -> None:
    """Print this package's gym-registered envs as a plain table.

    Uses ``os._exit(0)`` instead of ``simulation_app.close()`` because
    Isaac Sim 6 hangs on ``close()`` (see
    ``.claude/memory/reference_isaac_sim6_shutdown.md``). Without the hard
    exit the printed table never flushes before the hang.
    """
    import os
    import sys

    rows = _collect_rows()

    if not rows:
        print("No envs registered with prefixes:", ", ".join(_PACKAGE_PREFIXES))
    else:
        id_w = max(len(r[1]) for r in rows)
        cfg_w = max(len(r[2]) for r in rows)

        print(f"{'#':>3}  {'Task ID':<{id_w}}  {'env_cfg_entry_point':<{cfg_w}}")
        print(f"{'-'*3:>3}  {'-'*id_w:<{id_w}}  {'-'*cfg_w:<{cfg_w}}")
        for idx, task_id, cfg in rows:
            print(f"{idx:>3}  {task_id:<{id_w}}  {cfg:<{cfg_w}}")

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
