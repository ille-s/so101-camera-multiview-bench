# SPDX-License-Identifier: MIT
"""Parse CLI ``--episodes`` range-syntax into a sorted ``list[int]``.

Supported forms:

- ``'0'``         single ID
- ``'0,1,5'``     comma-list
- ``'0-10'``      inclusive range (lo ≤ hi)
- ``'0,5-10,15'`` mixed comma + range
- ``'all'``       auto-discover from ``<dataset_root>/trajectories/episode_*.npy``

The function is intentionally pure (no Isaac Sim deps) so it can be unit-tested
in isolation. Shared across console-scripts that operate on a subset of episodes
(``synthetic_multicam_recorder``, future tools).
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["parse_episodes_spec"]


def parse_episodes_spec(spec: str, dataset_root: Path) -> list[int]:
    """Parse a ``--episodes`` spec string into a sorted list of integer IDs.

    Args:
        spec: One of: ``'all'``, ``'<n>'``, ``'<n>,<m>'``, ``'<lo>-<hi>'``, or
            any comma-separated mix of singles + inclusive ranges. Whitespace
            around tokens is ignored. Case-insensitive for ``'all'``.
        dataset_root: Source LeRobot dataset directory. Used only when
            ``spec == 'all'`` — globs ``<dataset_root>/trajectories/episode_*.npy``.

    Returns:
        Sorted list of unique integer episode IDs. Duplicates within the
        spec are deduplicated.

    Raises:
        ValueError: If ``spec`` is empty/whitespace, malformed, or contains a
            reversed range (``hi < lo``). Also if ``spec == 'all'`` but the
            ``trajectories/`` directory has no matching files.
    """
    if spec is None:
        raise ValueError("episodes spec is None")
    spec = spec.strip()
    if not spec:
        raise ValueError("empty episodes spec")

    if spec.lower() == "all":
        traj_dir = dataset_root / "trajectories"
        ids = sorted(
            int(p.stem.split("_")[-1])
            for p in traj_dir.glob("episode_*.npy")
        )
        if not ids:
            raise ValueError(
                f"--episodes='all' but no trajectories/episode_*.npy in {traj_dir}"
            )
        return ids

    result: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            # Tolerate leading/trailing/duplicate commas (",0", "0,,5", "0,")
            continue
        if "-" in part:
            try:
                lo_s, hi_s = part.split("-", 1)
                lo = int(lo_s)
                hi = int(hi_s)
            except ValueError as e:
                raise ValueError(
                    f"malformed range token {part!r} (expected '<lo>-<hi>')"
                ) from e
            if hi < lo:
                raise ValueError(
                    f"reversed range {part!r}: hi ({hi}) < lo ({lo})"
                )
            result.update(range(lo, hi + 1))
        else:
            try:
                result.add(int(part))
            except ValueError as e:
                raise ValueError(
                    f"malformed integer token {part!r}"
                ) from e
    if not result:
        raise ValueError(f"empty episodes spec after parsing: {spec!r}")
    return sorted(result)
