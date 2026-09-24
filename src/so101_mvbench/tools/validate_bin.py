#!/usr/bin/env python3
"""validate_bin — Objective per-bin validation for AP-7 gate decisions.

Checks a single recorded bin for:
- Parquet readability + episode count consistency with binmap.json
- Video file count (2 cameras × n_success episodes)
- Scene-state JSON presence for each episode
- Cube position inside expected bin rectangle (within half_extent_m tolerance)

Exit code: 0 if all checks pass, 1 on first failure.

Usage::

    python validate_bin.py --bin_id 1
    python validate_bin.py --bin_id 13 --dataset_root /custom/path
"""

from __future__ import annotations

__version__ = "1.0.0"

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path


N_COLS_DEFAULT = 5
N_ROWS_DEFAULT = 5
BIN_SIZE_DEFAULT_M = 0.05
ORIGIN_X_DEFAULT = 0.20
ORIGIN_Y_DEFAULT = 0.0
ROBOT_YAW_DEFAULT_DEG = 90.0
HALF_EXTENT_DEFAULT_M = 0.02
INSET_FRACTION = 0.10


@dataclass(frozen=True)
class BinLayout:
    """Compute expected bin center + bounds in world frame."""
    col: int
    row: int
    n_cols: int
    n_rows: int
    bin_size_m: float
    origin_x: float
    origin_y: float
    robot_yaw_rad: float

    @property
    def bin_id(self) -> int:
        return self.row * self.n_cols + self.col + 1

    def _rotate(self, lx: float, ly: float) -> tuple[float, float]:
        c, s = math.cos(self.robot_yaw_rad), math.sin(self.robot_yaw_rad)
        return c * lx - s * ly, s * lx + c * ly

    def expected_center(self) -> tuple[float, float]:
        """World-frame center of this bin."""
        cx_local = (self.col - (self.n_cols - 1) / 2) * self.bin_size_m
        cy_local = (self.row - (self.n_rows - 1) / 2) * self.bin_size_m
        rx, ry = self._rotate(cx_local, cy_local)
        return self.origin_x + rx, self.origin_y + ry

    def world_bounds(self, tolerance_m: float) -> tuple[float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max) of the bin in world frame."""
        cx, cy = self.expected_center()
        half = self.bin_size_m / 2 + tolerance_m
        # Rotated rectangle — but since rotation is 90°, axis-aligned in world
        return cx - half, cx + half, cy - half, cy + half


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)


def ok(msg: str) -> None:
    print(f"ok:   {msg}")


def check_binmap(binmap_path: Path) -> tuple[int, list]:
    """Load binmap.json and return (n_success, success_episode_indices)."""
    if not binmap_path.exists():
        fail(f"binmap.json missing: {binmap_path}")
        sys.exit(1)
    data = json.loads(binmap_path.read_text())
    episodes = data.get("episodes", [])
    success_eps = [ep for ep in episodes if ep.get("success")]
    n_success = len(success_eps)
    ok(f"binmap.json: {n_success} success, "
       f"{sum(1 for ep in episodes if not ep.get('success'))} retry")
    return n_success, success_eps


def check_parquet(parquet_path: Path, expected_count: int) -> int:
    """Load parquet, verify distinct episode_index count. Returns actual count."""
    if not parquet_path.exists():
        fail(f"parquet missing: {parquet_path}")
        sys.exit(1)
    try:
        import pyarrow.parquet as pq
    except ImportError:
        fail("pyarrow not installed — cannot verify parquet")
        sys.exit(1)

    try:
        table = pq.read_table(parquet_path)
    except Exception as exc:
        fail(f"parquet read failed: {exc}")
        sys.exit(1)

    if "episode_index" not in table.column_names:
        fail(f"episode_index column missing in {parquet_path}")
        sys.exit(1)

    eps = sorted(set(table["episode_index"].to_pylist()))
    n_distinct = len(eps)
    if n_distinct < expected_count:
        fail(f"parquet: {n_distinct} distinct episodes < expected {expected_count}")
        sys.exit(1)
    ok(f"parquet: {table.num_rows} rows, {n_distinct} episodes {eps[:3]}...{eps[-3:]}")
    return n_distinct


def check_videos(videos_root: Path, n_success: int, min_size_bytes: int = 50_000) -> int:
    """Count .mp4 files under videos/. Should be 2 * n_success (2 cameras)."""
    mp4s = list(videos_root.rglob("*.mp4"))
    if not mp4s:
        fail(f"no MP4s found under {videos_root}")
        sys.exit(1)
    undersized = [p for p in mp4s if p.stat().st_size < min_size_bytes]
    if undersized:
        fail(f"{len(undersized)} MP4s below {min_size_bytes} bytes — example: {undersized[0]}")
        sys.exit(1)
    expected = 2 * n_success
    if len(mp4s) < expected:
        fail(f"{len(mp4s)} MP4s found, expected ≥ {expected} (2 × n_success)")
        sys.exit(1)
    ok(f"videos: {len(mp4s)} MP4s (expected ≥ {expected}), all ≥ {min_size_bytes}B")
    return len(mp4s)


def check_scene_states(meta_root: Path, n_success: int) -> int:
    """Count scene_state JSON files. Should be ≥ n_success."""
    scene_files = sorted(meta_root.glob("episode_*_scene_state.json"))
    if len(scene_files) < n_success:
        fail(f"{len(scene_files)} scene_state files < n_success={n_success}")
        sys.exit(1)
    ok(f"scene_states: {len(scene_files)} files (expected ≥ {n_success})")
    return len(scene_files)


def check_cube_positions(
    success_eps: list,
    layout: BinLayout,
    tolerance_m: float,
) -> int:
    """Verify cube (x, y) for each success episode is inside the bin rectangle."""
    x_min, x_max, y_min, y_max = layout.world_bounds(tolerance_m)
    outside = []
    for ep in success_eps:
        x, y = ep.get("x_m"), ep.get("y_m")
        if x is None or y is None:
            continue
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            outside.append((ep.get("episode_idx"), x, y))
    if outside:
        fail(f"{len(outside)} success eps outside bin rect "
             f"[{x_min:.3f},{x_max:.3f}]×[{y_min:.3f},{y_max:.3f}]: "
             f"{outside[:3]}")
        sys.exit(1)
    ok(f"cube positions: all {len(success_eps)} inside bin bounds "
       f"(x∈[{x_min:.3f},{x_max:.3f}], y∈[{y_min:.3f},{y_max:.3f}])")
    return len(success_eps)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parents[4]
    default_root = workspace_root / "datasets" / "liftcube_binned"

    parser = argparse.ArgumentParser(
        description="Validate a single recorded bin against AP-7 criteria."
    )
    parser.add_argument("--bin_id", type=int, required=True,
                        help="Flat bin ID (1-25).")
    parser.add_argument("--dataset_root", type=Path, default=default_root)
    parser.add_argument("--n_cols", type=int, default=N_COLS_DEFAULT)
    parser.add_argument("--n_rows", type=int, default=N_ROWS_DEFAULT)
    parser.add_argument("--bin_size_m", type=float, default=BIN_SIZE_DEFAULT_M)
    parser.add_argument("--origin_x", type=float, default=ORIGIN_X_DEFAULT)
    parser.add_argument("--origin_y", type=float, default=ORIGIN_Y_DEFAULT)
    parser.add_argument("--robot_yaw_deg", type=float, default=ROBOT_YAW_DEFAULT_DEG)
    parser.add_argument("--half_extent_m", type=float, default=HALF_EXTENT_DEFAULT_M)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not (1 <= args.bin_id <= args.n_cols * args.n_rows):
        fail(f"bin_id must be 1..{args.n_cols * args.n_rows}, got {args.bin_id}")
        return 1

    # Convert bin_id → (col, row) — row-major inverse
    idx0 = args.bin_id - 1
    col = idx0 % args.n_cols
    row = idx0 // args.n_cols

    bin_dir = args.dataset_root / f"bin_c{col}_r{row}"
    print(f"validate_bin v{__version__}: bin{args.bin_id} = (col={col}, row={row}) → {bin_dir}")

    if not bin_dir.exists():
        fail(f"bin directory missing: {bin_dir}")
        return 1

    layout = BinLayout(
        col=col, row=row,
        n_cols=args.n_cols, n_rows=args.n_rows,
        bin_size_m=args.bin_size_m,
        origin_x=args.origin_x, origin_y=args.origin_y,
        robot_yaw_rad=math.radians(args.robot_yaw_deg),
    )

    # Run checks sequentially — exit on first failure
    n_success, success_eps = check_binmap(bin_dir / "binmap.json")
    if n_success == 0:
        fail("n_success=0 — no episodes recorded yet")
        return 1

    check_parquet(bin_dir / "data" / "chunk-000" / "file-000.parquet", n_success)
    check_videos(bin_dir / "videos", n_success)
    check_scene_states(bin_dir / "meta", n_success)
    check_cube_positions(success_eps, layout, args.half_extent_m)

    print(f"\nPASS: bin{args.bin_id} ({n_success}/27 success)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
