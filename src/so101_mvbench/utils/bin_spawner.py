"""Deterministic bin-based stratified cube spawner for recording pipeline.

Divides the robot workspace into an n_cols × n_rows grid of bins. Within
each bin, a 3×3 sub-grid of deterministic positions is generated (corners,
edge midpoints, center). At each position, 3 fixed yaw rotations are
applied (0°, 22.5°, 45°). Episodes are scheduled block-wise: all episodes
for one bin complete before advancing to the next.

Adapted from ``leisaac_experiments.scene.sampling`` with simplifications:
- No ObjectRegistry dependency — ``half_extent_m`` passed directly.
- No ``load_from_hdf5()`` — resume exclusively via BinMap JSON sidecar.
- Deterministic sub-grid positions instead of least-visited-first random.
- Pre-computed schedule with cursor-based traversal.

Designed to be Isaac-Sim-free: no isaaclab imports at module level.
Integration with Isaac Lab happens via post-reset teleport in
``policy_recorder._apply_spawn_config()``.

BinMap / BinEpisodeRecord provide offline-persistable episode→bin tracking:
each recording session saves a JSON sidecar (<dataset>_binmap.json) that
maps every episode index to its bin coordinates, pose, and success flag.
"""

__version__ = "1.0.0"

import dataclasses
import json
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import ClassVar

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 2D yaw rotation helper (pure Python, no quaternion ambiguity)
# ------------------------------------------------------------------

def _rotate_yaw(
    yaw_rad: float,
    lx: float,
    ly: float,
) -> tuple[float, float]:
    """Rotate a 2D point (lx, ly) by *yaw_rad* around Z.

    Standard 2D rotation matrix::

        [ cos(yaw)  -sin(yaw) ] [ lx ]
        [ sin(yaw)   cos(yaw) ] [ ly ]

    Args:
        yaw_rad: Yaw angle in radians (positive = counter-clockwise).
        lx: Local x offset.
        ly: Local y offset.

    Returns:
        (world_dx, world_dy) — the rotated offsets.
    """
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return (c * lx - s * ly, s * lx + c * ly)


# ------------------------------------------------------------------
# WorkspaceBounds
# ------------------------------------------------------------------


@dataclass
class WorkspaceBounds:
    """Axis-aligned bounding box of the robot's reachable workspace.

    Min/max values are LOCAL — relative to (origin_x, origin_y).
    BinSpawner converts to world coordinates as: world = origin + local.

    This allows the same workspace shape to be re-used across different
    table/arm positions simply by changing origin_x/origin_y.

    Attributes:
        x_min_m: Minimum x bound relative to origin (meters).
        x_max_m: Maximum x bound relative to origin (meters).
        y_min_m: Minimum y bound relative to origin (meters).
        y_max_m: Maximum y bound relative to origin (meters).
        origin_x: World-frame x coordinate of the workspace origin (meters).
        origin_y: World-frame y coordinate of the workspace origin (meters).
        yaw_min_rad: Minimum yaw angle (radians). Default: -π.
        yaw_max_rad: Maximum yaw angle (radians). Default: +π.
    """

    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    origin_x: float = 0.0
    origin_y: float = 0.0
    yaw_min_rad: float = -math.pi
    yaw_max_rad: float = math.pi

    @classmethod
    def from_center(
        cls,
        center_x: float,
        center_y: float,
        n_cols: int,
        n_rows: int,
        bin_size: float,
        **kwargs,
    ) -> "WorkspaceBounds":
        """Create bounds from grid center, dimensions, and bin size.

        Computes x/y extents as center ± (n × bin_size) / 2. Any extra
        keyword arguments (origin_x, origin_y, yaw_*) are forwarded to
        the constructor.

        Args:
            center_x: Grid center x in local frame (meters).
            center_y: Grid center y in local frame (meters).
            n_cols: Number of grid columns.
            n_rows: Number of grid rows.
            bin_size: Size of each bin in meters.
            **kwargs: Forwarded to WorkspaceBounds (e.g. origin_x, origin_y).

        Returns:
            WorkspaceBounds with computed x/y limits.
        """
        half_x = (n_cols * bin_size) / 2.0
        half_y = (n_rows * bin_size) / 2.0
        return cls(
            x_min_m=center_x - half_x,
            x_max_m=center_x + half_x,
            y_min_m=center_y - half_y,
            y_max_m=center_y + half_y,
            **kwargs,
        )

    def safe_bounds(self, half_extent_m: float) -> "WorkspaceBounds":
        """Return inset bounds keeping an object fully within workspace.

        Shrinks x/y bounds by half_extent_m on each side. Origin and yaw
        are preserved unchanged.

        Args:
            half_extent_m: Object half-extent in meters.

        Returns:
            New WorkspaceBounds with inset x/y limits.

        Raises:
            ValueError: If margin makes bounds degenerate (max <= min).
        """
        x_min = self.x_min_m + half_extent_m
        x_max = self.x_max_m - half_extent_m
        y_min = self.y_min_m + half_extent_m
        y_max = self.y_max_m - half_extent_m
        if x_max <= x_min or y_max <= y_min:
            raise ValueError(
                f"half_extent_m={half_extent_m:.4f} is too large for workspace "
                f"({self.x_max_m - self.x_min_m:.3f} x "
                f"{self.y_max_m - self.y_min_m:.3f} m)."
            )
        return WorkspaceBounds(
            x_min_m=x_min,
            x_max_m=x_max,
            y_min_m=y_min,
            y_max_m=y_max,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            yaw_min_rad=self.yaw_min_rad,
            yaw_max_rad=self.yaw_max_rad,
        )


# ------------------------------------------------------------------
# SpawnConfig
# ------------------------------------------------------------------


@dataclass
class SpawnConfig:
    """Concrete spawn configuration for a single episode.

    All coordinates are in the **world frame** (origin offset applied).

    Attributes:
        object_key: Registry key of the spawned object (e.g. "cube").
        x_m: World-frame x position (meters).
        y_m: World-frame y position (meters).
        yaw_rad: Object yaw rotation (radians).
        bin_col: Grid column index of the containing bin.
        bin_row: Grid row index of the containing bin.
    """

    object_key: str
    x_m: float
    y_m: float
    yaw_rad: float
    bin_col: int
    bin_row: int

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SpawnConfig":
        """Deserialize from dict."""
        return cls(**d)


# ------------------------------------------------------------------
# BinSpawner — deterministic stratified schedule
# ------------------------------------------------------------------


class BinSpawner:
    """Deterministic bin-based stratified cube spawner.

    Pre-computes a schedule of all episodes at construction time:

    - For each trainable bin (block-wise): generate 3×3 sub-grid positions.
    - For each position: apply the 3 fixed yaw rotations in ``ROTATIONS_RAD``
      (0°, 22.5°, 45°).
    - Episodes are traversed via cursor: N advances, R retries.

    All returned coordinates are in the **world frame**:
    ``world = workspace.origin + rotate(local, robot_yaw)``

    Args:
        workspace: Physical bounds of the spawn region.
        n_cols: Number of grid columns.
        n_rows: Number of grid rows.
        half_extent_m: Object half-extent for safe_bounds margin.
        object_key: Object key stored in SpawnConfig (default "cube").
        robot_yaw_rad: Robot base yaw angle in radians for rotating
            local workspace offsets into world frame. Default 0.0.
        ood_bins: Set of (col, row) tuples to exclude from sampling.
    """

    ROTATIONS_RAD: ClassVar[list[float]] = [
        0.0,
        math.pi / 8,      # 22.5°
        math.pi / 4,      # 45°
    ]

    def __init__(
        self,
        workspace: WorkspaceBounds,
        n_cols: int,
        n_rows: int,
        half_extent_m: float,
        object_key: str = "cube",
        robot_yaw_rad: float = 0.0,
        ood_bins: frozenset[tuple[int, int]] = frozenset(),
        selected_bin: tuple[int, int] | None = None,
    ) -> None:
        assert n_cols > 0 and n_rows > 0, "Grid dimensions must be positive."
        for col, row in ood_bins:
            if not (0 <= col < n_cols and 0 <= row < n_rows):
                raise ValueError(
                    f"OOD bin ({col},{row}) out of range for "
                    f"{n_cols}×{n_rows} grid."
                )
        if selected_bin is not None:
            c, r = selected_bin
            if not (0 <= c < n_cols and 0 <= r < n_rows):
                raise ValueError(
                    f"Selected bin ({c},{r}) out of range for "
                    f"{n_cols}×{n_rows} grid."
                )
            if selected_bin in ood_bins:
                raise ValueError(f"Selected bin {selected_bin} is OOD.")
        self.workspace = workspace
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.half_extent_m = half_extent_m
        self.object_key = object_key
        self.robot_yaw_rad = robot_yaw_rad
        self.ood_bins = ood_bins
        self.selected_bin = selected_bin

        self._schedule: list[SpawnConfig] = self._build_schedule()
        self._cursor: int = 0
        self._current_config: SpawnConfig | None = None
        self._advance: bool = True

    # --- Grid geometry ---

    def _bin_bounds_local(
        self, col: int, row: int
    ) -> tuple[float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max) in LOCAL frame for a bin.

        Args:
            col: Column index.
            row: Row index.

        Returns:
            4-tuple of bin boundaries in local meters (relative to origin).
        """
        ws = self.workspace
        x_step = (ws.x_max_m - ws.x_min_m) / self.n_cols
        y_step = (ws.y_max_m - ws.y_min_m) / self.n_rows
        return (
            ws.x_min_m + col * x_step,
            ws.x_min_m + (col + 1) * x_step,
            ws.y_min_m + row * y_step,
            ws.y_min_m + (row + 1) * y_step,
        )

    # Fraction of bin size to inset sub-grid from bin edges.
    # Prevents positions from overlapping with neighboring bins.
    _INSET_FRACTION: float = 0.1

    def _subgrid_positions(
        self, col: int, row: int
    ) -> list[tuple[float, float]]:
        """Generate 3×3 deterministic sub-grid within a bin.

        Positions are inset slightly from bin edges (10% margin) so that
        each bin has exclusive positions with no overlap to neighbors.
        The center position (P4) is unchanged.

        Args:
            col: Column index.
            row: Row index.

        Returns:
            List of 9 (local_x, local_y) positions.
        """
        x_min, x_max, y_min, y_max = self._bin_bounds_local(col, row)
        margin_x = (x_max - x_min) * self._INSET_FRACTION
        margin_y = (y_max - y_min) * self._INSET_FRACTION
        x0 = x_min + margin_x
        x1 = (x_min + x_max) / 2.0
        x2 = x_max - margin_x
        y0 = y_min + margin_y
        y1 = (y_min + y_max) / 2.0
        y2 = y_max - margin_y
        return [
            (x0, y0), (x1, y0), (x2, y0),
            (x0, y1), (x1, y1), (x2, y1),
            (x0, y2), (x1, y2), (x2, y2),
        ]

    def _bin_order(self) -> list[tuple[int, int]]:
        """Return trainable bins in row-major order, skipping OOD.

        If selected_bin is set, returns only that bin.

        Returns:
            Ordered list of (col, row) tuples.
        """
        if self.selected_bin is not None:
            return [self.selected_bin]
        return [
            (c, r)
            for r in range(self.n_rows)
            for c in range(self.n_cols)
            if (c, r) not in self.ood_bins
        ]

    # --- Schedule ---

    def _build_schedule(self) -> list[SpawnConfig]:
        """Pre-compute all episodes: bin-wise → position-wise → rotation-wise.

        Returns:
            Flat list of SpawnConfig in execution order.
        """
        schedule: list[SpawnConfig] = []
        for col, row in self._bin_order():
            for local_x, local_y in self._subgrid_positions(col, row):
                rot_x, rot_y = _rotate_yaw(
                    self.robot_yaw_rad, local_x, local_y
                )
                world_x = self.workspace.origin_x + rot_x
                world_y = self.workspace.origin_y + rot_y
                for yaw in self.ROTATIONS_RAD:
                    schedule.append(SpawnConfig(
                        object_key=self.object_key,
                        x_m=world_x,
                        y_m=world_y,
                        yaw_rad=yaw,
                        bin_col=col,
                        bin_row=row,
                    ))
        return schedule

    # --- Cursor-based traversal ---

    def next_config(self) -> SpawnConfig:
        """Return the current spawn config, advancing if flagged.

        Called by _apply_spawn_config() after each env.reset(). Respects
        the _advance flag set by mark_success() / mark_retry().

        Returns:
            SpawnConfig for the current episode.

        Raises:
            StopIteration: If all episodes have been recorded.
        """
        if self._current_config is None or self._advance:
            if self._cursor >= len(self._schedule):
                raise StopIteration(
                    f"All {len(self._schedule)} episodes recorded."
                )
            self._current_config = self._schedule[self._cursor]
            self._advance = False
        return self._current_config

    def mark_success(self) -> None:
        """Mark current episode as success and advance cursor.

        Called on N-key (success) — next env.reset() will pick
        the next episode in the schedule.
        """
        self._cursor += 1
        self._advance = True

    def mark_retry(self) -> None:
        """Mark current episode for retry (cursor stays).

        Called on R-key (retry) — next env.reset() will repeat
        the same position and rotation.
        """
        # Intentionally empty: cursor stays, _advance stays False.
        pass

    def restore_from_binmap(self, bin_map: "BinMap") -> None:
        """Restore cursor position from a persisted BinMap.

        Sets cursor to the number of successful episodes, which
        corresponds to the next unrecorded position in the schedule.

        Args:
            bin_map: BinMap loaded from a JSON sidecar file.

        Raises:
            ValueError: If bin_map grid dimensions differ from this spawner.
        """
        if bin_map.n_cols != self.n_cols or bin_map.n_rows != self.n_rows:
            raise ValueError(
                f"BinMap grid ({bin_map.n_cols}×{bin_map.n_rows}) does not "
                f"match spawner grid ({self.n_cols}×{self.n_rows})."
            )
        n_success = sum(1 for ep in bin_map.episodes if ep.success)
        self._cursor = n_success
        logger.info(
            "Restored cursor to %d/%d from BinMap (%d successful episodes).",
            self._cursor, len(self._schedule), n_success,
        )

    # --- Properties ---

    @property
    def total_episodes(self) -> int:
        """Total number of episodes in the schedule."""
        return len(self._schedule)

    @property
    def remaining(self) -> int:
        """Number of episodes not yet recorded."""
        return max(0, self.total_episodes - self._cursor)

    @property
    def n_trainable_bins(self) -> int:
        """Number of bins available for sampling (excluding OOD)."""
        return self.n_cols * self.n_rows - len(self.ood_bins)

    @property
    def positions_per_bin(self) -> int:
        """Number of sub-grid positions per bin (always 9)."""
        return 9

    @property
    def episodes_per_bin(self) -> int:
        """Number of episodes per bin (positions × rotations)."""
        return self.positions_per_bin * len(self.ROTATIONS_RAD)

    @property
    def current_bin(self) -> tuple[int, int] | None:
        """(col, row) of the currently active bin, or None."""
        if self._current_config is None:
            return None
        return (self._current_config.bin_col, self._current_config.bin_row)


# ------------------------------------------------------------------
# BinMap — offline-persistable episode→bin index
# ------------------------------------------------------------------


class BinState(Enum):
    """Visual state of a single bin cell for display purposes."""

    UNVISITED = "unvisited"
    VISITED = "visited"
    ACTIVE = "active"
    OOD = "ood"


@dataclass
class BinEpisodeRecord:
    """Single episode entry in the BinMap index.

    Attributes:
        episode_idx: Dataset episode index.
        bin_col: Grid column of the spawned bin.
        bin_row: Grid row of the spawned bin.
        x_m: World-frame spawn x-coordinate (meters).
        y_m: World-frame spawn y-coordinate (meters).
        yaw_rad: Spawn yaw angle (radians).
        success: True if the episode was marked successful (N key).
    """

    episode_idx: int
    bin_col: int
    bin_row: int
    x_m: float
    y_m: float
    yaw_rad: float
    success: bool


@dataclass
class BinMap:
    """Offline-persistable index mapping episode indices to bin coordinates.

    Saved as a JSON sidecar file (<dataset>_binmap.json) alongside the
    dataset. Enables:

    - **Resume after crash:** reconstruct BinSpawner cursor from
      the count of successful episodes.
    - **Offline analysis:** inspect per-bin coverage, identify
      under-sampled bins, filter episodes by bin.

    Attributes:
        n_cols: Number of grid columns.
        n_rows: Number of grid rows.
        object_key: Active object key (matches BinSpawner.object_key).
        episodes: Ordered list of episode records (append-only).
        ood_bins: Set of (col, row) tuples excluded from sampling.
    """

    n_cols: int
    n_rows: int
    object_key: str
    episodes: list[BinEpisodeRecord] = field(default_factory=list)
    ood_bins: frozenset[tuple[int, int]] = field(default_factory=frozenset)
    workspace_origin_m: tuple[float, float, float] | None = None
    """Workspace-frame {W} origin of the scene the episodes were RECORDED in,
    expressed in that scene's world frame. Declares which frame the episode
    x_m/y_m coordinates live in, so a replay into a different scene can derive
    its rigid shift. ``None`` on datasets recorded before this field existed,
    which means the floor-level table scene (``HEMISPHERE_ORIGIN_W``)."""

    # --- Persistence ---

    def save(self, path: Path) -> None:
        """Save BinMap to a JSON sidecar file.

        Args:
            path: Destination file path.
        """
        payload = {
            "n_cols": self.n_cols,
            "n_rows": self.n_rows,
            "object_key": self.object_key,
            "ood_bins": sorted([c, r] for c, r in self.ood_bins),
            "episodes": [dataclasses.asdict(ep) for ep in self.episodes],
        }
        if self.workspace_origin_m is not None:
            payload["workspace_origin_m"] = list(self.workspace_origin_m)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.debug("BinMap saved to %s (%d episodes).", path, len(self.episodes))

    def get_episode_config(self, episode_idx: int) -> "SpawnConfig | None":
        """Lookup SpawnConfig for a specific dataset episode index.

        Args:
            episode_idx: Dataset episode index (matches BinEpisodeRecord.episode_idx).

        Returns:
            SpawnConfig reconstructed from the BinMap entry, or None if the
            episode_idx is not present. If multiple records share the same
            episode_idx (retries before success), returns the last occurrence.
        """
        rec = None
        for candidate in self.episodes:
            if candidate.episode_idx == episode_idx:
                rec = candidate
        if rec is None:
            return None
        return SpawnConfig(
            object_key=self.object_key,
            x_m=rec.x_m,
            y_m=rec.y_m,
            yaw_rad=rec.yaw_rad,
            bin_col=rec.bin_col,
            bin_row=rec.bin_row,
        )

    @classmethod
    def load(cls, path: Path) -> "BinMap":
        """Load BinMap from a JSON sidecar file.

        Args:
            path: Path to an existing ``_binmap.json`` file.

        Returns:
            Reconstructed BinMap instance.

        Raises:
            FileNotFoundError: If path does not exist.
        """
        with open(path) as f:
            payload = json.load(f)
        episodes = [BinEpisodeRecord(**ep) for ep in payload.get("episodes", [])]
        ood_bins = frozenset(
            tuple(pair) for pair in payload.get("ood_bins", [])
        )
        ws_origin = payload.get("workspace_origin_m")
        instance = cls(
            n_cols=payload["n_cols"],
            n_rows=payload["n_rows"],
            object_key=payload["object_key"],
            episodes=episodes,
            ood_bins=ood_bins,
            workspace_origin_m=tuple(ws_origin) if ws_origin is not None else None,
        )
        logger.info(
            "BinMap loaded from %s (%d episodes).", path, len(instance.episodes)
        )
        return instance

    # --- Episode tracking ---

    def log_episode(
        self, episode_idx: int, cfg: SpawnConfig, success: bool
    ) -> None:
        """Append an episode record to the index.

        Args:
            episode_idx: Dataset episode index.
            cfg: SpawnConfig used for this episode.
            success: True if the episode was accepted (N key).
        """
        self.episodes.append(
            BinEpisodeRecord(
                episode_idx=episode_idx,
                bin_col=cfg.bin_col,
                bin_row=cfg.bin_row,
                x_m=cfg.x_m,
                y_m=cfg.y_m,
                yaw_rad=cfg.yaw_rad,
                success=success,
            )
        )

    def visit_counts_grid(self) -> list[list[int]]:
        """Return n_rows × n_cols grid of successful episode counts.

        Returns:
            2-D list indexed as ``grid[row][col]``, counting only
            episodes where ``success=True``.
        """
        grid = [[0] * self.n_cols for _ in range(self.n_rows)]
        for ep in self.episodes:
            if ep.success:
                grid[ep.bin_row][ep.bin_col] += 1
        return grid

    # --- Rendering ---

    def render_ascii(self, active_col: int = -1, active_row: int = -1) -> str:
        """Render bin coverage as ASCII grid matching the robot's view.

        The grid is rotated 90° CW from local frame to match what the
        operator sees looking at the robot from behind (robot_yaw=90°):
        - Columns run left→right = World +Y → -Y
        - Rows run top→bottom = World +X (far) → -X (near robot)

        Cell format:
            ``[   ]`` — unvisited (no successful episode)
            ``[  N]`` — visited (N = number of successful episodes)
            ``[  *]`` — active (currently selected bin)
            ``[OOD]`` — out-of-distribution (excluded from sampling)

        Args:
            active_col: Column index of the currently active bin (-1 = none).
            active_row: Row index of the currently active bin (-1 = none).

        Returns:
            Multi-line string ready for ``print()``.
        """
        counts = self.visit_counts_grid()

        def _cell(col: int, row: int) -> str:
            if (col, row) in self.ood_bins:
                return "[OOD]"
            if col == active_col and row == active_row:
                return "[  *]"
            if counts[row][col] > 0:
                return f"[{counts[row][col]:3d}]"
            return "[   ]"

        # Rotated view: each display-row = one column (highest col first),
        # each display-col = one row (highest row first)
        lines = [f"Bin Coverage ({self.n_cols}×{self.n_rows}) — {self.object_key} "
                 f"(robot view, ←far  near→)"]
        for col in reversed(range(self.n_cols)):
            cells = ""
            for row in reversed(range(self.n_rows)):
                cells += _cell(col, row)
            prefix = f"c{col}  "
            lines.append(f"{prefix}{cells}")
        # Row labels at bottom
        row_labels = "    " + "".join(f" r{r:1d}  " for r in reversed(range(self.n_rows)))
        lines.append(row_labels)
        return "\n".join(lines)


# ------------------------------------------------------------------
# CLI helper
# ------------------------------------------------------------------


def parse_ood_bins(ood_str: str) -> frozenset[tuple[int, int]]:
    """Parse OOD bins from a semicolon-separated CLI string.

    Format: ``"col,row;col,row;..."``
    Empty string returns empty frozenset.

    Args:
        ood_str: CLI argument string.

    Returns:
        Frozenset of (col, row) tuples.
    """
    if not ood_str or not ood_str.strip():
        return frozenset()
    pairs = []
    for token in ood_str.split(";"):
        token = token.strip()
        if not token:
            continue
        parts = token.split(",")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid OOD bin format: '{token}'. Expected 'col,row'."
            )
        pairs.append((int(parts[0]), int(parts[1])))
    return frozenset(pairs)


# ------------------------------------------------------------------
# Isaac Sim integration — apply_cube_pose
# ------------------------------------------------------------------
# NOTE: This function imports torch + isaaclab lazily at call-time to keep
# the rest of this module Isaac-Sim-free (module-level imports stay pure).
# It is the shared implementation of `_apply_spawn_config` from
# gamepad_recorder.py Z.152-164 — mirror it exactly so original recording,
# replay recorder, and eval scene_loader all produce bit-identical cube poses.

def apply_cube_pose(
    env,
    x_m: float,
    y_m: float,
    yaw_rad: float,
    *,
    env_origins=None,
    cube_key: str = "cube",
) -> None:
    """Teleport cube to (x, y) with yaw rotation + zero velocity.

    Single source of truth for cube spawn, shared by the recorder
    (``teleop_recorder``), the replay (``multicam_replay``), and the eval
    scene loader (``scene_loader``).

    Quaternion is written to ``root_state[:, 3:7]`` in XYZW order (Isaac Lab
    post-PR #4437 convention, confirmed in base_rigid_object.py:173 docstring).
    Pure Z-rotation by yaw_rad: (qx=0, qy=0, qz=sin(yaw/2), qw=cos(yaw/2)).

    Velocity is zeroed explicitly (`root_state[:, 7:] = 0.0`) — fixes the
    "cube retains momentum between episodes" bug where
    ``write_root_pose_to_sim`` was used (which leaves velocity untouched).

    Z-position stays at ``default_root_state[:, 2]`` (cube spawn height from
    ArticulationCfg.init_state.pos.z). Matches original recording behavior.

    Args:
        env: Gymnasium-wrapped Isaac Lab environment.
        x_m: Target cube X position in meters (world-frame or env-local).
        y_m: Target cube Y position in meters.
        yaw_rad: Cube yaw rotation in radians (around Z-axis).
        env_origins: Per-env world-frame origin tensor of shape (num_envs, 3),
            required for multi-env (cube gets env_origins + x/y as world pos).
            Pass None for single-env (default).
        cube_key: Scene key for the cube asset (default "cube").
    """
    import torch  # lazy: keep module Isaac-Sim-free

    cube = env.unwrapped.scene[cube_key]
    root_state = cube.data.default_root_state.clone()

    if env_origins is not None:
        root_state[:, 0] = env_origins[:, 0] + x_m
        root_state[:, 1] = env_origins[:, 1] + y_m
    else:
        root_state[:, 0] = x_m
        root_state[:, 1] = y_m
    # z unchanged = default_root_state[:, 2]

    # Quaternion slots [3:7] in XYZW order (Isaac Lab post-PR #4437 convention,
    # see base_rigid_object.py:173). Pure Z-rotation by yaw_rad:
    # (qx=0, qy=0, qz=sin(yaw/2), qw=cos(yaw/2)). No roll/pitch for cube — the
    # cube is a simple rigid body spawned flat on the pad. Scene_state's
    # object_init_rpy_rad field is UNRELIABLE post-PR #4437 (scene_state.py:167
    # comment is stale, assumes WXYZ but root_quat_w now returns XYZW).
    root_state[:, 3] = 0.0
    root_state[:, 4] = 0.0
    root_state[:, 5] = math.sin(yaw_rad / 2.0)
    root_state[:, 6] = math.cos(yaw_rad / 2.0)

    # Zero linear + angular velocity (7:10 = lin_vel, 10:13 = ang_vel).
    root_state[:, 7:] = 0.0

    cube.write_root_state_to_sim(root_state)
