# SPDX-License-Identifier: MIT
"""Camera pose grid for the camera permutation experiment.

Defines a hemisphere grid of external camera poses in **workspace frame {W}**,
centered at the origin (0, 0, 0).  The caller must provide the transform
⁰ξ_W (workspace origin in world frame) to place cameras in Isaac Sim.

Frame convention (Corke, 2023, §2.1.2):
    {0}  World frame — Isaac Sim origin
    {W}  Workspace frame — centered on the manipulation target
    {C}  Camera frame — position on the hemisphere in {W}

    ⁰p_camera = ⁰ξ_W ⊕ ᵂp_camera

No Isaac Sim dependency at module load (torch pulled lazily inside the
setter functions). Re-exports ``HEMISPHERE_ORIGIN_W`` from
``so101_transforms`` as single source of truth (was
``DEFAULT_WORKSPACE_ORIGIN_W`` until 2026-05-12 — renamed to reflect
that this is the geometric origin of the camera hemisphere).

Usage:
    python camera_grid.py --list          # print all 24 poses (workspace frame)
    python camera_grid.py --visualize     # 3D scatter plot around origin
"""

from __future__ import annotations

__version__ = "1.1.0"

import argparse
import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

try:
    from so101_mvbench.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKSPACE_CENTER: tuple[float, float, float] = (0.0, 0.0, 0.0)
"""Default center for hemisphere grid — workspace frame {W} origin."""

# Re-export the canonical hemisphere origin from so101_transforms
# (single source of truth). Importing here so callers can do
# `from camera_grid import HEMISPHERE_ORIGIN_W` without an extra import.
from so101_mvbench.utils.so101_transforms import (  # noqa: E402
    HEMISPHERE_ORIGIN_W,
)

DEFAULT_RADIUS_M: float = 0.4
"""Distance from look-at target to each camera position in meters."""

DEFAULT_AZIMUTHS_DEG: tuple[float, ...] = (0, 45, 90, 135, 180, 225, 270, 315)
"""Azimuth angles for the coarse grid (degrees, 0 = +X axis)."""

DEFAULT_ELEVATIONS_DEG: tuple[float, ...] = (20, 45, 70)
"""Elevation angles for the coarse grid (degrees above horizontal)."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraPose:
    """A single camera position on the hemisphere.

    Attributes:
        id: Unique identifier, e.g. ``az045_el45``.
        azimuth_deg: Azimuth angle in degrees (0 = +X, 90 = +Y).
        elevation_deg: Elevation angle in degrees above horizontal.
        radius_m: Distance from look-at target in meters.
        x: World-frame X coordinate in meters.
        y: World-frame Y coordinate in meters.
        z: World-frame Z coordinate in meters.
    """

    id: str
    azimuth_deg: float
    elevation_deg: float
    radius_m: float
    x: float
    y: float
    z: float


# ---------------------------------------------------------------------------
# Coordinate math
# ---------------------------------------------------------------------------

def spherical_to_cartesian(
    azimuth_deg: float,
    elevation_deg: float,
    radius_m: float,
    center: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Convert spherical coordinates to Cartesian world coordinates.

    Args:
        azimuth_deg: Azimuth in degrees (0 = +X axis, 90 = +Y axis).
        elevation_deg: Elevation in degrees above the horizontal plane.
        radius_m: Distance from *center* in meters.
        center: (cx, cy, cz) look-at point in meters.

    Returns:
        (x, y, z) camera position in meters.
    """
    az_rad = math.radians(azimuth_deg)
    el_rad = math.radians(elevation_deg)

    cx, cy, cz = center
    x = cx + radius_m * math.cos(el_rad) * math.cos(az_rad)
    y = cy + radius_m * math.cos(el_rad) * math.sin(az_rad)
    z = cz + radius_m * math.sin(el_rad)
    return (x, y, z)


def _make_pose_id(azimuth_deg: float, elevation_deg: float) -> str:
    """Build a canonical pose ID like ``az045_el45``."""
    return f"az{int(azimuth_deg):03d}_el{int(elevation_deg):02d}"


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def generate_hemisphere_grid(
    azimuths_deg: tuple[float, ...] = DEFAULT_AZIMUTHS_DEG,
    elevations_deg: tuple[float, ...] = DEFAULT_ELEVATIONS_DEG,
    radius_m: float = DEFAULT_RADIUS_M,
    center: tuple[float, float, float] = WORKSPACE_CENTER,
) -> list[CameraPose]:
    """Generate a list of camera poses on a hemisphere.

    Args:
        azimuths_deg: Azimuth angles to sample.
        elevations_deg: Elevation angles to sample.
        radius_m: Hemisphere radius in meters.
        center: Look-at target (world frame).

    Returns:
        List of ``CameraPose`` objects, sorted by (elevation, azimuth).
    """
    poses: list[CameraPose] = []
    for el_deg in elevations_deg:
        for az_deg in azimuths_deg:
            x, y, z = spherical_to_cartesian(
                azimuth_deg=az_deg,
                elevation_deg=el_deg,
                radius_m=radius_m,
                center=center,
            )
            poses.append(CameraPose(
                id=_make_pose_id(az_deg, el_deg),
                azimuth_deg=az_deg,
                elevation_deg=el_deg,
                radius_m=radius_m,
                x=x,
                y=y,
                z=z,
            ))
    return poses


# ---------------------------------------------------------------------------
# Pre-built grid + lookup
# ---------------------------------------------------------------------------

COARSE_GRID: list[CameraPose] = generate_hemisphere_grid()
"""Default 24-pose hemisphere grid (8 azimuth × 3 elevation)."""

_POSE_INDEX: dict[str, CameraPose] = {p.id: p for p in COARSE_GRID}


# Specs for poses outside the regular 8×3 COARSE_GRID (azimuth, elevation, id).
# Kept separate to preserve grid semantics and to allow runtime regeneration
# at a custom radius via ``make_extra_poses(radius_m)``.
#   - el=90° top-down: singularity (all azimuths collapse to one point)
#   - el=-5°: near-ground poses for low-angle wrist-mimicking renders
_EXTRA_POSE_SPECS: tuple[tuple[float, float, str], ...] = (
    (0.0, 90.0, "az000_el90"),
    (0.0, -5.0, "az000_el-05"),
    (90.0, -5.0, "az090_el-05"),
    (270.0, -5.0, "az270_el-05"),
)


def make_extra_poses(radius_m: float = DEFAULT_RADIUS_M) -> list[CameraPose]:
    """Construct all non-grid singleton poses at *radius_m*.

    Returns the top-down singularity and the el=-5° near-ground poses,
    scaled to the requested radius. Callers that regenerate the hemisphere
    grid at runtime (e.g. ``external_batch_recorder --radius``) merge this
    into their lookup table alongside ``generate_hemisphere_grid()`` so
    every documented pose-id resolves.
    """
    poses = []
    for az, el, pid in _EXTRA_POSE_SPECS:
        x, y, z = spherical_to_cartesian(
            azimuth_deg=az, elevation_deg=el,
            radius_m=radius_m, center=WORKSPACE_CENTER,
        )
        poses.append(CameraPose(
            id=pid, azimuth_deg=az, elevation_deg=el,
            radius_m=radius_m, x=x, y=y, z=z,
        ))
    return poses


# Seed _POSE_INDEX with the extra (non-grid) poses at DEFAULT_RADIUS_M.
for _p in make_extra_poses(DEFAULT_RADIUS_M):
    _POSE_INDEX[_p.id] = _p


def get_pose_by_id(pose_id: str) -> CameraPose:
    """Look up a camera pose by its ID.

    Args:
        pose_id: Pose identifier, e.g. ``az045_el45``.

    Returns:
        The matching ``CameraPose``.

    Raises:
        KeyError: If *pose_id* is not in the grid.
    """
    if pose_id not in _POSE_INDEX:
        valid = ", ".join(sorted(_POSE_INDEX.keys()))
        raise KeyError(
            f"Unknown pose_id '{pose_id}'. Valid IDs: {valid}"
        )
    return _POSE_INDEX[pose_id]


def list_pose_ids() -> list[str]:
    """Return all pose IDs in grid order."""
    return [p.id for p in COARSE_GRID]


_POSE_ID_RE = re.compile(r"^az(\d+)_el(-?\d+)$")


def parse_elevations_from_pose_ids(pose_ids: Sequence[str]) -> tuple[float, ...]:
    """Extract the unique elevation angles (deg) from a list of pose IDs.

    Pose IDs follow ``az{azimuth:03d}_el{elevation:02d}`` (negatives with a
    leading dash, e.g. ``az000_el-05``). Used by callers that regenerate
    the hemisphere grid at runtime so they don't need a separate CLI flag
    for elevations — the IDs are self-describing.

    Args:
        pose_ids: List of pose-id strings.

    Returns:
        Sorted tuple of elevation angles in degrees.

    Raises:
        ValueError: If any pose ID does not match the expected format.
    """
    elevations: set[int] = set()
    for pid in pose_ids:
        m = _POSE_ID_RE.match(pid)
        if not m:
            raise ValueError(
                f"Pose ID '{pid}' does not match expected pattern "
                f"'az{{az:03d}}_el{{el:02d}}' (e.g. 'az045_el45', 'az000_el-05')."
            )
        elevations.add(int(m.group(2)))
    return tuple(sorted(elevations))


def resolve_poses(
    pose_ids: Sequence[str],
    radius_m: float = DEFAULT_RADIUS_M,
) -> list[CameraPose]:
    """Resolve a list of pose-IDs to ``CameraPose`` objects at *radius_m*.

    Elevations are auto-derived from the pose-IDs themselves
    (:py:func:`parse_elevations_from_pose_ids`). The hemisphere grid is
    regenerated at runtime so any well-formed combination of azimuth/
    elevation resolves consistently — including the singleton extras
    (top-down at el=90°, near-ground at el=-5°).

    This is the recommended API for callers that take a user-supplied
    list of pose-IDs (recorders, batch tools). For ad-hoc lookups of a
    single pose at DEFAULT_RADIUS_M, use :py:func:`get_pose_by_id`.

    Args:
        pose_ids: Pose-ID strings (``az###_el##``).
        radius_m: Hemisphere radius in meters.

    Returns:
        Camera poses in the same order as *pose_ids*.

    Raises:
        ValueError: If any pose ID cannot be resolved at *radius_m*.
    """
    elevations_deg = parse_elevations_from_pose_ids(pose_ids)
    by_id = {p.id: p for p in generate_hemisphere_grid(
        elevations_deg=elevations_deg, radius_m=radius_m,
    )}
    by_id.update({p.id: p for p in make_extra_poses(radius_m=radius_m)})
    try:
        return [by_id[pid] for pid in pose_ids]
    except KeyError as e:
        raise ValueError(
            f"Unknown pose_id {e!s} at radius={radius_m}m "
            f"(elevations auto-derived: {list(elevations_deg)}). "
            f"Available: {sorted(by_id.keys())}"
        ) from None


# ---------------------------------------------------------------------------
# Camera-pose helpers — getter + setters (single source of truth)
# ---------------------------------------------------------------------------

def get_pose_world_position(
    pose: CameraPose,
    hemisphere_origin: tuple[float, float, float] = HEMISPHERE_ORIGIN_W,
) -> tuple[float, float, float]:
    """Compute world-frame position of a hemisphere pose.

    Pure math, no Isaac Sim. Returns ``(pose.x + wx, pose.y + wy, pose.z + wz)``
    where ``pose.{x,y,z}`` is the {W}-frame offset and ``(wx, wy, wz)`` is the
    look-at target in world frame {0} (≡ ⁰ξ_W).

    Used by matplotlib visualizers (no sim) and any caller that needs the
    world-frame eye position without applying it to a camera asset.

    Args:
        pose: Hemisphere camera pose (in workspace frame {W}).
        hemisphere_origin: ⁰ξ_W transform (world frame) — point all cameras
            aim at. Coincides with workspace frame {W} origin.

    Returns:
        Tuple ``(x, y, z)`` in world frame, meters.
    """
    wx, wy, wz = hemisphere_origin
    return (pose.x + wx, pose.y + wy, pose.z + wz)


def _lookat_quat_xyzw_opengl(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> list[float]:
    """World quaternion (x, y, z, w) in the OpenGL/USD convention (forward -Z,
    up +Y) for a camera at ``eye`` looking at ``target`` with a FIXED world ``up``.

    Unlike Isaac Lab's ``set_world_poses_from_view`` (which derives the up from
    the stage up-axis and so rolls discontinuously near the pole), this uses the
    caller-supplied ``up`` constant, giving a continuous roll along a meridian.
    Degenerate case (view parallel to ``up``) falls back to world +X for ``up``.
    """
    import numpy as np
    e = np.asarray(eye, dtype=float)
    t = np.asarray(target, dtype=float)
    f = t - e
    nf = np.linalg.norm(f)
    if nf < 1e-9:
        return [0.0, 0.0, 0.0, 1.0]
    f = f / nf
    u = np.asarray(up, dtype=float)
    if abs(float(np.dot(f, u))) > 0.999:
        u = np.array([1.0, 0.0, 0.0])
    right = np.cross(f, u); right = right / np.linalg.norm(right)
    true_up = np.cross(right, f)
    R = np.column_stack((right, true_up, -f))  # X=right, Y=up, Z=-forward (OpenGL)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return [float(x), float(y), float(z), float(w)]


def set_external_camera_pose(
    env,
    pose: "CameraPose | str",
    hemisphere_origin: tuple[float, float, float] = HEMISPHERE_ORIGIN_W,
    camera_name: str = "camera_external",
    up_world: tuple[float, float, float] | None = None,
):
    """Set a SINGLE pose, broadcast to ALL envs via env_origins.

    Use for single-pose runs (hemisphere_screenshot, trajectory_replayer,
    eval/camera_override). All envs render the SAME camera position
    relative to their own env_origin — Isaac Lab's broadcast semantics
    on ``set_world_poses_from_view((1,3), (1,3))`` against an N-env scene.

    Args:
        env: Gym-wrapped Isaac Lab env (must have ``unwrapped.scene[camera_name]``).
        pose: CameraPose object or pose ID string (e.g. ``"az000_el45"``).
        hemisphere_origin: ⁰ξ_W transform (world frame) — point all cameras
            aim at. Coincides with workspace frame {W} origin.
        camera_name: Scene asset key for the camera (default ``"camera_external"``).
        up_world: Optional FIXED world up-vector. When given, the camera roll is
            derived from this constant up instead of the stage up-axis, so a
            camera swept along a meridian tilts continuously with no roll jump at
            the pole (used by the camera-drift ablation). Default ``None`` keeps
            the original ``set_world_poses_from_view`` behaviour (backward compat).

    Returns:
        Eye tensor of shape ``(num_envs, 3)`` — for logging/debug.

    Raises:
        KeyError: ``camera_name`` not in ``env.unwrapped.scene``.
        ValueError: ``pose`` is a string but not a known pose ID.
    """
    import torch
    scene = env.unwrapped.scene
    if camera_name not in scene.keys():
        raise KeyError(
            f"{camera_name} not in env.scene (available: "
            f"{[k for k in scene.keys() if k.startswith('camera_')]})"
        )

    if isinstance(pose, str):
        pose = get_pose_by_id(pose)

    env_origins = scene.env_origins  # (n, 3)
    device = env_origins.device

    wx, wy, wz = hemisphere_origin
    eye_local = torch.tensor(
        [[pose.x + wx, pose.y + wy, pose.z + wz]],
        device=device, dtype=torch.float32,
    )  # (1, 3)
    target_local = torch.tensor(
        [[wx, wy, wz]], device=device, dtype=torch.float32,
    )  # (1, 3)

    # Broadcast: (1,3) + (n,3) → (n,3). All envs get the same relative pose.
    eye = eye_local + env_origins
    target = target_local + env_origins

    if up_world is not None:
        # Fixed-up aim: same relative orientation for every env (rotation is
        # translation-invariant), so compute one quat and repeat over envs.
        q = _lookat_quat_xyzw_opengl(
            (pose.x + wx, pose.y + wy, pose.z + wz), (wx, wy, wz), up_world,
        )
        quat = torch.tensor([q], device=device, dtype=torch.float32).repeat(
            env_origins.shape[0], 1,
        )
        scene[camera_name].set_world_poses(
            positions=eye, orientations=quat, convention="opengl",
        )
        return eye

    scene[camera_name].set_world_poses_from_view(eye, target)
    return eye


def set_external_camera_poses(
    env,
    poses: "Sequence[CameraPose | str]",
    hemisphere_origin: tuple[float, float, float] = HEMISPHERE_ORIGIN_W,
    camera_name: str = "camera_external",
):
    """Set a DIFFERENT pose per env. Asserts ``len(poses) == num_envs``.

    Use for per-env permutation (external_batch_recorder runs N cameras
    simultaneously, env_i renders pose_i).

    Args:
        env: Gym-wrapped Isaac Lab env (must have ``unwrapped.scene[camera_name]``).
        poses: List of CameraPose objects or pose ID strings. Length MUST
            equal ``env.unwrapped.scene.num_envs``.
        hemisphere_origin: ⁰ξ_W transform (world frame) — point all cameras
            aim at. Coincides with workspace frame {W} origin.
        camera_name: Scene asset key for the camera (default ``"camera_external"``).

    Returns:
        Eye tensor of shape ``(num_envs, 3)`` — for logging/debug.

    Raises:
        KeyError: ``camera_name`` not in ``env.unwrapped.scene``.
        ValueError: ``len(poses) != num_envs`` or unknown pose ID string.
    """
    import torch
    scene = env.unwrapped.scene
    if camera_name not in scene.keys():
        raise KeyError(
            f"{camera_name} not in env.scene (available: "
            f"{[k for k in scene.keys() if k.startswith('camera_')]})"
        )

    n = scene.num_envs
    if len(poses) != n:
        raise ValueError(
            f"len(poses)={len(poses)} != num_envs={n} — "
            f"use set_external_camera_pose() for single-pose-broadcast."
        )

    poses_resolved = [
        get_pose_by_id(p) if isinstance(p, str) else p for p in poses
    ]

    env_origins = scene.env_origins  # (n, 3)
    device = env_origins.device

    wx, wy, wz = hemisphere_origin
    eyes_local = torch.tensor(
        [[p.x + wx, p.y + wy, p.z + wz] for p in poses_resolved],
        device=device, dtype=torch.float32,
    )  # (n, 3)
    targets_local = torch.tensor(
        [[wx, wy, wz]] * n, device=device, dtype=torch.float32,
    )  # (n, 3)

    eyes = eyes_local + env_origins
    targets = targets_local + env_origins

    scene[camera_name].set_world_poses_from_view(eyes, targets)
    return eyes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _draw_frame_axes(
    ax,
    origin: tuple[float, float, float],
    label: str,
    length: float = 0.08,
    lw: float = 2.0,
    fontsize: int = 13,
) -> None:
    """Draw RGB coordinate frame axes (X=red, Y=green, Z=blue) at *origin*.

    Follows the Corke (2023) Fig. 2.6 convention: colored arrows + frame label.
    """
    ox, oy, oz = origin
    ax.quiver(ox, oy, oz, length, 0, 0, color="#cc0000", linewidth=lw, arrow_length_ratio=0.15)
    ax.quiver(ox, oy, oz, 0, length, 0, color="#00aa00", linewidth=lw, arrow_length_ratio=0.15)
    ax.quiver(ox, oy, oz, 0, 0, length, color="#0044cc", linewidth=lw, arrow_length_ratio=0.15)
    ax.text(ox, oy, oz - 0.04, label, fontsize=fontsize, fontweight="bold",
            color="#333333", ha="center", va="top")


def _visualize_frames(scene_config_path: str | None = None,
                      save_path: str | None = None) -> None:
    """Plot the task's coordinate frames in 3D, for one scene.

    Every element is derived from the scene the config declares, so the plot
    cannot drift from the pipeline: {W} comes from the config's
    ``workspace_origin_m``, {B} and {P} from the {W}-relative anchors in
    ``so101_transforms``, and the camera positions from the config's own
    external cameras. Uses ``spatialmath.SE3.plot()`` for right-hand-rule
    frame rendering (Corke, 2023, §2.1.2).

    Args:
        scene_config_path: Scene config JSON. Defaults to the shipped
            six-camera configuration.
        save_path: Write the figure to this path instead of opening a window.
    """
    try:
        import matplotlib
        if save_path:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from spatialmath import SE3
    except ImportError:
        logger.error(
            "spatialmath + matplotlib required for --frames. "
            "Install: pip install 'so101_mvbench[viz]'"
        )
        raise SystemExit(1)

    from so101_mvbench.utils.scene_config import SceneConfig
    from so101_mvbench.utils.so101_transforms import (
        ARM_BASE_OFFSET,
        CUBE_SPAWN_Z_IN_W,
    )

    if scene_config_path is None:
        scene_config_path = str(
            Path(__file__).resolve().parent.parent
            / "tasks" / "scene_configs" / "lift_cube_6cam.json"
        )
    scene = SceneConfig.from_json(scene_config_path)

    # The workspace origin is the one value that changes between scenes.
    WORKSPACE_ORIGIN_W = tuple(scene.workspace_origin_m)
    # {W}-relative anchors, identical in every scene.
    ARM_BASE_IN_W = (ARM_BASE_OFFSET[0], ARM_BASE_OFFSET[1], -HEMISPHERE_ORIGIN_W[2])
    externals = [
        (name, c) for name, c in scene.cameras.items() if c.type == "external"
    ]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
    })
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    wx, wy, wz = WORKSPACE_ORIGIN_W

    # --- Coordinate frames via SE3.plot() ---
    # {0} World / Reference
    SE3().plot(frame="0", ax=ax, length=0.12, style="rviz", textcolor="black")

    # {B} Arm base, anchored {W}-relative like everything else
    base_world = tuple(w + a for w, a in zip(WORKSPACE_ORIGIN_W, ARM_BASE_IN_W))
    SE3(*base_world).plot(frame="B", ax=ax, length=0.08, style="rviz", textcolor="#555555")

    # {W} Workspace: the pose-graph root this scene declares
    SE3(wx, wy, wz).plot(frame="W", ax=ax, length=0.10, style="rviz", textcolor="black")

    # {P} Workpiece at its {W}-fixed spawn height
    SE3(wx, wy, wz + CUBE_SPAWN_Z_IN_W).plot(
        frame="P", ax=ax, length=0.06, style="rviz", textcolor="#555555")

    # {C} One frame per external camera of THIS scene config
    cam_positions = []
    for name, cam in externals:
        pos = spherical_to_cartesian(
            cam.azimuth_deg, cam.elevation_deg, cam.radius_m, WORKSPACE_ORIGIN_W,
        )
        cam_positions.append((name, pos))
        ax.scatter([pos[0]], [pos[1]], [pos[2]],
                   c="#cc0000", s=40, alpha=0.8, edgecolors="none")
        ax.text(pos[0], pos[1], pos[2], f"  {name}",  # type: ignore[call-overload]
                fontsize=9, color="#cc0000")
        # Workspace to camera: the experimental variable
        ax.plot([wx, pos[0]], [wy, pos[1]], [wz, pos[2]],
                color="#cc0000", linewidth=1.2, linestyle="--", alpha=0.6)
    if cam_positions:
        SE3(*cam_positions[0][1]).plot(
            frame="C", ax=ax, length=0.08, style="rviz", textcolor="#cc0000")

    # --- Transform arrows ---
    # World to workspace: the only edge that changes between scenes
    ax.plot([0, wx], [0, wy], [0, wz],
            color="#333333", linewidth=2, linestyle="--")
    ax.text(wx / 2 - 0.05, wy / 2, wz / 2 + 0.03,  # type: ignore[call-overload]
            r"$^0\xi_W$", fontsize=13, color="#333333")

    # --- Labels and title ---
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Coordinate frames of {Path(scene_config_path).stem}\n"
                 "spatialmath SE3.plot(), following Corke (2023)",
                 fontsize=14, pad=15)

    # Legend
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], color="red", lw=2, label="X axis"),
        Line2D([0], [0], color="green", lw=2, label="Y axis"),
        Line2D([0], [0], color="blue", lw=2, label="Z axis"),
        Line2D([0], [0], color="#333333", lw=2, ls="--", label=r"$^0\xi_W$ (scene property)"),
        Line2D([0], [0], color="#cc0000", lw=1.2, ls="--", label=r"$^W\xi_C$ (variable)"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=11,
              framealpha=0.9, edgecolor="black")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, facecolor="white")
        logger.info("Wrote %s", save_path)
    else:
        plt.show()


def _print_grid() -> None:
    """Print all poses as a formatted table."""
    header = f"{'ID':<14} {'Az':>5} {'El':>5} {'X':>8} {'Y':>8} {'Z':>8}"
    print(header)
    print("-" * len(header))
    for p in COARSE_GRID:
        print(f"{p.id:<14} {p.azimuth_deg:>5.0f} {p.elevation_deg:>5.0f} "
              f"{p.x:>8.4f} {p.y:>8.4f} {p.z:>8.4f}")
    print(f"\nTotal: {len(COARSE_GRID)} poses")


def _visualize_grid() -> None:
    """Show an interactive 3D scatter plot with filter checkboxes."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import CheckButtons
    except ImportError:
        logger.error("matplotlib required for --visualize. Install: pip install matplotlib")
        raise SystemExit(1)

    # Clean white style for thesis print
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 14,
        "axes.labelsize": 15,
        "axes.titlesize": 18,
        "legend.fontsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    cx, cy, cz = WORKSPACE_CENTER

    # High-contrast colors (colorblind-friendly)
    el_colors = {20: "#0072B2", 45: "#D55E00", 70: "#009E73"}
    el_labels_map = {20: "el=20\u00b0 (low)", 45: "el=45\u00b0 (mid)", 70: "el=70\u00b0 (high)"}
    az_values = sorted(set(int(p.azimuth_deg) for p in COARSE_GRID))
    el_values = sorted(set(int(p.elevation_deg) for p in COARSE_GRID))

    # Store artists per pose for toggling
    pose_artists: dict[str, list] = {}  # pose_id -> [scatter, text, line]

    for p in COARSE_GRID:
        color = el_colors[int(p.elevation_deg)]
        sc = ax.scatter(
            [p.x], [p.y], [p.z],
            c=color, s=90, depthshade=True, edgecolors="black",
            linewidths=0.8, zorder=5,
        )
        txt = ax.text(
            p.x, p.y, p.z, f"  {p.id}", fontsize=7,
            color="#333333", zorder=6,
        )
        line, = ax.plot(
            [p.x, cx], [p.y, cy], [p.z, cz],
            color="#999999", alpha=0.25, linewidth=0.5,
        )
        pose_artists[p.id] = [sc, txt, line]

    # Look-at target (always visible)
    ax.scatter(
        [cx], [cy], [cz], c="#CC0000", s=150, marker="X",
        linewidths=2.5, edgecolors="black", label="look-at target", zorder=10,
    )

    # Manual legend for elevation colors
    for el_deg, color in el_colors.items():
        ax.scatter([], [], [], c=color, s=60, edgecolors="black",
                   linewidths=0.8, label=el_labels_map[el_deg])

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(
        f"Camera Permutation Grid \u2014 {len(COARSE_GRID)} poses, r={DEFAULT_RADIUS_M}m",
        fontsize=14, pad=20,
    )
    ax.legend(loc="upper left", fontsize=13, framealpha=0.9, edgecolor="black")

    # --- Elevation checkboxes (left side) ---
    el_ax = fig.add_axes([0.01, 0.55, 0.12, 0.15])
    el_ax.set_title("Elevation", fontsize=11, fontweight="bold")
    el_check_labels = [f"{e}\u00b0" for e in el_values]
    el_check = CheckButtons(
        el_ax, el_check_labels,
        actives=[True] * len(el_values),
        label_props={"fontsize": [11] * len(el_values)},
        frame_props={"edgecolor": ["black"] * len(el_values)},
        check_props={"facecolor": [el_colors[e] for e in el_values]},
    )

    # --- Azimuth checkboxes (left side, below) ---
    az_ax = fig.add_axes([0.01, 0.10, 0.12, 0.40])
    az_ax.set_title("Azimuth", fontsize=11, fontweight="bold")
    az_check_labels = [f"{a}\u00b0" for a in az_values]
    az_check = CheckButtons(
        az_ax, az_check_labels,
        actives=[True] * len(az_values),
        label_props={"fontsize": [11] * len(az_values)},
        frame_props={"edgecolor": ["black"] * len(az_values)},
        check_props={"facecolor": ["#555555"] * len(az_values)},
    )

    # Track active filters
    active_els = set(el_values)
    active_azs = set(az_values)

    def _update_visibility() -> None:
        for p in COARSE_GRID:
            el = int(p.elevation_deg)
            az = int(p.azimuth_deg)
            visible = (el in active_els) and (az in active_azs)
            for artist in pose_artists[p.id]:
                artist.set_visible(visible)
        fig.canvas.draw_idle()

    def _on_el_click(label: str) -> None:
        el_deg = int(label.replace("\u00b0", ""))
        if el_deg in active_els:
            active_els.discard(el_deg)
        else:
            active_els.add(el_deg)
        _update_visibility()

    def _on_az_click(label: str) -> None:
        az_deg = int(label.replace("\u00b0", ""))
        if az_deg in active_azs:
            active_azs.discard(az_deg)
        else:
            active_azs.add(az_deg)
        _update_visibility()

    el_check.on_clicked(_on_el_click)
    az_check.on_clicked(_on_az_click)

    plt.subplots_adjust(left=0.16)
    plt.show()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Camera pose grid for permutation experiment",
    )
    parser.add_argument("--list", action="store_true", help="Print all poses")
    parser.add_argument("--list-ids", action="store_true", help="Print pose IDs only (one per line)")
    parser.add_argument("--visualize", action="store_true", help="3D scatter plot (camera hemisphere)")
    parser.add_argument("--frames", action="store_true",
                        help="3D coordinate frame plot for a scene config")
    parser.add_argument("--scene_config", type=str, default=None,
                        help="Scene config JSON for --frames "
                             "(default: the shipped six-camera config)")
    parser.add_argument("--save", type=str, default=None,
                        help="Write the --frames plot to this path instead of "
                             "opening a window")
    args = parser.parse_args()

    if not args.list and not args.list_ids and not args.visualize and not args.frames:
        parser.print_help()
        return

    if args.list:
        _print_grid()

    if args.list_ids:
        for pose_id in list_pose_ids():
            print(pose_id)

    if args.visualize:
        _visualize_grid()

    if args.frames:
        _visualize_frames(args.scene_config, args.save)


if __name__ == "__main__":
    main()
