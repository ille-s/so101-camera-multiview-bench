# SPDX-License-Identifier: MIT
"""Shared SO-101 joint constants and coordinate transforms.

Single source of truth for the scene geometry the whole pipeline shares:
the workspace-frame anchors, the rigid shift between scenes, the joint
metadata, and the raw-degrees to sim-radians transform pair.

Deliberately free of any Isaac Sim / AppLauncher import, so that recording,
replay and evaluation can read these constants at module level without
booting a simulator. Joint metadata mirrors the LeRobot interface
(``SO101_USD_MAPPING``, ``SO101_JOINT_ORDER``).
"""

from __future__ import annotations

__version__ = "1.0.0"

import torch

# ---------------------------------------------------------------------------
# Pipeline constants
# ---------------------------------------------------------------------------

FPS: int = 30
"""Simulation and recording frame rate (Hz)."""

WARMUP_STEPS: int = 60
"""Physics warmup after env.reset() — 2 seconds at FPS=30."""

# ---------------------------------------------------------------------------
# SO-101 scene geometry (Isaac Sim world frame {0})
# ---------------------------------------------------------------------------

HEMISPHERE_ORIGIN_W: tuple[float, float, float] = (0.20, 0.0, 0.12)
"""{W} origin of the retired floor-level table scene, in world frame {0}.

Cameras, cube and bin grid are all anchored in the workspace frame {W};
this constant is one scene's {W} origin expressed in {0} (⁰ξ_W per Corke
2023, §2.1.2). External cameras orbit that point and aim at it.

The scene it describes is no longer built: the package's base scene is the
cylindrical room (``BASE_SCENE_ORIGIN_W``). The constant stays because it is
the frame that **recorded data without a frame declaration** lives in, so it
remains the source origin when replaying or evaluating such a dataset.
"""

# ---------------------------------------------------------------------------
# Cylindrical-room scene geometry
#
# Every pipeline stage runs in the cylindrical room, where the rig stands on a
# pedestal table. An earlier floor-level table scene put the arm base at the
# world origin and spawned the cube on the floor; datasets recorded there are
# still replayable, because the bridge between the two scenes is ONE rigid
# translation applied to the entire coupled configuration -- arm, cube, action
# pad and camera hemisphere alike. The whole rig moves together, so the
# arm-to-cube geometry is untouched and a recorded joint trajectory reproduces
# the identical grasp.
#
# Single source of truth for these numbers. They used to be duplicated in
# recording/multicam_replay.py and evaluation/async_eval.py with a
# "keep in sync" comment; now both import from here.
# ---------------------------------------------------------------------------

TABLE_TOP_Z: float = 0.77
"""Tabletop SURFACE height in the cylindrical room: Top cylinder centre 0.745
plus half-height 0.025."""

PHASE3_FLOOR_Z: float = 0.0237
"""Floor surface the recorded rig sat on (= action-pad bottom: pad origin 0.0257
minus half-thickness 0.002). Verified empirically in a settled scene: arm-base
world z = 0.0, pad z = 0.0257, cube z = 0.0477. The arm ORIGIN sits ~2.4 cm
BELOW this surface -- its foot plate contacts the floor, not the origin. Do NOT
use the table surface as the arm origin; that over-lifts the rig by 2.4 cm and
leaves pad and cube floating."""

WORLD_DELTA_Z: float = TABLE_TOP_Z - PHASE3_FLOOR_Z  # = 0.7463
"""The one rigid z-shift that maps the recorded floor onto the table surface."""

ARM_BASE_OFFSET: tuple[float, float, float] = (-0.20, 0.0, WORLD_DELTA_Z)
"""Rigid translation applied to the ENTIRE coupled configuration when moving
from the recording scene into the cylindrical room."""

ROBOT_TABLE_POS: tuple[float, float, float] = ARM_BASE_OFFSET
"""Arm base world position on the pedestal table. Identical to ARM_BASE_OFFSET
because the arm base sits at the world origin in the recording scene."""

FLOOR_CUBE_SPAWN_Z: float = 0.05
"""Cube spawn height recorded in ``scene_state.object_init_pos_m.z``, expressed
in the recording scene's floor frame."""

TABLE_HEMISPHERE_ORIGIN: tuple[float, float, float] = (
    HEMISPHERE_ORIGIN_W[0] + ARM_BASE_OFFSET[0],
    HEMISPHERE_ORIGIN_W[1] + ARM_BASE_OFFSET[1],
    HEMISPHERE_ORIGIN_W[2] + ARM_BASE_OFFSET[2],
)  # = (0.0, 0.0, 0.8663)
"""Camera-hemisphere origin in the cylindrical room: the recording-scene origin
shifted by the same rigid delta as everything else. This is the workspace frame
{W} origin of the cylindrical-room scene, expressed in world frame {0}."""

BASE_SCENE_ORIGIN_W: tuple[float, float, float] = TABLE_HEMISPHERE_ORIGIN
"""{W} origin of the scene ``LiftCubeSceneCfg`` is authored in.

Since the cylroom migration the base scene IS the cylindrical room, so scene
overrides (RIG_SHIFT) are computed against this origin: for the shipped cylroom
configs the rig shift is zero and the base scene spawns as authored. Legacy
datasets recorded in the retired floor-level table scene still declare (or
default to) ``HEMISPHERE_ORIGIN_W`` as their SOURCE origin -- that constant
stays for data interpretation, this one describes the code's scene."""


CUBE_SPAWN_Z_IN_W: float = FLOOR_CUBE_SPAWN_Z - HEMISPHERE_ORIGIN_W[2]  # = -0.07
"""Cube spawn height expressed in the workspace frame {W}. Scene-independent:
the spawn height in any scene's world frame is its {W} origin z plus this."""


def rigid_shift_from_recording(
    workspace_origin_w: tuple[float, float, float],
    source_origin_w: tuple[float, float, float] = HEMISPHERE_ORIGIN_W,
) -> tuple[float, float, float]:
    """Translation that carries a recorded rig from its source scene into a target scene.

    A scene declares its workspace-frame {W} origin (``SceneConfig.
    workspace_origin_m``); recorded data declares the origin of the scene it was
    captured in (``binmap.json`` / ``scene_state`` ``workspace_origin_m``,
    defaulting to the floor-level table scene at ``HEMISPHERE_ORIGIN_W`` for
    datasets that predate the field). The rigid shift for the ENTIRE coupled rig
    -- arm base, cube, action pad, camera hemisphere -- is simply the difference
    between the two origins. This is the pose-graph property that keeps recorded
    joint trajectories valid across scenes: the rig moves as one, so the
    arm-to-cube geometry never changes.

    Table-scene recording replayed in the cylindrical room: returns exactly
    ``ARM_BASE_OFFSET``. Same scene on both sides: returns (0, 0, 0).
    """
    return (
        workspace_origin_w[0] - source_origin_w[0],
        workspace_origin_w[1] - source_origin_w[1],
        workspace_origin_w[2] - source_origin_w[2],
    )

# ---------------------------------------------------------------------------
# SO-101 joint metadata (from lerobot_interface.py)
# ---------------------------------------------------------------------------

SO101_JOINT_ORDER: list[str] = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
"""Joint names in LeRobot observation/action order (with .pos suffix)."""

SO101_USD_MAPPING: dict[str, dict[str, int]] = {
    "shoulder_pan": {"joint_min": -110, "joint_max": 110},
    "shoulder_lift": {"joint_min": -100, "joint_max": 100},
    "elbow_flex": {"joint_min": -100, "joint_max": 90},
    "wrist_flex": {"joint_min": -95, "joint_max": 95},
    "wrist_roll": {"joint_min": -160, "joint_max": 160},
    "gripper": {"joint_min": -10, "joint_max": 100},
}
"""USD joint limits in degrees — defines the mapping between normalized
[0, 1] space and physical joint angles."""

SO101_OBSERVATION_FEATURES: dict[str, type] = {
    "shoulder_pan.pos": float,
    "shoulder_lift.pos": float,
    "elbow_flex.pos": float,
    "wrist_flex.pos": float,
    "wrist_roll.pos": float,
    "gripper.pos": float,
    # Camera features are added dynamically based on env cameras.
}
"""Static observation feature schema (without cameras)."""

SO101_ACTION_FEATURES: dict[str, type] = {
    "shoulder_pan.pos": float,
    "shoulder_lift.pos": float,
    "elbow_flex.pos": float,
    "wrist_flex.pos": float,
    "wrist_roll.pos": float,
    "gripper.pos": float,
}
"""Action feature schema — 6 joint positions in raw degrees."""


# ---------------------------------------------------------------------------
# Coordinate transform functions
# ---------------------------------------------------------------------------

def build_joint_tensors(
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute joint min/max tensors for coordinate transforms.

    Args:
        device: Torch device (e.g. ``"cuda:0"`` or ``torch.device("cpu")``).

    Returns:
        Tuple of ``(joint_mins, joint_maxs)`` tensors, each shape ``(6,)``,
        containing USD joint limits in degrees.
    """
    joint_names = [j.split(".")[0] for j in SO101_JOINT_ORDER]
    joint_mins = torch.tensor(
        [SO101_USD_MAPPING[n]["joint_min"] for n in joint_names],
        dtype=torch.float32,
        device=device,
    )
    joint_maxs = torch.tensor(
        [SO101_USD_MAPPING[n]["joint_max"] for n in joint_names],
        dtype=torch.float32,
        device=device,
    )
    return joint_mins, joint_maxs


def raw_degrees_to_sim_radians(
    raw_values: torch.Tensor,
    joint_mins: torch.Tensor,
    joint_maxs: torch.Tensor,
) -> torch.Tensor:
    """Convert robot raw degrees to Isaac Sim joint position radians.

    Transform chain: raw_degrees → normalized [0,1] → mapped degrees → radians.

    Args:
        raw_values: Raw action values, shape ``(6,)``.
            Joints 0-4: range [-100, 100]. Gripper (5): range [0, 100].
        joint_mins: USD joint minimums in degrees, shape ``(6,)``.
        joint_maxs: USD joint maximums in degrees, shape ``(6,)``.

    Returns:
        Joint positions in radians, shape ``(6,)``.
    """
    normalized = torch.zeros_like(raw_values)
    normalized[:-1] = (raw_values[:-1] + 100.0) / 200.0  # joints: -100..100 → 0-1
    normalized[-1] = raw_values[-1] / 100.0               # gripper: 0..100 → 0-1
    mapped_deg = joint_mins + normalized * (joint_maxs - joint_mins)
    return mapped_deg * torch.pi / 180.0


def sim_radians_to_raw_degrees(
    sim_obs: torch.Tensor,
    joint_mins: torch.Tensor,
    joint_maxs: torch.Tensor,
) -> torch.Tensor:
    """Convert Isaac Sim joint positions (radians) to robot raw degrees.

    Inverse of :func:`raw_degrees_to_sim_radians`.

    Args:
        sim_obs: Joint positions in radians, shape ``(6,)``.
        joint_mins: USD joint minimums in degrees, shape ``(6,)``.
        joint_maxs: USD joint maximums in degrees, shape ``(6,)``.

    Returns:
        Raw degree values, shape ``(6,)``.
            Joints 0-4: range [-100, 100]. Gripper (5): range [0, 100].
    """
    mapped_deg = sim_obs * 180.0 / torch.pi
    normalized = (mapped_deg - joint_mins) / (joint_maxs - joint_mins)
    raw_degrees = torch.zeros_like(normalized)
    raw_degrees[:-1] = normalized[:-1] * 200.0 - 100.0  # joints: 0-1 → -100..100
    raw_degrees[-1] = normalized[-1] * 100.0             # gripper: 0-1 → 0..100
    return raw_degrees
