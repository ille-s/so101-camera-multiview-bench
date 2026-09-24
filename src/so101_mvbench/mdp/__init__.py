# SPDX-License-Identifier: MIT
"""MDP term-functions for LiftCube tasks (NVIDIA-style mdp/ library).

Naming follows the Isaac Lab Manager-Term-Pattern: every public function in
this package is meant to be used as `func=` in an `EventTermCfg` /
`ObservationTermCfg` / `RewardTermCfg`. The package also re-exports
`isaaclab.envs.mdp.*` so callers can write `mdp.reset_root_state_uniform`,
`mdp.image`, etc., without importing both namespaces.

Submodules:
    resets    Event-reset terms (DR functions: lighting, colors, yaw)

Helpers:
    configure_randomization  Mutates a LiftCubeEventCfg in place to
                             enable/disable visual + spatial DR groups.

Constants:
    COLOR_PALETTE            8-color discrete palette (Colosseum-style)
    RANDOMIZATION_DEFAULTS   Per-param ranges (visual + spatial)
    RANDOMIZATION_GROUPS     Group → param-list mapping
"""

from __future__ import annotations

import math

# Upstream Isaac Lab MDP terms (reset_root_state_uniform, image, joint_pos, ...)
from isaaclab.envs.mdp import *  # noqa: F401,F403

# Local DR functions
from .resets import (  # noqa: F401
    randomize_cube_color,
    randomize_light_exposure,
    randomize_robot_color,
    randomize_static_asset_yaw,
)


# ---------------------------------------------------------------------------
# Color palette — discrete, comparable colors (Colosseum-style color_list)
# ---------------------------------------------------------------------------

COLOR_PALETTE: dict[str, tuple[float, float, float]] = {
    "red":    (0.8, 0.2, 0.2),
    "green":  (0.2, 0.8, 0.2),
    "blue":   (0.2, 0.2, 0.8),
    "orange": (0.9, 0.5, 0.1),
    "yellow": (0.9, 0.9, 0.2),
    "purple": (0.6, 0.2, 0.8),
    "cyan":   (0.2, 0.8, 0.8),
    "white":  (0.95, 0.95, 0.95),
}


# ---------------------------------------------------------------------------
# Default ranges per group — override via configure_randomization()
#
# trajectory_independent: Whether the property can be varied WITHOUT re-recording
# the trajectory. Visual properties (lighting, colors) can be replayed with
# different values — the robot motion stays the same. Spatial / Physical
# properties change scene geometry or physics, requiring a new trajectory.
#
# This enables offline data augmentation: record 1 trajectory, replay N× with
# different visual properties → N× more training data without new recordings.
# ---------------------------------------------------------------------------

RANDOMIZATION_DEFAULTS: dict = {
    # --- VISUAL (trajectory_independent=True) ---
    "room_light_exposure": (-3.0, 1.0),
    "env_light_exposure": (-2.0, 4.0),
    "sky_dome_yaw": (math.pi / 2, -math.pi / 2),
    "robot_color": list(COLOR_PALETTE.values()),
    "cube_color": list(COLOR_PALETTE.values()),
    # --- SPATIAL (trajectory_independent=False) ---
    "cube_x": (-0.05, 0.05),
    "cube_y": (-0.05, 0.05),
    "action_pad_yaw": (math.pi - 0.2, math.pi + 0.2),
    # --- PHYSICAL (reserved) ---
    # "cube_mass": (0.03, 0.08),
    # "cube_friction": (0.5, 1.0),
    # "cube_size": (0.03, 0.05),
}

RANDOMIZATION_GROUPS: dict = {
    "visual": {
        "trajectory_independent": True,
        "params": [
            "room_light_exposure", "env_light_exposure", "sky_dome_yaw",
            "robot_color", "cube_color",
        ],
    },
    "spatial": {
        "trajectory_independent": False,
        "params": ["cube_x", "cube_y", "action_pad_yaw"],
    },
    "physical": {
        "trajectory_independent": False,
        "params": [],  # reserved
    },
}

# Zero-ranges for disabling randomization per group
_ZERO_VISUAL: dict = {
    "room_light_exposure": (-1.0, -1.0),   # midpoint of (-3, 1) — tested in hemisphere survey
    "env_light_exposure": (1.0, 1.0),      # midpoint of (-2, 4) — tested in hemisphere survey
    "sky_dome_yaw": (math.pi, math.pi),    # fixed orientation
    "robot_color": [(0.95, 0.95, 0.95)],   # fixed white
    "cube_color": [(0.8, 0.2, 0.2)],       # fixed red
}
_ZERO_SPATIAL: dict = {
    "cube_x": (0.0, 0.0),
    "cube_y": (0.0, 0.0),
    "action_pad_yaw": (math.pi, math.pi),  # fixed at 180°
}


def configure_randomization(
    event_cfg,
    groups: set[str] | None = None,
    overrides: dict | None = None,
) -> None:
    """Enable/disable randomization groups on a LiftCubeEventCfg instance.

    Mutates the EventCfg in-place — must be called BEFORE `gym.make()` since
    the EventManager snapshots the event-term params at scene-construction time.

    Args:
        event_cfg: The event config to modify in-place. Duck-typed: must
            expose `reset_room_light_exposure`, `reset_env_light_exposure`,
            `reset_sky_dome_orientation`, `reset_robot_color`,
            `reset_cube_color`, `reset_cube`, `reset_action_pad_orientation`
            attributes (matches LiftCubeEventCfg in
            `tasks/lift_cube_env_cfg.py`).
        groups: Set of enabled groups: subset of {"visual", "spatial"}.
            None → all enabled (default). Empty set → all disabled (used by
            the LiftCube-Sim base task = ID-eval, no DR).
        overrides: Per-parameter overrides. Example:
            {"cube_x": (-0.1, 0.1), "cube_y": (-0.1, 0.1)}.
    """
    if groups is None:
        groups = {"visual", "spatial"}
    if overrides is None:
        overrides = {}

    # Start from defaults, apply overrides
    params = dict(RANDOMIZATION_DEFAULTS)
    params.update(overrides)

    # Zero out disabled groups
    if "visual" not in groups:
        params.update(_ZERO_VISUAL)
    if "spatial" not in groups:
        params.update(_ZERO_SPATIAL)

    # Apply to event terms
    # VISUAL
    event_cfg.reset_room_light_exposure.params["exposure_range"] = params["room_light_exposure"]
    event_cfg.reset_env_light_exposure.params["exposure_range"] = params["env_light_exposure"]
    event_cfg.reset_sky_dome_orientation.params["yaw_range"] = params["sky_dome_yaw"]
    event_cfg.reset_robot_color.params["color_list"] = params["robot_color"]
    event_cfg.reset_cube_color.params["color_list"] = params["cube_color"]

    # SPATIAL
    event_cfg.reset_cube.params["pose_range"] = {
        "x": params["cube_x"], "y": params["cube_y"],
    }
    event_cfg.reset_action_pad_orientation.params["yaw_range"] = params["action_pad_yaw"]
