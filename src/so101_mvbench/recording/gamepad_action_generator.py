# SPDX-License-Identifier: MIT
"""Gamepad-driven action generator for human teleoperation.

Wraps ``gamepad_utils.JointGamepad`` in absolute mode to produce action dicts
compatible with ``LeRobotSO101Interface.real_to_sim_obs_processor()``.

Same interface as ``SineWaveActionGenerator``: call ``get_action()`` each frame
to get a dict of joint positions in LeRobot raw units.

The neutral (start) position is computed at runtime from the simulation's
init_state using ``get_raw_actions_from_radians()``, ensuring perfect alignment
with whatever USD model is loaded.
"""

from __future__ import annotations

__version__ = "1.0.0"

import logging
from importlib.resources import files as pkg_files
from pathlib import Path

import torch

from so101_mvbench.vendor.gamepad_utils import JointGamepad

logger = logging.getLogger(__name__)

# Canonical joint order from lerobot_so101_teleop
SO101_JOINT_ORDER = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

# Default config shipped with gamepad_utils
_DEFAULT_CONFIG = Path(
    str(pkg_files("so101_mvbench.vendor.gamepad_utils").joinpath("configs/so101_lerobot_teleop.json"))
)


class GamepadActionGenerator:
    """Gamepad action generator that outputs LeRobot raw-unit joint positions.

    Args:
        coord_iface: ``LeRobotSO101Interface`` instance (provides coordinate
            transforms and joint range info).
        initial_obs_rad: Initial joint positions from ``env.reset()`` observation
            (policy obs tensor, in radians, shape ``[6]``).
        config_path: Path to gamepad JSON config. If None, uses the default
            ``so101_lerobot_teleop.json`` shipped with ``gamepad_utils``.
    """

    def __init__(
        self,
        coord_iface,
        initial_obs_rad: torch.Tensor,
        config_path: str | Path | None = None,
    ):
        # Convert sim init_state (radians) -> LeRobot raw units
        neutral_raw: torch.Tensor = coord_iface.get_raw_actions_from_radians(initial_obs_rad)

        # Build neutral_pos dict for JointGamepad (keys without .pos suffix)
        joint_names = [j.split(".")[0] for j in SO101_JOINT_ORDER]
        neutral_dict = {
            name: neutral_raw[i].item()
            for i, name in enumerate(joint_names)
        }

        logger.info("GamepadActionGenerator: neutral positions (raw): %s", neutral_dict)

        # Resolve config path
        if config_path is None:
            config_path = _DEFAULT_CONFIG
        config_path = Path(config_path)

        logger.info("GamepadActionGenerator: config=%s", config_path)

        # JointGamepad in absolute mode — output values are in raw units
        self._gamepad = JointGamepad(
            config_path=config_path,
            sim_device="cpu",
            joint_order=joint_names[:5],  # body joints only (gripper handled separately)
            neutral_pos=neutral_dict,
        )

        logger.info("GamepadActionGenerator v%s initialized", __version__)
        logger.info("%s", self._gamepad)

    def get_action(self) -> dict[str, float]:
        """Return absolute joint positions in LeRobot raw units.

        Returns:
            Dict with keys like ``"shoulder_pan.pos"`` and float values
            in the range [-100, 100] for body joints, [0, 100] for gripper.
        """
        return self._gamepad.get_action()

    def reset(self):
        """Reset all joints to the neutral (init_state) position."""
        self._gamepad.reset()
