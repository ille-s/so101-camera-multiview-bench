# SPDX-License-Identifier: MIT
"""Gamepad Adapter - Loads Bluetooth Xbox profile from master_config.json."""

from __future__ import annotations

__version__ = "1.0.0"

import json
import logging
import os
from pathlib import Path

try:
    from so101_mvbench.logging import get_logger
    logger = get_logger(__name__)
except ImportError:  # vendored copy used standalone
    logger = logging.getLogger(__name__)


def _find_workspace_root() -> Path | None:
    """Find the workspace root via WORKSPACE_ROOT env var or walk-up."""
    ws = os.environ.get("WORKSPACE_ROOT")
    if ws:
        return Path(ws)
    # Walk up from this file looking for master_config.json
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "master_config.json").exists():
            return parent
    return None


class GamepadAdapter:
    """Standardized loader for the Bluetooth gamepad profile from master config."""

    @staticmethod
    def get_bluetooth_sim_config() -> dict:
        """Load and convert the Bluetooth profile from master_config.json.

        Returns a JointGamepad-compatible config dict with axis/button
        mappings and limits for joint-space control.
        """
        workspace_root = _find_workspace_root()
        if workspace_root is None:
            logger.error("Workspace root not found (no WORKSPACE_ROOT env var, no master_config.json in parents)")
            return {}

        config_path = workspace_root / "master_config.json"
        if not config_path.exists():
            config_path = workspace_root / "config" / "hardware" / "lerobot_master_config.json"

        if not config_path.exists():
            logger.error("Master config not found at %s", config_path)
            return {}

        try:
            with open(config_path) as f:
                master_cfg = json.load(f)

            profile = master_cfg.get("gamepad", {}).get("profiles", {}).get("Xbox Wireless Controller", {})
            if not profile:
                logger.error("Profile 'Xbox Wireless Controller' not found in master config.")
                return {}

            btns = profile.get("buttons", {})
            axs = profile.get("axes", {})

            logger.info("GamepadAdapter: Bluetooth profile (Xbox Wireless) loaded from %s", config_path)

            return {
                "dead_zone": 0.20,
                "global_scale": 0.02,
                "limits_unit": "deg",
                "joints": {
                    "shoulder_pan":  {"axis": axs.get("LX", 0), "scale": 1.0, "limits": [-80, 80]},
                    "shoulder_lift": {"axis": axs.get("LY", 1), "scale": 0.8, "invert": True, "limits": [-90, 90]},
                    "elbow_flex":    {"axis": axs.get("RY", 3), "scale": 0.8, "invert": True, "limits": [-90, 85]},
                    "wrist_flex":    {"axis": axs.get("RX", 2), "scale": 0.5, "limits": [-90, 65]},
                    "wrist_roll":    {"pos_button": btns.get("RB", 7), "neg_button": btns.get("LB", 6), "scale": 1.2, "limits": [-45, 45]},
                },
                "gripper": {
                    "open_axis": axs.get("LT", 4),
                    "close_axis": axs.get("RT", 5),
                    "trigger_threshold": 0.1,
                    "open_pos": 0.785,
                    "close_pos": -0.25,
                    "start_pos": 0.25,
                    "scale": 0.6,
                },
            }

        except Exception as e:
            logger.error("GamepadAdapter: Error loading Bluetooth config: %s", e)
            return {}
