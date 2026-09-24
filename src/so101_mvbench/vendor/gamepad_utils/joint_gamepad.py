# SPDX-License-Identifier: MIT
"""Joint-space gamepad controller for SO-ARM101.

Reads a JSON config file to map individual gamepad axes or buttons to robot joints.
Each joint has its own axis/button assignment, scale, inversion, and dead zone.
Uses the Linux joystick API (/dev/input/js*) directly -- no SDL/GLFW/pygame.

Supports two output modes:
  - "offset" (default): returns accumulated offsets from zero (backward-compat)
  - "absolute": returns absolute joint positions starting from a neutral pose
"""

from __future__ import annotations

__version__ = "1.0.0"

import json
import logging
import os
import struct
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

try:
    from so101_mvbench.logging import get_logger
    logger = get_logger(__name__)
except ImportError:  # vendored copy used standalone
    logger = logging.getLogger(__name__)

from ._joystick_utils import (
    JS_EVENT_AXIS,
    JS_EVENT_BUTTON,
    JS_EVENT_FMT,
    JS_EVENT_INIT,
    JS_EVENT_SIZE,
    find_js_device,
    read_device_name,
)
from .gamepad_adapter import GamepadAdapter

# Default joint order (matches old JointPositionActionCfg joint_names)
DEFAULT_JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]


class JointGamepad:
    """Gamepad controller that maps axes or buttons directly to individual joints.

    Args:
        config_path: Path to JSON config file. If None, loads Bluetooth
            profile from master_config.json via GamepadAdapter.
        sim_device: Torch device for output tensors.
        joint_order: Joint name order for the output tensor. Defaults to
            DEFAULT_JOINT_ORDER. Must match the config's joint keys.
        neutral_pos: Dict mapping joint names to their neutral (start)
            positions in output units. Required for absolute mode.
            Keys must match ``joint_order`` entries. If None, all joints
            start at zero (offset mode behavior).
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        sim_device: str = "cuda:0",
        joint_order: list[str] | None = None,
        neutral_pos: dict[str, float] | None = None,
    ):
        self._joint_order = joint_order or list(DEFAULT_JOINT_ORDER)

        # Load config via Adapter (Bluetooth Default) or path
        if config_path is None:
            self._config = GamepadAdapter.get_bluetooth_sim_config()
            if not self._config:
                logger.error("Failed to load automatic gamepad config. Falling back to empty.")
                self._config = {}
        else:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Gamepad config not found: {config_path}")
            with open(config_path) as f:
                self._config = json.load(f)
            logger.info("Loaded gamepad config from path: %s", config_path)

        self._dead_zone = self._config.get("dead_zone", 0.05)
        self._global_scale = self._config.get("global_scale", 0.02)
        self._joint_cfgs = self._config.get("joints", {})
        self._gripper_cfg = self._config.get("gripper", {})
        self._sim_device = sim_device

        # Output mode: "offset" (backward-compat) or "absolute"
        self._absolute = self._config.get("output_mode", "offset") == "absolute"

        # Unit conversion for limits ("deg" → radians, "raw"/"rad" → keep as-is)
        limits_unit = self._config.get("limits_unit", "rad")
        self._deg2rad = limits_unit == "deg"

        # Neutral positions for absolute mode
        self._neutral_pos = neutral_pos or {}

        # Build axis/button->joint mapping and collect limits
        self._axis_to_joint: dict[int, dict] = {}
        self._button_to_joint: list[dict] = []
        self._joint_limits: list[tuple[float, float]] = []

        for joint_name in self._joint_order:
            jcfg = self._joint_cfgs.get(joint_name)
            if jcfg is None:
                logger.warning("Joint '%s' not in config, will not be controlled.", joint_name)
                self._joint_limits.append((-3.14, 3.14))
                continue

            joint_info = {
                "joint": joint_name,
                "index": self._joint_order.index(joint_name),
                "scale": jcfg.get("scale", 1.0),
                "invert": jcfg.get("invert", False),
                "dead_zone": jcfg.get("dead_zone", self._dead_zone),
            }

            if "axis" in jcfg:
                self._axis_to_joint[jcfg["axis"]] = joint_info

            if "pos_button" in jcfg or "neg_button" in jcfg:
                btn_info = joint_info.copy()
                btn_info["pos_button"] = jcfg.get("pos_button")
                btn_info["neg_button"] = jcfg.get("neg_button")
                self._button_to_joint.append(btn_info)

            lim = jcfg.get("limits", [-180, 180])
            if self._deg2rad:
                lim = [np.radians(lim[0]), np.radians(lim[1])]
            # "raw" and "rad" both keep values as-is
            self._joint_limits.append((lim[0], lim[1]))

        # Gripper limits
        grip_lo = self._gripper_cfg.get("close_pos", 0.0)
        grip_hi = self._gripper_cfg.get("open_pos", 1.0)
        self._joint_limits.append((grip_lo, grip_hi))

        # Open joystick
        js_path = find_js_device()
        if js_path is None:
            raise RuntimeError("No joystick device found at /dev/input/js*")
        self._fd = os.open(js_path, os.O_RDONLY | os.O_NONBLOCK)
        self._js_path = js_path
        self._name = read_device_name(self._fd)

        logger.info("JointGamepad: %s (%s) mode=%s",
                     self._name, js_path, "absolute" if self._absolute else "offset")

        # State
        self._axes: dict[int, float] = {}
        self._buttons: dict[int, bool] = {}
        self._additional_callbacks: dict[int, Callable] = {}

        # Joint state: offsets (offset mode) or absolute positions (absolute mode)
        n_joints = len(self._joint_order)
        if self._absolute:
            self._joint_values = np.array(
                [self._neutral_pos.get(j, 0.0) for j in self._joint_order],
                dtype=np.float64,
            )
        else:
            self._joint_values = np.zeros(n_joints, dtype=np.float64)

        self._neutral_array = self._joint_values.copy()

        # Gripper state
        self._gripper_pos = self._gripper_cfg.get(
            "start_pos",
            self._gripper_cfg.get("close_pos", 0.0),
        )
        if self._absolute and "gripper" in self._neutral_pos:
            self._gripper_pos = self._neutral_pos["gripper"]
        self._gripper_neutral = self._gripper_pos

        # Drain initial events
        self._drain_events()

        # Log mapping
        for axis_id, info in sorted(self._axis_to_joint.items()):
            inv = " (inverted)" if info["invert"] else ""
            logger.info("  Axis %d -> %s (scale=%s%s)", axis_id, info["joint"], info["scale"], inv)
        for info in self._button_to_joint:
            logger.info("  Btns %s/%s -> %s (scale=%s)",
                        info["pos_button"], info["neg_button"], info["joint"], info["scale"])

    def __del__(self):
        try:
            os.close(self._fd)
        except Exception:
            pass

    @property
    def joint_limits(self) -> list[tuple[float, float]]:
        """Return (lo, hi) limits for each joint + gripper."""
        return self._joint_limits

    @property
    def joint_labels(self) -> list[str]:
        """Return joint names + gripper label."""
        return list(self._joint_order) + ["gripper"]

    @property
    def joint_norm_scales(self) -> list[float]:
        """Scale factors for normalized display [-100, 100]."""
        scales = []
        for lo, hi in self._joint_limits[:len(self._joint_order)]:
            half = (hi - lo) / 2.0
            scales.append(100.0 / half if half > 0 else 1.0)
        return scales

    def __str__(self) -> str:
        msg = f"Joint Gamepad Controller: {self.__class__.__name__}\n"
        msg += f"\tDevice: {self._name} ({self._js_path})\n"
        msg += f"\tMode: {'absolute' if self._absolute else 'offset'}\n"
        msg += f"\tGlobal scale: {self._global_scale}, Dead zone: {self._dead_zone}\n"
        msg += "\t----------------------------------------------\n"
        for axis_id, info in sorted(self._axis_to_joint.items()):
            inv = " (inv)" if info["invert"] else ""
            msg += f"\tAxis {axis_id} -> {info['joint']:16s} scale={info['scale']}{inv}\n"
        for info in self._button_to_joint:
            msg += f"\tBtns {info['pos_button']}/{info['neg_button']} -> {info['joint']:16s} scale={info['scale']}\n"
        open_ax = self._gripper_cfg.get("open_axis", 4)
        close_ax = self._gripper_cfg.get("close_axis", 5)
        msg += f"\tAxis {open_ax} (LT) -> gripper open, Axis {close_ax} (RT) -> gripper close\n"
        return msg

    def reset(self):
        """Reset joint values to neutral/zero and gripper to start position."""
        self._joint_values[:] = self._neutral_array
        self._gripper_pos = self._gripper_neutral

    def add_callback(self, key: int, func: Callable):
        """Register a callback for a gamepad button press."""
        self._additional_callbacks[key] = func

    def _drain_events(self):
        """Read all pending joystick events (non-blocking)."""
        while True:
            try:
                data = os.read(self._fd, JS_EVENT_SIZE)
                if len(data) < JS_EVENT_SIZE:
                    break
                _, value, type_, number = struct.unpack(JS_EVENT_FMT, data)
                raw_type = type_ & ~JS_EVENT_INIT

                if raw_type == JS_EVENT_AXIS:
                    self._axes[number] = value / 32767.0
                elif raw_type == JS_EVENT_BUTTON:
                    prev = self._buttons.get(number, False)
                    pressed = value == 1
                    self._buttons[number] = pressed
                    if pressed and not prev and not (type_ & JS_EVENT_INIT):
                        if number in self._additional_callbacks:
                            self._additional_callbacks[number]()
            except BlockingIOError:
                break

    def advance(self) -> torch.Tensor:
        """Read gamepad and return joint values + gripper position.

        In offset mode: returns accumulated offsets from zero.
        In absolute mode: returns absolute joint positions.

        Returns:
            6D tensor: [joint_0, ..., joint_4, gripper_pos]
        """
        self._drain_events()

        # Accumulate per-joint deltas from AXES
        for axis_id, info in self._axis_to_joint.items():
            raw = self._axes.get(axis_id, 0.0)
            if abs(raw) < info["dead_zone"]:
                raw = 0.0
            delta = raw * self._global_scale * info["scale"]
            if info["invert"]:
                delta = -delta
            self._joint_values[info["index"]] += delta

        # Accumulate per-joint deltas from BUTTONS
        for info in self._button_to_joint:
            val = 0.0
            if info["pos_button"] is not None and self._buttons.get(info["pos_button"], False):
                val += 1.0
            if info["neg_button"] is not None and self._buttons.get(info["neg_button"], False):
                val -= 1.0

            if val != 0.0:
                delta = val * self._global_scale * info["scale"]
                if info["invert"]:
                    delta = -delta
                self._joint_values[info["index"]] += delta

        # Clamp joint values to limits
        for i, (lo, hi) in enumerate(self._joint_limits[:len(self._joint_order)]):
            self._joint_values[i] = np.clip(self._joint_values[i], lo, hi)

        # Gripper: Incremental control via LT (open) and RT (close)
        open_ax = self._gripper_cfg.get("open_axis", 4)
        close_ax = self._gripper_cfg.get("close_axis", 5)

        trig_min = self._gripper_cfg.get("trigger_min", -1.0)
        lt_val = (self._axes.get(open_ax, trig_min) - trig_min) / (1.0 - trig_min)
        rt_val = (self._axes.get(close_ax, trig_min) - trig_min) / (1.0 - trig_min)

        thr = self._gripper_cfg.get("trigger_threshold", 0.1)
        if lt_val < thr:
            lt_val = 0.0
        if rt_val < thr:
            rt_val = 0.0

        gripper_scale = self._gripper_cfg.get("scale", 1.0)
        gripper_delta = (lt_val - rt_val) * self._global_scale * gripper_scale

        open_limit = self._gripper_cfg.get("open_pos", 1.0)
        close_limit = self._gripper_cfg.get("close_pos", 0.0)

        mi, ma = min(close_limit, open_limit), max(close_limit, open_limit)
        self._gripper_pos = np.clip(self._gripper_pos + gripper_delta, mi, ma)

        command = np.append(self._joint_values, self._gripper_pos)
        return torch.tensor(command, dtype=torch.float32, device=self._sim_device)

    def get_action(self) -> dict[str, float]:
        """Return current joint positions as a dict with '.pos'-suffixed keys.

        Calls advance() internally, then formats the result as a dict
        compatible with LeRobotSO101Interface.real_to_sim_obs_processor().

        Returns:
            Dict like {"shoulder_pan.pos": float, "gripper.pos": float, ...}
        """
        tensor = self.advance()
        values = tensor.tolist()
        result = {}
        for i, joint_name in enumerate(self._joint_order):
            result[f"{joint_name}.pos"] = values[i]
        result["gripper.pos"] = values[-1]
        return result
