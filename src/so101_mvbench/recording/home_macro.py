"""HomeMacro — leisaac-style PD-hold Auto-Pilot to default_joint_pos.

Y-button triggers a smooth 1.5s return to the neutral pose while preserving
the current gripper state. Produces BOTH the sim-space action (zero-offset
tensor for env.step) and the raw-unit action for parquet logging, so
observation.state and action remain consistent during the macro.

Defensive Level B:

- Shape sanity checks in __init__ and trigger()
- NaN guards on every physics read (current_pos, default_pos, final_pos)
- try/except around coord_iface calls (graceful error + abort)
- the exception handler in step() sets _steps_remaining=0 on failure
- Logs success only below 8.6 deg and warns only above 17 deg; in between it
  stays quiet, so a normal run does not bury the log in noise
- The init log prints every configuration parameter

Sync strategy:

- Normal completion (N ticks expired): action_gen.reset() → gamepad to neutral
- Cancel mid-glide (S/R key): _sync_gamepad_to_current_pos() → gamepad to
  actual sim state, preventing snap-back on next advance() call
"""

from __future__ import annotations

__version__ = "1.4.1"

import json
import math
from pathlib import Path

import torch

try:
    from so101_mvbench.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

_HOLD_STEPS_DEFAULT = 45  # 1.5 s @ 30 Hz
# "close enough to home" -- this is a teleop convenience move, not a precision task.
_HOME_REACHED_THRESHOLD_RAD = 0.15  # ~8.6 deg
# Beyond this the arm has barely moved at all, which means something is wrong.
_STUCK_THRESHOLD_RAD = 0.30         # ~17 deg
_EXPECTED_N_JOINTS = 6              # 5 arm + 1 gripper
_TRAPEZOID_ACCEL_FRAC = 1.0 / 3.0   # 1/3 accel + 1/3 cruise + 1/3 decel (classic)


def _trapezoidal_profile(t: float, accel_frac: float = _TRAPEZOID_ACCEL_FRAC) -> float:
    """Trapezoidal velocity profile — t ∈ [0, 1] → position ∈ [0, 1].

    3 phases:
      Phase 1 (t < f)        : accelerate with constant a — position is quadratic
      Phase 2 (f ≤ t ≤ 1-f)  : cruise at v_max — position is linear
      Phase 3 (t > 1-f)      : decelerate with constant -a — position is mirrored quadratic

    v_max chosen so total displacement = 1. For f=1/3: v_max = 1.5.
    Preferred over smoothstep 3t²-2t³ because max acceleration at t=0 is lower
    (constant a = v_max/f = 4.5 vs smoothstep's 6) and ramps predictably.
    """
    f = accel_frac
    if f <= 0.0 or f >= 0.5:
        raise ValueError(f"accel_frac must be in (0, 0.5), got {f}")
    v_max = 1.0 / (1.0 - f)
    if t < f:
        return 0.5 * v_max * t * t / f
    if t < 1.0 - f:
        return v_max * (t - 0.5 * f)
    return 1.0 - 0.5 * v_max * (1.0 - t) * (1.0 - t) / f

# The home pose is read from the shipped `config/robot_poses.json`. If that fails the
# code falls back to the environment's default_joint_pos, which is a DIFFERENT pose --
# see the warning in _load_home_from_master_config.
#
# The environment uses use_default_offset=False, so an action is an absolute joint
# angle in radians, not an offset from the current position.


class HomeMacro:
    """Y-button PD-hold auto-pilot to default_joint_pos.

    Args:
        env: Isaac Lab environment (for scene["robot"].data access).
        action_gen: GamepadActionGenerator instance (exposes _gamepad + reset).
        coord_iface: LeRobotSO101Interface (provides get_raw_actions_from_radians
            and .device).
        action_shape: Shape of the action tensor — must be (num_envs, n_joints).
        device: Target device for sim-space action tensor (typically env.device).
        hold_steps: Number of env.step() ticks to hold the macro (default 45).

    Raises:
        ValueError: If hold_steps <= 0 or action_shape is malformed.
        AttributeError: If coord_iface lacks get_raw_actions_from_radians.
    """

    def __init__(
        self,
        env,
        action_gen,
        coord_iface,
        action_shape: tuple[int, ...],
        device: str | torch.device,
        hold_steps: int = _HOLD_STEPS_DEFAULT,
    ) -> None:
        # --- Defensive Level B: Input validation ---
        if hold_steps <= 0:
            raise ValueError(f"hold_steps must be > 0, got {hold_steps}")
        if len(action_shape) != 2:
            raise ValueError(
                f"action_shape must be (num_envs, n_joints), got {action_shape}"
            )
        if action_shape[-1] != _EXPECTED_N_JOINTS:
            raise ValueError(
                f"action_shape[-1] must be {_EXPECTED_N_JOINTS} "
                f"(5 arm + 1 gripper), got {action_shape[-1]}"
            )
        if not hasattr(coord_iface, "get_raw_actions_from_radians"):
            raise AttributeError(
                "coord_iface missing 'get_raw_actions_from_radians' method"
            )
        if not hasattr(action_gen, "_gamepad"):
            logger.warning(
                "[Y-MACRO] action_gen has no _gamepad attr — cancel/sync will fail. "
                "Only safe with action_source=gamepad."
            )

        self._env = env
        self._action_gen = action_gen
        self._coord_iface = coord_iface
        self._hold_steps = hold_steps
        self._steps_remaining = 0
        self._sim_action: torch.Tensor | None = None
        self._real_action: torch.Tensor | None = None
        self._home_target_rad: torch.Tensor | None = None  # frozen at trigger time
        self._start_pos_rad: torch.Tensor | None = None    # frozen at trigger time (for interpolation)
        self._action_shape = action_shape
        self._device = device                      # env.device (for sim_action)
        self._coord_device = coord_iface.device    # for raw-unit conversion

        # Load home pose ONCE at init (static config, no runtime changes).
        self._home_arm_rad = self._load_home_from_master_config()

        logger.info(
            "[Y-MACRO] HomeMacro initialized | hold_steps=%d (%.2fs @ 30Hz) | "
            "action_shape=%s | env_device=%s | coord_device=%s | "
            "home_arm_rad=%s (deg=%s) | "
            "home_reached_threshold=%.2f rad (%.1f°)",
            hold_steps, hold_steps / 30.0,
            action_shape, device, self._coord_device,
            [round(v, 3) for v in self._home_arm_rad.tolist()],
            [round(math.degrees(v), 1) for v in self._home_arm_rad.tolist()],
            _HOME_REACHED_THRESHOLD_RAD, math.degrees(_HOME_REACHED_THRESHOLD_RAD),
        )

    def _load_home_from_master_config(self) -> torch.Tensor:
        """Load the home pose from the shipped ``config/robot_poses.json`` resource.

        Reads `poses.home` (unit: deg) and converts to radians.
        Returns arm-only pose (first 5 joints, gripper excluded — preserved at trigger).
        Falls back to env.default_joint_pos[:5] if the resource is missing or malformed.
        """
        # Package data, so it is resolved relative to __file__ and never via SETTINGS
        # (see so101_mvbench.settings: shipped data is not user-configurable).
        config_path = Path(__file__).parent / "config" / "robot_poses.json"
        try:
            cfg = json.loads(config_path.read_text())
            unit = cfg.get("unit", "deg")
            home_list = cfg.get("poses", {}).get("home")
            if home_list is None:
                raise KeyError("isaaclab.poses.home not found in master_config")
            if len(home_list) < 5:
                raise ValueError(
                    f"isaaclab.poses.home must have >= 5 entries, got {len(home_list)}"
                )
            home_arr = torch.tensor(
                home_list[:5], dtype=torch.float32, device=self._device
            )
            if unit == "deg":
                home_arr = home_arr * (math.pi / 180.0)
            elif unit != "rad":
                raise ValueError(f"Unknown unit '{unit}' in isaaclab config")
            logger.info(
                "[Y-MACRO] Home pose loaded from %s (unit=%s)",
                config_path.name, unit,
            )
            return home_arr.clone()
        except Exception as exc:
            logger.warning(
                "[Y-MACRO] Could not load home from master_config (%s) — "
                "fallback to env default_joint_pos[:5]", exc,
            )
            robot = self._env.unwrapped.scene["robot"]
            return robot.data.default_joint_pos[0][:5].clone().to(self._device)

    @property
    def is_active(self) -> bool:
        """True if a macro is currently running."""
        return self._steps_remaining > 0

    def trigger(self) -> None:
        """Y-button callback — snapshot current gripper + build macro tensors."""
        if self._steps_remaining > 0:
            logger.info(
                "[Y-MACRO] Y pressed — already running, ignored (%d steps left)",
                self._steps_remaining,
            )
            return

        try:
            robot = self._env.unwrapped.scene["robot"]
            current_pos = robot.data.joint_pos[0].clone()
            default_pos = robot.data.default_joint_pos[0]

            # Shape-Sanity-Check
            if current_pos.shape[0] != _EXPECTED_N_JOINTS:
                logger.error(
                    "[Y-MACRO] joint_pos shape mismatch: expected (%d,), got %s — aborted",
                    _EXPECTED_N_JOINTS, tuple(current_pos.shape),
                )
                return

            # NaN-Guard
            if torch.any(torch.isnan(current_pos)) or torch.any(torch.isnan(default_pos)):
                logger.error(
                    "[Y-MACRO] NaN in joint_pos (current=%s, default=%s) — trigger aborted",
                    current_pos.tolist(), default_pos.tolist(),
                )
                return

            # Target pose for the glide:
            # - arm joints 0..4 go to the configured home pose
            # - the gripper (index 5) holds its current position, so the cube stays held
            home_target_rad = torch.empty(
                _EXPECTED_N_JOINTS, dtype=current_pos.dtype, device=self._device
            )
            home_target_rad[:5] = self._home_arm_rad
            home_target_rad[5] = current_pos[-1]

            # CRITICAL: env has use_default_offset=False → action = absolute rad.
            # INTERPOLATION (not constant PD-hold) — prevents initial jerk that flings the cube.
            # At step(), compute s = smoothstep((N-remaining)/N) and target = start + s*(home-start).
            # Smoothstep: s(t) = 3t² - 2t³ → zero derivative at t=0 and t=1 → zero-jerk boundaries.
            self._start_pos_rad = current_pos.clone().to(self._device)  # frozen at trigger
            self._home_target_rad = home_target_rad.clone()

            # Populate with start_pos initially (zero delta → no jerk at t=0).
            # step() will compute the interpolated target each tick.
            self._sim_action = self._start_pos_rad.unsqueeze(0).expand(self._action_shape).clone()
            self._real_action = self._coord_iface.get_raw_actions_from_radians(
                self._start_pos_rad.clone().to(self._coord_device)
            )

        except Exception:
            logger.exception("[Y-MACRO] trigger failed — macro NOT started")
            self._sim_action = None
            self._real_action = None
            self._home_target_rad = None
            self._steps_remaining = 0
            return

        self._steps_remaining = self._hold_steps
        max_arm_move_deg = math.degrees(
            (current_pos[:-1] - self._home_arm_rad.to(current_pos.device)).abs().max().item()
        )
        logger.info(
            "[Y-MACRO] Home pressed — triggered | arm→home_pose=%s° | "
            "max_move=%.2f° | gripper HOLD @ %.3f rad | hold=%d steps (%.2fs @ 30Hz)",
            [round(math.degrees(v), 1) for v in self._home_arm_rad.tolist()],
            max_arm_move_deg, current_pos[-1].item(),
            self._hold_steps, self._hold_steps / 30.0,
        )

    def step(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Advance macro by one tick.

        Returns:
            Tuple (sim_action, real_action) to override the main-loop's action,
            or None if macro is inactive.
        """
        if self._steps_remaining == 0:
            return None
        try:
            self._steps_remaining -= 1

            # Trapezoidal velocity profile (User-Request 2026-04-18):
            #   Phase 1 (t < 1/3) : accel — constant acceleration, quadratic position
            #   Phase 2 (1/3..2/3): cruise — constant velocity, linear position
            #   Phase 3 (t > 2/3) : decel — constant deceleration, quadratic position
            # Vorteil gegen Smoothstep: peak acceleration 4.5 vs 6, no "schwinger" von
            # instant max-accel bei t=0 (smoothstep s''(0)=6, trapez ramps linear).
            elapsed = self._hold_steps - self._steps_remaining
            t = elapsed / float(self._hold_steps)
            s = _trapezoidal_profile(t)

            interp_rad = (
                self._start_pos_rad + s * (self._home_target_rad - self._start_pos_rad)
            )
            # Gripper: interp result is overwritten by main-loop gamepad pass-through
            # (macro only controls arm joints 0..4). Leave interp_rad[5] as-is;
            # gamepad_recorder.py will do `actions[:, :5] = macro[:, :5]`.

            sim_action = interp_rad.unsqueeze(0).expand(self._action_shape).clone()
            commanded_rad = interp_rad.clone().to(self._coord_device)
            real_action = self._coord_iface.get_raw_actions_from_radians(commanded_rad)

            self._sim_action = sim_action
            self._real_action = real_action

            if self._steps_remaining == 0:
                self._log_home_reached_and_complete()
                # Sync gamepad to WHERE THE ARM CURRENTLY IS (not neutral).
                self._sync_gamepad_to_current_pos()
                self._sim_action = None
                self._real_action = None
                self._home_target_rad = None
                self._start_pos_rad = None
            return sim_action, real_action
        except Exception:
            logger.exception("[Y-MACRO] step() failed — resetting macro state")
            self._steps_remaining = 0
            self._sim_action = None
            self._real_action = None
            self._home_target_rad = None
            self._start_pos_rad = None
            return None

    def cancel(self) -> None:
        """Abort macro on S/R keypress — sync gamepad to actual sim state."""
        if self._steps_remaining > 0:
            logger.info(
                "[Y-MACRO] cancelled (%d steps left) — syncing gamepad to sim state",
                self._steps_remaining,
            )
            try:
                self._sync_gamepad_to_current_pos()
            except Exception:
                logger.exception("[Y-MACRO] sync_gamepad failed during cancel — resetting")
                self._action_gen.reset()  # fallback
            self._sim_action = None
            self._real_action = None
            self._home_target_rad = None
            self._start_pos_rad = None
            self._steps_remaining = 0

    def _log_home_reached_and_complete(self) -> None:
        """Log home-reached status at macro-end. Target = _home_target_rad (arm joints only)."""
        try:
            if self._home_target_rad is None:
                return
            robot = self._env.unwrapped.scene["robot"]
            final_pos = robot.data.joint_pos[0]
            if torch.any(torch.isnan(final_pos)):
                logger.warning(
                    "[Y-MACRO] complete — NaN in final_pos, home-check skipped"
                )
                return
            # Compare ARM joints only (0..4) — gripper was intentionally held, not driven to home.
            target_arm = self._home_target_rad[:-1].to(final_pos.device)
            final_arm = final_pos[:-1]
            error = (final_arm - target_arm).abs()
            max_err = error.max().item()
            max_err_deg = math.degrees(max_err)

            if max_err <= _HOME_REACHED_THRESHOLD_RAD:
                logger.info(
                    "[Y-MACRO] HOME REACHED ✓ | max_arm_error=%.2f°",
                    max_err_deg,
                )
            elif max_err > _STUCK_THRESHOLD_RAD:
                # Only worth a warning when the arm barely moved -- that is a real fault
                logger.warning(
                    "[Y-MACRO] home NOT reached | max_arm_error=%.2f° "
                    "(>%.1f° stuck-threshold) | "
                    "arm barely moved; check the PD gains or raise hold_steps.",
                    max_err_deg, math.degrees(_STUCK_THRESHOLD_RAD),
                )
            # Between the two thresholds: no log. Overshooting slightly is normal and
            # logging it every episode would drown the interesting lines.
        except Exception:
            logger.exception("[Y-MACRO] home-reached check failed (non-fatal)")

    def _sync_gamepad_to_current_pos(self) -> None:
        """Read actual sim joint_pos, write into gamepad's internal state.

        Without this, canceling mid-glide causes a snap-back: gamepad thinks
        it's at the target (neutral), but arm is somewhere in between. Next
        advance() produces wrong offsets relative to home → hard snap.

        Ported from leisaac_experiments/.../fsm.py::_sync_gamepad_to_current_pos.
        """
        robot = self._env.unwrapped.scene["robot"]
        current_rad = robot.data.joint_pos[0].clone().to(self._coord_device)

        if torch.any(torch.isnan(current_rad)):
            logger.error(
                "[Y-MACRO] NaN in current joint_pos during cancel — "
                "sync aborted, falling back to reset"
            )
            self._action_gen.reset()
            return

        current_raw = self._coord_iface.get_raw_actions_from_radians(current_rad)
        gamepad = self._action_gen._gamepad
        # Sync ARM joints only (5). Gripper (joint_values has shape (5,), gripper separate).
        gamepad._joint_values[:] = current_raw[:-1].cpu().numpy()
        # CRITICAL: do NOT overwrite gamepad._gripper_pos — the user's commanded gripper
        # position must persist across the macro. If we sync to measured current_raw[-1],
        # PD-settling + cube-contact force drift mean commanded ≠ measured, and writing
        # measured back to gamepad causes a step-change on next tick → cube drops.
        # Leisaac macht das Gleiche — gripper ist always user-commanded, macro touches arm only.
        logger.info(
            "[Y-MACRO] gamepad arm-synced to current (raw)=%s | gripper preserved @ %.2f",
            [round(v, 2) for v in current_raw[:-1].tolist()],
            gamepad._gripper_pos,
        )
