"""Joint-space motion-profile primitives for "move to a pose" (sim + real).

Pure stdlib (math + time) — ZERO Isaac Lab / lerobot / numpy deps — so both the
sim teleop (Isaac env loop) and the real robot (send_action loop) import the SAME
cosine glide instead of each carrying a private copy. The execution loop is
injected via send_fn / get_pose_fn callbacks; this module owns ONLY the motion
profile, nothing robot- or sim-specific.

Consolidates the cosine S-curve duplicated (byte-identical formula
``alpha = (1 - cos(prog*pi)) / 2``) in:
  - artefacts/experiments/servo_jitter_tuning/elbow_diag.py::goto
  - artefacts/experiments/servo_jitter_tuning/run_jitter_experiment.py
  - lerobot_experiments/scripts/automated_jitter_benchmark.py
  - lerobot_so101_teleop_experiments/teleop/home_macro.py (sim)
"""
from __future__ import annotations

import math
import time
from typing import Callable, Mapping

__all__ = ["cosine_alpha", "interp_pose", "glide"]


def cosine_alpha(progress: float) -> float:
    """Cosine S-curve easing (smooth accel/decel): ``(1 - cos(p*pi)) / 2``.

    Clamps ``progress`` to ``[0, 1]``. ``alpha(0)=0``, ``alpha(0.5)=0.5``,
    ``alpha(1)=1``, monotonically increasing. Identical to the inline formula
    previously duplicated across the four call sites listed in the module docstring.
    """
    p = 0.0 if progress < 0.0 else 1.0 if progress > 1.0 else progress
    return (1.0 - math.cos(p * math.pi)) / 2.0


def interp_pose(
    start: Mapping[str, float],
    target: Mapping[str, float],
    alpha: float,
) -> dict[str, float]:
    """Per-joint linear blend ``start + (target - start) * alpha`` for each joint in ``target``.

    Only joints present in ``target`` are returned — a joint omitted from ``target``
    is never commanded (this is how the gripper is HELD during a home glide).
    ``alpha=0`` -> ``start`` values, ``alpha=1`` -> ``target`` values.
    """
    return {m: start[m] + (target[m] - start[m]) * alpha for m in target}


def glide(
    send_fn: Callable[[dict[str, float]], None],
    get_pose_fn: Callable[[], Mapping[str, float]],
    target: Mapping[str, float],
    duration_s: float,
    hz: float = 30.0,
    *,
    now_fn: Callable[[], float] = time.perf_counter,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, float]:
    """Cosine S-curve glide from the current pose to ``target`` over ``duration_s``.

    Moves exactly the joints in ``target`` — omit a joint (e.g. the gripper) to HOLD
    it. The start pose is read ONCE via ``get_pose_fn()`` (a joint missing there falls
    back to its ``target`` value, matching the original ``goto``). Each tick computes
    the interpolated pose and passes it to ``send_fn``. ``now_fn`` / ``sleep_fn`` are
    injectable for deterministic testing. Returns the final commanded pose (== target).

    Reproduces the duplicated real-robot ``goto()`` loop exactly:
        ``prog = min(1, (now-t0)/dur); cmd = interp(start, target, cosine_alpha(prog)); send; sleep(1/hz)``

    Raises:
        ValueError: if ``duration_s <= 0`` or ``hz <= 0`` (the originals would divide
            by zero on ``duration_s == 0``; this fails loud instead).
    """
    if duration_s <= 0.0:
        raise ValueError(f"duration_s must be > 0, got {duration_s}")
    if hz <= 0.0:
        raise ValueError(f"hz must be > 0, got {hz}")
    current = get_pose_fn()
    start = {m: (current[m] if m in current else target[m]) for m in target}
    t0 = now_fn()
    cmd = dict(start)
    while True:
        prog = min(1.0, (now_fn() - t0) / duration_s)
        cmd = interp_pose(start, target, cosine_alpha(prog))
        send_fn(cmd)
        if prog >= 1.0:
            break
        sleep_fn(1.0 / hz)
    return cmd
