"""Multi-env eval helpers — per-env obs slicing + per-env tracker init.

Role in the data flow:

- Data flow 3 (per-env inference): :func:`slice_visual_obs` fixes **Risk K-1** —
  the upstream ``sim_obs_to_policy_processor`` internally indexes ``visual[...][0]``,
  i.e. without slicing ALL envs would see env-0's image.
- Data flow 5 (per-env tracking): :func:`init_trackers` instantiates the existing
  ``LiftCubeEvalTracker`` N times (ADR-003 — no new vectorized tracker), each seeded
  from its own initial pose.

GPU-free: no Isaac Lab imports; unit-testable with CPU tensors / mocks
(``tests/test_multienv_obs_utils.py``).
"""

from __future__ import annotations

from typing import Any

import torch

from so101_mvbench.evaluation.eval_tracker import LiftCubeEvalTracker

__version__ = "0.1.0"


def slice_visual_obs(visual_obs: dict[str, Any], env_idx: int) -> dict[str, Any]:
    """Per-env view of the visual obs dict (Risk-K-1 fix).

    ``obs["visual"]`` maps camera-key -> tensor of shape ``(num_envs, H, W, C)``.
    The upstream ``sim_obs_to_policy_processor`` internally grabs ``visual[...][0]``;
    to feed environment ``env_idx`` we slice each camera tensor to
    ``[env_idx:env_idx+1]`` (the length-1 batch dim is preserved) — the downstream
    ``[0]`` then selects exactly ``env_idx``'s image instead of always env 0.

    Args:
        visual_obs: Dict camera-key -> tensor ``(num_envs, ...)``.
        env_idx: Environment index to extract.

    Returns:
        Dict camera-key -> tensor ``(1, ...)`` (view/slice, not a copy).
    """
    return {key: cam[env_idx : env_idx + 1] for key, cam in visual_obs.items()}


def init_trackers(
    n_envs: int,
    cube: Any,
    robot: Any,
    jaw_body_idx: int,
    home_pose_rad: list[float],
    *,
    home_pose_tol_rad: float | None = None,
    plateau_steps: dict[str, int] | None = None,
) -> list[LiftCubeEvalTracker]:
    """Build one ``LiftCubeEvalTracker`` per environment (ADR-003).

    Mirrors the single-env construction in ``eval_session.setup_episode`` but indexes
    each Isaac tensor per environment ``[i]`` instead of ``[0]``.

    Args:
        n_envs: Number of parallel environments.
        cube: Isaac RigidObject (``cube.data.root_link_pos_w`` shape ``(N, 3+)``).
        robot: Isaac Articulation (``robot.data.body_link_pos_w`` shape ``(N, B, 3)``).
        jaw_body_idx: Body index of the gripper jaw.
        home_pose_rad: Target home pose (>=5 arm joints).
        home_pose_tol_rad: Per-joint home tolerance (None -> tracker default).
        plateau_steps: Per-stage plateau early-termination budgets.

    Returns:
        List of ``n_envs`` independent trackers, each seeded from its env's initial
        cube height + gripper->cube distance.
    """
    trackers: list[LiftCubeEvalTracker] = []
    for i in range(n_envs):
        cube_xyz = cube.data.root_link_pos_w[i, :3]
        gripper_xyz = robot.data.body_link_pos_w[i, jaw_body_idx, :3]
        initial_dist = torch.norm(gripper_xyz - cube_xyz).item()
        trackers.append(
            LiftCubeEvalTracker(
                cube_xyz[2].item(),
                initial_dist,
                home_pose_rad,
                home_pose_tol_rad=home_pose_tol_rad,
                plateau_steps=plateau_steps,
            )
        )
    return trackers
