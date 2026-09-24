"""GPU-free unit tests for ``evaluation/multienv_obs_utils.py``.

Covers stage 1 of the multi-env eval MVP (see ADR canvas
``src/obsidian_msc/Research/eval-multienv-a1-architektur.canvas``):
- ``slice_visual_obs`` (Risk-K-1 fix): the env-1 slice MUST differ from the env-0 slice.
- ``init_trackers`` (ADR-003): N independent ``LiftCubeEvalTracker`` instances, each
  seeded from its own initial pose.

Runs without Isaac Sim / without GPU / without pytest:
    python3 tests/test_multienv_obs_utils.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

# Put the worktree `src` at the front of sys.path -> the worktree version (not the
# pip-installed main-checkout version) is tested.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from so101_mvbench.evaluation.eval_tracker import LiftCubeEvalTracker
from so101_mvbench.evaluation.multienv_obs_utils import (
    init_trackers,
    slice_visual_obs,
)

_HOME_POSE_RAD = [0.0, 0.0, 0.0, 1.57, -1.57, 0.0]


def test_slice_visual_obs_shape_and_per_env() -> None:
    """Slice keeps the length-1 batch dim and returns a DIFFERENT image per env."""
    n_envs, h, w, c = 3, 2, 2, 3
    visual = {"camera_external": torch.arange(n_envs * h * w * c).reshape(n_envs, h, w, c)}

    s0 = slice_visual_obs(visual, 0)
    s1 = slice_visual_obs(visual, 1)

    # Shape: (1, H, W, C) — batch dim preserved so downstream [0] indexes correctly.
    assert s1["camera_external"].shape == (1, h, w, c)
    # Slice == the correct env image (env_idx), not env 0.
    assert torch.equal(s1["camera_external"][0], visual["camera_external"][1])
    # Risk-K-1 core assert: env 1 != env 0 (else all envs would see env-0's image).
    assert not torch.equal(s0["camera_external"], s1["camera_external"])


def test_slice_visual_obs_multi_cam() -> None:
    """Multiple cameras are all sliced per env."""
    visual = {
        "camera_ego": torch.arange(2 * 1 * 1 * 3).reshape(2, 1, 1, 3),
        "camera_external": torch.arange(2 * 1 * 1 * 3).reshape(2, 1, 1, 3) + 100,
    }
    s = slice_visual_obs(visual, 1)
    assert set(s.keys()) == {"camera_ego", "camera_external"}
    assert s["camera_ego"].shape == (1, 1, 1, 3)
    assert torch.equal(s["camera_external"][0], visual["camera_external"][1])


def _mock_scene(cube_zs: list[float]) -> tuple[SimpleNamespace, SimpleNamespace, int]:
    """Mock cube/robot with per-env poses (CPU tensors, no Isaac)."""
    n = len(cube_zs)
    cube_pos = torch.tensor([[0.30, 0.0, z] for z in cube_zs], dtype=torch.float32)
    jaw_idx = 2
    n_bodies = 4
    body_pos = torch.zeros(n, n_bodies, 3, dtype=torch.float32)
    # Gripper (jaw) slightly above/beside the cube, minimally offset per env.
    for i in range(n):
        body_pos[i, jaw_idx] = torch.tensor([0.40 + 0.01 * i, 0.0, 0.20])
    cube = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=cube_pos))
    robot = SimpleNamespace(data=SimpleNamespace(body_link_pos_w=body_pos))
    return cube, robot, jaw_idx


def test_init_trackers_count_and_independence() -> None:
    """N trackers, distinct instances, each seeded from its own initial pose."""
    cube_zs = [0.05, 0.06, 0.07]
    cube, robot, jaw_idx = _mock_scene(cube_zs)

    trackers = init_trackers(
        len(cube_zs), cube, robot, jaw_idx, _HOME_POSE_RAD
    )

    assert len(trackers) == 3
    assert all(isinstance(t, LiftCubeEvalTracker) for t in trackers)
    # Distinct instances (no accidental sharing).
    assert len({id(t) for t in trackers}) == 3
    # Each tracker knows the initial_cube_z of ITS env (round(.,4)).
    for t, z in zip(trackers, cube_zs):
        assert t.result["initial_cube_z"] == round(z, 4)


def test_init_trackers_forwards_kwargs() -> None:
    """home_pose_tol_rad / plateau_steps are forwarded (no crash)."""
    cube, robot, jaw_idx = _mock_scene([0.05])
    trackers = init_trackers(
        1, cube, robot, jaw_idx, _HOME_POSE_RAD,
        home_pose_tol_rad=0.1,
        plateau_steps={"1_reach": 500, "2_lift": 500},
    )
    assert len(trackers) == 1
    assert trackers[0].result["success"] is False  # freshly initialized


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 — smoke runner should catch everything
            failed += 1
            print(f"  ERROR {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
