# SPDX-License-Identifier: MIT
"""Property tests for utils/camera_grid.py camera-pose helpers.

These tests are pure Python — no Isaac Sim required. They verify that the
new ``get_pose_world_position`` getter and the math used by the new
``set_external_camera_pose{,s}`` setters produce results bit-identical to
the pre-refactor inline code in:

    - evaluation/camera_override.py    (L73-82  pre-refactor)
    - recording/external_batch_recorder.py (L386-394 pre-refactor)
    - recording/trajectory_replayer.py  (L311+314 + L405 pre-refactor)
    - tools/hemisphere_screenshot.py    (L287-291 pre-refactor)

If the helper math drifts in the future, these tests fail at pytest time —
no Isaac Sim needed.

Run:
    PYTHONPATH=src/so101_mvbench/src:src/ws_utils/src \
        pytest -xvs src/so101_mvbench/.../tests/test_camera_pose_helpers.py
"""

from __future__ import annotations

import math
import pytest

from so101_mvbench.utils.camera_grid import (
    CameraPose,
    DEFAULT_RADIUS_M,
    HEMISPHERE_ORIGIN_W,
    COARSE_GRID,
    get_pose_by_id,
    get_pose_world_position,
)


# Canonical workspace origin — single source of truth in so101_transforms.
# Updated 2026-05-12: (0.15, 0.0, 0.12) → (0.20, 0.0, 0.12), aligning look-at
# with cube.init_state.pos.x = 0.20 (cube spawn x-center).
CANONICAL_HEMI_ORIGIN = (0.20, 0.0, 0.12)


# ---------------------------------------------------------------------------
# HEMISPHERE_ORIGIN_W matches the canonical value
# ---------------------------------------------------------------------------

def test_hemisphere_origin_w_is_canonical():
    """camera_grid.HEMISPHERE_ORIGIN_W must match CANONICAL_HEMI_ORIGIN (0.20, 0.0, 0.12)."""
    assert HEMISPHERE_ORIGIN_W == CANONICAL_HEMI_ORIGIN


def test_hemisphere_origin_w_matches_so101_transforms():
    """Both modules must declare the same canonical value (single source of truth)."""
    from so101_mvbench.utils.so101_transforms import (
        HEMISPHERE_ORIGIN_W as so101_const,
    )
    assert HEMISPHERE_ORIGIN_W == so101_const


# ---------------------------------------------------------------------------
# get_pose_world_position: math equivalence with pre-refactor inline code
# ---------------------------------------------------------------------------

def test_get_pose_world_position_matches_hemisphere_screenshot_inline():
    """Replica of hemisphere_screenshot.py:287 pre-refactor:

        eye = torch.tensor([[pose.x + wx, pose.y + wy, pose.z + wz]], ...)
    """
    pose = get_pose_by_id("az000_el45")
    wx, wy, wz = CANONICAL_HEMI_ORIGIN
    old_eye_tuple = (pose.x + wx, pose.y + wy, pose.z + wz)
    new_eye_tuple = get_pose_world_position(pose, CANONICAL_HEMI_ORIGIN)
    assert new_eye_tuple == old_eye_tuple


def test_get_pose_world_position_matches_trajectory_replayer_inline():
    """Replica of trajectory_replayer.py:311 + L405 pre-refactor."""
    pose = get_pose_by_id("az090_el45")
    wx, wy, wz = CANONICAL_HEMI_ORIGIN
    old = (pose.x + wx, pose.y + wy, pose.z + wz)
    new = get_pose_world_position(pose, CANONICAL_HEMI_ORIGIN)
    assert new == old


def test_get_pose_world_position_matches_external_batch_recorder_list_inline():
    """Replica of external_batch_recorder.py:386 pre-refactor (list version):

        eyes_local = [[p.x + wx, p.y + wy, p.z + wz] for p in poses]
    """
    pose_ids = ["az000_el45", "az090_el45", "az180_el45", "az270_el45", "az000_el90"]
    poses = [get_pose_by_id(pid) for pid in pose_ids]
    wx, wy, wz = CANONICAL_HEMI_ORIGIN

    old_eyes = [[p.x + wx, p.y + wy, p.z + wz] for p in poses]
    new_eyes = [list(get_pose_world_position(p, CANONICAL_HEMI_ORIGIN)) for p in poses]
    assert new_eyes == old_eyes


def test_get_pose_world_position_matches_camera_override_inline():
    """Replica of evaluation/camera_override.py:73-78 pre-refactor."""
    pose = get_pose_by_id("az045_el45")
    wx, wy, wz = CANONICAL_HEMI_ORIGIN

    # Old (from camera_override.py:73-75):
    # eye_local = torch.tensor([[pose.x + wx, pose.y + wy, pose.z + wz]], ...)
    old_eye_singleton = [pose.x + wx, pose.y + wy, pose.z + wz]
    new_eye_singleton = list(get_pose_world_position(pose, CANONICAL_HEMI_ORIGIN))
    assert new_eye_singleton == old_eye_singleton


# ---------------------------------------------------------------------------
# get_pose_world_position: edge cases
# ---------------------------------------------------------------------------

def test_get_pose_world_position_zero_radius_corner():
    """Pose at look-at target (radius=0) → eye == hemisphere_origin."""
    pose = CameraPose(
        id="origin", azimuth_deg=0.0, elevation_deg=0.0, radius_m=0.0,
        x=0.0, y=0.0, z=0.0,
    )
    assert get_pose_world_position(pose, CANONICAL_HEMI_ORIGIN) == CANONICAL_HEMI_ORIGIN


def test_get_pose_world_position_zero_hemisphere_origin():
    """hemisphere_origin=(0,0,0) → eye == pose.{x,y,z}."""
    pose = get_pose_by_id("az000_el45")
    result = get_pose_world_position(pose, (0.0, 0.0, 0.0))
    assert result == (pose.x, pose.y, pose.z)


def test_get_pose_world_position_default_argument():
    """Without explicit hemisphere_origin → uses HEMISPHERE_ORIGIN_W."""
    pose = get_pose_by_id("az000_el45")
    explicit = get_pose_world_position(pose, HEMISPHERE_ORIGIN_W)
    default = get_pose_world_position(pose)
    assert explicit == default


def test_get_pose_world_position_returns_tuple_of_three_floats():
    """Type contract: tuple[float, float, float]."""
    pose = get_pose_by_id("az000_el45")
    result = get_pose_world_position(pose)
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# Hemisphere geometry sanity: all poses on radius DEFAULT_RADIUS_M from origin {W}
# ---------------------------------------------------------------------------

def test_all_coarse_grid_poses_on_default_radius():
    """Geometric invariant: every CameraPose in COARSE_GRID has |pose.xyz| == DEFAULT_RADIUS_M."""
    for pose in COARSE_GRID:
        dist = math.sqrt(pose.x**2 + pose.y**2 + pose.z**2)
        assert math.isclose(dist, DEFAULT_RADIUS_M, abs_tol=1e-6), (
            f"Pose {pose.id} at distance {dist} != DEFAULT_RADIUS_M ({DEFAULT_RADIUS_M})"
        )


def test_get_pose_world_position_preserves_radius_in_workspace_frame():
    """If we subtract hemisphere_origin from the result, we get pose.xyz back."""
    pose = get_pose_by_id("az045_el45")
    wx, wy, wz = CANONICAL_HEMI_ORIGIN
    eye = get_pose_world_position(pose, CANONICAL_HEMI_ORIGIN)
    workspace_offset = (eye[0] - wx, eye[1] - wy, eye[2] - wz)
    assert math.isclose(workspace_offset[0], pose.x, abs_tol=1e-6)
    assert math.isclose(workspace_offset[1], pose.y, abs_tol=1e-6)
    assert math.isclose(workspace_offset[2], pose.z, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Setter signature smoke (no env — just verify callables + signature)
# ---------------------------------------------------------------------------

def test_setter_functions_are_importable():
    """set_external_camera_pose + set_external_camera_poses must be callable symbols."""
    from so101_mvbench.utils.camera_grid import (
        set_external_camera_pose,
        set_external_camera_poses,
    )
    assert callable(set_external_camera_pose)
    assert callable(set_external_camera_poses)


def test_setter_pose_param_accepts_str_or_camerapose():
    """set_external_camera_pose signature accepts both str and CameraPose."""
    import inspect
    from so101_mvbench.utils.camera_grid import set_external_camera_pose
    sig = inspect.signature(set_external_camera_pose)
    assert "pose" in sig.parameters
    assert "hemisphere_origin" in sig.parameters
    assert "camera_name" in sig.parameters
    # Default for camera_name should be "camera_external"
    assert sig.parameters["camera_name"].default == "camera_external"


def test_setter_poses_plural_signature():
    """set_external_camera_poses (plural) takes list parameter."""
    import inspect
    from so101_mvbench.utils.camera_grid import set_external_camera_poses
    sig = inspect.signature(set_external_camera_poses)
    assert "poses" in sig.parameters
    assert sig.parameters["camera_name"].default == "camera_external"
