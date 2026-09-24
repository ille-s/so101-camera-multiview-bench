# SPDX-License-Identifier: MIT
"""Characterisation tests for the workspace frame {W} and its placement in world {0}.

These pin the behaviour that the frame refactor must NOT change. The pose-graph
property the thesis relies on is not any absolute coordinate but this invariant:

    a scene change moves ⁰ξ_W and nothing else,
    so every object keeps its position *relative to* {W}.

Reference values are taken from a real recorded dataset
(``datasets/01_raw_gamepad/bin_c0_r0/binmap.json``, episodes 0-2). The geometry
constants are imported from ``so101_transforms`` -- their single source of truth
since the dedup -- so these tests exercise the real values, not restated copies.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from so101_mvbench.utils.bin_spawner import BinSpawner, WorkspaceBounds
from so101_mvbench.utils.scene_config import SceneConfig
from so101_mvbench.utils.so101_transforms import (
    ARM_BASE_OFFSET,
    CUBE_SPAWN_Z_IN_W,
    FLOOR_CUBE_SPAWN_Z,
    HEMISPHERE_ORIGIN_W,
    TABLE_HEMISPHERE_ORIGIN,
    rigid_shift_from_recording,
)

# ⁰ξ_W for the two scenes, expressed in world frame {0}.
TABLE_ORIGIN_W = HEMISPHERE_ORIGIN_W
CYLROOM_ORIGIN_W = TABLE_HEMISPHERE_ORIGIN

# Recording parameters of the released dataset (teleop_recorder.py:48-50 defaults).
GRID = dict(center_x=0.0, center_y=0.0, n_cols=5, n_rows=5, bin_size=0.05)


def _spawner(origin_x: float, origin_y: float) -> BinSpawner:
    bounds = WorkspaceBounds.from_center(origin_x=origin_x, origin_y=origin_y, **GRID)
    return BinSpawner(
        bounds,
        n_cols=GRID["n_cols"],
        n_rows=GRID["n_rows"],
        half_extent_m=0.02,
        robot_yaw_rad=math.pi / 2,
    )


def test_binspawner_reproduces_the_recorded_binmap() -> None:
    """The released dataset is reproducible from the spawner, so the frame math is pinned.

    binmap.json of bin_c0_r0 records x_m=0.32, y_m=-0.12 for bin (0,0) with the
    three fixed yaw rotations. Anything that changes these values invalidates the
    recorded episodes.
    """
    schedule = _spawner(*TABLE_ORIGIN_W[:2])._schedule

    assert len(schedule) == 675
    for cfg, expected_yaw in zip(schedule[:3], [0.0, math.pi / 8, math.pi / 4]):
        assert (cfg.bin_col, cfg.bin_row) == (0, 0)
        assert cfg.x_m == pytest.approx(0.32, abs=1e-9)
        assert cfg.y_m == pytest.approx(-0.12, abs=1e-9)
        assert cfg.yaw_rad == pytest.approx(expected_yaw, abs=1e-9)


def test_spawn_position_relative_to_the_workspace_origin_is_scene_independent() -> None:
    """THE invariant. Moving ⁰ξ_W must move the spawn with it, unchanged in {W}.

    This is what makes a recorded joint trajectory replayable in a different room:
    the arm-to-cube geometry is identical, only the whole rig sits elsewhere.
    """
    table = _spawner(*TABLE_ORIGIN_W[:2])._schedule[0]
    cylroom = _spawner(*CYLROOM_ORIGIN_W[:2])._schedule[0]

    in_table_frame = (table.x_m - TABLE_ORIGIN_W[0], table.y_m - TABLE_ORIGIN_W[1])
    in_cylroom_frame = (cylroom.x_m - CYLROOM_ORIGIN_W[0], cylroom.y_m - CYLROOM_ORIGIN_W[1])

    assert in_table_frame == pytest.approx(in_cylroom_frame, abs=1e-9)
    assert in_table_frame == pytest.approx((0.12, -0.12), abs=1e-9)
    assert table.yaw_rad == cylroom.yaw_rad


def test_todays_cylroom_placement_equals_the_workspace_origin_form() -> None:
    """The hand-applied ARM_BASE_OFFSET produces exactly the ⁰ξ_W form it will be replaced by.

    Today (multicam_replay.py:386-389) the cube world pose is built as
    ``floor spawn + ARM_BASE_OFFSET``. After the refactor it is built as
    ``⁰ξ_W(cylroom) + spawn in {W}``. Both must agree, otherwise the refactor moves
    the scene.
    """
    table_spawn = _spawner(*TABLE_ORIGIN_W[:2])._schedule[0]

    today = (
        table_spawn.x_m + ARM_BASE_OFFSET[0],
        table_spawn.y_m + ARM_BASE_OFFSET[1],
        FLOOR_CUBE_SPAWN_Z + ARM_BASE_OFFSET[2],
    )

    spawn_in_w = (
        table_spawn.x_m - TABLE_ORIGIN_W[0],
        table_spawn.y_m - TABLE_ORIGIN_W[1],
        FLOOR_CUBE_SPAWN_Z - TABLE_ORIGIN_W[2],
    )
    after = tuple(o + s for o, s in zip(CYLROOM_ORIGIN_W, spawn_in_w))

    assert today == pytest.approx(after, abs=1e-9)
    assert today == pytest.approx((0.12, -0.12, 0.7963), abs=1e-9)


def test_cylroom_origin_is_the_table_origin_plus_the_offset() -> None:
    """TABLE_HEMISPHERE_ORIGIN is the table origin shifted by the rigid delta.

    Both consumers (multicam_replay, async_eval) import it from so101_transforms.
    The literal pins the actual number: if someone edits any of the summands, this
    is the test that says the cylroom workspace origin moved.
    """
    computed = tuple(o + d for o, d in zip(TABLE_ORIGIN_W, ARM_BASE_OFFSET))
    assert computed == pytest.approx(CYLROOM_ORIGIN_W, abs=1e-9)
    assert CYLROOM_ORIGIN_W == pytest.approx((0.0, 0.0, 0.8663), abs=1e-9)


def test_scene_config_carries_the_workspace_origin() -> None:
    """A config with no origin field declares the base scene, not a retired one.

    The legacy fallback to the table scene applies to recorded *data* without a
    frame declaration, not to a scene config. Mixing the two once put the arm on
    the floor, so the two defaults are asserted separately.
    """
    from so101_mvbench.utils.so101_transforms import BASE_SCENE_ORIGIN_W

    assert SceneConfig().workspace_origin_m == BASE_SCENE_ORIGIN_W
    assert rigid_shift_from_recording(CYLROOM_ORIGIN_W) == pytest.approx(
        ARM_BASE_OFFSET, abs=1e-9
    ), "data without a declared frame must still be read as the table scene"


def test_shift_derived_from_shipped_config_equals_the_historical_offset() -> None:
    """The config-driven derivation reproduces ARM_BASE_OFFSET exactly.

    Stage 2 and stage 4 now compute their rigid shift from the scene config's
    workspace_origin_m instead of using the hardcoded constant. For the shipped
    cylroom configs both paths must agree to the last digit, otherwise the
    refactor moved the scene.
    """
    import so101_mvbench

    cfg_dir = Path(so101_mvbench.__file__).resolve().parent / "tasks" / "scene_configs"
    configs = sorted(cfg_dir.glob("*.json"))
    assert configs, f"no scene configs found in {cfg_dir}"
    for cfg_path in configs:
        cfg = SceneConfig.from_json(cfg_path)
        assert tuple(cfg.workspace_origin_m) == pytest.approx(CYLROOM_ORIGIN_W, abs=1e-9), cfg_path.name
        shift = rigid_shift_from_recording(tuple(cfg.workspace_origin_m))
        assert shift == pytest.approx(ARM_BASE_OFFSET, abs=1e-9), cfg_path.name


def test_shift_is_zero_when_source_and_target_are_the_same_scene() -> None:
    """Re-rendering a cylroom recording in the cylroom must not move anything."""
    assert rigid_shift_from_recording(CYLROOM_ORIGIN_W, CYLROOM_ORIGIN_W) == (0.0, 0.0, 0.0)


def test_source_spawn_height_is_workspace_fixed() -> None:
    """The cube spawn height is a {W} property; each scene's world value follows.

    source_value + shift must land at the same target-world height regardless of
    which scene the data was recorded in -- otherwise a cylroom recording would
    replay its cube at floor height.
    """
    for source in (TABLE_ORIGIN_W, CYLROOM_ORIGIN_W):
        spawn_z_source = source[2] + CUBE_SPAWN_Z_IN_W
        shift = rigid_shift_from_recording(CYLROOM_ORIGIN_W, source)
        assert spawn_z_source + shift[2] == pytest.approx(
            CYLROOM_ORIGIN_W[2] + CUBE_SPAWN_Z_IN_W, abs=1e-9
        )
    # And the table-scene value is the historical constant.
    assert TABLE_ORIGIN_W[2] + CUBE_SPAWN_Z_IN_W == pytest.approx(FLOOR_CUBE_SPAWN_Z)


def test_binmap_declares_its_recording_frame(tmp_path: Path) -> None:
    """BinMap round-trips the workspace origin; legacy files load as None."""
    from so101_mvbench.utils.bin_spawner import BinMap

    m = BinMap(n_cols=5, n_rows=5, object_key="cube",
               workspace_origin_m=CYLROOM_ORIGIN_W)
    p = tmp_path / "binmap.json"
    m.save(p)
    loaded = BinMap.load(p)
    assert loaded.workspace_origin_m == pytest.approx(CYLROOM_ORIGIN_W)

    legacy = BinMap(n_cols=5, n_rows=5, object_key="cube")
    p2 = tmp_path / "legacy.json"
    legacy.save(p2)
    import json
    assert "workspace_origin_m" not in json.loads(p2.read_text())
    assert BinMap.load(p2).workspace_origin_m is None


def test_rig_anchoring_is_target_only_data_shift_is_source_to_target() -> None:
    """The bug this pins: conflating the two put the arm on the floor.

    The scene furniture (arm base, pad) is re-anchored by target-vs-BASE-SCENE,
    no matter where the data came from; recorded coordinates move by
    source-vs-target. Since the base scene is authored natively in the cylroom,
    the rig shift for the shipped configs is zero -- while the data shift for a
    legacy table recording is the full table-to-cylroom delta.
    """
    from so101_mvbench.utils.so101_transforms import BASE_SCENE_ORIGIN_W

    rig = rigid_shift_from_recording(CYLROOM_ORIGIN_W, BASE_SCENE_ORIGIN_W)
    data_legacy = rigid_shift_from_recording(CYLROOM_ORIGIN_W, TABLE_ORIGIN_W)
    data_same = rigid_shift_from_recording(CYLROOM_ORIGIN_W, CYLROOM_ORIGIN_W)
    assert rig == (0.0, 0.0, 0.0)
    assert data_legacy == pytest.approx(ARM_BASE_OFFSET, abs=1e-9)
    assert data_same == (0.0, 0.0, 0.0)
