# SPDX-License-Identifier: MIT
"""Unit tests for ``utils.scene_config`` (M1 acceptance criteria).

Coverage:
- ``test_default_values``: SceneConfig() with no args yields documented defaults
- ``test_roundtrip``: load → save → load is bit-identical for content
- ``test_schema_version_mismatch``: incompatible major version raises ValueError
- ``test_empty_dict_load``: minimal valid JSON loads to default config
- ``test_cameras_dict_access``: cfg.cameras["back"] resolves to the right (az,el,r)
- ``test_lift_cube_6cam_json``: the shipped example loads and contains all
  6 cameras (wrist + 5 externals) with correct (az, el, r) per heute's
  ``assemble_all_bins.sh:34`` CAM_SPEC.
- ``test_external_missing_field_raises``: external cams need az+el+r (all three).
- ``test_ego_with_hemisphere_field_raises``: ego cams reject az/el/r.
- ``test_ego_without_offset_raises``: ego needs offset_pos_m + offset_rot_deg.
- ``test_radius_m_roundtrip``: per-camera (az, el, r) survives JSON load → save.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import so101_mvbench
from so101_mvbench.utils.so101_transforms import BASE_SCENE_ORIGIN_W
from so101_mvbench.utils.scene_config import (
    SCHEMA_VERSION,
    CameraSpec,
    ObjectSpec,
    SceneConfig,
)


def test_default_values() -> None:
    cfg = SceneConfig()
    assert cfg.schema_version == SCHEMA_VERSION
    assert cfg.workspace_origin_m == BASE_SCENE_ORIGIN_W
    assert cfg.cameras == {}
    assert cfg.objects == {}
    assert cfg.robot.type == "so101_white"
    assert cfg.robot.init_rot_xyzw == (0.0, 0.0, 0.7071, 0.7071)
    assert cfg.visual.dome_light_y_mirror is True


def test_roundtrip(tmp_path: Path) -> None:
    cfg = SceneConfig(
        cameras={
            "wrist": CameraSpec(
                type="ego",
                offset_pos_m=(-0.005, 0.06, -0.062),
                offset_rot_deg=(-45, 0, 0),
            ),
            "back": CameraSpec(
                type="external", azimuth_deg=180, elevation_deg=45, radius_m=0.4,
            ),
        },
        objects={
            "cube": ObjectSpec(
                type="cube",
                size_m=(0.04, 0.04, 0.04),
                color_rgb=(0.8, 0.2, 0.2),
            ),
        },
    )
    p = tmp_path / "test.json"
    cfg.to_json(p)
    cfg2 = SceneConfig.from_json(p)

    assert cfg2.cameras["wrist"].type == "ego"
    assert cfg2.cameras["wrist"].offset_pos_m == (-0.005, 0.06, -0.062)
    assert cfg2.cameras["back"].azimuth_deg == 180
    assert cfg2.cameras["back"].elevation_deg == 45
    assert cfg2.cameras["back"].radius_m == 0.4
    assert cfg2.objects["cube"].color_rgb == (0.8, 0.2, 0.2)
    assert cfg2.objects["cube"].size_m == (0.04, 0.04, 0.04)


def test_schema_version_mismatch(tmp_path: Path) -> None:
    p = tmp_path / "old.json"
    p.write_text(json.dumps({"schema_version": "0.9"}))
    with pytest.raises(ValueError, match="schema_version mismatch"):
        SceneConfig.from_json(p)


def test_minor_version_compatible(tmp_path: Path) -> None:
    """Future minor bumps within the same major (1.99 read by 1.2) load OK
    as long as no field-level validation is violated."""
    p = tmp_path / "future_minor.json"
    p.write_text(json.dumps({"schema_version": "1.99"}))
    cfg = SceneConfig.from_json(p)
    assert cfg.schema_version == "1.99"
    assert cfg.cameras == {}


def test_empty_dict_load(tmp_path: Path) -> None:
    """Minimal JSON loads to default SceneConfig."""
    p = tmp_path / "empty.json"
    p.write_text(json.dumps({"schema_version": SCHEMA_VERSION}))
    cfg = SceneConfig.from_json(p)
    assert cfg.workspace_origin_m == BASE_SCENE_ORIGIN_W
    assert cfg.cameras == {}
    assert cfg.robot.type == "so101_white"


def test_cameras_dict_access() -> None:
    cfg = SceneConfig(
        cameras={
            "back":  CameraSpec(type="external", azimuth_deg=180, elevation_deg=45, radius_m=0.4),
            "front": CameraSpec(type="external", azimuth_deg=  0, elevation_deg=45, radius_m=0.4),
            "top":   CameraSpec(type="external", azimuth_deg=  0, elevation_deg=90, radius_m=0.4),
        },
    )
    assert cfg.cameras["back"].azimuth_deg == 180
    assert cfg.cameras["back"].elevation_deg == 45
    assert cfg.cameras["front"].azimuth_deg == 0
    assert sorted(cfg.cameras.keys()) == ["back", "front", "top"]


def test_lift_cube_6cam_json() -> None:
    """The shipped example config loads and matches heute's CAM_SPEC mapping."""
    package_root = Path(so101_mvbench.__file__).resolve().parent
    p = package_root / "tasks" / "scene_configs" / "lift_cube_6cam.json"
    cfg = SceneConfig.from_json(p)

    expected_extern = {
        "front": (0,   45),
        "left":  (90,  45),
        "back":  (180, 45),
        "right": (270, 45),
        "top":   (0,   90),
    }
    assert set(cfg.cameras.keys()) == {"wrist", *expected_extern.keys()}
    for cam_name, (az, el) in expected_extern.items():
        c = cfg.cameras[cam_name]
        assert c.type == "external"
        assert c.azimuth_deg == az, (
            f"camera {cam_name!r}: expected azimuth_deg={az}, got {c.azimuth_deg!r}"
        )
        assert c.elevation_deg == el, (
            f"camera {cam_name!r}: expected elevation_deg={el}, got {c.elevation_deg!r}"
        )
        assert c.radius_m == pytest.approx(0.4), (
            f"camera {cam_name!r}: expected radius_m=0.4, got {c.radius_m!r}"
        )

    # ego defaults — none of the hemisphere fields, both offsets present
    assert cfg.cameras["wrist"].type == "ego"
    assert cfg.cameras["wrist"].offset_pos_m == (-0.005, 0.06, -0.062)
    assert cfg.cameras["wrist"].azimuth_deg is None
    assert cfg.cameras["wrist"].elevation_deg is None
    assert cfg.cameras["wrist"].radius_m is None

    # Cube + action_pad
    assert "cube" in cfg.objects
    assert cfg.objects["cube"].color_rgb == (0.8, 0.2, 0.2)
    assert "action_pad" in cfg.objects

    # Robot
    assert cfg.robot.init_rot_xyzw == (0.0, 0.0, 0.7071, 0.7071)
    assert "Wrist_Pitch" in cfg.robot.home_joint_pos_rad
    assert cfg.robot.home_joint_pos_rad["Wrist_Pitch"] == pytest.approx(1.5707963267948966)


def test_roundtrip_idempotent(tmp_path: Path) -> None:
    """Load → save → load → save should produce identical bytes."""
    package_root = Path(so101_mvbench.__file__).resolve().parent
    src = package_root / "tasks" / "scene_configs" / "lift_cube_6cam.json"

    cfg = SceneConfig.from_json(src)
    p1 = tmp_path / "first.json"
    p2 = tmp_path / "second.json"
    cfg.to_json(p1)
    SceneConfig.from_json(p1).to_json(p2)

    assert p1.read_bytes() == p2.read_bytes()


def test_external_missing_field_raises() -> None:
    """external CameraSpec needs all three of azimuth_deg, elevation_deg, radius_m."""
    # All three missing
    with pytest.raises(ValueError, match="azimuth_deg.*elevation_deg.*radius_m"):
        CameraSpec(type="external")
    # Only az missing
    with pytest.raises(ValueError, match="azimuth_deg"):
        CameraSpec(type="external", elevation_deg=45, radius_m=0.4)
    # Only el missing
    with pytest.raises(ValueError, match="elevation_deg"):
        CameraSpec(type="external", azimuth_deg=0, radius_m=0.4)
    # Only radius missing
    with pytest.raises(ValueError, match="radius_m"):
        CameraSpec(type="external", azimuth_deg=0, elevation_deg=45)


def test_ego_with_hemisphere_field_raises() -> None:
    """ego CameraSpec must not carry azimuth_deg / elevation_deg / radius_m."""
    base = dict(
        type="ego",
        offset_pos_m=(-0.005, 0.06, -0.062),
        offset_rot_deg=(-45, 0, 0),
    )
    with pytest.raises(ValueError, match="should not have azimuth_deg"):
        CameraSpec(**base, azimuth_deg=0)
    with pytest.raises(ValueError, match="should not have elevation_deg"):
        CameraSpec(**base, elevation_deg=45)
    with pytest.raises(ValueError, match="should not have radius_m"):
        CameraSpec(**base, radius_m=0.4)


def test_ego_without_offset_raises() -> None:
    """ego CameraSpec needs both offset_pos_m and offset_rot_deg."""
    with pytest.raises(ValueError, match="offset_pos_m \\+ offset_rot_deg"):
        CameraSpec(type="ego")
    with pytest.raises(ValueError, match="offset_pos_m \\+ offset_rot_deg"):
        CameraSpec(type="ego", offset_pos_m=(-0.005, 0.06, -0.062))
    with pytest.raises(ValueError, match="offset_pos_m \\+ offset_rot_deg"):
        CameraSpec(type="ego", offset_rot_deg=(-45, 0, 0))


def test_radius_m_roundtrip(tmp_path: Path) -> None:
    """Per-camera (az, el, r) survives JSON load → save → load. Distinct radii
    at the same (az, el) round-trip independently."""
    cfg = SceneConfig(
        cameras={
            "wrist": CameraSpec(
                type="ego",
                offset_pos_m=(-0.005, 0.06, -0.062),
                offset_rot_deg=(-45, 0, 0),
            ),
            "front_close": CameraSpec(
                type="external", azimuth_deg=0, elevation_deg=45, radius_m=0.30,
            ),
            "front_def":   CameraSpec(
                type="external", azimuth_deg=0, elevation_deg=45, radius_m=0.40,
            ),
            "front_far":   CameraSpec(
                type="external", azimuth_deg=0, elevation_deg=45, radius_m=0.80,
            ),
        },
    )
    p = tmp_path / "with_radius.json"
    cfg.to_json(p)
    cfg2 = SceneConfig.from_json(p)

    assert cfg2.cameras["wrist"].radius_m is None
    for name, expected_r in (
        ("front_close", 0.30), ("front_def", 0.40), ("front_far", 0.80),
    ):
        c = cfg2.cameras[name]
        assert c.azimuth_deg == 0
        assert c.elevation_deg == 45
        assert c.radius_m == pytest.approx(expected_r)
