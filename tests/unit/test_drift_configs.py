# SPDX-License-Identifier: MIT
"""Unit tests for the drifted-scene-config generator.

The generator's whole job is to change exactly one number and leave the rest of
the configuration alone, because an evaluation against the output is only a
camera-drift measurement if nothing else moved. These tests pin that.
"""

from __future__ import annotations

import json

import pytest

from so101_mvbench.tools.drift_configs import (
    drift_camera,
    parse_offsets,
    parse_up_world,
    variant_name,
)
from so101_mvbench.utils.scene_config import SceneConfig

SHIPPED = "src/so101_mvbench/tasks/scene_configs/lift_cube_6cam.json"


@pytest.fixture
def base() -> SceneConfig:
    return SceneConfig.from_json(SHIPPED)


def test_offset_moves_only_the_named_camera(base: SceneConfig) -> None:
    drifted = drift_camera(base, "front", "azimuth", 15.0)

    assert drifted.cameras["front"].azimuth_deg == base.cameras["front"].azimuth_deg + 15.0
    # every other camera identical
    for name, spec in base.cameras.items():
        if name == "front":
            continue
        assert drifted.cameras[name] == spec, f"{name} changed"


def test_offset_moves_only_the_named_axis(base: SceneConfig) -> None:
    drifted = drift_camera(base, "front", "azimuth", 15.0)
    src, dst = base.cameras["front"], drifted.cameras["front"]

    assert dst.elevation_deg == src.elevation_deg
    assert dst.radius_m == src.radius_m
    assert dst.resolution_hw == src.resolution_hw
    assert dst.focal_length == src.focal_length


def test_the_source_config_is_not_mutated(base: SceneConfig) -> None:
    before = base.cameras["front"].azimuth_deg
    drift_camera(base, "front", "azimuth", 15.0)
    assert base.cameras["front"].azimuth_deg == before


def test_non_camera_sections_survive(base: SceneConfig) -> None:
    drifted = drift_camera(base, "front", "elevation", -10.0)
    assert drifted.objects == base.objects
    assert drifted.robot == base.robot
    assert drifted.visual == base.visual
    assert drifted.workspace_origin_m == base.workspace_origin_m


def test_up_world_is_written_when_given(base: SceneConfig) -> None:
    assert base.cameras["top"].up_world is None
    drifted = drift_camera(base, "top", "elevation", -20.0, up_world=(0.0, 0.0, 1.0))
    assert drifted.cameras["top"].up_world == (0.0, 0.0, 1.0)


def test_the_wrist_camera_cannot_drift(base: SceneConfig) -> None:
    """It has no hemisphere pose, so asking must fail loudly rather than silently."""
    with pytest.raises(ValueError, match="external"):
        drift_camera(base, "wrist", "azimuth", 10.0)


def test_unknown_camera_names_the_available_ones(base: SceneConfig) -> None:
    with pytest.raises(KeyError, match="front"):
        drift_camera(base, "nope", "azimuth", 10.0)


def test_unknown_axis_is_rejected(base: SceneConfig) -> None:
    with pytest.raises(ValueError, match="axis"):
        drift_camera(base, "front", "sideways", 10.0)


def test_roundtrip_through_json_keeps_the_offset(base: SceneConfig, tmp_path) -> None:
    drifted = drift_camera(base, "front", "azimuth", -5.0, up_world=(0.0, 0.0, 1.0))
    path = tmp_path / "front_az-05.json"
    drifted.to_json(path)

    reloaded = SceneConfig.from_json(path)
    assert reloaded.cameras["front"].azimuth_deg == base.cameras["front"].azimuth_deg - 5.0
    assert reloaded.cameras["front"].up_world == (0.0, 0.0, 1.0)
    # the written file is a valid scene config, not just valid JSON
    assert json.loads(path.read_text())["schema_version"] == base.schema_version


def test_variant_names_state_sign_axis_and_magnitude() -> None:
    names = [variant_name("front", "azimuth", o) for o in (-15, -5, 5, 15)]
    assert names == ["front_az-15", "front_az-05", "front_az+05", "front_az+15"]
    assert len(set(names)) == len(names), "two offsets must not share a filename"
    # elevation and radius are distinguishable in the same directory
    assert variant_name("top", "elevation", -20) == "top_el-20"
    assert variant_name("front", "radius", 0.05) == "front_r+0p050"


def test_offset_and_up_world_parsing() -> None:
    assert parse_offsets("-15, -5,5 ,15") == [-15.0, -5.0, 5.0, 15.0]
    assert parse_up_world("0,0,1") == (0.0, 0.0, 1.0)
    assert parse_up_world(None) is None
    with pytest.raises(ValueError):
        parse_offsets("  ")
    with pytest.raises(ValueError):
        parse_up_world("0,1")
