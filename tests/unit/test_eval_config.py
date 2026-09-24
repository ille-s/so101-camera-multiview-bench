# SPDX-License-Identifier: MIT
"""Tests for ``evaluation.eval_config`` -- the shipped eval-config loader.

Replaces the parked ``test_config.py``: the loader used to be a private function
inside ``async_eval``, which launches Isaac Sim at import time and therefore
could not be unit-tested. Now it is an importable, Isaac-free module.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from so101_mvbench.evaluation.eval_config import load_eval_config


def test_shipped_config_loads_and_converts_to_radians() -> None:
    cfg = load_eval_config()

    # The shipped home pose is [0, 0, 0, 90, -90, 0] degrees.
    assert cfg["target_end_position_rad"] == pytest.approx(
        [0.0, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0]
    )
    # 8.6 deg tolerance -- the same "close enough to home" threshold METRICS.md documents.
    assert cfg["target_end_position_tolerance_rad"] == pytest.approx(math.radians(8.6))
    assert cfg["plateau_steps"]["1_reach"] == 600
    assert cfg["plateau_steps"]["2_lift"] == 600


def test_missing_file_fails_loud(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="eval config missing"):
        load_eval_config(tmp_path / "does_not_exist.json")


def test_missing_key_fails_loud(tmp_path: Path) -> None:
    p = tmp_path / "broken.json"
    p.write_text(json.dumps({"plateau_steps": {}}))
    with pytest.raises(KeyError):
        load_eval_config(p)


def test_degrees_are_converted_not_passed_through(tmp_path: Path) -> None:
    """A config written in degrees must never reach a consumer unconverted."""
    p = tmp_path / "custom.json"
    p.write_text(json.dumps({
        "target_end_position_deg": [180, 0, 0, 0, 0, 0],
        "target_end_position_tolerance_deg": 90,
        "plateau_steps": {"1_reach": 1},
    }))
    cfg = load_eval_config(p)
    assert cfg["target_end_position_rad"][0] == pytest.approx(math.pi)
    assert cfg["target_end_position_tolerance_rad"] == pytest.approx(math.pi / 2)
