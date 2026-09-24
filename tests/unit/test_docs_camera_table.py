# SPDX-License-Identifier: MIT
"""The documented camera geometry must be the shipped camera geometry.

The viewpoint set is the experimental variable of this benchmark, so a table
that drifts from the JSON is not a cosmetic problem: it misstates the setup a
reader would reproduce. A hand-written table plus this test is smaller than a
generator and a build hook, and catches the same failure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "src" / "so101_mvbench" / "tasks" / "scene_configs"
PAGE = REPO / "docs" / "concepts" / "camera_configurations.md"


def _shipped() -> list[tuple[str, str, dict]]:
    """(config stem, camera name, spec) for every camera of every config."""
    rows = []
    for path in sorted(CONFIG_DIR.glob("*.json")):
        cameras = json.loads(path.read_text())["cameras"]
        for name, spec in cameras.items():
            rows.append((path.stem, name, spec))
    return rows


def _fmt(value: float) -> str:
    """Render a JSON number the way the table writes it (60.0 -> 60)."""
    return str(int(value)) if float(value) == int(value) else str(value)


@pytest.mark.parametrize("stem,name,spec", _shipped(),
                         ids=lambda v: v if isinstance(v, str) else "")
def test_every_shipped_camera_appears_in_the_table(stem, name, spec) -> None:
    page = PAGE.read_text()
    assert stem in page, f"configuration {stem} is not documented at all"
    assert f"`{name}`" in page, f"camera {name} of {stem} is not in the table"

    if spec.get("type") == "ego":
        return  # no hemisphere coordinates to state

    for field, unit in (("azimuth_deg", "°"), ("elevation_deg", "°"),
                        ("radius_m", " m")):
        value = spec.get(field)
        if value is None:
            continue
        cell = f"{_fmt(value)}{unit}"
        assert cell in page, (
            f"{stem}/{name}: {field} is {value} in the JSON, but {cell!r} does "
            f"not appear in {PAGE.name}"
        )


def test_a_non_default_focal_length_is_stated() -> None:
    """A lens that differs from the default changes the image and must be named."""
    page = PAGE.read_text()
    for stem, name, spec in _shipped():
        focal = spec.get("focal_length")
        if focal is None:
            continue
        assert _fmt(focal) in page, (
            f"{stem}/{name} sets focal_length {focal}, which the table omits"
        )


def test_the_table_covers_every_shipped_config() -> None:
    """A new scene config must not silently stay undocumented."""
    page = PAGE.read_text()
    missing = [p.stem for p in CONFIG_DIR.glob("*.json") if p.stem not in page]
    assert not missing, f"scene configs absent from the table: {sorted(missing)}"
