# SPDX-License-Identifier: MIT
"""Unit tests for ``utils.episodes_spec.parse_episodes_spec``.

Coverage:
- Single integer
- Comma list
- Inclusive range
- Mixed comma + range
- Deduplication
- ``'all'`` glob (positive + empty-dir error)
- Whitespace tolerance
- Tolerant comma syntax (leading/trailing/double commas)
- Error cases: empty, malformed, reversed range
"""

from __future__ import annotations

from pathlib import Path

import pytest

from so101_mvbench.utils.episodes_spec import parse_episodes_spec


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_single_int(tmp_path: Path) -> None:
    assert parse_episodes_spec("0", tmp_path) == [0]
    assert parse_episodes_spec("17", tmp_path) == [17]


def test_comma_list(tmp_path: Path) -> None:
    assert parse_episodes_spec("0,1,5", tmp_path) == [0, 1, 5]
    assert parse_episodes_spec("5,1,0", tmp_path) == [0, 1, 5]  # sorted


def test_inclusive_range(tmp_path: Path) -> None:
    assert parse_episodes_spec("0-3", tmp_path) == [0, 1, 2, 3]
    assert parse_episodes_spec("5-5", tmp_path) == [5]  # single-element range


def test_mixed_comma_and_range(tmp_path: Path) -> None:
    assert parse_episodes_spec("0,5-7,10", tmp_path) == [0, 5, 6, 7, 10]


def test_deduplication(tmp_path: Path) -> None:
    """Overlapping ranges + repeats yield unique sorted IDs."""
    assert parse_episodes_spec("0,1,2,2-3", tmp_path) == [0, 1, 2, 3]
    assert parse_episodes_spec("5-7,6,7", tmp_path) == [5, 6, 7]


def test_whitespace_tolerance(tmp_path: Path) -> None:
    assert parse_episodes_spec(" 0 , 5 - 7 , 10 ", tmp_path) == [0, 5, 6, 7, 10]
    assert parse_episodes_spec("\t0\n", tmp_path) == [0]


def test_tolerant_comma_syntax(tmp_path: Path) -> None:
    """Leading / trailing / doubled commas are silently ignored."""
    assert parse_episodes_spec(",0", tmp_path) == [0]
    assert parse_episodes_spec("0,", tmp_path) == [0]
    assert parse_episodes_spec("0,,5", tmp_path) == [0, 5]


# ---------------------------------------------------------------------------
# 'all' globs trajectories/
# ---------------------------------------------------------------------------

def test_all_discovers_episodes(tmp_path: Path) -> None:
    traj_dir = tmp_path / "trajectories"
    traj_dir.mkdir()
    for idx in (0, 2, 5):
        (traj_dir / f"episode_{idx:03d}.npy").write_bytes(b"")
    assert parse_episodes_spec("all", tmp_path) == [0, 2, 5]


def test_all_case_insensitive(tmp_path: Path) -> None:
    traj_dir = tmp_path / "trajectories"
    traj_dir.mkdir()
    (traj_dir / "episode_001.npy").write_bytes(b"")
    assert parse_episodes_spec("ALL", tmp_path) == [1]
    assert parse_episodes_spec("All", tmp_path) == [1]


def test_all_empty_dir_raises(tmp_path: Path) -> None:
    (tmp_path / "trajectories").mkdir()
    with pytest.raises(ValueError, match="no trajectories"):
        parse_episodes_spec("all", tmp_path)


def test_all_missing_dir_raises(tmp_path: Path) -> None:
    """Non-existent trajectories/ dir: glob yields nothing → raises."""
    with pytest.raises(ValueError, match="no trajectories"):
        parse_episodes_spec("all", tmp_path)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_empty_spec_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty episodes spec"):
        parse_episodes_spec("", tmp_path)
    with pytest.raises(ValueError, match="empty episodes spec"):
        parse_episodes_spec("   ", tmp_path)


def test_only_commas_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty episodes spec after parsing"):
        parse_episodes_spec(",,", tmp_path)


def test_reversed_range_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reversed range"):
        parse_episodes_spec("5-2", tmp_path)


def test_malformed_range_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="malformed range"):
        parse_episodes_spec("0-a", tmp_path)
    with pytest.raises(ValueError, match="malformed range"):
        parse_episodes_spec("a-5", tmp_path)


def test_malformed_integer_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="malformed integer"):
        parse_episodes_spec("abc", tmp_path)
    with pytest.raises(ValueError, match="malformed integer"):
        parse_episodes_spec("0,abc,5", tmp_path)


def test_none_spec_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="None"):
        parse_episodes_spec(None, tmp_path)  # type: ignore[arg-type]
