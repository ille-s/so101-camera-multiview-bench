"""Tests for BinSpawner, BinMap, and deterministic schedule generation.

Simulates the policy_recorder.py recording loop:
  next_config() → recording → mark_success() (N) or mark_retry() (R)
  → BinMap.log_episode()

Run with:
    cd src/so101_mvbench
    python3 -m pytest src/so101_mvbench/tests/test_sampling.py -v
"""

import math
import tempfile
from pathlib import Path

import pytest

# Adjust import path — recording/ module lives next to the tests/ directory
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from so101_mvbench.utils.bin_spawner import (
    BinMap,
    BinEpisodeRecord,
    BinSpawner,
    SpawnConfig,
    WorkspaceBounds,
    parse_ood_bins,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ws() -> WorkspaceBounds:
    """10×10 cm workspace, no origin offset, no robot yaw."""
    return WorkspaceBounds(x_min_m=0.0, x_max_m=0.10, y_min_m=0.0, y_max_m=0.10)


@pytest.fixture
def ws_with_origin() -> WorkspaceBounds:
    """10×10 cm workspace at world (0.20, 0.0)."""
    return WorkspaceBounds(
        x_min_m=-0.05, x_max_m=0.05,
        y_min_m=-0.05, y_max_m=0.05,
        origin_x=0.20, origin_y=0.0,
    )


@pytest.fixture
def spawner_2x2(ws) -> BinSpawner:
    """2×2 grid, no OOD, no robot yaw."""
    return BinSpawner(ws, n_cols=2, n_rows=2, half_extent_m=0.02)


@pytest.fixture
def spawner_1x1(ws) -> BinSpawner:
    """1×1 grid (single bin)."""
    return BinSpawner(ws, n_cols=1, n_rows=1, half_extent_m=0.02)


# ---------------------------------------------------------------------------
# Schedule generation
# ---------------------------------------------------------------------------

class TestScheduleGeneration:
    """Verify the deterministic schedule is built correctly."""

    def test_schedule_length_2x2(self, spawner_2x2):
        """2×2 grid: 4 bins × 9 positions × 3 rotations = 108."""
        assert spawner_2x2.total_episodes == 4 * 9 * 3
        assert spawner_2x2.total_episodes == 108

    def test_schedule_length_1x1(self, spawner_1x1):
        """1×1 grid: 1 bin × 9 positions × 3 rotations = 27."""
        assert spawner_1x1.total_episodes == 27

    def test_schedule_length_with_ood(self, ws):
        """OOD bins reduce total episodes."""
        spawner = BinSpawner(
            ws, n_cols=3, n_rows=3, half_extent_m=0.02,
            ood_bins=frozenset({(1, 1), (2, 2)}),
        )
        n_trainable = 9 - 2
        assert spawner.total_episodes == n_trainable * 9 * 3
        assert spawner.n_trainable_bins == 7

    def test_block_wise_order(self, spawner_2x2):
        """Episodes are grouped by bin (block-wise)."""
        schedule = spawner_2x2._schedule
        eps_per_bin = 9 * 3  # 27

        # First 27 episodes should all be bin (0,0)
        for i in range(eps_per_bin):
            assert schedule[i].bin_col == 0
            assert schedule[i].bin_row == 0

        # Next 27 should be bin (1,0) — row-major order
        for i in range(eps_per_bin, 2 * eps_per_bin):
            assert schedule[i].bin_col == 1
            assert schedule[i].bin_row == 0

    def test_rotations_per_position(self, spawner_1x1):
        """Each position has exactly 3 rotations (0°, 22.5°, 45°) in order."""
        schedule = spawner_1x1._schedule
        expected_yaws = BinSpawner.ROTATIONS_RAD
        assert len(expected_yaws) == 3
        assert expected_yaws[0] == pytest.approx(0.0)
        assert expected_yaws[1] == pytest.approx(math.pi / 8)   # 22.5°
        assert expected_yaws[2] == pytest.approx(math.pi / 4)   # 45°
        for pos_idx in range(9):
            base = pos_idx * 3
            for rot_idx, expected_yaw in enumerate(expected_yaws):
                assert schedule[base + rot_idx].yaw_rad == pytest.approx(expected_yaw)

    def test_subgrid_9_positions(self, spawner_1x1):
        """Each bin produces exactly 9 unique positions."""
        schedule = spawner_1x1._schedule
        positions = set()
        for cfg in schedule:
            positions.add((round(cfg.x_m, 6), round(cfg.y_m, 6)))
        assert len(positions) == 9

    def test_subgrid_inset_from_boundaries(self, ws):
        """Sub-grid positions are inset from bin edges (no overlap with neighbors)."""
        spawner = BinSpawner(ws, n_cols=1, n_rows=1, half_extent_m=0.02)
        schedule = spawner._schedule
        positions = [(round(c.x_m, 6), round(c.y_m, 6)) for c in schedule]
        # ws [0, 0.1] × [0, 0.1], 10% inset → corners at (0.01, 0.01) etc.
        inset = 0.1 * 0.1  # 10% of bin size (0.1m for 1x1 grid)
        assert (round(inset, 6), round(inset, 6)) in positions          # bottom-left inset
        assert (round(0.1 - inset, 6), round(0.1 - inset, 6)) in positions  # top-right inset
        assert (0.05, 0.05) in positions   # center unchanged
        # Exact edges should NOT be present
        assert (0.0, 0.0) not in positions
        assert (0.1, 0.1) not in positions

    def test_ood_bins_excluded(self, ws):
        """OOD bins produce no episodes."""
        ood = frozenset({(0, 0)})
        spawner = BinSpawner(ws, n_cols=2, n_rows=2, half_extent_m=0.02, ood_bins=ood)
        for cfg in spawner._schedule:
            assert (cfg.bin_col, cfg.bin_row) != (0, 0)

    def test_ood_bin_out_of_range_raises(self, ws):
        """OOD bin outside grid dimensions raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            BinSpawner(ws, n_cols=2, n_rows=2, half_extent_m=0.02,
                       ood_bins=frozenset({(5, 5)}))

    def test_selected_bin_filters_schedule(self, ws):
        """selected_bin limits schedule to one bin (27 episodes)."""
        spawner = BinSpawner(ws, n_cols=3, n_rows=3, half_extent_m=0.02,
                             selected_bin=(1, 2))
        assert spawner.total_episodes == 27
        for cfg in spawner._schedule:
            assert cfg.bin_col == 1
            assert cfg.bin_row == 2

    def test_selected_bin_out_of_range_raises(self, ws):
        """selected_bin outside grid raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            BinSpawner(ws, n_cols=2, n_rows=2, half_extent_m=0.02,
                       selected_bin=(5, 5))

    def test_selected_bin_ood_raises(self, ws):
        """selected_bin that is OOD raises ValueError."""
        with pytest.raises(ValueError, match="OOD"):
            BinSpawner(ws, n_cols=2, n_rows=2, half_extent_m=0.02,
                       ood_bins=frozenset({(1, 1)}), selected_bin=(1, 1))

    def test_origin_offset_applied(self, ws_with_origin):
        """World coordinates include origin offset."""
        spawner = BinSpawner(
            ws_with_origin, n_cols=1, n_rows=1, half_extent_m=0.02
        )
        for cfg in spawner._schedule:
            # Origin is (0.20, 0.0), local range [-0.05, 0.05]
            # World x should be in [0.15, 0.25]
            assert 0.15 - 1e-6 <= cfg.x_m <= 0.25 + 1e-6
            assert -0.05 - 1e-6 <= cfg.y_m <= 0.05 + 1e-6

    def test_robot_yaw_rotates_positions(self, ws):
        """Non-zero robot yaw rotates local positions into world frame."""
        spawner_no_yaw = BinSpawner(
            ws, n_cols=1, n_rows=1, half_extent_m=0.02, robot_yaw_rad=0.0
        )
        spawner_yaw = BinSpawner(
            ws, n_cols=1, n_rows=1, half_extent_m=0.02,
            robot_yaw_rad=math.pi / 2,
        )
        # Same position index, different world coords due to rotation
        cfg_no = spawner_no_yaw._schedule[0]
        cfg_yaw = spawner_yaw._schedule[0]
        # At yaw=π/2: (x, y) → (-y, x)
        assert cfg_yaw.x_m == pytest.approx(-cfg_no.y_m, abs=1e-6)
        assert cfg_yaw.y_m == pytest.approx(cfg_no.x_m, abs=1e-6)


# ---------------------------------------------------------------------------
# Cursor state machine
# ---------------------------------------------------------------------------

class TestCursorStateMachine:
    """Verify next_config / mark_success / mark_retry behavior."""

    def test_first_call_returns_first_episode(self, spawner_1x1):
        cfg = spawner_1x1.next_config()
        assert cfg == spawner_1x1._schedule[0]

    def test_repeated_calls_without_advance_return_same(self, spawner_1x1):
        cfg1 = spawner_1x1.next_config()
        cfg2 = spawner_1x1.next_config()
        assert cfg1 is cfg2

    def test_mark_success_advances(self, spawner_1x1):
        cfg0 = spawner_1x1.next_config()
        spawner_1x1.mark_success()
        cfg1 = spawner_1x1.next_config()
        assert cfg1 == spawner_1x1._schedule[1]
        assert cfg1 is not cfg0

    def test_mark_retry_stays(self, spawner_1x1):
        cfg0 = spawner_1x1.next_config()
        spawner_1x1.mark_retry()
        cfg1 = spawner_1x1.next_config()
        assert cfg1 is cfg0

    def test_multiple_retries_then_success(self, spawner_1x1):
        cfg0 = spawner_1x1.next_config()
        for _ in range(5):
            spawner_1x1.mark_retry()
            assert spawner_1x1.next_config() is cfg0
        spawner_1x1.mark_success()
        cfg1 = spawner_1x1.next_config()
        assert cfg1 == spawner_1x1._schedule[1]

    def test_remaining_decrements(self, spawner_1x1):
        total = spawner_1x1.total_episodes
        assert spawner_1x1.remaining == total
        spawner_1x1.next_config()
        spawner_1x1.mark_success()
        assert spawner_1x1.remaining == total - 1

    def test_stop_iteration_when_exhausted(self, spawner_1x1):
        for _ in range(spawner_1x1.total_episodes):
            spawner_1x1.next_config()
            spawner_1x1.mark_success()
        assert spawner_1x1.remaining == 0
        with pytest.raises(StopIteration):
            spawner_1x1.next_config()

    def test_full_schedule_traversal(self, spawner_2x2):
        """Walk entire schedule — every config matches schedule order."""
        for i in range(spawner_2x2.total_episodes):
            cfg = spawner_2x2.next_config()
            assert cfg == spawner_2x2._schedule[i]
            spawner_2x2.mark_success()


# ---------------------------------------------------------------------------
# BinMap persistence
# ---------------------------------------------------------------------------

class TestBinMap:
    """Verify BinMap save/load and episode tracking."""

    def test_json_roundtrip(self):
        bm = BinMap(n_cols=3, n_rows=3, object_key="cube",
                     ood_bins=frozenset({(1, 1)}))
        cfg = SpawnConfig("cube", 0.5, -0.7, 1.23, 0, 0)
        bm.log_episode(0, cfg, success=True)
        bm.log_episode(1, cfg, success=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_binmap.json"
            bm.save(path)
            loaded = BinMap.load(path)

        assert loaded.n_cols == 3
        assert loaded.n_rows == 3
        assert loaded.object_key == "cube"
        assert len(loaded.episodes) == 2
        assert loaded.episodes[0].success is True
        assert loaded.episodes[1].success is False
        assert (1, 1) in loaded.ood_bins

    def test_visit_counts_grid(self):
        bm = BinMap(n_cols=2, n_rows=2, object_key="cube")
        cfg00 = SpawnConfig("cube", 0.1, 0.1, 0.0, 0, 0)
        cfg11 = SpawnConfig("cube", 0.2, 0.2, 0.0, 1, 1)
        bm.log_episode(0, cfg00, success=True)
        bm.log_episode(1, cfg00, success=True)
        bm.log_episode(2, cfg00, success=False)  # does not count
        bm.log_episode(3, cfg11, success=True)

        grid = bm.visit_counts_grid()
        assert grid[0][0] == 2  # bin (0,0)
        assert grid[1][1] == 1  # bin (1,1)
        assert grid[0][1] == 0  # bin (1,0)

    def test_render_ascii(self):
        bm = BinMap(n_cols=2, n_rows=2, object_key="cube",
                     ood_bins=frozenset({(1, 1)}))
        cfg = SpawnConfig("cube", 0.1, 0.1, 0.0, 0, 0)
        bm.log_episode(0, cfg, success=True)
        output = bm.render_ascii(active_col=0, active_row=0)
        assert "OOD" in output
        assert "[  *]" in output
        assert "cube" in output


# ---------------------------------------------------------------------------
# Resume from BinMap
# ---------------------------------------------------------------------------

class TestResume:
    """Verify BinSpawner.restore_from_binmap() cursor reconstruction."""

    def test_resume_cursor_position(self, ws):
        spawner = BinSpawner(ws, n_cols=2, n_rows=2, half_extent_m=0.02)
        bm = BinMap(n_cols=2, n_rows=2, object_key="cube")

        # Simulate 5 successful + 2 failed episodes
        for i in range(5):
            cfg = spawner.next_config()
            bm.log_episode(i, cfg, success=True)
            spawner.mark_success()
        for i in range(5, 7):
            cfg = spawner.next_config()
            bm.log_episode(i, cfg, success=False)
            spawner.mark_retry()

        # Create fresh spawner and restore
        spawner2 = BinSpawner(ws, n_cols=2, n_rows=2, half_extent_m=0.02)
        spawner2.restore_from_binmap(bm)
        assert spawner2._cursor == 5
        assert spawner2.remaining == spawner2.total_episodes - 5

        # Next config should match schedule[5]
        cfg_resumed = spawner2.next_config()
        assert cfg_resumed == spawner2._schedule[5]

    def test_resume_grid_mismatch_raises(self, ws):
        spawner = BinSpawner(ws, n_cols=2, n_rows=2, half_extent_m=0.02)
        bm = BinMap(n_cols=3, n_rows=3, object_key="cube")
        with pytest.raises(ValueError, match="does not match"):
            spawner.restore_from_binmap(bm)

    def test_resume_roundtrip_via_json(self, ws):
        """Save BinMap to JSON, load, restore — cursor correct."""
        spawner = BinSpawner(ws, n_cols=1, n_rows=1, half_extent_m=0.02)
        bm = BinMap(n_cols=1, n_rows=1, object_key="cube")

        # Record 10 successes
        for i in range(10):
            cfg = spawner.next_config()
            bm.log_episode(i, cfg, success=True)
            spawner.mark_success()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "binmap.json"
            bm.save(path)
            loaded = BinMap.load(path)

        spawner2 = BinSpawner(ws, n_cols=1, n_rows=1, half_extent_m=0.02)
        spawner2.restore_from_binmap(loaded)
        assert spawner2._cursor == 10
        assert spawner2.next_config() == spawner2._schedule[10]


# ---------------------------------------------------------------------------
# WorkspaceBounds
# ---------------------------------------------------------------------------

class TestWorkspaceBounds:
    """Verify WorkspaceBounds construction and safe_bounds."""

    def test_from_center(self):
        ws = WorkspaceBounds.from_center(
            center_x=0.0, center_y=0.0,
            n_cols=4, n_rows=4, bin_size=0.05,
        )
        assert ws.x_min_m == pytest.approx(-0.10)
        assert ws.x_max_m == pytest.approx(0.10)
        assert ws.y_min_m == pytest.approx(-0.10)
        assert ws.y_max_m == pytest.approx(0.10)

    def test_from_center_with_origin(self):
        ws = WorkspaceBounds.from_center(
            center_x=0.0, center_y=0.0,
            n_cols=5, n_rows=5, bin_size=0.05,
            origin_x=0.20, origin_y=0.0,
        )
        assert ws.origin_x == 0.20
        assert ws.origin_y == 0.0

    def test_safe_bounds_insets(self):
        ws = WorkspaceBounds(x_min_m=0.0, x_max_m=0.10, y_min_m=0.0, y_max_m=0.10)
        safe = ws.safe_bounds(half_extent_m=0.02)
        assert safe.x_min_m == pytest.approx(0.02)
        assert safe.x_max_m == pytest.approx(0.08)

    def test_safe_bounds_degenerate_raises(self):
        ws = WorkspaceBounds(x_min_m=0.0, x_max_m=0.03, y_min_m=0.0, y_max_m=0.03)
        with pytest.raises(ValueError, match="too large"):
            ws.safe_bounds(half_extent_m=0.02)


# ---------------------------------------------------------------------------
# parse_ood_bins
# ---------------------------------------------------------------------------

class TestParseOodBins:

    def test_empty_string(self):
        assert parse_ood_bins("") == frozenset()

    def test_single_bin(self):
        assert parse_ood_bins("3,1") == frozenset({(3, 1)})

    def test_multiple_bins(self):
        result = parse_ood_bins("3,1;4,1;3,2;4,2")
        assert result == frozenset({(3, 1), (4, 1), (3, 2), (4, 2)})

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid OOD"):
            parse_ood_bins("3,1,2")

    def test_whitespace_tolerance(self):
        result = parse_ood_bins(" 3,1 ; 4,2 ")
        assert result == frozenset({(3, 1), (4, 2)})
