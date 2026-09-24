"""EpisodeController — State Machine for gamepad recording episodes.

Manages S/R key transitions, BinSpawner advance/retry logic, BinMap
persistence, and scene state capture. Adapted from the TeleopFSM in
``leisaac_experiments/teleop/fsm.py``.

States::

    IDLE       — waiting for S (start recording)
    RECORDING  — S pressed, frames being collected via add_frame()

Transitions::

    IDLE  →  S  →  RECORDING
    RECORDING  →  S  →  save + IDLE
    RECORDING  →  R  →  discard + reset + IDLE
    IDLE  →  R  →  advance/retry + reset + IDLE
"""

__version__ = "1.0.1"

import logging
import math
from enum import IntEnum, auto

import torch

logger = logging.getLogger(__name__)


class EpisodeState(IntEnum):
    """Recording state machine states."""
    IDLE = auto()
    RECORDING = auto()


class EpisodeController:
    """S/R Key State Machine with BinSpawner + BinMap integration.

    The controller owns the episode lifecycle logic that was previously
    scattered across boolean flags in policy_recorder.py. The main loop
    only needs to call ``process_keys()`` and ``add_frame()`` each tick.

    Args:
        session: RecordingSession instance for save/discard/add_frame.
        env: Isaac Lab environment (for env.reset + cube teleport).
        spawner: BinSpawner instance, or None if bin spawning is off.
        bin_map: BinMap instance, or None.
        binmap_path: Path to binmap.json, or None.
        scene_state_fn: Callable(env, episode, repo_root, seed, object_key)
            for saving scene state after reset.
        apply_spawn_fn: Callable(env, spawner) for cube teleport after reset.
        action_gen: Action generator (has .reset() method).
        repo_root: Dataset root path (for scene_state save).
        seed: Random seed.
        object_key: Scene object key (e.g. "cube").
    """

    def __init__(
        self,
        session,
        env,
        spawner=None,
        bin_map=None,
        binmap_path=None,
        scene_state_fn=None,
        apply_spawn_fn=None,
        action_gen=None,
        repo_root: str = "",
        seed: int = 42,
        object_key: str = "cube",
        start_episode: int = 0,
    ) -> None:
        self.session = session
        self.env = env
        self.spawner = spawner
        self.bin_map = bin_map
        self.binmap_path = binmap_path
        self._save_scene_state = scene_state_fn
        self._apply_spawn = apply_spawn_fn
        self.action_gen = action_gen
        self.repo_root = repo_root
        self.seed = seed
        self.object_key = object_key

        self.state = EpisodeState.IDLE
        self.episode = start_episode
        self.step = 0

    # --- Key handlers ---

    def on_key_s(self) -> None:
        """Handle S key press (start/stop recording)."""
        if self.state == EpisodeState.IDLE:
            self.state = EpisodeState.RECORDING
            self.step = 0
            logger.info(">>> RECORDING STARTED — episode %d", self.episode + 1)

        elif self.state == EpisodeState.RECORDING:
            logger.info("<<< RECORDING STOPPED — %d frames captured", self.step)
            self.session.save_episode()
            self._log_success()
            self.step = 0
            self.episode += 1
            self.state = EpisodeState.IDLE

    def on_key_r(self) -> None:
        """Handle R key press (discard during recording, or advance/retry in IDLE)."""
        if self.state == EpisodeState.RECORDING:
            # Discard in-progress episode
            self.session.discard_episode()
            self.step = 0
            logger.info("R during recording → episode discarded")
            self.state = EpisodeState.IDLE

        elif self.state == EpisodeState.IDLE:
            if self.spawner is not None:
                if self.spawner._advance:
                    logger.info("R pressed → advance to next position")
                else:
                    self._log_retry()
                    self.spawner.mark_retry()

        # Always reset env on R (both IDLE and after discard)
        self._reset_env()

    def on_auto_episode_end(self) -> None:
        """Handle auto-record episode completion (episode_length_steps reached).

        State stays ``RECORDING`` so the next auto-episode starts capturing
        frames immediately — this is the auto_record contract.
        """
        if self.state != EpisodeState.RECORDING:
            # Guard: prevent saving an empty buffer if called while IDLE
            return
        logger.info("Episode %d complete (%d frames) — saving...",
                    self.episode + 1, self.step)
        self.session.save_episode()
        self._log_success()
        self.step = 0
        self.episode += 1
        self._reset_env()

    # --- Frame handling ---

    def add_frame(self, frame: dict) -> None:
        """Add frame if currently recording."""
        if self.state == EpisodeState.RECORDING:
            self.session.add_frame(frame)
            self.step += 1

    @property
    def is_recording(self) -> bool:
        """True if currently recording frames."""
        return self.state == EpisodeState.RECORDING

    @property
    def is_done(self) -> bool:
        """True if BinSpawner schedule is exhausted."""
        if self.spawner is not None:
            return self.spawner.remaining == 0
        return False

    # --- Internal helpers ---

    def _reset_env(self) -> None:
        """Reset environment, apply spawn config, save scene state."""
        with torch.inference_mode():
            self.env.reset()
        if self.spawner is not None and self._apply_spawn is not None:
            self._apply_spawn(self.env, self.spawner)
        if self._save_scene_state is not None:
            self._save_scene_state(
                self.env, self.episode, self.repo_root,
                self.seed, self.object_key,
            )
        if self.action_gen is not None and hasattr(self.action_gen, "reset"):
            self.action_gen.reset()

    def _log_success(self) -> None:
        """Log successful episode to BinMap + advance spawner."""
        if self.spawner is not None and self.bin_map is not None:
            self.bin_map.log_episode(
                self.episode, self.spawner._current_config, success=True,
            )
            self.bin_map.save(self.binmap_path)
            self.spawner.mark_success()
            self._print_spawner_status()

    def _log_retry(self) -> None:
        """Log retry to BinMap (no spawner advance)."""
        if self.spawner is not None and self.bin_map is not None:
            self.bin_map.log_episode(
                self.episode, self.spawner._current_config, success=False,
            )
            self.bin_map.save(self.binmap_path)
            logger.info("R pressed → retry bin (%d,%d) yaw=%.0f°",
                        self.spawner._current_config.bin_col,
                        self.spawner._current_config.bin_row,
                        math.degrees(self.spawner._current_config.yaw_rad))

    def _print_spawner_status(self) -> None:
        """Print BinSpawner debug info + ASCII grid."""
        n_rot = len(self.spawner.ROTATIONS_RAD)
        idx_in_bin = self.spawner._cursor % self.spawner.episodes_per_bin
        pos_idx = idx_in_bin // n_rot
        rot_idx = idx_in_bin % n_rot

        cfg = self.spawner._current_config
        if cfg is not None:
            logger.info(
                "\n"
                "  ┌─ BinSpawner ─────────────────────────────\n"
                "  │ Bin (%d,%d)  Position P%d/9  Rotation %d/3 (%.1f°)\n"
                "  │ Cube: (%.3f, %.3f)  Episode %d/%d\n"
                "  │ Bin progress: %d/%d done\n"
                "  └───────────────────────────────────────────",
                cfg.bin_col, cfg.bin_row, pos_idx, rot_idx + 1,
                math.degrees(cfg.yaw_rad),
                cfg.x_m, cfg.y_m,
                self.spawner._cursor + 1, self.spawner.total_episodes,
                idx_in_bin, self.spawner.episodes_per_bin,
            )

        logger.info("\n%s", self.bin_map.render_ascii(
            active_col=cfg.bin_col if cfg else -1,
            active_row=cfg.bin_row if cfg else -1,
        ))
