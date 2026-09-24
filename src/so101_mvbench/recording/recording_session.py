"""RecordingSession — Context Manager for LeRobotDataset lifecycle.

Wraps LeRobotDataset + VideoEncodingManager to guarantee finalize()
runs on exit (Ctrl+C, exception, or normal completion). Based on
the pattern from ``lerobot/scripts/lerobot_record.py``.

Usage::

    with RecordingSession(repo_id, root, fps, features) as session:
        session.add_frame(frame_dict)
        session.save_episode()
    # finalize() called automatically — parquet footer written

The ``atexit`` hook provides an additional safety net in case __exit__
is bypassed (e.g. os._exit or SIGKILL).
"""

__version__ = "1.2.0"

import atexit
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class RecordingSession:
    """Context Manager for LeRobotDataset with streaming_encoding.

    Handles dataset creation/resume, VideoEncodingManager lifecycle,
    and guaranteed finalize() on exit.

    Args:
        repo_id: LeRobot repo ID (e.g. "local/liftcube_binned").
        root: Dataset root directory path.
        fps: Recording framerate.
        features: Feature dict for LeRobotDataset.create().
        task_name: Human-readable task name for frame dicts.
    """

    def __init__(
        self,
        repo_id: str,
        root: str,
        fps: int,
        features: dict,
        task_name: str = "",
    ) -> None:
        self.repo_id = repo_id
        self.root = Path(root)
        self.fps = fps
        self.features = features
        self.task_name = task_name
        self.dataset: "LeRobotDataset | None" = None
        self._video_mgr: "VideoEncodingManager | None" = None
        self._finalized = False

    def __enter__(self) -> "RecordingSession":
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.video_utils import VideoEncodingManager

        os.environ["HF_LEROBOT_HOME"] = str(self.root.parent)

        # Resume only if dataset is complete (info.json + episodes parquet)
        episodes_dir = self.root / "meta" / "episodes"
        has_valid_dataset = (
            self.root.exists()
            and (self.root / "meta" / "info.json").exists()
            and episodes_dir.exists()
            and any(episodes_dir.rglob("*.parquet"))
        )

        if has_valid_dataset:
            try:
                # streaming_encoding=True has to be passed again when resuming:
                # LeRobot does not persist it in info.json. Leaving it at the
                # default False silently falls back to writing one PNG per frame
                # synchronously, which collapses the recording frame rate.
                # See lerobot_dataset.py:771.
                self.dataset = LeRobotDataset(
                    repo_id=self.repo_id,
                    root=str(self.root),
                    streaming_encoding=True,
                )
                assert self.dataset is not None  # for type-checker
                # Validate caller-provided config matches the resumed dataset.
                # Silent fps/feature drift would only surface in cryptic
                # add_frame errors later — fail early instead.
                resumed_fps = self.dataset.meta.fps
                if resumed_fps != self.fps:
                    raise ValueError(
                        f"fps mismatch on resume: caller={self.fps}, "
                        f"dataset={resumed_fps}"
                    )
                logger.info("Resumed dataset: %s (%d episodes, streaming_encoding=True)",
                            self.root, self.dataset.meta.total_episodes)
            except (FileNotFoundError, NotADirectoryError, ValueError) as e:
                # Only "dataset unloadable" errors trigger recreate. Other
                # exceptions (TypeError, PermissionError, MemoryError, ...)
                # propagate so we do NOT destructively shutil.rmtree a
                # working dataset directory.
                logger.warning("Dataset unloadable — recreating: %s (%s)", self.root, e)
                has_valid_dataset = False

        if not has_valid_dataset:
            # Remove empty/incomplete directory (LeRobotDataset.create needs exist_ok=False)
            if self.root.exists():
                import shutil
                shutil.rmtree(self.root)
                logger.info("Removed incomplete dataset dir: %s", self.root)
            self.dataset = LeRobotDataset.create(
                repo_id=self.repo_id,
                fps=self.fps,
                features=self.features,
                root=str(self.root),
                robot_type="so101_follower",
                streaming_encoding=True,
            )
            # Rotate the parquet files aggressively (1 MB instead of the default).
            # Each rotation makes LeRobot call _close_writer(), which writes the
            # Thrift footer -- and a parquet file without its footer is unreadable.
            # So a kill -9 can now cost at most the file currently open, rather than
            # a whole multi-episode chunk, which is what happened on 2026-04-18.
            self.dataset.meta.update_chunk_settings(
                video_files_size_in_mb=1,
                data_files_size_in_mb=1,
            )
            logger.info("Created dataset (streaming_encoding=True): %s",
                        self.root)

        self._video_mgr = VideoEncodingManager(self.dataset)
        self._video_mgr.__enter__()
        self._finalized = False

        atexit.register(self._finalize)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._finalize(exc_type, exc_val, exc_tb)

    def _finalize(self, exc_type=None, exc_val=None, exc_tb=None) -> None:
        """Flush parquet footer + close video encoder. Idempotent.

        Forwards exception info to VideoEncodingManager.__exit__ so it can
        react to in-flight exceptions. atexit calls with no args (defaults
        to None, None, None).

        NOTE: VideoEncodingManager.__exit__ already calls dataset.finalize()
        internally (see lerobot video_utils.py:1087). We do NOT call it a
        second time — LeRobotDataset._close_writer() has no idempotency
        guard, so a double-close would raise RuntimeError and potentially
        leave the Thrift footer unwritten. This is the likely mechanism
        behind the 2026-04-18 data-loss incident.
        """
        if self._finalized:
            return
        self._finalized = True
        try:
            if self._video_mgr is not None:
                self._video_mgr.__exit__(exc_type, exc_val, exc_tb)
            if self.dataset is not None:
                logger.info("Dataset finalized: %d episodes, %d frames",
                            self.dataset.meta.total_episodes,
                            self.dataset.meta.total_frames)
        except Exception:
            logger.exception("Dataset finalize failed")

    # --- Frame operations ---

    def add_frame(self, frame: dict) -> None:
        """Add a single frame to the current episode buffer."""
        if "task" not in frame:
            frame["task"] = self.task_name
        self.dataset.add_frame(frame)

    def save_episode(self) -> None:
        """Save the current episode (flush to parquet + video)."""
        self.dataset.save_episode()

    def discard_episode(self) -> None:
        """Discard the current in-progress episode (clear buffer)."""
        self.dataset.clear_episode_buffer()
        logger.info("Episode buffer cleared (discarded)")

    # --- Properties ---

    @property
    def total_episodes(self) -> int:
        """Number of saved episodes."""
        return self.dataset.meta.total_episodes if self.dataset else 0

    @property
    def total_frames(self) -> int:
        """Total frames across all saved episodes."""
        return self.dataset.meta.total_frames if self.dataset else 0
