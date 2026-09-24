# SPDX-License-Identifier: MIT
"""Pipeline utilities for LeRobot v3.0 dataset lifecycle management.

Provides centralized dataset validation, cleanup, and resume handling
used by all pipeline scripts (recorder, visualizer, trainer, inference).
"""

from __future__ import annotations

__version__ = "1.0.0"

import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

try:
    from so101_mvbench.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages LeRobot v3.0 dataset lifecycle -- validation, cleanup, resume."""

    @staticmethod
    def validate_dataset(dataset_path: Path) -> dict:
        """Check dataset integrity and return a status dict.

        Args:
            dataset_path: Root path of the LeRobot dataset.

        Returns:
            Dict with keys:
                exists (bool), valid (bool), total_episodes (int),
                total_frames (int), valid_files (int), corrupt_files (list[str])
        """
        result = {
            "exists": False,
            "valid": False,
            "total_episodes": 0,
            "total_frames": 0,
            "valid_files": 0,
            "corrupt_files": [],
        }

        if not dataset_path.exists():
            return result
        result["exists"] = True

        info_file = dataset_path / "meta" / "info.json"
        tasks_file = dataset_path / "meta" / "tasks.parquet"
        if not info_file.exists() or not tasks_file.exists():
            return result

        try:
            with open(info_file) as f:
                info = json.load(f)
            result["total_episodes"] = info.get("total_episodes", 0)
            result["total_frames"] = info.get("total_frames", 0)
        except (json.JSONDecodeError, OSError):
            return result

        # Validate parquet files
        data_dir = dataset_path / "data"
        if not data_dir.exists():
            result["valid"] = result["total_episodes"] == 0
            return result

        parquet_files = sorted(data_dir.rglob("*.parquet"))
        for pf in parquet_files:
            if DatasetManager._is_valid_parquet(pf):
                result["valid_files"] += 1
            else:
                result["corrupt_files"].append(str(pf.relative_to(dataset_path)))

        result["valid"] = len(result["corrupt_files"]) == 0 and result["valid_files"] > 0
        return result

    @staticmethod
    def _is_valid_parquet(path: Path) -> bool:
        """Check if a parquet file is readable."""
        try:
            import pyarrow.parquet as pq
            pq.read_metadata(path)
            return True
        except Exception:
            return False

    @staticmethod
    def cleanup_corrupt_tail(dataset_path: Path) -> bool:
        """Remove corrupt trailing parquet file(s) from aborted recording.

        When Ctrl+C kills the recorder mid-write, the async_episode_processor
        leaves a corrupt .parquet file. This method detects and removes it,
        then updates info.json to match the remaining valid data.

        Args:
            dataset_path: Root path of the LeRobot dataset.

        Returns:
            True if cleanup was performed, False if dataset was already clean.
        """
        status = DatasetManager.validate_dataset(dataset_path)
        if not status["exists"] or not status["corrupt_files"]:
            return False

        cleaned = False
        for corrupt_rel in status["corrupt_files"]:
            corrupt_path = dataset_path / corrupt_rel
            if corrupt_path.exists():
                logger.warning("Removing corrupt file: %s", corrupt_rel)
                corrupt_path.unlink()
                cleaned = True

        if cleaned:
            DatasetManager._recount_info_json(dataset_path)

        return cleaned

    @staticmethod
    def _recount_info_json(dataset_path: Path):
        """Recount episodes/frames from valid parquet files and update info.json."""
        info_file = dataset_path / "meta" / "info.json"
        if not info_file.exists():
            return

        data_dir = dataset_path / "data"
        parquet_files = sorted(data_dir.rglob("*.parquet")) if data_dir.exists() else []

        total_frames = 0
        total_episodes = 0
        for pf in parquet_files:
            if DatasetManager._is_valid_parquet(pf):
                try:
                    import pyarrow.parquet as pq
                    meta = pq.read_metadata(pf)
                    total_frames += meta.num_rows
                    total_episodes += 1
                except Exception:
                    pass

        try:
            with open(info_file) as f:
                info = json.load(f)
            info["total_episodes"] = total_episodes
            info["total_frames"] = total_frames
            info["splits"] = {"train": f"0:{total_episodes}"}
            with open(info_file, "w") as f:
                json.dump(info, f, indent=4)
            logger.info("Updated info.json: %d episodes, %d frames", total_episodes, total_frames)
        except (json.JSONDecodeError, OSError):
            logger.exception("Failed to update info.json")

    @staticmethod
    def handle_existing_dataset(dataset_path: Path, auto: bool = False) -> str:
        """Check for existing dataset and prompt user for action.

        Automatically cleans up corrupt trailing files before presenting
        the menu. In auto mode or non-interactive sessions, makes a safe
        default choice without prompting.

        Args:
            dataset_path: Root path of the LeRobot dataset.
            auto: If True, skip menu (delete incomplete, resume valid).

        Returns:
            One of: "continue", "fresh", "quit".
        """
        if not dataset_path.exists():
            return "fresh"

        # Auto-cleanup corrupt files first
        DatasetManager.cleanup_corrupt_tail(dataset_path)

        status = DatasetManager.validate_dataset(dataset_path)
        is_valid = status["valid"]

        if auto:
            if not is_valid:
                logger.warning("Auto mode: removing incomplete dataset: %s", dataset_path)
                shutil.rmtree(dataset_path, ignore_errors=True)
                return "fresh"
            return "continue"

        # Non-interactive fallback (no TTY)
        if not sys.stdin.isatty():
            if is_valid:
                logger.info("No TTY: resuming existing valid dataset at %s", dataset_path)
                return "continue"
            logger.warning("No TTY: removing incomplete dataset at %s", dataset_path)
            shutil.rmtree(dataset_path, ignore_errors=True)
            return "fresh"

        # Interactive menu
        print()
        print("================================================================")
        print("  EXISTING DATASET DETECTED")
        print(f"  Path:      {dataset_path}")
        print(f"  Valid:     {'YES' if is_valid else 'NO (incomplete)'}")
        print(f"  Episodes:  {status['total_episodes']}")
        print(f"  Frames:    {status['total_frames']}")
        if status["corrupt_files"]:
            print(f"  Cleaned:   {len(status['corrupt_files'])} corrupt file(s) removed")
        print("================================================================")
        if is_valid:
            print("  [c] CONTINUE: Resume recording (add new episodes)")
        print("  [b] BACKUP:   Rename to _backup_TIMESTAMP and start fresh")
        print("  [d] DELETE:   Delete and start fresh")
        print("  [q] QUIT:     Abort")
        print("================================================================")

        while True:
            try:
                choice = input("  Selection (c/b/d/q): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return "quit"

            if choice == "c" and is_valid:
                logger.info("Resuming recording in existing dataset...")
                return "continue"
            elif choice == "b":
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = dataset_path.parent / f"{dataset_path.name}_backup_{ts}"
                try:
                    shutil.move(str(dataset_path), str(backup_path))
                    logger.info("Backed up to: %s", backup_path.name)
                except OSError:
                    logger.exception("Backup failed — aborting")
                    return "quit"
                return "fresh"
            elif choice == "d":
                logger.warning("Deleting existing dataset: %s", dataset_path)
                shutil.rmtree(dataset_path, ignore_errors=True)
                return "fresh"
            elif choice == "q":
                return "quit"
            else:
                valid_choices = "c/b/d/q" if is_valid else "b/d/q"
                print(f"  Invalid choice. Enter {valid_choices}.")
