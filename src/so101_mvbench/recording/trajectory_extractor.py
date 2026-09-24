# SPDX-License-Identifier: MIT
"""Extract per-episode action trajectories from a LeRobot v3.0 dataset.

Reads parquet files and saves each episode's action sequence as a
standalone ``.npy`` file.  This decouples the trajectory replay from the
full LeRobot import chain (torch, lerobot, etc.).

Usage:
    python trajectory_extractor.py \\
        --dataset_root datasets/so101_gamepad_teleop \\
        --output_dir datasets/trajectories/so101_gamepad_teleop
"""

from __future__ import annotations

__version__ = "1.0.0"

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

try:
    import pyarrow.parquet as pq
except ImportError:
    print("ERROR: pyarrow is required. Install: pip install pyarrow", file=sys.stderr)
    raise SystemExit(1)

try:
    from so101_mvbench.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


def extract_actions_from_parquet(parquet_path: Path) -> np.ndarray:
    """Read a single parquet file and return the action column as ndarray.

    Args:
        parquet_path: Path to a LeRobot v3.0 parquet file.

    Returns:
        Array of shape ``(num_frames, 6)`` with dtype float32.
        Each row contains 6 joint positions in raw degrees.
    """
    table = pq.read_table(parquet_path, columns=["action"])
    # action column is fixed_size_list<float>[6]
    actions_list = table["action"].to_pylist()
    actions = np.array(actions_list, dtype=np.float32)  # (num_frames, 6)
    return actions


def extract_actions_by_episode(parquet_path: Path) -> dict[int, np.ndarray]:
    """Read a chunked parquet file and split actions by ``episode_index``.

    LeRobot v3 concatenates multiple episodes into one parquet file;
    the ``episode_index`` column identifies which rows belong to which episode.

    Args:
        parquet_path: Path to a LeRobot v3.0 data parquet.

    Returns:
        Mapping ``episode_index -> actions array`` (shape ``(length, 6)``).
    """
    table = pq.read_table(parquet_path, columns=["action", "episode_index"])
    ep_col = table["episode_index"].to_numpy()
    actions_list = table["action"].to_pylist()
    actions = np.array(actions_list, dtype=np.float32)

    out: dict[int, np.ndarray] = {}
    for ep_idx in np.unique(ep_col):
        mask = ep_col == ep_idx
        out[int(ep_idx)] = actions[mask]
    return out


def extract_dataset(
    dataset_root: Path,
    output_dir: Path,
) -> dict:
    """Extract all episodes from a LeRobot v3.0 dataset to .npy files.

    Args:
        dataset_root: Root directory of the LeRobot dataset.
        output_dir: Directory to write .npy files and metadata.

    Returns:
        Metadata dict written to ``trajectories_meta.json``.

    Raises:
        FileNotFoundError: If dataset_root or info.json does not exist.
        ValueError: If no parquet files are found.
    """
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Dataset info not found: {info_path}")

    with open(info_path) as f:
        info = json.load(f)

    total_episodes = info["total_episodes"]
    fps = info["fps"]
    logger.info(
        "Dataset: %s — %d episodes, %d total frames, %d FPS",
        dataset_root.name,
        total_episodes,
        info.get("total_frames", -1),
        fps,
    )

    # Discover parquet files
    parquet_files = sorted(dataset_root.glob("data/chunk-*/file-*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {dataset_root / 'data'}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes_meta: list[dict] = []
    total_frames_extracted = 0

    # LeRobot v3 concatenates multiple episodes per parquet file.  Split by
    # the ``episode_index`` column so one .npy is written per logical episode,
    # not per parquet file.
    for pq_path in parquet_files:
        by_ep = extract_actions_by_episode(pq_path)
        for ep_idx in sorted(by_ep.keys()):
            actions = by_ep[ep_idx]
            num_frames = actions.shape[0]

            out_path = output_dir / f"episode_{ep_idx:03d}.npy"
            np.save(out_path, actions)

            episodes_meta.append({
                "index": ep_idx,
                "num_frames": num_frames,
                "source_file": pq_path.name,
            })
            total_frames_extracted += num_frames

            logger.info(
                "  Episode %03d: %d frames → %s",
                ep_idx,
                num_frames,
                out_path.name,
            )

    if len(episodes_meta) != total_episodes:
        logger.warning(
            "Extracted %d episodes but info.json claims %d",
            len(episodes_meta),
            total_episodes,
        )

    # Write metadata
    meta = {
        "source_dataset": str(dataset_root),
        "fps": fps,
        "num_episodes": len(episodes_meta),
        "total_frames": total_frames_extracted,
        "joint_names": [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ],
        "units": "raw_degrees",
        "dtype": "float32",
        "episodes": episodes_meta,
    }

    meta_path = output_dir / "trajectories_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "Extraction complete: %d episodes, %d frames → %s",
        len(episodes_meta),
        total_frames_extracted,
        output_dir,
    )
    return meta


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract action trajectories from LeRobot v3.0 dataset",
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        required=True,
        help="Root directory of the LeRobot dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for .npy files",
    )
    args = parser.parse_args()

    if not args.dataset_root.exists():
        logger.error("Dataset root does not exist: %s", args.dataset_root)
        raise SystemExit(1)

    extract_dataset(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
