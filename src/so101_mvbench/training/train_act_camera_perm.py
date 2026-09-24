# SPDX-License-Identifier: MIT
"""ACT trainer wrapper for camera-permutation experiments.

Expands `--cameras wrist front` into the LeRobot
`--policy.input_features` JSON, then invokes `lerobot.scripts.lerobot_train`
with the rest of the args.

Reduces a typical Phase-1 baseline command from ~10 lines (multi-line bash
with embedded JSON) to a single line:

    train_act_camera_perm \\
        --cameras wrist front \\
        --dataset bin_c1_r1__wrist_front_sideLR_top \\
        --steps 30000 --batch_size 8 \\
        --output_dir outputs/2cam_wrist_front

Cameras are looked up from `_CAMERA_SHAPES` (default: 480×640 RGB for all
five workspace cameras: wrist, front, side_l, side_r, top).

Output dir contains LeRobot's standard `checkpoints/`, `train.log`, etc.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

__version__ = "1.0.0"


# Per-camera input shapes (CHW, RGB). Adapt if you record at a different
# resolution.
_CAMERA_SHAPES: dict[str, list[int]] = {
    "wrist":  [3, 480, 640],
    "front":  [3, 480, 640],
    "side_l": [3, 480, 640],
    "side_r": [3, 480, 640],
    "top":    [3, 480, 640],
    "front_bottom": [3, 480, 640],
    "left_bottom":  [3, 480, 640],
    "right_bottom": [3, 480, 640],
    "left":   [3, 480, 640],
    "right":  [3, 480, 640],
    "back":   [3, 480, 640],
}

# Robot has 6 joints (5 arm + 1 gripper). State is observed at every frame.
_STATE_SHAPE: list[int] = [6]


def build_input_features(cameras: list[str]) -> dict[str, dict]:
    """Build the LeRobot `policy.input_features` dict from a list of cameras.

    Args:
        cameras: list of camera names (e.g. ["wrist", "front"]). Each must
            exist as a key in `_CAMERA_SHAPES`.

    Returns:
        A dict like::

            {
              "observation.images.wrist": {"type": "VISUAL", "shape": [3,480,640]},
              "observation.images.front": {"type": "VISUAL", "shape": [3,480,640]},
              "observation.state":         {"type": "STATE",  "shape": [6]},
            }
    """
    if not cameras:
        raise ValueError("--cameras requires at least one camera name")

    unknown = [c for c in cameras if c not in _CAMERA_SHAPES]
    if unknown:
        known = ", ".join(_CAMERA_SHAPES)
        raise ValueError(f"unknown camera(s): {unknown}. Known: {known}")

    features: dict[str, dict] = {}
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "type": "VISUAL",
            "shape": _CAMERA_SHAPES[cam],
        }
    features["observation.state"] = {"type": "STATE", "shape": _STATE_SHAPE}
    return features


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ACT trainer wrapper for camera-permutation experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--cameras", nargs="+", required=True,
                        help="Camera names (e.g. wrist front). Must be subset "
                             f"of {sorted(_CAMERA_SHAPES)}.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name under datasets/liftcube_binned/ "
                             "(e.g. bin_c1_r1__wrist_front_sideLR_top).")
    parser.add_argument("--dataset_root", type=Path, default=None,
                        help="Override dataset root path. Default: "
                             "datasets/liftcube_binned/<DATASET>.")
    parser.add_argument("--episodes", type=str, default=None,
                        help="JSON list of episode indices to train on "
                             "(e.g. '[0,1,3,4,...]'). Default: all episodes "
                             "of the dataset.")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="LeRobot --output_dir target.")
    parser.add_argument("--job_name", type=str, default=None,
                        help="LeRobot --job_name. Default: basename of output_dir.")
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Dataloader workers. More = faster data loading. Default: 4.")
    parser.add_argument("--save_freq", type=int, default=2_000)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy_device", type=str, default="cuda")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Upload checkpoints to HuggingFace Hub. "
                             "Default: false.")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B logging. Default: disabled.")
    args = parser.parse_args()

    dataset_root = args.dataset_root or Path("datasets/liftcube_binned") / args.dataset
    job_name = args.job_name or args.output_dir.name

    features = build_input_features(args.cameras)
    features_json = json.dumps(features, separators=(",", ":"))

    # LeRobot's lerobot_train asserts the output_dir does NOT exist yet
    # (refuses to overwrite non-empty dir without resume=True). We create
    # only the parent so lerobot can mkdir its own fresh output_dir.
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id=local/{args.dataset}",
        f"--dataset.root={dataset_root}",
        "--policy.type=act",
        f"--policy.device={args.policy_device}",
        f"--policy.push_to_hub={'true' if args.push_to_hub else 'false'}",
        f"--policy.input_features={features_json}",
        f"--output_dir={args.output_dir}",
        f"--job_name={job_name}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        f"--save_freq={args.save_freq}",
        f"--log_freq={args.log_freq}",
        f"--seed={args.seed}",
        f"--wandb.enable={'true' if args.wandb else 'false'}",
    ]
    if args.wandb:
        # Log metrics only — NEVER upload checkpoints as wandb artifacts.
        # lerobot default is disable_artifact=false → uploads every ckpt
        # (~4.5 GB/run), fills small quotas + ~/.cache/wandb/artifacts.
        # Models live in outputs/<run>/checkpoints/ (LFS). See CLAUDE.md §4a.
        cmd.append("--wandb.disable_artifact=true")
    if args.episodes:
        cmd.append(f"--dataset.episodes={args.episodes}")

    print(f"[train_act_camera_perm] cameras: {args.cameras}")
    print(f"[train_act_camera_perm] running: {' '.join(cmd)}")
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
