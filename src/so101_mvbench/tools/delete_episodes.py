#!/usr/bin/env python3
"""Delete specific episodes from a LeRobot v3 dataset.

Removes episode data from parquet, videos, scene_states, and binmap.
Re-indexes remaining episodes so indices are contiguous.

Usage:
    # List all episodes
    python delete_episodes.py --dataset_root /path/to/bin_c0_r0 --list

    # Delete episodes 1 and 3
    python delete_episodes.py --dataset_root /path/to/bin_c0_r0 --delete 1,3

    # Dry run (show what would be deleted)
    python delete_episodes.py --dataset_root /path/to/bin_c0_r0 --delete 1,3 --dry_run
"""

VERSION = "1.0.0"

import argparse
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np


def list_episodes(root: Path) -> None:
    """Print all episodes with frame counts."""
    info = json.load(open(root / "meta" / "info.json"))
    print(f"Dataset: {root}")
    print(f"Total: {info['total_episodes']} episodes, {info['total_frames']} frames")
    print()

    # Read parquet to get per-episode frame counts
    data_files = sorted((root / "data").rglob("*.parquet"))
    if not data_files:
        print("  No parquet data files found.")
        return

    all_tables = [pq.read_table(str(f)) for f in data_files]
    table = pa.concat_tables(all_tables)
    ep_indices = table["episode_index"].to_pylist()

    episodes = {}
    for idx in ep_indices:
        episodes[idx] = episodes.get(idx, 0) + 1

    print(f"  {'Ep':>4}  {'Frames':>7}  {'Duration':>8}  Videos")
    print(f"  {'─'*4}  {'─'*7}  {'─'*8}  {'─'*30}")
    fps = info.get("fps", 30)
    for ep_idx in sorted(episodes.keys()):
        n_frames = episodes[ep_idx]
        duration = n_frames / fps
        # Check video files
        videos = []
        for vid_dir in sorted((root / "videos").iterdir()):
            vid_file = vid_dir / f"chunk-000/file-{ep_idx:03d}.mp4"
            if vid_file.exists():
                size_mb = vid_file.stat().st_size / (1024 * 1024)
                videos.append(f"{vid_dir.name.split('.')[-1]}:{size_mb:.1f}MB")
        vid_str = ", ".join(videos) if videos else "none"
        print(f"  {ep_idx:4d}  {n_frames:7d}  {duration:7.1f}s  {vid_str}")

    # Binmap info
    binmap_path = root / "binmap.json"
    if binmap_path.exists():
        bm = json.load(open(binmap_path))
        print(f"\n  BinMap: {len(bm['episodes'])} entries "
              f"({sum(1 for e in bm['episodes'] if e['success'])} success, "
              f"{sum(1 for e in bm['episodes'] if not e['success'])} retry)")


def delete_episodes(root: Path, delete_indices: list[int], dry_run: bool = False) -> None:
    """Delete specified episodes and re-index remaining."""
    info_path = root / "meta" / "info.json"
    info = json.load(open(info_path))
    total_eps = info["total_episodes"]

    # Validate indices
    for idx in delete_indices:
        if idx < 0 or idx >= total_eps:
            raise ValueError(f"Episode {idx} out of range [0, {total_eps - 1}]")

    delete_set = set(delete_indices)
    print(f"{'DRY RUN: ' if dry_run else ''}Deleting episodes {sorted(delete_set)} "
          f"from {root}")

    # --- 1. Filter parquet data ---
    data_files = sorted((root / "data").rglob("*.parquet"))
    all_tables = [pq.read_table(str(f)) for f in data_files]
    table = pa.concat_tables(all_tables)

    # Filter out deleted episodes
    ep_col = table["episode_index"].to_pylist()
    keep_mask = [idx not in delete_set for idx in ep_col]
    filtered = table.filter(keep_mask)

    # Re-index episode_index to be contiguous
    old_eps = sorted(set(ep_col) - delete_set)
    remap = {old: new for new, old in enumerate(old_eps)}
    new_ep_col = [remap[idx] for idx in filtered["episode_index"].to_pylist()]
    filtered = filtered.set_column(
        filtered.schema.get_field_index("episode_index"),
        "episode_index",
        pa.array(new_ep_col, type=pa.int64()),
    )

    # Re-index frame_index and index to be contiguous
    n_remaining = len(filtered)
    for col_name in ["frame_index", "index"]:
        if col_name in filtered.column_names:
            filtered = filtered.set_column(
                filtered.schema.get_field_index(col_name),
                col_name,
                pa.array(list(range(n_remaining)), type=pa.int64()),
            )

    kept_frames = len(filtered)
    kept_episodes = len(old_eps)
    print(f"  Parquet: {len(table)} → {kept_frames} frames "
          f"({len(table) - kept_frames} removed)")

    if not dry_run:
        # Remove old files, write single new file
        for f in data_files:
            f.unlink()
        out_path = root / "data" / "chunk-000" / "file-000.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(filtered, str(out_path))

    # --- 2. Delete video files ---
    for vid_dir in sorted((root / "videos").iterdir()):
        for idx in sorted(delete_set):
            vid_file = vid_dir / f"chunk-000/file-{idx:03d}.mp4"
            if vid_file.exists():
                print(f"  Video: rm {vid_file.relative_to(root)}")
                if not dry_run:
                    vid_file.unlink()

        # Rename remaining videos to close gaps
        if not dry_run:
            chunk_dir = vid_dir / "chunk-000"
            remaining = sorted(chunk_dir.glob("*.mp4"))
            for new_idx, f in enumerate(remaining):
                new_name = chunk_dir / f"file-{new_idx:03d}.mp4"
                if f != new_name:
                    f.rename(new_name)

    # --- 3. Delete scene_state files ---
    for idx in sorted(delete_set):
        ss = root / "meta" / f"episode_{idx:06d}_scene_state.json"
        if ss.exists():
            print(f"  Scene state: rm {ss.name}")
            if not dry_run:
                ss.unlink()

    # Rename remaining scene_states
    if not dry_run:
        remaining_ss = sorted((root / "meta").glob("episode_*_scene_state.json"))
        for new_idx, f in enumerate(remaining_ss):
            new_name = f.parent / f"episode_{new_idx:06d}_scene_state.json"
            if f != new_name:
                f.rename(new_name)

    # --- 4. Update meta/info.json ---
    print(f"  info.json: {total_eps} → {kept_episodes} episodes, "
          f"{info['total_frames']} → {kept_frames} frames")
    if not dry_run:
        info["total_episodes"] = kept_episodes
        info["total_frames"] = kept_frames
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    # --- 5. Update meta/episodes parquet ---
    # Each episode may have a different schema (first episode lacks image stats).
    # Write kept episodes as individual files to preserve schemas.
    ep_meta_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if ep_meta_files:
        kept = []
        total_meta = 0
        for f in ep_meta_files:
            t = pq.read_table(str(f))
            total_meta += len(t)
            for i in range(len(t)):
                if t["episode_index"][i].as_py() not in delete_set:
                    row = t.slice(i, 1)
                    kept.append(row)
        print(f"  Episodes meta: {total_meta} → {len(kept)} entries")
        if not dry_run:
            for f in ep_meta_files:
                f.unlink()
            out_dir = root / "meta" / "episodes" / "chunk-000"
            out_dir.mkdir(parents=True, exist_ok=True)
            for new_idx, row in enumerate(kept):
                row = row.set_column(
                    row.schema.get_field_index("episode_index"),
                    "episode_index",
                    pa.array([new_idx], type=pa.int64()),
                )
                pq.write_table(row, str(out_dir / f"file-{new_idx:03d}.parquet"))

    # --- 6. Update binmap.json ---
    binmap_path = root / "binmap.json"
    if binmap_path.exists():
        bm = json.load(open(binmap_path))
        # Remove entries for deleted episodes (by episode_idx)
        bm["episodes"] = [
            e for e in bm["episodes"]
            if e["episode_idx"] not in delete_set
        ]
        # Re-index remaining successful episodes
        success_idx = 0
        for e in bm["episodes"]:
            if e["success"]:
                e["episode_idx"] = success_idx
                success_idx += 1
        n_removed = len(delete_set)
        print(f"  BinMap: removed {n_removed} episode entries")
        if not dry_run:
            with open(binmap_path, "w") as f:
                json.dump(bm, f, indent=2)

    print(f"\n{'DRY RUN complete.' if dry_run else 'Done.'}")


def main():
    parser = argparse.ArgumentParser(description="Delete episodes from LeRobot v3 dataset.")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Path to dataset directory.")
    parser.add_argument("--list", action="store_true",
                        help="List all episodes with frame counts.")
    parser.add_argument("--delete", type=str, default=None,
                        help="Comma-separated episode indices to delete (e.g. '1,3').")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show what would be deleted without doing it.")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    if not root.exists():
        print(f"Error: {root} does not exist.")
        return

    if args.list:
        list_episodes(root)
    elif args.delete:
        indices = [int(x.strip()) for x in args.delete.split(",")]
        delete_episodes(root, indices, dry_run=args.dry_run)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
