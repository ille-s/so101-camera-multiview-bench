#!/usr/bin/env python3
"""Merge multiple single-bin LeRobot datasets into one multi-bin training dataset.

Usage:
    python -m so101_mvbench.assembly.bin_fusion \\
        --datasets_root /path/to/datasets \\
        --target /path/to/merged_pilot \\
        --bins bin_c1_r1 bin_c2_r1
"""

VERSION = "1.0.0"

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from so101_mvbench.logging import get_logger, configure_root_logger

logger = get_logger(__name__)
configure_root_logger()

from lerobot.datasets.aggregate import aggregate_datasets, validate_all_metadata
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


FLOAT_TOLERANCE = 1e-6


@dataclass
class BinSource:
    bin_id: str
    path: Path
    episode_count: int = 0
    frame_count: int = 0
    global_episode_offset: int = 0


@dataclass
class FusionConfig:
    target: Path
    datasets_root: Path
    bins: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    exclude_where: list[tuple[str, str]] = field(default_factory=list)
    strict: bool = False
    dry_run: bool = False


def discover_bins(datasets_root: Path) -> list[str]:
    """Find all bin_c*_r* directories under datasets_root."""
    bins = sorted(
        d.name for d in datasets_root.iterdir()
        if d.is_dir() and d.name.startswith("bin_c") and "_r" in d.name
        and not d.name.startswith(".")
    )
    logger.info("Discovered %d bins under %s", len(bins), datasets_root)
    return bins


def resolve_selection(
    available: list[str],
    explicit_bins: list[str],
    exclude: list[str],
) -> list[str]:
    """Apply bin-level selection and exclusion."""
    selected = explicit_bins if explicit_bins else available
    missing = set(selected) - set(available)
    if missing:
        raise ValueError(f"Bins not found: {sorted(missing)}")
    result = [b for b in selected if b not in exclude]
    if not result:
        raise ValueError("Selection is empty after exclusion")
    return sorted(result)


def load_binmap_episodes(binmap_path: Path) -> list[dict]:
    """Load binmap.json and return only success=true entries."""
    try:
        data = json.loads(binmap_path.read_text())
    except (json.JSONDecodeError, FileNotFoundError) as e:
        raise ValueError(f"Cannot read {binmap_path}: {e}") from e
    episodes = data.get("episodes", [])
    return [ep for ep in episodes if ep.get("success", False)]


def matches_filter(episode: dict, key: str, value: str) -> bool:
    """Check if an episode's property matches a filter value."""
    if key not in episode:
        raise ValueError(f"Unknown filter key '{key}' (available: {list(episode.keys())})")
    ep_val = episode[key]
    if isinstance(ep_val, bool):
        return ep_val == (value.lower() == "true")
    if isinstance(ep_val, (int, float)):
        try:
            return abs(ep_val - float(value)) < FLOAT_TOLERANCE
        except ValueError:
            raise ValueError(f"Cannot compare '{key}'={ep_val} with '{value}'") from None
    return str(ep_val) == value


def apply_episode_filters(
    episodes: list[dict],
    exclude_where: list[tuple[str, str]],
) -> list[int]:
    """Return episode indices that survive all exclude-where filters."""
    surviving = []
    for ep in episodes:
        excluded = any(
            matches_filter(ep, key, value) for key, value in exclude_where
        )
        if not excluded:
            surviving.append(ep["episode_idx"])
    return surviving


def validate_pre_merge(sources: list[BinSource], strict: bool) -> list[str]:
    """Run pre-merge checks. Returns list of warnings."""
    warnings = []
    all_meta = []
    for src in sources:
        info_path = src.path / "meta" / "info.json"
        if not info_path.exists():
            raise ValueError(f"Missing {info_path}")
        all_meta.append(LeRobotDatasetMetadata(src.bin_id, root=src.path))

    try:
        validate_all_metadata(all_meta)
    except ValueError as e:
        raise ValueError(f"Pre-merge validation failed: {e}") from e

    episode_counts = [src.episode_count for src in sources]
    if len(set(episode_counts)) > 1:
        msg = f"Episode counts vary: {dict(zip([s.bin_id for s in sources], episode_counts))}"
        if strict:
            raise ValueError(msg)
        warnings.append(msg)
        logger.warning(msg)

    return warnings


def _extract_frame(video_path: Path, frame_idx: int) -> "torch.Tensor":
    """Extract a single frame from a video file."""
    import torch
    from torchcodec.decoders import VideoDecoder
    decoder = VideoDecoder(str(video_path))
    return decoder[frame_idx].data


def _frames_match(frame_a: "torch.Tensor", frame_b: "torch.Tensor") -> bool:
    """Check if two frames are pixel-identical."""
    import torch
    if frame_a.shape != frame_b.shape:
        return False
    return torch.equal(frame_a, frame_b)


def _build_episode_frame_map(target: Path) -> list[dict]:
    """Build per-episode global frame ranges from the merged parquet."""
    import pyarrow.parquet as pq
    data = pq.read_table(target / "data" / "chunk-000" / "file-000.parquet")
    ep_col = data.column("episode_index").to_pylist()
    frame_col = data.column("frame_index").to_pylist()

    episodes: dict[int, dict] = {}
    for ep, fr in zip(ep_col, frame_col):
        if ep not in episodes:
            episodes[ep] = {"local_start": fr, "local_end": fr, "count": 0, "global_start": 0}
        episodes[ep]["local_end"] = fr
        episodes[ep]["count"] += 1

    offset = 0
    for ep in sorted(episodes):
        episodes[ep]["global_start"] = offset
        offset += episodes[ep]["count"]

    return [{"ep": ep, **episodes[ep]} for ep in sorted(episodes)]


def _build_source_episode_map(src: BinSource) -> list[dict]:
    """Build per-episode frame ranges from a source bin's parquet."""
    import pyarrow.parquet as pq

    parquet_files = sorted((src.path / "data").rglob("*.parquet"))
    if not parquet_files:
        return []

    import pyarrow as pa
    data = pa.concat_tables([pq.read_table(f) for f in parquet_files])
    ep_col = data.column("episode_index").to_pylist()
    frame_col = data.column("frame_index").to_pylist()

    episodes: dict[int, dict] = {}
    for ep, fr in zip(ep_col, frame_col):
        if ep not in episodes:
            episodes[ep] = {"local_start": fr, "local_end": fr, "count": 0, "global_start": 0}
        episodes[ep]["local_end"] = fr
        episodes[ep]["count"] += 1

    offset = 0
    for ep in sorted(episodes):
        episodes[ep]["global_start"] = offset
        offset += episodes[ep]["count"]

    return [{"ep": ep, **episodes[ep]} for ep in sorted(episodes)]


def validate_video_content(
    target: Path,
    sources: list[BinSource],
) -> list[str]:
    """Verify merged videos by comparing mid-episode frames against source bins.

    LeRobot v3.0 concatenates all episode videos into one MP4 per camera-key
    per chunk. Episode boundaries are resolved via timestamp indices in the
    ``episodes/*.parquet`` metadata.
    Ref: https://huggingface.co/docs/lerobot/en/using_dataset_tools

    For each source bin, picks 3 sample episodes (first, middle, last) and
    compares the mid-frame of each episode in the merged video against the
    same frame in the source video. This catches wrong ordering, corrupted
    concatenation, or shifted frame indices.
    """
    warnings = []
    results = []
    info = json.loads((target / "meta" / "info.json").read_text())
    features = info.get("features", {})
    video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]

    if not video_keys:
        warnings.append("No video features found — skipping video content validation")
        return warnings

    frame_pairs: list[tuple] = []
    merged_ep_map = _build_episode_frame_map(target)
    video_key = video_keys[0]
    merged_video = target / "videos" / video_key / "chunk-000" / "file-000.mp4"
    if not merged_video.exists():
        warnings.append(f"Merged video not found: {merged_video}")
        return warnings

    merged_ep_offset = 0
    for src in sources:
        src_video = src.path / "videos" / video_key / "chunk-000" / "file-000.mp4"
        if not src_video.exists():
            warnings.append(f"Source video not found: {src_video}")
            merged_ep_offset += src.episode_count
            continue

        src_ep_map = _build_source_episode_map(src)
        if not src_ep_map:
            warnings.append(f"No parquet data for {src.bin_id}")
            merged_ep_offset += src.episode_count
            continue

        sample_indices = [0, len(src_ep_map) // 2, len(src_ep_map) - 1]
        sample_indices = sorted(set(sample_indices))

        # Source bins may have per-episode video files (file-000.mp4, file-001.mp4, ...)
        src_video_dir = src.path / "videos" / video_key / "chunk-000"
        per_episode_videos = sorted(src_video_dir.glob("file-*.mp4")) if src_video_dir.exists() else []

        for si in sample_indices:
            src_ep = src_ep_map[si]
            merged_ep_idx = merged_ep_offset + si

            if merged_ep_idx >= len(merged_ep_map):
                warnings.append(f"Merged episode {merged_ep_idx} out of range")
                continue

            merged_ep = merged_ep_map[merged_ep_idx]
            mid_local = src_ep["count"] // 2
            merged_global_frame = merged_ep["global_start"] + mid_local

            try:
                m_frame = _extract_frame(merged_video, merged_global_frame)

                # Source: per-episode files → frame index is local to that file
                if len(per_episode_videos) > si:
                    s_frame = _extract_frame(per_episode_videos[si], mid_local)
                else:
                    s_frame = _extract_frame(src_video, src_ep["global_start"] + mid_local)

                match = _frames_match(m_frame, s_frame)
                label = f"{src.bin_id}_ep{src_ep['ep']:02d}_srcf{mid_local}_mergedf{merged_global_frame}"
                result = {
                    "source": src.bin_id,
                    "source_ep": src_ep["ep"],
                    "merged_ep": merged_ep["ep"],
                    "local_frame": mid_local,
                    "merged_global_frame": merged_global_frame,
                    "match": match,
                }
                results.append(result)
                frame_pairs.append((label, m_frame, s_frame, match))

                if match:
                    logger.info(
                        "Video PASS: %s ep%d mid-frame %d == merged ep%d frame %d",
                        src.bin_id, src_ep["ep"], mid_local,
                        merged_ep["ep"], merged_global_frame,
                    )
                else:
                    msg = (
                        f"Video FAIL: {src.bin_id} ep{src_ep['ep']} mid-frame "
                        f"{mid_local} != merged ep{merged_ep['ep']} frame "
                        f"{merged_global_frame}"
                    )
                    warnings.append(msg)
                    logger.warning(msg)
            except Exception as e:
                warnings.append(f"Video check error {src.bin_id} ep{src_ep['ep']}: {e}")

        merged_ep_offset += src.episode_count

    passed = sum(1 for r in results if r["match"])
    total = len(results)
    logger.info("Video validation: %d/%d checks passed", passed, total)

    manifest_path = target / "MANIFEST_fusion.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["validation"]["video_content"] = results
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    _generate_diff_heatmaps(target, frame_pairs)

    return warnings


def _generate_diff_heatmaps(target: Path, frame_pairs: list[tuple]) -> None:
    """Save side-by-side diff heatmaps (merged | source | jet diff) as PNG."""
    if not frame_pairs:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    proof_dir = target / "validation_proof"
    proof_dir.mkdir(exist_ok=True)

    for label, m_tensor, s_tensor, match in frame_pairs:
        m = m_tensor.permute(1, 2, 0).cpu().numpy().astype(np.float32)
        s = s_tensor.permute(1, 2, 0).cpu().numpy().astype(np.float32)

        diff = np.sqrt(np.mean((m - s) ** 2, axis=2))
        max_diff = float(diff.max())

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].imshow(m.astype(np.uint8))
        axes[0].set_title("Merged")
        axes[0].axis("off")

        axes[1].imshow(s.astype(np.uint8))
        axes[1].set_title("Source")
        axes[1].axis("off")

        im = axes[2].imshow(diff, cmap="jet", vmin=0, vmax=max(max_diff, 1))
        axes[2].set_title(f"Diff (max={max_diff:.1f})")
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        status = "IDENTICAL" if max_diff == 0 else f"MAX DIFF={max_diff:.1f}"
        plt.suptitle(f"{label} — {status}", fontsize=14)
        plt.tight_layout()
        plt.savefig(proof_dir / f"{label}_diff.png", dpi=100, bbox_inches="tight")
        plt.close()

    logger.info("Diff heatmaps saved to %s (%d images)", proof_dir, len(frame_pairs))


def validate_post_merge(
    target: Path,
    expected_episodes: int,
    sources: list[BinSource],
) -> list[str]:
    """Run post-merge checks including video content verification."""
    warnings = []
    for subdir in ("data", "meta", "videos"):
        if not (target / subdir).exists():
            raise ValueError(f"Missing {target / subdir} after merge")

    info = json.loads((target / "meta" / "info.json").read_text())
    actual_episodes = info.get("total_episodes", 0)
    if actual_episodes != expected_episodes:
        raise ValueError(
            f"Episode count mismatch: expected {expected_episodes}, got {actual_episodes}"
        )

    features = info.get("features", {})
    for key, feat in features.items():
        if "observation.images" in key:
            dtype = feat.get("dtype", "")
            if dtype not in ("video", "image"):
                warnings.append(f"Image feature '{key}' has dtype '{dtype}' (expected video/image)")
                logger.warning("R2 mitigation: %s has dtype %s", key, dtype)

    return warnings


def write_manifest(
    target: Path,
    config: FusionConfig,
    sources: list[BinSource],
    pre_warnings: list[str],
    post_warnings: list[str],
) -> None:
    """Write MANIFEST_fusion.json with full provenance."""
    total_episodes = sum(s.episode_count for s in sources)
    total_frames = sum(s.frame_count for s in sources)

    manifest = {
        "schema_version": "1.0",
        "pipeline_type": "fusion",
        "run_name": target.name,
        "tool_version": VERSION,
        "inputs": {
            "datasets_root": str(config.datasets_root),
            "selected_bins": [s.bin_id for s in sources],
            "excluded_bins": config.exclude,
            "source_bins": [
                {
                    "bin_id": s.bin_id,
                    "path": str(s.path),
                    "episode_count": s.episode_count,
                    "frame_count": s.frame_count,
                    "global_episode_offset": s.global_episode_offset,
                }
                for s in sources
            ],
        },
        "outputs": {
            "target_path": str(target),
            "total_episodes": total_episodes,
            "total_frames": total_frames,
        },
        "config": {
            "strict": config.strict,
            "dry_run": config.dry_run,
            "exclude_where": [
                {"key": k, "value": v} for k, v in config.exclude_where
            ],
        },
        "validation": {
            "pre_merge": {"warnings": pre_warnings},
            "post_merge": {"warnings": post_warnings},
        },
        "tools": {
            "package": "lerobot-so101-teleop-experiments",
            "tool_version": VERSION,
        },
    }
    manifest_path = target / "MANIFEST_fusion.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info("Manifest written to %s", manifest_path)


def fuse_bins(config: FusionConfig) -> None:
    """Main fusion logic."""
    if config.target.exists():
        raise ValueError(f"Target already exists: {config.target}")

    available = discover_bins(config.datasets_root)
    selected = resolve_selection(available, config.bins, config.exclude)
    logger.info("Selected %d bins: %s", len(selected), selected)

    sources: list[BinSource] = []
    repo_ids: list[str] = []
    roots: list[Path] = []
    offset = 0

    for bin_id in selected:
        bin_path = config.datasets_root / bin_id
        info_path = bin_path / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        ep_count = info.get("total_episodes", 0)
        frame_count = info.get("total_frames", 0)

        if config.exclude_where:
            binmap_path = bin_path / "binmap.json"
            if binmap_path.exists():
                success_eps = load_binmap_episodes(binmap_path)
                surviving = apply_episode_filters(success_eps, config.exclude_where)
                logger.info(
                    "Bin %s: %d/%d episodes survive filter",
                    bin_id, len(surviving), len(success_eps),
                )
                if len(surviving) < len(success_eps):
                    raise NotImplementedError(
                        f"--exclude-where would drop episodes in {bin_id} "
                        f"({len(surviving)}/{len(success_eps)} survive), but "
                        f"episode-level filtering requires temp-dataset creation "
                        f"(not yet implemented in v1.0). Use --exclude to drop "
                        f"entire bins instead."
                    )

        src = BinSource(
            bin_id=bin_id,
            path=bin_path,
            episode_count=ep_count,
            frame_count=frame_count,
            global_episode_offset=offset,
        )
        sources.append(src)
        repo_ids.append(bin_id)
        roots.append(bin_path)
        offset += ep_count

    total_episodes = sum(s.episode_count for s in sources)
    logger.info(
        "Fusion plan: %d bins, %d total episodes",
        len(sources), total_episodes,
    )

    pre_warnings = validate_pre_merge(sources, config.strict)

    if config.dry_run:
        logger.info("DRY RUN — validation passed, no output created")
        for src in sources:
            logger.info(
                "  %s: %d episodes, offset=%d",
                src.bin_id, src.episode_count, src.global_episode_offset,
            )
        return

    logger.info("Starting aggregate_datasets()...")
    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=config.target.name,
        roots=roots,
        aggr_root=config.target,
    )
    logger.info("aggregate_datasets() complete")

    post_warnings = validate_post_merge(config.target, total_episodes, sources)
    write_manifest(config.target, config, sources, pre_warnings, post_warnings)

    # Video content validation runs AFTER manifest write — it updates the manifest
    video_warnings = validate_video_content(config.target, sources)
    if video_warnings:
        for w in video_warnings:
            logger.warning(w)

    logger.info(
        "Fusion complete: %s (%d episodes from %d bins)",
        config.target, total_episodes, len(sources),
    )


def parse_exclude_where(values: list[str] | None) -> list[tuple[str, str]]:
    """Parse 'key=value' strings into (key, value) tuples."""
    if not values:
        return []
    result = []
    for v in values:
        if "=" not in v:
            raise ValueError(f"Invalid --exclude-where format: '{v}' (expected key=value)")
        key, val = v.split("=", 1)
        result.append((key.strip(), val.strip()))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge multiple single-bin LeRobot datasets into one.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--target", type=Path, required=True,
        help="Path for the merged output dataset (must not exist)",
    )
    parser.add_argument(
        "--bins", nargs="+", default=[],
        help="Explicit bin IDs to include (default: all bin_c*_r*)",
    )
    parser.add_argument(
        "--exclude", nargs="+", default=[],
        help="Bin IDs to exclude from selection",
    )
    parser.add_argument(
        "--exclude-where", nargs="+", dest="exclude_where_raw", default=[],
        help="Episode-level filter as key=value (e.g. yaw_rad=0.0)",
    )
    parser.add_argument(
        "--datasets_root", type=Path,
        default=Path("datasets/liftcube_binned"),
        help="Root directory containing bin_c*_r* dirs",
    )
    parser.add_argument("--strict", action="store_true", help="Warnings become errors")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run", help="Validate only")
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = FusionConfig(
        target=args.target.resolve(),
        datasets_root=args.datasets_root.resolve(),
        bins=args.bins,
        exclude=args.exclude,
        exclude_where=parse_exclude_where(args.exclude_where_raw),
        strict=args.strict,
        dry_run=args.dry_run,
    )

    logger.info("bin_fusion v%s", VERSION)
    logger.info("Target: %s", config.target)
    logger.info("Datasets root: %s", config.datasets_root)

    fuse_bins(config)


if __name__ == "__main__":
    main()
