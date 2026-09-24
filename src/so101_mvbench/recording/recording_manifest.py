"""Recording manifest — documents provenance of recorded datasets.

Writes a JSON manifest alongside each recording that captures:
- Input files (trajectories, scene states)
- Recording settings (seed, determinism, neutral scene)
- Script version
- Timestamps

This enables traceability: given any _ego_base/ or _external_*/ directory,
the manifest tells you exactly how it was produced.
"""

__version__ = "1.2.0"

import json
import datetime
from pathlib import Path


def _collect_outputs(
    output_dir: Path,
    pose_subdirs: list[str] | None = None,
) -> dict:
    """Scan the output directory for generated files.

    Args:
        output_dir: Root output directory written by the recorder.
        pose_subdirs: Optional list of per-pose subdir names (e.g.
            ``["_external_az000_el45", ...]``). When provided, scanning is
            restricted to these subdirs — the manifest then documents ONLY
            the current run's outputs, not stale data from sibling
            archives or prior runs that also live under output_dir.
            When None, behaves as before (legacy single-dir callers like
            ``ego_recorder`` whose output_dir IS the dataset dir).

    Returns:
        Dict with videos, parquet files, total_frames, total_episodes
        read from info.json if available.
    """
    outputs = {"dataset_dir": output_dir.name}

    if pose_subdirs is None:
        scan_roots = [output_dir]
    else:
        scan_roots = [output_dir / s for s in pose_subdirs if (output_dir / s).exists()]

    # Videos
    videos = sorted(
        str(p.relative_to(output_dir))
        for root in scan_roots for p in root.rglob("*.mp4")
    )
    if videos:
        outputs["videos"] = videos

    # Parquet
    parquets = sorted(
        str(p.relative_to(output_dir))
        for root in scan_roots for p in (root / "data").rglob("*.parquet")
        if p.exists()
    )
    if parquets:
        outputs["parquet"] = parquets

    # info.json metadata — only valid for single-dir callers
    if pose_subdirs is None:
        info_path = output_dir / "meta" / "info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text())
            outputs["total_frames"] = info.get("total_frames")
            outputs["total_episodes"] = info.get("total_episodes")
            outputs["fps"] = info.get("fps")

    return outputs


def write_manifest(
    output_dir: Path,
    *,
    script_name: str,
    script_version: str,
    trajectory_dir: str,
    episode_indices: list[int] | None,
    scene_state_path: str | None,
    seed: int,
    camera_pose_ids: list[str] | None = None,
    camera_radius_m: float | None = None,
    num_envs: int = 1,
    pose_subdirs: list[str] | None = None,
    extra: dict | None = None,
) -> Path:
    """Write a recording_manifest.json to the output directory.

    Scans the output directory for generated files (videos, parquet)
    and includes them in the manifest for full traceability.

    Args:
        output_dir: Directory where the manifest is saved (e.g. _ego_base/).
        script_name: Name of the recording script.
        script_version: Script VERSION string.
        trajectory_dir: Path to the trajectory npy directory.
        episode_indices: List of episode indices replayed, or None for all.
        scene_state_path: Path to scene_state.json, or None.
        seed: Random seed used.
        camera_pose_ids: List of camera pose IDs (external only).
        num_envs: Number of parallel environments.
        pose_subdirs: Optional list of per-pose subdir names to restrict the
            file scan (e.g. ``["_external_az000_el45", ...]``). Required for
            batch recorders writing the manifest one level above the
            per-pose dataset dirs — without it the scan picks up stale
            archival data co-located under output_dir.
        extra: Additional key-value pairs to include.

    Returns:
        Path to the written manifest file.
    """
    manifest = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "script": script_name,
        "script_version": script_version,
        "inputs": {
            "trajectory_dir": str(trajectory_dir),
            "episode_indices": episode_indices,
            "scene_state": str(scene_state_path) if scene_state_path else None,
        },
        "settings": {
            "seed": seed,
            "enhanced_determinism": True,
            "num_envs": num_envs,
        },
        "outputs": _collect_outputs(output_dir, pose_subdirs=pose_subdirs),
    }

    if camera_pose_ids is not None:
        manifest["camera_pose_ids"] = camera_pose_ids
    if camera_radius_m is not None:
        manifest["settings"]["camera_radius_m"] = camera_radius_m

    if extra:
        manifest.update(extra)

    manifest_path = output_dir / "recording_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path
