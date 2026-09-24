# Data layout

Every dataset in this pipeline is a
[LeRobot v3](https://github.com/huggingface/lerobot) dataset: parquet for
actions and proprioceptive state, MP4 per camera stream, and a `meta/`
directory describing both. What differs between the top-level directories
is *provenance*, and the naming encodes it.

## Top-level structure

```bash
datasets/
├── 01_raw_gamepad/<bin>/            # stage 1 — human teleoperation, the only non-synthetic data
├── lift_cube_6cam_cylroom/<bin>/    # stage 2b — per-bin multi-camera replay
└── 04_fused/<merged_name>/          # stage 2c — trainer-ready, several bins concatenated
```

The replay directory is named `<scene_config_stem>_cylroom` by
`multicam_replay`, so the camera set is readable from the path.

## Raw recording — `01_raw_gamepad/<bin>/`

```bash
datasets/01_raw_gamepad/bin_c1_r1/
├── meta/
│   ├── info.json                        # LeRobot v3 dataset metadata
│   ├── episodes/chunk-000/file-000.parquet
│   └── episode_000000_scene_state.json  # cube pose + workspace_origin_m, one per episode
├── data/chunk-000/file-000.parquet      # observations + actions
├── videos/observation.images.<camera>/chunk-000/
├── trajectories/episode_000.npy         # extracted action trajectories (stage 2 input)
└── binmap.json                          # bin progress + workspace_origin_m
```

This is the only directory a human produced. Everything else is derived
from `trajectories/` plus `meta/*_scene_state.json`.

## Per-bin replay — `lift_cube_6cam_cylroom/<bin>/`

```bash
datasets/lift_cube_6cam_cylroom/bin_c0_r0/
├── meta/
│   ├── info.json
│   └── scene_config.json               # archived camera set — guards --mode append
├── data/chunk-000/file-000.parquet
├── videos/observation.images.{wrist,front,left,right,top,back}/chunk-000/
└── timing.json                         # per-phase wall clock of the replay
```

One directory per bin, 27 episodes each in the reference setup.

## Fused training dataset — `04_fused/<merged_name>/`

```bash
datasets/04_fused/merged_6cam_cylroom/
├── meta/, data/, videos/
├── MANIFEST_fusion.json                # which bin contributed which episode range
└── validation_proof/                   # pixel-diff evidence from the merge
```

`MANIFEST_fusion.json` is the authoritative bin-to-episode map — the file
to read when translating "hold out bin `bin_c0_r0`" into episode indices, and
when checking whether an evaluated bin was part of training.

## Getting the reference corpus

The corpus used during development (`merged_6cam_cylroom`, 540 episodes, 20
bins × 27, six cameras) is not yet published.

:::{admonition} Hub upload pending
:class: warning

A Hugging Face Hub repository exists but currently holds an older,
538-episode fusion from a different scene — not the corpus described here.
Publishing `merged_6cam_cylroom` is an open task. The repository ID and the
revision tag will be filled in here once the upload exists. Until then, the download
below is a template.
:::

```bash
# adjust: both values come from the upload, which does not exist yet, so this
# block is a template. It runs as written and fails at the Hub, not in bash.
REPO_ID=owner/merged_6cam_cylroom
REVISION=v1.0.0

huggingface-cli download "$REPO_ID" \
    --repo-type dataset \
    --revision "$REVISION" \
    --local-dir datasets/04_fused/merged_6cam_cylroom
```

Always pull and cite a *pin* (`--revision` with a tag or commit), never the
bare repository name. Hub repos can be renamed, extended, or rewritten — a
reproduction would then silently run on different data.

## What you can rely on

- **Videos are capped at 1 MB per file**, set by
  `update_chunk_settings(video_files_size_in_mb=1)` after dataset creation,
  against LeRobot's default of 200 MB. A full-length episode passes that cap on
  its own, so each episode normally lands in its own MP4 and stays individually
  inspectable. Episodes short enough to fit together share a file.
- **Recorded data declares its frame.** `workspace_origin_m` in
  `binmap.json` and every `scene_state.json` names the scene the episodes
  were captured in. A dataset without the field is read as the retired
  floor-level table scene. Mixing frames in one batch is refused
  loudly. See {doc}`coordinate_frames`.
- **`scene_state.json` is mandatory for replay.** Cube positions vary per
  episode, so without scene-state restoration the gripper closes on empty air.
- **The training data is fully synthetic.** All camera streams, wrist
  included, were rendered during replay — only the action trajectory is
  human. See {doc}`../pipeline/02_assembly`.

## Inspecting a dataset

```bash
rerun datasets/04_fused/merged_6cam_cylroom/
```

Streams the episodes into the [Rerun](https://rerun.io/) viewer through its
native LeRobot v3 loader. This needs `rerun-sdk >= 0.31`, because older SDKs cannot read
v3 datasets.
