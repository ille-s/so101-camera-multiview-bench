# Stage 2 — Assembly

Your recording captured one camera. The trainer needs all of them. This stage
replays the recorded action trajectory in the simulator and renders every
camera of a scene configuration in one pass, one dataset per bin, then
concatenates those bins into the corpus you train on.

```{raw} html
<video autoplay loop muted playsinline width="100%" poster="../_static/demo/stage2_assembly.jpg">
  <source src="../_static/demo/stage2_assembly.mp4" type="video/mp4">
</video>
```


Because all views come from the same replayed trajectory, policies trained on
different camera subsets see identical motions, cube poses and contacts. That
is what makes a comparison between camera sets a comparison of cameras.

```{mermaid}
flowchart LR
    REC[01_raw_gamepad/bin_c1_r1<br/>recording] -->|trajectories/*.npy<br/>+ scene_state| SMR[multicam_replay<br/>num_envs=1, N cameras]
    CFG[scene config JSON<br/>lift_cube_6cam.json] --> SMR
    SMR -->|one dataset per bin| BIN[(lift_cube_6cam_cylroom/bin_c1_r1/)]
    BIN --> FUS[bin_fusion]
    BIN2[(bin_c2_r1)] -.-> FUS
    BIN3[(bin_c3_r1)] -.-> FUS
    FUS -->|training-ready| OUT[(04_fused/merged_6cam_cylroom/)]
```

## How to use

### 1. Make sure the trajectories exist

This stage reads `<bin>/trajectories/*.npy`, not the recorded dataset. If that
directory is missing, run the extraction from
{doc}`stage 1, step 6 <01_recording>` first.

### 2. Choose the cameras

The camera set is a JSON file, not code. `lift_cube_6cam.json` defines six
views, and each external one is an azimuth, an elevation and a radius on a
hemisphere around the workspace:

```json
"front": { "type": "external", "azimuth_deg": 0, "elevation_deg": 45, "radius_m": 0.4 }
```

Adding a viewpoint means adding such an entry. Nothing else changes.

### 3. Render one bin

```bash
python -m so101_mvbench.recording.multicam_replay \
    --dataset_root datasets/01_raw_gamepad/bin_c1_r1 \
    --episodes all \
    --scene_config src/so101_mvbench/tasks/scene_configs/lift_cube_6cam.json \
    --output_base datasets \
    --mode create --headless
```

One simulator boot handles the whole bin. The output lands in
`datasets/<config_stem>_cylroom/<bin>/`, so the camera set is readable from
the path.

Start with `--episodes 0` to render a single episode and look at it before
committing an hour of GPU time. `--episodes` also takes comma lists and ranges
(`0,5-10`).

:::{note}
The replay deliberately runs one environment at a time. Rendering many
environments in parallel degrades the image quality of the tiled cameras. At
one environment, all cameras render at full quality in the same step.
:::

### 4. Repeat for every bin

```bash
for bin in datasets/01_raw_gamepad/bin_c*_r*; do
    python -m so101_mvbench.recording.multicam_replay \
        --dataset_root "$bin" --episodes all \
        --scene_config src/so101_mvbench/tasks/scene_configs/lift_cube_6cam.json \
        --output_base datasets --mode create --headless
done
```

Expect roughly 20 minutes per bin. Run this under a watchdog that kills a bin
whose log has stopped moving, see the deadlock below.

### 5. Fuse the bins into one dataset

```bash
export TMPDIR="datasets/04_fused/.tmp"
mkdir -p "$TMPDIR"   # nothing creates it, and tempfile does not either
python -m so101_mvbench.assembly.bin_fusion \
    --datasets_root datasets/lift_cube_6cam_cylroom \
    --target datasets/04_fused/merged_6cam_cylroom \
    --bins bin_c1_r1 bin_c2_r1 bin_c3_r1
```

`--bins` picks bins explicitly, `--exclude` drops them, `--exclude-where`
filters at episode level (`yaw_rad=0.0`), and `--dry-run` validates without
writing anything.

:::{danger}
**Set `TMPDIR` on the same partition as the target.** LeRobot's video
concatenation moves files with `shutil.move`. Across partitions that raises
`OSError: Invalid cross-device link`, and the fallback copy then trips over
read-only MP4s. This is the single most common failure of this step.
:::

### 6. Check the merge

`bin_fusion` pixel-validates the concatenation itself and leaves the evidence
in `validation_proof/`. Confirm the episode count matches your bins, and read
`MANIFEST_fusion.json` to see which bin contributed which episode range. That
file is what you consult later to hold out a bin.

## What you get

```bash
datasets/lift_cube_6cam_cylroom/bin_c1_r1/   # step 3, one dataset, six camera streams
├── meta/scene_config.json                   # the camera set, archived
├── videos/observation.images.{wrist,front,left,right,top,back}/
└── timing.json                              # how long each phase took

datasets/04_fused/merged_6cam_cylroom/       # step 5, trainer-ready
├── MANIFEST_fusion.json                     # bin to episode-range map
└── validation_proof/                        # pixel-diff evidence of the merge
```

:::{important}
**Everything in here is synthetic, including the wrist view.** Every frame was
rendered during replay. The only human part is the action trajectory. A visual state
that differed while you were recording but resets to scene defaults during
replay simply will not be in the training data. Audit the scene config before
you render twenty bins with it.
:::

:::{note}
**Decide the out-of-distribution split here, not while recording.** Record all
bins, then choose which ones enter the corpus via `--bins` and `--exclude`.
That way you can revisit the split without re-recording anything.
:::

## When something goes wrong

See {doc}`../troubleshooting`, which collects the failures of every stage in one place: a hung encoder, the cross-device TMPDIR error, duplicated episodes after an append.

## A note on older datasets

Earlier corpora were built by a four-stage chain that rendered one camera pose
per pass and merged afterwards. `multicam_replay` replaced it, and those tools
are not part of this repository. Both produce the same LeRobot v3 layout, so
nothing downstream can tell them apart.

## Where to go next

- {doc}`03_training` — train a policy on the fused dataset
- {doc}`../concepts/camera_configurations` — what the viewpoints look like
- {doc}`../concepts/data_layout` — what each directory contains
