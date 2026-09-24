# Running the pipeline end to end

The complete procedure across all five stages, from demonstrations to an
evaluation report. Each step links to the page documenting it in detail. For
a minimal install check, see {doc}`first_steps`.

```{contents}
:local:
:depth: 1
```

## Before you start

Hardware, software versions and the calibration references are in
{doc}`requirements`. Two things matter for the run itself: the simulator
stages have to run one at a time, because concurrent Isaac Sim processes
exhaust GPU memory and an aborted one can hold its allocation until it is
killed, and a full recording grid is hours of manual work.

## 1. Environment

```bash
cd so101-camera-multiview-bench
source .venv/bin/activate
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export OMNI_KIT_ACCEPT_EULA=YES
```

Those three lines belong in every shell that runs a simulator command. They
keep the machine's globally installed Python packages out of this environment.
`PYTHONPATH` is usually set by a sourced ROS setup and would pull ROS packages
in, `PYTHONNOUSERSITE` stops packages under `~/.local` from shadowing the ones
you installed here, and `OMNI_KIT_ACCEPT_EULA` answers the licence prompt that
would otherwise stall a headless run.
{doc}`installation` explains each one, and covers installing Isaac Sim and
PyTorch cu128.

Verify the environment with:

```bash
python -m so101_mvbench.tools.list_envs   # prints the LiftCube-* task IDs
```

## 2. Get demonstrations

**Record new demonstrations** (the path that produces new data):

```bash
python -m so101_mvbench.recording.teleop_recorder \
    --task LiftCube-Sim --action_source gamepad \
    --num_episodes 27 \
    --spawn_bins --spawn_bin_col 1 --spawn_bin_row 1 \
    --repo_id local/bin_c1_r1 --repo_root datasets/01_raw_gamepad \
    --task_name 'lift the cube' --seed 42
```

Guide the arm to the cube, lift it, and press **Y** to return to the home
pose. Repeat per bin (the recorder appends `bin_c<col>_r<row>` to the root
itself). Controls and constraints: {doc}`../pipeline/01_recording`.

**Or use an existing dataset.** Any LeRobot v3 dataset with the camera
streams and `meta/*_scene_state.json` files works from stage 3 onward. The
reference corpus and the status of its Hub upload are described in
{doc}`../concepts/data_layout`.

Check a recorded bin before spending GPU time on it:

```bash
# adjust: BIN_ID counts row-major from 1, so column 1 / row 1 on a 5-column
# grid is 1*5 + 1 + 1 = 7. The tool prints which bin it resolved to.
BIN_ID=7

python -m so101_mvbench.tools.validate_bin \
    --bin_id "$BIN_ID" --dataset_root datasets/01_raw_gamepad/
```

## 3. Render the camera views and fuse

```bash
# Extract per-episode trajectories (CPU-only)
for bin in datasets/01_raw_gamepad/bin_c*_r*; do
    python -m so101_mvbench.recording.trajectory_extractor \
        --dataset_root "$bin" --output_dir "$bin/trajectories"
done

# Render all cameras of each bin in one replay pass
for bin in datasets/01_raw_gamepad/bin_c*_r*; do
    python -m so101_mvbench.recording.multicam_replay \
        --dataset_root "$bin" --episodes all \
        --scene_config src/so101_mvbench/tasks/scene_configs/lift_cube_6cam.json \
        --output_base datasets --mode create --headless
done

# Concatenate the per-bin datasets into one training dataset
export TMPDIR="datasets/04_fused/.tmp"
mkdir -p "$TMPDIR"   # nothing creates it, and tempfile does not either
python -m so101_mvbench.assembly.bin_fusion \
    --datasets_root datasets/lift_cube_6cam_cylroom \
    --target datasets/04_fused/my_corpus \
    --bins bin_c1_r1 bin_c2_r1
```

The camera set is defined in the scene-config JSON — changing it is an edit
there, not a code change. Append mode, the TMPDIR pitfall, and a known
encoder deadlock: {doc}`../pipeline/02_assembly`.

## 4. Train a policy

```bash
python -m so101_mvbench.training.train_act_camera_perm \
    --cameras wrist front \
    --dataset my_corpus \
    --dataset_root datasets/04_fused/my_corpus \
    --steps 65000 \
    --output_dir outputs/2cam_wrist_front
```

`--cameras` selects this policy's subset from the streams the dataset
provides — the variable a viewpoint comparison manipulates. Holdout bins
are excluded via `--episodes`. Details: {doc}`../pipeline/03_training`.

## 5. Evaluate

One policy, one bin, in-distribution:

```bash
python -m so101_mvbench.evaluation.async_eval \
    --mode async --num_envs 10 --num_episodes 27 \
    --policy_path outputs/2cam_wrist_front/checkpoints/065000/pretrained_model \
    --scene_config src/so101_mvbench/tasks/scene_configs/lift_cube_6cam.json \
    --dataset_root datasets/01_raw_gamepad/bin_c1_r1 \
    --output_dir outputs/2cam_wrist_front/eval_cylroom_async_bin_c1_r1/run01 \
    --seed 42 --settle_steps 130 --full_report --headless
```

For a *comparison* between camera configurations, use the campaign
orchestrator instead of repeating the command by hand — it applies one
protocol to every policy and is resume-safe:

```bash
POLICY_ROOT=outputs DATASETS_ROOT=datasets/01_raw_gamepad OUT_ROOT=outputs/campaign \
POLICIES="2cam_wrist_front 2cam_wrist_left" \
bash scripts/run_campaign.sh
```

`DRY_RUN=1` prints the exact per-policy commands without booting Isaac.
Protocol rationale, OOD variants, and what to check after a killed run:
{doc}`../pipeline/04_evaluation`.

:::{warning}
Redirect Isaac Sim output to a file — never pipe it through `tail` or
`grep`. `SIGPIPE` kills the simulator instantly. `tail -F` on the log file
afterwards is fine.
:::

## 6. Report

```bash
python -m so101_mvbench.evaluation.report \
    --dir outputs/2cam_wrist_front/eval_cylroom_async_bin_c1_r1/run01

python -m so101_mvbench.evaluation.aggregate_report \
    --run_dirs outputs/2cam_wrist_front/eval_cylroom_async_bin_c1_r1/run* \
    --output_dir outputs/2cam_wrist_front/eval_cylroom_async_bin_c1_r1/aggregate
```

CPU-only and repeatable. The aggregate gives mean ± std across the repeats.
That spread measures floating-point non-determinism rather than seed
variation, so report both together. Available plots: {doc}`../pipeline/05_reporting`.

## Expected results

| Stage | Artefacts | Runtime | Completed correctly when |
|-------|-----------|---------|--------------------------|
| 1 Recording | `01_raw_gamepad/<bin>/` — parquet, MP4s, a `scene_state.json` per episode plus one look-ahead, `binmap.json` | manual, 1–2 min per episode | `validate_bin` passes for that bin's ID |
| 2 Replay + fusion | `lift_cube_6cam_cylroom/<bin>/` with one MP4 stream per camera, then `04_fused/<name>/` with `MANIFEST_fusion.json` and `validation_proof/` | about 20 min per bin, fusion minutes on CPU | The episode count matches the source and the pixel-diff evidence in `validation_proof/` shows no deviation |
| 3 Training | `checkpoints/<step>/pretrained_model/` and `training_state/` | under 2 h for 65 k steps, two cameras | Loss drops steeply and then flattens, and checkpoints appear at the save frequency |
| 4 Evaluation | `results_async.csv` and `timing_async.json`, plus `eval_summary.json`, the per-episode CSV, `quality_gate.json` and videos when `--full_report` is set | minutes per run, hours for a campaign | Episodes reach `2_lift` or `3_home` rather than staying at `0_miss`, and the videos show the gripper closing on the cube |
| 5 Reporting | `report/` with plots and `report.md`, plus `aggregate/` across repeats | seconds | The reported success rate matches what the videos show, and the cross-run spread stays within a few percentage points |

Two frequent failure modes:

- **Every episode ends `0_miss`, the gripper closes on empty space.** The
  scene state was not restored — check that `--dataset_root` points at the
  *recording* bin (the directory with `binmap.json` and `meta/`).
- **Training loss decreases but success stays near zero.** Usually a camera
  mismatch: the policy was trained on streams the evaluation scene config
  does not provide. Compare `--cameras` against the scene-config camera
  names.

Success rates are interpretable only with their conditions. Report episode
count, evaluated bins, checkpoint step, and number of repeats alongside
every number.
