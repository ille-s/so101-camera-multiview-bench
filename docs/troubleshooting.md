# Troubleshooting

Everything that has gone wrong often enough to be worth writing down, grouped
by where it bites. Each entry names the symptom you actually see, then what is
happening, then what to do.

```{contents}
:local:
:depth: 2
```

## Environment

### Imports fail, or a package behaves like a different version

Something outside this environment is on the import path. The usual source is
ROS: sourcing a ROS setup script exports
`PYTHONPATH=/opt/ros/<distro>/lib/python3.X/site-packages`, and many setups do
that from `~/.bashrc` for every shell.

1. Check what you inherited:

   ```bash
   echo "$PYTHONPATH"
   ```

2. Clear it and keep the user-wide directory out as well, in the same shell
   that will run the command:

   ```bash
   unset PYTHONPATH
   export PYTHONNOUSERSITE=1
   ```

3. If the wrong version persists, confirm which file is actually imported:

   ```bash
   python -c "import so101_mvbench; print(so101_mvbench.__file__)"
   ```

### The simulator starts and then just sits there

It is waiting for you to accept the licence agreement, which a headless run
never shows.

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

### `Bus Error` when the simulator boots

PyTorch was installed from the `cu130` wheels, which ship unversioned NVIDIA
libraries that Isaac Sim cannot load. Reinstall pinned to `cu128`:

```bash
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128 \
    --force-reinstall --no-deps
```

### Video decoding fails inside LeRobot

`torchcodec` has to match the PyTorch ABI. Version 0.10.0 works with
torch 2.10 and CUDA 12.8. Reinstall it after any PyTorch change.

## Recording

### Nothing appears on the console

Isaac Lab's engine swallows `print()`. The package logs through
`so101_mvbench.logging` instead. Point it at a file:

```bash
export SO101_BENCH_LOG_FILE=record.log
```

### The simulator dies the moment you look at its output

You piped it. A pipe delivers `SIGPIPE` when the reader goes away, and Isaac
Sim does not survive that.

1. Redirect to a file instead:

   ```bash
   python -m so101_mvbench.recording.teleop_recorder --task LiftCube-Sim --action_source gamepad --num_episodes 27 --spawn_bins --spawn_bin_col 1 --spawn_bin_row 1 --repo_id local/bin_c1_r1 --repo_root datasets/01_raw_gamepad --task_name 'lift the cube' > record.log 2>&1
   ```

2. Watch the file, which is safe:

   ```bash
   tail -F record.log
   ```

### The parquet file will not open

The recording ended without a clean stop, so the writer never wrote its
footer. Confirm it:

```bash
# adjust: the bin whose parquet will not open
BIN=datasets/01_raw_gamepad/bin_c1_r1

python -c "import pyarrow.parquet as pq; pq.read_metadata('$BIN/data/chunk-000/file-000.parquet')"
# a broken file answers with: ArrowInvalid: Couldn't deserialize thrift
```

There is no reliable repair, so re-record that bin. To avoid it next time, let
the process finish on its own after `RECORDING STOPPED` appears. The recording
session already handles the exits it can intercept, which are a key press, a
`Ctrl+C` and an exception. A `SIGKILL` or an out-of-memory kill stays
unrecoverable.

### `validate_bin` reports a bin that does not exist

Bin IDs are row-major and start at 1, so `--bin_id 1` is column 0, row 0. For
a 5-column grid the ID of column `c`, row `r` is `r * 5 + c + 1`. The tool
prints the bin it resolved to, so read that line before the verdict.

### `validate_bin` finds fewer MP4s than it expects

It expects two videos per successful episode. Video files are capped at 1 MB,
and a full-length episode passes that cap on its own, so it normally gets its
own file. Very short test episodes fit together and share one, which is
harmless.

## Replay and fusion

### A bin hangs and ignores `SIGTERM`

The video encoder occasionally deadlocks at an episode transition. The process
stays busy and the log stops growing.

1. Confirm by the log's modification time, not by CPU load. A deadlocked
   encoder looks busy.
2. `SIGKILL` the process.
3. Re-run that bin. Bulk replays belong under a watchdog that kills on a
   stalled log.

### `OSError: Invalid cross-device link` during fusion

LeRobot's video concatenation moves files with `shutil.move`. Across
partitions that fails, and the fallback copy then trips over read-only MP4s.
Put the temporary directory on the same partition as the target:

```bash
export TMPDIR="datasets/04_fused/.tmp"
mkdir -p "$TMPDIR"   # nothing creates it, and tempfile does not either
```

### An episode appears twice after appending

`--mode append` with an index that is already present duplicates it, because
LeRobot assigns a fresh internal ID. Delete first, then re-render:

```bash
# adjust: the replay dataset that holds the bad episodes, and the recording it came from
REPLAY=datasets/lift_cube_6cam_cylroom/bin_c1_r1
BIN=datasets/01_raw_gamepad/bin_c1_r1
SCENE=src/so101_mvbench/tasks/scene_configs/lift_cube_6cam.json

python -m so101_mvbench.tools.delete_episodes \
    --dataset_root "$REPLAY" --delete 5,17

python -m so101_mvbench.recording.multicam_replay \
    --dataset_root "$BIN" --episodes 5,17 \
    --scene_config "$SCENE" --output_base datasets \
    --mode append --headless
```

Note which path goes where. You delete from the *replay* dataset and re-render
from the *recording*.

### Append refuses to run

The camera set no longer matches the archived `meta/scene_config.json`. That
is deliberate: appending a different camera set would write an inconsistent
dataset. Either use the original scene config or start a new dataset with
`--mode create`.

## Training

### The checkpoint selector crashes with a `TypeError`

A known defect in LeRobot 0.5.1. `ACTPolicy.forward()` dereferences the
`(None, None)` that the VAE encoder returns in eval mode. The selector works
around it by running the policy in train mode with gradients disabled and a
fixed seed, so the VAE stays alive, no weights change, and sampling stays
deterministic. If you hit the same crash in your own code, do the same.

### There is no `train.log`

Nothing writes one. The loss curve goes to standard output, so redirect it if
you want to keep it:

```bash
# adjust: the camera set, the fused dataset, and where the run goes
CAMERAS="wrist front"
DATASET=merged_6cam_cylroom
DATASET_ROOT=datasets/04_fused/merged_6cam_cylroom
OUT=outputs/2cam_wrist_front

# the log lives NEXT TO the run directory: lerobot_train refuses to start if
# --output_dir already exists, and the shell would create it by opening the
# redirect there before python is even reached
LOG="${OUT}.log"

python -m so101_mvbench.training.train_act_camera_perm \
    --cameras $CAMERAS \
    --dataset "$DATASET" \
    --dataset_root "$DATASET_ROOT" \
    --steps 65000 \
    --output_dir "$OUT" > "$LOG" 2>&1 &
```

### Two camera sets end up with near-identical final loss

Expected. Training loss does not rank camera configurations, task success
does, which is what {doc}`pipeline/04_evaluation` is for.

## Evaluation

### Every episode ends at `0_miss` and the gripper closes on nothing

The recorded starting conditions were not restored, so the cube is not where
the policy is reaching. `--dataset_root` has to point at the stage-1
**recording** directory, the one holding `binmap.json` and
`meta/*_scene_state.json`, not at the replay output.

### Success rate is lower than it should be, with no error anywhere

Inference and training disagree about temporal ensembling. Check the
checkpoint:

```bash
# adjust: the checkpoint you are about to evaluate
CKPT=outputs/2cam_wrist_front/checkpoints/065000/pretrained_model

grep temporal_ensemble_coeff "$CKPT/config.json"
```

The evaluator runs without ensembling by default. If the policy was trained
with it, add `--temporal_ensemble`.

### Stage 5 has nothing to read

The evaluation ran without `--full_report`, so `eval_summary.json`,
`eval_results_random_*.csv` and `quality_gate.json` were never written. They
cannot be reconstructed, so the run has to be repeated.

### A campaign was killed and resumed

Do not trust the run count. The runner resumes past a killed worker, so a
policy can stay half-evaluated, and a kill during a write can truncate a CSV,
a JSON or an MP4.

1. Parse every JSON and CSV in the campaign.
2. Count the runs per policy against what you expect.
3. `ffprobe` the videos around the point of the abort.

Complete is not the same as intact.

## Data

### Positions in an old dataset look shifted

The dataset predates the frame declaration and is being read in the wrong
scene. Recordings carry their scene's workspace origin in `binmap.json` and in
each `scene_state.json`. A dataset without that field is interpreted as the
retired floor-level table scene, which is usually right. See
{doc}`concepts/coordinate_frames`.

### There are more scene states than episodes

There is one extra, and that is normal. The recorder writes the state of the
upcoming episode at each reset, so the last one describes an episode that was
never recorded. Count episodes from `binmap.json`.

### The Rerun viewer cannot open a dataset

The LeRobot v3 loader arrived in `rerun-sdk` 0.31, and the environment that
runs the pipeline usually carries an older one:

```bash
rerun --version
```

Install the viewer separately if it is older.

## Still stuck

- {doc}`getting-started/requirements` — versions this was verified against
- {doc}`getting-started/installation` — the install, including the CUDA notes
- {doc}`pipeline/01_recording` — the stages, each with its own recipe
