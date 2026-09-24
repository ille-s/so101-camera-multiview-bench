# Stage 4 — Evaluation

Put the trained policy back into the simulator, replay the recorded starting
conditions, and measure how far it gets. The output is a per-episode record of
which stage of the task the policy reached.

```{raw} html
<video autoplay loop muted playsinline width="100%" poster="../_static/demo/stage4_evaluation.jpg">
  <source src="../_static/demo/stage4_evaluation.mp4" type="video/mp4">
</video>
```


```{mermaid}
flowchart LR
    CK[checkpoints/065000/<br/>pretrained_model] --> EVAL[async_eval]
    HOLD[recording bin<br/>scene states + binmap] --> EVAL
    CFG[scene config JSON] --> EVAL
    EVAL -->|per episode| TRK[eval_tracker<br/>4 stages]
    EVAL -->|MP4 per camera| VID[videos/]
    TRK --> CSV[results CSV]
    TRK --> JSN[eval_summary.json]
```

## How to use

### 1. Check how the policy was trained

ACT can smooth its output by averaging overlapping action chunks (temporal
ensembling). Inference has to match training. Look at the checkpoint's
`config.json`:

```bash
# adjust: the checkpoint you are about to evaluate
CKPT=outputs/2cam_wrist_front/checkpoints/065000/pretrained_model

grep temporal_ensemble_coeff "$CKPT/config.json"
```

The evaluator runs **without** ensembling by default. If the policy was
trained with it, add `--temporal_ensemble`. Getting this wrong costs success
rate silently, with no error anywhere.

### 2. Point it at the recording, not the replay

`--dataset_root` is the stage-1 bin directory, the one with `binmap.json` and
`meta/*_scene_state.json`. The evaluator restores each episode's recorded cube
pose from there. Pointing it at the replay output is the most common mistake
on this page, and it fails in a way that looks like a broken policy.

### 3. Run one evaluation

```bash
python -m so101_mvbench.evaluation.async_eval \
    --mode async --num_envs 10 --num_episodes 27 \
    --policy_path outputs/2cam_wrist_front/checkpoints/065000/pretrained_model \
    --scene_config src/so101_mvbench/tasks/scene_configs/lift_cube_6cam.json \
    --dataset_root datasets/01_raw_gamepad/bin_c0_r0 \
    --output_dir outputs/2cam_wrist_front/eval_cylroom_async_bin_c0_r0/run01 \
    --seed 42 --settle_steps 130 --full_report --headless
```

**Do not omit `--full_report`.** It writes the files stage 5 reads. Without
it you get only a compact CSV and a timing file, and the reporting step has
nothing to work with.

```{raw} html
<video autoplay loop muted playsinline width="100%" poster="../_static/demo/parallel_eval.jpg">
  <source src="../_static/demo/parallel_eval.mp4" type="video/mp4">
</video>
```

`--num_envs` copies of the scene run side by side and step independently: each
copy starts its next episode as soon as its own is done, instead of waiting
for the slowest one in a batch. The copies are separate scene instances, so
episodes stay independent.

To evaluate under a domain-randomisation variant, pass e.g.
`--task LiftCube-Sim-OOD-Light` ({doc}`../concepts/eval_variants`).

### 4. For comparisons, use the campaign runner

One evaluation answers "does this policy work". Comparing camera sets needs
several bins, several repeats, and above all the *same protocol* for every
policy:

```bash
POLICY_ROOT=outputs DATASETS_ROOT=datasets/01_raw_gamepad OUT_ROOT=outputs/campaign \
POLICIES="2cam_wrist_front 2cam_wrist_left" \
BINS="bin_c0_r0 bin_c2_r2" \
bash scripts/run_campaign.sh
```

Defaults: checkpoint `065000`, 27 episodes, seed 42, 10 repeats, ten
environments, `--settle_steps 130 --record_all_envs --verify_placement`. It
skips any run whose `timing_async.json` already exists, so an interrupted
campaign resumes. `DRY_RUN=1` prints the commands without booting anything.

:::{important}
Resist the urge to hand-write the comparison commands. The runner exists so
that protocol parity is structural rather than remembered. If you need a
variant, override the environment variables.
:::

### 5. Understand two protocol choices

**Repeats share one seed.** The spread across `run01..run10` measures
floating-point non-determinism in the physics and on the GPU, which is the
noise floor of the measurement. Cube positions come from the recorded scene
states, not from the seed, so varying the seed would not vary the task, it
would only blur what the spread means.

**`--settle_steps 130` is not padding.** The renderer needs those steps to
clear the ghost frame after the cube is teleported into place. They apply to
the policy's first observation and to the recorded video alike, so the video
stays an honest record of what the policy saw.

### 6. Read the outcome

The tracker records how far each episode got:

| Stage | Reached when |
|-------|--------------|
| `0_miss` | the gripper never got to the cube |
| `1_reach` | the gripper came within a fixed distance of the cube |
| `2_lift` | the cube rose above its starting height, it left the table |
| `3_home` | the arm returned to the home pose *with the cube still held* |

Success means `3_home`: completing the cycle that was demonstrated, not merely
lifting. Episodes that stall are cut short after 600 steps without progress
(10 s at 60 Hz) instead of burning the full budget. The home pose and the
plateau budget come from `evaluation/config/eval.json`. The stage thresholds
themselves (12 cm to count as reaching, 1 cm of lift, 8.6° per joint for
"home") are constants at the top of `evaluation/eval_tracker.py` and are not
configurable.

Then confirm against the videos. A success rate that the recordings do not
support is a bug, not a result.

## What you get

```bash
outputs/2cam_wrist_front/eval_cylroom_async_bin_c0_r0/run01/
├── results_async.csv               # per-episode stage results
├── timing_async.json               # wall clock, also the resume marker
├── eval_summary.json               # with --full_report
├── eval_results_random_none.csv    # with --full_report
├── quality_gate.json               # with --full_report
├── joint_states/                   # per-episode joint CSVs, on by default
└── videos/env<N>/episode_<NNN>/    # one MP4 per camera per recorded episode
```

The videos show what the policy saw rather than a bystander view: the recorder
is handed the policy's own camera list, and it captures after any degradation
has been applied. They are not a pixel-exact record, though. The policy reads
480 by 640 at every control step. The video is 320 by 240 at 10 fps and keeps
every fifteenth step (`--video_interval`). Use them to see what happened, and
the frame dumps from `--capture_frames` when the pixels themselves matter. The file names are
the environment's camera slots (`ego.mp4`, `external.mp4`), which the policy
sees under its own names, so `ego.mp4` is the stream a wrist-camera policy
consumes as `wrist`. They are large and reproducible, so keep them out of
version control.

## Useful extra flags

| Flag | What it does |
|------|--------------|
| `--verify_placement <csv>` | Writes measured against expected cube world position per episode. The ground-truth check that frames are handled correctly. |
| `--record_all_envs` | Record video from every environment, not just the first. |
| `--no-record_video` | Skip video entirely. |
| `--video_interval N` | Record every Nth frame. |
| `--no-dump_joint_states` | Skip the per-episode joint CSVs under `joint_states/`. They are written by default. |
| `--gui` | Show the simulator window instead of running headless. |

## When something goes wrong

See {doc}`../troubleshooting`, which collects the failures of every stage in one place: every episode ending at `0_miss`, a killed campaign, stage 5 finding nothing to read.

## Where to go next

- {doc}`05_reporting` — turn these files into plots and numbers
- {doc}`../concepts/eval_variants` — the randomisation variants
- `evaluation/METRICS.md` in the repository — what every CSV column means
