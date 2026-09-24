# Stage 3 — Training

Train an [ACT](https://arxiv.org/abs/2304.13705) imitation-learning policy on
the fused dataset. The dataset carries every camera. Which of them this policy
sees is decided here, with `--cameras`. Run the same command twice with
different camera sets and you have two policies that differ in exactly one
thing.

```{raw} html
<video autoplay loop muted playsinline width="100%" poster="../_static/demo/stage3_training.jpg">
  <source src="../_static/demo/stage3_training.mp4" type="video/mp4">
</video>
```


```{mermaid}
flowchart LR
    DS[(Fused dataset<br/>04_fused/merged_6cam_cylroom)] --> T[train_act_camera_perm<br/>--cameras wrist front]
    HO[holdout bins] -->|excluded via --episodes| T
    T -->|every save_freq steps| CK[checkpoints/<br/>002500..065000/]
    T -->|loss curve| TL[train.log]
    CK --> EVAL([Stage 4 — Evaluation])
```

## How to use

### 1. Pick the cameras

Camera names must match the streams in your dataset and be keys of
`_CAMERA_SHAPES` in `training/train_act_camera_perm.py`. For a dataset built
with `lift_cube_6cam.json` those are `wrist`, `front`, `left`, `right`, `top`,
`back`. All are registered as 480 × 640 RGB. Recording at another resolution
means editing that table.

### 2. Decide what to hold out

Two questions need different holdouts:

| Question | Hold out | How |
|----------|----------|-----|
| Which checkpoint of this run is best? | a few episodes | list them in `<run_dir>/holdout_episodes.json` |
| How does the policy do at cube positions it never saw? | whole bins | exclude their episodes with `--episodes` |

Bins map to episode indices through `MANIFEST_fusion.json` in the fused
dataset. Look them up there and pass the complement. Write down which episodes
went in: the run directory name is not evidence.

### 3. Train

```bash
python -m so101_mvbench.training.train_act_camera_perm \
    --cameras wrist front \
    --dataset merged_6cam_cylroom \
    --dataset_root datasets/04_fused/merged_6cam_cylroom \
    --steps 65000 \
    --batch_size 8 \
    --output_dir outputs/2cam_wrist_front
```

The wrapper turns `--cameras` into LeRobot's `--policy.input_features` JSON and
builds a fixed `lerobot_train` command line from its own flags. It is not a
passthrough: a flag it does not define is an error, so reaching a LeRobot
option it does not expose means adding it here first.

:::{note}
`--dataset` on its own resolves under `datasets/liftcube_binned/<name>`, a
legacy default. For anything else pass `--dataset_root` explicitly.
:::

### 4. Watch the loss

The loss curve goes to standard output, not to a file. Redirect it if you
want to keep it, which is also what makes the run tailable:

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

tail -F "$LOG"
```

Expect a steep drop over the first few thousand steps, then a long flat
stretch. Add `--wandb` to mirror the metrics to Weights and Biases.

:::{warning}
The wrapper appends `--wandb.disable_artifact=true` for a reason. LeRobot's
default uploads *every* checkpoint, gigabytes per run, and stages the same
volume again in the local cache. Metrics belong in wandb. The checkpoints stay in
`outputs/<run>/checkpoints/`.
:::

### 5. Pick a checkpoint

For a single policy, let the selector rank checkpoints by loss on the
held-out episodes:

```bash
python -m so101_mvbench.training.checkpoint_selector \
    --run_dir outputs/2cam_wrist_front \
    --dataset_root datasets/04_fused/merged_6cam_cylroom
```

It reads `<run_dir>/holdout_episodes.json`, evaluates every checkpoint, and
writes `best_checkpoint.json` plus a loss plot.

:::{important}
**Do not do this when comparing camera sets.** Evaluate all policies at the
same fixed step instead. Otherwise a difference in success rate can just as
well come from two different training lengths as from the cameras, and you
cannot tell which.
:::

## What you get

```bash
outputs/2cam_wrist_front/
├── checkpoints/
│   ├── 002500/
│   │   └── pretrained_model/
│   │       ├── config.json
│   │       └── model.safetensors
│   ├── ...
│   └── 065000/
│       ├── pretrained_model/     # the weights, plus the pre/post-processors
│       └── training_state/       # optimizer state, for resuming
└── checkpoints/last -> 065000/   # symlink to the newest checkpoint
```

Every run writes only into its own `--output_dir`, so runs stay
self-contained. That matters when sweeping camera configurations: nothing
leaks between them.

## Settings worth knowing

| Flag | Default | What it does |
|------|---------|--------------|
| `--cameras` | — | Which camera streams this policy receives. |
| `--dataset` / `--dataset_root` | — | Dataset name and explicit path. |
| `--episodes` | all | Episode subset, how bins are held out. |
| `--steps` | `30000` | Training steps. |
| `--batch_size` | `8` | ACT default. |
| `--save_freq` / `--log_freq` | | How often to checkpoint and log. |
| `--seed` | | Training seed. |
| `--num_workers` | | Dataloader workers. |
| `--output_dir` | — | Where everything lands. |
| `--wandb` | off | Mirror metrics to Weights and Biases. |

## When something goes wrong

See {doc}`../troubleshooting`, which collects the failures of every stage in one place: the selector crashing in the policy, the missing `train.log`.

## Where to go next

- {doc}`04_evaluation` — run the checkpoint in the simulator
- {doc}`../concepts/camera_configurations` — the viewpoints and subsets
- ACT: Zhao et al., 2023 — <https://arxiv.org/abs/2304.13705>
