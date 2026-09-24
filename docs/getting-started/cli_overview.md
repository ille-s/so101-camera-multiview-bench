# Command overview

Every pipeline stage and tool runs as a Python module: `python -m
so101_mvbench.<module>`. There are no console-script shims, so what you see below
is the complete surface. Every command accepts `--help` without booting the
simulator, with one exception: `list_envs` has no argument parsing at all and
starts Isaac Sim on import, because listing the registered tasks is what it
does. Give it no flags.

Commands marked *sim* need Isaac Sim and a GPU. The others run anywhere the
package is installed.

## How to read the commands

Every command in this documentation is written to be pasted into a terminal
and run as it stands, from the repository root.

Where a command needs a value from you, the block sets it as a shell variable
on its first lines, marked with `# adjust:`. Change those lines, then paste
the rest unchanged:

```bash
# adjust: the bin you recorded
BIN=datasets/01_raw_gamepad/bin_c1_r1

python -m so101_mvbench.recording.trajectory_extractor \
    --dataset_root "$BIN" --output_dir "$BIN/trajectories"
```

Nothing else in a command needs editing. Anything adjustable is a variable at
the top of its block, so a block never contains a placeholder that would break
the shell if you pasted it unchanged.

Paths that appear literally, such as `datasets/01_raw_gamepad/bin_c1_r1`, are
the conventional locations this documentation uses throughout. They work as
they stand if you followed the previous stage.

## Pipeline

| Command | Stage | Sim | Purpose |
|---------|-------|-----|---------|
| `python -m so101_mvbench.recording.teleop_recorder` | 1 | yes | Record demonstrations (gamepad or scripted sine) into a LeRobot dataset with per-episode scene state and bin map. |
| `python -m so101_mvbench.recording.trajectory_extractor` | 2a | no | Extract per-episode joint trajectories to `.npy`, decoupling replay from the LeRobot import chain. |
| `python -m so101_mvbench.recording.multicam_replay` | 2b | yes | Re-render recorded episodes from every camera of a scene config, one pass, one dataset per bin. |
| `python -m so101_mvbench.assembly.bin_fusion` | 2c | no | Fuse per-bin datasets into one trainer-ready dataset. |
| `python -m so101_mvbench.training.train_act_camera_perm` | 3 | no | Train an ACT policy on a chosen camera subset (wraps `lerobot_train`). |
| `python -m so101_mvbench.training.checkpoint_selector` | 3b | no | Rank checkpoints of a finished run by evaluation loss. |
| `python -m so101_mvbench.evaluation.async_eval` | 4 | yes | Evaluate a policy over N parallel environments. With `--full_report` it also writes the files stage 5 reads. |
| `python -m so101_mvbench.evaluation.report` | 5 | no | Plots and a written summary for a single evaluation run. |
| `python -m so101_mvbench.evaluation.aggregate_report` | 5 | no | Cross-run statistical aggregation. |

## Audit tools

| Command | Sim | Question it answers |
|---------|-----|---------------------|
| `python -m so101_mvbench.audit.determinism` | yes | Does moving a camera change the physics? (If yes, every camera ablation is invalid.) |
| `python -m so101_mvbench.audit.cube_replay` | yes | Do episodes diverge across parallel environments? |
| `python -m so101_mvbench.audit.cube_plot` | no | Plot the divergence data produced by `cube_replay`. |

## Dataset tools

| Command | Sim | Purpose |
|---------|-----|---------|
| `python -m so101_mvbench.tools.list_envs` | yes | Print the registered `LiftCube-*` task IDs. |
| `python -m so101_mvbench.tools.validate_bin` | no | Integrity check for a recorded bin: parquet readability, MP4 presence, scene-state coverage. |
| `python -m so101_mvbench.tools.delete_episodes` | no | Remove episodes from a LeRobot dataset, renumbering safely. |
| `python -m so101_mvbench.tools.drift_configs` | no | Write scene configs with one camera coordinate shifted, for a drift sweep ({doc}`../experiments/camera_drift`). |

## Running a whole campaign

`scripts/run_campaign.sh` drives stage 4 across N policies x M bins x R
same-seed repeats, one Isaac boot per policy, resume-safe. Data locations come
from three required environment variables:

```bash
POLICY_ROOT=outputs \
DATASETS_ROOT=datasets/01_raw_gamepad \
OUT_ROOT=outputs/campaign \
bash scripts/run_campaign.sh
```

`DRY_RUN=1` prints the per-policy commands without booting Isaac. See the
header comment of the script for every tunable.
