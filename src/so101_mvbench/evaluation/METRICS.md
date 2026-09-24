# Evaluation metrics — LiftCube task

How an episode is scored, when it is cut short, and what the run-level quality gate
checks. This file is the contract; `eval_tracker.py` implements it.

## Task progress stages

An episode's score is the highest stage it ever reached. Stages only go up, so a
policy that lifts the cube and then drops it still scores `2_lift`.

| Stage | Score | Reached when | What it means |
|---|---|---|---|
| `0_miss` | 0.00 | the gripper never came within 12 cm of the cube | the policy missed entirely |
| `1_reach` | 0.33 | jaw-to-cube distance drops below 12 cm | it found the cube but did not grasp |
| `2_lift` | 0.66 | `cube_z` peaks more than 1 cm above where it started | the cube left the table |
| `3_home` | 1.00 | arm back at home AND still holding, for 0.5 s | success |

Distances are 3D, between the jaw body and the cube centre.

### What `3_home` requires exactly

Three conditions, all at once:

- **Arm is home**: `max(|joint_i − home_pose_i|) < 0.15 rad` (about 8.6°) across arm
  joints 0..4. The gripper is excluded, since it is holding something.
- **Cube still held**: `cube_z > initial_z − 1 cm`.
- **Sustained**: both hold for 30 consecutive steps (0.5 s at 60 Hz). A single frame
  passing through the home pose does not count.

`3_home` is gated on `2_lift`: an arm that returns home without ever having lifted
the cube scores `1_reach`, not success.

## Early termination

An episode is cut short when it can no longer produce a useful outcome. This saves
simulation time and keeps failed episodes from padding the timing statistics.

| Reason | Triggered when | Why |
|---|---|---|
| `no_approach` | after 10 s the gripper-to-cube distance has not dropped by at least 50 %, and the stage is still below `1_reach` | the SO-101 crosses its workspace in about 3 s, so 10 s is a threefold margin |
| `cube_lost` | `cube_z < initial_z − 2 cm` | the cube is below the table surface and cannot be recovered |
| `timeout` | the episode length is reached (`EPISODE_LENGTH_S × FPS`) | ordinary end of episode |

## Grasp attempts

An attempt starts when `cube_z` rises 2 cm above its initial height and ends when it
falls back below 1 cm. The two thresholds differ on purpose: a single threshold would
count one shaky grasp as dozens of attempts as the cube jitters across it.

Several attempts in one episode indicate an unstable grasp, even when the episode
eventually succeeds.

## Quality gate

`async_eval.py` writes `quality_gate.json` into the output directory. It describes the
health of the RUN, not the quality of the policy, and is what an automated check should
assert on.

| Key | Meaning | Expected |
|---|---|---|
| `num_episodes_completed` | episodes that finished | equal to the number requested |
| `nan_guard_triggered` | policy actions containing NaN or Inf | 0 |
| `startup_warnings` | non-fatal warnings raised during startup | 0 |
| `success_rate_pct` | overall success rate | recorded, not asserted -- this is the result, not a health signal |

A run that fails any of the first three is not evidence about the policy, whatever the
success rate says.

## Where this is implemented

- `eval_tracker.py` — stage detection, early termination, attempt counting
- `async_eval.py` — runs the episodes and writes `quality_gate.json`
- `report.py` — turns the per-episode CSV into plots and a written summary
