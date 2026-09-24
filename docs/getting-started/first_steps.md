# First steps

A short smoke test after a fresh install. If everything here works, continue
with {doc}`running_the_pipeline`.

## 1. List registered environments

```bash
python -m so101_mvbench.tools.list_envs
```

This is the cheapest Isaac Sim boot. It prints the registered `LiftCube-*`
task IDs. If nothing is printed, the install is incomplete — see
{doc}`installation`.

## 2. Record one scripted episode

No gamepad needed — `--action_source sine` drives a scripted motion:

```bash
python -m so101_mvbench.recording.teleop_recorder \
    --task LiftCube-Sim --action_source sine --auto_record \
    --num_episodes 1 --episode_length_s 3 --seed 42 \
    --repo_id smoke/test --repo_root datasets/smoke_test \
    --task_name 'lift the cube' --headless
```

Drop `--headless` to watch it in the Isaac Sim window. With a gamepad,
use `--action_source gamepad` instead and drive the arm yourself
({doc}`../pipeline/01_recording`).

:::{warning}
After `RECORDING STOPPED` appears in the log, wait for the process to exit
on its own before intervening — `save_episode()` flushes frames on the Carb
main thread, and killing it mid-flush corrupts the parquet file.
:::

## 3. Inspect the recording

```bash
rerun datasets/smoke_test/
```

Opens the [Rerun viewer](https://rerun.io/) on the recorded episode: joint
trajectories and camera streams. This needs `rerun-sdk >= 0.31`, the first
version whose LeRobot loader reads v3 datasets. Expect to install it
separately, because the environment that runs the pipeline tends to carry an
older one (0.26 in the setup this was developed on, which cannot open a v3
dataset). Check with `rerun --version`.

## 4. Clean up

```bash
rm -rf datasets/smoke_test
```

## Where next

- {doc}`running_the_pipeline` — all five stages end to end.
- {doc}`cli_overview` — every command in one table.
