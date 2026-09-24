# Requirements

What you need before {doc}`installation`, and what this project deliberately
leaves to other people's documentation.

## Hardware

| | |
|---|---|
| GPU | CUDA-capable, and it has to be a GPU. Isaac Sim does not run on CPU alone. Developed on an RTX 5090 (sm_120, Blackwell) on driver 580. |
| Disk | A few GB for a multi-bin six-camera dataset, and a few GB per training run. Evaluation videos add up quickly, so plan for them. |
| Gamepad | An Xbox-compatible controller, needed only to record new demonstrations. Everything downstream runs without one. |
| Robot | **None.** The pipeline is simulation-only. A physical SO-101 is neither required nor used. |

## Software

The environment the pipeline was developed and run on, not minimums. Four of
these are attested by the evaluation logs committed under `log/`: Ubuntu
24.04.4, Python 3.12, Isaac Sim 6.0 and NVIDIA driver 580. The rest are the
versions of that machine, recorded here because a reproduction needs them, not
because each was separately tested.

| Component | Version | Note |
|---|---|---|
| OS | Ubuntu 24.04 LTS | Other distributions are untested rather than unsupported. |
| Python | 3.12 | The package requires 3.11 or newer, which rules out a stock Ubuntu 22.04. |
| Isaac Sim | 6.0.0 | From NVIDIA's package index, not PyPI. |
| Isaac Lab | 3.0.2 | Installed alongside Isaac Sim. The logged runs used a source checkout rather than the `pip install` on the install page. |
| LeRobot | 0.5.1 | Dataset format, ACT policy, training entry point. |
| PyTorch | 2.10 with CUDA 12.8 | `cu130` wheels crash Isaac Sim, so pin `cu128`. |
| torchcodec | 0.10.0 | Has to match the PyTorch ABI. |
| Rerun | 0.31 or newer, optional | Only for viewing datasets. The environment that runs the pipeline usually carries an older one, so install it separately. |

{doc}`installation` walks through the actual install, including the three
environment variables that keep the machine's global packages out of it.

## Robot calibration is out of scope

Calibrating a physical SO-101 is a solved, well documented problem, and this
project does not repeat it. You only need it if you go beyond what is
documented here, in one of two ways:

- you drive the simulation with a **physical leader arm** instead of a gamepad
- you deploy a trained policy onto a **physical follower arm**

Neither is part of this pipeline. If you do either, use one of these:

| Source | What it covers |
|---|---|
| [LeRobot SO-101 guide](https://huggingface.co/docs/lerobot/en/so101) | Assembly, motor IDs and baudrates, then `lerobot-calibrate` for follower and leader, with a video of the calibration motion. |
| [NVIDIA, Calibrating the SO-101](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/07-calibrating-so101.html) | The same calibration inside a full sim-to-real course: powering up, identifying USB ports, calibrating both arms, and a script that checks the result. |
| [liorbenhorin/lerobot_so101_teleop](https://github.com/liorbenhorin/lerobot_so101_teleop) | The Isaac Lab teleoperation environment this project builds on. Assumes an already calibrated leader arm and shows how to wire it to the simulation. MIT licensed. |

The gamepad path used here needs none of that. The controller maps to joint
targets directly, and the mapping lives in a JSON file you can edit
({doc}`../pipeline/01_recording`).

## Where to go next

- {doc}`installation` — set up the environment
- {doc}`first_steps` — check that it works
- {doc}`running_the_pipeline` — the whole procedure
