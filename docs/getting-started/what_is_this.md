# Overview

A framework for **camera-viewpoint experiments in imitation learning** with the low-cost
[SO-ARM101](https://github.com/TheRobotStudio/SO-ARM100) manipulator in
[NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/).

The question it addresses is which viewpoints a manipulation policy requires. Studying
this on physical hardware is expensive: every camera configuration requires repositioning
and re-calibrating the rig, and re-recording the full set of demonstrations. Each
configuration therefore rests on different demonstrations, which confounds the viewpoint
effect with variation in the operator's execution.

This framework separates the two. The task is teleoperated once, and the recorded trajectory
is then re-rendered from an arbitrary set of viewpoints. Policies trained on different
camera subsets consequently share identical motions, object poses and physics, so the
camera configuration remains the only variable.

## Method

```{mermaid}
flowchart LR
    A[Gamepad teleop<br/>N episodes per bin] --> B[Trajectory<br/>extraction]
    B --> C[Multi-camera replay<br/>all views in one pass]
    C --> D[Multi-bin fusion<br/>one training dataset]
    D --> E[ACT training<br/>any camera subset]
    E --> F[Sim evaluation<br/>ID + OOD variants]
    F --> G[Reporting<br/>per-run + cross-run plots]
```

Each stage runs as a Python module (`python -m so101_mvbench.<module>`, see
{doc}`cli_overview`) and is documented on its own page, beginning with
{doc}`../pipeline/01_recording`.

Two properties establish the internal validity of the comparison:

- **Shared demonstrations.** All camera streams are rendered post-hoc from the recorded
  action trajectory. Policies trained on different camera subsets are therefore exposed
  to identical motions, object poses and contact dynamics.
- **Fixed evaluation protocol.** Episodes are initialised from deterministic scene
  states, repeated runs use a fixed seed, and a campaign runner applies identical
  settings across policies. Differences in success rate are consequently attributable to
  the camera configuration rather than to run conditions.

## Applications

- Comparison of camera subsets (wrist only, wrist plus external, multi-view) on one task.
- Generalisation studies: withholding cube-position bins, or evaluating under the
  provided out-of-distribution variants (position grid, lighting and colour
  randomisation).
- Ablation of any parameter exposed by the scene configuration: viewpoint angles, camera
  count, scene layout.
- Standalone use of individual stages. Recorder, fusion tool, evaluator and report
  generator operate independently of one another.

Outside the intended scope: deployment on physical hardware (the pipeline is
simulation-only), policy architectures other than ACT, and tasks other than the provided
`LiftCube`. The task is defined as a standard Isaac Lab environment configuration and can
be substituted.

## Requirements

A CUDA-capable GPU, Isaac Sim and no robot. The full list, with the versions
this was verified against, is in {doc}`requirements`.

## Scope of this documentation

This site documents the package `so101_mvbench`. The joystick interface it
uses (`gamepad_utils`) is vendored inside the package under
`so101_mvbench.vendor` — there is no separate dependency to install.

Proceed with {doc}`installation`, then {doc}`running_the_pipeline` for the complete
end-to-end procedure.

*Developed within an academic research project on viewpoint selection in imitation
learning.*
