# SO-101 Multi-View Benchmark

**Which camera viewpoints does a manipulation policy actually need, how many
of them, and from where?**

That question is easy to ask and expensive to answer. This benchmark answers
it by construction: teleoperate the task once, re-render that single
demonstration from every viewpoint you care about, train one policy per camera
subset, and let an automated evaluation decide which subset wins.

```{raw} html
<video autoplay loop muted playsinline width="100%" poster="_static/demo/six_viewpoints.jpg">
  <source src="_static/demo/six_viewpoints.mp4" type="video/mp4">
</video>
```

## Why the question is hard on real hardware

Every new camera setup means repositioning and recalibrating the rig, then
recording a fresh set of demonstrations. Each configuration therefore rests on
different human executions, so the effect of the camera is tangled up with the
effect of the operator having a better or worse day. You end up comparing
recording sessions, not viewpoints.

Replaying one demonstration removes the confound. Identical motions, object
poses and contacts feed every camera configuration, and the camera is the only
thing left that differs.

## How it works, stage by stage

### 1. Record the demonstration once

A human drives the arm with a gamepad. The cube is placed on a deterministic
grid, so the demonstrations cover the workspace instead of clustering wherever
the operator felt comfortable.

```{raw} html
<video autoplay loop muted playsinline width="100%" poster="_static/demo/stage1_recording.jpg">
  <source src="_static/demo/stage1_recording.mp4" type="video/mp4">
</video>
```

Across a 5 by 4 grid that is 540 demonstrations covering 180 distinct cube
positions, each seen at three orientations. Seen from the wrist camera, whose
framing is identical in every episode, the coverage is the point:

```{raw} html
<video autoplay loop muted playsinline width="70%" poster="_static/demo/spawn_coverage.jpg">
  <source src="_static/demo/spawn_coverage.mp4" type="video/mp4">
</video>
```

### 2. Render it from every viewpoint

The recorded trajectory is replayed in the simulator, and all cameras of a
scene configuration render in the same pass. Viewpoints are given as an angle
and a distance, so adding one is an edit to a JSON file rather than a change
to the code.

Each viewpoint is a point on a hemisphere around the workspace: an azimuth, an
elevation, a radius. Blue are the positions the grid offers, red the ones this
configuration uses, green the point they all look at.

```{image} _static/hemisphere/hemisphere_scene.jpg
:alt: Camera positions on the hemisphere around the workspace
:width: 100%
```

```{raw} html
<video autoplay loop muted playsinline width="100%" poster="_static/demo/stage2_assembly.jpg">
  <source src="_static/demo/stage2_assembly.mp4" type="video/mp4">
</video>
```

### 3. Train one policy per camera subset

The dataset carries every viewpoint. Which of them a policy is allowed to see
is one flag. Two runs that differ only in that flag are a controlled
comparison.

```{raw} html
<video autoplay loop muted playsinline width="100%" poster="_static/demo/stage3_training.jpg">
  <source src="_static/demo/stage3_training.mp4" type="video/mp4">
</video>
```

### 4. Evaluate every policy under one protocol, automatically

This is the part that makes the question answerable at scale. Evaluation
restores the recorded starting conditions, runs ten copies of the scene at
once, and repeats each cell with a fixed seed so the spread across repeats
tells you the measurement's own noise. A campaign runner applies the identical
protocol to every policy, so a difference in success rate is a difference
between cameras.

```{raw} html
<video autoplay loop muted playsinline width="100%" poster="_static/demo/parallel_eval.jpg">
  <source src="_static/demo/parallel_eval.mp4" type="video/mp4">
</video>
```

### 5. Read the answer, not just the videos

Every run is turned into per-episode outcomes and plots without you touching a
spreadsheet. Left, the policy attempting the task and failing, then
succeeding. Right, the same bin counted up: how far each of its 27 episodes
got.

```{raw} html
<video autoplay loop muted playsinline width="100%" poster="_static/demo/eval_and_report.jpg">
  <source src="_static/demo/eval_and_report.mp4" type="video/mp4">
</video>
```

## Where a policy holds, and where it stops

A camera set that wins on the positions it was trained on has not proven much.
The workspace is split into bins, and which bins a policy trains on decides
what a result means.

Inside the training bins, the cube is placed *between* the positions the
policy saw. This measures precision within the distribution it knows.

```{image} _static/diagrams/bin_grid_id_interp_setup.png
:alt: In-distribution interpolation between trained cube placements
:width: 85%
```

Whole bins are withheld from training and evaluated afterwards. Success there
measures spatial generalisation rather than recall, and the withheld set is
spread across corners and interior so both extrapolation and gaps are covered.

```{image} _static/diagrams/bin_grid_ood_setup.png
:alt: Out-of-distribution test with withheld bins
:width: 85%
```

Beyond position, the same policy can be re-run under lighting and colour
randomisation, so a viewpoint that only works in perfect conditions shows
itself.

## What you get

- A complete pipeline from teleoperated recording to a written report, every
  step documented as a command you can paste.
- Deterministic replay that turns one demonstration into any set of
  viewpoints.
- Policy training on any subset of them, one flag apart.
- Automated evaluation: recorded starting conditions, ten parallel
  environments, same-seed repeats, one protocol for every policy.
- In-distribution and out-of-distribution regimes, plus randomisation
  variants.
- Reports that state the spread across repeats instead of a single flattering
  number.
- Two experiments beyond the standard comparison: degrading one camera at a
  time to find which one the policy leans on, and moving a camera off its
  trained pose to see what that costs.

## What awaits you

| | |
|---|---|
| {doc}`getting-started/requirements` | What you need. A GPU, and no robot. |
| {doc}`getting-started/installation` | Isaac Sim, LeRobot, this package. |
| {doc}`getting-started/first_steps` | A five-minute check that it runs. |
| {doc}`getting-started/running_the_pipeline` | The whole procedure end to end. |
| {doc}`pipeline/01_recording` | The five stages, each as a step-by-step recipe. |
| {doc}`concepts/coordinate_frames` | The ideas behind it: frames, viewpoints, test regimes, data layout. |
| {doc}`experiments/camera_importance` | Studies this pipeline supports beyond the standard run. |
| {doc}`troubleshooting` | When something goes wrong. |

## Scope

Simulation only, on NVIDIA Isaac Lab with the
[SO-ARM101](https://github.com/TheRobotStudio/SO-ARM100), a low-cost
open-source arm. Nothing here deploys to physical hardware, and no robot is
required to run it. The shipped task is picking up a cube, defined as an
ordinary Isaac Lab environment that you can replace.

```{toctree}
:caption: Getting Started
:maxdepth: 2
:hidden:

getting-started/what_is_this
getting-started/requirements
getting-started/installation
getting-started/first_steps
getting-started/running_the_pipeline
getting-started/cli_overview
```

```{toctree}
:caption: Pipeline
:maxdepth: 2
:hidden:

pipeline/01_recording
pipeline/02_assembly
pipeline/03_training
pipeline/04_evaluation
pipeline/05_reporting
```

```{toctree}
:caption: Experiments
:maxdepth: 1
:hidden:

experiments/camera_importance
experiments/camera_drift
experiments/beyond_rgb
```

```{toctree}
:caption: Concepts
:maxdepth: 1
:hidden:

concepts/architecture
concepts/coordinate_frames
concepts/camera_configurations
concepts/eval_variants
concepts/data_layout
```

```{toctree}
:caption: Reference
:maxdepth: 1
:hidden:

troubleshooting
api/index
```
