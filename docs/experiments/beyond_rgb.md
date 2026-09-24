# Depth instead of colour, and other things this cannot do yet

The two experiments in this chapter work because the pipeline already carries
what they need. This page is the other half of the picture: the study that
would fit here and does not run, and why.

## Depth versus colour

The natural next question after "which viewpoint" is "which modality". A
wrist-plus-top pair sees the cube from close range and from above, which is
exactly the geometry where depth should beat colour, and where the comparison
is cheap because both modalities can be rendered from the same demonstration.

```{image} ../_static/experiments/rgb_vs_depth.jpg
:alt: The wrist and top views as colour and as depth, the same frame
:width: 85%
```

The same frame as colour and as depth. The wrist view is where the difference
bites: in colour the cube is a red patch against a grey table, in depth it is
a raised block with an edge, and the gripper jaws stand out as the nearest
thing in the frame.

```{raw} html
<video autoplay loop muted playsinline width="100%" poster="../_static/experiments/exp_depth.jpg">
  <source src="../_static/experiments/exp_depth.mp4" type="video/mp4">
</video>
```

Recorded during the study this repository was cut from, in the research
workspace rather than here.

**This does not run today.** Every camera in the shipped scene is fixed to
colour:

```python
# tasks/lift_cube_env_cfg.py
data_types=["rgb"],
```

The same is true of the scene-configuration schema, which describes a
viewpoint's position, resolution and lens but has no notion of a channel. So a
depth study would need a change in four places, not one:

| Where | What is missing |
|---|---|
| `utils/scene_config.py` | A modality field on `CameraSpec`, so a configuration can ask for depth |
| `tasks/lift_cube_env_cfg.py` | `data_types` derived from that field instead of hard-coded |
| `recording/multicam_replay.py` | Writing a depth stream into the dataset, which is a different dtype and range than an 8-bit image |
| `training/train_act_camera_perm.py` | A shape entry for a single-channel input, since `_CAMERA_SHAPES` assumes three |

None of that is deep, but it is a real change through the whole chain, and it
touches the dataset format. It is not a flag.

:::{note}
Isaac Sim renders depth perfectly well: the camera type used here supports
`distance_to_image_plane` alongside `rgb`. The gap is in this pipeline, not in
the simulator.
:::

## What else is out of reach

| Question | Why it does not fit today |
|---|---|
| Does this transfer to a real arm? | The pipeline is simulation-only by design. Nothing here deploys, and no calibration path exists ({doc}`../getting-started/requirements`). |
| Does another policy architecture rank viewpoints differently? | Training wraps LeRobot's ACT specifically. A second architecture means a second trainer. |
| Does the ranking hold on another task? | The task is an ordinary Isaac Lab environment and can be replaced, but the tracker's success criterion is written for lift-and-return. A new task needs its own criterion. |

## Why say this at all

A framework that lists what it cannot do is easier to trust than one that
implies everything is possible. If you need depth, you now know it is a
four-file change rather than a missing flag, and you know where to start.

## Where to go next

- {doc}`camera_importance` — what does run, on the modality you have
- {doc}`../concepts/camera_configurations` — the schema a modality field would extend
