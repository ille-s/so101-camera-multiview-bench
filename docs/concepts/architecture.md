# Architecture

End-to-end view of the pipeline: how human-recorded trajectories flow through
replay, fusion, training, evaluation, and reporting.

```{image} ../_static/diagrams/pipeline_overview.png
:alt: The five pipeline stages with the artefact each one produces
:width: 100%
```

Every stage reads files and writes files, and nothing is passed in memory, so
each one is re-runnable in isolation. The chain is not a strict one-in-one-out
line: evaluation, for instance, takes the stage-3 checkpoint, a stage-1
recording bin for the cube placements, and a scene configuration. The dashed
loop is a practice rather than a mechanism: the determinism audits
(`audit.determinism`, `audit.cube_replay`) are run by hand, and episodes that
fail them are re-recorded before assembly. No script enforces that.

## Where the human is

Only in stage 1. The operator teleoperates the arm, and everything downstream
is deterministic replay, training, and measurement. This is what makes camera
configurations comparable: policies trained on different camera subsets share
one set of human demonstrations.

## What each stage produces

| Stage | Command | Artefact |
|-------|---------|----------|
| 1 Record | `recording.teleop_recorder` | LeRobot dataset per bin, plus the per-episode scene state and the bin map |
| 2 Assemble | `recording.trajectory_extractor` -> `recording.multicam_replay` -> `assembly.bin_fusion` | one multi-camera dataset per bin, fused into the training corpus |
| 3 Train | `training.train_act_camera_perm` | ACT checkpoints for one camera subset |
| 4 Evaluate | `evaluation.async_eval` | per-episode results, summary, videos |
| 5 Report | `evaluation.report`, `evaluation.aggregate_report` | plots and written summaries, single-run and cross-run |

## Frames and scene

Every stage in the diagram above runs inside the same environment, and it is a
deliberately plain one:

<!-- Source: thesis figure "The simulation scene shared by all experiments". -->

```{image} ../_static/diagrams/scene_annotated.png
:alt: The cylindrical room seen from outside, with the light source in the ceiling and the arm on a round table at the centre
:width: 55%
:align: center
```

A closed cylinder, a round table, and the arm on it. The lighting is a disc in
the ceiling for the visible pool of light plus a dome for the ambient fill.
There is no window and no textured wall, so a camera moved to a new viewpoint
sees the same room as the one it replaced, from a different angle. That is
what makes a viewpoint comparison a comparison of viewpoints.

Plain is not the same as fixed. Both light exposures are randomisable, and
that is one of the out-of-distribution axes rather than an accident of the
scene ({doc}`eval_variants`). What the policy is asked to generalise over is
chosen, and it lives in the scene configuration: where the cameras sit, and
where the cube spawns.

Every scene is described by one value: the position of the workspace frame
{W}. Cameras, cube, bin grid and arm base are anchored relative to it, which
is what lets a recording made in one scene replay in another. The contract is
on {doc}`coordinate_frames`.

## Where to go next

- {doc}`../pipeline/01_recording` — recording stage details
- {doc}`../pipeline/02_assembly` — replay and assembly details
- {doc}`coordinate_frames` — frames and rotation conventions
- {doc}`camera_configurations` — the viewpoints and trained subsets
- {doc}`data_layout` — what each dataset directory contains
