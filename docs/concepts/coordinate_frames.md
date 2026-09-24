# Coordinate frames

The scene is modelled as a pose graph: every component is positioned relative to
whichever frame is most natural for it, and scenes differ only in where the
graph's root sits in the world. This page is the contract. The constants themselves live
in `so101_mvbench.utils.so101_transforms`.

```{image} ../_static/diagrams/pose_graph.png
:alt: Pose graph of the task, following Corke's frame notation
:width: 100%
```

Read it as two chains that meet at the cube {P}. The arm chain runs world
{0} -> arm base {B} -> end effector {E} -> wrist camera {C_w}, given by the
scene and the robot's kinematics. The workspace chain runs {0} -> workspace
{W} -> external camera {C_E}, and the bin spawner places the cube relative to
{W}. Three edges carry the task: the two cameras *observe* the cube, and the
policy has to *learn* `{E} -> {P}`, the transform that closes the loop,
from those observations alone. `{W} -> {C_E}` is the experimental variable —
changing it is what a camera configuration does.

Moving the rig into another scene changes `{0} -> {W}` and `{0} -> {B}` by the
same rigid translation, so every edge inside the two chains is untouched. That
is why a recorded joint trajectory replays unchanged in a different scene.

## The workspace frame {W}

All task geometry is authored in one frame, the **workspace frame {W}**: the
bin grid is centred on it, external cameras orbit it on a hemisphere (azimuth,
elevation, radius), the cube spawns on it, and the arm base sits at a fixed
offset from it. A scene places this frame in the world by declaring one value:

```json
"workspace_origin_m": [0.0, 0.0, 0.8663]
```

in its scene config (`SceneConfig.workspace_origin_m`). That triple is the
{W} origin expressed in the scene's world frame — nothing else about the task
geometry changes between scenes.

Fixed {W}-relative anchors (identical in every scene):

| Anchor | Position in {W} |
|---|---|
| Bin grid centre | `(0, 0)` |
| Arm base | `(-0.20, 0, -0.12)` |
| Cube spawn height | `z = -0.07` (`CUBE_SPAWN_Z_IN_W`) |
| Camera hemisphere centre | `(0, 0, 0)` |

## The two origins that matter

| Constant | Value | Meaning |
|---|---|---|
| `HEMISPHERE_ORIGIN_W` | `(0.20, 0.0, 0.12)` | {W} origin of the retired floor-level table scene. Kept solely to interpret legacy datasets. |
| `BASE_SCENE_ORIGIN_W` | `(0.0, 0.0, 0.8663)` | {W} origin of the cylindrical room — the scene `LiftCubeSceneCfg` is authored in. |

Their difference, `(-0.20, 0.0, 0.7463)`, is the historical `ARM_BASE_OFFSET`:
the rigid translation that carried the whole floor-scene rig onto the pedestal
table.

## Recorded data declares its frame

Every recording writes its scene's {W} origin into both sidecars:

- `binmap.json` → top-level `workspace_origin_m`
- `meta/episode_*_scene_state.json` → `workspace_origin_m`

A dataset **without** the field predates the mechanism and by definition means
the floor-level table scene (`HEMISPHERE_ORIGIN_W`). That is why the released
corpus records the `bin_c0_r0` cube at world `x = 0.32` while a fresh cylroom
recording of the same bin records `x = 0.12`: both are `0.12` in {W}.

## Two shifts, not one

Replay and evaluation derive two distinct translations, and conflating them is
the documented failure mode (it once put the arm on the floor):

- **`RIG_SHIFT` = target origin − `BASE_SCENE_ORIGIN_W`.** Re-anchors the scene
  furniture (robot base, action pad, cube spawn, external camera). Zero for the
  shipped configs, because the base scene already is the cylindrical room.
- **`DATA_SHIFT` = target origin − source origin.** Transports recorded
  coordinates (binmap x/y, spawn height) from the scene the data was captured
  in. Zero when replaying a cylroom recording in the cylroom, and the full
  table-to-cylroom delta for a legacy dataset.

Because the whole rig moves rigidly, the arm-to-cube geometry is identical in
every scene and recorded joint trajectories replay unchanged — this is the
pose-graph property the entire cross-scene pipeline rests on. Verified
end-to-end: the cube lands at the predicted world position with
`err_xy = 0.0 m` for both legacy and native recordings
(`multicam_replay --verify_placement`).

## Conventions worth pinning

- **Robot forward = +X.** The SO-ARM101 USD mesh defaults to −Y, and a
  +90° yaw in `init_state.rot` turns it to +X.
- **Quaternion convention is XYZW since [Isaac Lab PR #4437].** Helpers
  in `so101_white.py` and `lift_cube_env_cfg.py` swizzle the
  `euler_angles_to_quat()` output (which is WXYZ) accordingly.
- **`XFormPrim.set_local_poses()` is WXYZ.** Different convention than
  `OffsetCfg.rot` and `init_state.rot` — easy to mix up.

[Isaac Lab PR #4437]: https://github.com/isaac-sim/IsaacLab/pull/4437
