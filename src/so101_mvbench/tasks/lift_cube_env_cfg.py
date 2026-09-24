# SPDX-License-Identifier: MIT
#
# Based on: third_party/lerobot_so101_teleop/source/lerobot_so101_teleop/tasks/base/base_env_cfg.py
# Modified for the camera-permutation pipeline. Self-contained: the config classes are
# defined here rather than imported from upstream.
#
# WXYZ→XYZW Fix: Isaac Lab PR #4437 (Commit 9659a5cefe7, branch feature/isaacsim-6-0)
# changed OffsetCfg.rot convention from WXYZ to XYZW. euler_angles_to_quat() from
# isaacsim.core.utils.rotations still returns WXYZ — needs [[1,2,3,0]] swizzle.

"""LiftCube task — cube on a pedestal table in the cylindrical room.

Standalone env config with white SO-ARM101 robot, cylindrical room shell, action
pad, and a single rigid cube. No rewards or terminations — teleop/replay only.

Authored natively in the cylindrical room since the scene migration: the arm is
table-mounted at (-0.20, 0, 0.7463), the cube spawns at the workspace origin
(0, 0) on the tabletop, and the retired floor-level table scene survives only as
the frame legacy datasets are interpreted in (``HEMISPHERE_ORIGIN_W``).

All MDP functions use Isaac Lab native APIs or are inlined here.
No dependency on lerobot_so101_teleop.mdp.
"""

from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import JointPositionActionCfg
from isaaclab.envs.mdp import image, joint_pos, reset_joints_by_offset
from isaaclab.envs.mdp.terminations import time_out
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils
from isaacsim.core.utils.rotations import euler_angles_to_quat

# Our custom robot config (USDs originally from liorbenhorin/lerobot_so101_teleop, MIT license)
from so101_mvbench.assets import ASSETS_DIR
from so101_mvbench.assets.so101_white import SO101_WHITE_CFG
from so101_mvbench.utils.so101_transforms import BASE_SCENE_ORIGIN_W

_ASSETS_DIR = str(ASSETS_DIR)


# ---------------------------------------------------------------------------
# Quaternion helper — PR #4437 WXYZ→XYZW migration
# ---------------------------------------------------------------------------

def _euler_to_quat_xyzw(euler_deg: list[float]) -> tuple[float, ...]:
    """Convert Euler angles to XYZW quaternion for Isaac Lab OffsetCfg.rot.

    euler_angles_to_quat() returns WXYZ, but Isaac Lab's OffsetCfg.rot expects
    XYZW since PR #4437 (commit 9659a5cefe7).
    """
    q = euler_angles_to_quat(np.array(euler_deg), degrees=True)
    return tuple(q[[1, 2, 3, 0]])


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

@configclass
class LiftCubeSceneCfg(InteractiveSceneCfg):
    """Scene: white SO-ARM101 + room + action pad + cube + cameras."""

    env_spacing = 4.0
    num_envs = 1

    # --- Environment ---
    room = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Room",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{_ASSETS_DIR}/usd/cylindrical_room_shell.usda",
        ),
    )
    room_light = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Room/lights/DiskLight")
    env_light = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Room/lights/DomeLight")

    # Visual-only round pad on the tabletop, held <1 mm proud of the surface
    # (any proud collider would collapse the arm; the cube rests on the table
    # collider). Same asset and height the replay stage uses.
    action_pad = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ActionPad",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{_ASSETS_DIR}/usd/action-pad-round.usda",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.7681),
            rot=(0.0, 0.0, 0.0, 1.0),  # XYZW identity
        ),
    )

    # --- Robot (table-mounted; foot plate on the 0.77 tabletop) ---
    robot = SO101_WHITE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=SO101_WHITE_CFG.init_state.replace(pos=(-0.20, 0.0, 0.7463)),
    )

    # --- Cameras (Pinhole, XYZW quaternions) ---
    camera_ego = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/gripper_cam",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.4,
            horizontal_aperture=10.0,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.005, 0.06, -0.062),
            rot=_euler_to_quat_xyzw([-45, 0, 0]),
            convention="opengl",
        ),
    )

    # External camera — frontal-side view near table height
    # Position and angle from upstream Rock-A-Stack-Hard task (liorbenhorin)
    # which provides a better view of the grasp than the old top-down angle.
    camera_external = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/external_cam",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=5.0,
            horizontal_aperture=10.0,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.37, 0.0, 0.8013),
            rot=_euler_to_quat_xyzw([115, 0, 90]),
            convention="opengl",
        ),
    )

    # --- Object ---
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.2, 0.2),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.7963),
        ),
    )


# ---------------------------------------------------------------------------
# MDP: Actions
# ---------------------------------------------------------------------------

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_positions = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"],
        scale=1,
        use_default_offset=False,
    )


# ---------------------------------------------------------------------------
# MDP: Observations
# ---------------------------------------------------------------------------

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_obs = ObsTerm(func=joint_pos)

        def __post_init__(self) -> None:
            self.enable_corruption = False

    @configclass
    class VisualCfg(ObsGroup):
        camera_ego = ObsTerm(
            func=image,
            params={
                "sensor_cfg": SceneEntityCfg("camera_ego"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        camera_external = ObsTerm(
            func=image,
            params={
                "sensor_cfg": SceneEntityCfg("camera_external"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    visual: VisualCfg = VisualCfg()


# ---------------------------------------------------------------------------
# MDP: Events
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Randomization Groups (Colosseum-style, cf. pumacayCOLOSSEUMBenchmarkEvaluating2024)
#
#   VISUAL:   Lighting, sky dome, robot color — appearance only
#   SPATIAL:  Cube position, action pad orientation — scene layout
#   PHYSICAL: (reserved) Cube mass, friction, size — not yet implemented
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Event functions (inlined — no lerobot_so101_teleop dependency)
# ---------------------------------------------------------------------------

# The domain-randomisation term functions, their constants and configure_randomization
# live in `so101_mvbench.mdp`, following the NVIDIA mdp/-library layout.
# Re-imported here only so the EventCfg definitions below can reference them.
from so101_mvbench import mdp
from so101_mvbench.mdp import (  # re-exported for callers
    COLOR_PALETTE,
    RANDOMIZATION_DEFAULTS,
    RANDOMIZATION_GROUPS,
    configure_randomization,
)
from so101_mvbench.mdp.resets import (
    randomize_cube_color,
    randomize_light_exposure,
    randomize_robot_color,
    randomize_static_asset_yaw,
)


@configclass
class LiftCubeEventCfg:
    """Events: robot reset, randomization (visual + spatial), cube reset.

    Randomization follows the Colosseum benchmark grouping:
    - VISUAL: lighting exposure, sky dome orientation, robot color
    - SPATIAL: cube XY position, action pad yaw
    - PHYSICAL: (reserved for future: mass, friction, size)

    Use configure_randomization() to enable/disable groups or override ranges.
    """

    # --- Always active (not randomized) ---
    reset_robot_position = EventTerm(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"],
            ),
            "position_range": (0, 0),
            "velocity_range": (0, 0),
        },
    )

    # --- VISUAL group ---
    reset_robot_color = EventTerm(
        func=randomize_robot_color,
        mode="reset",
        params={"color_list": RANDOMIZATION_DEFAULTS["robot_color"]},
    )

    reset_cube_color = EventTerm(
        func=randomize_cube_color,
        mode="reset",
        params={
            "color_list": RANDOMIZATION_DEFAULTS["cube_color"],
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )
    reset_room_light_exposure = EventTerm(
        func=randomize_light_exposure,
        mode="reset",
        params={
            "exposure_range": RANDOMIZATION_DEFAULTS["room_light_exposure"],
            "asset_cfg": SceneEntityCfg("room_light"),
        },
    )

    reset_env_light_exposure = EventTerm(
        func=randomize_light_exposure,
        mode="reset",
        params={
            "exposure_range": RANDOMIZATION_DEFAULTS["env_light_exposure"],
            "asset_cfg": SceneEntityCfg("env_light"),
        },
    )

    reset_sky_dome_orientation = EventTerm(
        func=randomize_static_asset_yaw,
        mode="reset",
        params={
            "yaw_range": RANDOMIZATION_DEFAULTS["sky_dome_yaw"],
            "asset_cfg": SceneEntityCfg("env_light"),
        },
    )

    # --- SPATIAL group ---
    reset_action_pad_orientation = EventTerm(
        func=randomize_static_asset_yaw,
        mode="reset",
        params={
            "yaw_range": RANDOMIZATION_DEFAULTS["action_pad_yaw"],
            "asset_cfg": SceneEntityCfg("action_pad"),
        },
    )

    reset_cube = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": RANDOMIZATION_DEFAULTS["cube_x"],
                "y": RANDOMIZATION_DEFAULTS["cube_y"],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

@configclass
class LiftCubeEnvCfg(ManagerBasedRLEnvCfg):
    """Base environment for cube pick-and-lift with SO-ARM101.

    Phase-C eval variants: this class is the **ID** ("in-distribution") base.
    Subclasses adjust `cube_noise_m`, `cube_rot_noise_deg`, and
    `randomization_groups` to define OOD-light / OOD-visual variants;
    the rest of the env (scene, observations, actions, events) is shared.
    """

    scene: LiftCubeSceneCfg = LiftCubeSceneCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: LiftCubeEventCfg = LiftCubeEventCfg()

    @configclass
    class TerminationsCfg:
        """Terminations: time-out only (no success detection)."""
        time_out = DoneTerm(func=time_out, time_out=True)

    rewards = None
    terminations: TerminationsCfg = TerminationsCfg()

    # ---------------- Eval-variant knobs (read by eval_policy.py) ----------------
    # Per-episode cube perturbation, applied during setup_episode (NOT via
    # the EventManager). Defaults: ID, no perturbation.
    cube_noise_m: float = 0.0
    cube_rot_noise_deg: float = 0.0

    # EventManager DR groups — empty tuple = no Manager-level DR (ID).
    # Valid entries: "visual" (lighting + colors + sky_dome_yaw),
    # "spatial" (cube_xy + action_pad_yaw).
    # Threaded into configure_randomization() at env-build time.
    randomization_groups: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 5
        self.scene.num_envs = 1
        self.viewer.eye = (-0.25, -0.4, 0.22)
        self.viewer.lookat = BASE_SCENE_ORIGIN_W
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.render.rendering_mode = "quality"
        self.sim.render.enable_translucency = False


@configclass
class LiftCubeOodLightEnvCfg(LiftCubeEnvCfg):
    """OOD-light: cube position perturbed by ±2cm XY and ±10° yaw.

    No Manager-level visual or spatial DR — just per-episode cube perturbation
    on top of the recorded scene_state. Used to measure policy robustness
    against small spatial deviations of the manipulation target.
    """
    cube_noise_m: float = 0.02
    cube_rot_noise_deg: float = 10.0


@configclass
class LiftCubeOodVisualEnvCfg(LiftCubeEnvCfg):
    """OOD-visual: full visual DR (lighting, sky-dome yaw, robot/cube colors).

    Cube position stays at recorded scene_state (no spatial noise); only
    Manager-level "visual" group is enabled, randomizing per-episode at
    env.reset() via the EventCfg terms (configure_randomization).
    """
    randomization_groups: tuple[str, ...] = ("visual",)


@configclass
class LiftCubeOodPositionEnvCfg(LiftCubeEnvCfg):
    """OOD-position: deterministic grid positions, IID rotation + visual.

    Cube positions from an alternative binmap (midpoint interpolation +
    extrapolation beyond training bounds). No noise, no visual DR.
    Scene states wrap cyclically when the OOD grid has more episodes
    than recorded scene_states.
    """
    binmap_name: str = "binmap_ood.json"
    scene_state_cyclic: bool = True


# ---------------------------------------------------------------------------
# Multi-external camera variants — one subclass per camera count.
#
# Each variant adds N external cameras to the scene and corresponding
# ObsTerms. Positions are overridden at eval time by camera_override;
# defaults here are irrelevant but visually distinct for debugging.
# ---------------------------------------------------------------------------

def _ext_camera(prim_suffix: str) -> TiledCameraCfg:
    """Factory for additional external cameras (same optics as camera_external)."""
    return TiledCameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_suffix}",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=5.0,
            horizontal_aperture=10.0,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.37, 0.0, 0.8013),
            rot=_euler_to_quat_xyzw([115, 0, 90]),
            convention="opengl",
        ),
    )


def _ext_obs(scene_key: str) -> ObsTerm:
    """Factory for external camera ObsTerms."""
    return ObsTerm(
        func=image,
        params={
            "sensor_cfg": SceneEntityCfg(scene_key),
            "data_type": "rgb",
            "normalize": False,
        },
    )


@configclass
class LiftCube2ExtSceneCfg(LiftCubeSceneCfg):
    """Scene with 2 external cameras (for 2-external or wrist+2-external policies)."""
    camera_external_2 = _ext_camera("external_cam_2")


@configclass
class LiftCube2ExtEnvCfg(LiftCubeEnvCfg):
    """2 external cameras — use with ``--task LiftCube-Sim-2Ext``."""
    scene: LiftCube2ExtSceneCfg = LiftCube2ExtSceneCfg()

    @configclass
    class ObsCfg(ObservationsCfg):
        @configclass
        class VisualCfg(ObservationsCfg.VisualCfg):
            camera_external_2 = _ext_obs("camera_external_2")
        visual: VisualCfg = VisualCfg()
    observations: ObsCfg = ObsCfg()


@configclass
class LiftCube2ExtOodPositionEnvCfg(LiftCube2ExtEnvCfg):
    """2 external cameras + OOD-Position grid (deterministic 4×4 cube offsets)."""
    binmap_name: str = "binmap_ood.json"
    scene_state_cyclic: bool = True


@configclass
class LiftCube3ExtSceneCfg(LiftCube2ExtSceneCfg):
    """Scene with 3 external cameras."""
    camera_external_3 = _ext_camera("external_cam_3")


@configclass
class LiftCube3ExtEnvCfg(LiftCubeEnvCfg):
    """3 external cameras — use with ``--task LiftCube-Sim-3Ext``."""
    scene: LiftCube3ExtSceneCfg = LiftCube3ExtSceneCfg()

    @configclass
    class ObsCfg(ObservationsCfg):
        @configclass
        class VisualCfg(ObservationsCfg.VisualCfg):
            camera_external_2 = _ext_obs("camera_external_2")
            camera_external_3 = _ext_obs("camera_external_3")
        visual: VisualCfg = VisualCfg()
    observations: ObsCfg = ObsCfg()


@configclass
class LiftCube4ExtSceneCfg(LiftCube3ExtSceneCfg):
    """Scene with 4 external cameras."""
    camera_external_4 = _ext_camera("external_cam_4")


@configclass
class LiftCube4ExtEnvCfg(LiftCubeEnvCfg):
    """4 external cameras — use with ``--task LiftCube-Sim-4Ext``."""
    scene: LiftCube4ExtSceneCfg = LiftCube4ExtSceneCfg()

    @configclass
    class ObsCfg(ObservationsCfg):
        @configclass
        class VisualCfg(ObservationsCfg.VisualCfg):
            camera_external_2 = _ext_obs("camera_external_2")
            camera_external_3 = _ext_obs("camera_external_3")
            camera_external_4 = _ext_obs("camera_external_4")
        visual: VisualCfg = VisualCfg()
    observations: ObsCfg = ObsCfg()


@configclass
class LiftCube5ExtSceneCfg(LiftCube4ExtSceneCfg):
    """Scene with 5 external cameras (4-Ext + camera_external_5)."""
    camera_external_5 = _ext_camera("external_cam_5")


@configclass
class LiftCube5ExtEnvCfg(LiftCubeEnvCfg):
    """5 external cameras (wrist + 5 ext = 6-cam policy total).

    Used by ``LiftCube-Sim-6Cam`` for evaluating 6-camera policies trained on
    ``merged_6cam_phase2`` (cameras: wrist + back/front/left/right/top).
    """
    scene: LiftCube5ExtSceneCfg = LiftCube5ExtSceneCfg()

    @configclass
    class ObsCfg(ObservationsCfg):
        @configclass
        class VisualCfg(ObservationsCfg.VisualCfg):
            camera_external_2 = _ext_obs("camera_external_2")
            camera_external_3 = _ext_obs("camera_external_3")
            camera_external_4 = _ext_obs("camera_external_4")
            camera_external_5 = _ext_obs("camera_external_5")
        visual: VisualCfg = VisualCfg()
    observations: ObsCfg = ObsCfg()


@configclass
class LiftCube6ExtSceneCfg(LiftCube5ExtSceneCfg):
    """Scene with 6 external cameras (5-Ext + camera_external_6).

    Reserved for future variants that need 6 externals without wrist.
    """
    camera_external_6 = _ext_camera("external_cam_6")
