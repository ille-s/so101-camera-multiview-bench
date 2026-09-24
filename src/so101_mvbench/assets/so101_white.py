# SPDX-License-Identifier: MIT
#
# SO-ARM101 Articulation Config — white variant, Multi-Env compatible.
#
# Sources:
#   - USD assets based on SO-ARM101-USD.usd from lerobot_so101_teleop (Lior Ben Horin)
#     https://github.com/liorbenhorin/lerobot_so101_teleop
#     Material color changed from yellow to white (diffuse_color_constant).
#
#   - Actuator parameters: copied from upstream SO101_CFG
#     lerobot_so101_teleop/assets/so101.py (Lior Ben Horin)
#
# The USD: SO-ARM101-USD-white-multienv.usd
#
# The upstream asset carries a `root_joint` (a PhysicsFixedJoint) that corrupts the
# articulation's mesh positions during sim.reset() as soon as num_envs > 1 -- the cube,
# table and markers stay visible while the arm alone vanishes to ~1e12 m. This USD has
# that joint removed; `fix_root_link=True` below makes Isaac Lab create the fixed joint
# at runtime instead, which is multi-env safe.
#
# A second variant (…-artapi.usd) solved the same bug the other way, by keeping the
# root_joint and moving PhysicsArticulationRootAPI onto it. It was removed on
# 2026-08-12 after both were spawned side by side and measured: identical root_pos_w,
# identical root_quat_w (yaw -180.000 deg), identical joint_pos, visually identical
# renders. Two files, one behaviour, 22.4 MB for the redundancy. If the multi-env bug
# ever resurfaces, that is the alternative fix -- it is in the git history, not here.
#
# XYZW Convention: isaaclab/assets/asset_base_cfg.py:37 — init_state.rot is (x,y,z,w)

__version__ = "3.0.0"

from pathlib import Path

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaacsim.core.utils.rotations import euler_angles_to_quat

_HERE = Path(__file__).parent


def _euler_to_quat_xyzw(euler_deg: list[float]) -> tuple[float, ...]:
    """Convert Euler XYZ degrees to XYZW quaternion for Isaac Lab configs.

    euler_angles_to_quat() returns WXYZ, but Isaac Lab expects XYZW
    since PR #4437 (commit 9659a5cefe7).
    """
    q = euler_angles_to_quat(np.array(euler_deg), degrees=True)
    return tuple(q[[1, 2, 3, 0]])

SO101_WHITE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_HERE / "usd" / "SO-ARM101-USD-white-multienv.usd"),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        rot=_euler_to_quat_xyzw([0, 0, 90]),  # +90° Yaw → robot faces +X
        joint_pos={
            "Rotation": -0.2736,
            "Pitch": -0.6109,
            "Elbow": -0.0745,
            "Wrist_Pitch": 1.5148,
            "Wrist_Roll": -1.6034,
            "Jaw": -0.1465,
        },
    ),
    actuators={
        # Source: lerobot_so101_teleop SO101_CFG — per-joint tuning for STS3215 servos
        "rotation": ImplicitActuatorCfg(
            joint_names_expr=["Rotation"],
            effort_limit_sim=30, stiffness=55, damping=0.7,
        ),
        "pitch": ImplicitActuatorCfg(
            joint_names_expr=["Pitch"],
            effort_limit_sim=30, stiffness=30, damping=0.8,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["Elbow"],
            effort_limit_sim=30, stiffness=25, damping=0.7,
        ),
        "wrist_pitch": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Pitch"],
            effort_limit_sim=30, stiffness=12, damping=0.5,
        ),
        "wrist_roll": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Roll"],
            effort_limit_sim=30, stiffness=7, damping=0.5,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["Jaw"],
            effort_limit_sim=30, stiffness=4, damping=0.3,
        ),
    },
)
