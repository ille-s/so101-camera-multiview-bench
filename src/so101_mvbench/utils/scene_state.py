# SPDX-License-Identifier: MIT

"""Scene State — capture, save, and load exact scene conditions per episode.

Stores all randomizable parameters so that trajectory replays can
reproduce identical scenes across camera permutations. Uses self-documenting
dict format with named keys (not positional arrays).

Conventions:
    - Orientation: Euler RPY (rad), not quaternions
    - Unit suffixes: _m for meters, _rad for radians
    - Dict keys match joint/axis names for unambiguous access
"""

from __future__ import annotations

__version__ = "1.1.0"

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Joint names for the SO-ARM101 (lerobot_so101_teleop USD model)
SO101_JOINT_NAMES = [
    "Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw",
]


@dataclass
class SceneState:
    """Exact scene conditions for one episode.

    Saved as JSON after env.reset(), before the first action.
    The replayer loads this to reproduce identical conditions.

    All numeric fields use physical unit suffixes (_m, _rad).
    Vector quantities use dicts with named keys for self-documentation.

    Args:
        episode_index: Episode number within the recording session.
        seed: Random seed used for env.reset().
        task: Isaac Lab task name (e.g. "LiftCube-Sim").
        object_key: Scene key of the tracked object (e.g. "cube", "yellow_ring").
        object_init_pos_m: Object position after reset {"x", "y", "z"}.
        object_init_rpy_rad: Object orientation after reset {"roll", "pitch", "yaw"}.
        action_pad_yaw_rad: Action pad yaw angle after reset.
        room_light_exposure: Room light exposure value after reset.
        env_light_exposure: Environment light exposure value after reset.
        sky_dome_yaw_rad: Sky dome yaw angle after reset.
        robot_color_rgb: Robot diffuse color {"r", "g", "b"} in [0, 1].
        object_color_rgb: Object (cube) diffuse color {"r", "g", "b"} in [0, 1].
        object_texture: Object (cube) texture path or "" if none applied.
        table_texture: Action pad texture path or "" if none.
        table_form: Action pad USD asset name (e.g. "action-pad-round.usda").
        robot_default_joint_pos_rad: Robot default joint positions {"Rotation", ...}.
        look_at_m: Camera look-at point {"x", "y", "z"}.
    """

    episode_index: int
    seed: int
    task: str
    object_key: str
    object_init_pos_m: dict[str, float] = field(default_factory=dict)
    object_init_rpy_rad: dict[str, float] = field(default_factory=dict)
    action_pad_yaw_rad: float = 0.0
    room_light_exposure: float = 0.0
    env_light_exposure: float = 0.0
    sky_dome_yaw_rad: float = 0.0
    robot_color_rgb: dict[str, float] = field(default_factory=dict)
    object_color_rgb: dict[str, float] = field(default_factory=dict)
    object_texture: str = ""
    table_texture: str = ""
    table_form: str = ""
    robot_default_joint_pos_rad: dict[str, float] = field(default_factory=dict)
    look_at_m: dict[str, float] = field(default_factory=dict)
    workspace_origin_m: list[float] | None = None
    """Workspace-frame {W} origin of the scene this episode was RECORDED in
    (world coordinates). Declares the frame of ``object_init_pos_m``. ``None``
    on episodes recorded before this field existed, which means the floor-level
    table scene (``HEMISPHERE_ORIGIN_W``)."""

    # --- Getters (for Isaac Sim API) ---

    def get_object_pos_tuple(self) -> tuple[float, float, float]:
        """Return object position as (x, y, z) tuple."""
        p = self.object_init_pos_m
        return (p["x"], p["y"], p["z"])

    def get_look_at_tuple(self) -> tuple[float, float, float]:
        """Return look-at point as (x, y, z) tuple."""
        la = self.look_at_m
        return (la["x"], la["y"], la["z"])

    def get_joint_pos_list(self) -> list[float]:
        """Return joint positions in SO-ARM101 order.

        Order: Rotation, Pitch, Elbow, Wrist_Flex, Wrist_Roll, Gripper.
        """
        return [self.robot_default_joint_pos_rad[j] for j in SO101_JOINT_NAMES]

    # --- Factory helpers ---

    @staticmethod
    def _read_table_texture(stage, unwrapped) -> str:
        """Read action pad diffuse texture filename from USD."""
        try:
            import isaaclab.sim as _sim
            asset = unwrapped.scene["action_pad"]
            prim_path = getattr(asset, "_prim_path", None) if not hasattr(asset, "cfg") else asset.cfg.prim_path
            if prim_path is None:
                return ""
            prims = _sim.find_matching_prims(prim_path)
            if prims:
                pad_prim = prims[0]
                for child in pad_prim.GetAllChildren():
                    file_attr = child.GetAttribute("inputs:file")
                    if file_attr.IsValid():
                        asset_path = str(file_attr.Get().resolvedPath or file_attr.Get().path)
                        return Path(asset_path).name if asset_path else ""
        except Exception:
            pass
        return ""

    @staticmethod
    def _read_table_form(unwrapped) -> str:
        """Read action pad USD asset filename from spawn config."""
        try:
            pad_cfg = unwrapped.cfg.scene.action_pad
            if hasattr(pad_cfg, "spawn") and hasattr(pad_cfg.spawn, "usd_path"):
                return Path(pad_cfg.spawn.usd_path).name
        except Exception:
            pass
        return ""

    # --- Factory ---

    @classmethod
    def from_env(
        cls,
        env: Any,
        episode_index: int,
        seed: int,
        look_at: tuple[float, float, float],
        object_key: str = "cube",
        workspace_origin_m: tuple[float, float, float] | None = None,
    ) -> SceneState:
        """Capture scene state from a live Isaac Sim environment.

        Must be called after env.reset() and before the first env.step().

        Args:
            env: Gymnasium-wrapped Isaac Lab environment.
            episode_index: Current episode number.
            seed: Seed used for this reset.
            look_at: Camera look-at point (x, y, z) in meters.
            object_key: Scene key of the object to track.

        Returns:
            SceneState with all captured values.
        """
        unwrapped = env.unwrapped

        # Object pose
        obj_pos = [0.0, 0.0, 0.0]
        obj_rpy = [0.0, 0.0, 0.0]
        if object_key in unwrapped.scene.keys():
            obj = unwrapped.scene[object_key]
            obj_pos = obj.data.root_pos_w[0].cpu().tolist()
            # Isaac Lab returns root_quat_w as [x, y, z, w] post PR #4437
            # (see CLAUDE.md §2). _quat_to_rpy() expects [w, x, y, z]
            # (ROS convention), so we re-order before handing off.
            # Prior versions read the XYZW tensor as if it were WXYZ,
            # which swapped the yaw rotation into the roll slot — scene_state
            # JSONs written before scene_state.py v1.1.0 carry this bug.
            q_xyzw = obj.data.root_quat_w[0].cpu().tolist()
            q_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
            obj_rpy = _quat_to_rpy(q_wxyz)

        # Robot default joint positions
        robot = unwrapped.scene["robot"]
        joints = robot.data.default_joint_pos[0].cpu().tolist()
        joint_dict = {}
        for i, name in enumerate(SO101_JOINT_NAMES):
            if i < len(joints):
                joint_dict[name] = round(joints[i], 6)

        # Task name
        # Use gym task ID (e.g. "LiftCube-Sim") if available,
        # otherwise fall back to config class name (e.g. "LiftCubeEnvCfg")
        task_name = getattr(unwrapped, "spec", None)
        if task_name is not None and hasattr(task_name, "id"):
            task_name = task_name.id
        else:
            task_name = type(unwrapped.cfg).__name__

        # Read randomized visual/spatial parameters from USD stage
        stage = None
        action_pad_yaw = 0.0
        room_light_exp = 0.0
        env_light_exp = 0.0
        sky_dome_yaw = 0.0
        robot_color = {"r": 0.95, "g": 0.95, "b": 0.95}
        object_color = {"r": 0.8, "g": 0.2, "b": 0.2}  # default cube red

        try:
            from pxr import UsdGeom
            import isaaclab.sim as sim_utils
            from isaaclab.sim import get_current_stage
            stage = get_current_stage()

            def _resolve(asset_name: str):
                """Resolve prim path for any asset type via find_matching_prims.

                Handles AssetBase (.cfg.prim_path), RigidObject (.cfg.prim_path),
                and XformPrimView (._prim_path) which has no .cfg attribute.
                """
                asset = unwrapped.scene[asset_name]
                if hasattr(asset, "cfg"):
                    prim_path = asset.cfg.prim_path
                else:
                    # XformPrimView (extras: room_light, env_light, action_pad)
                    prim_path = getattr(asset, "_prim_path", None)
                    if prim_path is None:
                        return None
                prims = sim_utils.find_matching_prims(prim_path)
                return prims[0] if prims else None

            # ActionPad yaw — from orient XformOp (Gf.Quatd WXYZ)
            pad_prim = _resolve("action_pad")
            if pad_prim is not None:
                for op in UsdGeom.Xformable(pad_prim).GetOrderedXformOps():
                    if "orient" in op.GetOpName():
                        q = op.Get()
                        w, (_, _, z) = q.GetReal(), q.GetImaginary()
                        action_pad_yaw = 2 * math.atan2(z, w)
                        break

            # Room light exposure
            room_prim = _resolve("room_light")
            if room_prim is not None:
                attr = room_prim.GetAttribute("inputs:exposure")
                if attr.IsValid():
                    room_light_exp = float(attr.Get())

            # Env light exposure + sky dome yaw
            env_prim = _resolve("env_light")
            if env_prim is not None:
                attr = env_prim.GetAttribute("inputs:exposure")
                if attr.IsValid():
                    env_light_exp = float(attr.Get())
                for op in UsdGeom.Xformable(env_prim).GetOrderedXformOps():
                    if "orient" in op.GetOpName():
                        q = op.Get()
                        w, (_, _, z) = q.GetReal(), q.GetImaginary()
                        sky_dome_yaw = 2 * math.atan2(z, w)
                        break

            # Robot color — inputs:diffuse_color_constant (PreviewSurfaceCfg shader)
            robot_prim = _resolve("robot")
            if robot_prim is not None:
                shader_path = str(robot_prim.GetPath()) + "/Looks/material_a_3d_printed/Shader"
                shader_prims = sim_utils.find_matching_prims(shader_path)
                if shader_prims:
                    color_attr = shader_prims[0].GetAttribute("inputs:diffuse_color_constant")
                    if color_attr.IsValid():
                        c = color_attr.Get()
                        robot_color = {"r": round(float(c[0]), 4), "g": round(float(c[1]), 4), "b": round(float(c[2]), 4)}

            # Object (cube) color — inputs:diffuseColor (UsdPreviewSurface from CuboidCfg)
            if object_key in unwrapped.scene.keys():
                obj_prim = _resolve(object_key)
                if obj_prim is not None:
                    shader_path = str(obj_prim.GetPath()) + "/geometry/material/Shader"
                    shader = stage.GetPrimAtPath(shader_path)
                    if shader.IsValid():
                        color_attr = shader.GetAttribute("inputs:diffuseColor")
                        if color_attr.IsValid():
                            c = color_attr.Get()
                            object_color = {"r": round(float(c[0]), 4), "g": round(float(c[1]), 4), "b": round(float(c[2]), 4)}

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "Failed to read randomization params from USD stage: %s — using defaults (0.0)", e,
            )

        return cls(
            episode_index=episode_index,
            seed=seed,
            task=task_name,
            object_key=object_key,
            object_init_pos_m={"x": round(obj_pos[0], 6), "y": round(obj_pos[1], 6), "z": round(obj_pos[2], 6)},
            object_init_rpy_rad={
                "roll": round(obj_rpy[0], 6),
                "pitch": round(obj_rpy[1], 6),
                "yaw": round(obj_rpy[2], 6),
            },
            action_pad_yaw_rad=round(action_pad_yaw, 6),
            room_light_exposure=round(room_light_exp, 6),
            env_light_exposure=round(env_light_exp, 6),
            sky_dome_yaw_rad=round(sky_dome_yaw, 6),
            robot_color_rgb=robot_color,
            object_color_rgb=object_color,
            object_texture="",  # reserved for future cube texture randomization
            table_texture=cls._read_table_texture(stage, unwrapped) if stage else "",
            table_form=cls._read_table_form(unwrapped),
            robot_default_joint_pos_rad=joint_dict,
            look_at_m={"x": round(look_at[0], 6), "y": round(look_at[1], 6), "z": round(look_at[2], 6)},
            workspace_origin_m=(
                [round(v, 6) for v in workspace_origin_m]
                if workspace_origin_m is not None else None
            ),
        )

    # --- Persistence ---

    def save(self, path: Path) -> None:
        """Save scene state to JSON file.

        Args:
            path: Output file path (e.g. episode_000_scene_state.json).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> SceneState:
        """Load scene state from JSON file.

        Args:
            path: Input file path.

        Returns:
            SceneState instance.
        """
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def rpy_to_quat_xyzw(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Convert RPY Euler angles (ZYX aerospace convention) to XYZW quaternion.

    Inverse of ``_quat_to_rpy``. Result matches Isaac Lab's post-PR-#4437
    XYZW convention used by ``write_root_pose_to_sim`` and ``init_state.rot``.

    Args:
        roll: Rotation around X-axis in radians.
        pitch: Rotation around Y-axis in radians.
        yaw: Rotation around Z-axis in radians.

    Returns:
        (qx, qy, qz, qw) quaternion tuple.
    """
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


def _quat_to_rpy(quat: list[float]) -> list[float]:
    """Convert quaternion [w, x, y, z] to RPY [roll, pitch, yaw] in radians.

    Uses the standard aerospace convention (ZYX Euler angles).
    """
    w, x, y, z = quat

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # gimbal lock
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return [roll, pitch, yaw]
