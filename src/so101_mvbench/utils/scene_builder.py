"""Shared scene builder — single source of truth for scene setup across all stages.

Used by: gamepad_recorder, ego_recorder, external_batch_recorder, eval_policy.
Guarantees identical visual scene for the same inputs, regardless of which stage
calls it.
"""

from __future__ import annotations

__version__ = "1.0.0"

import dataclasses
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def resolve_all_prim_paths(env, asset_name: str) -> list[str]:
    """Resolve USD prim paths for ALL envs (Multi-Env safe).

    Handles AssetBaseCfg/RigidObjectCfg/ArticulationCfg (.cfg.prim_path)
    and XformPrimView (._prim_path) which has no .cfg attribute.
    Returns every match for the ``{ENV_REGEX_NS}`` pattern.
    """
    import isaaclab.sim as sim_utils

    scene = getattr(env, "unwrapped", env).scene
    asset = scene[asset_name]
    if hasattr(asset, "cfg"):
        prim_path = asset.cfg.prim_path
    else:
        prim_path = getattr(asset, "_prim_path", None)
        if prim_path is None:
            return []
    return [str(p.GetPath()) for p in sim_utils.find_matching_prims(prim_path)]


@dataclass(frozen=True)
class VisualConfig:
    """Visual scene properties applied after env.reset().

    Default values match ``_ZERO_SPATIAL`` / ``_ZERO_VISUAL`` from
    ``mdp/__init__.py`` — deterministic ID-replay baseline.

    Immutable (frozen). Create OOD variants via ``dataclasses.replace()``.
    """

    action_pad_yaw_rad: float = math.pi
    room_light_exposure: float = -1.0
    env_light_exposure: float = 1.0
    sky_dome_yaw_rad: float = math.pi
    robot_color_rgb: tuple[float, float, float] = (0.95, 0.95, 0.95)
    cube_color_rgb: tuple[float, float, float] = (0.8, 0.2, 0.2)
    dome_light_y_mirror: bool = True


ID_VISUAL_CONFIG = VisualConfig()


def load_visual_config(path: str | Path) -> VisualConfig:
    """Load a VisualConfig from a JSON file.

    Only fields present in the JSON are overridden — missing fields keep
    their ID defaults. This allows OOD configs that vary a single parameter.

    Example JSON (vary only cube color)::

        {"cube_color_rgb": [0.2, 0.8, 0.2]}
    """
    import json

    data = json.loads(Path(path).read_text())
    for key in ("robot_color_rgb", "cube_color_rgb"):
        if key in data and isinstance(data[key], list):
            data[key] = tuple(data[key])
    return dataclasses.replace(ID_VISUAL_CONFIG, **data)


def apply_scene(
    env,
    episode_idx: int,
    binmap,
    *,
    visual: VisualConfig = ID_VISUAL_CONFIG,
    cube_noise_m: float = 0.0,
    cube_rot_noise_deg: float = 0.0,
) -> dict:
    """Set up the complete scene — robot joints, cube pose, visual properties.

    Call AFTER ``env.reset()`` and BEFORE warmup steps.
    Multi-Env safe: applies to ALL environment instances.

    Args:
        env: Gym-wrapped Isaac Lab environment.
        episode_idx: Episode index for BinMap lookup.
        binmap: ``BinMap`` instance with per-episode cube configs.
        visual: Visual scene configuration. Use ``ID_VISUAL_CONFIG`` (default)
            for in-distribution replay, or a custom ``VisualConfig`` for OOD.
        cube_noise_m: Additive uniform noise on cube x/y (for OOD eval).
        cube_rot_noise_deg: Additive uniform noise on cube yaw (for OOD eval).

    Returns:
        Dict with actually placed cube coordinates:
        ``{"cube_x_m": float, "cube_y_m": float, "cube_yaw_rad": float}``.
    """
    _reset_robot_joints(env)
    placed_cube = _apply_cube_pose(env, episode_idx, binmap, cube_noise_m, cube_rot_noise_deg)
    _apply_visual_properties(env, visual)
    return placed_cube


def _reset_robot_joints(env) -> None:
    """Reset robot joints to default pose + zero velocity (all envs)."""
    scene = env.unwrapped.scene
    robot = scene["robot"]
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)


def _apply_cube_pose(
    env,
    episode_idx: int,
    binmap,
    cube_noise_m: float,
    cube_rot_noise_deg: float,
) -> dict:
    """Cube pose from BinMap (authoritative source).

    Returns:
        Dict with actually placed coordinates:
        ``{"cube_x_m": float, "cube_y_m": float, "cube_yaw_rad": float}``.
    """
    from so101_mvbench.utils.bin_spawner import apply_cube_pose

    cfg = binmap.get_episode_config(episode_idx)
    if cfg is None:
        raise RuntimeError(
            f"BinMap has no entry for episode_idx={episode_idx}"
        )

    cx = cfg.x_m + random.uniform(-cube_noise_m, cube_noise_m)
    cy = cfg.y_m + random.uniform(-cube_noise_m, cube_noise_m)
    yaw_noise_rad = math.radians(
        random.uniform(-cube_rot_noise_deg, cube_rot_noise_deg)
    )
    yaw_rad = cfg.yaw_rad + yaw_noise_rad

    env_origins = env.unwrapped.scene.env_origins
    apply_cube_pose(env, cx, cy, yaw_rad, env_origins=env_origins)
    logger.info(
        "  ep%03d cube: bin=(%d,%d) pos=(%.3f, %.3f) yaw=%.1f°",
        episode_idx, cfg.bin_col, cfg.bin_row, cx, cy,
        math.degrees(yaw_rad),
    )
    return {"cube_x_m": cx, "cube_y_m": cy, "cube_yaw_rad": yaw_rad}


def _apply_visual_properties(env, visual: VisualConfig) -> None:
    """Apply all visual properties to ALL env instances (Multi-Env safe)."""
    from pxr import Gf, Sdf, UsdGeom
    import isaaclab.sim as sim_utils
    from isaaclab.sim import get_current_stage

    stage = get_current_stage()
    scene = env.unwrapped.scene

    env_light_paths = resolve_all_prim_paths(env, "env_light") if "env_light" in scene.keys() else []
    robot_paths = resolve_all_prim_paths(env, "robot") if "robot" in scene.keys() else []

    with Sdf.ChangeBlock():
        # --- Light exposures ---
        for light_key, exposure in [
            ("room_light", visual.room_light_exposure),
            ("env_light", visual.env_light_exposure),
        ]:
            if light_key not in scene.keys():
                continue
            paths = env_light_paths if light_key == "env_light" else resolve_all_prim_paths(env, light_key)
            for path in paths:
                prim = stage.GetPrimAtPath(path)
                if prim.IsValid():
                    prim.GetAttribute("inputs:exposure").Set(exposure)

        # --- Sky dome yaw (env_light orient) ---
        if env_light_paths:
            quat = Gf.Quatd(
                math.cos(visual.sky_dome_yaw_rad / 2),
                0, 0,
                math.sin(visual.sky_dome_yaw_rad / 2),
            )
            for path in env_light_paths:
                prim = stage.GetPrimAtPath(path)
                for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
                    if "orient" in op.GetOpName():
                        op.Set(quat)
                        break

        # --- Sky dome Y-mirror scale (skip = leave untouched, not reset) ---
        if visual.dome_light_y_mirror and env_light_paths:
            scale = Gf.Vec3d(1.0, -1.0, 1.0)
            for path in env_light_paths:
                prim = stage.GetPrimAtPath(path)
                for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
                    if "scale" in op.GetOpName():
                        op.Set(scale)
                        break

        # --- Action pad yaw ---
        if "action_pad" in scene.keys():
            quat = Gf.Quatd(
                math.cos(visual.action_pad_yaw_rad / 2),
                0, 0,
                math.sin(visual.action_pad_yaw_rad / 2),
            )
            for path in resolve_all_prim_paths(env, "action_pad"):
                prim = stage.GetPrimAtPath(path)
                for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
                    if "orient" in op.GetOpName():
                        op.Set(quat)
                        break

        # --- Robot color ---
        if robot_paths:
            r, g, b = visual.robot_color_rgb
            for path in robot_paths:
                shader_path = path + "/Looks/material_a_3d_printed/Shader"
                shader_prims = sim_utils.find_matching_prims(shader_path)
                if shader_prims:
                    shader_prims[0].GetAttribute(
                        "inputs:diffuse_color_constant"
                    ).Set(Gf.Vec3f(r, g, b))

        # --- Cube color ---
        if "cube" in scene.keys():
            r, g, b = visual.cube_color_rgb
            for path in resolve_all_prim_paths(env, "cube"):
                shader = stage.GetPrimAtPath(
                    path + "/geometry/material/Shader"
                )
                if shader.IsValid():
                    shader.GetAttribute("inputs:diffuseColor").Set(
                        Gf.Vec3f(r, g, b)
                    )

    logger.info(
        "  Scene visual applied: pad=%.1f° light=(%.1f,%.1f) dome=%.1f° y_mirror=%s",
        math.degrees(visual.action_pad_yaw_rad),
        visual.room_light_exposure, visual.env_light_exposure,
        math.degrees(visual.sky_dome_yaw_rad),
        visual.dome_light_y_mirror,
    )
