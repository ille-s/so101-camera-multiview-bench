# SPDX-License-Identifier: MIT
"""Event-reset terms for LiftCube tasks — visual / spatial randomization.

All functions here have the standard Isaac Lab reset-term signature
``func(env, env_ids, **params)`` and are wired into a `LiftCubeEventCfg`
via `EventTermCfg(func=randomize_X, mode="reset", params={...})`.

Multi-Env-safe prim resolver: ``resolve_all_prim_paths`` from ``utils.scene_builder``.
"""

from __future__ import annotations

import math
import random

import torch
from pxr import Gf, Sdf

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import get_current_stage

from so101_mvbench.utils.scene_builder import resolve_all_prim_paths


# ---------------------------------------------------------------------------
# Visual DR (trajectory-independent: replay-safe)
# ---------------------------------------------------------------------------

def randomize_light_exposure(
    env,
    env_ids: torch.Tensor | None,
    exposure_range: tuple[float, float],
    asset_cfg: SceneEntityCfg | None = None,
) -> None:
    """Randomize light exposure via USD attribute (`inputs:exposure`).

    Applies to ALL env instances (Multi-Env safe).
    """
    stage = get_current_stage()
    all_paths = resolve_all_prim_paths(env, asset_cfg.name)
    exposure = random.uniform(*exposure_range)
    with Sdf.ChangeBlock():
        for path in all_paths:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                prim.GetAttribute("inputs:exposure").Set(exposure)


def randomize_static_asset_yaw(
    env,
    env_ids: torch.Tensor | None,
    yaw_range: tuple[float, float],
    asset_cfg: SceneEntityCfg | None = None,
) -> None:
    """Randomize asset yaw (Z-rotation) via USD orient XformOp.

    Sets only the orient op, preserving translate. Uses Gf.Quatd (WXYZ)
    directly to avoid XYZW/WXYZ confusion between isaaclab and isaacsim APIs.

    Applies to ALL env instances (Multi-Env safe).
    """
    from pxr import UsdGeom

    stage = get_current_stage()
    all_paths = resolve_all_prim_paths(env, asset_cfg.name)
    yaw = random.uniform(*yaw_range)
    quat = Gf.Quatd(math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))

    with Sdf.ChangeBlock():
        for path in all_paths:
            prim = stage.GetPrimAtPath(path)
            for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
                if "orient" in op.GetOpName():
                    op.Set(quat)
                    break


def randomize_robot_color(
    env,
    env_ids: torch.Tensor | None,
    color_list: list[tuple[float, float, float]],
) -> None:
    """Randomize robot mesh color from a discrete palette.

    Shader attribute: ``inputs:diffuse_color_constant`` (PreviewSurface).
    Path ``/Looks/material_a_3d_printed/Shader`` is specific to the SO-ARM101 USD.
    Applies to ALL env instances (Multi-Env safe).
    """
    color = random.choice(color_list)
    all_paths = resolve_all_prim_paths(env, "robot")
    with Sdf.ChangeBlock():
        for path in all_paths:
            shader_path = path + "/Looks/material_a_3d_printed/Shader"
            shader_prims = sim_utils.find_matching_prims(shader_path)
            if shader_prims:
                shader_prims[0].GetAttribute("inputs:diffuse_color_constant").Set(Gf.Vec3f(*color))


def randomize_cube_color(
    env,
    env_ids: torch.Tensor | None,
    color_list: list[tuple[float, float, float]],
    asset_cfg: SceneEntityCfg | None = None,
) -> None:
    """Randomize cube color from a discrete palette.

    Shader attribute: ``inputs:diffuseColor`` (UsdPreviewSurface from CuboidCfg).
    Applies to ALL env instances (Multi-Env safe).
    """
    color = random.choice(color_list)
    all_paths = resolve_all_prim_paths(env, asset_cfg.name)
    stage = get_current_stage()
    with Sdf.ChangeBlock():
        for path in all_paths:
            shader = stage.GetPrimAtPath(path + "/geometry/material/Shader")
            if shader.IsValid():
                shader.GetAttribute("inputs:diffuseColor").Set(Gf.Vec3f(*color))
