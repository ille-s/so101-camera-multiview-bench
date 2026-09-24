# SPDX-License-Identifier: MIT
"""Task package — env-cfg classes + gym.register hooks."""

import gymnasium as gym
from isaaclab_tasks.utils import import_packages

# Auto-import sub-modules so their module-level definitions are loaded
# (NVIDIA-style pattern, matches sim_to_real_so101.tasks).
_BLACKLIST_PKGS = ["utils"]
import_packages(__name__, _BLACKLIST_PKGS)

_ENTRY_POINT = "isaaclab.envs:ManagerBasedRLEnv"

# LiftCube-Sim — base task, no DR (ID = in-distribution).
gym.register(
    id="LiftCube-Sim",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.lift_cube_env_cfg:LiftCubeEnvCfg"},
)

# LiftCube-Sim-OOD-Light — cube position perturbed (±2cm XY, ±10° yaw).
gym.register(
    id="LiftCube-Sim-OOD-Light",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.lift_cube_env_cfg:LiftCubeOodLightEnvCfg"},
)

# LiftCube-Sim-OOD-Visual — full visual DR (lighting, sky_dome_yaw, colors).
gym.register(
    id="LiftCube-Sim-OOD-Visual",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.lift_cube_env_cfg:LiftCubeOodVisualEnvCfg"},
)

# LiftCube-Sim-OOD-Position — deterministic OOD grid (binmap_ood.json), IID rotation + visual.
gym.register(
    id="LiftCube-Sim-OOD-Position",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.lift_cube_env_cfg:LiftCubeOodPositionEnvCfg"},
)

# Multi-external camera variants — one per camera count.
gym.register(
    id="LiftCube-Sim-2Ext-OOD-Position",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.lift_cube_env_cfg:LiftCube2ExtOodPositionEnvCfg"},
)
gym.register(
    id="LiftCube-Sim-2Ext",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.lift_cube_env_cfg:LiftCube2ExtEnvCfg"},
)
gym.register(
    id="LiftCube-Sim-3Ext",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.lift_cube_env_cfg:LiftCube3ExtEnvCfg"},
)
gym.register(
    id="LiftCube-Sim-4Ext",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.lift_cube_env_cfg:LiftCube4ExtEnvCfg"},
)
# LiftCube-Sim-6Cam — wrist + 5 externals = 6-cam total (back/front/left/right/top + wrist).
# Used for evaluating policies trained on `merged_6cam_phase2`.
gym.register(
    id="LiftCube-Sim-6Cam",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.lift_cube_env_cfg:LiftCube5ExtEnvCfg"},
)
