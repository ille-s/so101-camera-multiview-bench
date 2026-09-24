"""SceneStateLoader — apply per-episode scene_state JSONs to Isaac Sim (single-env).

Rewrite of design00 scene_loader.py without the ``except Exception: return {}``
silent-continue anti-pattern. All failures raise with clear messages.

Responsibility
--------------
- Load ``episode_NNNNNN_scene_state.json`` files from a meta-dir into memory
- For each episode: write USD attributes into Isaac Sim (lights, yaws, colors, cube)

Not our job
-----------
- Validate JSON schema / ranges / NaN → ``startup.py`` did that already
- Own the list of keys                → ``shared.scene_state.SceneState`` is SOT
- Multi-env broadcasting              → eval is single-env, ``prims[0]`` is the only prim

Version 1.1.0
"""

from __future__ import annotations

__version__ = "1.1.0"

import json
from pathlib import Path
from typing import TYPE_CHECKING

from so101_mvbench.logging import get_logger

if TYPE_CHECKING:
    from so101_mvbench.utils.scene_builder import VisualConfig

logger = get_logger(__name__)


class SceneApplyError(RuntimeError):
    """Raised when applying scene_state to Isaac Sim fails. NOT silent."""


class SceneStateLoader:
    """Iterator over scene_state.json files + Isaac Sim USD applicator."""

    def __init__(self, scene_state_dir: Path, binmap_path: Path, device: str):
        self._device = device
        self._states: list[dict] = []

        # startup already validated dir + files; we just load into memory.
        for p in sorted(scene_state_dir.glob("episode_*_scene_state.json")):
            self._states.append(json.loads(p.read_text()))
        if not self._states:
            raise SceneApplyError(f"scene_state_dir empty: {scene_state_dir}")

        # BinMap is the authoritative cube-yaw source
        # (scene_state.object_init_rpy_rad unreliable post PR #4437 — see
        # docs/plans/2026-04-19 post-mortem).
        from so101_mvbench.utils.bin_spawner import BinMap
        self._binmap = BinMap.load(binmap_path)
        logger.info(
            "SceneStateLoader v%s: %d scene_states, binmap with %d episodes",
            __version__, len(self._states), len(self._binmap.episodes),
        )

    @property
    def count(self) -> int:
        return len(self._states)

    def state_for_episode(self, episode_idx: int) -> dict:
        """Return scene_state dict for episode index (0-based).

        Wraps with modulo when episode_idx >= len(states), so OOD grids
        with more episodes than scene_states reuse visual properties cyclically.
        """
        if episode_idx < 0:
            raise SceneApplyError(f"episode_idx={episode_idx} must be >= 0")
        wrapped = episode_idx % len(self._states)
        if wrapped != episode_idx:
            logger.debug(
                "  ep%03d → scene_state[%d] (cyclic wrap, %d states)",
                episode_idx, wrapped, len(self._states),
            )
        return self._states[wrapped]

    # -----------------------------------------------------------------------
    # Apply — write to Isaac Sim. Raise on failure, NOT silent-continue.
    # -----------------------------------------------------------------------

    @staticmethod
    def visual_config_from_scene_state(ss: dict) -> "VisualConfig":
        """Build a VisualConfig from a scene_state dict (IID reproduction)."""
        from so101_mvbench.utils.scene_builder import VisualConfig

        rc = ss.get("robot_color_rgb", {})
        cc = ss.get("object_color_rgb", {})
        return VisualConfig(
            action_pad_yaw_rad=ss.get("action_pad_yaw_rad", 3.141593),
            room_light_exposure=ss.get("room_light_exposure", -1.0),
            env_light_exposure=ss.get("env_light_exposure", 1.0),
            sky_dome_yaw_rad=ss.get("sky_dome_yaw_rad", 3.141593),
            robot_color_rgb=(rc.get("r", 0.95), rc.get("g", 0.95), rc.get("b", 0.95)),
            cube_color_rgb=(cc.get("r", 0.8), cc.get("g", 0.2), cc.get("b", 0.2)),
            dome_light_y_mirror=True,
        )

    def apply(
        self,
        env,
        episode_idx: int,
        *,
        cube_noise_m: float = 0.0,
        cube_rot_noise_deg: float = 0.0,
        visual_override: "VisualConfig | None" = None,
    ) -> dict:
        """Apply scene_state for ``episode_idx`` to Isaac Sim.

        Args:
            visual_override: If None (IID), visual config is built from
                scene_state.json (reproduces training scene exactly).
                If a VisualConfig is passed (OOD), it overrides all visual
                properties for controlled variation testing.

        Returns dict with ``"scene_state"`` (the loaded JSON) and
        ``"placed_cube"`` (actually placed coordinates from scene_builder).
        """
        from so101_mvbench.utils.scene_builder import apply_scene

        ss = self.state_for_episode(episode_idx)

        if visual_override is not None:
            visual = visual_override
        else:
            visual = self.visual_config_from_scene_state(ss)

        placed_cube = apply_scene(
            env,
            episode_idx,
            self._binmap,
            visual=visual,
            cube_noise_m=cube_noise_m,
            cube_rot_noise_deg=cube_rot_noise_deg,
        )

        logger.info(
            "  ep%03d scene applied (cube, visual=%s)",
            episode_idx,
            "override" if visual_override is not None else "from_scene_state",
        )
        return {"scene_state": ss, "placed_cube": placed_cube}
