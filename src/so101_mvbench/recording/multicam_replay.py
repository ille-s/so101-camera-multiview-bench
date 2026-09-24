#!/usr/bin/env python3
"""cylroom_replay.py — Phase-4 replay/record in the cylindrical-room scene.

Standalone recorder for the table-mounted cylindrical-room scene (Phase 4).
Replays existing Phase-3 gamepad trajectories (``trajectories/episode_*.npy`` +
``meta/episode_*_scene_state.json``) into the new scene where the arm sits on a
pedestal table (base at (-0.20, 0, 0.74), workspace center == table center) and
synthesizes all N cameras in ONE env (num_envs=1).

WHY THE TRAJECTORIES STAY VALID (verified 2026-06-03):
    Joint trajectories are frame-agnostic. The arm has the SAME orientation
    (yaw +90 deg, faces +X) in both the Phase-3 floor scene (base at origin) and
    this table scene (base at (-0.20, 0, 0.74)). Only the base TRANSLATION differs
    (no rotation). So the whole arm-relative world content — the cube and the
    camera hemisphere — is shifted rigidly by ``DATA_SHIFT`` (source-scene origin
    to target-scene origin, both declared); the recorded joint angles then
    reproduce the identical grasp relative to the base.

DESIGN — reuses the proven ``synthetic_multicam_recorder`` machinery as a NEW FILE
(imports ``LiftCubeSceneCfg`` etc. but edits NO existing config → byte-identical,
branch-isolated). Phase-4 overrides vs that recorder:
    1. ``room.spawn.usd_path`` -> ``cylindrical_room_shell.usda`` (room+table+lights,
       no arm — the arm is spawned by the robot field, table-mounted).
    3. cube world pose + ``DATA_SHIFT`` (post scene_loader.apply).
    4. external-camera hemisphere origin = the scene config's workspace origin.

CLI mirrors synthetic_multicam_recorder (``--mode {create,append}``, ``--episodes``
range-syntax or 'all').

Version: 0.1.0
"""

from __future__ import annotations

VERSION = "0.1.0"

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
from pathlib import Path


_T_SCRIPT_START = time.time()

# Scene geometry: the scene config's workspace_origin_m is the pose-graph root;
# RIG_SHIFT and DATA_SHIFT are derived from it after the config is loaded
# (below). The named constants (TABLE_TOP_Z, ...) live in so101_transforms.


# ---------------------------------------------------------------------------
# CLI args — parsed BEFORE Isaac Sim boot
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Phase-4: replay Phase-3 trajectory in the cylindrical-room table scene + record cameras."
)
parser.add_argument("--dataset_root", type=str, required=True,
                    help="Source LeRobot dataset (contains trajectories/ + meta/)")
parser.add_argument("--episodes", type=str, required=True,
                    help="Episodes: '0', '0,1,5', '0-10', '0,5-10', or 'all'.")
parser.add_argument("--scene_config", type=str, required=True,
                    help="Path to scene_config JSON (M1 schema 1.2+). Defines cameras.")
from so101_mvbench.settings import SETTINGS  # noqa: E402
# SETTINGS.datasets_root is a plain field, not require_datasets_root(): the latter
# raises when the directory is absent, which at module level would break --help.
parser.add_argument("--output_base", type=str,
                    default=str(SETTINGS.datasets_root),
                    help="Datasets root; <config_stem>_cylroom/<source_bin> is appended.")
parser.add_argument("--mode", type=str, choices=["create", "append"], default="create")
parser.add_argument("--room_usd", type=str, default=None,
                    help="Override room shell USD (default: assets/usd/cylindrical_room_shell.usda).")
parser.add_argument("--pad_usd", type=str, default=None,
                    help="Override action_pad USD (default: assets/usd/action-pad-round.usda, "
                         "visual-only matte disk).")
parser.add_argument("--task_name", type=str, default="lift_cube_replay_cylroom")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true")
parser.add_argument("--pad_z_offset", type=float, default=None,
                    help="Override action_pad floor-frame z offset (default 0.0222). "
                         "Use for height sweeps; pad-top(world) = offset + 0.7463 + 0.002.")
parser.add_argument("--screenshot", type=str, default=None,
                    help="Diagnostic: after settle, render the 'right' cam to this PNG and "
                         "exit (no dataset, no replay). For pad-height sweeps.")
parser.add_argument("--gui_hold", action="store_true",
                    help="Diagnostic: load the scene, settle the cube, then HOLD (arm at "
                         "start pose) keeping the GUI open for inspection until the window is "
                         "closed. No dataset, no replay. Run WITHOUT --headless.")
parser.add_argument("--verify_placement", type=str, default=None,
                    help="Verify cube placement for ALL --episodes: reset+apply+place+settle, "
                         "read cube world pos, compare to scene_state+WORLD_DELTA, APPEND rows to "
                         "this CSV, then exit (no replay/record). Run per bin to audit all 540.")
parser.add_argument("--diag_cube", action="store_true",
                    help="Diagnostic: log cube world pos at reset / after apply / after "
                         "_place_cube_table / after settle vs expected, then exit (no replay). "
                         "Verifies the cube lands at floor_scene_state + WORLD_DELTA.")

from isaaclab.app import AppLauncher  # noqa: E402
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# Pre-flight (cheap, before Isaac Sim boot)
_ds_root = Path(args_cli.dataset_root)
_scene_cfg_path = Path(args_cli.scene_config)
_binmap_path = _ds_root / "binmap.json"
_pkg_root = Path(__file__).resolve().parents[1]  # .../so101_mvbench
_shell_usd = Path(args_cli.room_usd) if args_cli.room_usd else (
    _pkg_root / "assets" / "usd" / "cylindrical_room_shell.usda"
)
# Default action pad: ROUND matte disk, VISUAL-ONLY (no collider — any proud collider
# collapses the arm). Kept for contrast on the table. Pad top is held <1 mm above the
# table surface (see pad z = 0.0222 below -> top ~0.7705, 0.5 mm proud) so it stays
# visible without occluding the table-resting cube. Alternatives via --pad_usd:
# action-pad-none.usda (no pad) / action-pad-plain.usda (square).
_none_pad_usd = _pkg_root / "assets" / "usd" / "action-pad-none.usda"
_round_pad_usd = _pkg_root / "assets" / "usd" / "action-pad-round.usda"
_pad_usd = Path(args_cli.pad_usd) if args_cli.pad_usd else _round_pad_usd

from so101_mvbench.utils.episodes_spec import parse_episodes_spec  # noqa: E402
try:
    _episode_ids = parse_episodes_spec(args_cli.episodes, _ds_root)
except (ValueError, FileNotFoundError) as _e:
    print(f"ERR: --episodes parse failed: {_e}", file=sys.stderr)
    sys.exit(2)

for _p, _desc in (
    (_scene_cfg_path, "scene_config"),
    (_binmap_path, "binmap.json (needed by SceneStateLoader)"),
    (_shell_usd, "room shell USD"),
):
    if not _p.exists():
        print(f"ERR: {_desc} not found: {_p}", file=sys.stderr)
        sys.exit(2)
for _ep_idx in _episode_ids:
    _npy_path = _ds_root / "trajectories" / f"episode_{_ep_idx:03d}.npy"
    _ss_path = _ds_root / "meta" / f"episode_{_ep_idx:06d}_scene_state.json"
    if not _npy_path.exists():
        print(f"ERR: trajectory not found: {_npy_path}", file=sys.stderr)
        sys.exit(2)
    if not _ss_path.exists():
        print(f"ERR: scene_state not found: {_ss_path}", file=sys.stderr)
        sys.exit(2)

# Output: <output_base>/<scene_config_stem>_cylroom/<source_bin>/
_config_stem = _scene_cfg_path.stem
_source_bin = _ds_root.name
_output_root = Path(args_cli.output_base) / f"{_config_stem}_cylroom" / _source_bin
args_cli.output_root = str(_output_root)

if args_cli.mode == "create":
    if _output_root.exists():
        print(f"ERR: --mode=create but output_root exists: {_output_root}\n"
              f"     Use --mode=append or remove the dir for a fresh build.",
              file=sys.stderr)
        sys.exit(2)
elif args_cli.mode == "append":
    if not _output_root.exists():
        print(f"ERR: --mode=append but output_root doesn't exist: {_output_root}",
              file=sys.stderr)
        sys.exit(2)
    _archived_cfg_path = _output_root / "meta" / "scene_config.json"
    if not _archived_cfg_path.exists():
        print(f"ERR: --mode=append: no archived meta/scene_config.json: {_archived_cfg_path}",
              file=sys.stderr)
        sys.exit(2)
    from so101_mvbench.utils.scene_config import SceneConfig as _SC
    try:
        _cur_cfg = _SC.from_json(_scene_cfg_path)
        _archived_cfg = _SC.from_json(_archived_cfg_path)
    except (ValueError, KeyError) as _e:
        print(f"ERR: scene_config load failed during compat check: {_e}", file=sys.stderr)
        sys.exit(2)
    if _cur_cfg.cameras != _archived_cfg.cameras:
        print(f"ERR: --mode=append: camera-set mismatch\n"
              f"     archived: {sorted(_archived_cfg.cameras.keys())}\n"
              f"     supplied: {sorted(_cur_cfg.cameras.keys())}",
              file=sys.stderr)
        sys.exit(2)

print(f">>> Phase-4 cylroom replay | mode={args_cli.mode} | output_root: {_output_root}", flush=True)
print(f">>> shell_usd: {_shell_usd}", flush=True)
print(f">>> episodes ({len(_episode_ids)}): {_episode_ids}", flush=True)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ---------------------------------------------------------------------------
# Imports that require an active SimulationApp
# ---------------------------------------------------------------------------

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import so101_mvbench.tasks  # noqa: F401, E402  (gym registration)

from so101_mvbench.tasks.lift_cube_env_cfg import (  # noqa: E402
    LiftCubeSceneCfg,
    _euler_to_quat_xyzw,
    configure_randomization,
)
from so101_mvbench.evaluation.scene_loader import (  # noqa: E402
    SceneStateLoader,
)
from so101_mvbench.utils import camera_grid  # noqa: E402
from so101_mvbench.utils.so101_transforms import (  # noqa: E402
    build_joint_tensors,
    raw_degrees_to_sim_radians,
    sim_radians_to_raw_degrees,
)
from so101_mvbench.utils.scene_config import (  # noqa: E402
    CameraSpec,
    SceneConfig,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")

# Scene geometry is driven by two declared origins: the scene config's
# workspace_origin_m is the pose-graph root {W} of the TARGET scene, and the
# binmap's workspace_origin_m is the root of the scene the data was RECORDED in
# (absent on legacy datasets = the floor-level table scene). The rigid shift for
# the recorded rig (arm, cube, pad, camera hemisphere) is the difference. Table
# recording replayed in the cylroom reproduces the historical ARM_BASE_OFFSET;
# re-rendering a cylroom recording in the same room yields a zero shift.
import json as _json  # noqa: E402

from so101_mvbench.utils.so101_transforms import (  # noqa: E402
    BASE_SCENE_ORIGIN_W,
    CUBE_SPAWN_Z_IN_W,
    HEMISPHERE_ORIGIN_W,
    rigid_shift_from_recording,
)

_scene_config = SceneConfig.from_json(_scene_cfg_path)
WS_ORIGIN: tuple[float, float, float] = tuple(_scene_config.workspace_origin_m)
_src = _json.loads(_binmap_path.read_text()).get("workspace_origin_m")
SOURCE_ORIGIN: tuple[float, float, float] = tuple(_src) if _src is not None else HEMISPHERE_ORIGIN_W
# Two DIFFERENT translations, easy to conflate (and conflating them put the arm
# on the floor during a cylroom-to-cylroom replay):
#  - RIG_SHIFT anchors the scene furniture. The base LiftCubeSceneCfg is authored
#    natively in the cylindrical room, so robot base, action pad and the cube's
#    placeholder spawn move by target-vs-BASE_SCENE (zero for the shipped
#    configs) regardless of where the data came from.
#  - DATA_SHIFT transports the recorded coordinates (binmap x/y, spawn height)
#    from the SOURCE scene into the target. Same scene on both sides -> zero.
RIG_SHIFT: tuple[float, float, float] = rigid_shift_from_recording(WS_ORIGIN, BASE_SCENE_ORIGIN_W)
DATA_SHIFT: tuple[float, float, float] = rigid_shift_from_recording(WS_ORIGIN, SOURCE_ORIGIN)
# Cube spawn height is {W}-fixed; its world value in the SOURCE frame:
SOURCE_CUBE_SPAWN_Z: float = SOURCE_ORIGIN[2] + CUBE_SPAWN_Z_IN_W

FPS = 30
W, H = 640, 480
WARMUP_STEPS = 60
SETTLE_STEPS = 30
# Render-product priming before the first add_frame. The 6 TiledCameras need
# their RTX render products fully compiled before streaming-encode starts;
# too few steps here can leave encoders fed with not-yet-ready frames. The
# recorder uses 3; we raise headroom for the fresh cylindrical scene.
PROPAGATION_STEPS = 20


# ---------------------------------------------------------------------------
# Camera build from scene_config JSON (identical to synthetic_multicam_recorder).
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class _BuiltCamera:
    field: str
    cfg: TiledCameraCfg
    pose: "camera_grid.CameraPose | None"


def _build_camera(name: str, spec: CameraSpec) -> _BuiltCamera:
    field = f"camera_{name}"
    if spec.type == "ego":
        cfg = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper/" + name + "_link",
            update_period=0.0, height=H, width=W, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=spec.focal_length,
                horizontal_aperture=spec.horizontal_aperture,
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=tuple(spec.offset_pos_m),
                rot=_euler_to_quat_xyzw(list(spec.offset_rot_deg)),
                convention="opengl",
            ),
        )
        return _BuiltCamera(field=field, cfg=cfg, pose=None)

    if spec.type == "external":
        x, y, z = camera_grid.spherical_to_cartesian(
            spec.azimuth_deg, spec.elevation_deg, spec.radius_m, camera_grid.WORKSPACE_CENTER,
        )
        pose = camera_grid.CameraPose(
            id=name, azimuth_deg=spec.azimuth_deg, elevation_deg=spec.elevation_deg,
            radius_m=spec.radius_m, x=x, y=y, z=z,
        )
        cfg = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/" + name + "_ext",
            update_period=0.0, height=H, width=W, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=spec.focal_length,
                horizontal_aperture=spec.horizontal_aperture,
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.57, 0.0, 0.055),  # dummy — overwritten by set_external_camera_pose
                rot=_euler_to_quat_xyzw([115, 0, 90]),
                convention="opengl",
            ),
        )
        return _BuiltCamera(field=field, cfg=cfg, pose=pose)

    raise ValueError(f"camera {name!r}: unknown type {spec.type!r}")


def _build_cameras_from_config(cfg: SceneConfig) -> list[_BuiltCamera]:
    return [_build_camera(name, spec) for name, spec in cfg.cameras.items()]


def _make_recorder_scene_cfg(built: list[_BuiltCamera]) -> type:
    fields = [
        (rec.field, TiledCameraCfg, dataclasses.field(default_factory=lambda c=rec.cfg: c))
        for rec in built
    ]
    DynamicCls = dataclasses.make_dataclass(
        "CylRoomSceneCfg", fields, bases=(LiftCubeSceneCfg,),
    )
    return configclass(DynamicCls)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mark(msg: str) -> None:
    print(f"\n>>> {msg}\n", flush=True)


def _build_features(scene_config: SceneConfig) -> dict:
    joint_names = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                   "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
    features: dict = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": joint_names},
        "action":            {"dtype": "float32", "shape": (6,), "names": joint_names},
    }
    for name in scene_config.cameras:
        features[f"observation.images.{name}"] = {
            "dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channels"],
        }
    return features




def _place_cube_table(scene, placed_cube: dict, world_delta) -> None:
    """Place the cube ABSOLUTELY at ``source_value + DATA_SHIFT`` (one transform).

    Frame-consistent with the other two anchors — the arm base and the camera
    hemisphere are shifted by the same source-to-target delta. ``apply_scene``
    placed the cube x,y in the SOURCE scene's frame (and its yaw); here we WRITE
    the final world pose ``(source x, source y, SOURCE_CUBE_SPAWN_Z) + shift``
    directly (per env via ``env_origins``). Absolute write — no relative read of
    ``root_state_w`` and no reliance on the env_cfg default — so a double-shift
    (the 1.54 m drop bug) is structurally impossible. Keeps the apply yaw
    (quat 3:7), zeroes velocity. Run inside ``torch.inference_mode`` (Isaac Lab
    cached-tensor constraint).
    """
    cube = scene["cube"]
    eo = scene.env_origins  # (n, 3) world-frame env origins
    root_state = cube.data.root_state_w.clone()  # (n, 13): pos(3) quat(4) linvel(3) angvel(3)
    root_state[:, 0] = eo[:, 0] + placed_cube["cube_x_m"] + world_delta[0]
    root_state[:, 1] = eo[:, 1] + placed_cube["cube_y_m"] + world_delta[1]
    root_state[:, 2] = eo[:, 2] + SOURCE_CUBE_SPAWN_Z + world_delta[2]
    root_state[:, 7:13] = 0.0
    cube.write_root_state_to_sim(root_state)


def _set_pad_yaw(env, yaw_rad: float = 0.0) -> None:
    """Pin the action_pad yaw to a fixed, axis-aligned value (USD orient op).

    ``scene_loader.apply`` (via ``apply_scene``) writes the per-episode
    ``scene_state.action_pad_yaw_rad`` (a floor-training DR artifact) onto the
    pad. In the fixed table scene the pad is a static workspace mat that should
    stay aligned with the 5x5 grid / world axes, so we overwrite the yaw right
    after apply. Mirrors apply_scene's pad-orient mechanism (USD stage write,
    not a physics tensor → safe inside/outside inference_mode).
    """
    import math as _m
    import omni.usd
    from pxr import Gf, UsdGeom
    from so101_mvbench.utils.scene_builder import resolve_all_prim_paths

    stage = omni.usd.get_context().get_stage()
    quat = Gf.Quatd(_m.cos(yaw_rad / 2.0), 0.0, 0.0, _m.sin(yaw_rad / 2.0))
    for path in resolve_all_prim_paths(env, "action_pad"):
        prim = stage.GetPrimAtPath(path)
        for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
            if "orient" in op.GetOpName():
                op.Set(quat)
                break


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    t_main = time.time()
    phases: dict[str, float] = {"sim_boot": t_main - _T_SCRIPT_START}
    _mark(f"0. main() entered  [sim_boot {phases['sim_boot']:.1f}s]")

    # ----- 1. scene_config -----
    scene_config = _scene_config  # loaded once at module level (drives the geometry)
    cam_names = list(scene_config.cameras.keys())
    logger.info(
        "cylroom_replay v%s | scene_config=%s | %d cams (%s) | bin=%s | eps=%s",
        VERSION, _scene_cfg_path.name, len(cam_names), ", ".join(cam_names),
        _ds_root.name, _episode_ids,
    )

    # ----- 2. cameras + dynamic SceneCfg -----
    built = _build_cameras_from_config(scene_config)
    CylRoomSceneCfg = _make_recorder_scene_cfg(built)

    # ----- 3. pre-scan trajectory lengths + parse_env_cfg + Phase-4 overrides -----
    ep_n_frames: dict[int, int] = {}
    for ep_idx in _episode_ids:
        _arr = np.load(_ds_root / "trajectories" / f"episode_{ep_idx:03d}.npy", mmap_mode="r")
        ep_n_frames[ep_idx] = int(_arr.shape[0])
    max_n_frames = max(ep_n_frames.values())
    logger.info("Episode lengths: %s (max=%d, total=%d)",
                ep_n_frames, max_n_frames, sum(ep_n_frames.values()))

    env_cfg = parse_env_cfg(
        "LiftCube-Sim", device=args_cli.device,
        num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.scene = CylRoomSceneCfg()
    env_cfg.scene.num_envs = 1

    # --- Scene overrides ---
    # The base scene is authored natively in the cylroom, so nothing needs
    # re-anchoring for the shipped configs (RIG_SHIFT is zero). What remains:
    # CLI asset overrides, the historical --pad_z_offset semantics, and the
    # generic additive shift for a future foreign target scene.
    env_cfg.scene.room.spawn.usd_path = str(_shell_usd)
    env_cfg.scene.action_pad.spawn.usd_path = str(_pad_usd)
    # --pad_z_offset keeps its historical floor-frame meaning; expressed {W}-fixed
    # it lands at the same tabletop height in any scene.
    _pad_off = args_cli.pad_z_offset if args_cli.pad_z_offset is not None else 0.0218
    env_cfg.scene.action_pad.init_state.pos = (
        WS_ORIGIN[0], WS_ORIGIN[1],
        WS_ORIGIN[2] + (_pad_off - HEMISPHERE_ORIGIN_W[2]),
    )
    if RIG_SHIFT != (0.0, 0.0, 0.0):
        for _obj in (env_cfg.scene.robot, env_cfg.scene.cube):
            _p0 = _obj.init_state.pos
            _obj.init_state.pos = (
                _p0[0] + RIG_SHIFT[0], _p0[1] + RIG_SHIFT[1], _p0[2] + RIG_SHIFT[2],
            )
    # Reset-time spawn placeholder only: _place_cube_table sets the final cube pose
    # absolutely (recorded position + DATA_SHIFT) after apply, so this value never
    # enters the shift accounting and cannot double-shift. It sits at the workspace
    # frame's spawn height purely so env.reset's transient spawn is not inside the
    # table.
    env_cfg.scene.cube.init_state.pos = (
        WS_ORIGIN[0], WS_ORIGIN[1], WS_ORIGIN[2] + CUBE_SPAWN_Z_IN_W,
    )
    _mark(f"3. overrides: room->{_shell_usd.name}, robot base->{env_cfg.scene.robot.init_state.pos} yaw+90, "
          f"pad->{_pad_usd.name} @ {env_cfg.scene.action_pad.init_state.pos}")

    env_cfg.episode_length_s = (max_n_frames / FPS) + 60.0
    configure_randomization(env_cfg.events, groups=set())

    # ----- 4. gym.make -----
    _t = time.time()
    _mark("4. gym.make — SLOW (asset load + shader compile per TiledCamera)")
    env = gym.make("LiftCube-Sim", cfg=env_cfg)
    device = env.unwrapped.device
    _mark(f"4.OK env created  [{time.time() - _t:.1f}s]")

    scene_loader = SceneStateLoader(
        scene_state_dir=_ds_root / "meta", binmap_path=_binmap_path, device=str(device),
    )
    joint_mins, joint_maxs = build_joint_tensors(device)

    # ----- 5. initial reset + scene_apply + cube shift -----
    _mark(f"5. initial env.reset() + scene_apply + cube-shift (ep {_episode_ids[0]})")
    if args_cli.verify_placement:
        import csv as _csv
        _scene = env.unwrapped.scene
        wd = DATA_SHIFT
        rows = []
        with torch.inference_mode():
            for ep in _episode_ids:
                env.reset()
                res = scene_loader.apply(env, ep, cube_noise_m=0.0,
                                         cube_rot_noise_deg=0.0, visual_override=None)
                pc = res["placed_cube"]
                _place_cube_table(_scene, pc, DATA_SHIFT)
                for _ in range(30):  # settle
                    env.step(_scene["robot"].data.default_joint_pos.clone())
                a = _scene["cube"].data.root_state_w[0, 0:3].tolist()
                ex, ey, ez = pc["cube_x_m"] + wd[0], pc["cube_y_m"] + wd[1], SOURCE_CUBE_SPAWN_Z + wd[2]
                err = ((a[0] - ex) ** 2 + (a[1] - ey) ** 2) ** 0.5
                rows.append([_source_bin, ep, round(ex, 4), round(ey, 4), round(ez, 4),
                             round(a[0], 4), round(a[1], 4), round(a[2], 4), round(err, 4)])
        _csvp = Path(args_cli.verify_placement)
        _hdr = not _csvp.exists()
        with open(_csvp, "a", newline="") as f:
            w = _csv.writer(f)
            if _hdr:
                w.writerow(["bin", "ep", "exp_x", "exp_y", "exp_z", "act_x", "act_y", "act_z", "err_xy_m"])
            w.writerows(rows)
        logger.info("verify_placement: %d eps -> %s (max err_xy=%.4f m)",
                    len(rows), _csvp, max((r[8] for r in rows), default=0.0))
        os._exit(0)

    if args_cli.diag_cube:
        _scene = env.unwrapped.scene
        _df = open("/tmp/cube_diag_result.txt", "w")
        def _w(line):
            _df.write(line + "\n"); _df.flush(); os.fsync(_df.fileno())
        def _cp(tag):
            p = _scene["cube"].data.root_state_w[0, 0:3].tolist()
            _w(f"{tag:24s} cube_w = ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})")
            return p
        with torch.inference_mode():
            env.reset()
            _cp("after env.reset()")
            res = scene_loader.apply(env, _episode_ids[0], cube_noise_m=0.0,
                                     cube_rot_noise_deg=0.0, visual_override=None)
            pc = res.get("placed_cube", {})
            _w(f"apply placed_cube = {pc}")
            p_apply = _cp("after apply (pre-place)")
            _place_cube_table(_scene, pc, DATA_SHIFT)
            _cp("after _place_cube_table")
            for _ in range(30):
                env.step(env.unwrapped.scene["robot"].data.default_joint_pos.clone())
            _cp("after 30 settle steps")
        off = DATA_SHIFT
        cx, cy = pc.get("cube_x_m", float('nan')), pc.get("cube_y_m", float('nan'))
        _w(f"DATA_SHIFT = {off}")
        _w(f"EXPECTED final ~ placed({cx:.3f},{cy:.3f})+offset = ({cx+off[0]:.4f}, {cy+off[1]:.4f})")
        match = abs(p_apply[0]-cx) < 0.01 and abs(p_apply[1]-cy) < 0.01
        _w(f"VERDICT: post-apply read {'MATCHES' if match else 'does NOT match'} placed_cube"
           f" -> {'no stale-read' if match else 'STALE-READ confirmed'}")
        _df.close()
        env.close(); os._exit(0)
    with torch.inference_mode():
        env.reset()
        res = scene_loader.apply(env, _episode_ids[0], cube_noise_m=0.0,
                                 cube_rot_noise_deg=0.0, visual_override=None)
        _place_cube_table(env.unwrapped.scene, res["placed_cube"], DATA_SHIFT)
        _set_pad_yaw(env, 0.0)  # axis-aligned pad, overrides scene_state DR yaw

    # ----- 6. initial warmup -----
    _first_ep_actions = np.load(
        _ds_root / "trajectories" / f"episode_{_episode_ids[0]:03d}.npy"
    ).astype(np.float32)
    first_action_raw_t = torch.from_numpy(_first_ep_actions[0]).to(device)
    first_action_sim = raw_degrees_to_sim_radians(first_action_raw_t, joint_mins, joint_maxs)
    first_action = first_action_sim.unsqueeze(0).expand(args_cli.num_envs, -1).clone()

    _mark(f"6. initial warmup {WARMUP_STEPS} steps")
    with torch.inference_mode():
        for _ in range(WARMUP_STEPS):
            env.step(first_action)

    # ----- 7. external camera look-at (shifted to table center) -----
    ext_records = [rec for rec in built if rec.pose is not None]
    _mark(f"7. set look-at for {len(ext_records)} external cameras "
          f"(hemisphere_origin={WS_ORIGIN})")
    for rec in ext_records:
        eye = camera_grid.set_external_camera_pose(
            env, rec.pose, hemisphere_origin=WS_ORIGIN, camera_name=rec.field,
        )
        logger.info("  %-22s eye=(%+0.3f, %+0.3f, %+0.3f)",
                    rec.field, eye[0, 0].item(), eye[0, 1].item(), eye[0, 2].item())
    _mark(f"7b. prime render products ({PROPAGATION_STEPS} render steps)")
    with torch.inference_mode():
        for _ in range(PROPAGATION_STEPS):
            env.step(first_action)

    # ----- diagnostic: pad-height screenshot (no dataset / no replay) -----
    if args_cli.screenshot:
        _mark(f"SCREENSHOT mode: pad_z_offset={_pad_off} (pad-top world ~{_pad_off + RIG_SHIFT[2] + 0.002:.4f})")
        with torch.inference_mode():
            for _ in range(40):  # extra settle so the cube fully rests
                env.step(first_action)
        rgb = env.unwrapped.scene["camera_right"].data.output["rgb"][0].cpu().numpy()
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
        from PIL import Image
        Image.fromarray(rgb[..., :3]).save(args_cli.screenshot)
        logger.info("screenshot saved: %s", args_cli.screenshot)
        _mark("SCREENSHOT done — exiting")
        os._exit(0)  # OS reclaims VRAM; skip env.close() (Isaac6 close() can hang)

    # ----- diagnostic: GUI hold (scene loaded, arm at start pose, window open) -----
    if args_cli.gui_hold:
        _mark("GUI_HOLD: scene loaded — holding start pose; close the window to exit")
        with torch.inference_mode():
            while simulation_app.is_running():
                env.step(first_action)
        _mark("GUI_HOLD: window closed — exiting")
        os._exit(0)

    # ----- 8. LeRobotDataset create / open -----
    output_root = Path(args_cli.output_root).resolve()
    repo_id = f"local/{_config_stem}_cylroom_{_source_bin}"
    if args_cli.mode == "create":
        _mark("8. LeRobotDataset.create — streaming_encoding=True")
        ds = LeRobotDataset.create(
            repo_id=repo_id, root=output_root, fps=FPS,
            features=_build_features(scene_config), use_videos=True,
            streaming_encoding=True,
        )
        ds.meta.update_chunk_settings(video_files_size_in_mb=1)
        (output_root / "meta" / "scene_config.json").write_text(_scene_cfg_path.read_text())
    else:
        _mark(f"8. open existing LeRobotDataset (append {len(_episode_ids)} eps)")
        ds = LeRobotDataset(repo_id, root=output_root, streaming_encoding=True)
        logger.info("Existing: %d eps, %d frames", ds.meta.total_episodes, ds.meta.total_frames)

    # ----- 9. replay loop -----
    _mark(f"9. replay loop over {len(_episode_ids)} episodes")
    t_replay_start = time.time()
    scene = env.unwrapped.scene
    rc = 0
    frame_idx = -1
    ep_idx = -1
    per_ep: list[dict] = []
    try:
        with torch.inference_mode():
            for ep_pos, ep_idx in enumerate(_episode_ids):
                n_frames = ep_n_frames[ep_idx]
                logger.info("--- Episode %d/%d (ep_idx=%d, %d frames) ---",
                            ep_pos + 1, len(_episode_ids), ep_idx, n_frames)

                if ep_pos == 0:
                    actions = _first_ep_actions
                else:
                    actions = np.load(
                        _ds_root / "trajectories" / f"episode_{ep_idx:03d}.npy"
                    ).astype(np.float32)
                actions_raw_t = torch.from_numpy(actions).to(device)
                first_sim = raw_degrees_to_sim_radians(actions_raw_t[0], joint_mins, joint_maxs)
                first_action = first_sim.unsqueeze(0).expand(args_cli.num_envs, -1).clone()

                env.reset()
                res = scene_loader.apply(env, ep_idx, cube_noise_m=0.0,
                                         cube_rot_noise_deg=0.0, visual_override=None)
                _place_cube_table(scene, res["placed_cube"], DATA_SHIFT)
                _set_pad_yaw(env, 0.0)  # axis-aligned pad, overrides scene_state DR yaw

                for _ in range(SETTLE_STEPS):
                    env.step(first_action)

                t_loop = time.time()
                for frame_idx in range(n_frames):
                    raw = actions_raw_t[frame_idx]
                    sim_rad = raw_degrees_to_sim_radians(raw, joint_mins, joint_maxs)
                    act = sim_rad.unsqueeze(0).expand(args_cli.num_envs, -1).clone()
                    env.step(act)

                    joint_pos_rad = scene["robot"].data.joint_pos[0]
                    joint_pos_raw = sim_radians_to_raw_degrees(joint_pos_rad, joint_mins, joint_maxs)
                    frame = {
                        "observation.state": joint_pos_raw.cpu().numpy().astype(np.float32),
                        "action":            raw.cpu().numpy().astype(np.float32),
                        "task":              args_cli.task_name,
                    }
                    for cam_short in cam_names:
                        rgb = scene[f"camera_{cam_short}"].data.output["rgb"]
                        frame[f"observation.images.{cam_short}"] = rgb[0].cpu().numpy()
                    ds.add_frame(frame)

                    if frame_idx == 0 or (frame_idx + 1) % 30 == 0:
                        logger.info("  ep=%d frame %d/%d", ep_idx, frame_idx + 1, n_frames)
                replay_secs = time.time() - t_loop

                t_save = time.time()
                ds.save_episode()
                save_secs = time.time() - t_save
                per_ep.append({
                    "ep_idx": ep_idx, "n_frames": n_frames,
                    "replay_s": round(replay_secs, 3),
                    "replay_fps": round(n_frames / replay_secs, 2) if replay_secs > 0 else 0.0,
                    "save_s": round(save_secs, 3),
                })
                logger.info("  ep=%d done: replay=%.1fs (%.1f fps), save=%.2fs",
                            ep_idx, replay_secs,
                            n_frames / replay_secs if replay_secs > 0 else 0.0, save_secs)
    except KeyboardInterrupt:
        logger.error("Interrupted at ep_idx=%d frame=%d", ep_idx, frame_idx)
        rc = 130
    phases["replay_loop"] = time.time() - t_replay_start

    # ----- 10. finalize -----
    _mark("10. ds.finalize()")
    ds.finalize()
    logger.info("Dataset: %d eps, %d frames", ds.meta.total_episodes, ds.meta.total_frames)

    # ----- 11. close -----
    _mark("11. env.close()")
    env.close()
    phases["total"] = time.time() - _T_SCRIPT_START

    timing_json = {
        "version": VERSION, "scene_config": str(_scene_cfg_path),
        "rig_shift": RIG_SHIFT, "data_shift": DATA_SHIFT, "hemisphere_origin": WS_ORIGIN,
        "n_streams": len(cam_names), "n_episodes": len(_episode_ids),
        "episode_ids": _episode_ids, "per_episode": per_ep,
        "phase_seconds": {k: round(v, 3) for k, v in phases.items()},
        "total_minutes": round(phases["total"] / 60.0, 2),
    }
    (output_root / "timing.json").write_text(json.dumps(timing_json, indent=2))

    # Clean up leftover temp dirs (tmpXXXX/.space/context.mdb — an Isaac/Kit LMDB
    # cache created via tempfile under the dataset root during recording). We exit via
    # os._exit (to dodge the Isaac Sim 6 close() hang), which SKIPS Python's atexit/
    # tempfile finalizers, so these orphan dirs would otherwise leak into the dataset.
    import shutil as _shutil
    for _tmp in output_root.glob("tmp*"):
        if _tmp.is_dir():
            _shutil.rmtree(_tmp, ignore_errors=True)

    logger.info("DONE rc=%d total=%.1fs -> %s", rc, phases["total"], output_root)
    return rc


if __name__ == "__main__":
    try:
        rc = main()
    except KeyboardInterrupt:
        logger.info("Ctrl-C — exiting")
        rc = 130
    os._exit(rc)
