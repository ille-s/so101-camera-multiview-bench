#!/usr/bin/env python3
"""async_eval_sandbox.py — sandbox: async per-env episode refill vs batched lockstep.

Follow-up to the discarded multi-env-eval path (``tiledcamera_edge_artifact``). There
the speed K.O. was the synchronous lockstep in ``cylroom_eval.py``: a global
``env.reset()`` per episode + ``while not all(done)`` + force-terminate-ALL on any gym
truncation. One timeout episode blocks the 9 finished envs -> only ~2x instead of ~10x.

DEXTRAH (``third_party/DEXTRAH/dextrah_lab/distillation/eval.py``) avoids this: no global
reset, it refills only the finished envs (``all_done_indices``) while the others keep
running. This sandbox implements exactly that and measures, on the SAME 27 distinct
episodes with N envs, whether async refill beats batched lockstep — and (``--capture_frames``)
whether the tiled-render artifact shifts the success rate vs single-env.

The sandbox is STANDALONE (production ``cylroom_eval.py`` is untouched). Because
``cylroom_eval`` is an AppLauncher entry script (parses args + boots Isaac at import), its
setup helpers are copied here, not imported. The cube pose per episode comes purely from
``binmap.get_episode_config(ep)`` (no sim side effects) so a finished env can be refilled
with ``cube.write_root_state_to_sim(state, env_ids=[i])`` while the others run on.

Two modes (identical setup, only the loop differs):
  * ``--mode lockstep`` — batched over the 27 distinct episodes (each env a different episode
    per batch), global reset + ``while not all(done)`` barrier per batch = the faithful current
    production behaviour.
  * ``--mode async`` — ONE global reset, ``episode_length_s`` large (env time_out never fires,
    so we own all resets), per-env step budget + manual per-env reset + a settle phase that
    overlaps with the still-running envs (ONE global sim clock, no separate settle loop).

Usage:
    python async_eval_sandbox.py --mode async --num_envs 10 --num_episodes 27 \\
        --policy_path .../pretrained_model --dataset_root datasets/01_raw_gamepad/bin_c0_r0 \\
        --scene_config .../lift_cube_6cam.json --output_dir _runs/stage2_async

BATCH (campaign) mode — many (bin x seed) runs in ONE Isaac boot (one loaded policy); each run stays
a fully separate eval, written to output_dir/<bin>__seed<seed>/ (see
docs/plans/2026-06-12_holdout_campaign.md):
    python async_eval_sandbox.py --mode async --num_envs 10 --num_episodes 27 \\
        --policy_path .../pretrained_model \\
        --dataset_roots "datasets/01_raw_gamepad/bin_c0_r0,datasets/01_raw_gamepad/bin_c2_r2" \\
        --seeds "42,43,44" --full_report \\
        --scene_config .../lift_cube_6cam.json --output_dir _runs/holdout_campaign/<policy>

Version: see VERSION
"""

from __future__ import annotations

VERSION = "0.4.3"

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI — parsed BEFORE Isaac Sim boot
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Sandbox: async per-env episode refill vs batched lockstep sim-eval."
)
parser.add_argument("--mode", choices=["lockstep", "async"], required=True,
                    help="lockstep = batched barrier (current prod); async = per-env queue refill.")
parser.add_argument("--policy_path", type=Path, required=True,
                    help="Path to the pretrained_model/ directory of the checkpoint.")
parser.add_argument("--dataset_root", type=Path, default=None,
                    help="Single-run: source bin with meta/episode_*_scene_state.json + binmap.json. "
                         "Required unless --dataset_roots (batch) is given.")
parser.add_argument("--dataset_roots", type=str, default=None,
                    help="BATCH (campaign) mode: comma-separated bin dirs. Loops ALL (bin x --seeds) runs in "
                         "ONE Isaac boot (env+policy+cameras loaded once); each run stays a fully separate "
                         "eval (own reset(seed), results, report) under output_dir/<bin>__seed<seed>/. "
                         "Overrides --dataset_root/--seed.")
parser.add_argument("--seeds", type=str, default=None,
                    help="BATCH mode variant A: comma-separated seeds; each bin runs once per seed "
                         "(-> <bin>__seed<seed>/). Mutually exclusive with --num_runs.")
parser.add_argument("--num_runs", type=int, default=None,
                    help="BATCH mode variant B (= the established eval_2cam.sh n-repeat convention): repeat "
                         "each bin N times with the SAME --seed (default 42) -> <bin>__run<NN>/. The spread "
                         "over runs isolates pure CUDA/PhysX FP non-determinism (no seed variation). "
                         "Mutually exclusive with --seeds.")
parser.add_argument("--scene_config", type=Path, required=True,
                    help="Recording scene_config JSON (camera az/el/r) for external-cam aim.")
parser.add_argument("--task", type=str, default=None,
                    help="Gym env task id. Default: AUTO-select by the policy's external-camera count "
                         "(from rename_map): only `external` -> LiftCube-Sim; `external_2` present -> "
                         "LiftCube-Sim-2Ext; `external_3` -> LiftCube-Sim-3Ext. The env must provide a "
                         "camera slot per policy camera; the base LiftCube-Sim scene has only {ego, "
                         "external}, so 2-external / wrist+2-external policies need the 2Ext scene. "
                         "Set explicitly only to force a specific env.")
parser.add_argument("--output_dir", type=Path, required=True,
                    help="Directory for results_<mode>.csv + timing_<mode>.json.")
parser.add_argument("--num_envs", type=int, default=10,
                    help="Parallel environments (queue spreads episodes across them).")
parser.add_argument("--num_episodes", type=int, default=27,
                    help="Number of distinct episodes to evaluate (1-indexed 1..num_episodes).")
parser.add_argument("--seed", type=int, default=42, help="Base seed.")
parser.add_argument("--episode_length_s", type=float, default=20.0,
                    help="Per-episode time budget (caps the per-env step budget).")
# ── RTX/DLSS DENOISER-GHOST SETTLE — DEFAULTS RAISED 60 -> 130 (2026-06-20) ──────────────
# After a per-env teleport/reset the RTX/DLSS denoiser needs ~120 sim-steps to clear the
# post-teleport "two cubes" ghost. At settle=60 the ghost is still in the first ~2 captured
# frames AND the policy's FIRST OBSERVATION -> can shift the success rate. The GOLD cylroom_v2
# 2/3cam thesis evals ran ghost-free at 130 (run_holdout_campaign.sh SETTLE_STEPS default 130);
# a run at 60 is NOT comparable to them (caught 2026-06-20: a 6cam eval at the old 60 default
# was quarantined + re-run). So BOTH defaults are 130 now — every mode (async=settle_steps,
# lockstep=warmup_steps) starts ghost-free + GOLD-comparable out of the box, no caller has to
# remember to pass it. Lower ONLY for a deliberate timing experiment.
# (VERSION intentionally NOT bumped: default-value + doc change only, no harness-logic change.)
parser.add_argument("--warmup_steps", type=int, default=130,
                    help="Physics+render settle after the one-time global reset (= lockstep's per-batch "
                         "settle). Default raised 60->130 on 2026-06-20 (RTX/DLSS denoiser ghost, see "
                         "comment above).")
parser.add_argument("--settle_steps", type=int, default=130,
                    help="async: hold-steps after a per-env manual reset before the episode counts "
                         "(physics-settle AND render-settle). The RTX/DLSS denoiser needs ~120 sim-steps "
                         "to clear the post-teleport ghost ('two cubes'); 60 leaves it in the first ~2 "
                         "captured frames AND the policy's first observation. DEFAULT 130 = ghost-free "
                         "(raised 60->130 on 2026-06-20, see comment above). Externally settable via "
                         "run_holdout_campaign.sh SETTLE_STEPS. NOTE: lockstep uses --warmup_steps.")
parser.add_argument("--dump_joint_states", action=argparse.BooleanOptionalAction, default=True,
                    help="Dump per-frame joint trajectory per episode (collapse diagnosis).")
parser.add_argument("--verify_placement", action="store_true", default=False,
                    help="Stage-1: assert per-env cube readback == expected binmap pose (FAIL-LOUD).")
parser.add_argument("--capture_frames", action="store_true", default=False,
                    help="Stage-3: dump each env's first running frame per camera (native res).")
parser.add_argument("--record_video", action=argparse.BooleanOptionalAction, default=True,
                    help="Record env-0's per-episode rollouts (policy cams) as MP4 into "
                         "<output_dir>/videos/env0/episode_NNN/. DEFAULT ON (user rule 2026-06-13: eval runs "
                         "must always have video so results are verifiable). Pass --no-record_video ONLY for "
                         "pure timing experiments where capture overhead must be excluded.")
# ── SVT-AV1 MULTI-CAMERA DEADLOCK — DEFAULT RAISED 2 -> 15 (2026-06-20) ──────────────
# The per-episode video encode is SYNCHRONOUS in the main thread (see "SYNCHRONOUS
# encode" below). With many cameras (a 5-6cam policy writes 5-6 MP4s/episode) a low
# capture interval makes the encoder do huge per-episode work, and SVT-AV1 then
# DEADLOCKS at an episode/run transition: the whole eval freezes — process alive on
# the GPU, log frozen, no traceback — until a watchdog SIGKILLs it.
#   Observed 2026-06-20: 6cam eval, video_interval=2, hung at run 28/50 for ~2.5 h.
#   With interval=15 the same 6cam eval ran 50/50 with 0 hangs.
# Higher interval = fewer captured frames = ~Nx less encode work = no deadlock, and
# shorter clips (15 -> ~2-8 s timelapse vs interval-2's 18-60 s slow-mo monsters).
# LOWER THIS ONLY for <=2-cam policies, or knowingly accept the deadlock risk + run
# under a stall-watchdog (e.g. resume_eval_6cam_watchdog.sh).
parser.add_argument("--video_interval", type=int, default=15,
                    help="Capture every Nth step for video. LOWER = smoother + bigger + "
                         "MORE SVT-AV1 DEADLOCK RISK with many cameras (see comment above). "
                         "Default raised 2->15 on 2026-06-20 after a 6cam video deadlock.")
parser.add_argument("--record_all_envs", action="store_true", default=False,
                    help="Verification: record EVERY env's rollout (both lockstep AND async), not just "
                         "env-0. Robust to which episode the non-deterministic run fails. Writes "
                         "<output_dir>/videos/env<i>/episode_NNN/. Slow (per-env GPU->CPU copies) — "
                         "verification only, never timing runs.")
parser.add_argument("--full_report", action="store_true", default=False,
                    help="Production parity: also emit eval_summary.json + eval_results_random_none.csv "
                         "(production schema) + quality_gate.json via the production ResultsWriter, so the "
                         "run is eval_report-compatible and does the SAME finalize work as the single-env "
                         "production eval (fair async-vs-single-env comparison). Run `eval_report --dir "
                         "<output_dir>` afterwards for the 8 plots + report.md (separate CPU post-step, like "
                         "run_report.sh).")
parser.add_argument("--table_collision_margin_m", type=float, default=0.02,
                    help="Single-env parity (cylroom_eval): fail an episode 'table_collision' when the jaw "
                         "body stays below table_top+margin for --table_collision_frames frames "
                         "(into-table press = invalid grasp). 0 disables.")
parser.add_argument("--table_collision_frames", type=int, default=10,
                    help="Consecutive below-surface frames before 'table_collision' fires (cylroom_eval default 10).")
parser.add_argument("--temporal_ensemble", action=argparse.BooleanOptionalAction, default=False,
                    help="ACT temporal ensembling. Default OFF (phase-3/4 trained TE=None).")
parser.add_argument("--temporal_ensemble_coeff", type=float, default=0.01,
                    help="TE coefficient when --temporal_ensemble is set.")
parser.add_argument("--room_usd", type=str, default=None, help="Override room shell USD.")
parser.add_argument("--pad_usd", type=str, default=None, help="Override action_pad USD.")
parser.add_argument("--pad_z_offset", type=float, default=None, help="Override pad floor-z offset.")
parser.add_argument("--render_interval", type=int, default=3,
                    help="sim.render_interval (sim-steps between camera render passes). "
                         "DEFAULT 3 = 40 Hz render (integrated 2026-06-12 from the render-throttle A/B: "
                         "~1.14-1.21x speedup over 60 Hz with success parity; a buffer above the 30 FPS "
                         "training cadence, chosen over 30 Hz for render-rate margin). "
                         "Render rate (Sim-Time) = (1/sim.dt) / render_interval: 2->60 Hz (legacy/EnvCfg "
                         "default), 3->40 Hz (default here), 4->30 Hz (exact training cadence, marginally "
                         "faster). Throttles ONLY the GPU camera render; PhysX (120 Hz) and "
                         "env.step()/control (60 Hz) are unchanged. video_interval=2 stays frame-distinct "
                         "for render_interval<=4.")
parser.add_argument("--gui", action="store_true", default=False, help="Isaac Sim GUI (default headless).")
# Camera-importance augmentation (no-op unless --augment_cameras given). Parity with cylroom_eval.
parser.add_argument("--augment_cameras", type=str, default=None,
                    help="Comma-list of POLICY cam names to augment (e.g. 'wrist' or 'front,top'). "
                         "None = off. Single name = single-cam importance; multiple = leave-one-out.")
parser.add_argument("--augment", type=str, default="blackout",
                    choices=["blackout", "gauss", "blur", "invert", "contrast", "canny"],
                    help="Augmentation kind applied to the target cameras.")
parser.add_argument("--augment_strength", type=float, default=0.0,
                    help="gauss=sigma(0-1) · blur=odd kernel · contrast=factor · canny=upper thresh.")

AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True, enable_cameras=True)
args_cli = parser.parse_args()
if args_cli.gui:
    args_cli.headless = False

# Cheap pre-flight (before the slow Isaac Sim boot).
_scene_cfg_path = args_cli.scene_config
# Batch (campaign) vs single-run dispatch. Batch: loop (bin x seed) in ONE boot; each run stays a fully
# separate eval (see _eval_run). _ds_root/_meta_dir/_binmap_path stay as the FIRST bin for the pre-boot
# banner only — every run resolves its own local paths from its ds_root.
_batch_bins = [Path(b.strip()) for b in args_cli.dataset_roots.split(",") if b.strip()] if args_cli.dataset_roots else None
_batch_seeds = [int(s.strip()) for s in args_cli.seeds.split(",") if s.strip()] if args_cli.seeds else None
if args_cli.seeds and args_cli.num_runs:
    print("ERR: --seeds and --num_runs are mutually exclusive", file=sys.stderr)
    sys.exit(2)
if _batch_bins is not None and not (_batch_seeds or args_cli.num_runs):
    print("ERR: --dataset_roots requires --seeds or --num_runs", file=sys.stderr)
    sys.exit(2)
if _batch_bins is None and args_cli.dataset_root is None:
    print("ERR: need --dataset_root (single-run) or --dataset_roots + --seeds/--num_runs (batch)", file=sys.stderr)
    sys.exit(2)
_ds_root = _batch_bins[0] if _batch_bins is not None else args_cli.dataset_root
_binmap_path = _ds_root / "binmap.json"
_meta_dir = _ds_root / "meta"
# src/so101_mvbench/evaluation/ -> the package root is 4 parents up
# (.../so101_mvbench). Resolve the package assets relative to it.
# Shipped package data is resolved relative to __file__, never via SETTINGS
# (see so101_mvbench.settings). parents[1] is the package dir, so this also
# works from an installed wheel where no src/ layer exists.
_pkg_root = Path(__file__).resolve().parents[1]  # .../so101_mvbench
_assets_usd = _pkg_root / "assets" / "usd"
_shell_usd = Path(args_cli.room_usd) if args_cli.room_usd else (_assets_usd / "cylindrical_room_shell.usda")
_pad_usd = Path(args_cli.pad_usd) if args_cli.pad_usd else (_assets_usd / "action-pad-round.usda")
_preflight: list[tuple[Path, str]] = [
    (args_cli.policy_path, "policy_path"),
    (_scene_cfg_path, "scene_config"),
    (_shell_usd, "room shell USD"),
    (_pad_usd, "action_pad USD"),
]
for _b in (_batch_bins if _batch_bins is not None else [_ds_root]):
    _preflight.append((_b / "meta", f"{_b.name}/meta (scene_state jsons)"))
    _preflight.append((_b / "binmap.json", f"{_b.name}/binmap.json"))
for _p, _desc in _preflight:
    if not _p.exists():
        print(f"ERR: {_desc} not found: {_p}", file=sys.stderr)
        sys.exit(2)

# Scene geometry comes from so101_transforms (single source of truth); imported
# below with the other post-AppLauncher imports.
# Cube-settled criterion (= eval_session.warmup_until_settled SETTLE_THRESHOLD). After the fixed
# settle/warmup the cube vel must be below this, else the reset was not clean → WARN.
SETTLE_VEL_THRESHOLD: float = 0.001  # m/s

print(f">>> async_eval_sandbox v{VERSION} | mode={args_cli.mode} | num_envs={args_cli.num_envs}", flush=True)
print(f">>> policy={args_cli.policy_path} | bin={_ds_root.name}", flush=True)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Imports that require an active SimulationApp
# ---------------------------------------------------------------------------

import csv  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import so101_mvbench.tasks  # noqa: F401, E402  (gym registration)

from so101_mvbench.tasks.lift_cube_env_cfg import (  # noqa: E402
    _euler_to_quat_xyzw,
    configure_randomization,
)
from so101_mvbench.utils import camera_grid  # noqa: E402
from so101_mvbench.utils.so101_transforms import (  # noqa: E402
    BASE_SCENE_ORIGIN_W,
    CUBE_SPAWN_Z_IN_W,
    HEMISPHERE_ORIGIN_W,
    TABLE_TOP_Z,
    rigid_shift_from_recording,
)
from so101_mvbench.evaluation.eval_config import load_eval_config as _load_eval_config  # noqa: E402
from so101_mvbench.utils.scene_config import SceneConfig  # noqa: E402
from so101_mvbench.utils.bin_spawner import BinMap  # noqa: E402
from so101_mvbench.evaluation.scene_loader import SceneStateLoader  # noqa: E402
from so101_mvbench.evaluation.policy_wrapper import (  # noqa: E402
    SimLeRobotSO101Interface,
    rename_map_for_policy,
)
from so101_mvbench.evaluation.eval_tracker import LiftCubeEvalTracker  # noqa: E402
from so101_mvbench.evaluation.augmentation import (  # noqa: E402
    augment_image_frame, check_strength)
from so101_mvbench.evaluation.multienv_obs_utils import (  # noqa: E402
    slice_visual_obs,
)
from so101_mvbench.evaluation.video_recorder import (  # noqa: E402
    DEFAULT_FPS, DEFAULT_RESIZE, _encode_videos)
from so101_mvbench.logging import configure_root_logger, get_logger  # noqa: E402

logger = get_logger(__name__)
configure_root_logger()

# Scene geometry is driven by two declared origins: the scene config's
# workspace_origin_m is the pose-graph root {W} of the TARGET scene, and the
# binmap's workspace_origin_m is the root of the scene the data was RECORDED in
# (absent on legacy datasets = the floor-level table scene). The rigid shift for
# the recorded rig (arm, cube, pad, camera hemisphere) is the difference. Table
# recording evaluated in the cylroom reproduces the historical ARM_BASE_OFFSET;
# same scene on both sides yields a zero shift. All batch bins must agree on
# their frame -- mixing recordings from different scenes in one campaign would
# silently misplace every cube.
_scene_config = SceneConfig.from_json(_scene_cfg_path)
WS_ORIGIN: tuple[float, float, float] = tuple(_scene_config.workspace_origin_m)

def _binmap_origin(binmap_path: Path) -> tuple[float, float, float] | None:
    d = json.loads(binmap_path.read_text())
    o = d.get("workspace_origin_m")
    return tuple(o) if o is not None else None

_source_origins = {
    b.name: _binmap_origin(b / "binmap.json")
    for b in (_batch_bins if _batch_bins is not None else [_ds_root])
}
if len(set(_source_origins.values())) > 1:
    print(f"ERR: batch bins were recorded in different frames: {_source_origins}",
          file=sys.stderr)
    sys.exit(2)
_src = next(iter(_source_origins.values()))
SOURCE_ORIGIN: tuple[float, float, float] = _src if _src is not None else HEMISPHERE_ORIGIN_W
# Two DIFFERENT translations (see multicam_replay for the full story): RIG_SHIFT
# anchors the furniture of the target scene (base scene cfg is authored in the
# cylroom); DATA_SHIFT transports recorded coordinates from the source scene.
RIG_SHIFT: tuple[float, float, float] = rigid_shift_from_recording(WS_ORIGIN, BASE_SCENE_ORIGIN_W)
DATA_SHIFT: tuple[float, float, float] = rigid_shift_from_recording(WS_ORIGIN, SOURCE_ORIGIN)
# Cube spawn height is {W}-fixed; its world value in the SOURCE frame:
SOURCE_CUBE_SPAWN_Z: float = SOURCE_ORIGIN[2] + CUBE_SPAWN_Z_IN_W


# ---------------------------------------------------------------------------
# Scene + eval helpers (copied from cylroom_eval.py — keep in sync).
# ---------------------------------------------------------------------------

def _cube_pose_for_episode(binmap, episode_idx: int) -> dict:
    """Pure binmap lookup -> placed cube dict. No sim side effects (unlike scene_loader.apply)."""
    cfg = binmap.get_episode_config(episode_idx)
    if cfg is None:
        raise RuntimeError(f"BinMap has no entry for episode_idx={episode_idx}")
    return {"cube_x_m": cfg.x_m, "cube_y_m": cfg.y_m, "cube_yaw_rad": cfg.yaw_rad}


def _expected_cube_world_xyz(scene, placed_cube: dict, env_id: int) -> tuple[float, float, float]:
    eo = scene.env_origins
    return (
        eo[env_id, 0].item() + placed_cube["cube_x_m"] + DATA_SHIFT[0],
        eo[env_id, 1].item() + placed_cube["cube_y_m"] + DATA_SHIFT[1],
        eo[env_id, 2].item() + SOURCE_CUBE_SPAWN_Z + DATA_SHIFT[2],
    )


def _place_cube_env(scene, placed_cube: dict, env_id: int, device) -> None:
    """Per-env absolute cube write (env_ids=[i]) — the async-refill core.

    Mirrors cylroom_eval._place_cube_table but writes ONLY row env_id so the other
    (still-running) envs are untouched. Velocity rows (7:13) zeroed.
    """
    cube = scene["cube"]
    i = env_id
    ids = torch.tensor([i], device=device, dtype=torch.long)
    new_state = cube.data.root_state_w[i:i + 1].clone()  # (1, 13)
    ex, ey, ez = _expected_cube_world_xyz(scene, placed_cube, i)
    new_state[0, 0] = ex
    new_state[0, 1] = ey
    new_state[0, 2] = ez
    # BUGFIX 2026-06-14: write the per-episode yaw quaternion (was missing -> cube always landed at
    # yaw=0° from the _reset_idx identity, breaking 18/27 rotated episodes; see async_eval_validity AP).
    # Slots [3:7] = XYZW (Isaac Lab post-PR#4437), pure-Z yaw — BIT-IDENTICAL to apply_cube_pose
    # (bin_spawner.py): (qx=0, qy=0, qz=sin(yaw/2), qw=cos(yaw/2)).
    yaw = placed_cube["cube_yaw_rad"]
    new_state[0, 3] = 0.0
    new_state[0, 4] = 0.0
    new_state[0, 5] = math.sin(yaw / 2.0)
    new_state[0, 6] = math.cos(yaw / 2.0)
    new_state[0, 7:13] = 0.0
    cube.write_root_state_to_sim(new_state, env_ids=ids)
    cube.reset(env_ids=ids)  # clear external wrenches for env i (residual contact forces)


def _reset_robot_env(scene, default_joint_pos, env_id: int, device) -> None:
    """PROPER per-env articulation reset (mid-run, env_ids=[i] only).

    A raw ``write_joint_state_to_sim`` teleports the joints but leaves (a) the PD position
    *targets* on the previous episode's last commanded pose and (b) the actuator internal state +
    external wrenches — so the controller fights the teleport on the next step → ~55deg articulation
    explosion (Stage-1 smoke v0.1.0: 3/4 mid-run refills collapsed). A full env.reset() clears all of
    this; here we replicate that per-env: state + targets + ``Articulation.reset`` + flush.
    """
    robot = scene["robot"]
    ids = torch.tensor([env_id], device=device, dtype=torch.long)
    pos = default_joint_pos.unsqueeze(0)  # (1, n_joints)
    vel = torch.zeros_like(pos)
    robot.write_joint_state_to_sim(pos, vel, env_ids=ids)
    # Reset the PD targets to the new default (else the controller yanks toward the old target).
    robot.set_joint_position_target(pos, env_ids=ids)
    robot.set_joint_velocity_target(vel, env_ids=ids)
    robot.reset(env_ids=ids)        # clear actuator internal state + external wrenches for env i
    robot.write_data_to_sim()       # flush targets/wrench before the next env.step


def _cube_settled_vel(scene, env_id: int) -> float:
    """Cube linear-velocity magnitude (m/s) for env_id — settle-cleanliness check."""
    return torch.norm(scene["cube"].data.root_lin_vel_w[env_id]).item()


def _set_pad_yaw(env, yaw_rad: float = 0.0) -> None:
    import omni.usd
    from pxr import Gf, UsdGeom
    from so101_mvbench.utils.scene_builder import resolve_all_prim_paths

    stage = omni.usd.get_context().get_stage()
    quat = Gf.Quatd(math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0))
    for path in resolve_all_prim_paths(env, "action_pad"):
        prim = stage.GetPrimAtPath(path)
        for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
            if "orient" in op.GetOpName():
                op.Set(quat)
                break


def _discover_cameras(env) -> dict:
    cameras = {}
    for obj in env.unwrapped.scene.keys():
        if obj.startswith("camera_"):
            cfg = getattr(env.unwrapped.scene.cfg, obj)
            cameras[obj.replace("camera_", "")] = {"height": cfg.height, "width": cfg.width}
    return cameras


def _find_jaw_body_idx(robot) -> int:
    for i, name in enumerate(n.lower() for n in robot.body_names):
        if "jaw" in name:
            return i
    raise RuntimeError(f"no 'jaw' body in robot.body_names: {robot.body_names}")


def _aim_external_cameras(env, scene_config: SceneConfig, rename_map: dict) -> None:
    for env_key, policy_cam in rename_map.items():
        if env_key == "ego":
            continue
        if policy_cam not in scene_config.cameras:
            raise RuntimeError(f"policy external camera {policy_cam!r} not in scene_config")
        spec = scene_config.cameras[policy_cam]
        x, y, z = camera_grid.spherical_to_cartesian(
            spec.azimuth_deg, spec.elevation_deg, spec.radius_m, camera_grid.WORKSPACE_CENTER,
        )
        pose = camera_grid.CameraPose(
            id=policy_cam, azimuth_deg=spec.azimuth_deg, elevation_deg=spec.elevation_deg,
            radius_m=spec.radius_m, x=x, y=y, z=z,
        )
        up_world = getattr(spec, "up_world", None)
        camera_grid.set_external_camera_pose(
            env, pose, hemisphere_origin=WS_ORIGIN,
            camera_name=f"camera_{env_key}", up_world=up_world,
        )
        logger.info("  aimed camera_%s -> %s (az=%.0f el=%.0f r=%.2f up=%s)",
                    env_key, policy_cam, spec.azimuth_deg, spec.elevation_deg,
                    spec.radius_m, up_world)


def _build_policy_iface(idx: int, device, cameras: dict, rename_map: dict) -> SimLeRobotSO101Interface:
    pi = SimLeRobotSO101Interface(
        device=str(device), port=None, id=f"async_sandbox_{idx}",
        cameras=cameras, fps=30, kind="follower", rename_map=rename_map,
    )
    pi.init_device(visualize=False)
    if args_cli.temporal_ensemble:
        pi.make_policy(str(args_cli.policy_path), temporal_ensemble_coeff=args_cli.temporal_ensemble_coeff)
    else:
        pi.make_policy(str(args_cli.policy_path), force_disable_te=True)
    return pi


def _make_tracker(cube, robot, jaw_body_idx, env_id, eval_cfg) -> LiftCubeEvalTracker:
    """One fresh tracker seeded from env_id's CURRENT (settled) pose."""
    cube_xyz = cube.data.root_link_pos_w[env_id, :3]
    gripper_xyz = robot.data.body_link_pos_w[env_id, jaw_body_idx, :3]
    initial_dist = torch.norm(gripper_xyz - cube_xyz).item()
    return LiftCubeEvalTracker(
        cube_xyz[2].item(), initial_dist, eval_cfg["target_end_position_rad"],
        home_pose_tol_rad=eval_cfg["target_end_position_tolerance_rad"],
        plateau_steps=eval_cfg["plateau_steps"],
    )


# Camera-importance augmentation (ported from cylroom_eval). Optional: no-op unless
# --augment_cameras is set in main(). Reaches policy input AND frame dumps AND videos.
_AUGMENT_KEYS: list[str] = []
_AUGMENT_RNG = None


def _augment_obs(obs: dict) -> None:
    """Augment the target obs cameras in place (no-op when _AUGMENT_KEYS empty).

    augment_image_frame operates on ONE uint8 HWC numpy frame; bridge each
    (N, H, W, C) cuda obs tensor through numpy per env+camera and back."""
    if not _AUGMENT_KEYS:
        return
    for key in _AUGMENT_KEYS:
        t = obs["visual"][key]
        arr = t.cpu().numpy()
        for i in range(arr.shape[0]):
            arr[i] = augment_image_frame(arr[i], args_cli.augment, args_cli.augment_strength, _AUGMENT_RNG)
        obs["visual"][key] = torch.from_numpy(arr).to(t.device)


def _dump_env_frame(obs, out_dir: Path, episode_idx: int, env_id: int) -> None:
    """Dump env_id's current frame per camera (native res, no resize) for validity inspection."""
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    for cam_key, tensor in obs.get("visual", {}).items():
        cam = cam_key.replace("camera_", "", 1)
        arr = tensor[env_id].detach().cpu().numpy()
        if arr.dtype != np.uint8:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(arr).save(out_dir / f"ep{episode_idx + 1:03d}_env{env_id:02d}_{cam}.png")


def _dump_joint_states(joint_traj, joint_names, out_dir: Path, ep_idx: int, env_id: int) -> None:
    """Per-frame joint trajectory (rad) -> CSV for collapse diagnosis (healthy <~1deg/frame)."""
    if not joint_traj:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = joint_names or [f"joint{j}" for j in range(len(joint_traj[0]))]
    with open(out_dir / f"episode_{ep_idx + 1:03d}_env{env_id:02d}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", *cols])
        for fi, row in enumerate(joint_traj):
            w.writerow([fi, *(f"{v:.6f}" for v in row)])


# ---------------------------------------------------------------------------
# Result + CSV helpers
# ---------------------------------------------------------------------------

def _table_collision_step(scene, gripper_z: float, env_id: int, counters: list[int],
                          fired: list[bool], margin: float, frames: int) -> None:
    """cylroom_eval table_collision detector (single-env parity): jaw below table_top+margin for
    `frames` consecutive frames -> fired (= invalid into-table press). Mutates counters/fired in place."""
    if margin <= 0 or fired[env_id]:
        return
    table_z = scene.env_origins[env_id, 2].item() + TABLE_TOP_Z
    if gripper_z < table_z + margin:
        counters[env_id] += 1
    else:
        counters[env_id] = 0
    if counters[env_id] >= frames:
        fired[env_id] = True


def _result_row(tracker, ep_idx, env_id, steps, joint_traj, table_collision: bool = False) -> dict:
    max_joint_delta_deg = 0.0
    if len(joint_traj) > 1:
        jt = np.asarray(joint_traj, dtype=float)
        max_joint_delta_deg = float(np.degrees(np.abs(np.diff(jt, axis=0)).max()))
    row = dict(tracker.result)
    if table_collision:   # cylroom_eval: tc overrides the tracker verdict -> FAIL
        row["termination_reason"] = "table_collision"
        row["success"] = False
    row["episode"] = ep_idx + 1
    row["env_id"] = env_id
    row["steps"] = steps
    row["max_joint_delta_deg"] = round(max_joint_delta_deg, 2)
    return row


def _write_results(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_full_report(rows, out_dir, meta_dir, policy_path, success_rate, mode, num_envs,
                       n_requested, video_settings, nan_guard_triggered: int = 0,
                       task: str = "LiftCube-Sim") -> None:
    """Production-parity output: eval_summary.json + eval_results_random_none.csv + quality_gate.json.

    Reuses the production ``ResultsWriter`` so the run is ``eval_report``-compatible and performs the
    SAME finalize work as the single-env production eval (fair async-vs-single-env comparison). Cube
    position/yaw per episode come from the dataset's ``meta/episode_*_scene_state.json`` — the same
    source the production ``eval_policy`` reads ``object_init_pos_m`` from.
    """
    import math

    from so101_mvbench.evaluation.results_writer import ResultsWriter

    # cube_yaw from binmap.json (authoritative — same source as cylroom_eval's placed_cube["cube_yaw_rad"]
    # and eval_report's _load_binmap_yaw). NOT scene_state object_init_rpy_rad.yaw: that can be XYZW/WXYZ-
    # corrupted in pre-v1.1.0 JSONs, and eval_report does NOT re-derive the column for our CSV (its binmap
    # overwrite early-returns because we write training_cube_*). scene_state is only a fallback.
    bmp = meta_dir.parent / "binmap.json"
    yaw_by_idx: dict[int, float] = {}
    if bmp.is_file():
        bm = json.loads(bmp.read_text())
        yaw_by_idx = {int(e["episode_idx"]): float(e["yaw_rad"]) for e in bm.get("episodes", [])
                      if e.get("episode_idx") is not None and e.get("yaw_rad") is not None}
    aug: list[dict] = []
    for r in rows:
        ep = int(r["episode"]) - 1
        ssf = meta_dir / f"episode_{ep:06d}_scene_state.json"
        ss = json.loads(ssf.read_text()) if ssf.is_file() else {}
        pos = ss.get("object_init_pos_m", {})
        yaw = math.degrees(yaw_by_idx.get(ep, ss.get("object_init_rpy_rad", {}).get("yaw", 0.0)))
        rr = dict(r)
        rr["run"] = 1                                            # cylroom_eval uses run=1
        rr["scene_state_idx"] = ep
        rr["cube_x_m"] = round(pos.get("x", 0.0), 4)
        rr["cube_y_m"] = round(pos.get("y", 0.0), 4)
        rr["cube_yaw_deg"] = round(yaw, 1)
        rr["training_cube_x_m"] = round(pos.get("x", 0.0), 4)
        rr["training_cube_y_m"] = round(pos.get("y", 0.0), 4)
        aug.append(rr)

    cfg = {
        # task = literal gym-task id (display-only field). NB cylroom_eval labels its writer-cfg task
        # "cylroom" (a scene-identity label, NOT the gym id); user chose the honest gym id here.
        "policy_path": str(policy_path), "task": task,
        "randomize": "none", "cube_noise": 0.0, "cube_rot_noise": 0.0,
        "scene_state": str(meta_dir), "num_runs": 1, "num_episodes_per_run": len(aug),
    }
    ResultsWriter(out_dir, cfg).write(aug, run_success_rates=[success_rate], video_settings=video_settings)
    (out_dir / "quality_gate.json").write_text(json.dumps({   # cylroom_eval schema (single-env parity)
        "num_envs": num_envs,
        "num_episodes_requested": n_requested,
        "num_episodes_completed": len(aug),
        "num_results": len(aug),
        "nan_guard_triggered": nan_guard_triggered,   # sandbox zeroes+continues (cylroom aborts) — count = zeroing events
        "success_rate_pct": round(success_rate, 2),
        "bin": meta_dir.parent.name,
        "policy_path": str(policy_path),
        # Camera-importance provenance (read by aggregate_cam_importance.py).
        "augment_cameras": args_cli.augment_cameras or "none",
        "augment_kind": args_cli.augment if args_cli.augment_cameras else "none",
        "augment_strength": args_cli.augment_strength if args_cli.augment_cameras else 0.0,
    }, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    device = torch.device(args_cli.device if hasattr(args_cli, "device") else "cuda:0")
    num_envs = args_cli.num_envs
    if num_envs < 1:
        raise ValueError(f"--num_envs must be >=1, got {num_envs}")

    # 1. Resolve the policy's cameras + AUTO-SELECT the env task by external-camera count.
    # The env must expose a camera slot for every rename_map key (ego + external/external_2/...). The base
    # LiftCube-Sim scene has only {ego, external}, so a 2-external (or wrist+2-external) policy needs the
    # 2Ext scene. rename_map only reads the policy's config.json (no env), so we pick the task FIRST.
    rename_map = rename_map_for_policy(args_cli.policy_path)

    # Camera-importance augmentation: resolve the SET of obs camera keys to augment from the
    # policy cam names (single-cam OR leave-one-out). No-op when --augment_cameras is unset.
    global _AUGMENT_KEYS, _AUGMENT_RNG
    if args_cli.augment_cameras is not None:
        env_key_by_policy_cam = {v: k for k, v in rename_map.items()}
        req = [c.strip() for c in args_cli.augment_cameras.split(",") if c.strip()]
        bad = [c for c in req if c not in env_key_by_policy_cam]
        if bad:
            raise RuntimeError(
                f"--augment_cameras {bad} not cameras of this policy: "
                f"{sorted(env_key_by_policy_cam)} (rename_map={rename_map})")
        check_strength(args_cli.augment, args_cli.augment_strength)
        _AUGMENT_KEYS = [f"camera_{env_key_by_policy_cam[c]}" for c in req]
        _AUGMENT_RNG = np.random.default_rng(args_cli.seed)
        logger.warning("AUGMENT ON: policy cams %s -> obs keys %s | kind=%s strength=%s "
                       "(reaches policy input, frame dumps AND videos)",
                       req, _AUGMENT_KEYS, args_cli.augment, args_cli.augment_strength)

    if args_cli.task:
        task = args_cli.task
    elif "external_3" in rename_map:
        task = "LiftCube-Sim-3Ext"
    elif "external_2" in rename_map:
        task = "LiftCube-Sim-2Ext"
    else:
        task = "LiftCube-Sim"
    logger.info("Task %s (%s) for policy %s | rename_map=%s",
                task, "explicit" if args_cli.task else "auto", args_cli.policy_path.name, rename_map)

    # 1b. Env cfg (mirror cylroom_eval) ------------------------------------------------
    env_cfg = parse_env_cfg(task, device=str(device), num_envs=1, use_fabric=True)
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.physx.enable_enhanced_determinism = True
    env_cfg.scene.replicate_physics = False
    env_cfg.scene.env_spacing = 8.0
    env_cfg.episode_length_s = args_cli.episode_length_s

    # Render-rate throttle: render the cameras below control rate (training is 30 FPS Sim-Time; the EnvCfg
    # default render_interval=decimation=2 renders at 60 Hz = 2x oversampled). DEFAULT here is now 3 = 40 Hz
    # (integrated from the render-throttle A/B); pass --render_interval 2 for legacy 60 Hz. PhysX / control
    # rate are untouched -- only GPU render passes drop. See AP 2026-06-12_cylroom_render_throttle.md.
    if args_cli.render_interval is not None:
        if args_cli.render_interval < 1:
            raise ValueError(f"--render_interval must be >=1, got {args_cli.render_interval}")
        env_cfg.sim.render_interval = args_cli.render_interval
    render_hz = (1.0 / env_cfg.sim.dt) / env_cfg.sim.render_interval
    logger.info("Render: render_interval=%d -> %.1f Hz Sim-Time (decimation=%d, sim.dt=1/%.0f; "
                "PhysX/control unchanged)", env_cfg.sim.render_interval, render_hz,
                env_cfg.decimation, 1.0 / env_cfg.sim.dt)

    # Scene overrides: base scene is authored natively in the cylroom (RIG_SHIFT
    # zero for the shipped configs); see multicam_replay for the full story.
    env_cfg.scene.room.spawn.usd_path = str(_shell_usd)
    env_cfg.scene.action_pad.spawn.usd_path = str(_pad_usd)
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
    # The task variant is what selects the randomization: LiftCube-Sim leaves
    # randomization_groups empty, the OOD variants fill it. Hard-coding an empty
    # set here made every -OOD- task render identically to the baseline.
    configure_randomization(env_cfg.events,
                            groups=set(env_cfg.randomization_groups))

    args_cli.output_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(task, cfg=env_cfg)
    env.unwrapped.cfg.episode_length_s = args_cli.episode_length_s
    max_steps_per_episode = env.unwrapped.max_episode_length
    logger.info("sandbox v%s | mode=%s | N=%d | bin=%s | budget=%d steps/episode",
                VERSION, args_cli.mode, num_envs, _ds_root.name, max_steps_per_episode)

    # 2. Cameras + policies + loaders (rename_map computed above for the task auto-select).
    env_cams = _discover_cameras(env)
    cameras = {k: env_cams[k] for k in rename_map if k in env_cams}
    missing = [k for k in rename_map if k not in env_cams]
    if missing:
        raise RuntimeError(f"rename_map keys {missing} missing in env cameras {list(env_cams)}")
    logger.info("Policy cameras: %s | rename_map: %s", list(cameras.keys()), rename_map)

    policy_ifaces = [_build_policy_iface(i, device, cameras, rename_map) for i in range(num_envs)]
    scene_config = _scene_config  # loaded once at module level (drives the geometry)

    robot = env.unwrapped.scene["robot"]
    cube = env.unwrapped.scene["cube"]
    scene = env.unwrapped.scene
    jaw_body_idx = _find_jaw_body_idx(robot)
    eval_cfg = _load_eval_config()
    default_joint_pos = robot.data.default_joint_pos[0]
    default_action = default_joint_pos.unsqueeze(0).expand(num_envs, -1).contiguous()

    # Everything above is bin/seed-INDEPENDENT (one Isaac boot, one loaded policy, aimed cameras come
    # per-run). Dispatch: batch (campaign) loops (bin x seed) through _eval_run in THIS one boot — each
    # run stays a fully separate eval (own reset(seed), scene_states, results, report). Single-run is
    # one _eval_run call with the legacy args (behaviour unchanged).
    shared = dict(env=env, scene=scene, robot=robot, cube=cube, jaw_body_idx=jaw_body_idx,
                  eval_cfg=eval_cfg, policy_ifaces=policy_ifaces, rename_map=rename_map,
                  scene_config=scene_config, num_envs=num_envs, default_joint_pos=default_joint_pos,
                  default_action=default_action, device=device,
                  max_steps_per_episode=max_steps_per_episode, env_cfg=env_cfg, render_hz=render_hz,
                  task=task)
    if _batch_bins is not None:
        # Variant A (--seeds): one run per (bin, seed) -> <bin>__seed<seed>/.
        # Variant B (--num_runs, = eval_2cam.sh convention): N repeats per bin with the SAME seed,
        # laid out per the established outputs/ convention (runs live NEXT TO the policy checkpoints):
        # <output_dir>/eval_cylroom_async_<bin>/run<NN>/ — DISTINCT from the legacy single-env
        # eval_cylroom_<bin>/ dirs (different eval engine: async N=10 @40Hz; never mix/overwrite those).
        # Spread over the runs isolates pure FP non-determinism.
        if args_cli.num_runs:
            jobs = [(b, args_cli.seed, f"eval_cylroom_async_{b.name}/run{r:02d}", r)
                    for b in _batch_bins for r in range(1, args_cli.num_runs + 1)]
            logger.info("BATCH: %d bins x %d runs (SAME seed %d) = %d separate runs in ONE boot",
                        len(_batch_bins), args_cli.num_runs, args_cli.seed, len(jobs))
        else:
            jobs = [(b, s, f"{b.name}__seed{s}", 1) for b in _batch_bins for s in _batch_seeds]
            logger.info("BATCH: %d bins x %d seeds = %d separate runs in ONE boot",
                        len(_batch_bins), len(_batch_seeds), len(jobs))
        for job_i, (ds_root, seed, run_name, run_idx) in enumerate(jobs, start=1):
            run_dir = args_cli.output_dir / run_name
            if (run_dir / f"timing_{args_cli.mode}.json").exists():
                logger.info("=== BATCH run %d/%d: %s — already done, SKIP (resume) ===",
                            job_i, len(jobs), run_name)
                continue
            logger.info("=== BATCH run %d/%d: %s (bin=%s seed=%d) ===",
                        job_i, len(jobs), run_name, ds_root.name, seed)
            _eval_run(ds_root=ds_root, seed=seed, out_dir=run_dir, run_idx=run_idx, **shared)
    else:
        _eval_run(ds_root=args_cli.dataset_root, seed=args_cli.seed, out_dir=args_cli.output_dir,
                  run_idx=1, **shared)
    return 0


def _eval_run(*, env, scene, robot, cube, jaw_body_idx, eval_cfg, policy_ifaces, rename_map, scene_config,
              num_envs, default_joint_pos, default_action, device, max_steps_per_episode, env_cfg,
              render_hz, task: str, ds_root: Path, seed: int, out_dir: Path, run_idx: int = 1) -> None:
    """ONE fully separate eval on (ds_root, seed) -> out_dir, reusing the booted env + loaded policy.

    This is the former per-run body of main(), verbatim but parameterised: local scene_loader/binmap
    from THIS bin (no module-global reassignment), own env.reset(seed), own warmup + camera re-aim,
    own results/timing/full_report. Nothing is shared between runs except the boot + policy weights
    (policy/processors are reset per episode inside the run loops)."""
    meta_dir = ds_root / "meta"
    binmap_path = ds_root / "binmap.json"
    scene_loader = SceneStateLoader(meta_dir, binmap_path, str(device))
    binmap = BinMap.load(binmap_path)

    # Resolve episode list (0-indexed).
    if scene_loader.count < args_cli.num_episodes:
        raise RuntimeError(f"{ds_root.name}: scene_loader has {scene_loader.count} eps, "
                           f"need {args_cli.num_episodes}")
    ep_idx_list = list(range(args_cli.num_episodes))
    logger.info("Evaluating %d distinct episodes in %s mode (bin=%s seed=%d)",
                len(ep_idx_list), args_cli.mode, ds_root.name, seed)

    # Per-run setup: global reset(seed) + warmup + camera aim + static visuals (ep0).
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    with torch.inference_mode():
        obs, _ = env.reset(seed=seed)
        for _ in range(args_cli.warmup_steps):
            obs = env.step(default_action)[0]
        _aim_external_cameras(env, scene_config, rename_map)
        for _ in range(20):  # prime render products after camera pose-set
            obs = env.step(default_action)[0]
        # Static visuals once (dome light etc. are constant across episodes in cylroom).
        scene_loader.apply(env, ep_idx_list[0],
                           cube_noise_m=env_cfg.cube_noise_m,
                           cube_rot_noise_deg=env_cfg.cube_rot_noise_deg)
        _set_pad_yaw(env, 0.0)
        for _ in range(args_cli.warmup_steps):
            obs = env.step(default_action)[0]

    results: list[dict] = []
    # Per-env busy intervals in GLOBAL sim steps -> gantt (settling+running = busy; idle = gap).
    busy_intervals: list[list[list[int]]] = [[] for _ in range(num_envs)]

    # Per-env video recorders (both modes). record_all_envs -> every env (verification: robust to
    # which episode the non-deterministic run fails); record_video alone -> env-0 only (demo).
    rec_envs = (set(range(num_envs)) if args_cli.record_all_envs
                else ({0} if args_cli.record_video else set()))
    video_recorders = [
        _SyncVideoRecorder(base_dir=(out_dir / "videos" / f"env{i}") if i in rec_envs else None,
                           policy_cams=list(rename_map.keys()), interval=args_cli.video_interval)
        for i in range(num_envs)
    ]
    if rec_envs:
        logger.info("Video recording enabled for envs %s (mode=%s) — SYNCHRONOUS encode (no threads)",
                    sorted(rec_envs), args_cli.mode)

    t0 = time.perf_counter()
    nan_count = [0]   # NaN/Inf-action events that got zeroed (cylroom_eval-parity quality_gate field)
    with torch.inference_mode():
        if args_cli.mode == "async":
            global_step = _run_async(
                env, scene, robot, cube, jaw_body_idx, eval_cfg, policy_ifaces, binmap,
                ep_idx_list, num_envs, default_joint_pos, default_action, device,
                max_steps_per_episode, results, busy_intervals, frames_dir, obs, video_recorders,
                nan_count, out_dir,
            )
        else:
            global_step = _run_lockstep(
                env, scene, robot, cube, jaw_body_idx, eval_cfg, policy_ifaces, binmap,
                scene_loader, ep_idx_list, num_envs, default_joint_pos, default_action, device,
                max_steps_per_episode, results, busy_intervals, frames_dir, video_recorders,
                nan_count, out_dir,
            )
    wallclock = time.perf_counter() - t0
    for vr in video_recorders:
        if vr.enabled:
            vr.wait()

    # 4. Finalize.
    n_succ = sum(1 for r in results if r.get("success"))
    success_rate = 100.0 * n_succ / max(len(results), 1)
    eps_per_s = len(results) / wallclock if wallclock > 0 else 0.0
    _write_results(results, out_dir / f"results_{args_cli.mode}.csv")

    # Production-parity finalize (eval_summary + eval_results CSV + quality_gate) — same finalize work
    # as the single-env production eval, so an async run is a fair drop-in comparison + eval_report-ready.
    output_write_s = 0.0
    if args_cli.full_report:
        _ow0 = time.perf_counter()
        video_settings = {"record_all_envs": args_cli.record_all_envs,
                          "video_interval": args_cli.video_interval}
        _write_full_report(results, out_dir, meta_dir, args_cli.policy_path,
                          success_rate, args_cli.mode, num_envs, len(ep_idx_list), video_settings,
                          nan_guard_triggered=nan_count[0], task=task)
        output_write_s = time.perf_counter() - _ow0
        logger.info("full_report: eval_summary.json + eval_results_random_none.csv + quality_gate.json in "
                    "%.3fs — run `eval_report --dir %s` for the 8 plots + report.md",
                    output_write_s, out_dir)

    timing = {
        "version": VERSION,
        "mode": args_cli.mode,
        "num_envs": num_envs,
        "bin": ds_root.name,
        "seed": seed,
        "run": run_idx,
        "num_episodes": len(ep_idx_list),
        "wallclock_s": round(wallclock, 2),
        "output_write_s": round(output_write_s, 3),
        "wallclock_total_s": round(wallclock + output_write_s, 2),
        "full_report": bool(args_cli.full_report),
        "render_interval": env_cfg.sim.render_interval,
        "render_hz": round(render_hz, 1),
        "global_steps": global_step,
        "eps_per_s": round(eps_per_s, 4),
        "success_rate_pct": round(success_rate, 1),
        "n_success": n_succ,
        "max_steps_per_episode": max_steps_per_episode,
        "per_env_busy_intervals": busy_intervals,
        "per_env_episode_count": [len(b) for b in busy_intervals],
    }
    (out_dir / f"timing_{args_cli.mode}.json").write_text(json.dumps(timing, indent=2))
    logger.info("DONE bin=%s seed=%d mode=%s N=%d | %d eps in %.1fs = %.3f eps/s | success %.1f%% (%d/%d)",
                ds_root.name, seed, args_cli.mode, num_envs, len(results), wallclock, eps_per_s,
                success_rate, n_succ, len(results))
    print(f">>> RESULT bin={ds_root.name} seed={seed} mode={args_cli.mode} N={num_envs} "
          f"eps_per_s={eps_per_s:.3f} wallclock={wallclock:.1f}s success={success_rate:.1f}%", flush=True)


def _infer_env(policy_ifaces, obs, i):
    """Serial per-env inference (Risk-K-1 visual slice). Returns (n_joints,) action tensor."""
    frame = policy_ifaces[i].sim_obs_to_policy_processor(
        obs["policy"][i].clone(), slice_visual_obs(obs["visual"], i)
    )
    av = policy_ifaces[i].predict_action(frame)
    return policy_ifaces[i].prediction_to_sim_processor(av, frame, log=False).reshape(-1)


class _SyncVideoRecorder:
    """Crash-safe per-env video recorder: encodes SYNCHRONOUSLY (no ThreadPoolExecutor).

    Why synchronous: the all-env path with one ThreadPoolExecutor per env (10 concurrent encode
    threads) tripped Isaac Sim's carb tasking mutex (Mutex.cpp:103 "Recursion not allowed") and
    crashed the session 2026-06-10. `_encode_videos` is pure imageio/PIL (no omni/carb), so the
    fault was purely the background-thread concurrency during sim stepping. Encoding inline at the
    episode barrier removes ALL background threads -> no carb contention. Same interface as the
    production VideoRecorder (`enabled` / `capture` / `flush` / `wait`) so call sites are unchanged.

    ⚠️ DEADLOCK CAVEAT (many cameras + low --video_interval): because this encode is SYNCHRONOUS,
    a 5-6cam policy (5-6 MP4s/episode) at a small video_interval makes SVT-AV1 do huge per-episode
    work and can DEADLOCK at an episode/run transition — the eval freezes (process alive on GPU,
    log frozen, no traceback). See the --video_interval argparse comment; its default was raised
    2->15 on 2026-06-20 after a 6cam eval hung 2.5 h at run 28/50.
    """

    def __init__(self, base_dir, policy_cams, interval, resize=DEFAULT_RESIZE, fps=DEFAULT_FPS):
        self._base_dir = base_dir
        self._policy_cams = set(policy_cams) if policy_cams else None
        self._interval = interval
        self._resize = resize
        self._fps = fps
        self._buffers: dict[str, list] = {}

    @property
    def enabled(self) -> bool:
        return self._base_dir is not None

    def capture(self, obs: dict, step: int) -> None:
        if not self.enabled or step % self._interval != 0:
            return
        for cam_key, img_tensor in obs.get("visual", {}).items():
            cam_name = cam_key.replace("camera_", "", 1)
            if self._policy_cams is not None and cam_name not in self._policy_cams:
                continue
            img = img_tensor[0].detach().cpu().numpy()
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            self._buffers.setdefault(cam_name, []).append(img)

    def flush(self, ep_dir) -> None:
        if not any(self._buffers.values()):
            return
        ep_dir.mkdir(parents=True, exist_ok=True)
        _encode_videos(self._buffers, ep_dir, self._resize, self._fps)  # SYNCHRONOUS — no thread
        self._buffers = {}

    def wait(self) -> None:
        pass  # synchronous — nothing pending


def _slice_obs_env(obs: dict, i: int) -> dict:
    """View of obs with only env i, kept as a length-1 batch so the recorder (which slices [0])
    records env i. Used to record any/all envs without touching the production VideoRecorder."""
    vis = obs.get("visual", {})
    return {"visual": {cam: t[i:i + 1] for cam, t in vis.items()}}


def _run_async(env, scene, robot, cube, jaw_body_idx, eval_cfg, policy_ifaces, binmap,
               ep_idx_list, num_envs, default_joint_pos, default_action, device,
               max_steps_per_episode, results, busy_intervals, frames_dir, obs, video_recorders,
               nan_count, out_dir) -> int:
    """Queue-refill: ONE global reset, per-env step budget + manual reset + overlapping settle."""
    queue = deque(ep_idx_list)
    # Per-env state. phase: 'idle' | 'settling' | 'running'.
    phase = ["idle"] * num_envs
    settle_left = [0] * num_envs
    assigned_ep = [-1] * num_envs
    step_count = [0] * num_envs
    env_start_step = [0] * num_envs
    trackers: list[LiftCubeEvalTracker | None] = [None] * num_envs
    joint_traj: list[list[list[float]]] = [[] for _ in range(num_envs)]
    actions = default_action.clone()
    tc_margin = args_cli.table_collision_margin_m
    tc_frames = max(1, args_cli.table_collision_frames)
    tc_counters = [0] * num_envs
    tc_fired = [False] * num_envs

    def assign(i: int, global_step: int) -> None:
        ep = queue.popleft()
        placed = _cube_pose_for_episode(binmap, ep)
        # PROPER per-env reset: scene.reset + event-manager reset + obs/action/reward manager resets
        # for env i — the SANCTIONED path step() auto-calls for terminated envs (line 221), which a
        # raw write_joint_state_to_sim skips. Then override the cube/robot to THIS episode's state.
        ids = torch.tensor([i], device=device, dtype=torch.long)
        env.unwrapped._reset_idx(ids)
        _place_cube_env(scene, placed, i, device)
        _reset_robot_env(scene, default_joint_pos, i, device)
        policy_ifaces[i].policy.reset()
        policy_ifaces[i].preprocessor.reset()
        policy_ifaces[i].postprocessor.reset()
        if args_cli.verify_placement:
            _verify_placement(scene, placed, i)
        assigned_ep[i] = ep
        phase[i] = "settling"
        settle_left[i] = args_cli.settle_steps
        step_count[i] = 0
        tc_counters[i] = 0
        tc_fired[i] = False
        joint_traj[i] = []
        env_start_step[i] = global_step
        actions[i] = default_joint_pos
        logger.info("  [assign] env%02d <- ep%03d (queue left=%d)", i, ep + 1, len(queue))

    global_step = 0
    for i in range(num_envs):
        if queue:
            assign(i, global_step)

    safety_cap = len(ep_idx_list) * (max_steps_per_episode + args_cli.settle_steps) + 1000
    _augment_obs(obs)  # initial post-warmup obs feeds the first inference
    while any(p in ("settling", "running") for p in phase) and global_step < safety_cap:
        for i in range(num_envs):
            if phase[i] == "running":
                actions[i] = _infer_env(policy_ifaces, obs, i)
            else:
                actions[i] = default_joint_pos
        if torch.isnan(actions).any() or torch.isinf(actions).any():
            nan_count[0] += 1
            logger.error("NaN/Inf action at global step %d — zeroing offenders", global_step)
            actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        obs = env.step(actions)[0]
        _augment_obs(obs)  # before next inference AND dump/video this iteration
        global_step += 1

        for i in range(num_envs):
            if phase[i] == "settling":
                settle_left[i] -= 1
                if settle_left[i] <= 0:
                    v = _cube_settled_vel(scene, i)
                    if v > SETTLE_VEL_THRESHOLD:
                        logger.warning("  env%02d cube NOT settled after %d steps (vel=%.4f > %.3f m/s) "
                                       "— increase --settle_steps", i, args_cli.settle_steps, v, SETTLE_VEL_THRESHOLD)
                    trackers[i] = _make_tracker(cube, robot, jaw_body_idx, i, eval_cfg)
                    phase[i] = "running"
                    step_count[i] = 0
                    env_start_step[i] = global_step
                    if args_cli.capture_frames:
                        _dump_env_frame(obs, frames_dir, assigned_ep[i], i)
            elif phase[i] == "running":
                step_count[i] += 1
                cube_xyz = cube.data.root_link_pos_w[i, :3]
                gripper_xyz = robot.data.body_link_pos_w[i, jaw_body_idx, :3]
                dist_3d = torch.norm(gripper_xyz - cube_xyz).item()
                aj = robot.data.joint_pos[i].cpu().tolist()
                joint_traj[i].append(aj)
                trackers[i].update(cube_xyz[2].item(), dist_3d, step_count[i], arm_joint_pos=aj)
                _table_collision_step(scene, gripper_xyz[2].item(), i, tc_counters, tc_fired, tc_margin, tc_frames)
                if video_recorders[i].enabled:
                    video_recorders[i].capture(_slice_obs_env(obs, i), step_count[i])
                if trackers[i].should_terminate or tc_fired[i] or step_count[i] >= max_steps_per_episode:
                    results.append(_result_row(trackers[i], assigned_ep[i], i, step_count[i], joint_traj[i],
                                               table_collision=tc_fired[i]))
                    busy_intervals[i].append([env_start_step[i], global_step])
                    if args_cli.dump_joint_states:
                        _dump_joint_states(joint_traj[i], getattr(robot, "joint_names", None),
                                           out_dir / "joint_states", assigned_ep[i], i)
                    if video_recorders[i].enabled:
                        video_recorders[i].flush(out_dir / "videos" / f"env{i}"
                                                 / f"episode_{assigned_ep[i] + 1:03d}")
                    if queue:
                        assign(i, global_step)
                    else:
                        phase[i] = "idle"
                        actions[i] = default_joint_pos
        if global_step % 200 == 0:
            n_run = sum(1 for p in phase if p == "running")
            n_set = sum(1 for p in phase if p == "settling")
            logger.info("  async progress: gstep=%d | running=%d settling=%d queue=%d done=%d",
                        global_step, n_run, n_set, len(queue), len(results))
    if global_step >= safety_cap:
        logger.error("async safety_cap %d hit — aborting (stuck env?)", safety_cap)
    return global_step


def _run_lockstep(env, scene, robot, cube, jaw_body_idx, eval_cfg, policy_ifaces, binmap,
                  scene_loader, ep_idx_list, num_envs, default_joint_pos, default_action, device,
                  max_steps_per_episode, results, busy_intervals, frames_dir, video_recorders,
                  nan_count, out_dir) -> int:
    """Batched barrier: each env a different episode per batch, global reset + while-not-all-done.

    Faithful current-production behaviour, but over DISTINCT episodes (each env its own) so it is
    apples-to-apples with async on the SAME 27 episodes. A finished env holds (idles) until the
    slowest in the batch terminates -> the lockstep cost, visible as a gantt gap.
    """
    actions = default_action.clone()
    global_step = 0
    tc_margin = args_cli.table_collision_margin_m
    tc_frames = max(1, args_cli.table_collision_frames)
    batches = [ep_idx_list[s:s + num_envs] for s in range(0, len(ep_idx_list), num_envs)]
    obs = None
    for batch in batches:
        n_used = len(batch)
        with torch.inference_mode():
            # Per-env PROPER reset — IDENTICAL to async's setup (only the SCHEDULING differs: barrier
            # vs refill). The earlier per-batch `env.reset()` + manual `_reset_robot_env` under-cleaned
            # cross-batch state → later batches failed baseline-100% episodes (v0.2.0 c2_r2: lockstep
            # 77.8% vs async 88.9% vs baseline 92.6%). `_reset_idx` is the sanctioned manager-level
            # reset; mirroring it makes the success rates match so the speedup isolates the scheduling.
            placed_cubes = []
            for j, ep in enumerate(batch):
                placed = _cube_pose_for_episode(binmap, ep)
                env.unwrapped._reset_idx(torch.tensor([j], device=device, dtype=torch.long))
                _place_cube_env(scene, placed, j, device)
                _reset_robot_env(scene, default_joint_pos, j, device)
                placed_cubes.append(placed)
                policy_ifaces[j].policy.reset()
                policy_ifaces[j].preprocessor.reset()
                policy_ifaces[j].postprocessor.reset()
            _set_pad_yaw(env, 0.0)
            for _ in range(args_cli.warmup_steps):
                obs = env.step(default_action)[0]
                global_step += 1   # count the warmup/settle (like async) so it's a VISIBLE gap
                                   # before the batch on the global-step axis — not an invisible delay
            _augment_obs(obs)  # post-warmup obs feeds the first inference AND the dump below
            maxv = max(_cube_settled_vel(scene, j) for j in range(n_used))
            if maxv > SETTLE_VEL_THRESHOLD:
                logger.warning("  lockstep batch cubes NOT settled after %d warmup steps "
                               "(max vel=%.4f > %.3f m/s) — increase --warmup_steps",
                               args_cli.warmup_steps, maxv, SETTLE_VEL_THRESHOLD)
            if args_cli.verify_placement:
                for j, placed in enumerate(placed_cubes):
                    _verify_placement(scene, placed, j)
            if args_cli.capture_frames:
                for j in range(n_used):
                    _dump_env_frame(obs, frames_dir, batch[j], j)

        trackers = [_make_tracker(cube, robot, jaw_body_idx, j, eval_cfg) for j in range(n_used)]
        joint_traj: list[list[list[float]]] = [[] for _ in range(n_used)]
        done = [False] * n_used
        done_step = [0] * n_used
        tc_counters = [0] * n_used
        tc_fired = [False] * n_used
        batch_start = global_step
        step = 0
        actions[:] = default_joint_pos
        while not all(done) and step < max_steps_per_episode:
            for j in range(n_used):
                actions[j] = default_joint_pos if done[j] else _infer_env(policy_ifaces, obs, j)
            if torch.isnan(actions).any() or torch.isinf(actions).any():
                nan_count[0] += 1
                actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
            obs = env.step(actions)[0]
            _augment_obs(obs)  # parity with async path
            step += 1
            global_step += 1
            for j in range(n_used):
                if done[j]:
                    continue
                cube_xyz = cube.data.root_link_pos_w[j, :3]
                gripper_xyz = robot.data.body_link_pos_w[j, jaw_body_idx, :3]
                dist_3d = torch.norm(gripper_xyz - cube_xyz).item()
                aj = robot.data.joint_pos[j].cpu().tolist()
                joint_traj[j].append(aj)
                trackers[j].update(cube_xyz[2].item(), dist_3d, step, arm_joint_pos=aj)
                _table_collision_step(scene, gripper_xyz[2].item(), j, tc_counters, tc_fired, tc_margin, tc_frames)
                if video_recorders[j].enabled:
                    video_recorders[j].capture(_slice_obs_env(obs, j), step)
                if trackers[j].should_terminate or tc_fired[j]:
                    done[j] = True
                    done_step[j] = global_step
                    actions[j] = default_joint_pos
            if step % 200 == 0:
                logger.info("  lockstep progress: batch %s step=%d done=%d/%d",
                            [e + 1 for e in batch], step, sum(done), n_used)
        for j in range(n_used):
            end = done_step[j] if done[j] else global_step
            results.append(_result_row(trackers[j], batch[j], j, step, joint_traj[j],
                                       table_collision=tc_fired[j]))
            busy_intervals[j].append([batch_start, end])  # gap [end, batch_end] = lockstep idle
            if args_cli.dump_joint_states:
                _dump_joint_states(joint_traj[j], getattr(robot, "joint_names", None),
                                   out_dir / "joint_states", batch[j], j)
            if video_recorders[j].enabled:
                video_recorders[j].flush(out_dir / "videos" / f"env{j}"
                                         / f"episode_{batch[j] + 1:03d}")
        logger.info("  [lockstep] batch %s done at global_step=%d", [e + 1 for e in batch], global_step)
    return global_step


def _verify_placement(scene, placed_cube: dict, env_id: int) -> None:
    """FAIL-LOUD: per-env cube readback must match the expected binmap world pose."""
    ex, ey, ez = _expected_cube_world_xyz(scene, placed_cube, env_id)
    actual = scene["cube"].data.root_link_pos_w[env_id, :3].cpu().tolist()
    err = max(abs(actual[0] - ex), abs(actual[1] - ey), abs(actual[2] - ez))
    # Yaw quaternion readback (XYZW): qz=sin(yaw/2), qw=cos(yaw/2). Guards the 2026-06-14 cube-yaw
    # bugfix against regression — a position-only write would read back yaw≈0 here and FAIL LOUD.
    q = scene["cube"].data.root_state_w[env_id, 3:7].cpu().tolist()  # XYZW
    yaw_actual = 2.0 * math.atan2(q[2], q[3])
    yaw_exp = float(placed_cube["cube_yaw_rad"])
    yaw_err = abs((math.degrees(yaw_actual) - math.degrees(yaw_exp) + 180.0) % 360.0 - 180.0)
    logger.info("  [verify] env%02d expected=(%.3f,%.3f,%.3f|yaw=%.1f°) actual=(%.3f,%.3f,%.3f|yaw=%.1f°) "
                "err=%.4fm yaw_err=%.1f°", env_id, ex, ey, ez, math.degrees(yaw_exp),
                actual[0], actual[1], actual[2], math.degrees(yaw_actual), err, yaw_err)
    if err > 0.01:
        raise RuntimeError(
            f"PLACEMENT MISMATCH env{env_id}: expected ({ex:.3f},{ey:.3f},{ez:.3f}) "
            f"!= actual ({actual[0]:.3f},{actual[1]:.3f},{actual[2]:.3f}) err={err:.4f}m "
            f"(cross-env contamination or env_origins bug)"
        )
    if yaw_err > 2.0:
        raise RuntimeError(
            f"YAW MISMATCH env{env_id}: expected yaw={math.degrees(yaw_exp):.1f}° "
            f"!= actual yaw={math.degrees(yaw_actual):.1f}° (cube quaternion not written — "
            f"the 2026-06-14 _place_cube_env yaw bug regressed)"
        )


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        logger.exception("async_eval_sandbox failed")
        rc = 1
    sys.stdout.flush()
    sys.stderr.flush()
    # Isaac Sim 6.0 simulation_app.close() hangs for non-recording scripts -> os._exit is the
    # documented workaround (CLAUDE.md submodule §6).
    os._exit(rc)
