# SPDX-License-Identifier: MIT
"""Multi-env cube-state CSV logger for Multi-Env-Divergence audit.

Replays one or more episodes' action trajectories across N envs in parallel
(broadcast), logs cube pos+quat per env per frame to CSV. No video rendering.

Single Isaac Sim instance handles all (bin, episode) pairs sequentially —
amortizes the ~60s sim startup over the whole batch.

Usage:
    # Single episode
    python -m so101_mvbench.audit.cube_replay \\
        --bins bin_c4_r3 --episodes 6 --num_envs 6 \\
        --output_root datasets/audit_cube_trajectories

    # Full bulk: all bins, all episodes per bin
    python -m so101_mvbench.audit.cube_replay \\
        --bins ALL --episodes ALL --num_envs 6 \\
        --output_root datasets/audit_cube_trajectories
"""

VERSION = "1.0.0"

import argparse
import csv
import json
import time
from pathlib import Path

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--bins", required=True, help="Comma-separated bin names, or 'ALL'.")
p.add_argument("--episodes", default="ALL",
               help="Comma-separated episode indices applied to every bin, or 'ALL'.")
p.add_argument("--num_envs", type=int, default=6)
p.add_argument("--datasets_root", default="datasets/01_raw_gamepad")
p.add_argument("--output_root", required=True,
               help="Output dir; CSV per ep written to <output_root>/<bin>/cube_states_ep_NN.csv.")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = False
# args.headless honored from AppLauncher CLI: pass --headless for batch, omit for GUI viewer
sim_app = AppLauncher(args).app

import numpy as np
import torch
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks  # noqa: F401
import so101_mvbench.tasks  # noqa: F401
from so101_mvbench.utils.bin_spawner import BinMap
from so101_mvbench.utils.scene_builder import apply_scene
from so101_mvbench.utils.so101_transforms import (
    FPS, build_joint_tensors, raw_degrees_to_sim_radians,
)
from so101_mvbench.tasks.lift_cube_env_cfg import configure_randomization


def discover_bins(root: Path) -> list[str]:
    return sorted(d.name for d in root.iterdir()
                  if d.is_dir() and d.name.startswith("bin_c") and "_r" in d.name)


def load_bin_episodes(bin_dir: Path, ep_filter: list[int] | None) -> list[tuple[int, np.ndarray]]:
    """Load .npy trajectories for one bin, return list of (ep_idx, actions) tuples."""
    meta = json.loads((bin_dir / "trajectories" / "trajectories_meta.json").read_text())
    out = []
    for ep_meta in meta["episodes"]:
        idx = ep_meta["index"]
        if ep_filter is not None and idx not in ep_filter:
            continue
        npy = bin_dir / "trajectories" / f"episode_{idx:03d}.npy"
        if npy.exists():
            out.append((idx, np.load(npy)))
    return out


# --- Resolve bins ---
datasets_root = Path(args.datasets_root)
output_root = Path(args.output_root)
output_root.mkdir(parents=True, exist_ok=True)

if args.bins == "ALL":
    bin_names = discover_bins(datasets_root)
else:
    bin_names = [b.strip() for b in args.bins.split(",") if b.strip()]

ep_filter = None
if args.episodes != "ALL":
    ep_filter = [int(x.strip()) for x in args.episodes.split(",")]

# --- Load all trajectories upfront; find global max for episode_length_s ---
payloads: list[tuple[str, list[tuple[int, np.ndarray]], BinMap]] = []
global_max_frames = 0
total_eps = 0
for bn in bin_names:
    bd = datasets_root / bn
    if not (bd / "binmap.json").exists():
        print(f"[skip] {bn}: no binmap.json")
        continue
    eps = load_bin_episodes(bd, ep_filter)
    if not eps:
        print(f"[skip] {bn}: no matching episodes")
        continue
    payloads.append((bn, eps, BinMap.load(bd / "binmap.json")))
    total_eps += len(eps)
    for _, a in eps:
        global_max_frames = max(global_max_frames, a.shape[0])
print(f"# Plan: {len(payloads)} bins, {total_eps} episodes, longest={global_max_frames} frames")

# --- Env setup (one sim, reused for all episodes) ---
cfg = parse_env_cfg("LiftCube-Sim", device=args.device, num_envs=1)
cfg.scene.num_envs = args.num_envs
cfg.scene.env_spacing = 8.0
cfg.scene.replicate_physics = False
cfg.sim.physx.enable_enhanced_determinism = True
cfg.seed = 42
cfg.episode_length_s = (global_max_frames / FPS) + 60  # CLAUDE.md §4 PFLICHT
configure_randomization(cfg.events, groups=set())
for attr in ("camera_ego", "camera_external"):
    if hasattr(cfg.scene, attr):
        setattr(cfg.scene, attr, None)
if hasattr(cfg.observations, "visual"):
    cfg.observations.visual = None

env = gym.make("LiftCube-Sim", cfg=cfg)
device = env.unwrapped.device
mins, maxs = build_joint_tensors(device)
cube = env.unwrapped.scene["cube"]

def to_sim(raw):
    return raw_degrees_to_sim_radians(torch.tensor(raw, dtype=torch.float32, device=device), mins, maxs)


# --- Main loop ---
t_start_all = time.time()
ep_done = 0
with torch.inference_mode():
    for bn, eps, binmap in payloads:
        bin_out = output_root / bn
        bin_out.mkdir(parents=True, exist_ok=True)
        for ep_idx, traj in eps:
            ep_done += 1
            T = traj.shape[0]
            t_ep = time.time()
            env.reset()
            apply_scene(env, ep_idx, binmap)
            warm = to_sim(traj[0]).unsqueeze(0).expand(args.num_envs, -1).contiguous()
            for _ in range(60):
                env.step(warm)
            rows = []
            for t in range(T):
                a = to_sim(traj[t]).unsqueeze(0).expand(args.num_envs, -1).contiguous()
                env.step(a)
                pos = cube.data.root_pos_w.cpu().numpy()           # (N,3) world frame
                quat = cube.data.root_quat_w.cpu().numpy()         # (N,4) XYZW (PR #4437)
                for e in range(args.num_envs):
                    rows.append((bn, ep_idx, e, t,
                                 *pos[e].tolist(), *quat[e].tolist()))
            out_csv = bin_out / f"cube_states_ep_{ep_idx:02d}.csv"
            with open(out_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["bin","episode","env","frame","x","y","z","qx","qy","qz","qw"])
                w.writerows(rows)
            dt = time.time() - t_ep
            print(f"[{ep_done}/{total_eps}] {bn} ep{ep_idx:02d}: T={T}, dt={dt:.1f}s, csv={out_csv.name}", flush=True)

print(f"DONE: {ep_done} episodes in {(time.time()-t_start_all)/60:.1f} min")
env.close()
import os; os._exit(0)
