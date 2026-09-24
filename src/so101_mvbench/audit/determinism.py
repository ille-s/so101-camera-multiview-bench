# SPDX-License-Identifier: MIT

"""Determinism Gate — verify physics is identical across camera permutations.

Creates ONE environment, then runs N camera configurations (default: 8
azimuth positions, 0°–315° at 45° steps) by repositioning the camera and
resetting. Logs per-frame object + robot state to .npy.

Supports two action modes:
  - Sine wave (default): synthetic deterministic actions
  - Trajectory replay (--trajectory_dir): real recorded .npy actions

This is a GATE test: if physics differs between camera positions (same seed),
the camera permutation experiment is invalid.

Usage:
    python -m so101_mvbench.audit.determinism --headless
"""

VERSION = "1.2.0"

# ---------------------------------------------------------------------------
# Isaac Sim MUST be launched before any other imports.
# ---------------------------------------------------------------------------
import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Determinism gate test.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument(
    "--task", type=str,
    default="LiftCube-Sim",
    help="Isaac Lab task name.",
)
parser.add_argument(
    "--num_steps", type=int, default=200,
    help="Number of simulation steps per run (default: 200).",
)
_SCRIPT_DIR = Path(__file__).resolve().parent
_SUBMODULE_ROOT = _SCRIPT_DIR.parent.parent.parent
parser.add_argument(
    "--output_dir", type=str,
    default=str(_SUBMODULE_ROOT / "artefacts" / "experiments" / "determinism_test"),
    help="Directory for .npy output files.",
)
parser.add_argument(
    "--trajectory_dir", type=str, default=None,
    help="Path to extracted .npy trajectories. If set, replays real actions instead of sine wave.",
)
parser.add_argument(
    "--episode_index", type=int, default=0,
    help="Episode index to replay (only used with --trajectory_dir).",
)
parser.add_argument(
    "--scene_state", type=str, default=None,
    help="Path to scene_state.json from so101_mvbench.recording. Restores cube position for accurate replay.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# All other imports AFTER simulation_app is created.
# ---------------------------------------------------------------------------
import json
import math
import sys

import numpy as np
import torch
from PIL import Image

import gymnasium as gym

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import so101_mvbench.tasks  # noqa: F401 — registers LiftCube
from so101_mvbench.logging import get_logger, configure_root_logger
from so101_mvbench.utils.scene_state import SceneState
from so101_mvbench.utils.so101_transforms import build_joint_tensors, raw_degrees_to_sim_radians
from so101_mvbench.tasks.lift_cube_env_cfg import configure_randomization

logger = get_logger(__name__)
configure_root_logger()

# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------
# 8 azimuth positions (45° steps) at elevation 45° — covers full hemisphere
# All same seed → scene must be identical, only viewpoint changes
RUNS = [
    {"name": f"run_az{az:03d}", "seed": 42, "camera_az_deg": az, "camera_el_deg": 45}
    for az in range(0, 360, 45)
]

LOOK_AT = (0.15, 0.0, 0.10)
CAMERA_RADIUS_M = 0.6
WARMUP_STEPS = 60


def _spherical_to_cartesian(
    az_deg: float,
    el_deg: float,
    radius_m: float,
    center: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Convert spherical coordinates to cartesian (world frame)."""
    az_rad = math.radians(az_deg)
    el_rad = math.radians(el_deg)
    x = center[0] + radius_m * math.cos(el_rad) * math.cos(az_rad)
    y = center[1] + radius_m * math.cos(el_rad) * math.sin(az_rad)
    z = center[2] + radius_m * math.sin(el_rad)
    return (x, y, z)


def _generate_sine_actions(
    num_steps: int,
    num_joints: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate deterministic sine-wave actions (offsets from default_joint_pos).

    Returns:
        Tensor of shape (num_steps, num_joints) — unbatched actions.
    """
    actions = torch.zeros(num_steps, num_joints, device=device)
    freqs_hz = [0.10, 0.15, 0.12, 0.08, 0.11, 0.0]
    amplitudes_rad = [0.15, 0.10, 0.12, 0.08, 0.10, 0.0]

    for step in range(num_steps):
        t = step / 30.0
        for j in range(num_joints):
            actions[step, j] = amplitudes_rad[j] * math.sin(
                2 * math.pi * freqs_hz[j] * t
            )
    return actions


def _freeze_all_randomization(env_cfg) -> None:
    """Zero out all event randomization for deterministic replay."""
    configure_randomization(env_cfg.events, groups=set())
    logger.info("All event randomization frozen to zero.")


def _find_object_key(env) -> str | None:
    """Find a trackable object in the scene."""
    for candidate in ["yellow_ring", "cube", "rock_a_stack_base"]:
        if candidate in env.unwrapped.scene.keys():
            return candidate
    return None


def _collect_run_data(
    env,
    actions: torch.Tensor,
    num_steps: int,
    object_key: str | None,
    warmup_offset: int = WARMUP_STEPS,
) -> dict:
    """Step through actions and collect per-frame state data.

    Args:
        warmup_offset: Index offset into actions tensor. For sine-wave mode,
            the first WARMUP_STEPS actions are used during warmup, so data
            collection starts at WARMUP_STEPS. For trajectory mode, warmup
            uses neutral actions (not from the trajectory), so offset is 0.
    """
    robot_joint_pos = []
    object_states = []

    for step_idx in range(num_steps):
        action = actions[warmup_offset + step_idx].unsqueeze(0)
        env.step(action)

        robot = env.unwrapped.scene["robot"]
        joint_pos = robot.data.joint_pos[0].cpu().numpy().copy()
        robot_joint_pos.append(joint_pos)

        if object_key is not None:
            obj = env.unwrapped.scene[object_key]
            pos = obj.data.root_pos_w[0].cpu().numpy().copy()
            quat = obj.data.root_quat_w[0].cpu().numpy().copy()
            object_states.append(np.concatenate([pos, quat]))

    result = {"robot_joint_pos": np.array(robot_joint_pos)}
    if object_states:
        result["object_state"] = np.array(object_states)
    return result


def _save_run(result: dict, run_cfg: dict, output_dir: Path) -> None:
    """Save run results to disk."""
    run_dir = output_dir / run_cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    np.save(run_dir / "robot_joint_pos.npy", result["robot_joint_pos"])
    logger.info("Saved robot_joint_pos.npy: shape=%s", result["robot_joint_pos"].shape)

    if "object_state" in result:
        np.save(run_dir / "object_state.npy", result["object_state"])
        logger.info("Saved object_state.npy: shape=%s", result["object_state"].shape)

    with open(run_dir / "run_config.json", "w") as f:
        json.dump(run_cfg, f, indent=2)


def main():
    """Run determinism gate test with a single env instance."""
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_steps = args_cli.num_steps

    logger.info("Determinism test: %d runs, %d steps each", len(RUNS), num_steps)
    logger.info("Output: %s", output_dir)

    # --- Create env ONCE ---
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = RUNS[0]["seed"]
    env_cfg.sim.physx.enable_enhanced_determinism = True
    _freeze_all_randomization(env_cfg)

    # Restore cube position from recording's scene_state (if provided)
    if args_cli.scene_state:
        ss = SceneState.load(Path(args_cli.scene_state))
        pos = ss.get_object_pos_tuple()
        if hasattr(env_cfg.scene, "cube"):
            env_cfg.scene.cube.init_state.pos = pos
            logger.info("Restored cube init_state.pos from scene_state: (%.4f, %.4f, %.4f)", *pos)
        else:
            logger.warning("--scene_state provided but scene has no 'cube' entity")

    # Override camera to pinhole
    if hasattr(env_cfg.scene, "camera_external"):
        env_cfg.scene.camera_external.spawn = sim_utils.PinholeCameraCfg(
            focal_length=5.0,
            horizontal_aperture=10.0,
        )

    env = gym.make(args_cli.task, cfg=env_cfg)
    device = env.unwrapped.device

    num_joints = env.unwrapped.action_manager.action.shape[-1]
    neutral = torch.zeros(1, num_joints, device=device)
    object_key = _find_object_key(env)

    # --- Action source: trajectory .npy or sine wave ---
    if args_cli.trajectory_dir:
        traj_path = Path(args_cli.trajectory_dir) / f"episode_{args_cli.episode_index:03d}.npy"
        if not traj_path.exists():
            logger.error("Trajectory not found: %s", traj_path)
            env.close()
            return
        raw_actions_np = np.load(traj_path)  # (N_frames, 6), raw degrees
        num_steps = raw_actions_np.shape[0]
        logger.info("Loaded real trajectory: %s (%d frames)", traj_path.name, num_steps)

        joint_mins, joint_maxs = build_joint_tensors(device)
        actions = torch.stack([
            raw_degrees_to_sim_radians(
                torch.tensor(raw_actions_np[i], dtype=torch.float32, device=device),
                joint_mins, joint_maxs,
            ) for i in range(num_steps)
        ])  # (num_steps, 6) — sim radians, no warmup prefix
        warmup_offset = 0
    else:
        actions = _generate_sine_actions(num_steps + WARMUP_STEPS, num_joints, device)
        warmup_offset = WARMUP_STEPS

    if object_key:
        logger.info("Tracking object: %s", object_key)
    else:
        logger.warning("No known object found in scene. Logging robot only.")

    results = {}

    # --- Run all camera configurations with same env ---
    for run_cfg in RUNS:
        run_name = run_cfg["name"]
        logger.info("--- %s: seed=%d, camera=az%d_el%d ---",
                    run_name, run_cfg["seed"],
                    run_cfg["camera_az_deg"], run_cfg["camera_el_deg"])

        # Reset env
        env.reset(seed=run_cfg["seed"])

        # Reposition camera
        cam_pos = _spherical_to_cartesian(
            run_cfg["camera_az_deg"],
            run_cfg["camera_el_deg"],
            CAMERA_RADIUS_M,
            LOOK_AT,
        )
        eyes = torch.tensor([cam_pos], dtype=torch.float32, device=device)
        targets = torch.tensor([LOOK_AT], dtype=torch.float32, device=device)
        env.unwrapped.scene["camera_external"].set_world_poses_from_view(eyes, targets)
        logger.info("Camera at (%.3f, %.3f, %.3f)", *cam_pos)

        # Warmup
        for _ in range(WARMUP_STEPS):
            env.step(neutral)
        logger.info("Warmup complete (%d steps)", WARMUP_STEPS)

        # Capture camera frame for visual verification
        run_dir = output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        if "camera_external" in env.unwrapped.scene.keys():
            cam = env.unwrapped.scene["camera_external"]
            cam.update(dt=0.0)
            rgb_data = cam.data.output["rgb"]  # (1, H, W, 4) RGBA
            if rgb_data is not None and rgb_data.numel() > 0:
                frame = rgb_data[0, :, :, :3].cpu().numpy()  # (H, W, 3)
                Image.fromarray(frame).save(run_dir / "frame_000.png")
                logger.info("Saved frame_000.png: %dx%d", frame.shape[1], frame.shape[0])

        # Capture scene state (all randomizable parameters)
        scene_state = SceneState.from_env(
            env=env,
            episode_index=0,
            seed=run_cfg["seed"],
            look_at=LOOK_AT,
            object_key=object_key if object_key else "unknown",
        )
        scene_state.save(run_dir / "scene_state.json")
        logger.info("Saved scene_state.json")

        # Collect data
        result = _collect_run_data(env, actions, num_steps, object_key, warmup_offset)
        _save_run(result, run_cfg, output_dir)
        results[run_name] = result

    # --- Comparison: all pairs against first run (reference) ---
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)

    PASS_THRESHOLD = 1e-5
    run_names = [r["name"] for r in RUNS]
    ref_name = run_names[0]
    all_pass = True

    for other_name in run_names[1:]:
        robot_diff = np.abs(
            results[ref_name]["robot_joint_pos"] - results[other_name]["robot_joint_pos"]
        ).max()
        ref_az = next(r["camera_az_deg"] for r in RUNS if r["name"] == ref_name)
        other_az = next(r["camera_az_deg"] for r in RUNS if r["name"] == other_name)

        obj_diff_str = ""
        if "object_state" in results[ref_name] and "object_state" in results[other_name]:
            obj_diff = np.abs(
                results[ref_name]["object_state"] - results[other_name]["object_state"]
            )
            obj_pos_diff = obj_diff[:, :3].max()
            obj_quat_diff = obj_diff[:, 3:7].max()
            obj_diff_str = f"  obj_pos={obj_pos_diff:.2e}  obj_quat={obj_quat_diff:.2e}"

        passed = robot_diff < PASS_THRESHOLD
        if not passed:
            all_pass = False
        status = "PASS" if passed else "FAIL"
        logger.info("%s (az=%d°) vs %s (az=%d°): robot=%.2e %s%s",
                    ref_name, ref_az, other_name, other_az, robot_diff, status, obj_diff_str)

    logger.info("-" * 60)
    if all_pass:
        logger.info("PASS: Camera change does NOT affect physics (all %d pairs < %.0e)",
                    len(run_names) - 1, PASS_THRESHOLD)
    else:
        logger.info("FAIL: Camera change AFFECTS physics (threshold: %.0e)", PASS_THRESHOLD)

    logger.info("=" * 60)
    logger.info("DETERMINISM_TEST_FINISHED")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
