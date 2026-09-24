# SPDX-License-Identifier: MIT

"""Smoke test: spawn LiftCube scene, verify cube + white robot visible.

Category A script — calls simulation_app.close() at end.
"""

VERSION = "1.0.0"

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="LiftCube scene smoke test.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Post-AppLauncher imports ---
import sys
import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

# Add pipeline dir for tasks import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Register LiftCube task
import so101_mvbench.tasks  # noqa: F401 — triggers gym.register()

from so101_mvbench.logging import get_logger, configure_root_logger

logger = get_logger(__name__)
configure_root_logger()

# --- Main ---
def main():
    logger.info("Creating LiftCube environment...")

    from isaaclab_tasks.utils import parse_env_cfg
    env_cfg = parse_env_cfg(
        "LiftCube-Sim",
        device=args_cli.device,
        num_envs=1,
    )

    env = gym.make("LiftCube-Sim", cfg=env_cfg)
    logger.info("Environment created!")

    obs, _ = env.reset()
    logger.info("Reset done.")

    # Verify scene contents
    scene_keys = list(env.unwrapped.scene.keys())
    logger.info("Scene keys: %s", scene_keys)

    assert "cube" in scene_keys, "FAIL: cube not in scene!"
    assert "robot" in scene_keys, "FAIL: robot not in scene!"

    cube = env.unwrapped.scene["cube"]
    pos = cube.data.root_pos_w[0].cpu().tolist()
    logger.info("Cube position: (%.3f, %.3f, %.3f)", *pos)

    robot = env.unwrapped.scene["robot"]
    joints = robot.data.joint_pos[0].cpu().tolist()
    logger.info("Robot joints: %s", [f"{j:.3f}" for j in joints])

    # Step a few times to verify physics works
    neutral = torch.zeros(1, env.unwrapped.action_manager.action.shape[-1],
                          device=env.unwrapped.device)
    for i in range(30):
        env.step(neutral)
    logger.info("30 neutral steps OK — physics stable.")

    # Final cube position (should be close to init)
    pos2 = cube.data.root_pos_w[0].cpu().tolist()
    logger.info("Cube after 30 steps: (%.3f, %.3f, %.3f)", *pos2)

    logger.info("LIFT_CUBE_SMOKE_TEST_PASSED")
    logger.info("Scene is open — inspect in GUI. Press Ctrl+C to exit.")

    # Keep scene alive for visual inspection
    try:
        while True:
            simulation_app.update()
    except KeyboardInterrupt:
        logger.info("Ctrl+C — shutting down.")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
