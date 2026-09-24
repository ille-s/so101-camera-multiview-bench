"""Gamepad recorder — modular teleop recording for LiftCube.

Thin main script that delegates to:
- RecordingSession — dataset lifecycle (Context Manager, finalize guarantee)
- EpisodeController — state machine (S/R key handling, BinSpawner)
- GamepadActionGenerator / SineWaveActionGenerator — action sources

Replaces the monolithic policy_recorder.py.

Usage:
    gamepad_recorder --task=LiftCube-Sim --num_episodes=27 --output_dir=datasets/...
"""

__version__ = "2.1.1"

# ---------------------------------------------------------------------------
# Isaac Sim MUST be launched before any other imports.
# ---------------------------------------------------------------------------
import argparse
import math

from isaaclab.app import AppLauncher

# WORKAROUND: LiftCubeEnvCfg.episode_length_s defaults to 5s → time_out DoneTerm
# teleports the cube every 150 frames. We disable it for recording by setting
# a long horizon. TODO(#008): replace with a proper ReplayEnvCfg.
_NO_TIMEOUT_EPISODE_LENGTH_S = 9999

parser = argparse.ArgumentParser(description="Gamepad recorder for LiftCube teleop.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None, help="Isaac Lab task name.")
parser.add_argument("--repo_id", type=str, required=True, help="LeRobot dataset repo ID.")
parser.add_argument("--repo_root", type=str, required=True, help="Local dataset root path.")
parser.add_argument("--task_name", type=str, required=True, help="Human-readable task name.")
parser.add_argument("--num_episodes", type=int, default=10, help="Episodes (auto_record mode).")
parser.add_argument("--episode_length_s", type=float, default=5.0, help="Episode length (s).")
parser.add_argument("--auto_record", action="store_true", default=False)
parser.add_argument("--action_source", choices=["sine", "gamepad"], default="sine")
parser.add_argument("--gamepad_config", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--object_key", type=str, default="cube")
parser.add_argument("--randomize", type=str, default="all")
parser.add_argument("--scene_config", type=str, default=None,
                    help="Scene config JSON. When given, recording happens in the scene it "
                         "describes (room USD + workspace origin) instead of the default "
                         "floor-level table scene; scene_state and binmap then declare that "
                         "origin so replay/eval derive their shift instead of assuming it.")
# Bin spawning
parser.add_argument("--spawn_bins", action="store_true", default=False)
parser.add_argument("--spawn_n_cols", type=int, default=5)
parser.add_argument("--spawn_n_rows", type=int, default=5)
parser.add_argument("--spawn_bin_size", type=float, default=0.05)
parser.add_argument("--spawn_center_x", type=float, default=0.0)
parser.add_argument("--spawn_center_y", type=float, default=0.0)
parser.add_argument("--spawn_ood_bins", type=str, default="")
parser.add_argument("--spawn_bin_col", type=int, default=None)
parser.add_argument("--spawn_bin_row", type=int, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# Pre-flight: abort BEFORE Sim-launch when --action_source=gamepad but no
# device exists. Without this check, JointGamepad.__init__ raises after Sim
# is already up and Carb threads block on cleanup — requires kill -9 and
# in one case froze the whole host.
if args_cli.action_source == "gamepad":
    import sys
    from so101_mvbench.vendor.gamepad_utils import find_js_device  # sim-free, safe pre-AppLauncher
    if find_js_device() is None:
        print("ERROR: No gamepad found at /dev/input/js*. "
              "Plug in controller and re-run.", file=sys.stderr)
        sys.exit(1)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Post-SimulationApp imports
# ---------------------------------------------------------------------------
import json
from pathlib import Path

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import so101_mvbench.tasks  # noqa: F401

from so101_mvbench.vendor.lerobot_so101_teleop.keyboard import KeyboardControl
from so101_mvbench.vendor.lerobot_so101_teleop.lerobot_interface import LeRobotSO101Interface

from so101_mvbench.logging import get_logger, configure_root_logger
from so101_mvbench.utils.bin_spawner import BinMap, BinSpawner, WorkspaceBounds, parse_ood_bins , apply_cube_pose
from so101_mvbench.utils.scene_state import SceneState
from so101_mvbench.utils.so101_transforms import FPS
from so101_mvbench.tasks.lift_cube_env_cfg import configure_randomization

from so101_mvbench.recording.recording_session import RecordingSession
from so101_mvbench.recording.episode_controller import EpisodeController

logger = get_logger(__name__)
configure_root_logger()


# ---------------------------------------------------------------------------
# SineWaveActionGenerator (inline — no Isaac Lab dependency)
# ---------------------------------------------------------------------------

class SineWaveActionGenerator:
    """Safe sinusoidal joint trajectories for smoke tests."""

    NEUTRAL_DEG = {
        "shoulder_pan.pos":  math.degrees(-0.2736),
        "shoulder_lift.pos": math.degrees(-0.6109),
        "elbow_flex.pos":    math.degrees(-0.0745),
        "wrist_flex.pos":    math.degrees( 1.5148),
        "wrist_roll.pos":    math.degrees(-1.6034),
        "gripper.pos":       math.degrees(-0.1465),
    }
    AMPLITUDE_DEG = {
        "shoulder_pan.pos": 20.0, "shoulder_lift.pos": 15.0,
        "elbow_flex.pos": 15.0, "wrist_flex.pos": 10.0,
        "wrist_roll.pos": 20.0, "gripper.pos": 30.0,
    }
    FREQ_HZ = {
        "shoulder_pan.pos": 0.20, "shoulder_lift.pos": 0.15,
        "elbow_flex.pos": 0.17, "wrist_flex.pos": 0.13,
        "wrist_roll.pos": 0.22, "gripper.pos": 0.10,
    }
    JOINT_ORDER = list(NEUTRAL_DEG.keys())

    def __init__(self, fps: int = 30):
        self._fps = fps
        self._step = 0

    def get_action(self) -> dict:
        t = self._step / self._fps
        action = {
            k: self.NEUTRAL_DEG[k] + self.AMPLITUDE_DEG[k]
               * math.sin(2 * math.pi * self.FREQ_HZ[k] * t)
            for k in self.JOINT_ORDER
        }
        self._step += 1
        return action

    def reset(self):
        self._step = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WS_ORIGIN_FOR_SCENE_STATE: tuple[float, float, float] | None = None
"""Set once at startup when --scene_config is given; recorded into every
scene_state so downstream consumers know which frame the episode lives in."""


def _save_scene_state(env, episode_index, repo_root, seed, object_key):
    """Capture and save scene state after env.reset()."""
    state = SceneState.from_env(
        env=env, episode_index=episode_index, seed=seed,
        look_at=(0.15, 0.0, 0.10), object_key=object_key,
        workspace_origin_m=_WS_ORIGIN_FOR_SCENE_STATE,
    )
    path = Path(repo_root) / "meta" / f"episode_{episode_index:06d}_scene_state.json"
    state.save(path)
    logger.info("Saved scene_state: %s", path)


def _apply_spawn_config(env, spawner):
    """Teleport cube to BinSpawner position after env.reset()."""
    cfg = spawner.next_config()
    apply_cube_pose(env, cfg.x_m, cfg.y_m, cfg.yaw_rad)


def _build_features(cameras: dict) -> dict:
    """Build LeRobotDataset feature dict from camera discovery."""
    joint_names = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                   "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": joint_names},
        "action": {"dtype": "float32", "shape": (6,), "names": joint_names},
    }
    for cam_name, cam_info in cameras.items():
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": (cam_info["height"], cam_info["width"], 3),
            "names": ["height", "width", "channels"],
        }
    return features


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("gamepad_recorder.py v%s", __version__)

    # --- Keyboard ---
    try:
        keyboard_control = KeyboardControl()
    except Exception:
        logger.exception("KeyboardControl init failed — aborting")
        return

    # --- Environment ---
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = _NO_TIMEOUT_EPISODE_LENGTH_S

    # --- Optional scene override (record in a different target scene) ---
    # The LiftCube scene config values are authored natively in the cylindrical
    # room (BASE_SCENE_ORIGIN_W), so the rigid shift is derived against that
    # frame -- zero for the shipped configs. Everything the rig couples to moves
    # together: robot base, cube spawn, action pad, and the external camera
    # (the wrist camera is gripper-relative and follows).
    from so101_mvbench.utils.so101_transforms import (
        BASE_SCENE_ORIGIN_W,
        rigid_shift_from_recording,
    )
    _ws_origin = BASE_SCENE_ORIGIN_W
    if args_cli.scene_config:
        from so101_mvbench.utils.scene_config import SceneConfig

        _scene_config = SceneConfig.from_json(Path(args_cli.scene_config))
        _ws_origin = tuple(_scene_config.workspace_origin_m)
        _shift = rigid_shift_from_recording(_ws_origin, BASE_SCENE_ORIGIN_W)
        if _shift != (0.0, 0.0, 0.0):
            # Foreign target scene: move the whole rig additively. NEVER assign
            # the shift as a position -- the base scene is authored natively in
            # the cylroom, and overwriting its values with a shift once put the
            # arm on the floor.
            for _obj in (
                env_cfg.scene.robot,
                env_cfg.scene.cube,
                env_cfg.scene.action_pad,
            ):
                _p0 = _obj.init_state.pos
                _obj.init_state.pos = (
                    _p0[0] + _shift[0], _p0[1] + _shift[1], _p0[2] + _shift[2],
                )
            _e = env_cfg.scene.camera_external.offset.pos
            env_cfg.scene.camera_external.offset.pos = (
                _e[0] + _shift[0], _e[1] + _shift[1], _e[2] + _shift[2],
            )
        logger.info("Scene config: %s | origin=%s shift=%s",
                    Path(args_cli.scene_config).name, _ws_origin, _shift)
    global _WS_ORIGIN_FOR_SCENE_STATE
    _WS_ORIGIN_FOR_SCENE_STATE = _ws_origin

    # --- Randomization + BinSpawner ---
    randomize_map = {
        "all": {"visual", "spatial"}, "visual": {"visual"},
        "spatial": {"spatial"}, "none": set(),
    }
    groups = randomize_map.get(args_cli.randomize, {"visual", "spatial"})

    spawner = None
    bin_map = None
    binmap_path = None

    if args_cli.spawn_bins:
        cube_pos = env_cfg.scene.cube.init_state.pos
        ood_bins = parse_ood_bins(args_cli.spawn_ood_bins)

        # Single-bin selection
        selected_bin = None
        if args_cli.spawn_bin_col is not None and args_cli.spawn_bin_row is not None:
            selected_bin = (args_cli.spawn_bin_col, args_cli.spawn_bin_row)
            args_cli.repo_root = str(
                Path(args_cli.repo_root) / f"bin_c{selected_bin[0]}_r{selected_bin[1]}"
            )
        elif (args_cli.spawn_bin_col is None) != (args_cli.spawn_bin_row is None):
            raise ValueError("--spawn_bin_col and --spawn_bin_row must both be set or both omitted.")

        ws = WorkspaceBounds.from_center(
            center_x=args_cli.spawn_center_x, center_y=args_cli.spawn_center_y,
            n_cols=args_cli.spawn_n_cols, n_rows=args_cli.spawn_n_rows,
            bin_size=args_cli.spawn_bin_size,
            origin_x=cube_pos[0], origin_y=cube_pos[1],
        )
        spawner = BinSpawner(
            ws, n_cols=args_cli.spawn_n_cols, n_rows=args_cli.spawn_n_rows,
            half_extent_m=0.02, object_key=args_cli.object_key,
            robot_yaw_rad=math.pi / 2, ood_bins=ood_bins,
            selected_bin=selected_bin,
        )

        binmap_path = Path(args_cli.repo_root) / "binmap.json"
        if binmap_path.exists():
            bin_map = BinMap.load(binmap_path)
            if bin_map.workspace_origin_m != (_ws_origin if _ws_origin is None
                                              else tuple(_ws_origin)):
                raise ValueError(
                    f"binmap workspace origin {bin_map.workspace_origin_m} does not match "
                    f"this run's scene origin {_ws_origin}. Resuming a dataset recorded in a "
                    f"different scene would mix frames -- record into a fresh --repo_root."
                )
            spawner.restore_from_binmap(bin_map)
        else:
            bin_map = BinMap(
                n_cols=args_cli.spawn_n_cols, n_rows=args_cli.spawn_n_rows,
                object_key=args_cli.object_key, ood_bins=ood_bins,
                workspace_origin_m=_ws_origin,
            )

        groups.discard("spatial")
        logger.info("BinSpawner: %d episodes, %d remaining",
                    spawner.total_episodes, spawner.remaining)

    configure_randomization(env_cfg.events, groups=groups)
    env = gym.make(args_cli.task, cfg=env_cfg)

    # --- Initial reset ---
    obs, _ = env.reset()
    if spawner is not None:
        _apply_spawn_config(env, spawner)

    # --- Cameras ---
    cameras = {}
    for obj in env.unwrapped.scene.keys():
        if obj.startswith("camera_"):
            cfg = getattr(env.unwrapped.scene.cfg, obj)
            cameras[obj.replace("camera_", "")] = {"height": cfg.height, "width": cfg.width}

    # --- Coordinate transforms ---
    coord_iface = LeRobotSO101Interface(
        device=env.unwrapped.device, port="none", id="recorder",
        cameras=cameras, fps=FPS, kind="leader",
    )

    # --- Action generator ---
    if args_cli.action_source == "gamepad":
        from so101_mvbench.recording.gamepad_action_generator import GamepadActionGenerator
        action_gen = GamepadActionGenerator(coord_iface, obs["policy"][0], args_cli.gamepad_config)
    else:
        action_gen = SineWaveActionGenerator(fps=FPS)

    # --- Warmup: drive the arm to its neutral pose, deterministically ---
    # The environment is configured with JointPositionActionCfg(use_default_offset=False),
    # so an action is an ABSOLUTE joint position, not a delta.
    #
    # We therefore command default_joint_pos directly rather than asking the action
    # generator. Asking it would be wrong for --action_source gamepad: it reads the
    # current stick position, and a resting stick is rarely exactly centred. The arm
    # would settle a little off neutral and every episode would inherit that offset.
    #
    # 60 steps is enough for the PD controller to converge.
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    neutral_joint_pos = env.unwrapped.scene["robot"].data.default_joint_pos[0]
    actions[:] = neutral_joint_pos
    with torch.inference_mode():
        for _ in range(60):
            env.step(actions)
    if hasattr(action_gen, "reset"):
        action_gen.reset()

    # --- Y-Button Home-Macro (leisaac-style PD-hold Auto-Pilot) ---
    from so101_mvbench.recording.home_macro import HomeMacro

    import os as _os
    # 120 steps at 30 Hz is 4 seconds, which the smoothstep interpolation covers
    # without a visible jerk. Override with HOME_HOLD_STEPS to tune.
    _hold_steps = int(_os.environ.get("HOME_HOLD_STEPS", "120"))
    home_macro = HomeMacro(
        env=env,
        action_gen=action_gen,
        coord_iface=coord_iface,
        action_shape=actions.shape,
        device=env.unwrapped.device,
        hold_steps=_hold_steps,
    )

    if args_cli.action_source == "gamepad":
        # Y-Button-Index aus gamepad_utils Config lesen (Single Source of Truth,
        # The pad-specific button mapping belongs to the gamepad package, not to a
        # workspace config. Note the Bluetooth Xbox Wireless Controller reports Y as
        # button 4, not the Linux-standard 3.
        import os, json
        from importlib.resources import files as _pkg_files
        _gp_cfg_path = Path(str(_pkg_files("so101_mvbench.vendor.gamepad_utils").joinpath(
            "configs/so101_lerobot_teleop.json"
        )))
        home_button = None
        try:
            _gp_cfg = json.loads(_gp_cfg_path.read_text())
            btn = _gp_cfg.get("button_labels", {}).get("Y")
            if btn is not None:
                home_button = int(btn)
                logger.info("[Y-MACRO] Y button index %d loaded from gamepad_utils config",
                            home_button)
        except Exception as exc:
            logger.warning("[Y-MACRO] Could not parse gamepad config %s: %s",
                           _gp_cfg_path, exc)
        if home_button is None:
            home_button = int(os.environ.get("HOME_BUTTON", "4"))  # BT Xbox Y=4
            logger.warning(
                "[Y-MACRO] Y button not found in gamepad config — fallback to %d "
                "(override via HOME_BUTTON env var)", home_button,
            )
        action_gen._gamepad.add_callback(home_button, home_macro.trigger)
        logger.info("[Y-MACRO] Gamepad Y button (index %d) bound to HomeMacro.trigger()",
                    home_button)

    # --- Quit key (Q) ---
    # Ctrl+C is unreliable with Isaac Sim (Jupyter server intercepts SIGINT).
    # Extend existing KeyboardControl with Q handler (same subscriber).
    import carb.input
    keyboard_control.quit_requested = False
    _orig_cb = keyboard_control._on_keyboard_event

    def _extended_cb(event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "Q":
                keyboard_control.quit_requested = True
                logger.info("Quit requested (Q pressed)")
                return True
        return _orig_cb(event, *args, **kwargs)

    # Re-subscribe with extended callback (guard against missing original subscription)
    _input = carb.input.acquire_input_interface()
    if keyboard_control._sub_keyboard:
        _input.unsubscribe_to_keyboard_events(
            keyboard_control._keyboard, keyboard_control._sub_keyboard
        )
        keyboard_control._sub_keyboard = _input.subscribe_to_keyboard_events(
            keyboard_control._keyboard, _extended_cb
        )
        logger.info("Keyboard: S=start/stop, R=reset/advance, Q=quit")
    else:
        logger.warning("Keyboard: _sub_keyboard is None — Q-handler not installed")

    # --- Recording Session (Context Manager → guaranteed finalize) ---
    features = _build_features(cameras)
    episode_length_steps = max(1, int(args_cli.episode_length_s * FPS))

    with RecordingSession(
        repo_id=args_cli.repo_id, root=args_cli.repo_root,
        fps=FPS, features=features, task_name=args_cli.task_name,
    ) as session:

        # Ensure finalize runs on SIGTERM / SIGINT (Window-Close, pkill, Ctrl-C).
        # Does NOT protect against SIGKILL (-9) — that's kernel-level and unfangbar.
        import signal as _signal
        def _on_terminate(signum, frame):
            logger.warning("Signal %d received — finalizing dataset before exit", signum)
            try:
                session._finalize()
            finally:
                import os as _os
                _os._exit(128 + signum)
        _signal.signal(_signal.SIGTERM, _on_terminate)
        _signal.signal(_signal.SIGINT, _on_terminate)

        start_episode = session.total_episodes
        _save_scene_state(env, start_episode, args_cli.repo_root, args_cli.seed, args_cli.object_key)

        controller = EpisodeController(
            session=session, env=env,
            spawner=spawner, bin_map=bin_map, binmap_path=binmap_path,
            scene_state_fn=_save_scene_state,
            apply_spawn_fn=_apply_spawn_config if spawner else None,
            action_gen=action_gen,
            repo_root=args_cli.repo_root,
            seed=args_cli.seed, object_key=args_cli.object_key,
            start_episode=start_episode,
        )

        if args_cli.auto_record:
            controller.on_key_s()  # start recording immediately

        logger.info("Main loop. auto_record=%s, episode_length_steps=%d",
                    args_cli.auto_record, episode_length_steps)

        while simulation_app.is_running() and not keyboard_control.quit_requested:
            if args_cli.auto_record and controller.episode >= args_cli.num_episodes:
                logger.info("Recorded %d episodes — done.", controller.episode)
                break
            if controller.is_done:
                logger.info("BinSpawner: all episodes recorded — done.")
                break

            with torch.inference_mode():
                raw_action = action_gen.get_action()
                real_action, mapped_action = coord_iface.real_to_sim_obs_processor(raw_action)
                actions[:] = mapped_action

                # While the home macro runs it owns ONLY the arm joints (0..4). The
                # gripper (5) keeps following the gamepad, so the operator can hold on
                # to the cube while the arm travels home.
                # (servo PD hält die commanded position, egal was der Arm macht).
                macro_result = home_macro.step()
                if macro_result is not None:
                    sim_override, real_override = macro_result
                    actions[:, :5] = sim_override[:, :5]
                    real_action[:5] = real_override[:5]

                obs, _, _, _, _ = env.step(actions)

                # Key handling — S/R cancels active macro first
                if keyboard_control.reset_world:
                    keyboard_control.reset_world = False
                    home_macro.cancel()
                    controller.on_key_r()
                    continue

                if not args_cli.auto_record:
                    if keyboard_control.recording and not controller.is_recording:
                        home_macro.cancel()
                        controller.on_key_s()
                    elif not keyboard_control.recording and controller.is_recording:
                        home_macro.cancel()
                        controller.on_key_s()

                # Frames recorded during the home macro are KEPT on purpose: the move
                # from the grasp point back to the home pose IS the lifting half of the
                # pick-and-lift task. Dropping them would leave the policy with grasping
                # demonstrations only, and it would never learn to lift.
                if controller.is_recording:
                    real_obs, visual_buffers = coord_iface.sim_to_real_dataset_processor(
                        obs["policy"][0], obs["visual"],
                    )
                    frame = {
                        "observation.state": real_obs.cpu().numpy(),
                        "action": real_action.cpu().numpy(),
                    }
                    for cam_name in cameras:
                        frame[f"observation.images.{cam_name}"] = (
                            visual_buffers[cam_name].cpu().numpy()
                        )
                    controller.add_frame(frame)

                # Auto-end
                if args_cli.auto_record and controller.is_recording:
                    if controller.step >= episode_length_steps:
                        controller.on_auto_episode_end()

        if keyboard_control.quit_requested:
            logger.info("Recording stopped (Q pressed)")

    # RecordingSession.__exit__ guarantees finalize()
    env.close()
    # Explicit int return — console-script wrapper does sys.exit(main()),
    # and Isaac Sim's app.post_quit() rejects None with TypeError.
    return 0


if __name__ == "__main__":
    main()
    import os
    os._exit(0)
