"""LiftCube policy eval — single-env, modular Isaac Lab pipeline.

Modules (clean dependency order):
    startup          Input-validation + sanity-checks (no Isaac-Sim deps)
    eval_tracker     4-stage success metric (no Isaac-Sim deps)
    results_writer   CSV + JSON output (no Isaac-Sim deps)
    camera_override  set_world_poses_from_view helper
    scene_loader     scene_state.json iterator + Isaac Sim applicator
    policy_wrapper   SimLeRobotSO101Interface subclass (no scservo_sdk)
    eval_session     Per-episode 5-step setup helper
    eval_policy      Main orchestrator (entry point: `lerobot_eval`)
    eval_report      Post-analysis (pandas + matplotlib)

Eval-Variants are encoded as gym-task IDs registered in `tasks/__init__.py`:
LiftCube-Sim (ID), LiftCube-Sim-OOD-Light, LiftCube-Sim-OOD-Visual.
The DR knobs (cube_noise_m, cube_rot_noise_deg, randomization_groups)
live on the env_cfg subclass — eval_policy reads them after parse_env_cfg.
"""

from .eval_tracker import LiftCubeEvalTracker
from .results_writer import ResultsWriter

__all__ = ["LiftCubeEvalTracker", "ResultsWriter"]
