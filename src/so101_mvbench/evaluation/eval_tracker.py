"""Task progress tracker for LiftCube evaluation.

Tracks 4 monotonic progress stages, grasp attempts, and early termination
conditions. Designed to be testable without Isaac Sim — accepts raw floats,
no torch/Isaac Lab dependency.

Stages (v2.0.0 — post-HomeMacro recording era):
    0_miss  — Gripper never reached the cube (score=0.0)
    1_reach — Gripper within REACH_DIST of cube (score=0.33)
    2_lift  — Cube peak z > initial_z + LIFT_DELTA_REL (score=0.66);
    "cube has left the table", very permissive, peak-based.
    3_home  — Arm has reached the home pose AND cube still held,
    sustained for HOME_HOLD_STEPS (score=1.0, final success).

Breaking-change vs v1.0.0:
    - Stage "2_pick" merged into "2_lift" (same semantic: cube off the table).
    - LIFT_HEIGHT (absolute 12cm) removed — replaced by relative LIFT_DELTA_REL.
      Motivation: after training with the Y-button Home-Macro, the cube ends
      up at an arbitrary z-height in the home pose, not at a fixed 12cm.
    - HOLD_STEPS/SMOOTH_WINDOW (2s sustained at 12cm) replaced by composite
      home-reached criterion: arm_at_home ∧ cube_not_dropped, sustained
      HOME_HOLD_STEPS. The cube MUST still be in the gripper when the arm
      reaches home — otherwise it is a failed task, not a success.

Early Termination:
    ET-1 no_approach  — dist_3d not reduced by 50% after 10s (stage<1 guard)
    ET-2 cube_lost    — cube_z dropped below table level (ET_CUBE_LOST_DELTA)

References:
    - Cherry Shen (so101_bench): 5 progress stages for pick-and-place.
    - DEXTRAH (NVlabs): 2s sustained success timeout.
    - ``home_macro._HOME_REACHED_THRESHOLD_RAD``: source of truth for
      REACH_HOME_RAD (8.6°), kept in sync manually.
"""

VERSION = "2.1.0"

# ---------------------------------------------------------------------------
# Thresholds — derived from task physics, not tuned on evaluation data.
# ---------------------------------------------------------------------------
REACH_DIST = 0.12          # 12cm 3D Euclidean (jaw pivot → cube center; contact empirically at ~7-10cm)
LIFT_DELTA_REL = 0.01      # 1cm above initial cube_z — "cube has left the table"
REACH_HOME_RAD = 0.15      # 8.6° max per-joint error — arm reached home pose
HOME_HOLD_STEPS = 30       # 0.5s @ 60Hz — sustain arm_at_home AND cube_held
CUBE_HOLD_TOLERANCE = 0.01 # 1cm: cube not dropped (stricter than ET_CUBE_LOST)

ET_APPROACH_STEP = 600     # 10s — 3× the physical travel time (~3s for 30cm)
ET_APPROACH_RATIO = 0.5    # dist must decrease by ≥50%
ET_CUBE_LOST_DELTA = 0.02  # 2cm below initial = off the table (episode end)

ET_PLATEAU_STEPS_DEFAULT = {    # per-stage fallback if caller passes None.
    "1_reach": 600,              # 10s @ 60Hz — reach→lift transition budget
    "2_lift":  600,              # 10s @ 60Hz — lift→home transition budget
}                                # Runtime values come from evaluation/config/eval.json
                                 # (key plateau_steps) via _load_eval_config()
                                 # and are threaded through setup_episode.

# Attempt hysteresis
ATTEMPT_UP = 0.02          # cube_z > initial + 2cm → attempt starts
ATTEMPT_DOWN = 0.01        # cube_z < initial + 1cm → attempt ends

STAGES = {0: "0_miss", 1: "1_reach", 2: "2_lift", 3: "3_home"}
SCORES = {0: 0.0, 1: 0.33, 2: 0.66, 3: 1.0}


class LiftCubeEvalTracker:
    """Per-episode eval tracker.  Call update() once per sim step."""

    def __init__(
        self,
        initial_cube_z: float,
        initial_dist_3d: float,
        home_pose_rad: list[float],
        home_pose_tol_rad: float | None = None,
        plateau_steps: dict[str, int] | None = None,
    ):
        """
        Args:
            initial_cube_z: Cube z-coordinate at episode start (m).
            initial_dist_3d: Initial 3D gripper-to-cube distance (m).
            home_pose_rad: Target home pose for arm joints 0..4 (rad).
                Gripper (joint 5) is not checked — Y-macro holds it.
            home_pose_tol_rad: Per-joint max error for arm_at_home detection
                (rad). If None, falls back to REACH_HOME_RAD (0.15 = 8.6°).
                Authoritative source: evaluation/config/eval.json
                target_end_position_tolerance_deg.
            plateau_steps: Per-stage max-stage-stuck ET threshold dict
                (e.g. {"1_reach": 600, "2_lift": 600}). Falls back to
                ET_PLATEAU_STEPS_DEFAULT if None. Authoritative source:
                evaluation/config/eval.json plateau_steps.
        """
        if len(home_pose_rad) < 5:
            raise ValueError(
                f"home_pose_rad must have >= 5 entries, got {len(home_pose_rad)}"
            )
        self._initial_cube_z = initial_cube_z
        self._initial_dist_3d = initial_dist_3d
        self._home_pose_rad = list(home_pose_rad[:5])  # arm joints only

        # Stage tracking (monotonic — only increases)
        self._max_stage = 0
        self._time_to_reach = -1
        self._time_to_lift = -1
        self._time_to_home = -1

        # Cube tracking
        self._min_dist = initial_dist_3d
        self._max_height = initial_cube_z

        # Home-hold counter (consecutive steps with arm_at_home ∧ cube_held)
        self._home_hold_counter = 0
        self._home_hold_total = 0

        # Attempt tracking (hysteresis)
        self._in_attempt = False
        self._num_attempts = 0

        # Early termination
        self._termination_reason: str | None = None
        self._termination_step = -1
        self._step = 0

        # Plateau tracking — step of last _max_stage advance.
        # Initialized to 0 (== episode start). First stage transition at step N
        # sets this to N, so plateau timer is relative to last progression.
        self._last_stage_advance_step = 0
        self._plateau_steps: dict[str, int] = (
            dict(plateau_steps) if plateau_steps is not None
            else dict(ET_PLATEAU_STEPS_DEFAULT)
        )
        self._reach_home_rad: float = (
            home_pose_tol_rad if home_pose_tol_rad is not None
            else REACH_HOME_RAD
        )

    # ------------------------------------------------------------------
    def update(
        self,
        cube_z: float,
        dist_3d: float,
        step: int,
        arm_joint_pos: list[float] | None = None,
    ) -> None:
        """Feed one sim step.

        Args:
            cube_z: Current cube z-coordinate (m).
            dist_3d: Current 3D gripper-to-cube distance (m).
            step: Sim step index (monotonic).
            arm_joint_pos: Current joint positions in rad (>=5-dim).
                Required for Stage-3 home-reached detection. If None,
                Stage 3 cannot be reached.
        """
        self._step = step
        self._min_dist = min(self._min_dist, dist_3d)
        self._max_height = max(self._max_height, cube_z)

        # --- Stage transitions (monotonic) ---
        prev_stage = self._max_stage
        if dist_3d < REACH_DIST and self._max_stage < 1:
            self._max_stage = 1
            self._time_to_reach = step

        if cube_z > self._initial_cube_z + LIFT_DELTA_REL and self._max_stage < 2:
            self._max_stage = 2
            self._time_to_lift = step

        # Stage 3 gating: only after Stage 2 (no home-success without lift).
        if self._max_stage >= 2 and arm_joint_pos is not None:
            arm_err = max(
                abs(arm_joint_pos[i] - self._home_pose_rad[i]) for i in range(5)
            )
            arm_at_home = arm_err < self._reach_home_rad
            cube_still_held = cube_z > self._initial_cube_z - CUBE_HOLD_TOLERANCE

            if arm_at_home and cube_still_held:
                self._home_hold_counter += 1
                self._home_hold_total += 1
                if self._home_hold_counter >= HOME_HOLD_STEPS and self._max_stage < 3:
                    self._max_stage = 3
                    self._time_to_home = step
            else:
                self._home_hold_counter = 0

        # Track last stage advance for plateau-ET.
        if self._max_stage > prev_stage:
            self._last_stage_advance_step = step

        # --- Attempt tracking (hysteresis) ---
        if cube_z > self._initial_cube_z + ATTEMPT_UP and not self._in_attempt:
            self._in_attempt = True
            self._num_attempts += 1
        elif cube_z < self._initial_cube_z + ATTEMPT_DOWN and self._in_attempt:
            self._in_attempt = False

        # --- Early termination ---
        if self._termination_reason is not None:
            return

        if cube_z < self._initial_cube_z - ET_CUBE_LOST_DELTA:
            self._termination_reason = "cube_lost"
            self._termination_step = step

        elif (step == ET_APPROACH_STEP
              and self._max_stage < 1     # only if gripper never reached the cube
              and self._initial_dist_3d > 0
              and dist_3d > self._initial_dist_3d * (1 - ET_APPROACH_RATIO)):
            self._termination_reason = "no_approach"
            self._termination_step = step

        elif (self._max_stage >= 1
              and self._max_stage < 3):
            stage_name = STAGES[self._max_stage]  # "1_reach" or "2_lift"
            threshold = self._plateau_steps.get(stage_name, 600)
            if (step - self._last_stage_advance_step) >= threshold:
                self._termination_reason = f"plateau_{stage_name}"
                self._termination_step = step

    # ------------------------------------------------------------------
    @property
    def should_terminate(self) -> bool:
        """True when the episode should stop — either early failure or success.

        Early-termination on success is a speed optimization: once Stage 3
        (arm_at_home ∧ cube_held, 0.5s sustained) is reached, the remaining
        frames carry no signal. Stopping saves ~60-70% of eval wall-time for
        successful episodes. Fail-episodes still run to EPISODE_LENGTH_S to
        give ET-1 (no_approach @ step 600) its full detection window.
        """
        return self._termination_reason is not None or self._max_stage == 3

    @property
    def result(self) -> dict:
        """Episode summary dict — maps directly to CSV row fields."""
        reason = self._termination_reason or (
            "success" if self._max_stage == 3 else "timeout"
        )
        return {
            "task_progress_label": STAGES[self._max_stage],
            "task_progress_score": SCORES[self._max_stage],
            "time_to_reach": self._time_to_reach,
            "time_to_lift": self._time_to_lift,
            "time_to_home": self._time_to_home,
            "min_gripper_cube_dist": round(self._min_dist, 4),
            "max_height": round(self._max_height, 4),
            "home_hold_total_steps": self._home_hold_total,
            "initial_cube_z": round(self._initial_cube_z, 4),
            "num_attempts": self._num_attempts,
            "termination_reason": reason,
            "termination_step": (
                self._termination_step if self._termination_step > 0
                else self._step
            ),
            "success": self._max_stage == 3,
        }
