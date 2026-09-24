"""CSV + JSON results writer for evaluation runs.

Writes per-episode results to CSV (Pandas-ready) and an aggregated
summary to JSON. Handles multi-run statistics (mean ± std).
"""

VERSION = "2.1.0"

import csv
import json
from pathlib import Path

from so101_mvbench.logging import get_logger

logger = get_logger(__name__)

# Schema aligns with eval_tracker v2.0.0 (Stages: miss/reach/lift/home).
# Removed: time_to_pick (merged with time_to_lift), hold_duration_steps.
# Added:   time_to_home, home_hold_total_steps.
CSV_FIELDNAMES = [
    "run", "episode", "scene_state_idx", "env_id",
    "task_progress_label", "task_progress_score",
    "time_to_reach", "time_to_lift", "time_to_home",
    "min_gripper_cube_dist", "max_height", "home_hold_total_steps",
    "initial_cube_z", "num_attempts",
    "termination_reason", "termination_step",
    "success", "steps",
    "randomize", "cube_noise", "cube_rot_noise", "checkpoint",
    "cube_x_m", "cube_y_m", "cube_yaw_deg",
    "training_cube_x_m", "training_cube_y_m",
    # Articulation-collapse indicator (max single-frame arm-joint jump, deg).
    # Written by cylroom_eval; blank for evaluators that don't track it.
    "max_joint_delta_deg",
    # Camera-importance augmentation condition (cylroom_eval --augment_*).
    # Written by cylroom_eval ("none"/0.0 when off); blank for other evaluators.
    "augment_cameras", "augment_kind", "augment_strength",
]


class ResultsWriter:
    """Writes evaluation results to CSV + JSON."""

    def __init__(self, output_dir: Path | None, config: dict):
        """
        Args:
            output_dir: Directory for output files. None = next to checkpoint.
            config: Eval config dict with keys: policy_path, task, randomize,
                    cube_noise, cube_rot_noise, scene_state, num_runs,
                    num_episodes_per_run.
        """
        if output_dir:
            self._out_dir = Path(output_dir)
            self._out_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._out_dir = Path(config["policy_path"]).parent.parent.parent
        self._config = config

    def write(
        self,
        results: list[dict],
        run_success_rates: list[float],
        video_settings: dict | None = None,
    ) -> None:
        """Write CSV + JSON + log summary table."""
        if not results:
            return

        self._write_csv(results)
        self._write_json(results, run_success_rates, video_settings)
        self._log_summary(results)

    def _write_csv(self, results: list[dict]) -> None:
        csv_path = self._out_dir / f"eval_results_random_{self._config['randomize']}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                row = dict(r)
                row["randomize"] = self._config["randomize"]
                row["cube_noise"] = self._config["cube_noise"]
                row["cube_rot_noise"] = self._config["cube_rot_noise"]
                row["checkpoint"] = Path(self._config["policy_path"]).parent.name
                writer.writerow(row)
        logger.info("CSV saved to %s", csv_path)

    def _write_json(
        self,
        results: list[dict],
        run_success_rates: list[float],
        video_settings: dict | None,
    ) -> None:
        total = len(results)
        successes = sum(1 for r in results if r["success"])
        rate = (successes / total * 100.0) if total else 0.0

        summary = {
            "policy_path": self._config["policy_path"],
            "task": self._config["task"],
            "randomize": self._config["randomize"],
            "cube_noise": self._config["cube_noise"],
            "cube_rot_noise": self._config["cube_rot_noise"],
            "scene_state": self._config.get("scene_state", ""),
            "num_runs": self._config.get("num_runs", 1),
            "num_episodes_per_run": self._config.get("num_episodes_per_run", total),
            "total_episodes": total,
            "total_successes": successes,
            "overall_success_rate": round(rate, 1),
            "avg_max_height": round(
                sum(r["max_height"] for r in results) / total, 4
            ),
        }
        if video_settings:
            summary["video_settings"] = video_settings

        if len(run_success_rates) > 1:
            mean_r = sum(run_success_rates) / len(run_success_rates)
            std_r = (sum((r - mean_r) ** 2 for r in run_success_rates)
                     / len(run_success_rates)) ** 0.5
            summary["per_run_success_rates"] = [round(r, 1) for r in run_success_rates]
            summary["mean_success_rate"] = round(mean_r, 1)
            summary["std_success_rate"] = round(std_r, 1)

        json_path = self._out_dir / "eval_summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Summary saved to %s", json_path)

    def _log_summary(self, results: list[dict]) -> None:
        logger.info("%-4s %-8s %-8s %-10s %-10s %-8s %-12s",
                     "Run", "Episode", "Scene", "Stage", "MaxH", "Attempts", "Termination")
        for r in results:
            logger.info(
                "%-4d %-8d %-8d %-10s %-10.4f %-8d %-12s",
                r["run"], r["episode"], r["scene_state_idx"],
                r["task_progress_label"], r["max_height"],
                r["num_attempts"], r["termination_reason"],
            )
