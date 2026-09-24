"""eval_report: standalone post-analysis (pandas + matplotlib).

Reads eval_results_random_*.csv + eval_summary.json + quality_gate.json from
an eval output directory and produces:

    stage_outcome.png           — the 4 task stages, split by termination reason
    attempts_histogram.png      — histogram of grasp attempts per episode
    cube_position_analysis.png  — cube-xy scatter colored by success/stage
    cube_rotation_analysis.png  — cube-yaw histogram split by success
    episode_timeline.png        — per-episode outcome Gantt chart
    timing_and_distance.png     — time_to_reach/lift/home + min dist scatter
    report.txt                  — plain-text summary
    report.md                   — markdown-formatted briefing

Standalone: no Jupyter / nbconvert dependency, no simulator needed.

Usage:
    eval_report --dir <eval_output>     # via pyproject [project.scripts] entry-point

Metrics doc: see ``METRICS.md`` next to this file.

Version 1.1.0 — added cube_position/rotation/timeline/timing plots.
"""

# Version history:
# Version 1.2.0 — cube_position rebuilt: marker-shape encoding for roll,
# holdout edge style, roll-stratified success rate panel.
# Version 1.3.0 — cube_position restored to the training-overlay style:
# training-pos overlay, yaw arrows, FAIL=X/SUCCESS=O, and a
# distance-to-nearest-training boxplot. The v1.2.0 shape-encoded view moved
# to a separate artifact
# `holdout_vs_training_analysis.png` that is skipped when no
# holdout_episodes.json is discoverable.
# Version 1.4.0 — cube-yaw is now sourced from `binmap.json` (authoritative,
# same source scene_loader.py uses). Previous versions derived
# it from `scene_state.object_init_rpy_rad` which is unreliable
# pre-scene_state.py v1.1.0 (quat XYZW/WXYZ swap). Removed
# `cube_roll_deg` / `cube_pitch_deg` CSV enrichment — recording
# pipeline only varies yaw (bin_spawner.py:812-821), so
# roll/pitch carry no information. Plot labels updated
# "roll" → "yaw" across both cube_position and holdout plots.
# Version 1.5.0 — cylroom/OOD-scene hardening + new figure:
# (a) cube_position "Training pos" overlay only when the bin was
# actually trained (data-driven via train_manifest.json);
# OOD-holdout bins get a neutral label. Episode numbers drawn
# at each arrow tip.
# (b) timing_and_distance uses RELATIVE lift height
# (max_height − initial_cube_z) — absolute world-Z clumped all
# points near the table height (~0.79 m) in the cylroom scene.
# (c) termination_reasons no longer silently drops plateau reasons
# (TERM_ORDER extended with plateau_1_reach/plateau_2_lift;
# unknown reasons are appended, never reindex-dropped).
# (d) cube_rotation yaw bins are data-driven (was hardcoded
# −180..180, squashing the few discrete yaw variants).
# (e) NEW stage_sunburst.png — radial stage-progression funnel
# (area = episode count).
# (f) episode_timeline: speaking termination identifiers + a
# footer glossary (minimal IDs in-plot, full text below axes).
# Version 1.6.0 — cube_position: the second panel (distance-to-training boxplot)
# and its dead distance computation are removed: in ID mode
# every distance is 0, so the panel rendered empty. That
# comparison lives in holdout_vs_training_analysis.png.
# The per-episode numbers from v1.5.0 (a) were dropped here and
# restored in v1.8.0 -- see there.
# Version 1.7.0 — figure-standards pass: every axis carries a label, units moved
# to square brackets, column names replaced by plain labels, the
# em-dash dropped from rendered titles, TERM_ORDER reordered so the
# terminal "success" sits at the right end, and the legend helper
# moved to plot_style.py so the other generators share it.
# Version 1.8.0 — cube_position: episode numbers are back (they answer "which
# episode failed here", which nothing else shows per position),
# now with a white outline instead of a filled box so a number
# no longer covers the neighbouring arrow. The subtitle is gone:
# at noise=0 it only restated the default, and its one piece of
# real content, what the arrow direction means, is a legend
# entry now. cube_rotation draws one slim bar per recorded yaw
# instead of 5-degree bins, which had implied a range of angles
# that was never recorded, and rounded 22.5 to 20 on the ticks.
# Version 1.9.0 — stage_distribution, termination_reasons and stage_sunburst are
# replaced by ONE stage_outcome.png. Over 10,800 episodes the three
# showed the same four numbers: every stage had exactly one dominant
# termination reason, and the 12 episodes that broke the mapping
# (timeout instead of plateau) were invisible in the stage chart.
# The reason is now a stacked segment inside the stage bar.
# episode_timeline lost its four-clause title and its baked-in
# glossary box; the glossary is a table in report.md instead.

from __future__ import annotations

__version__ = "1.9.0"

import argparse
import json
import math
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patheffects as mpatheffects
import matplotlib.pyplot as plt
import pandas as pd

from .plot_style import legend_below

STAGE_ORDER = ["0_miss", "1_reach", "2_lift", "3_home"]
STAGE_COLORS = {
    "0_miss": "#d32f2f",
    "1_reach": "#ff9800",
    "2_lift": "#fdd835",
    "3_home": "#4caf50",
}
# Ordered by pipeline stage: success first, then the early-termination reasons in
# funnel order. plateau_1_reach / plateau_2_lift are the dominant failure modes in
# the plateau-ET eval (added 2026-05); they MUST be listed or _plot_terminations'
# reindex silently drops them (old TERM_ORDER predated plateau-ET → ~580 episodes
# vanished from the cylroom_v2 termination charts).
# Pipeline order: earliest failure first, the terminal success last, so both the
# bar chart and the text report read left to right along the task.
TERM_ORDER = [
    "no_approach", "table_collision", "plateau_1_reach", "cube_lost",
    "plateau_2_lift", "timeout", "success",
]
TERM_COLORS = {
    "success": "#4caf50",
    "no_approach": "#ff9800",
    "table_collision": "#8e24aa",  # policy drives gripper into the table (cylroom_eval v0.4.0)
    "plateau_1_reach": "#fb8c00",
    "cube_lost": "#d32f2f",
    "plateau_2_lift": "#e65100",
    "timeout": "#9e9e9e",
}


def _load(eval_dir: Path) -> tuple[pd.DataFrame, dict, dict | None]:
    """Load CSV, summary JSON, and (optional) quality-gate JSON."""
    csv_files = sorted(eval_dir.glob("eval_results_random_*.csv"))
    if not csv_files:
        raise SystemExit(f"no eval_results_*.csv in {eval_dir}")
    df = pd.read_csv(csv_files[0])
    summary = json.loads((eval_dir / "eval_summary.json").read_text())
    gate_path = eval_dir / "quality_gate.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else None
    return df, summary, gate


def _title(bin_label: str, text: str) -> str:
    """Prefix a plot title with the bin identifier, if provided."""
    return f"{bin_label} - {text}" if bin_label else text


def _plot_stage_outcome(df: pd.DataFrame, out_path: Path, bin_label: str = "") -> None:
    """How far the episodes got, and why each one stopped, in one chart.

    This replaces the former ``stage_distribution`` / ``termination_reasons``
    pair. Over 10,800 episodes of the cylinder-room corpus the two carried the
    same information: every stage had exactly one dominant reason, and only 12
    episodes broke the mapping by timing out instead of plateauing. Those 12
    were invisible in the stage chart, because a timeout at stage 1 looked like
    any other episode stuck at reach. Stacking the reason inside the stage bar
    keeps the ladder readable and puts the exception where it can be seen.
    """
    stages = df["task_progress_label"].value_counts().reindex(
        STAGE_ORDER, fill_value=0)
    raw = df["termination_reason"].value_counts()
    reasons = [t for t in TERM_ORDER if t in raw.index]
    reasons += [t for t in raw.index if t not in TERM_ORDER]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bottoms = [0] * len(STAGE_ORDER)
    for reason in reasons:
        heights = [int(((df["task_progress_label"] == stage)
                        & (df["termination_reason"] == reason)).sum())
                   for stage in STAGE_ORDER]
        if not any(heights):
            continue
        ax.bar(STAGE_ORDER, heights, bottom=bottoms,
               color=TERM_COLORS.get(reason, "#9e9e9e"),
               edgecolor="black", linewidth=0.5, label=reason)
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    for x, total in zip(STAGE_ORDER, stages.values):
        if total > 0:
            ax.text(x, total + 0.2, str(int(total)),
                    ha="center", va="bottom", fontweight="bold")

    ax.set_xlabel("Task stage")
    ax.set_ylabel("Episodes")
    # Says what the reader gets, not which two columns were joined. The old
    # "Task Progress Stage and Termination" was the two former titles glued
    # together, and "task progress stage" is the CSV column name.
    ax.set_title(_title(bin_label, "How far the episodes got, and why they stopped"))
    legend_below(ax, max_ncol=4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _load_binmap_yaw(scene_state_dir: Path) -> dict[int, float]:
    """Read cube-yaw per scene-state index from the dataset's binmap.json.

    Convention (design01): ``datasets/liftcube_binned/<bin>/meta/`` is the
    scene_state dir; ``datasets/liftcube_binned/<bin>/binmap.json`` sits one
    level up. Binmap stores the authoritative yaw applied at recording time
    via ``bin_spawner.apply_cube_pose(..., yaw_rad)`` (bin_spawner.py:793).

    Returns empty dict when:
      - binmap.json does not exist (legacy non-binned dataset)
      - parsing fails
      - binmap schema lacks ``episodes[*].episode_idx`` / ``.yaw_rad`` keys
    """
    candidates = [
        scene_state_dir.parent / "binmap.json",
        scene_state_dir / "binmap.json",
    ]
    for binmap_path in candidates:
        if binmap_path.exists():
            try:
                d = json.loads(binmap_path.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
            out: dict[int, float] = {}
            for e in d.get("episodes", []):
                idx = e.get("episode_idx")
                yaw = e.get("yaw_rad")
                if idx is None or yaw is None:
                    continue
                # Episodes can appear multiple times (recording retries).
                # Last-wins is the convention used by bin_progress / validate
                # tooling — mirror it here for consistency.
                out[int(idx)] = math.degrees(float(yaw))
            return out
    return {}


def _enrich_with_scene_state(df: pd.DataFrame, scene_state_dir: Path) -> pd.DataFrame:
    """Overwrite `cube_yaw_deg` with values sourced from binmap.json (authoritative).

    Intentionally NOT using scene_state.object_init_rpy_rad — that field was
    unreliable pre-scene_state.py v1.1.0 due to an XYZW/WXYZ quaternion
    misread (see CLAUDE.md §2). The binmap, written directly by the recording
    spawner, was always correct.

    Always overwrites the column when binmap is available (not merely
    backfills) because pre-v1.4 eval_policy.py wrote `cube_yaw_deg` from
    `action_pad_yaw_rad` — the pad rotation, not the cube rotation — which
    is constant and meaningless for plotting. Binmap has the true cube yaw.

    Skipped when CSV contains ``training_cube_x_m`` — that column signals
    the placed_cube pipeline (v1.0.0+) which already writes correct yaw.

    No-op if scene_state_dir is missing or no binmap.json is discoverable.
    """
    if "training_cube_x_m" in df.columns:
        return df
    if "scene_state_idx" not in df.columns or not scene_state_dir.exists():
        return df
    yaw_by_idx = _load_binmap_yaw(scene_state_dir)
    if not yaw_by_idx:
        return df
    df = df.copy()
    df["cube_yaw_deg"] = [
        yaw_by_idx.get(int(i), float("nan")) for i in df["scene_state_idx"]
    ]
    return df


def _discover_holdout_episodes(policy_path: str | Path) -> set[int]:
    """Resolve holdout_episodes.json relative to the policy's run_dir.

    Policy path convention: <run_dir>/checkpoints/<step>/pretrained_model
    → run_dir is parents[2]. holdout_episodes.json (design01) sits there.
    Returns empty set if no file or legacy (non-holdout) training.
    """
    try:
        run_dir = Path(policy_path).resolve().parents[2]
    except (IndexError, OSError):
        return set()
    ho = run_dir / "holdout_episodes.json"
    if not ho.exists():
        return set()
    try:
        d = json.loads(ho.read_text())
        return set(int(x) for x in d.get("holdout_episodes", []))
    except (json.JSONDecodeError, ValueError, TypeError):
        return set()


def _bin_was_trained(policy_path: str | Path, bin_label: str) -> bool | None:
    """Whether ``bin_label`` was part of the policy's training set.

    Reads the policy's ``train_manifest.json`` (written by write_train_manifest;
    sits at <run_dir>/train_manifest.json, run_dir = policy parents[2]). Returns:
      - True  if bin_label is in ``train_bins``,
      - False if bin_label is in ``heldout_bins`` (OOD — never trained),
      - None  if no manifest / bin not listed (unknown → caller keeps legacy
              behaviour so non-manifest runs are unaffected).

    Used so the cube-position plot only draws the "Training pos" overlay when
    training data actually existed at those positions — for OOD-holdout bins the
    eval positions were never trained and labelling them "Training" is wrong.
    """
    try:
        run_dir = Path(policy_path).resolve().parents[2]
    except (IndexError, OSError):
        return None
    mf = run_dir / "train_manifest.json"
    if not mf.exists():
        return None
    try:
        d = json.loads(mf.read_text())
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    if bin_label in (d.get("heldout_bins") or []):
        return False
    if bin_label in (d.get("train_bins") or []):
        return True
    return None


def _pick_varying_rotation_axis(df: pd.DataFrame) -> tuple[str | None, str]:
    """Return (column, label) for the varying rotation axis, or (None, "").

    Recording pipeline pure-rotates the cube around Z (yaw) — see
    ``bin_spawner.py:812-821``. There is no roll or pitch variation to
    visualize. We nonetheless expose this function because the plot code
    still supports rotation-axis-less datasets (constant yaw → arrow-free
    scatter).
    """
    col = "cube_yaw_deg"
    if col in df.columns and df[col].nunique(dropna=True) > 1:
        return col, "yaw"
    return None, ""


def _draw_rotation_arrow(
    ax, x: float, y: float, angle_deg: float,
    color: str, arrow_length: float = 0.008, alpha: float = 0.7,
    angle_offset_deg: float = 0.0,
) -> None:
    """Small arrow at (x,y) whose direction encodes a rotation angle.

    ``angle_offset_deg`` follows the convention used for each rotation axis:
      - yaw  → +90° (0° yaw = forward/+y), matching the t36 Cherry-Shen pattern
               from ``auswertung.ipynb`` Cell 10 lines 22-29
      - roll → 0°  (arrow points +x for roll=0°, rotates CCW with increasing roll)
      - pitch → 0°
    """
    import numpy as np
    rad = np.deg2rad(angle_deg + angle_offset_deg)
    dx = arrow_length * np.cos(rad)
    dy = arrow_length * np.sin(rad)
    ax.arrow(x, y, dx, dy, head_width=0.003, head_length=0.002,
             fc=color, ec=color, alpha=alpha, zorder=5)


# Per-axis conventions used when the corresponding column is the varying one.
_ROTATION_AXIS_OFFSETS = {"yaw": 90.0, "roll": 0.0, "pitch": 0.0}


def _load_training_positions(
    scene_state_dir: Path, holdout_episodes: set[int]
) -> list[dict]:
    """Load training scene-state rows as list of dicts, excluding holdout episodes.

    Each row has keys: x, y. Rotation fields are deliberately not returned —
    ``_plot_cube_positions`` only uses positions for the blue-circle overlay,
    and pre-v1.4 rotation reads from scene_state.rpy were unreliable. Consumers
    needing cube yaw should call ``_load_binmap_yaw`` instead.
    """
    rows: list[dict] = []
    for f in sorted(scene_state_dir.glob("episode_*_scene_state.json")):
        try:
            ep_idx = int(f.name.split("_")[1])
        except (ValueError, IndexError):
            continue
        if ep_idx in holdout_episodes:
            continue
        ss = json.loads(f.read_text())
        pos = ss.get("object_init_pos_m", {})
        rows.append({"x": pos.get("x", 0.0), "y": pos.get("y", 0.0)})
    return rows


def _plot_cube_positions(
    df: pd.DataFrame,
    out_path: Path,
    scene_state_dir: Path | None = None,
    holdout_episodes: set[int] | None = None,
    cube_noise: float = 0.0,
    cube_rot_noise: float = 0.0,
    bin_was_trained: bool | None = None,
) -> None:
    """t36-style: training-pos overlay + rotation-arrows, single panel.

    Layout follows `auswertung.ipynb` Cell 10 from `t36_eval_benchmark/
    run_20260414_181108`, adapted for datasets where the varying rotation
    axis is roll or pitch (not yaw). Works for all eval modes.

    One axes: training positions (blue open circles, deduplicated to unique XY),
    eval success (green filled circle), eval fail (red X), a rotation-arrow per
    episode. Nothing is labelled per episode.

    The distance-to-nearest-training boxplot that used to sit in a second panel
    is gone: in ID mode every distance is 0, so the panel rendered empty except
    for a pointer textbox. That comparison lives in
    `holdout_vs_training_analysis.png`.

    Arrow angle encodes the **first varying** rotation axis among
    {roll, pitch, yaw}. Per-axis visual convention via ``_ROTATION_AXIS_OFFSETS``.
    If no axis varies, arrows are omitted entirely (pure scatter).

    No-op if `scene_state_dir` is missing or empty.
    """
    if "cube_x_m" not in df.columns or "cube_y_m" not in df.columns:
        return
    if scene_state_dir is None or not scene_state_dir.exists():
        return
    import numpy as np

    holdout_episodes = holdout_episodes or set()
    train_rows = _load_training_positions(scene_state_dir, holdout_episodes)
    if not train_rows:
        return

    # Bin identifier for the title, extracted from scene_state_dir. E.g.
    # ``/.../datasets/liftcube_binned/bin_c1_r1/meta`` → ``bin_c1_r1``.
    bin_label = scene_state_dir.parent.name if scene_state_dir.name == "meta" \
        else scene_state_dir.name

    # Markers are plotted at the TRUE (cube_x_m, cube_y_m). No synthetic jitter.
    # Three roll repeats at the same XY will STACK on top of each other; the
    # visual disambiguation is left to the rotation-arrows, which fan out at
    # different roll angles from the shared origin, and to the complementary
    # `holdout_vs_training_analysis.png` plot.
    arrow_len = 0.007  # 7 mm — arrow tips stay inside half the 20 mm grid cell

    # Pick varying rotation axis for arrow direction (roll → pitch → yaw).
    rot_col, rot_label = _pick_varying_rotation_axis(df)
    angle_offset = _ROTATION_AXIS_OFFSETS.get(rot_label, 0.0)

    # Unique training XY for the blue-circle overlay. The training rotation
    # variants per XY are NOT shown as additional arrows — the eval arrows
    # already carry that information. A single open circle per XY stays clean.
    train_unique: dict[tuple[float, float], list[dict]] = {}
    for r in train_rows:
        key = (round(r["x"], 4), round(r["y"], 4))
        train_unique.setdefault(key, []).append(r)
    n_unique_train = len(train_unique)

    success_df = df[df["success"] == True]  # noqa: E712
    fail_df = df[df["success"] == False]    # noqa: E712

    fig, ax = plt.subplots(figsize=(8, 6))
    # 1. Position overlay (deduped, one open circle per unique XY). Only labelled
    #    "Training pos" when this bin was ACTUALLY in the policy's training set.
    #    For OOD-holdout bins (bin_was_trained is False) the eval positions were
    #    never trained → draw a neutral "Eval-pos grid" circle instead (or skip).
    tu_x = [k[0] for k in train_unique]
    tu_y = [k[1] for k in train_unique]
    if bin_was_trained is True:
        ax.scatter(tu_x, tu_y, c="none", edgecolor="#1565c0", s=180,
                   linewidth=2,
                   label=f"Training pos ({n_unique_train} unique XY, "
                         f"{len(train_rows)} scene-states)",
                   zorder=2, marker="o")
    elif bin_was_trained is None:
        # Unknown provenance (no manifest) → legacy neutral grid, no claim.
        ax.scatter(tu_x, tu_y, c="none", edgecolor="#90a4ae", s=180,
                   linewidth=2,
                   label=f"Cube-pos grid ({n_unique_train} unique XY)",
                   zorder=2, marker="o")
    # bin_was_trained is False (OOD-holdout): no overlay — these positions were
    # never trained; SUCCESS/FAIL markers below carry all the information.

    # 2. Eval rotation-arrows (from true XY, direction = varying rotation axis)
    #    plus the episode number at each tip, so a failure can be traced back to
    #    the episode that produced it. The number sits BEYOND the arrow head and
    #    carries a white outline rather than a filled box: a box at this density
    #    (3 yaw variants per position) covered the neighbouring arrow.
    ep_col = "episode" if "episode" in df.columns else None
    halo = [mpatheffects.withStroke(linewidth=2.2, foreground="white")]
    if rot_col:
        for _, row in df.iterrows():
            color = "#4caf50" if row["success"] else "#d32f2f"
            _draw_rotation_arrow(ax, row["cube_x_m"], row["cube_y_m"],
                                 row[rot_col], color, alpha=0.7,
                                 arrow_length=arrow_len,
                                 angle_offset_deg=angle_offset)
            if ep_col is not None:
                rad = np.deg2rad(row[rot_col] + angle_offset)
                reach = arrow_len + 0.0034
                ax.text(row["cube_x_m"] + reach * np.cos(rad),
                        row["cube_y_m"] + reach * np.sin(rad),
                        str(int(row[ep_col])), fontsize=6.5, color=color,
                        ha="center", va="center", zorder=7, fontweight="bold",
                        path_effects=halo)
    elif ep_col is not None:
        # No rotation variation, so no arrows to hang the number on.
        for _, row in df.iterrows():
            color = "#2e7d32" if row["success"] else "#b71c1c"
            ax.text(row["cube_x_m"], row["cube_y_m"] + 0.0018,
                    str(int(row[ep_col])), fontsize=6.5, color=color,
                    ha="center", va="bottom", zorder=6, fontweight="bold",
                    path_effects=halo)

    # 3. Eval markers at the TRUE XY (no jitter). Stacked repeats with identical
    #    outcome blend visually; mixed outcomes show red X on top of green O via
    #    zorder ordering.
    if len(success_df) > 0:
        ax.scatter(success_df["cube_x_m"], success_df["cube_y_m"],
                   c="#4caf50", s=60, edgecolor="black", linewidth=0.5,
                   label=f"SUCCESS ({len(success_df)})", zorder=4, marker="o")
    if len(fail_df) > 0:
        ax.scatter(fail_df["cube_x_m"], fail_df["cube_y_m"],
                   c="#d32f2f", s=90, linewidth=1.8,
                   label=f"FAIL ({len(fail_df)})", zorder=5, marker="x")

    ax.set_xlabel("Cube X [m]", fontsize=12)
    ax.set_ylabel("Cube Y [m]", fontsize=12)
    if cube_noise == 0.0 and cube_rot_noise == 0.0:
        # Nothing was jittered, so there is nothing to announce.
        subtitle = "OOD-Position grid" if len(df) > 27 else ""
    else:
        subtitle = f"pos noise=±{cube_noise}m, rot noise=±{cube_rot_noise}°"
    if bin_was_trained is False:
        title_lead = f"{bin_label} - Cube Pose: Eval (OOD-Holdout, not trained)"
    elif bin_was_trained is True:
        title_lead = f"{bin_label} - Cube Pose: Eval vs Training"
    else:
        title_lead = f"{bin_label} - Cube Pose: Eval"
    ax.set_title(f"{title_lead}\n({subtitle})" if subtitle else title_lead,
                 fontsize=11)
    # Legend as a horizontal row below the x-axis. The episode count and the XY
    # ranges that used to sit in a box on the right are metadata, not plotted
    # data, so they belong in the caption rather than inside the figure.
    if rot_col:
        handles, _ = ax.get_legend_handles_labels()
        handles.append(mlines.Line2D([], [], marker=(3, 0, 0), linestyle="",
                                     color="#9e9e9e", markersize=9,
                                     label=f"arrow direction = cube {rot_label}"))
        legend_below(ax, handles=handles, fontsize=8, max_ncol=4)
    else:
        legend_below(ax, fontsize=8, max_ncol=3)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_holdout_vs_training(
    df: pd.DataFrame,
    out_path: Path,
    holdout_episodes: set[int] | None = None,
    bin_label: str = "",
) -> None:
    """Holdout-vs-Training generalization view (design01 runs).

    Orthogonal to the t36-style `_plot_cube_positions`: focuses on the
    training-vs-holdout generalization axis rather than position-offset.

    - Marker shape encodes the rotation axis that actually varies in the run
      (roll for bin_c1_r1, yaw for classic bins). Constant-axis data gets no
      shape encoding — everything becomes 'o'.
    - Episodes that share the exact same (x,y) are jittered into a small circle
      around the base position so overlapping repeats become visible.
    - Marker edge distinguishes training (thin gray) from holdout (thick black).
    - Right panel: success rate stratified by the same rotation axis.

    Skipped (no-op) if no holdout episodes are passed — this plot is only
    meaningful for design01 runs with an explicit holdout set.
    """
    if not holdout_episodes:
        return
    if "cube_x_m" not in df.columns or "cube_y_m" not in df.columns:
        return

    import numpy as np

    df = df.copy()
    df["is_holdout"] = (
        df["scene_state_idx"].isin(holdout_episodes)
        if "scene_state_idx" in df.columns
        else False
    )

    rot_col, rot_label = _pick_varying_rotation_axis(df)
    if rot_col:
        unique_rots = sorted(df[rot_col].unique())
        shape_map = {r: m for r, m in zip(unique_rots, ["o", "s", "D", "^", "v", "P"])}
    else:
        unique_rots, shape_map = [], {}

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel("Cube X [m]")
    axes[0].set_ylabel("Cube Y [m]")

    # Panel A — jittered XY × shape × color × edge
    sort_col = rot_col if rot_col else "episode"
    for (x, y), group in df.groupby(["cube_x_m", "cube_y_m"]):
        n = len(group)
        jitter_r = 0.003 if n > 1 else 0.0
        for k, (_, row) in enumerate(group.sort_values(sort_col).iterrows()):
            theta = 2 * np.pi * k / max(n, 1) + np.pi / 2
            ox, oy = jitter_r * np.cos(theta), jitter_r * np.sin(theta)
            color = "#4caf50" if row["success"] else "#d32f2f"
            marker = shape_map.get(row[rot_col], "o") if rot_col else "o"
            edge_color = "black" if row["is_holdout"] else "#555555"
            edge_width = 2.0 if row["is_holdout"] else 0.6
            axes[0].scatter([x + ox], [y + oy], s=150, c=color, marker=marker,
                            edgecolors=edge_color, linewidths=edge_width,
                            alpha=0.9, zorder=3)

    n_succ = int(df["success"].sum())
    n_fail = int((~df["success"]).sum())
    n_holdout = int(df["is_holdout"].sum())
    n_train = int((~df["is_holdout"]).sum())
    handles = [
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor="#4caf50",
                      markeredgecolor="black", markersize=11,
                      label=f"success ({n_succ})"),
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor="#d32f2f",
                      markeredgecolor="black", markersize=11,
                      label=f"fail ({n_fail})"),
    ]
    if rot_col:
        for r in unique_rots:
            handles.append(mlines.Line2D([], [], marker=shape_map[r], color="w",
                                         markerfacecolor="gray", markeredgecolor="black",
                                         markersize=10,
                                         label=f"{rot_label}={r:.1f}°"))
    if n_holdout > 0:
        handles.append(mlines.Line2D([], [], marker="o", color="w", markerfacecolor="gray",
                                     markeredgecolor="black", markeredgewidth=2.0,
                                     markersize=11, label=f"holdout ({n_holdout})"))
        handles.append(mlines.Line2D([], [], marker="o", color="w", markerfacecolor="gray",
                                     markeredgecolor="#555555", markeredgewidth=0.6,
                                     markersize=11, label=f"training ({n_train})"))
    legend_below(axes[0], handles=handles, fontsize=8, max_ncol=3)
    title_extra = f" | {rot_label}=shape" if rot_col else ""
    axes[0].set_title(_title(
        bin_label,
        f"Cube XY × Success × Train/Holdout{title_extra}  (n={len(df)})",
    ))

    # Panel B — success-rate stratification by the varying rotation axis
    ax2 = axes[1]
    if rot_col:
        stats = df.groupby(rot_col).agg(n=("success", "size"),
                                        n_succ=("success", "sum")).reset_index()
        stats["sr"] = 100.0 * stats["n_succ"] / stats["n"]
        labels = [f"{rot_label}={r:.1f}°\n({s}/{n})"
                  for r, n, s in zip(stats[rot_col], stats["n"], stats["n_succ"])]
        bar_colors = [
            "#4caf50" if sr == 100 else "#ff9800" if sr >= 80 else "#d32f2f"
            for sr in stats["sr"]
        ]
        bars = ax2.bar(labels, stats["sr"], color=bar_colors,
                       edgecolor="black", linewidth=0.8)
        ax2.set_ylabel("Success rate [%]")
        ax2.set_ylim(0, 110)
        ax2.set_title(_title(bin_label, f"Success Rate stratified by {rot_label}"))
        ax2.grid(True, alpha=0.3, axis="y")
        for bar, sr in zip(bars, stats["sr"]):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 1.5, f"{sr:.1f}%",
                     ha="center", fontsize=10, fontweight="bold")
    else:
        # Fallback — legacy per-stage XY scatter when no rotation variation
        ax2.set_aspect("equal")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel("Cube X [m]")
        ax2.set_ylabel("Cube Y [m]")
        for stage in STAGE_ORDER:
            sub = df[df["task_progress_label"] == stage]
            if len(sub) == 0:
                continue
            ax2.scatter(sub["cube_x_m"], sub["cube_y_m"],
                        c=STAGE_COLORS[stage], s=80, alpha=0.7,
                        label=f"{stage} ({len(sub)})",
                        edgecolors="black", linewidths=0.5)
        ax2.set_title("Cube XY × Max Stage")
        legend_below(ax2, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_cube_rotations(df: pd.DataFrame, out_path: Path, bin_label: str = "") -> None:
    """Cube yaw split by success. Diagnoses yaw-dependent failures.

    The recording pipeline spawns a handful of DISCRETE yaw variants, so the
    default is one narrow bar per value that actually occurs. A histogram would
    be a lie here: five-degree bins drew a block from 0 to 5 degrees where the
    data holds exactly one angle, and the reader sees a range that was never
    recorded. Only when the yaws really are spread out does binning make sense.
    """
    if "cube_yaw_deg" not in df.columns:
        return
    import numpy as np
    fig, ax = plt.subplots(figsize=(8, 4))
    yaws = df["cube_yaw_deg"].values
    successes = df["success"].values
    labels = [f"success ({successes.sum()})", f"fail ({(~successes).sum()})"]
    uniq = np.unique(yaws)

    if len(uniq) <= 12:
        # Discrete: a slim bar sits ON each recorded angle. The x-axis stays
        # numeric so the true spacing between the angles remains visible.
        span = float(uniq.max() - uniq.min()) or 1.0
        bar_w = max(span * 0.012, 0.35)
        n_succ = [int(((yaws == v) & successes).sum()) for v in uniq]
        n_fail = [int(((yaws == v) & ~successes).sum()) for v in uniq]
        ax.bar(uniq, n_succ, width=bar_w, color="#4caf50",
               edgecolor="black", linewidth=0.6, label=labels[0])
        ax.bar(uniq, n_fail, width=bar_w, bottom=n_succ, color="#d32f2f",
               edgecolor="black", linewidth=0.6, label=labels[1])
        ax.set_xticks(uniq)
        ax.set_xlim(uniq.min() - span * 0.12 - bar_w,
                    uniq.max() + span * 0.12 + bar_w)
    else:
        y_lo, y_hi = float(np.min(yaws)), float(np.max(yaws))
        width = 5.0 if max(y_hi - y_lo, 1.0) <= 60 else 20.0
        bins = np.arange(np.floor((y_lo - width) / width) * width,
                         y_hi + 2 * width, width)
        ax.hist([yaws[successes], yaws[~successes]], bins=bins, label=labels,
                color=["#4caf50", "#d32f2f"], stacked=True,
                edgecolor="black", alpha=0.8)
    ax.set_xlabel("Cube yaw [deg]")
    ax.set_ylabel("Episodes")
    ax.set_title(_title(bin_label, "Cube Rotation × Outcome"))
    legend_below(ax)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_episode_timeline(df: pd.DataFrame, out_path: Path,
                           bin_label: str = "") -> list[tuple[str, str]]:
    """Per-episode horizontal bar showing stage reached + termination step.

    Adds design00-style annotations (from auswertung_template.ipynb):
      - Vertical dotted line at ET_APPROACH_STEP=600 (ET-1 no_approach threshold)
      - Termination markers at termination_step colored/shaped by reason:
        success=★ green, no_approach=✗ orange, cube_lost=✗ red (ET-2),
        plateau_1_reach/plateau_2_lift=◆ gray, timeout=◊ gray.
      - reach/lift/home pipe markers (design00-style)
    """
    n = len(df)
    fig, ax = plt.subplots(figsize=(12, max(4, n * 0.28)))
    ep_indices = df["episode"].values

    # Termination-reason marker style table. ``id`` = short identifier for the
    # in-plot legend (kept minimal); ``desc`` = full explanation, rendered in a
    # separate footer legend below the axes (two legends in-plot would be noisy).
    term_style = {
        "success":         dict(marker="*", color="#4caf50", size=14,
                                id="success",     desc="lifted + returned home"),
        "no_approach":     dict(marker="X", color="#ff9800", size=10,
                                id="no_approach", desc="gripper never reached the cube by step 600"),
        "cube_lost":       dict(marker="X", color="#d32f2f", size=11,
                                id="cube_lost",   desc="cube dropped below start height"),
        "plateau_1_reach": dict(marker="P", color="#fb8c00", size=11,
                                id="stuck@reach", desc="at cube but no grasp/lift within 10 s"),
        "plateau_2_lift":  dict(marker="P", color="#e65100", size=11,
                                id="stuck@lift",  desc="cube lifted but no home-return within 10 s"),
        "timeout":         dict(marker="d", color="#9e9e9e", size=9,
                                id="timeout",     desc="step limit reached, no success"),
    }
    seen_terms: set[str] = set()

    for _, row in df.iterrows():
        stage = row["task_progress_label"]
        color = STAGE_COLORS.get(stage, "#9e9e9e")
        ax.barh(row["episode"], row["termination_step"] if row.get("termination_step", -1) > 0
                                else row["steps"],
                color=color, edgecolor="black", alpha=0.85, height=0.7)

        # Time-to-stage pipe markers (design00 convention)
        for col, m_color in [
            ("time_to_reach", "#1565c0"),
            ("time_to_lift", "#ef6c00"),
            ("time_to_home", "#2e7d32"),
        ]:
            if col in df.columns and row[col] > 0:
                ax.plot(row[col], row["episode"], "|", color=m_color,
                        markersize=14, markeredgewidth=2)

        # Termination marker at termination_step, styled by reason
        term = str(row.get("termination_reason", "timeout"))
        style = term_style.get(term, term_style["timeout"])
        term_step = row.get("termination_step", row["steps"])
        if term_step > 0:
            ax.plot(term_step, row["episode"], style["marker"],
                    color=style["color"], markersize=style["size"],
                    markeredgecolor="black", markeredgewidth=0.8)
            seen_terms.add(term)

    # ET-1 vertical threshold line (no_approach fires at step 600 if stage<1)
    ax.axvline(x=600, color="red", linestyle=":", alpha=0.45, linewidth=1.2)

    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Episode")
    ax.set_title(_title(bin_label, "Episode Timeline"))
    ax.set_yticks(ep_indices)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    # Build legend from stages + termination reasons actually present
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = []
    for s in STAGE_ORDER:
        if s in df["task_progress_label"].values:
            handles.append(Patch(facecolor=STAGE_COLORS[s], label=s, edgecolor="black"))
    # Stage-timing markers (always show if any rows have them)
    for lbl, m_color in [("reach", "#1565c0"), ("lift", "#ef6c00"), ("home", "#2e7d32")]:
        handles.append(Line2D([0], [0], marker="|", linestyle="", color=m_color,
                              markersize=10, markeredgewidth=2, label=lbl))
    # Termination markers that appeared in data — MINIMAL identifiers only.
    for term in ("success", "no_approach", "cube_lost",
                 "plateau_1_reach", "plateau_2_lift", "timeout"):
        if term in seen_terms:
            s = term_style[term]
            handles.append(Line2D([0], [0], marker=s["marker"], linestyle="",
                                  color=s["color"], markersize=9,
                                  markeredgecolor="black", markeredgewidth=0.6,
                                  label=s["id"]))
    # ET-1 threshold (short identifier; explained in the footer legend)
    handles.append(Line2D([0], [0], color="red", linestyle=":", linewidth=1.5,
                          label="ET-1 (600)"))

    # Legend below the axes: short identifiers only.
    legend_below(ax, handles=handles, fontsize=8, max_ncol=5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # The identifiers in the legend are short on purpose. Their full meaning is
    # returned for report.md instead of being baked into the PNG, where it used
    # to sit as a monospace block under the axes.
    glossary = [(term_style[t]["id"], term_style[t]["desc"])
                for t in ("success", "no_approach", "cube_lost",
                          "plateau_1_reach", "plateau_2_lift", "timeout")
                if t in seen_terms]
    glossary.append(("ET-1 (600)",
                     "no_approach fires at step 600 (10 s) if the gripper "
                     "never reached the cube"))
    return glossary


def _plot_timing_and_distance(df: pd.DataFrame, out_path: Path, bin_label: str = "") -> None:
    """Scatter of min_gripper_cube_dist vs RELATIVE lift height, colored by outcome."""
    if "min_gripper_cube_dist" not in df.columns or "max_height" not in df.columns:
        return
    df = df.copy()
    # Y must be the RELATIVE lift (how high the cube was raised), not the absolute
    # world-Z. In the cylroom table scene the cube rests at z≈0.79 m, so absolute
    # max_height (~0.79–0.85) clumps every point near 0.8 and makes the 0.06 m lift
    # threshold meaningless. Subtract the resting height so failures sit at ~0 and
    # successes at the actual lift amount. Falls back to absolute if the column is
    # missing (legacy room-scene CSVs with floor at z=0).
    if "initial_cube_z" in df.columns:
        df["lift_height"] = df["max_height"] - df["initial_cube_z"]
        ylabel = "Lift height [m]"
    else:
        df["lift_height"] = df["max_height"]
        ylabel = "Max cube height [m]"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A — min dist × lift height × outcome
    for success, sub in df.groupby("success"):
        color = "#4caf50" if success else "#d32f2f"
        label = "success" if success else "fail"
        axes[0].scatter(sub["min_gripper_cube_dist"], sub["lift_height"],
                        c=color, s=80, alpha=0.7, label=f"{label} ({len(sub)})",
                        edgecolors="black", linewidths=0.5)
    axes[0].axhline(0.06, color="gray", linestyle="--", alpha=0.5, label="lift threshold (0.06m)")
    axes[0].axhline(0.0, color="black", linestyle="-", alpha=0.25, linewidth=0.8)
    axes[0].set_xlabel("Min gripper-cube distance [m]")
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(_title(bin_label, "Reach Quality × Lift Height"))
    legend_below(axes[0], fontsize=9, max_ncol=3)
    axes[0].grid(True, alpha=0.3)

    # Panel B — time_to_reach × time_to_lift
    if "time_to_reach" in df.columns and "time_to_lift" in df.columns:
        sub = df[df["time_to_lift"] > 0]  # only episodes that lifted
        if len(sub) > 0:
            axes[1].scatter(sub["time_to_reach"], sub["time_to_lift"],
                            c="#1976d2", s=80, alpha=0.7,
                            edgecolors="black", linewidths=0.5)
            axes[1].plot([0, sub["time_to_reach"].max() * 1.1],
                         [0, sub["time_to_reach"].max() * 1.1],
                         "k--", alpha=0.3, label="reach = lift (fast grip)")
        axes[1].set_xlabel("Time to reach [steps]")
        axes[1].set_ylabel("Time to lift [steps]")
        axes[1].set_title(_title(bin_label, f"Reach-to-Lift Latency (n={len(sub)})"))
        legend_below(axes[1], fontsize=9)
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_attempts(df: pd.DataFrame, out_path: Path, bin_label: str = "") -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    max_attempts = int(df["num_attempts"].max()) if len(df) else 0
    ax.hist(
        df["num_attempts"],
        bins=range(0, max_attempts + 3),
        color="#42a5f5", edgecolor="black", linewidth=0.5, align="left",
    )
    ax.set_xlabel("Grasp attempts")
    ax.set_ylabel("Episodes")
    ax.set_title(_title(bin_label, "Attempts per Episode"))
    ax.set_xticks(range(0, max_attempts + 2))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _render_text_report(df: pd.DataFrame, summary: dict, gate: dict | None) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append(f"Eval Report: {summary.get('policy_path', '?')}")
    lines.append("=" * 60)
    lines.append(f"Task              : {summary.get('task', '?')}")
    lines.append(f"Randomize         : {summary.get('randomize', '?')}")
    lines.append(
        f"Cube noise        : pos=±{summary.get('cube_noise', 0)} m, "
        f"yaw=±{summary.get('cube_rot_noise', 0)}°"
    )
    lines.append(f"Episodes          : {len(df)}")
    lines.append(
        f"Success rate      : {summary.get('total_successes', 0)}/"
        f"{summary.get('total_episodes', 0)} "
        f"({summary.get('overall_success_rate', 0)}%)"
    )
    lines.append(f"Avg max_height    : {summary.get('avg_max_height', 0)} m")
    if gate:
        lines.append("")
        lines.append("-- quality gate --")
        for k, v in gate.items():
            lines.append(f"  {k:30s}: {v}")
    lines.append("")
    lines.append("-- stage distribution --")
    for s in STAGE_ORDER:
        n = int((df["task_progress_label"] == s).sum())
        lines.append(f"  {s:10s}: {n}")
    lines.append("")
    lines.append("-- termination reasons --")
    for t in TERM_ORDER:
        n = int((df["termination_reason"] == t).sum())
        if n > 0:
            lines.append(f"  {t:12s}: {n}")
    return "\n".join(lines)


def _render_md_report(summary: dict, gate: dict | None, txt: str,
                      timeline_glossary: list[tuple[str, str]] | None = None) -> str:
    md = [
        f"# Eval Report: `{Path(summary.get('policy_path', '?')).parent.name}`",
        "",
        f"- **Task**: `{summary.get('task', '?')}`",
        f"- **Randomize**: `{summary.get('randomize', '?')}`",
        f"- **Cube noise**: ±{summary.get('cube_noise', 0)} m / "
        f"±{summary.get('cube_rot_noise', 0)}°",
        f"- **Success rate**: **{summary.get('overall_success_rate', 0)}%** "
        f"({summary.get('total_successes', 0)}/{summary.get('total_episodes', 0)})",
        "",
    ]
    if gate:
        md.append("## Quality Gate")
        md.append("")
        for k, v in gate.items():
            md.append(f"- `{k}`: `{v}`")
        md.append("")
    md.append("## Plots")
    md.append("")
    md.append("![Task progress stage and termination](stage_outcome.png)")
    md.append("![Attempts](attempts_histogram.png)")
    md.append("![Cube position analysis](cube_position_analysis.png)")
    md.append("![Reach quality x lift height](timing_and_distance.png)")
    md.append("![Episode timeline](episode_timeline.png)")
    md.append("")
    md.append("In the timeline, the bar colour is the highest stage an episode "
              "reached, the `|` marks are reach, lift and home, and the marker "
              "at the end says why the episode stopped:")
    md.append("")
    if timeline_glossary:
        md.append("| Marker | Meaning |")
        md.append("|---|---|")
        for ident, desc in timeline_glossary:
            md.append(f"| `{ident}` | {desc} |")
        md.append("")
    md.append("## Full summary")
    md.append("")
    md.append("```")
    md.append(txt)
    md.append("```")
    return "\n".join(md)


def main() -> int:
    p = argparse.ArgumentParser(description="eval_report: sim-eval post-analysis")
    p.add_argument("--dir", type=Path, required=True,
                   help="Eval output directory (containing eval_results_*.csv + eval_summary.json)")
    args = p.parse_args()

    eval_dir = args.dir.resolve()
    df, summary, gate = _load(eval_dir)

    report_dir = eval_dir / "report"
    report_dir.mkdir(exist_ok=True)

    # Enrich DataFrame for plots that need roll/pitch (CSVs written before
    # eval_policy.py exposed RPY will be missing those columns).
    scene_state_dir_str = summary.get("scene_state")
    scene_state_dir = Path(scene_state_dir_str) if scene_state_dir_str else None
    if scene_state_dir is not None:
        df = _enrich_with_scene_state(df, scene_state_dir)
    holdout_eps = _discover_holdout_episodes(summary.get("policy_path", ""))

    # Bin identifier used as title prefix on ALL plots (CLAUDE.md §17-analog
    # convention: every eval artifact must be traceable to its recorded bin).
    bin_label = ""
    if scene_state_dir is not None:
        bin_label = scene_state_dir.parent.name if scene_state_dir.name == "meta" \
            else scene_state_dir.name

    _plot_stage_outcome(df, report_dir / "stage_outcome.png", bin_label=bin_label)
    _plot_attempts(df, report_dir / "attempts_histogram.png", bin_label=bin_label)
    # Was the evaluated bin actually in this policy's training set? Drives whether
    # the "Training pos" overlay is shown (only when real training data existed).
    bin_was_trained = _bin_was_trained(summary.get("policy_path", ""), bin_label)
    _plot_cube_positions(
        df, report_dir / "cube_position_analysis.png",
        scene_state_dir=scene_state_dir,
        holdout_episodes=holdout_eps,
        cube_noise=float(summary.get("cube_noise", 0.0)),
        cube_rot_noise=float(summary.get("cube_rot_noise", 0.0)),
        bin_was_trained=bin_was_trained,
    )
    _plot_holdout_vs_training(
        df, report_dir / "holdout_vs_training_analysis.png",
        holdout_episodes=holdout_eps,
        bin_label=bin_label,
    )
    _plot_cube_rotations(df, report_dir / "cube_rotation_analysis.png", bin_label=bin_label)
    timeline_glossary = _plot_episode_timeline(
        df, report_dir / "episode_timeline.png", bin_label=bin_label)
    _plot_timing_and_distance(df, report_dir / "timing_and_distance.png", bin_label=bin_label)

    txt = _render_text_report(df, summary, gate)
    (report_dir / "report.txt").write_text(txt + "\n")
    (report_dir / "report.md").write_text(
        _render_md_report(summary, gate, txt, timeline_glossary) + "\n")

    print(txt)
    print(f"\nArtefacts: {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
