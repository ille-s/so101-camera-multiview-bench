# SPDX-License-Identifier: MIT
"""Read per-episode CSVs from cube_replay, generate 3-panel divergence plot
per episode + a global summary.csv.

CPU-only — no simulator needed (numpy + matplotlib + scipy).
"""

# ================================================================================
# METRICS — exact formulas
# ================================================================================
#
# Notation:
#   E              = number of parallel envs
#   T              = number of frames in episode
#
#   p_e(t) ∈ ℝ³    = cube position (x, y, z) in env e at frame t, in m
#                    (env_origin-offset subtracted: each env's cube is reported in
#                     that env's local frame so they are directly comparable)
#   q_e(t) ∈ ℝ⁴    = cube unit quaternion (XYZW, Isaac Lab post-PR-4437) in env e at frame t
#   P              = { (i, j) : 0 ≤ i < j < E } — all unordered env-pairs
#   R(q)           = rotation matrix / Rotation object from quaternion q
#   · / abs / ‖·‖₂ = component-wise dot product / absolute value / Euclidean L2 norm
#
# --- Position divergence (pairwise) -------------------------------------------------
#
#   max_pos_norm_mm  = 1000 · max_{(i,j)∈P, t} ‖ p_i(t) − p_j(t) ‖₂
#   max_dx_mm        = 1000 · max_{(i,j)∈P, t} | x_i(t) − x_j(t) |
#   max_dy_mm        = 1000 · max_{(i,j)∈P, t} | y_i(t) − y_j(t) |
#   max_dz_mm        = 1000 · max_{(i,j)∈P, t} | z_i(t) − z_j(t) |
#   mean_pos_norm_mm = 1000 · mean_t  max_{(i,j)∈P} ‖ p_i(t) − p_j(t) ‖₂
#   frames_pos_gt_3mm  = #{ t : max_{(i,j)∈P} ‖p_i(t) − p_j(t)‖ > 0.003 }   # flag threshold
#   frames_pos_gt_10mm = #{ t : max_{(i,j)∈P} ‖p_i(t) − p_j(t)‖ > 0.010 }   # moderate
#   frames_pos_gt_15mm = #{ t : max_{(i,j)∈P} ‖p_i(t) − p_j(t)‖ > 0.015 }   # severe
#
# --- Rotation divergence (geodesic angle, pairwise) ---------------------------------
#
#   max_quat_angle_deg = (180/π) · max_{(i,j)∈P, t}  2·arccos( | q_i(t) · q_j(t) | )
#
# The factor 2 and the abs are essential:
#   - abs handles q ≡ −q (both represent same rotation)
#   - arccos(|q·q'|) gives HALF the geodesic angle; 2·arccos(|·|) gives the full
#     geodesic distance on SO(3) in radians.
#
# --- Rotation per-axis (signed Euler vs reference env 0) ----------------------------
#
# For each env e (e ≠ 0) at each frame t, compute the RELATIVE rotation, then
# decompose:
#
#   R_rel(t)               = R(q_0(t))⁻¹ · R(q_e(t))
#   (Δroll, Δpitch, Δyaw)[e,t] = R_rel(t).as_euler("xyz", degrees=True)   # signed
#
# Then over the episode:
#
#   max_droll_to_env0_deg  = max_{e>0, t} | Δroll[e, t] |
#   max_dpitch_to_env0_deg = max_{e>0, t} | Δpitch[e, t] |
#   max_dyaw_to_env0_deg   = max_{e>0, t} | Δyaw[e, t] |
#
# CRITICAL: this is NOT euler(R_e) − euler(R_0). Naive subtraction of Euler angles
# is wrong under angle wrapping and gimbal lock. The correct procedure is to first
# compose the relative rotation, then decompose that single rotation.
#
# --- Flag ---------------------------------------------------------------------------
#
#   flag = 1  iff  max_pos_norm_mm > 3.0  OR  max_quat_angle_deg > 3.0
#        = 0  otherwise
#
# ================================================================================

VERSION = "1.0.0"

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(csv_path: Path):
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return None
    T = max(int(r["frame"]) for r in rows) + 1
    E = max(int(r["env"]) for r in rows) + 1
    pos = np.zeros((E, T, 3)); quat = np.zeros((E, T, 4))
    for r in rows:
        e, t = int(r["env"]), int(r["frame"])
        pos[e, t] = [float(r["x"]), float(r["y"]), float(r["z"])]
        quat[e, t] = [float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])]
    origins = pos[:, 0, :] - pos[0:1, 0, :]
    pos_rel = pos - origins[:, None, :]
    return T, E, pos_rel, quat


def per_env_to_ref_metrics(pos_rel: np.ndarray, quat: np.ndarray, ref: int = 0):
    """Per-env distance and rotation-angle relative to a reference env.

    For each env e at each frame t::

        dist[e, t]    = ‖ p_e(t) − p_ref(t) ‖₂              [meters]
        ang_deg[e, t] = 2·arccos( |q_e(t) · q_ref(t)| )      [degrees]

    where:
      - p_e(t) ∈ ℝ³ is the cube position in env e at frame t (env_origin-offset
        already subtracted so envs are comparable in a common local frame)
      - q_e(t) ∈ ℝ⁴ is the unit quaternion in XYZW order (Isaac Lab convention since PR #4437)
      - q_e · q_ref denotes the component-wise dot product
      - the absolute value handles the quaternion sign ambiguity: q and −q
        represent the same rotation, so the distance must be sign-invariant

    The factor 2 in ``2·arccos(|q·q'|)`` is the geodesic angle on SO(3); for unit
    quaternions, ``arccos(|q·q'|)`` is HALF the rotation angle between R(q) and R(q').

    Shapes: pos_rel (E, T, 3), quat (E, T, 4). Returns (dist, ang_deg) of shape (E, T).
    """
    dist = np.linalg.norm(pos_rel - pos_rel[ref:ref+1], axis=-1)
    dot = np.abs((quat * quat[ref:ref+1]).sum(axis=-1))
    ang_deg = np.degrees(2 * np.arccos(np.clip(dot, 0, 1)))
    return dist, ang_deg


def per_env_euler_to_ref(quat: np.ndarray, ref: int = 0):
    """Signed Euler decomposition of the relative rotation R_ref⁻¹·R_e per env.

    For each env e (e ≠ ref) at each frame t:

        R_e(t)    = Rotation.from_quat(q_e(t))   # quat already XYZW, scipy convention matches
        R_ref(t)  = Rotation.from_quat(q_ref(t))
        R_rel(t)  = R_ref(t)⁻¹ · R_e(t)                       # relative rotation
        (roll, pitch, yaw)[e, t] = R_rel(t).as_euler("xyz", degrees=True)

    where:
      - q_e is the unit quaternion in XYZW order (Isaac Lab convention since PR #4437,
        matches scipy.spatial.transform.Rotation "scalar-last")
      - "xyz" denotes the intrinsic Tait-Bryan convention (X, then Y′, then Z″)
      - Output values are SIGNED degrees in (−180°, +180°] per axis

    CRITICAL: this is NOT euler(R_e) − euler(R_ref), which would be wrong under
    angle wrapping and gimbal lock. The correct way is to first compose the
    relative rotation R_rel = R_ref⁻¹·R_e, then decompose that ONE rotation.

    Quaternion sign ambiguity (q ≡ −q): scipy.spatial.transform.Rotation handles
    this internally — Rotation.from_quat(q) and Rotation.from_quat(−q) yield
    identical rotation objects.

    Shapes: quat (E, T, 4). Returns euler (E, T, 3) signed degrees [roll, pitch, yaw].
    The slice [ref] stays zero (R_rel is identity for the reference env itself).
    """
    from scipy.spatial.transform import Rotation as R
    # quat is already in XYZW (Isaac Lab post-PR-4437, matches scipy convention).
    E, T = quat.shape[:2]
    ref_inv = R.from_quat(quat[ref].reshape(-1, 4)).inv()
    out = np.zeros((E, T, 3), dtype=np.float64)
    for e in range(E):
        if e == ref:
            continue
        rel = ref_inv * R.from_quat(quat[e].reshape(-1, 4))
        out[e] = rel.as_euler("xyz", degrees=True).reshape(T, 3)
    return out


def pairwise_max(pos_rel: np.ndarray, quat: np.ndarray):
    """Per-frame maxima over all env-pairs of position & rotation divergence.

    Let P = { (i, j) : 0 ≤ i < j < E } be the set of unordered env-pairs
    (size n_pairs = E·(E−1)/2). For each frame t we compute::

        max_norm[t]     = max_{(i,j)∈P}   ‖ p_i(t) − p_j(t) ‖₂        [meters]
        max_dx[t]       = max_{(i,j)∈P}   | x_i(t) − x_j(t) |          [meters]
        max_dy[t]       = max_{(i,j)∈P}   | y_i(t) − y_j(t) |
        max_dz[t]       = max_{(i,j)∈P}   | z_i(t) − z_j(t) |
        max_quat_deg[t] = max_{(i,j)∈P}   2·arccos( |q_i(t) · q_j(t)| ) · 180/π

    where:
      - p_e(t) = (x_e, y_e, z_e)(t) ∈ ℝ³ is the cube position in env e at frame t
        (in m, env_origin-offset subtracted, common local frame)
      - q_e(t) ∈ ℝ⁴ is the unit quaternion in XYZW order (Isaac Lab post-PR-4437) of the cube in env e
      - the absolute value of the dot product handles the q ≡ −q sign
        ambiguity, making the angular distance sign-invariant
      - ``2·arccos(|q·q'|)`` is the geodesic rotation distance on SO(3) in
        radians; multiplied by 180/π → degrees.

    The per-axis components (dx, dy, dz) are useful for diagnostics: they reveal
    WHERE the divergence sits (lateral vs vertical), which the Euclidean norm
    alone hides.

    Shapes: pos_rel (E, T, 3), quat (E, T, 4).
    Returns: (max_norm, max_dx, max_dy, max_dz, max_quat_deg), each shape (T,).
    """
    E, T, _ = pos_rel.shape
    pairs = list(combinations(range(E), 2))
    pair_diff = np.zeros((len(pairs), T, 3))
    qd = np.zeros((len(pairs), T))
    for k, (i, j) in enumerate(pairs):
        pair_diff[k] = pos_rel[i] - pos_rel[j]
        qd[k] = np.abs((quat[i] * quat[j]).sum(axis=-1))
    norm = np.linalg.norm(pair_diff, axis=-1)               # (n_pairs, T)
    abs_diff = np.abs(pair_diff)                            # (n_pairs, T, 3)
    return (norm.max(axis=0),
            abs_diff[..., 0].max(axis=0),
            abs_diff[..., 1].max(axis=0),
            abs_diff[..., 2].max(axis=0),
            np.degrees(2 * np.arccos(np.clip(qd, 0, 1))).max(axis=0))


def plot_episode(csv_path: Path, png_path: Path):
    loaded = load_csv(csv_path)
    if loaded is None:
        return None
    T, E, pos_rel, quat = loaded
    dist, ang = per_env_to_ref_metrics(pos_rel, quat, ref=0)
    max_norm, max_dx, max_dy, max_dz, max_q = pairwise_max(pos_rel, quat)
    deuler = per_env_euler_to_ref(quat, ref=0)  # (E, T, 3) signed roll/pitch/yaw vs env0
    bin_name = csv_path.parent.name
    ep_idx = int(csv_path.stem.split("_")[-1])

    cmap = plt.get_cmap("tab10")
    t = np.arange(T) / 30.0
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    n_legend_cols = max(2, (E + 4) // 5)  # 2 cols for <=10 envs, more for larger

    for e in range(E):
        axes[0].plot(t, pos_rel[e, :, 2], label=f"env{e}", color=cmap(e), alpha=0.85)
    axes[0].set_ylabel("cube Z [m]")
    axes[0].grid()
    axes[0].set_title(f"{bin_name} ep{ep_idx:02d} ({T} frames = {T/30:.1f}s) — Multi-Env divergence, num_envs={E}")

    for e in range(1, E):
        axes[1].plot(t, dist[e] * 1000, label=f"env{e}", color=cmap(e), alpha=0.85)
    axes[1].set_ylabel("‖pos − pos$_{env0}$‖ [mm]\n(Euclidean dist per env)")
    axes[1].axhline(3, ls=":", color="gray", linewidth=0.8)
    axes[1].grid()

    for e in range(1, E):
        axes[2].plot(t, ang[e], label=f"env{e}", color=cmap(e), alpha=0.85)
    axes[2].set_ylabel("∠(quat, quat$_{env0}$) [deg]\n(rotation diff per env)")
    axes[2].axhline(3, ls=":", color="gray", linewidth=0.8)
    axes[2].set_xlabel("t [s]"); axes[2].grid()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=min(len(handles), n_legend_cols * 3), fontsize=8,
               columnspacing=1.4, handletextpad=0.5, frameon=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(png_path, dpi=100)
    plt.close(fig)

    # Pairwise pos: norm + per-axis abs max (all over env-pairs × frames)
    # Quat: pairwise geodesic angle (sign-invariant) + per-env Euler-axes vs env0 (signed)
    return {
        "bin": bin_name,
        "episode": ep_idx,
        "T_frames": T,
        "num_envs": E,
        # --- pos (mm) ---
        "max_pos_norm_mm": float(max_norm.max() * 1000),       # ‖p_i − p_j‖ pairwise
        "max_dx_mm": float(max_dx.max() * 1000),               # |x_i − x_j| pairwise
        "max_dy_mm": float(max_dy.max() * 1000),
        "max_dz_mm": float(max_dz.max() * 1000),
        "mean_pos_norm_mm": float(max_norm.mean() * 1000),
        "frames_pos_gt_3mm": int((max_norm > 0.003).sum()),
        "frames_pos_gt_10mm": int((max_norm > 0.010).sum()),
        "frames_pos_gt_15mm": int((max_norm > 0.015).sum()),
        # --- rot (deg) ---
        "max_quat_angle_deg": float(max_q.max()),              # geodesic ∠(q_i, q_j) pairwise
        "max_droll_to_env0_deg": float(np.abs(deuler[:, :, 0]).max()),   # signed Δroll vs env0
        "max_dpitch_to_env0_deg": float(np.abs(deuler[:, :, 1]).max()),
        "max_dyaw_to_env0_deg": float(np.abs(deuler[:, :, 2]).max()),
        "mean_quat_angle_deg": float(max_q.mean()),
        "frames_quat_gt_3deg": int((max_q > 3).sum()),
        # --- flag ---
        "flag": int(max_norm.max() > 0.003 or max_q.max() > 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Audit output root containing <bin>/cube_states_ep_NN.csv files.")
    ap.add_argument("--summary", default=None,
                    help="Path to summary CSV; default <root>/summary.csv")
    args = ap.parse_args()

    root = Path(args.root)
    summary_path = Path(args.summary) if args.summary else (root / "summary.csv")
    fields = ["bin","episode","T_frames","num_envs",
              "max_pos_norm_mm","max_dx_mm","max_dy_mm","max_dz_mm","mean_pos_norm_mm",
              "frames_pos_gt_3mm","frames_pos_gt_10mm","frames_pos_gt_15mm",
              "max_quat_angle_deg","max_droll_to_env0_deg","max_dpitch_to_env0_deg","max_dyaw_to_env0_deg",
              "mean_quat_angle_deg","frames_quat_gt_3deg","flag"]
    summaries = []
    csv_files = sorted(root.rglob("cube_states_ep_*.csv"))
    print(f"# Processing {len(csv_files)} CSVs")
    for i, csv_path in enumerate(csv_files, 1):
        png_path = csv_path.with_suffix(".png")
        try:
            s = plot_episode(csv_path, png_path)
            if s is not None:
                summaries.append(s)
                print(f"[{i}/{len(csv_files)}] {s['bin']} ep{s['episode']:02d}: "
                      f"pos={s['max_pos_norm_mm']:.2f}mm quat={s['max_quat_angle_deg']:.2f}deg flag={s['flag']}")
        except Exception as exc:
            print(f"[{i}/{len(csv_files)}] {csv_path}: FAIL {exc}")
    summaries.sort(key=lambda s: (-s["flag"], -s["max_pos_norm_mm"]))
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summaries)
    n_flag = sum(s["flag"] for s in summaries)
    print(f"\nWrote {summary_path}\n  total: {len(summaries)} eps, flagged: {n_flag} ({100*n_flag/max(len(summaries),1):.1f}%)")


if __name__ == "__main__":
    main()
