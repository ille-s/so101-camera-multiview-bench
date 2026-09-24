"""eval_aggregate_report — cross-run statistical analysis of multiple eval-runs.

Reads N eval-output directories (each containing eval_summary.json +
eval_results_random_*.csv from a single run_eval.sh execution) and produces
a cross-run aggregation report:

    aggregate_report.md            — markdown summary (mean ± std)
    aggregate_report.txt           — plain-text summary
    plots/cross_run_success_rate.png      — boxplot of success_rate per run
    plots/per_episode_success.png         — fraction-success per episode (1-indexed)
    plots/cross_run_attempts.png          — distribution of num_attempts per run
    plots/cross_run_stage_distribution.png — stacked bar of stages per run

Designed for thesis-grade statistics: mean, std, min/max, per-episode flakiness.

Standalone — no Jupyter / nbconvert dependency, no simulator needed.

Usage:
    eval_aggregate_report --run_dirs <dir1> <dir2> ... --output_dir <agg_dir>

Each <dir_N> must contain:
    - eval_summary.json
    - eval_results_random_*.csv

Version 1.0.0
"""

from __future__ import annotations

__version__ = "1.0.0"

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .plot_style import legend_below
import pandas as pd


def _load_run(run_dir: Path) -> tuple[dict, pd.DataFrame]:
    """Load one run's summary + per-episode CSV.

    Args:
        run_dir: Path to run<N>/ subdir containing eval_summary.json + CSV.

    Returns:
        (summary_dict, episodes_df). Raises if files are missing.
    """
    summary_path = run_dir / "eval_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"eval_summary.json missing in {run_dir}")
    summary = json.loads(summary_path.read_text())

    csv_files = sorted(run_dir.glob("eval_results_random_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"no eval_results_random_*.csv in {run_dir}")
    df = pd.read_csv(csv_files[0])
    df["run_dir"] = run_dir.name
    return summary, df


def _aggregate_summaries(summaries: list[dict]) -> dict:
    """Compute mean ± std for top-level metrics across runs."""
    success_rates = [s["overall_success_rate"] for s in summaries]
    avg_heights = [s["avg_max_height"] for s in summaries]
    return {
        "num_runs": len(summaries),
        "success_rate": {
            "mean": float(np.mean(success_rates)),
            "std": float(np.std(success_rates)),
            "min": float(np.min(success_rates)),
            "max": float(np.max(success_rates)),
            "values": success_rates,
        },
        "avg_max_height": {
            "mean": float(np.mean(avg_heights)),
            "std": float(np.std(avg_heights)),
            "min": float(np.min(avg_heights)),
            "max": float(np.max(avg_heights)),
        },
        "task": summaries[0].get("task"),
        "num_episodes_per_run": summaries[0].get("num_episodes_per_run"),
    }


def _per_episode_success_fraction(all_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Compute success fraction per episode across runs.

    Returns DataFrame with columns: episode, n_runs, n_success, frac_success.
    Identifies flaky episodes (frac_success in (0, 1)).
    """
    combined = pd.concat(all_dfs, ignore_index=True)
    grouped = combined.groupby("episode")["success"].agg(
        n_runs="count",
        n_success="sum",
    )
    grouped["frac_success"] = grouped["n_success"] / grouped["n_runs"]
    return grouped.reset_index()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_cross_run_success_rate(summaries: list[dict], out_path: Path) -> None:
    """Boxplot + scatter of success_rate per run."""
    rates = [s["overall_success_rate"] for s in summaries]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([rates], showmeans=True, meanline=True)
    ax.scatter([1] * len(rates), rates, color="red", alpha=0.7, s=40, zorder=3)
    ax.set_xticks([1])
    ax.set_xticklabels([f"N={len(rates)} runs"])
    ax.set_ylabel("Success rate (%)")
    ax.set_title(f"Cross-run success rate (mean={np.mean(rates):.1f}%, std={np.std(rates):.1f})")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_per_episode_success(per_ep_df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart: success-fraction per episode (flaky vs always-pass/always-fail)."""
    episodes = per_ep_df["episode"]
    fracs = per_ep_df["frac_success"]
    colors = [
        "green" if f == 1.0 else "red" if f == 0.0 else "orange"
        for f in fracs
    ]
    fig, ax = plt.subplots(figsize=(12, 5))
    heights = [max(f * 100, 2.0) for f in fracs]
    ax.bar(episodes, heights, color=colors)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success fraction (%)")
    ax.set_title("Per-episode success fraction across runs (green=always-pass, red=always-fail, orange=flaky)")
    ax.set_ylim(0, 105)
    ax.set_xticks(episodes)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_cross_run_attempts(all_dfs: list[pd.DataFrame], out_path: Path) -> None:
    """Histogram of num_attempts across all run × episode combinations."""
    combined = pd.concat(all_dfs, ignore_index=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(combined["num_attempts"], bins=range(0, int(combined["num_attempts"].max()) + 2),
            edgecolor="black", alpha=0.75)
    ax.set_xlabel("num_attempts (per episode)")
    ax.set_ylabel("Frequency (across all runs × episodes)")
    ax.set_title(f"Attempt distribution — N={len(all_dfs)} runs × {len(combined) // len(all_dfs)} episodes")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_cross_run_stage_distribution(summaries: list[dict], all_dfs: list[pd.DataFrame], out_path: Path) -> None:
    """Stacked bar: stage-counts per run (averaged + per-run)."""
    stages = ["0_miss", "1_reach", "2_lift", "3_home"]
    fig, ax = plt.subplots(figsize=(10, 5))
    bottoms = np.zeros(len(all_dfs))
    colors = {"0_miss": "#cc0000", "1_reach": "#cc6600", "2_lift": "#cccc00", "3_home": "#00aa00"}
    for stage in stages:
        counts = []
        for df in all_dfs:
            counts.append(int((df["task_progress_label"] == stage).sum()))
        ax.bar(range(len(all_dfs)), counts, bottom=bottoms,
               label=stage, color=colors[stage], edgecolor="black", linewidth=0.5)
        bottoms += np.array(counts)
    ax.set_xticks(range(len(all_dfs)))
    ax.set_xticklabels([d["run_dir"].iloc[0] for d in all_dfs], rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(f"Stage distribution per run (N={len(all_dfs)} runs × {summaries[0]['num_episodes_per_run']} eps)")
    legend_below(ax)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _format_markdown(agg: dict, per_ep_df: pd.DataFrame, run_dirs: list[Path]) -> str:
    sr = agg["success_rate"]
    h = agg["avg_max_height"]
    flaky = per_ep_df[(per_ep_df["frac_success"] > 0) & (per_ep_df["frac_success"] < 1)]
    always_fail = per_ep_df[per_ep_df["frac_success"] == 0]
    always_pass = per_ep_df[per_ep_df["frac_success"] == 1]

    lines = [
        f"# Aggregate Eval Report — {agg['task']}",
        "",
        f"- **Runs aggregated**: {agg['num_runs']}",
        f"- **Episodes per run**: {agg['num_episodes_per_run']}",
        f"- **Total episode-runs**: {agg['num_runs'] * agg['num_episodes_per_run']}",
        "",
        "## Cross-Run Statistics",
        "",
        "| Metric | Mean | Std | Min | Max |",
        "|---|---|---|---|---|",
        f"| Success rate (%) | {sr['mean']:.2f} | {sr['std']:.2f} | {sr['min']:.2f} | {sr['max']:.2f} |",
        f"| avg_max_height (m) | {h['mean']:.4f} | {h['std']:.4f} | {h['min']:.4f} | {h['max']:.4f} |",
        "",
        f"Per-run success rates: `{[f'{r:.1f}' for r in sr['values']]}`",
        "",
        "## Per-Episode Robustness",
        "",
        f"- **Always pass** ({len(always_pass)} eps): {always_pass['episode'].tolist()}",
        f"- **Always fail** ({len(always_fail)} eps): {always_fail['episode'].tolist()}",
        f"- **Flaky** ({len(flaky)} eps): "
        + ", ".join(
            f"ep{int(r['episode'])}={int(r['n_success'])}/{int(r['n_runs'])}"
            for _, r in flaky.iterrows()
        ),
        "",
        "## Plots",
        "",
        "![Cross-run success rate](plots/cross_run_success_rate.png)",
        "![Per-episode success fraction](plots/per_episode_success.png)",
        "![Cross-run attempts](plots/cross_run_attempts.png)",
        "![Cross-run stage distribution](plots/cross_run_stage_distribution.png)",
        "",
        "## Source Runs",
        "",
        *[f"- `{rd}`" for rd in run_dirs],
        "",
    ]
    return "\n".join(lines)


def _format_text(agg: dict, per_ep_df: pd.DataFrame, run_dirs: list[Path]) -> str:
    sr = agg["success_rate"]
    h = agg["avg_max_height"]
    return (
        "=" * 60 + "\n"
        f"Aggregate Eval Report — {agg['task']}\n"
        + "=" * 60 + "\n"
        f"Runs aggregated   : {agg['num_runs']}\n"
        f"Episodes per run  : {agg['num_episodes_per_run']}\n\n"
        f"Success rate (%)  : mean={sr['mean']:.2f}  std={sr['std']:.2f}"
        f"  range=[{sr['min']:.2f}, {sr['max']:.2f}]\n"
        f"avg_max_height (m): mean={h['mean']:.4f} std={h['std']:.4f}\n\n"
        f"Per-run rates: {[f'{r:.1f}' for r in sr['values']]}\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Cross-run statistical aggregation of multiple eval-runs.",
    )
    p.add_argument("--run_dirs", nargs="+", type=Path, required=True,
                   help="Paths to run<N>/ sub-dirs (each must contain eval_summary.json + CSV).")
    p.add_argument("--output_dir", type=Path, required=True,
                   help="Output dir for aggregate_report.md, .txt, plots/.")
    args = p.parse_args()

    if len(args.run_dirs) < 2:
        print("ERROR: need at least 2 run_dirs for cross-run stats", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Load all runs
    summaries: list[dict] = []
    dfs: list[pd.DataFrame] = []
    for rd in args.run_dirs:
        s, df = _load_run(rd)
        summaries.append(s)
        dfs.append(df)

    # Aggregate
    agg = _aggregate_summaries(summaries)
    per_ep = _per_episode_success_fraction(dfs)

    # Plots
    _plot_cross_run_success_rate(summaries, plots_dir / "cross_run_success_rate.png")
    _plot_per_episode_success(per_ep, plots_dir / "per_episode_success.png")
    _plot_cross_run_attempts(dfs, plots_dir / "cross_run_attempts.png")
    _plot_cross_run_stage_distribution(summaries, dfs, plots_dir / "cross_run_stage_distribution.png")

    # Markdown + text
    md = _format_markdown(agg, per_ep, args.run_dirs)
    txt = _format_text(agg, per_ep, args.run_dirs)
    (args.output_dir / "aggregate_report.md").write_text(md)
    (args.output_dir / "aggregate_report.txt").write_text(txt)

    # Persist agg-stats as JSON for downstream tools
    (args.output_dir / "aggregate_summary.json").write_text(
        json.dumps(agg, indent=2)
    )

    print(txt)
    print(f"Artefacts: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
