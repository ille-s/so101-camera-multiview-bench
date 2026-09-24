#!/usr/bin/env python3
"""eval_checkpoint_selector.py — Cherry-Shen-Style Best-Checkpoint-Finder.

Evaluates all `checkpoints/NNNNNN/pretrained_model/` directories of a training
run against a holdout episode set (eval_loss on the holdout subset) and writes
the best step + per-step results to `<run_dir>/best_checkpoint.json` (plus
`eval_vs_train_loss.png` for visual inspection).

Console-script entry: `eval_checkpoint_selector` (after `pip install -e .`).

Usage:
    eval_checkpoint_selector --run_dir outputs/2cam_wrist_front \\
        --dataset_root datasets/liftcube_binned/bin_c1_r1__wrist_front_sideLR_top

Reads:
    <run_dir>/checkpoints/NNNNNN/pretrained_model/   — all checkpoints
    <run_dir>/holdout_episodes.json                  — held-out episode indices
    <run_dir>/train.log (optional)                   — for train-loss overlay

Writes:
    <run_dir>/best_checkpoint.json   — {best_step, best_eval_loss, all_results}
    <run_dir>/eval_vs_train_loss.png — overlay plot (skipped with --no-plot)

Design decisions (verified against Cherry Shen's so101_bench/scripts/train/train.py
and LeRobot 0.5.1 source, 2026-04-19):

1. **policy.train() + deterministic seed**: Cherry's original uses eval(),
   but LeRobot 0.5.1 modeling_act.py:149 has a bug — the KLD-loss block lacks
   an `and self.training` guard. In eval mode mu_hat/log_sigma_x2_hat are
   None → TypeError in line 155. We use policy.train() (VAE encoder active)
   combined with `torch.manual_seed(seed)` before each checkpoint eval to
   make the VAE reparameterization-sampling deterministic. torch.no_grad()
   still prevents weight updates.

2. **Preprocessor applied explicitly**: LeRobot 0.5.1 moved normalization out
   of the policy's forward() into a separate NormalizerProcessorStep (loaded
   via make_pre_post_processors). Without this, loss values are computed on
   unnormalized inputs — orders of magnitude larger than train-time loss, and
   not comparable. See lerobot_train.py:414 `batch = preprocessor(batch)`.

3. **Empty holdout guard**: Cherry's train.py:472 has `if eval_dataset is not None`.
   We mirror that: empty holdout list raises ValueError rather than silently
   producing a zero-frame dataset. LeRobot 0.5.1 `LeRobotDataset(episodes=[])`
   loads ZERO frames (pa_ds.field.isin([]) = empty filter), not "all" —
   subsequent `ds[0]` access would raise IndexError. The guard catches the
   logical error early with an actionable message.
"""

__version__ = "2.0.0"

import argparse
import json
import logging
import os
import re
import signal
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from so101_mvbench.evaluation.plot_style import legend_below

import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors

# Required so draccus choice-registry picks up ACT/Diffusion/etc.
import lerobot.policies  # noqa: F401

logger = logging.getLogger(__name__)


@dataclass
class RunLayout:
    """Resolved file paths the selector operates on.

    For design03: holdout/ckpt/train_log live in the TRAINING capsule;
    best_json/plot live in the SELECTION capsule (=write target).
    For legacy/in-place mode: read AND write paths share a common run_dir.
    """

    layout: str  # "selection_capsule" | "capsule" | "legacy"
    holdout_json: Path
    ckpt_dir: Path
    best_json_out: Path
    plot_png_out: Path
    train_log: Path | None  # auto-discovered, optional


def detect_run_layout(run_dir: Path) -> RunLayout:
    """Detect whether `run_dir` is a Capsule (input/proc/output) or Legacy run.

    Capsule (run_train.sh v3.0.0+, in-place selector mode):
        run_dir/
        ├── input/holdout_episodes.json
        ├── processing/train.log
        └── output/{checkpoints/, best_checkpoint.json, eval_vs_train_loss.png}

    Legacy (run_train.sh v2.0.0):
        run_dir/
        ├── holdout_episodes.json
        ├── checkpoints/{NNNNNN/, last/pretrained_model/train.log}
        ├── best_checkpoint.json
        └── eval_vs_train_loss.png

    Decision rule: if run_dir/input/ AND run_dir/output/ both exist, it's a Capsule.
    Otherwise it's Legacy. Mismatch (e.g. only input/ exists) is an error.
    """
    has_input = (run_dir / "input").is_dir()
    has_output = (run_dir / "output").is_dir()
    has_legacy_ckpt = (run_dir / "checkpoints").is_dir()

    if has_input and has_output:
        candidates = [
            run_dir / "processing" / "train.log",
            run_dir / "output" / "checkpoints" / "last" / "pretrained_model" / "train.log",
        ]
        train_log = next((p for p in candidates if p.exists()), None)
        return RunLayout(
            layout="capsule",
            holdout_json=run_dir / "input" / "holdout_episodes.json",
            ckpt_dir=run_dir / "output" / "checkpoints",
            best_json_out=run_dir / "output" / "best_checkpoint.json",
            plot_png_out=run_dir / "output" / "eval_vs_train_loss.png",
            train_log=train_log,
        )
    elif has_legacy_ckpt and not has_input:
        candidate = run_dir / "checkpoints" / "last" / "pretrained_model" / "train.log"
        train_log = candidate if candidate.exists() else None
        return RunLayout(
            layout="legacy",
            holdout_json=run_dir / "holdout_episodes.json",
            ckpt_dir=run_dir / "checkpoints",
            best_json_out=run_dir / "best_checkpoint.json",
            plot_png_out=run_dir / "eval_vs_train_loss.png",
            train_log=train_log,
        )
    else:
        raise FileNotFoundError(
            f"{run_dir} matches neither Capsule (input/ + output/) nor Legacy "
            f"(checkpoints/) layout. Found: input/={has_input}, output/={has_output}, "
            f"checkpoints/={has_legacy_ckpt}."
        )


def _resolve_workspace_dir() -> Path:
    """Selector lives at ``<workspace>/src/so101_mvbench/src/so101_mvbench/training/``.

    Climb 5 levels to reach <workspace>.
    """
    return Path(__file__).resolve().parents[5]


def _resolve_dataset_root(training_run: Path, override: Path | None) -> Path:
    """Resolve dataset root: CLI override > training_run/input/dataset symlink."""
    if override is not None:
        return override.resolve()
    ds_link = training_run / "input" / "dataset"
    if not ds_link.exists():
        raise FileNotFoundError(
            f"No dataset symlink at {ds_link}. Pass --dataset_root explicitly."
        )
    return ds_link.resolve()


def create_selection_capsule(
    training_run: Path,
    dataset_root: Path,
    selector_args: dict,
) -> tuple[Path, RunLayout]:
    """Create a fresh selection capsule under `selections/<job>__<train_ts>/<sel_ts>/`.

    Writes input sidecars (training_run symlink, holdout copy, dataset symlink,
    selector_args.json) + README.md + processing/ skeleton, then returns the
    capsule path along with a RunLayout pointing reads at the training capsule
    and writes at this new selection capsule.

    Reads from training capsule:
      - input/holdout_episodes.json (copied verbatim — selection capsule must
        be self-contained per schema)
      - output/checkpoints/                 (read-only, never moved)
      - processing/train.log               (auto-discovered for plot overlay)
    """
    if not (training_run / "input" / "holdout_episodes.json").exists():
        raise FileNotFoundError(
            f"Training capsule missing input/holdout_episodes.json: {training_run}. "
            "Train via run_train.sh v4.0.0+ first."
        )
    if not (training_run / "output" / "checkpoints").is_dir():
        raise FileNotFoundError(
            f"Training capsule missing output/checkpoints/: {training_run}"
        )

    train_ts = training_run.name                         # "<ts>_<policy>"
    job_name = training_run.parent.name                  # "<job_name>"
    sel_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    workspace = _resolve_workspace_dir()
    capsule = workspace / "selections" / f"{job_name}__{train_ts}" / sel_ts

    if capsule.exists():
        raise FileExistsError(f"Selection capsule already exists: {capsule}")

    inp = capsule / "input"
    proc = capsule / "processing"
    out = capsule / "output"
    inp.mkdir(parents=True)
    proc.mkdir()
    out.mkdir()

    # input/training_run → training capsule (absolute, mount-stable)
    (inp / "training_run").symlink_to(training_run.resolve())

    # input/holdout_episodes.json → verbatim copy from training (self-contained)
    holdout_src = training_run / "input" / "holdout_episodes.json"
    holdout_dst = inp / "holdout_episodes.json"
    holdout_dst.write_bytes(holdout_src.read_bytes())

    # input/dataset → absolute symlink
    (inp / "dataset").symlink_to(dataset_root.resolve())

    # input/selector_args.json — frozen forensic record of CLI args
    (inp / "selector_args.json").write_text(json.dumps(selector_args, indent=2))

    # processing/ skeleton — selector.log handler is attached in main()
    (proc / "selector.log").touch()

    # README.md (status patched by main() on completion / signal)
    started = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    readme = f"""# Selection Capsule: {job_name} / {train_ts} / {sel_ts}

**Stage:** 3 of 4 (Selection) — design03 stage-as-capsule
**Status:** in_progress
**Started:** {started}
**Selector:** eval_checkpoint_selector.py v{__version__}

Schema: [SELECTION_CAPSULE_SCHEMA.md](../../../src/so101_mvbench/docs/SELECTION_CAPSULE_SCHEMA.md)

## Layout

```
{sel_ts}/
├── README.md
├── input/                      ── E
│   ├── training_run -> {training_run}
│   ├── holdout_episodes.json
│   ├── dataset -> {dataset_root}
│   └── selector_args.json
├── processing/                 ── V
│   └── selector.log
└── output/                     ── A
    ├── best_checkpoint.json
    └── eval_vs_train_loss.png  (skipped with --no-plot)
```

## Lineage

- **Upstream:**  `input/training_run` → trainings/{job_name}/{train_ts}/
- **Downstream:** evaluations/{job_name}__{train_ts}__step<best>__<mode>/<eval_ts>/
                  copies `output/best_checkpoint.json` to its `input/best_checkpoint_ref.json`
"""
    (capsule / "README.md").write_text(readme)

    # Build read/write layout
    candidates = [
        training_run / "processing" / "train.log",
        training_run / "output" / "checkpoints" / "last" / "pretrained_model" / "train.log",
    ]
    train_log = next((p for p in candidates if p.exists()), None)

    layout = RunLayout(
        layout="selection_capsule",
        holdout_json=holdout_dst,                                  # in selection
        ckpt_dir=training_run / "output" / "checkpoints",          # in training
        best_json_out=out / "best_checkpoint.json",                # in selection
        plot_png_out=out / "eval_vs_train_loss.png",               # in selection
        train_log=train_log,                                       # in training
    )
    return capsule, layout


def _patch_readme_status(readme_path: Path, new_status: str) -> None:
    """Idempotent status patch: only matches the literal 'in_progress' line."""
    if not readme_path.exists():
        return
    text = readme_path.read_text()
    patched = re.sub(
        r"^\*\*Status:\*\* in_progress$",
        f"**Status:** {new_status}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if patched != text:
        readme_path.write_text(patched)


def eval_policy_on_dataset(
    policy,
    preprocessor,
    dataset: LeRobotDataset,
    device: torch.device,
    batch_size: int = 32,
) -> float:
    """Compute mean forward-loss on a dataset without gradient computation.

    - policy.train() keeps VAE encoder active (LeRobot 0.5.1 bug: eval-mode
      forward() crashes on KLD-block with None mu_hat, see modeling_act.py:149)
    - preprocessor(batch) applied to normalize inputs (LeRobot 0.5.1 API)
    - torch.no_grad() prevents weight updates
    - Seed is set BEFORE this function is called, to make VAE sampling deterministic
    """
    policy.train()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in loader:
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(device, non_blocking=True)
            batch = preprocessor(batch)   # normalize inputs
            loss, _ = policy.forward(batch)
            total_loss += loss.item()
            num_batches += 1
    return total_loss / num_batches if num_batches > 0 else float("inf")


def evaluate_all_checkpoints(
    layout: RunLayout,
    dataset_root: Path,
    batch_size: int,
    device: str,
    seed: int = 42,
) -> tuple[dict, list[dict]]:
    """Evaluate every checkpoint in layout.ckpt_dir against the holdout set."""
    if not layout.holdout_json.exists():
        raise FileNotFoundError(
            f"No holdout_episodes.json at {layout.holdout_json} — "
            "train via run_train.sh first (writes input/holdout_episodes.json in capsule layout)."
        )
    meta = json.loads(layout.holdout_json.read_text())
    holdout_eps = meta["holdout_episodes"]
    if not holdout_eps:
        raise ValueError(
            "holdout_episodes list is empty — cannot evaluate without holdout. "
            "Re-train with a profile that has holdout.episodes set."
        )
    logger.info("Holdout episodes: %s", holdout_eps)

    dataset_id = dataset_root.name
    repo_id = f"local/liftcube_binned_{dataset_id}"

    ds_meta = LeRobotDatasetMetadata(repo_id, root=str(dataset_root))

    ckpt_dirs = sorted(
        d for d in layout.ckpt_dir.iterdir()
        if d.is_dir() and d.name.isdigit()
    )
    if not ckpt_dirs:
        raise FileNotFoundError(f"No numeric checkpoint dirs in {layout.ckpt_dir}")
    logger.info("Found %d checkpoints: %s ... %s",
                len(ckpt_dirs), ckpt_dirs[0].name, ckpt_dirs[-1].name)

    # Load first checkpoint's config once to resolve delta_timestamps
    first_pretrained = ckpt_dirs[0] / "pretrained_model"
    policy_config = PreTrainedConfig.from_pretrained(str(first_pretrained))
    delta_timestamps = resolve_delta_timestamps(policy_config, ds_meta)

    eval_dataset = LeRobotDataset(
        repo_id,
        root=str(dataset_root),
        episodes=holdout_eps,
        delta_timestamps=delta_timestamps,
    )
    logger.info("Holdout dataset: %d eps, %d frames",
                eval_dataset.num_episodes, eval_dataset.num_frames)

    torch_device = torch.device(device)
    results = []
    best = {"step": -1, "eval_loss": float("inf")}

    for ckpt_dir in ckpt_dirs:
        step = int(ckpt_dir.name)
        pretrained_path = ckpt_dir / "pretrained_model"

        policy_config = PreTrainedConfig.from_pretrained(str(pretrained_path))
        policy_config.pretrained_path = str(pretrained_path)
        policy_config.device = device
        policy = make_policy(policy_config, ds_meta=ds_meta)

        # Preprocessor is checkpoint-specific (stats embedded in saved artifacts).
        preprocessor, _ = make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=str(pretrained_path),
        )

        # Deterministic seed per checkpoint → VAE reparameterization sampling
        # returns identical latents across selector runs (reproducibility).
        torch.manual_seed(seed)
        if torch_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        eval_loss = eval_policy_on_dataset(
            policy, preprocessor, eval_dataset, torch_device, batch_size
        )
        results.append({"step": step, "eval_loss": eval_loss})

        is_new_best = eval_loss < best["eval_loss"]
        marker = " ← best so far" if is_new_best else ""
        logger.info("  Step %6d: eval_loss = %.4f%s", step, eval_loss, marker)
        if is_new_best:
            best = {"step": step, "eval_loss": eval_loss}

        del policy, preprocessor
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()

    return best, results


_TRAIN_LOG_PATTERN = re.compile(r"step:(\d+)K.*?loss:(\d+\.\d+)")


def parse_train_loss_from_log(log_path: Path) -> list[tuple[int, float]]:
    """Parse LeRobot training log for per-K-step average train-loss.

    LeRobot logs lines like::

        INFO 2026-04-22 14:21:49 ot_train.py:439 step:30K smpl:241K ep:490 epch:20.41 loss:0.052 ...

    The step is K-rounded (floor division by 1000). Each K-bucket contains
    ~log_freq log entries (e.g. 10 entries per K at log_freq=100). We aggregate
    loss per bucket via mean so the returned list has one (step, mean_loss)
    tuple per 1000-step window.

    Returns tuples of (step_absolute=K*1000, mean_loss), sorted by step.
    Unreadable / non-matching log is tolerated — returns [] in that case.
    """
    if not log_path.exists():
        logger.warning("train-log not found: %s — skipping train-loss overlay", log_path)
        return []

    bucket: dict[int, list[float]] = defaultdict(list)
    try:
        with log_path.open(errors="replace") as f:
            for line in f:
                m = _TRAIN_LOG_PATTERN.search(line)
                if m:
                    step_k = int(m.group(1))
                    loss = float(m.group(2))
                    bucket[step_k].append(loss)
    except OSError as e:
        logger.warning("cannot read train-log %s: %s — skipping train-loss overlay", log_path, e)
        return []

    if not bucket:
        logger.warning(
            "no 'step:NK ... loss:X.XXX' lines in %s — skipping train-loss overlay",
            log_path,
        )
        return []

    return sorted((k * 1000, sum(v) / len(v)) for k, v in bucket.items())


def write_plot(
    results: list[dict],
    best: dict,
    plot_path: Path,
    title: str,
    train_data: list[tuple[int, float]] | None = None,
) -> None:
    """Generate an eval-loss-over-steps plot, optionally with train-loss overlay."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plot")
        return

    steps = [r["step"] for r in results]
    losses = [r["eval_loss"] for r in results]
    fig, ax = plt.subplots(figsize=(10, 6))

    # Look up train-loss at best step (exact K-bucket match) so we can
    # annotate it in the plot. None if no train_data or no matching bucket.
    train_at_best: float | None = None
    if train_data:
        tr_steps, tr_losses = zip(*train_data)
        ax.plot(
            tr_steps, tr_losses, "-",
            color="#ef6c00", alpha=0.65, linewidth=1.5,
            label="Train-Loss (mean per 1k-step window)",
        )
        train_at_best = next((l for s, l in train_data if s == best["step"]), None)

    ax.plot(steps, losses, "o-", color="#1565c0", linewidth=2, markersize=7,
            label="Eval-Loss (Holdout)")

    if train_at_best is not None:
        best_label = (
            f"Best: step {best['step']} "
            f"(eval={best['eval_loss']:.3f}, train={train_at_best:.3f})"
        )
    else:
        best_label = f"Best: step {best['step']} (eval_loss={best['eval_loss']:.3f})"
    ax.axvline(best["step"], color="red", linestyle="--", alpha=0.5, label=best_label)

    # Markers on both curves at best_step so the user can read the values off.
    ax.plot([best["step"]], [best["eval_loss"]], "o",
            color="red", markersize=10, markerfacecolor="none", markeredgewidth=2)
    if train_at_best is not None:
        ax.plot([best["step"]], [train_at_best], "o",
                color="red", markersize=10, markerfacecolor="none", markeredgewidth=2)
        # Small text label next to the train-marker for direct readability.
        ax.annotate(
            f"{train_at_best:.3f}",
            xy=(best["step"], train_at_best),
            xytext=(8, -12), textcoords="offset points",
            fontsize=9, color="#ef6c00",
        )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"Checkpoint Selection — {title}")
    legend_below(ax)
    ax.grid(alpha=0.3)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote: %s", plot_path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--training_run", type=Path,
        help="Training capsule path (design03 mode, preferred). Selector creates "
             "its own selection capsule under selections/<job>__<train_ts>/<sel_ts>/.",
    )
    mode.add_argument(
        "--run_dir", type=Path,
        help="Legacy in-place mode (design02 or pre-capsule). Selector reads and "
             "writes within run_dir. Requires --dataset_root explicitly.",
    )
    parser.add_argument(
        "--dataset_root", type=Path, default=None,
        help="Dataset root (contains meta/episodes/ + data/). With --training_run, "
             "auto-resolved from <training_run>/input/dataset symlink unless overridden.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Eval-Batch-Size (default: 32)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for inference (default: cuda)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for VAE reparameterization sampling (default: 42). "
             "Ensures two runs of the selector produce identical eval_loss.",
    )
    parser.add_argument(
        "--no-plot", dest="plot", action="store_false",
        help="Skip generating eval_vs_train_loss.png plot (default: plot enabled)",
    )
    parser.set_defaults(plot=True)
    parser.add_argument(
        "--train_log", type=Path, default=None,
        help="Optional LeRobot training log file. If provided, train-loss is "
             "extracted and overlayed in the plot alongside eval-loss.",
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    args = parser.parse_args()

    capsule_path: Path | None = None
    readme_path: Path | None = None

    if args.training_run is not None:
        # design03 mode — create new selection capsule
        training_run = args.training_run.resolve()
        if not training_run.is_dir():
            parser.error(f"--training_run not a directory: {training_run}")
        dataset_root = _resolve_dataset_root(training_run, args.dataset_root)

        selector_args = {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "device": args.device,
            "training_run_path": str(training_run),
            "dataset_root": str(dataset_root),
            "selector_version": __version__,
            "started": _now_iso(),
        }
        capsule_path, layout = create_selection_capsule(
            training_run, dataset_root, selector_args
        )
        readme_path = capsule_path / "README.md"

        # Tee logs to processing/selector.log
        fh = logging.FileHandler(capsule_path / "processing" / "selector.log")
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
        ))
        logging.getLogger().addHandler(fh)

        logger.info("eval_checkpoint_selector v%s — design03 mode", __version__)
        logger.info("training_run:    %s", training_run)
        logger.info("dataset_root:    %s", dataset_root)
        logger.info("selection_caps:  %s", capsule_path)
    else:
        # Legacy in-place mode
        run_dir = args.run_dir.resolve()
        if args.dataset_root is None:
            parser.error("--dataset_root is required with --run_dir (legacy mode)")
        dataset_root = args.dataset_root.resolve()
        layout = detect_run_layout(run_dir)
        logger.info("eval_checkpoint_selector v%s — legacy mode (%s)",
                    __version__, layout.layout)
        logger.info("run_dir:      %s", run_dir)
        logger.info("dataset_root: %s", dataset_root)

    if layout.train_log:
        logger.info("Auto-discovered train.log: %s", layout.train_log)

    # Signal trap (selection-capsule mode only — patch README on SIGINT/SIGTERM)
    if readme_path is not None:
        def _on_signal(signum, _frame):
            _patch_readme_status(
                readme_path,
                f"interrupted by SIG{signal.Signals(signum).name} ({_now_iso()})",
            )
            os._exit(130)
        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)

    try:
        best, results = evaluate_all_checkpoints(
            layout=layout,
            dataset_root=dataset_root,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
        )

        output = {
            "best_step": best["step"],
            "best_eval_loss": round(best["eval_loss"], 5),
            "all_results": [
                {"step": r["step"], "eval_loss": round(r["eval_loss"], 5)}
                for r in results
            ],
        }
        layout.best_json_out.parent.mkdir(parents=True, exist_ok=True)
        layout.best_json_out.write_text(json.dumps(output, indent=2))
        logger.info("Wrote: %s", layout.best_json_out)
        logger.info("=" * 60)
        logger.info("BEST CHECKPOINT: step=%d, eval_loss=%.4f",
                    best["step"], best["eval_loss"])
        last_step = results[-1]["step"]
        last_loss = results[-1]["eval_loss"]
        if best["step"] != last_step:
            delta_pct = (last_loss - best["eval_loss"]) / best["eval_loss"] * 100
            logger.info("Overfitting detected: last step %d has +%.1f%% higher eval_loss",
                        last_step, delta_pct)
        else:
            logger.info("No overfitting — last checkpoint is best")
        logger.info("=" * 60)

        if args.plot:
            log_path = args.train_log if args.train_log else layout.train_log
            train_data = parse_train_loss_from_log(log_path) if log_path else None
            if train_data:
                logger.info("Train-loss overlay: %d K-buckets parsed from %s",
                            len(train_data), log_path)
            title = capsule_path.parent.name if capsule_path else args.run_dir.name
            write_plot(
                results, best,
                plot_path=layout.plot_png_out,
                title=title,
                train_data=train_data,
            )
        else:
            logger.info("Plot generation skipped (--no-plot)")

        if readme_path is not None:
            _patch_readme_status(readme_path, f"completed ({_now_iso()})")
    except Exception as e:
        if readme_path is not None:
            _patch_readme_status(readme_path, f"failed: {type(e).__name__} ({_now_iso()})")
        raise


if __name__ == "__main__":
    main()
