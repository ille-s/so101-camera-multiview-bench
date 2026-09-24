#!/usr/bin/env python3
"""collect_wandb_runs.py — build the training-run table for the Sphinx docs.

Reads the local Weights-and-Biases run directories that ``lerobot_train``
left under ``outputs/<run>/wandb/latest-run/`` and ``outputs/<campaign>/<run>/
wandb/latest-run/``, and emits a Markdown table.

NOT currently included by any documentation page — the run overview is an
open task. Whoever wires it in: check first WHICH runs belong in it. The
current campaign runs are the nested ones (``cylroom_v2`` and friends) trained
on ``merged_6cam_cylroom``; the flat ``outputs/2cam_*`` runs used an earlier
dataset and are not comparable to them.

Stdlib only — no ``wandb`` package, no network. Three files per run:

    files/wandb-summary.json   final logged metrics (train/loss, train/steps)
    files/wandb-metadata.json  the full ``lerobot_train`` argv + start time
    files/output.log           carries the wandb run URL

Output lands NEXT TO this script (``wandb_runs.md``), per the workspace
artifact co-location rule.

Usage:
    python3 collect_wandb_runs.py                  # auto-detect workspace root
    python3 collect_wandb_runs.py --outputs <dir>  # explicit outputs/ dir
    python3 collect_wandb_runs.py --selftest       # parser self-check

Version: 1.1.0
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

VERSION = "1.1.0"

_URL_RE = re.compile(r"https://wandb\.ai/[^\s\"']+/runs/[A-Za-z0-9]+")


def workspace_root(start: Path) -> Path:
    """Walk up until a directory holds both CLAUDE.md and outputs/."""
    for parent in [start, *start.parents]:
        if (parent / "CLAUDE.md").is_file() and (parent / "outputs").is_dir():
            return parent
    raise SystemExit("workspace root not found — pass --outputs explicitly")


def arg_value(args: list[str], key: str) -> str | None:
    """Value of ``--key=value`` in a lerobot_train argv list."""
    prefix = f"--{key}="
    for a in args:
        if a.startswith(prefix):
            return a[len(prefix):]
    return None


def cameras_from_args(args: list[str]) -> str:
    """Camera slots, derived from the policy input_features JSON."""
    raw = arg_value(args, "policy.input_features")
    if not raw:
        return "?"
    try:
        features = json.loads(raw)
    except json.JSONDecodeError:
        return "?"
    cams = [k.split(".")[-1] for k in features if k.startswith("observation.images.")]
    return " + ".join(cams) if cams else "?"


def from_config_yaml(text: str) -> dict:
    """Fallback for resumed runs, whose argv is just ``--config_path/--resume``.

    wandb dumps the fully resolved training config next to the summary, so the
    same facts are recoverable there. Read by regex — stdlib has no YAML parser
    and pulling one in for four scalars is not worth a dependency.
    """
    cams = re.findall(r"observation\.images\.(\w+):", text)
    repo = re.search(r"^\s+repo_id: (?!null)(\S+)", text, re.MULTILINE)
    batch = re.search(r"^batch_size:\s*\n\s*value: (\d+)", text, re.MULTILINE)
    seed = re.search(r"^seed:\s*\n\s*value: (\d+)", text, re.MULTILINE)
    return {
        "cameras": " + ".join(dict.fromkeys(cams)) if cams else "?",
        "dataset": repo.group(1) if repo else "?",
        "batch": batch.group(1) if batch else "?",
        "seed": seed.group(1) if seed else "?",
    }


def read_run(run_dir: Path) -> dict | None:
    """One ``outputs/<name>/wandb/latest-run/`` directory → one table row."""
    files = run_dir / "files"
    summary_path = files / "wandb-summary.json"
    meta_path = files / "wandb-metadata.json"
    if not (summary_path.is_file() and meta_path.is_file()):
        return None

    summary = json.loads(summary_path.read_text())
    meta = json.loads(meta_path.read_text())
    args = meta.get("args", [])

    log = files / "output.log"
    url_match = _URL_RE.search(log.read_text(errors="ignore")) if log.is_file() else None

    row = {
        "cameras": cameras_from_args(args),
        "dataset": arg_value(args, "dataset.repo_id") or "?",
        "batch": arg_value(args, "batch_size") or "?",
        "seed": arg_value(args, "seed") or "?",
    }
    if row["cameras"] == "?":  # resumed run — recover from the wandb config dump
        cfg = files / "config.yaml"
        if cfg.is_file():
            row = from_config_yaml(cfg.read_text(errors="ignore"))

    # <campaign>/<run> for nested runs, plain <run> otherwise — the bare run
    # name is not unique across campaigns.
    rel = run_dir.parents[1]
    name = rel.name if rel.parent.name == "outputs" else f"{rel.parent.name}/{rel.name}"

    return {
        "run": name,
        "cameras": row["cameras"],
        "dataset": row["dataset"].split("/")[-1],
        "steps": summary.get("train/steps"),
        "loss": summary.get("train/loss"),
        "l1": summary.get("train/l1_loss"),
        "batch": row["batch"],
        "seed": row["seed"],
        "started": (meta.get("startedAt") or "?")[:10],
        "url": url_match.group(0) if url_match else None,
    }


def fmt(rows: list[dict]) -> str:
    head = (
        "| Run | Cameras | Dataset | Steps | Final loss | Final L1 | Batch | Seed | Started | wandb |\n"
        "|-----|---------|---------|-------|-----------|----------|-------|------|---------|-------|\n"
    )
    body = ""
    for r in rows:
        link = f"[{r['url'].rsplit('/', 1)[-1]}]({r['url']})" if r["url"] else "—"
        loss = f"{r['loss']:.4f}" if isinstance(r["loss"], (int, float)) else "—"
        l1 = f"{r['l1']:.4f}" if isinstance(r["l1"], (int, float)) else "—"
        steps = f"{int(r['steps']):,}" if isinstance(r["steps"], (int, float)) else "—"
        body += (
            f"| `{r['run']}` | {r['cameras']} | `{r['dataset']}` | {steps} | {loss} | {l1} "
            f"| {r['batch']} | {r['seed']} | {r['started']} | {link} |\n"
        )
    return head + body


def selftest() -> None:
    args = [
        "--dataset.repo_id=local/merged_6cam_phase3",
        "--steps=65000",
        '--policy.input_features={"observation.images.front":{"type":"VISUAL"},'
        '"observation.images.top":{"type":"VISUAL"},"observation.state":{"type":"STATE"}}',
    ]
    assert arg_value(args, "steps") == "65000"
    assert arg_value(args, "missing") is None
    assert cameras_from_args(args) == "front + top"
    assert cameras_from_args(["--steps=1"]) == "?"
    assert _URL_RE.search("run at https://wandb.ai/team/lerobot/runs/qmqfeci0 done")

    yaml = (
        "batch_size:\n    value: 8\n"
        "dataset:\n    value:\n        repo_id: local/merged_6cam_phase3\n"
        "seed:\n    value: 42\n"
        "policy:\n    value:\n        input_features:\n"
        "            observation.images.back:\n                type: VISUAL\n"
        "            observation.images.top:\n                type: VISUAL\n"
        "        repo_id: null\n"
    )
    assert from_config_yaml(yaml) == {
        "cameras": "back + top",
        "dataset": "local/merged_6cam_phase3",
        "batch": "8",
        "seed": "42",
    }
    print("selftest OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outputs", type=Path, default=None,
                        help="Training outputs directory (default: <workspace>/outputs)")
    parser.add_argument("--output_file", type=Path,
                        default=Path(__file__).resolve().parent / "wandb_runs.md")
    parser.add_argument("--selftest", action="store_true")
    a = parser.parse_args()

    if a.selftest:
        selftest()
        return

    # Runs live at two depths: outputs/<run>/ for one-off trainings, and
    # outputs/<campaign>/<run>/ for the grouped campaigns (cylroom_v2 and
    # friends). Globbing only the first level silently drops the campaigns —
    # which is where the campaign policies are.
    outputs = a.outputs or workspace_root(Path(__file__).resolve()) / "outputs"
    run_dirs = sorted({*outputs.glob("*/wandb/latest-run"),
                       *outputs.glob("*/*/wandb/latest-run")})
    rows = [row for d in run_dirs if (row := read_run(d)) is not None]
    if not rows:
        raise SystemExit(f"no wandb runs found under {outputs}")

    rows.sort(key=lambda r: (r["cameras"].count("+"), r["run"]))
    a.output_file.write_text(
        f"<!-- Generated by docs/sphinx/tools/collect_wandb_runs.py v{VERSION}. "
        "Do not edit by hand. -->\n\n" + fmt(rows)
    )
    print(f"{len(rows)} runs → {a.output_file}")


if __name__ == "__main__":
    main()
