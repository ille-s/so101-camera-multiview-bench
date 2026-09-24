# SPDX-License-Identifier: MIT
"""Loader for the shipped evaluation config (``evaluation/config/eval.json``).

Lives in its own module, free of any Isaac import, so it can be unit-tested and
reused. It used to be a private function inside ``async_eval``, which launches
Isaac Sim at import time -- the reason its old test had to be parked.

The JSON stores degrees because humans wrote it; consumers work in radians, so
the conversion happens here, once.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

_DEFAULT_PATH = Path(__file__).parent / "config" / "eval.json"


def load_eval_config(path: Path | None = None) -> dict:
    """Read the eval config and convert its angles to radians.

    Args:
        path: Override for tests. Defaults to the shipped package resource.

    Returns:
        Dict with ``target_end_position_rad`` (list, one entry per joint),
        ``target_end_position_tolerance_rad`` and ``plateau_steps``.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If a required key is missing.
    """
    cfg_path = path if path is not None else _DEFAULT_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(f"eval config missing: {cfg_path}")
    cfg = json.loads(cfg_path.read_text())
    deg = cfg["target_end_position_deg"]
    return {
        "target_end_position_rad": [math.radians(v) for v in deg],
        "target_end_position_tolerance_rad": math.radians(
            cfg["target_end_position_tolerance_deg"]
        ),
        "plateau_steps": cfg["plateau_steps"],
    }
