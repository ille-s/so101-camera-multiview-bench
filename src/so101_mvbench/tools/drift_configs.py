#!/usr/bin/env python3
"""Generate drifted scene configurations for a camera-sensitivity sweep.

Takes a scene configuration, picks one external camera, and writes one copy per
requested offset with a single spherical coordinate shifted. Everything else is
carried over unchanged, so an evaluation against the copies differs from the
baseline in the camera pose and in nothing else.

Usage:
    # Six azimuth offsets around the front camera's trained pose
    python -m so101_mvbench.tools.drift_configs \\
        --scene_config src/so101_mvbench/tasks/scene_configs/lift_cube_6cam.json \\
        --camera front --axis azimuth --offsets=-15,-10,-5,5,10,15 \\
        --output_dir configs/drift

    # Tilt the top camera along a meridian, with a fixed up-vector
    python -m so101_mvbench.tools.drift_configs \\
        --scene_config src/so101_mvbench/tasks/scene_configs/lift_cube_6cam.json \\
        --camera top --axis elevation --offsets=-30,-20,-10,10,20,30 \\
        --up_world 0,0,1 --include_baseline --output_dir configs/drift_top

Elevation sweeps should pass ``--up_world`` (see :doc:`the drift experiment
</experiments/camera_drift>`): without a fixed up-vector the camera roll is
derived from the stage up-axis and flips near the pole, which shows up as a
success-rate drop that is really a rotated image.
"""

from __future__ import annotations

VERSION = "1.0.0"

import argparse
import copy
import sys
from pathlib import Path

from so101_mvbench.utils.scene_config import SceneConfig

# Which CameraSpec field each axis moves, and the unit it is written in.
_AXIS_FIELD = {
    "azimuth": ("azimuth_deg", "deg"),
    "elevation": ("elevation_deg", "deg"),
    "radius": ("radius_m", "m"),
}


def parse_offsets(raw: str) -> list[float]:
    """Parse a comma-separated offset list, rejecting an empty result."""
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("--offsets is empty")
    return [float(v) for v in values]


def parse_up_world(raw: str | None) -> tuple[float, float, float] | None:
    """Parse ``x,y,z`` into a tuple, or return None when not given."""
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"--up_world needs three components, got {raw!r}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def variant_name(camera: str, axis: str, offset: float) -> str:
    """Build a stem that sorts by offset and states the axis, e.g. ``front_az+05``."""
    short = {"azimuth": "az", "elevation": "el", "radius": "r"}[axis]
    if axis == "radius":
        return f"{camera}_{short}{offset:+.3f}".replace(".", "p")
    return f"{camera}_{short}{offset:+03.0f}"


def drift_camera(
    base: SceneConfig,
    camera: str,
    axis: str,
    offset: float,
    up_world: tuple[float, float, float] | None = None,
) -> SceneConfig:
    """Return a copy of *base* with one camera's coordinate shifted by *offset*.

    Args:
        base: The configuration to derive from. Left untouched.
        camera: Name of an external camera in that configuration.
        axis: One of ``azimuth``, ``elevation``, ``radius``.
        offset: Added to the camera's current value.
        up_world: Optional fixed world up-vector written onto the camera.

    Returns:
        A new SceneConfig. The caller writes it with ``to_json``.

    Raises:
        KeyError: *camera* is not in the configuration.
        ValueError: *camera* is not external, or *axis* is unknown.
    """
    if axis not in _AXIS_FIELD:
        raise ValueError(f"axis must be one of {sorted(_AXIS_FIELD)}, got {axis!r}")
    if camera not in base.cameras:
        raise KeyError(
            f"camera {camera!r} not in this configuration. Available: "
            f"{sorted(base.cameras)}"
        )

    cfg = copy.deepcopy(base)
    spec = cfg.cameras[camera]
    if spec.type != "external":
        raise ValueError(
            f"camera {camera!r} is {spec.type!r}. Only external cameras have a "
            f"hemisphere pose to drift. The wrist camera rides on the gripper."
        )

    field, _unit = _AXIS_FIELD[axis]
    setattr(spec, field, getattr(spec, field) + offset)
    if up_world is not None:
        spec.up_world = up_world
    return cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write drifted copies of a scene config for a camera sweep.",
    )
    parser.add_argument("--scene_config", type=Path, required=True,
                        help="Scene configuration to derive from.")
    parser.add_argument("--camera", type=str, required=True,
                        help="Name of the external camera to move, e.g. front.")
    parser.add_argument("--axis", choices=sorted(_AXIS_FIELD), default="azimuth",
                        help="Which spherical coordinate to shift. Default: azimuth.")
    parser.add_argument("--offsets", type=str, required=True,
                        help="Comma-separated offsets, written with an equals sign so "
                             "that a leading minus is not read as a flag: "
                             "--offsets=-15,-10,-5,5,10,15. Degrees for azimuth "
                             "and elevation, metres for radius.")
    parser.add_argument("--up_world", type=str, default=None,
                        help="Fixed world up-vector 'x,y,z' written onto the camera. "
                             "Recommended for elevation sweeps.")
    parser.add_argument("--include_baseline", action="store_true",
                        help="Also write the unmodified configuration, so the "
                             "baseline lives in the same directory as the sweep.")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Directory for the generated configurations.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    base = SceneConfig.from_json(args.scene_config)
    offsets = parse_offsets(args.offsets)
    up_world = parse_up_world(args.up_world)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    if args.include_baseline:
        path = args.output_dir / f"{args.camera}_baseline.json"
        base.to_json(path)
        written.append(path)

    for offset in offsets:
        cfg = drift_camera(base, args.camera, args.axis, offset, up_world)
        path = args.output_dir / f"{variant_name(args.camera, args.axis, offset)}.json"
        cfg.to_json(path)
        written.append(path)

    spec = base.cameras[args.camera]
    field, unit = _AXIS_FIELD[args.axis]
    print(f"{args.camera}: {args.axis} {getattr(spec, field)} {unit} at the trained pose")
    for path in written:
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
