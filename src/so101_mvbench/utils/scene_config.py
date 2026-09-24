# SPDX-License-Identifier: MIT
"""SceneConfig — unified scene declaration schema.

Single source of truth for scene structure: cameras, objects, robot,
visuals, and the scene's workspace origin (``workspace_origin_m``).
Consumed by ``teleop_recorder``, ``multicam_replay``, and ``async_eval``
via ``SceneConfig.from_json``; the shipped configs live in
``tasks/scene_configs/``.
- ... and 5 more (see plan-file "9 fragmentierte Configs" inventory)

Schema versioning: major version (``1.x``) must match ``SCHEMA_VERSION``;
minor bumps are forward-compatible for *new optional fields with defaults*,
but they may also tighten validation on existing fields (e.g. a previously
optional field becoming required for one ``type`` variant). The major bump
is reserved for changes that rename or remove fields or alter the semantics
of existing values.

Version history:

``1.0``
    Initial schema.
``1.1``
    ``CameraSpec.radius_m`` is required for external entries and forbidden
    on ego entries.
``1.2``
    ``CameraSpec.pose_id`` is removed. External cameras declare their
    hemisphere position via three explicit floats (``azimuth_deg``,
    ``elevation_deg``, ``radius_m``). Pre-1.2 configs must be migrated by
    splitting ``pose_id`` into the two angle fields.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from so101_mvbench.utils.so101_transforms import BASE_SCENE_ORIGIN_W

__version__ = "1.2.0"

SCHEMA_VERSION = "1.2"


@dataclass
class CameraSpec:
    """One camera in the scene.

    ``ego``      — gripper-mounted (use ``offset_pos_m`` + ``offset_rot_deg``
                   relative to the gripper joint).
    ``external`` — hemisphere-mounted around the workspace origin. Position
                   is declared by three explicit floats: ``azimuth_deg``,
                   ``elevation_deg``, ``radius_m``. All three are REQUIRED.
                   The world-frame ``(x, y, z)`` is computed from them by
                   ``utils/camera_grid.spherical_to_cartesian``.

    Schema 1.2 removed the string-form ``pose_id`` (e.g. ``"az000_el45"``):
    carrying both numbers and a string-encoding of two of them invited
    parsing, key mangling, and registry mutation downstream.
    """

    type: Literal["ego", "external"]
    # external: hemisphere position around the workspace center
    azimuth_deg: float | None = None
    elevation_deg: float | None = None
    radius_m: float | None = None
    # ego: gripper-relative pose
    offset_pos_m: tuple[float, float, float] | None = None
    offset_rot_deg: tuple[float, float, float] | None = None
    # common
    resolution_hw: tuple[int, int] = (480, 640)
    focal_length: float = 5.0
    horizontal_aperture: float = 10.0
    # optional fixed world up-vector for the camera roll (external only). When
    # set, the camera roll is derived from this constant up instead of the stage
    # up-axis, so a camera swept along a meridian tilts continuously without a
    # roll jump at the pole (camera-drift ablation). Default None = stage up-axis.
    up_world: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.type == "external":
            missing = [
                name for name, val in (
                    ("azimuth_deg", self.azimuth_deg),
                    ("elevation_deg", self.elevation_deg),
                    ("radius_m", self.radius_m),
                ) if val is None
            ]
            if missing:
                raise ValueError(
                    f"external camera needs {missing!r} (all three required: "
                    f"azimuth_deg, elevation_deg, radius_m)"
                )
        elif self.type == "ego":
            for name, val in (
                ("azimuth_deg", self.azimuth_deg),
                ("elevation_deg", self.elevation_deg),
                ("radius_m", self.radius_m),
            ):
                if val is not None:
                    raise ValueError(
                        f"ego camera should not have {name} "
                        f"(gripper-relative offset, no spherical concept)"
                    )
            if self.offset_pos_m is None or self.offset_rot_deg is None:
                raise ValueError(
                    "ego camera needs offset_pos_m + offset_rot_deg "
                    "(gripper-relative pose)"
                )


@dataclass
class ObjectSpec:
    """One scene object: rigid cube or static USD asset.

    ``init_pos_m`` is expressed in the **workspace frame {W}**, not in world
    coordinates: add the scene's ``workspace_origin_m`` to get the world pose.
    Declaring it {W}-relative is what makes an object placement portable
    between scenes (see :doc:`the frame contract </concepts/coordinate_frames>`).
    """

    type: Literal["cube", "usd_asset"]
    init_pos_m: tuple[float, float, float] | None = None
    init_rpy_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size_m: tuple[float, float, float] | None = None
    color_rgb: tuple[float, float, float] = (0.8, 0.8, 0.8)
    mass_kg: float = 0.05
    usd_path: str | None = None


@dataclass
class RobotSpec:
    """Robot config: type, initial pose, default + home joint positions.

    Joint dicts use *Isaac-USD* names (``Rotation``, ``Pitch``, ``Elbow``,
    ``Wrist_Pitch``, ``Wrist_Roll``, ``Jaw``) for consistency with
    ``LiftCubeEnvCfg.actions.joint_positions.joint_names``.
    """

    type: str = "so101_white"
    init_pos_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    init_rot_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.7071, 0.7071)
    default_joint_pos_rad: dict[str, float] = field(default_factory=dict)
    home_joint_pos_rad: dict[str, float] = field(default_factory=dict)
    color_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class VisualSpec:
    """Visual defaults: lighting, sky dome orientation, action pad."""

    room_light_exposure: float = 1.0
    env_light_exposure: float = 1.0
    sky_dome_yaw_rad: float = 0.0
    action_pad_yaw_rad: float = 0.0
    dome_light_y_mirror: bool = True


@dataclass
class SceneConfig:
    """Single source of truth for scene declaration."""

    schema_version: str = SCHEMA_VERSION
    workspace_origin_m: tuple[float, float, float] = BASE_SCENE_ORIGIN_W
    """{W} origin in world coordinates. Defaults to the scene the package
    ships (the cylindrical room). A config that omits it therefore declares
    the current base scene, not a retired one. The legacy fallback for
    *recorded data* without a frame declaration is a separate rule and lives
    in the consumers (see ``so101_transforms.rigid_shift_from_recording``)."""
    cameras: dict[str, CameraSpec] = field(default_factory=dict)
    objects: dict[str, ObjectSpec] = field(default_factory=dict)
    robot: RobotSpec = field(default_factory=RobotSpec)
    visual: VisualSpec = field(default_factory=VisualSpec)

    @classmethod
    def from_json(cls, path: Path | str) -> SceneConfig:
        """Load SceneConfig from a JSON file.

        Hard-fails when the JSON's ``schema_version`` major component does
        not match :data:`SCHEMA_VERSION`. JSON list-fields are converted
        back to tuples for type-consistency with downstream Isaac Lab APIs.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        version = str(data.get("schema_version", "0"))
        if _major(version) != _major(SCHEMA_VERSION):
            raise ValueError(
                f"schema_version mismatch: file {path} has {version!r}, "
                f"expected major {_major(SCHEMA_VERSION)!r}"
            )

        return cls._from_dict(data)

    def to_json(self, path: Path | str) -> None:
        """Serialize SceneConfig to JSON (pretty-printed, 2-space indent)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
            f.write("\n")

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> SceneConfig:
        cameras = {
            name: CameraSpec(**_coerce_tuples(spec))
            for name, spec in data.get("cameras", {}).items()
        }
        objects = {
            name: ObjectSpec(**_coerce_tuples(spec))
            for name, spec in data.get("objects", {}).items()
        }
        robot = RobotSpec(**_coerce_tuples(data.get("robot", {})))
        visual = VisualSpec(**data.get("visual", {}))

        ws_origin = data.get("workspace_origin_m")
        # No second default here: an omitted origin means the base scene, and
        # the dataclass field is the one place that decides which scene that is.
        ws_origin = tuple(ws_origin) if ws_origin is not None else BASE_SCENE_ORIGIN_W

        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            workspace_origin_m=ws_origin,
            cameras=cameras,
            objects=objects,
            robot=robot,
            visual=visual,
        )


_TUPLE_FIELDS = frozenset({
    "offset_pos_m", "offset_rot_deg", "resolution_hw",
    "init_pos_m", "init_rpy_rad", "size_m", "color_rgb",
    "init_rot_xyzw", "up_world",
})


def _coerce_tuples(spec: dict[str, Any]) -> dict[str, Any]:
    """Convert known list-fields to tuples (JSON lacks tuple literals)."""
    out = dict(spec)
    for key in _TUPLE_FIELDS:
        if key in out and isinstance(out[key], list):
            out[key] = tuple(out[key])
    return out


def _major(version: str) -> str:
    """Return major version (e.g. ``"1.2"`` → ``"1"``)."""
    return version.split(".", 1)[0]
