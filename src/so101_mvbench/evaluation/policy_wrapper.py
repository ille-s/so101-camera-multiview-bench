"""SimLeRobotSO101Interface — Sim-only subclass of upstream LeRobotSO101Interface.

Policy inference for Isaac-Sim eval must use the EXACT SAME chain as on-robot
inference to keep Sim/Hardware parity. Upstream already provides that chain in
`LeRobotSO101Interface` at `third_party/lerobot_so101_teleop/.../lerobot_interface.py`.

Upstream's `init_device()` calls `make_robot_from_config()` which instantiates a
`SO101FollowerConfig`. That drags in `scservo_sdk` (Feetech Motor SDK) — NOT
a dependency of this package. So we override ONLY `init_device()`
to skip the hardware-connection and build a `_RobotStub` that provides the four
downstream attributes `make_policy()` and `predict_action()` need:

    self.robot.observation_features   (used in make_policy)
    self.robot.action_features        (used in make_policy)
    self.robot.name                   (used for DummyDatasetMeta)
    self.robot.robot_type             (used in predict_action per-step)

All OTHER methods (make_policy, sim_obs_to_policy_processor, predict_action,
prediction_to_sim_processor) are inherited UNCHANGED from upstream.

Rename-map env → policy is auto-detected from policy's config.json
input_features. Supports three layouts:

  - wrist + N external cameras (e.g. wrist+front, wrist+front+top)
  - N external cameras without wrist (e.g. front+top, side_l+side_r)
  - wrist-only (1-cam)

Convention: env 'ego' → policy 'wrist'; env 'external', 'external_2', …
→ non-wrist cameras sorted alphabetically. Falls back to legacy
POLICY_RENAME_MAPS only if config.json read fails.

Version 1.2.0 (multi-external + no-wrist rename_map auto-detect)
"""

from __future__ import annotations

__version__ = "1.2.0"

import json

from dataclasses import dataclass, field
from pathlib import Path

from so101_mvbench.vendor.lerobot_so101_teleop.lerobot_interface import (
    LeRobotSO101Interface,
    DummyDatasetMeta,
)
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.policies.factory import make_policy as lerobot_make_policy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import make_default_processors

from so101_mvbench.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Hardcoded rename-map per policy: env cam key → policy feature cam key.
# Startup reads this to verify policy's config.json image features match.
# ---------------------------------------------------------------------------

POLICY_RENAME_MAPS: dict[str, dict[str, str]] = {
    # Policy #1 (1-cam wrist): env camera_ego → policy observation.images.wrist
    "act_bin_c1_r1__wrist": {
        "ego": "wrist",
    },
    # Policy #1'' (1-cam wrist, 30k + Holdout) — same camera wiring as #1
    "act_bin_c1_r1__wrist__holdout": {
        "ego": "wrist",
    },
    # Policy #1''_50k (1-cam wrist, 50k + Holdout, diagnostic run) — same wiring
    "act_bin_c1_r1__wrist__holdout_50k": {
        "ego": "wrist",
    },
    # Smoke-test profile (1k Steps, wrist-only) — CI-like pipeline validation
    "_smoke_test": {
        "ego": "wrist",
    },
    # Policy #2 (2-cam wrist+front): ego → wrist, external → front
    "act_bin_c1_r1__wrist_front": {
        "ego": "wrist",
        "external": "front",
    },
}


def _autodetect_rename_map(policy_path: Path) -> dict[str, str] | None:
    """Auto-detect rename-map from policy config.json input_features.

    Maps Isaac Sim env camera names to the policy's expected image keys.
    Non-wrist cameras are sorted alphabetically and assigned to env sensors
    ``external``, ``external_2``, ``external_3``, ``external_4`` in order.

    Supported layouts:
      - wrist-only:          ``{"ego": "wrist"}``
      - wrist + 1 external:  ``{"ego": "wrist", "external": "front"}``
      - wrist + N external:  ``{"ego": "wrist", "external": "front", "external_2": "top"}``
      - no-wrist (N ext):    ``{"external": "front", "external_2": "top"}``

    Returns None if auto-detection isn't possible (no config.json, >4
    non-wrist cameras) — caller falls back to POLICY_RENAME_MAPS lookup.
    """
    config_path = policy_path / "config.json"
    if not config_path.is_file():
        return None
    try:
        cfg = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    input_features = cfg.get("input_features", {})
    cam_keys = [
        key[len("observation.images."):]
        for key in input_features
        if key.startswith("observation.images.")
    ]

    if not cam_keys:
        return None

    _EXT_NAMES = ["external", "external_2", "external_3", "external_4", "external_5", "external_6"]

    rename_map: dict[str, str] = {}
    if "wrist" in cam_keys:
        rename_map["ego"] = "wrist"
        non_wrist = sorted(c for c in cam_keys if c != "wrist")
        if len(non_wrist) == 0:
            return rename_map  # 1-cam wrist-only
        if len(non_wrist) <= len(_EXT_NAMES):
            for env_key, policy_cam in zip(_EXT_NAMES, non_wrist):
                rename_map[env_key] = policy_cam
            return rename_map
        return None  # 5+ non-wrist — not auto-resolvable

    # No-wrist policies (external-only, e.g. front+top)
    cam_sorted = sorted(cam_keys)
    if 1 <= len(cam_sorted) <= len(_EXT_NAMES):
        return dict(zip(_EXT_NAMES, cam_sorted))
    return None


def rename_map_for_policy(policy_path: Path) -> dict[str, str]:
    """Resolve env→policy camera rename-map for the given policy checkpoint.

    Strategy:
      1. Auto-detect from policy config.json input_features (NVIDIA-style layout)
      2. Fallback: lookup hardcoded POLICY_RENAME_MAPS by JOB_NAME
         (legacy capsule layout: .../train/<JOB_NAME>/<ts>/checkpoints/<step>/pretrained_model)
    """
    auto = _autodetect_rename_map(policy_path)
    if auto is not None:
        return auto

    # Legacy fallback — derive JOB_NAME from path structure.
    # Try parents[2] (NVIDIA layout) first, then parents[3] (capsule layout).
    for depth in (2, 3):
        try:
            candidate = policy_path.parents[depth].name
        except IndexError:
            continue
        if candidate in POLICY_RENAME_MAPS:
            return POLICY_RENAME_MAPS[candidate]

    raise KeyError(
        f"cannot resolve rename-map for policy_path={policy_path}. "
        f"Auto-detect failed (no config.json or unrecognized cam-set), "
        f"and JOB_NAME not in POLICY_RENAME_MAPS={sorted(POLICY_RENAME_MAPS.keys())}."
    )


# ---------------------------------------------------------------------------
# Robot stub — replaces `self.robot = make_robot_from_config(...)` in Sim
# ---------------------------------------------------------------------------

@dataclass
class _RobotStub:
    """Minimal stand-in for a real SO101Follower robot.

    Provides exactly the 4 attributes that upstream `make_policy` and
    `predict_action` access via `self.robot`:
    """

    name: str
    robot_type: str
    observation_features: dict
    action_features: dict


# ---------------------------------------------------------------------------
# Drift guard — detect upstream API changes that would silently break us
# ---------------------------------------------------------------------------

_EXPECTED_UPSTREAM_METHODS = (
    "init_device",
    "make_policy",
    "sim_obs_to_policy_processor",
    "predict_action",
    "prediction_to_sim_processor",
)


def _check_upstream_drift() -> None:
    """Assert that upstream's LeRobotSO101Interface has the methods we rely on.

    Upstream changes to these methods could silently break our subclass.
    Fail loud at import time instead of at step 11 of episode 1.
    """
    for method in _EXPECTED_UPSTREAM_METHODS:
        if not hasattr(LeRobotSO101Interface, method):
            raise RuntimeError(
                f"upstream LeRobotSO101Interface has no {method!r} — "
                f"API drift detected. Expected methods: {_EXPECTED_UPSTREAM_METHODS}"
            )


_check_upstream_drift()


# ---------------------------------------------------------------------------
# Sim-only subclass
# ---------------------------------------------------------------------------

class SimLeRobotSO101Interface(LeRobotSO101Interface):
    """Sim-only subclass — skips Feetech hardware init.

    Inherits make_policy / predict_action / sim_obs_to_policy_processor /
    prediction_to_sim_processor from upstream UNCHANGED. Only init_device is
    overridden to bypass scservo_sdk.
    """

    def init_device(self, visualize: bool = False):  # noqa: ARG002 — keep upstream sig
        """Build a `_RobotStub` with just the 4 attributes downstream methods need.

        No self.robot = make_robot_from_config(self.cfg). No scservo_sdk.
        No rerun init (we run headless or in the eval GUI viewport separately).

        Feature-dict format MUST match the real SO101Follower (see
        ``lerobot/robots/so_follower/so_follower.py:66-81``)::

            motor features:   dict[str, type]   -> {"joint_name.pos": float, ...}
            camera features:  dict[str, tuple]  -> {"cam_name": (H, W, 3), ...}
        """
        # Joint state features — dict[str, type]
        motor_feats: dict[str, type] = {j: float for j in self.SO101_JOINT_ORDER}

        # Camera features — renamed to policy-side names, as dict[str, tuple]
        camera_feats: dict[str, tuple] = {}
        for cam_key, cam_info in self.cameras.items():
            renamed = self.rename_map[cam_key] if self.rename_map else cam_key
            camera_feats[renamed] = (cam_info["height"], cam_info["width"], 3)

        observation_features = {**motor_feats, **camera_feats}
        action_features = dict(motor_feats)  # joint targets only

        self.robot = _RobotStub(
            name="so101_follower_sim",
            robot_type="so101_follower",
            observation_features=observation_features,
            action_features=action_features,
        )
        logger.info(
            "SimLeRobotSO101Interface v%s: %d motor feats + %d camera feats "
            "(no hardware connection)",
            __version__, len(motor_feats), len(camera_feats),
        )

    def make_policy(
        self,
        name_or_path: str,
        *,
        temporal_ensemble_coeff: float | None = None,
        force_disable_te: bool = False,
    ):
        """Override upstream make_policy to accept temporal_ensemble_coeff override.

        Body mirrors upstream LeRobotSO101Interface.make_policy 1:1 (same calls,
        same order) with ONE addition: if ``temporal_ensemble_coeff`` is set
        (or ``force_disable_te=True``), the value is written onto ``policy_config``
        BEFORE ``lerobot.policies.factory.make_policy`` builds the model. This is
        necessary because ACT wires up its ``TemporalEnsembler`` inside
        ``__init__`` — post-hoc config edits do NOT take effect (the ensembler
        dataclass is bound at init-time).

        TE-semantics:
          - ``force_disable_te=True``       → policy_config.temporal_ensemble_coeff = None
          - ``temporal_ensemble_coeff=X``   → policy_config.temporal_ensemble_coeff = X
          - neither                         → keep checkpoint's config default
        """
        _, self.robot_action_processor, self.robot_observation_processor = (
            make_default_processors()
        )

        self.dataset_features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=self.robot_observation_processor,
                initial_features=create_initial_features(
                    observation=self.robot.observation_features
                ),
                use_videos=True,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=self.robot_action_processor,
                initial_features=create_initial_features(
                    action=self.robot.action_features
                ),
                use_videos=True,
            ),
        )

        policy_config = PreTrainedConfig.from_pretrained(name_or_path)
        policy_config.pretrained_path = name_or_path
        policy_config.device = self.device
        # --- TE coeff override (not in upstream lerobot_so101_teleop) ---
        prev = getattr(policy_config, "temporal_ensemble_coeff", "?")
        if force_disable_te:
            policy_config.temporal_ensemble_coeff = None
            logger.info(
                "policy_config.temporal_ensemble_coeff: %s → None (force_disable_te)",
                prev,
            )
        elif temporal_ensemble_coeff is not None:
            policy_config.temporal_ensemble_coeff = temporal_ensemble_coeff
            logger.info(
                "policy_config.temporal_ensemble_coeff: %s → %s",
                prev, temporal_ensemble_coeff,
            )
        else:
            logger.info(
                "policy_config.temporal_ensemble_coeff: kept from checkpoint (%s)",
                prev,
            )

        self.dataset_meta = DummyDatasetMeta(self.dataset_features, self.robot.name)
        self.policy = lerobot_make_policy(policy_config, ds_meta=self.dataset_meta)
        logger.info("[INFO]: Policy loaded")

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=name_or_path,
            dataset_stats={},
            preprocessor_overrides={
                "device_processor": {"device": policy_config.device},
            },
        )
        logger.info("[INFO]: Preprocessor and postprocessor loaded")
