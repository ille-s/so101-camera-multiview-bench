"""Image augmentations at the sim->policy boundary (camera-importance + general).

``augment_image_frame()`` is the single dispatch point where the TiledCamera frame
handed from Isaac Sim to the ACT policy can be transformed by arbitrary image-
processing algorithms. Pure numpy/OpenCV on ONE uint8 HWC RGB frame -> a NEW frame
of the same shape; importable WITHOUT an Isaac Sim boot (CPU-testable). A new algo
= one function + one ``_AUGMENTATIONS`` entry; no eval-loop change.

Families:
  degradations (camera-importance study, strength-graded): blackout, gauss, blur
  transforms   (general image processing):                 invert, contrast, canny
  ... vieles mehr — just register another function.
"""

from __future__ import annotations

VERSION = "1.0.0"

import cv2
import numpy as np


# --- individual image-processing algorithms (each: img, strength, rng -> img) ---

def _blackout(img: np.ndarray, strength: float, rng) -> np.ndarray:
    """Sensor outage — zero frame. strength ignored."""
    return np.zeros_like(img)


def _gauss(img: np.ndarray, strength: float, rng) -> np.ndarray:
    """Additive Gaussian pixel noise. strength = sigma on a 0-1 scale (*255), clamped."""
    rng = rng if rng is not None else np.random.default_rng()
    noise = rng.normal(0.0, float(strength) * 255.0, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _blur(img: np.ndarray, strength: float, rng) -> np.ndarray:
    """Gaussian blur. strength = odd kernel size >= 3 (cv2 derives sigma)."""
    k = int(strength)
    if k < 3 or k % 2 == 0:
        raise ValueError(f"blur strength must be an odd kernel size >= 3, got {strength}")
    return cv2.GaussianBlur(img, (k, k), 0)


def _invert(img: np.ndarray, strength: float, rng) -> np.ndarray:
    """Invert colours (255 - img). strength ignored."""
    return 255 - img


def _contrast(img: np.ndarray, strength: float, rng) -> np.ndarray:
    """Increase contrast around mid-grey. strength = factor (>1 stronger; default 2.0)."""
    factor = float(strength) if strength and strength > 0 else 2.0
    return np.clip((img.astype(np.float32) - 127.5) * factor + 127.5, 0, 255).astype(np.uint8)


def _canny(img: np.ndarray, strength: float, rng) -> np.ndarray:
    """Canny edge map as an abstracted 3-channel input. strength = upper threshold.

    obs is RGB -> use COLOR_RGB2GRAY (NOT BGR; the wrong code would mis-weight the
    luminance). Edges (1-channel) are replicated to 3 channels so the frame keeps
    the (H, W, 3) contract the policy expects.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    high = int(strength) if strength and strength > 0 else 150
    low = max(1, high // 2)
    edges = cv2.Canny(gray, low, high)
    return np.repeat(edges[:, :, None], 3, axis=2)


_AUGMENTATIONS = {
    "blackout": _blackout,
    "gauss": _gauss,
    "blur": _blur,
    "invert": _invert,
    "contrast": _contrast,
    "canny": _canny,
}
AUGMENTATION_KINDS = tuple(_AUGMENTATIONS)

# Kinds with no meaningful zero: gauss at sigma 0 returns the frame untouched (a
# run that degrades nothing while the CSV still calls it a gauss run), and blur at
# kernel 0 raises deep inside the rollout. blackout and invert ignore strength,
# contrast and canny fall back to 2.0 and 150.
STRENGTH_REQUIRED = ("gauss", "blur")


def check_strength(kind: str, strength: float) -> None:
    """Raise if *kind* was asked for without a strength that means anything.

    async_eval calls this before it builds the environment, so a mislabelled
    sweep dies in about ten seconds rather than after an hour of evaluation.
    It cannot fire earlier than that: AppLauncher boots Isaac at import time,
    well before main() sees the parsed flags.
    """
    if kind in STRENGTH_REQUIRED and not strength:
        raise ValueError(
            f"--augment {kind} needs --augment_strength "
            f"(gauss=sigma on a 0-1 scale, blur=odd kernel >= 3). "
            f"blackout and invert ignore it, contrast and canny fall back "
            f"to 2.0 and 150.")


def augment_image_frame(
    img: np.ndarray,
    kind: str,
    strength: float = 0.0,
    rng: "np.random.Generator | None" = None,
) -> np.ndarray:
    """Apply ONE image augmentation at the sim->policy boundary (registry dispatch).

    Args:
        img: uint8 HWC RGB frame ``(H, W, C)`` — the native obs frame (CLAUDE.md §4).
        kind: registry key (see ``AUGMENTATION_KINDS``); selects the algorithm.
        strength: per-kind knob — gauss=sigma(0-1) · blur=odd kernel · contrast=factor
            · canny=upper threshold; ignored for blackout/invert.
        rng: ``np.random.Generator`` for stochastic kinds (gauss) — reproducibility.

    Returns:
        A NEW contiguous uint8 HWC RGB array, same shape as ``img`` (input untouched,
        no channel swap — preserves the obs->policy->video format contract).
    """
    if img.dtype != np.uint8:
        raise TypeError(f"expected uint8 camera obs (CLAUDE.md §4), got {img.dtype}")
    if img.ndim != 3:
        raise ValueError(f"expected one (H, W, C) frame, got shape {img.shape}")
    if kind not in _AUGMENTATIONS:
        raise ValueError(f"unknown augmentation {kind!r} (valid: {AUGMENTATION_KINDS})")
    out = _AUGMENTATIONS[kind](img, strength, rng)
    return np.ascontiguousarray(out, dtype=np.uint8)
