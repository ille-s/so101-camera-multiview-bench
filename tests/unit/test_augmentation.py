#!/usr/bin/env python3
"""CPU unit checks for evaluation.augmentation (sim->policy image augmentations).

Pure numpy/OpenCV on CPU — no Isaac Sim, no GPU. Verifies every registered
augmentation (blackout/gauss/blur/invert/contrast/canny), the format invariant
(uint8 HWC RGB in == out shape, no channel swap), input immutability and error
paths. Also writes the AP visualization (one panel per augmentation) next to the
pilot's artefacts.

The bulk of this file is a standalone self-check that also writes a panel
figure, so it runs as a script rather than under pytest:

    PYTHONPATH=src python tests/unit/test_augmentation.py

The pytest-collected part is at the bottom, for the checks worth running on
every commit.
"""

from __future__ import annotations

VERSION = "1.1.0"

import sys
from pathlib import Path

import numpy as np
import pytest

from so101_mvbench.evaluation.augmentation import (
    AUGMENTATION_KINDS,
    STRENGTH_REQUIRED,
    augment_image_frame,
    check_strength,
)

VIZ_DIR = Path(__file__).resolve().parents[3] / "artefacts" / "experiments" / "cam_importance_pilot"


def _test_image(h: int = 240, w: int = 320) -> np.ndarray:
    """Synthetic uint8 (H, W, C) RGB frame: gradient + checkerboard + colour squares."""
    yy = np.linspace(0, 255, h).reshape(h, 1) * np.ones((1, w))
    xx = np.linspace(0, 255, w).reshape(1, w) * np.ones((h, 1))
    checker = (((np.arange(h).reshape(h, 1) // 16 + np.arange(w).reshape(1, w) // 16) % 2)
               * 255).astype(float)
    img = np.stack([yy, xx, checker], axis=-1)
    img[40:90, 40:90] = [255, 0, 0]
    img[40:90, 120:170] = [0, 255, 0]
    return np.clip(img, 0, 255).astype(np.uint8)


def main() -> int:
    img = _test_image()
    orig = img.copy()
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    print(f"test_augmentation v{VERSION} — input {img.shape} {img.dtype} | kinds={AUGMENTATION_KINDS}")

    # --- format invariant: every kind keeps shape/dtype/contiguity ---
    for kind, s in [("blackout", 0), ("gauss", 0.15), ("blur", 15), ("invert", 0),
                    ("contrast", 2.0), ("canny", 150)]:
        out = augment_image_frame(img, kind, s, np.random.default_rng(42))
        ok = (out.shape == img.shape and out.dtype == np.uint8 and out.flags["C_CONTIGUOUS"])
        check(f"{kind}: shape/dtype/contiguous", ok, f"{out.shape} {out.dtype}")

    # --- blackout ---
    check("blackout: all zeros", bool((augment_image_frame(img, "blackout", 0) == 0).all()))

    # --- gauss: monotone in sigma + seeded determinism ---
    stds = [float((augment_image_frame(img, "gauss", s, np.random.default_rng(42)).astype(float)
                   - img.astype(float)).std()) for s in (0.05, 0.15, 0.4)]
    check("gauss: residual std monotone in sigma", stds[0] < stds[1] < stds[2],
          f"stds={[round(s, 1) for s in stds]}")
    g1 = augment_image_frame(img, "gauss", 0.15, np.random.default_rng(7))
    g2 = augment_image_frame(img, "gauss", 0.15, np.random.default_rng(7))
    g3 = augment_image_frame(img, "gauss", 0.15, np.random.default_rng(8))
    check("gauss: same seed -> identical", bool((g1 == g2).all()))
    check("gauss: different seed -> different", not bool((g1 == g3).all()))

    # --- blur: monotone smoothing ---
    grad = lambda a: float(np.abs(a[1:].astype(float) - a[:-1].astype(float)).mean())
    g_in = grad(img)
    gks = [grad(augment_image_frame(img, "blur", k)) for k in (5, 15, 31)]
    check("blur: smoother than input + monotone", g_in > gks[0] > gks[1] > gks[2],
          f"in={g_in:.2f} -> {[round(v, 2) for v in gks]}")

    # --- invert ---
    check("invert: == 255 - img", bool((augment_image_frame(img, "invert", 0) == (255 - img)).all()))

    # --- contrast: increases spread ---
    c = augment_image_frame(img, "contrast", 2.0)
    check("contrast: std increases", c.astype(float).std() > img.astype(float).std(),
          f"{img.astype(float).std():.1f} -> {c.astype(float).std():.1f}")

    # --- canny: binary edge map, 3 channels ---
    e = augment_image_frame(img, "canny", 150)
    uniq = set(np.unique(e).tolist())
    check("canny: 3ch + values in {0,255}", e.shape[2] == 3 and uniq <= {0, 255}, f"uniq={sorted(uniq)}")

    # --- input immutability + error paths ---
    check("input not modified in place", bool((img == orig).all()))
    for bad, exc in [((img.astype(np.float32), "blackout", 0), TypeError),
                     ((img, "blur", 4.0), ValueError),
                     ((img, "does_not_exist", 0), ValueError)]:
        try:
            augment_image_frame(*bad)
            check(f"reject {bad[1]}/{bad[2]}", False)
        except exc:
            check(f"reject {bad[1]}/{bad[2]}", True)

    # --- AP visualization: one panel per augmentation ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        panels = [("original", img),
                  ("blackout", augment_image_frame(img, "blackout", 0)),
                  ("gauss σ0.15", augment_image_frame(img, "gauss", 0.15, np.random.default_rng(42))),
                  ("blur k15", augment_image_frame(img, "blur", 15)),
                  ("invert", augment_image_frame(img, "invert", 0)),
                  ("contrast ×2", augment_image_frame(img, "contrast", 2.0)),
                  ("canny 150", augment_image_frame(img, "canny", 150))]
        fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 3.0))
        for ax, (title, t) in zip(axes, panels):
            ax.imshow(t)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.suptitle("augment_image_frame() CPU unit check — synthetic frame")
        fig.tight_layout()
        VIZ_DIR.mkdir(parents=True, exist_ok=True)
        out_png = VIZ_DIR / "unit_check_augmentations.png"
        fig.savefig(out_png, dpi=120)
        print(f"  [viz ] {out_png}")
    except ImportError:
        print("  [viz ] SKIPPED — matplotlib not available")

    print(f"{'PASS' if not failures else 'FAIL'}: {len(failures)} failure(s)"
          + (f" -> {failures}" if failures else ""))
    return 1 if failures else 0


@pytest.mark.parametrize("kind", STRENGTH_REQUIRED)
def test_graded_kinds_refuse_a_missing_strength(kind: str) -> None:
    """The failure mode this guards is silent, which is why it is guarded.

    gauss at sigma 0 returns the frame untouched, so the run would degrade
    nothing while the results CSV still labels it a gauss run. blur at kernel 0
    raises, but only once the simulator is up and the rollout has started.
    """
    with pytest.raises(ValueError, match="augment_strength"):
        check_strength(kind, 0.0)


@pytest.mark.parametrize("kind", [k for k in AUGMENTATION_KINDS
                                  if k not in STRENGTH_REQUIRED])
def test_the_other_kinds_run_without_a_strength(kind: str) -> None:
    """blackout and invert ignore it, contrast and canny have a fallback."""
    check_strength(kind, 0.0)
    out = augment_image_frame(_test_image(32, 32), kind, 0.0)
    assert out.shape == (32, 32, 3)
    assert out.dtype == np.uint8


def test_a_real_strength_is_always_accepted() -> None:
    for kind, strength in (("gauss", 0.15), ("blur", 15.0)):
        check_strength(kind, strength)


if __name__ == "__main__":
    sys.exit(main())
