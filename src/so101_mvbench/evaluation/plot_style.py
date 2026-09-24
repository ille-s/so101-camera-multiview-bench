"""Shared matplotlib conventions for the plots this package writes.

One rule so far, but it applies to every figure: the legend belongs BELOW the
x-axis. An outside-right legend eats roughly a quarter of the width for a few
short entries, and an in-plot one covers data -- in the episode timeline it
literally hid a bar. Vertical space is the cheap direction.
"""

from __future__ import annotations

__version__ = "1.0.0"


def legend_below(ax, handles=None, fontsize=9, max_ncol=4, y=None) -> None:
    """Place ``ax``'s legend as a horizontal row underneath the x-axis.

    ``handles`` overrides the artists collected from the axes, for figures that
    build proxy handles. Does nothing when there is nothing to label.

    ``y`` is an axes-fraction offset. Left at None it is derived from a constant
    PHYSICAL gap, because the same fraction means different distances on a tall
    square plot and on a short wide one: at a fixed -0.09 the legend landed on
    top of the x-axis label of the wide rotation and timing figures.
    """
    if y is None:
        # Whatever already sits under the axes has to be cleared: the x-axis
        # label, and rotated tick labels, which are what the aggregate charts
        # use and which are taller than upright ones.
        rotated = any(t.get_rotation() % 180 != 0 for t in ax.get_xticklabels())
        gap_in = 0.30 + (0.32 if ax.get_xlabel() else 0.0) + (0.32 if rotated else 0.0)
        height_in = ax.get_position().height * ax.figure.get_figheight()
        y = -gap_in / max(height_in, 0.5)
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    else:
        labels = [h.get_label() for h in handles]
    if not handles:
        return
    ax.legend(handles=handles, labels=labels, loc="upper center",
              bbox_to_anchor=(0.5, y), ncol=min(len(handles), max_ncol),
              columnspacing=1.4, handletextpad=0.5, fontsize=fontsize,
              frameon=True)
