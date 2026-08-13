# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared matplotlib style for all paper figures.

Applying this in every figure script gives one consistent look: identical font
sizes, line widths, marker sizes, and DPI across figures, so that after LaTeX
scaling they read as a single set. Import and call ``apply()`` at the top of a
plotting script, after ``import matplotlib.pyplot as plt``.

Figure *sizes* are set per script using the PANELS helper so that every panel
has the same physical dimensions and every figure has the same height.
"""

import matplotlib.pyplot as plt

# Fixed native height for every figure, and two native widths chosen so that a
# full-width (figure*) figure at \textwidth and a single-column figure at
# \columnwidth both scale by the same factor in LaTeX. Equal scale + equal
# height => identical on-page font size and identical figure height everywhere.
FIG_H = 3.8
WIDTH_FULL = 9.6  # displayed at \textwidth  (2- or 3-panel figure*)
WIDTH_COL = 4.7  # displayed at \columnwidth (single-panel figure)


def figsize(n_panels: int) -> tuple[float, float]:
    """Consistent figure size: single-column for one panel, else full width."""
    return (WIDTH_COL if n_panels <= 1 else WIDTH_FULL, FIG_H)


def apply() -> None:
    """Set the shared rcParams used by every figure in the paper."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "axes.grid": False,
            "grid.alpha": 0.3,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "figure.dpi": 110,
        }
    )
