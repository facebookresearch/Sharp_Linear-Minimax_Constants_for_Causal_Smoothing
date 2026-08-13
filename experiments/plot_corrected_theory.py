# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Create theory figures from the corrected infinite-cascade approximations."""

import matplotlib as mpl

mpl.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import paper_style
from .optimal_cascade import CascadeFilter, peano_from_grid

paper_style.apply()

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=paper_style.figsize(1))
for p, color in zip((0, 1, 2), ("#1b9e77", "#d95f02", "#7570b3")):
    shape = CascadeFilter(p, cells=120)
    x, g = shape.grid(10001)
    ax.plot(x, g, lw=1.8, color=color, label=rf"$p={p}$")
ax.axhline(0, color="black", lw=0.6)
ax.set_xlim(0, 1.002)
ax.set_xlabel("normalized time")
ax.set_ylabel("filter value")
ax.set_title("Optimal filter shapes")
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(FIGURES / "fig_impulse_responses.png", dpi=180)
fig.savefig(FIGURES / "fig_impulse_responses.pdf")
plt.close(fig)

shape = CascadeFilter(2, cells=140)
x, g = shape.grid(30001)
phi = peano_from_grid(x, g, 2)

fig, axes = plt.subplots(1, 2, figsize=paper_style.figsize(2))
axes[0].plot(x, phi, color="#2166ac", lw=1.4)
axes[0].axhline(0, color="black", lw=0.5)
axes[0].set_xlim(0, 0.98)
axes[0].set_title("Visible lobes")
axes[0].set_xlabel("normalized time")
axes[0].set_ylabel(r"Peano kernel $\Phi$")

knots = shape.breaks[1:-1]
visible = (1.0 - knots) > 1e-12
indices = np.arange(1, len(knots) + 1)[visible]
axes[1].semilogy(indices, 1.0 - knots[visible], "o-", color="#b2182b", ms=3, lw=1.0)
axes[1].set_title("Knot accumulation")
axes[1].set_xlabel("knot index")
axes[1].set_ylabel("distance to endpoint")
for ax in axes:
    ax.grid(alpha=0.18)
fig.tight_layout()
fig.savefig(FIGURES / "fig_kernel_sign_changes.png", dpi=180)
fig.savefig(FIGURES / "fig_kernel_sign_changes.pdf")
plt.close(fig)

print("Saved corrected theory figures")
