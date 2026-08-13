# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Experiment 3 (REWRITTEN): shape-quality (K) comparison of causal smoothers.

The original 1-Euro part compared its AVERAGE-case empirical MSE to the WORST-case
bound K_p* sigma^rate, giving "ratios" of 0.04-0.68 that look like 1-Euro beats the
minimax-optimal filter -- a category error (average vs worst case; and 1-Euro is
nonlinear so it has no fixed K).

Fix:
  * Put the LINEAR filters on the common worst-case K-scale: optimal g*, the
    BB/BB-type family, and causal Savitzky-Golay (continuous-window limit).  This is the honest,
    apples-to-apples comparison and reproduces the letter's mechanism (the BB-family gap
    is the missing negative sidelobes; SG HAS sidelobes so it closes the gap and
    even beats the BB-type extension at p=2).
  * 1-Euro is nonlinear/adaptive -> no single K.  Frozen (beta=0) it is a p=0 EMA,
    whose shape quality equals the EMA/alpha-filter constant (~1.19 = BB at p=0) and
    which has NO p>=1 version.  Its adaptivity is evaluated in the AVERAGE-case
    tracking study (exp4), against the optimal filter -- not against a worst-case bound.
"""

import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
from numpy.typing import NDArray

mpl.use("Agg")
import matplotlib.pyplot as plt

from . import paper_style
from .optimal_cascade import (
    K_STAR,
    savitzky_golay_continuous_constant,
    shape_constant_from_taps,
)

paper_style.apply()

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# corrected optimal constants (from exp2, certified)
K_OPT = {p: K_STAR[p] for p in range(3)}
K_BB = {0: 1.191, 1: 1.790, 2: 2.554}  # continuous BB/BB-type limits
FloatArray = NDArray[np.float64]


# ---- Savitzky-Golay continuous-limit K ----
def sg_weights(W: int, p: int) -> FloatArray:
    t = np.arange(W, dtype=float)
    X = np.column_stack([t**k for k in range(p + 1)])
    return np.linalg.lstsq(X, np.eye(W), rcond=None)[0][0, :]


def sg_K(p: int, Ws: list[int]) -> list[float]:
    return [shape_constant_from_taps(sg_weights(W, p), p) for W in Ws]


# ---- EMA (= frozen 1-Euro, = BB p=0) K, to show 1-Euro is a p=0 filter ----
def ema_K(alpha: float, nmax: int = 4000) -> float:
    j = np.arange(nmax)
    h = alpha * (1 - alpha) ** j
    return shape_constant_from_taps(h, 0)


print("=== Savitzky-Golay K (continuous-window limit) ===")
sg_limit = {}
sg_curves = {}
for p in range(3):
    Ws = list(range(p + 2, 200, 2))
    Ks = sg_K(p, Ws)
    sg_limit[p] = savitzky_golay_continuous_constant(p)
    sg_curves[p] = (Ws, Ks)
    print(
        f"  p={p}: SG->{sg_limit[p]:.4f}  optimal={K_OPT[p]:.4f}  "
        f"gap={100 * (sg_limit[p] - K_OPT[p]) / K_OPT[p]:.1f}%   (BB-family gap={100 * (K_BB[p] - K_OPT[p]) / K_OPT[p]:.1f}%)"
    )

print("\n=== Frozen 1-Euro (= EMA, p=0) ===")
# K -> continuous limit as alpha->0 (finite-alpha has an O(alpha) correction)
ema_small = [ema_K(a) for a in (0.02, 0.01, 0.005)]
ema_limit = ema_small[-1]
print(
    f"  EMA K (alpha=0.02,0.01,0.005) = {[f'{v:.4f}' for v in ema_small]}  "
    f"-> {ema_limit:.3f} as alpha->0  (= BB p=0 = {K_BB[0]}); 1-Euro has no p>=1 version."
)

results = {
    "K_optimal": K_OPT,
    "K_bb": K_BB,
    "K_sg_limit": {str(p): sg_limit[p] for p in range(3)},
    "bb_gap_pct": {str(p): 100 * (K_BB[p] - K_OPT[p]) / K_OPT[p] for p in range(3)},
    "sg_gap_pct": {str(p): 100 * (sg_limit[p] - K_OPT[p]) / K_OPT[p] for p in range(3)},
    "one_euro_note": "nonlinear/adaptive: no fixed K; frozen = EMA (p=0, K~1.19); "
    "adaptive evaluation vs optimal is in exp4",
    "ema_K": float(ema_limit),
}
with (RESULTS / "exp3_results.json").open("w") as f:
    json.dump(results, f, indent=2)

# ---------------- plot ----------------
fig, axes = plt.subplots(1, 3, figsize=paper_style.figsize(3))

ax = axes[0]
x = np.arange(3)
w = 0.26
ax.bar(x - w, [K_OPT[p] for p in range(3)], w, color="#2166AC", label="Optimal $g^*$")
ax.bar(x, [K_BB[p] for p in range(3)], w, color="#B2182B", label="BB / BB-type")
ax.bar(x + w, [sg_limit[p] for p in range(3)], w, color="#4DAF4A", label="Causal SG")
# 1-Euro/EMA reference line removed (nonlinear; outside the linear-minimax comparison)
ax.set_xticks(x)
ax.set_xticklabels(["$p=0$", "$p=1$", "$p=2$"])
ax.set_ylabel("$K$ (worst-case shape quality)")
ax.set_title("Shape-quality on a common scale")
ax.legend(fontsize=8.5)
ax.grid(True, alpha=0.3, axis="y")

ax2 = axes[1]
for i, p in enumerate(range(3)):
    bb = 100 * (K_BB[p] - K_OPT[p]) / K_OPT[p]
    sg = 100 * (sg_limit[p] - K_OPT[p]) / K_OPT[p]
    ax2.bar(
        i - 0.16, bb, 0.32, color="#B2182B", label="BB / BB-type" if i == 0 else None
    )
    ax2.bar(i + 0.16, sg, 0.32, color="#4DAF4A", label="SG" if i == 0 else None)
    ax2.text(i - 0.16, bb + 0.15, f"{bb:.1f}", ha="center", fontsize=8)
    ax2.text(i + 0.16, sg + 0.15, f"{sg:.1f}", ha="center", fontsize=8)
ax2.set_xticks(range(3))
ax2.set_xticklabels(["$p=0$", "$p=1$", "$p=2$"])
ax2.set_ylabel("% above optimal")
ax2.set_title("Sub-optimality gap")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis="y")

ax3 = axes[2]
for p in range(3):
    Ws, Ks = sg_curves[p]
    ax3.plot(Ws, Ks, lw=1.5, label=f"$p={p}$", color=f"C{p}")
    ax3.axhline(sg_limit[p], ls=":", color=f"C{p}", alpha=0.6)
ax3.set_xlabel("SG window $N_w$")
ax3.set_ylabel("$K$")
ax3.set_title("Causal SG $\\to$ continuous limit")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 200)

plt.tight_layout()
plt.savefig(FIGURES / "exp3_filter_comparison.png", dpi=150, bbox_inches="tight")
plt.savefig(FIGURES / "exp3_filter_comparison.pdf", bbox_inches="tight")
print(f"\nSaved exp3 outputs under {ROOT}")
