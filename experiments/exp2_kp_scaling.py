# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Plot the analytic p=0 and certified p=1 through p=4 sharp constants."""

import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

from . import paper_style
from .optimal_cascade import K_STAR

paper_style.apply()

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

BB = {0: 1.191, 1: 1.790, 2: 2.554}  # continuous BB/BB-type estimates

results = {}
print("Sharp constants (p=0 analytic; p=1,...,4 certified)\n")
Kvals = dict(K_STAR)
for p in range(5):
    results[p] = {
        "p": p,
        "K": Kvals[p],
        "certified": p > 0,
        "status": "analytic" if p == 0 else "certified",
        "structure": "one piece" if p == 0 else "infinite endpoint cascade",
    }
    print(f"p={p}: K_p*={Kvals[p]:.14f}")

# BB gaps
for p in range(3):
    results[p]["K_bb"] = BB[p]
    results[p]["bb_gap_pct"] = (BB[p] - Kvals[p]) / Kvals[p] * 100.0

print(
    "\nK increments (near-linear growth):",
    [f"{Kvals[p] - Kvals[p - 1]:.4f}" for p in range(1, 5)],
)
b, a = np.polyfit(list(range(5)), [Kvals[p] for p in range(5)], 1)
print(f"linear fit  K_p* ~ {a:.4f} + {b:.4f} * p")
with (RESULTS / "exp2_results.json").open("w") as f:
    json.dump({str(k): v for k, v in results.items()}, f, indent=2)

# ---------------- plot ----------------
fig, ax1 = plt.subplots(figsize=paper_style.figsize(1))

ps = list(range(5))
Ks = [Kvals[p] for p in ps]
ax1.plot(ps, Ks, "o-", color="#2166AC", lw=2, ms=8, label=r"$K_p^*$ (sharp)", zorder=5)
xx = np.linspace(0, 4, 100)
ax1.plot(
    xx,
    a + b * xx,
    "--",
    color="gray",
    lw=1,
    alpha=0.8,
    label=rf"$\approx {a:.2f} + {b:.2f}\,p$",
)
pb = [0, 1, 2]
ax1.plot(
    pb,
    [BB[p] for p in pb],
    "s--",
    color="#B2182B",
    lw=2,
    ms=7,
    label="BB / BB-type (numerical)",
)
ax1.set_xlabel("Smoothness order $p$")
ax1.set_ylabel("Shape-quality constant")
ax1.set_title("Sharp minimax constants vs order")
ax1.legend(fontsize=8.5)
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES / "exp2_kp_scaling.png", dpi=150, bbox_inches="tight")
plt.savefig(FIGURES / "exp2_kp_scaling.pdf", bbox_inches="tight")
print(f"\nSaved exp2 outputs under {ROOT}")
