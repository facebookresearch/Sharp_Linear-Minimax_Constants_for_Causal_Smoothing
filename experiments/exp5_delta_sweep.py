# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Experiment 5: finite-sampling-rate convergence of a sampled causal filter.

Studies how one moment-corrected discretization of the continuous optimum
approaches K_p* as the sampling step Delta tends to zero.  This is a
filterwise calculation, not a discrete-time minimax result.

Mechanism (no Monte-Carlo; a deterministic quadrature study):
  * Build a deep corrected cascade approximation of g*.
  * Discretize it on M cells by the EXACT cell integrals h0_j = int_cell g*, then
    PROJECT onto the discrete moment constraints sum_j h_j j^m = delta_{m0}
    (m=0..p) so the discrete filter reproduces degree-<=p polynomials exactly --
    the same moment-exact FIR of g* used by exp4.gstar_taps.
  * K(M) is the scale-invariant quality of this particular discretized filter.
    Its convergence is a filterwise numerical check, not a proof that the full
    discrete minimax constant converges.

What is reported (see the findings block printed at the end):
  1. p=2 headline: K(M), rel(M) over a geometric M grid; the convergence rate
     from an ASYMPTOTIC (tail) log-log fit AND the naive all-points fit (the two
     differ because small M is pre-asymptotic).
  2. rel(M) at M=10,30,100,300, evaluated directly.
  3. p=1 and p=0 sweeps for the trend across orders.

The Peano-kernel norm is integrated exactly on each polynomial sampling cell.
"""

import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
from numpy.typing import NDArray

mpl.use("Agg")
import matplotlib.pyplot as plt

from . import paper_style
from .optimal_cascade import CascadeFilter, K_STAR, shape_constant_from_taps

paper_style.apply()

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
FloatArray = NDArray[np.float64]


def sweep(p: int, Ms: list[int]) -> tuple[FloatArray, FloatArray]:
    """Evaluate moment-corrected discretizations of the cascade approximation."""
    shape = CascadeFilter(p, cells=120)
    Ks = np.array(
        [shape_constant_from_taps(shape.cell_taps(int(M), True), p) for M in Ms]
    )
    rels = (Ks - K_STAR[p]) / K_STAR[p]
    return Ks, rels


def loglog_slope(
    Ms: list[int], rels: FloatArray, mask: NDArray[np.bool_] | None = None
) -> tuple[float, float]:
    """Least-squares slope of log|rel| vs log M (optionally over a sub-mask)."""
    x = np.log(np.asarray(Ms, float))
    y = np.log(np.abs(rels))
    if mask is not None:
        x, y = x[mask], y[mask]
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope), float(intercept)


def rel_at(M_query: int, Ms: list[int], rels: FloatArray) -> float:
    """|rel| at an arbitrary M by linear interpolation in (log M, log|rel|) space
    (the curve is a near-straight line there in the asymptotic range)."""
    lm = np.log(np.asarray(Ms, float))
    ly = np.log(np.abs(rels))
    return float(np.exp(np.interp(np.log(M_query), lm, ly)))


# ----------------------------------------------------------------------
# 1) headline p=2 sweep on the geometric grid
# ----------------------------------------------------------------------
Ms_full = [
    8,
    10,
    12,
    16,
    24,
    30,
    32,
    48,
    64,
    96,
    100,
    128,
    192,
    256,
    300,
    384,
    512,
    768,
    1024,
    1536,
    2048,
]
K2, rel2 = sweep(2, Ms_full)

print(f"=== p=2 (headline, K_2* = {K_STAR[2]:.6f}) ===")
print(f"{'M':>5} {'K(M)':>12} {'rel(M)':>13} {'|rel|':>11}")
for M, K, r in zip(Ms_full, K2, rel2):
    print(f"{M:>5} {K:>12.6f} {r:>+13.4e} {abs(r):>11.4e}")

# asymptotic (tail) vs naive (all-points) log-log fit.  Small M is pre-asymptotic
# so the tail fit is the relevant empirical rate; the all-points fit is reported
# alongside to show the pre-asymptotic bias.
tail_mask = np.array(Ms_full) >= 512
slope_all2, _ = loglog_slope(Ms_full, rel2)
slope_tail2, _ = loglog_slope(Ms_full, rel2, tail_mask)
print(
    f"\nlog|rel| vs log M slope:  all-M = {slope_all2:.3f}   "
    f"tail (M>=512) = {slope_tail2:.3f}"
)
print(
    f"-> asymptotic rate ~ O(M^{slope_tail2:.2f}) = O(Delta^{-slope_tail2:.2f}); "
    f"O(Delta) would be slope -1."
)

# 2) percent-level test at M = 10, 30, 100, 300
q_Ms = [10, 30, 100, 300]
rel_q2 = {M: rel_at(M, Ms_full, rel2) for M in q_Ms}
print("\nrelative deviation at target tap counts:")
for M in q_Ms:
    print(f"  M={M:>4}:  |rel| = {rel_q2[M] * 100:7.3f} %")

# ----------------------------------------------------------------------
# 3) trend across orders: p=1 and p=0 (a few M values is enough; p=2 is detailed)
# ----------------------------------------------------------------------
Ms_other = [8, 10, 16, 30, 32, 64, 100, 128, 256, 300, 512, 1024, 2048]
K1, rel1 = sweep(1, Ms_other)
K0, rel0 = sweep(0, Ms_other)
tail_mask_o = np.array(Ms_other) >= 512
slope_tail1, _ = loglog_slope(Ms_other, rel1, tail_mask_o)
slope_tail0, _ = loglog_slope(Ms_other, rel0, tail_mask_o)
slope_all1, _ = loglog_slope(Ms_other, rel1)
slope_all0, _ = loglog_slope(Ms_other, rel0)

for p, Ms, K, r, st in [
    (1, Ms_other, K1, rel1, slope_tail1),
    (0, Ms_other, K0, rel0, slope_tail0),
]:
    print(f"\n=== p={p} (K_{p}* = {K_STAR[p]:.6f}) ===")
    print(f"{'M':>5} {'K(M)':>12} {'rel(M)':>13}")
    for MM, KK, rr in zip(Ms, K, r):
        print(f"{MM:>5} {KK:>12.6f} {rr:>+13.4e}")
    print(
        f"tail (M>=512) slope = {st:.3f};  "
        f"|rel| at M=10/30/100/300 = "
        + ", ".join(f"{rel_at(M, Ms, r) * 100:.2f}%" for M in q_Ms)
    )

# ----------------------------------------------------------------------
# save results
# ----------------------------------------------------------------------
results = {
    "description": "Finite-sampling-rate convergence of a sampled causal "
    "filter: K(M)=shape_constant_from_taps(moment-exact discretization "
    "of g* on M cells) -> K_p* as M=rho/Delta -> infinity. "
    "rel(M)=(K(M)-K_p*)/K_p* is the finite-rate correction.",
    "K_star": {str(k): v for k, v in K_STAR.items()},
    "p2": {
        "M": Ms_full,
        "K": [float(v) for v in K2],
        "rel": [float(v) for v in rel2],
        "slope_loglog_all": slope_all2,
        "slope_loglog_tail_Mge512": slope_tail2,
        "rel_at": {str(M): rel_q2[M] for M in q_Ms},
    },
    "p1": {
        "M": Ms_other,
        "K": [float(v) for v in K1],
        "rel": [float(v) for v in rel1],
        "slope_loglog_all": slope_all1,
        "slope_loglog_tail_Mge512": slope_tail1,
        "rel_at": {str(M): rel_at(M, Ms_other, rel1) for M in q_Ms},
    },
    "p0": {
        "M": Ms_other,
        "K": [float(v) for v in K0],
        "rel": [float(v) for v in rel0],
        "slope_loglog_all": slope_all0,
        "slope_loglog_tail_Mge512": slope_tail0,
        "rel_at": {str(M): rel_at(M, Ms_other, rel0) for M in q_Ms},
    },
    "verdict": (
        f"For p=2 the observed tail rate is approximately O(M^{slope_tail2:.2f}), "
        "and K(M) approaches K_p* from below. The absolute relative deviations "
        f"are {100 * rel_q2[30]:.3f}%, {100 * rel_q2[100]:.3f}%, and "
        f"{100 * rel_q2[300]:.3f}% at M=30, 100, and 300. This is a "
        "filterwise numerical rate for the chosen cell integration and "
        "moment correction, not a discrete-minimax convergence theorem."
    ),
}
with (RESULTS / "exp5_results.json").open("w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {RESULTS / 'exp5_results.json'}")

# ----------------------------------------------------------------------
# figure: log-log |rel(M)| vs M, one line per p, with fitted tail slope
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=paper_style.figsize(1))
series = [
    (2, Ms_full, rel2, slope_tail2, "#2166AC", "o"),
    (1, Ms_other, rel1, slope_tail1, "#4DAF4A", "s"),
    (0, Ms_other, rel0, slope_tail0, "#B2182B", "^"),
]
for p, Ms, rels, st, col, mk in series:
    ax.loglog(
        Ms,
        np.abs(rels),
        mk + "-",
        color=col,
        ms=5,
        lw=1.4,
        label=f"$p={p}$  (tail slope ${st:.2f}$)",
    )

# O(1/M) reference guide anchored to the p=2 tail
Mg = np.array([512, 2048], float)
anchor = np.abs(rel2)[np.array(Ms_full) == 512][0]
ax.loglog(
    Mg,
    anchor * (Mg / 512.0) ** (-1.0),
    "k--",
    lw=1.0,
    alpha=0.7,
    label=r"$O(1/M)$ guide",
)

# percent-level reference line
ax.axhline(0.01, color="gray", ls=":", lw=1.0)
ax.text(Ms_full[0], 0.0115, "1% level", fontsize=8.5, color="gray")

ax.set_xlabel(r"tap count $N=\rho/\Delta$ (effective memory in samples)")
ax.set_ylabel(r"$|\,K(N)-K_p^*\,|\,/\,K_p^*$")
ax.set_title(
    "Finite-rate behavior of a moment-corrected sampled filter\n"
    r"(moment-exact discretization of $g^*$)"
)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=9, loc="lower left")
plt.tight_layout()
plt.savefig(FIGURES / "exp5_delta_sweep.png", dpi=150, bbox_inches="tight")
plt.savefig(FIGURES / "exp5_delta_sweep.pdf", bbox_inches="tight")
print(f"Saved exp5 figures under {FIGURES}")
