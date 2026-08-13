# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Experiment 1 (REWRITTEN): calibrated Monte-Carlo validation of the sharp
constant K_p*.

The original version (a) never actually ran the optimal filter (dead code),
(b) measured AVERAGE-case MSE on benign signals and compared it to a WORST-case
bound, and (c) used an uncalibrated signal amplitude -- so the points fell far
below the theory line with the wrong slope.

Here we validate the worst-case theory on its own terms:
  * the OPTIMAL filter g* (perfect spline) is discretized and actually applied;
  * for each sigma the bandwidth rho* is set to the worst-case optimum;
  * a piecewise-constant approximation to the continuous least-favorable
    derivative s^{(p+1)} = D*sign(Phi_rho) is integrated exactly.  Its
    finite-rate MSE must approach
    K_p* D^{2/(2p+3)} sigma^{4(p+1)/(2p+3)} in both slope and constant.
We also plot the AVERAGE-case MSE (benign signal) to show -- correctly labelled --
that it sits below the worst-case bound (as it must).
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
    CascadeFilter,
    integrate_piecewise_constant_top_derivative,
    K_STAR,
    peano_from_grid,
)

np.random.seed(0)

paper_style.apply()

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

D = 1.0
sigmas = np.logspace(-2, 0, 13)
N = 600  # samples across the (normalized) support [0,1]
N_TRIALS = 400
FloatArray = NDArray[np.float64]


def gstar_float(p: int) -> tuple[FloatArray, FloatArray, float, float, float]:
    """Return a deep cascade approximation and the certified constant."""
    shape = CascadeFilter(p, cells=120)
    h = shape.cell_taps(N, correct_moments=True)
    xs, gv = shape.grid(12001)
    phi = peano_from_grid(xs, gv, p)
    interval_midpoints = (np.arange(N - 1, dtype=float) + 0.5) / N
    phi_sign = np.sign(np.interp(interval_midpoints, xs, phi))
    L2 = float(np.trapezoid(gv * gv, xs))
    L1 = float(np.trapezoid(np.abs(phi), xs))
    return h, phi_sign, L2, L1, K_STAR[p]


def worst_case_signal(phi_sign: FloatArray, p: int, dt: float) -> FloatArray:
    """Build an in-class, piecewise-constant approximation to the adversary."""
    return integrate_piecewise_constant_top_derivative(D * phi_sign, dt, p + 1)


def benign_signal(p: int, dt: float, N: int) -> FloatArray:
    """An in-class signal with a sinusoidal piecewise-constant top derivative."""
    u = (np.arange(N - 1) + 0.5) * dt
    T = N * dt
    deriv = D * np.sin(2 * np.pi * 1.3 * u / T + 0.7)
    return integrate_piecewise_constant_top_derivative(deriv, dt, p + 1)


results = {}
fig, axes = plt.subplots(1, 3, figsize=paper_style.figsize(3))
titles = [r"$p=0$: position", r"$p=1$: pos.+vel.", r"$p=2$: pos.+vel.+accel."]

for p in range(3):
    q = 2 * p + 3
    r = 2 * (p + 1)
    h, phi_sign, L2, L1, K = gstar_float(p)

    mse_theory, mse_wc, mse_avg, bias_check = [], [], [], []
    for sigma in sigmas:
        # worst-case-optimal bandwidth rho*:  min A rho^{2(p+1)} + B/rho
        A = (D * L1) ** 2
        B = sigma**2 * L2
        rho = (B / (2 * (p + 1) * A)) ** (1.0 / (2 * p + 3))
        dt = rho / N  # real-time sample step
        sigma_d = sigma * np.sqrt(1.0 / dt)  # sigma^2 = sigma_d^2 * dt

        mse_theory.append(K * D ** (2 / q) * sigma ** (2 * r / q))

        # ---- worst case (g*) ----
        F = worst_case_signal(phi_sign, p, dt)
        s0 = F[0]  # = 0 (present value)
        bias = s0 - np.dot(h, F)  # piecewise-constant in-class adversarial bias
        bias_check.append(abs(bias) / (D * rho ** (p + 1) * L1))  # should be ~1
        errs = []
        for _ in range(N_TRIALS):
            w = np.random.randn(len(h)) * sigma_d
            est = np.dot(h, F + w)
            errs.append((s0 - est) ** 2)
        mse_wc.append(np.mean(errs))

        # ---- average case (benign signal, same g*, same bandwidth) ----
        Fb = benign_signal(p, dt, len(h))
        s0b = Fb[0]
        eb = []
        for _ in range(N_TRIALS):
            w = np.random.randn(len(h)) * sigma_d
            eb.append((s0b - np.dot(h, Fb + w)) ** 2)
        mse_avg.append(np.mean(eb))

    results[p] = {
        "sigma": sigmas.tolist(),
        "K_star": K,
        "mse_theory": mse_theory,
        "mse_worstcase_emp": mse_wc,
        "mse_average_emp": mse_avg,
        "bias_calibration": bias_check,
    }

    ax = axes[p]
    ax.loglog(
        sigmas,
        mse_theory,
        "b-",
        lw=2,
        label=rf"theory $K_{p}^*\,\sigma^{{{2 * r}/{q}}}$",
    )
    ax.loglog(sigmas, mse_wc, "r^", ms=7, alpha=0.85, label=r"$g^*$ adversarial")
    ax.loglog(sigmas, mse_avg, "gv", ms=6, alpha=0.7, label=r"$g^*$ benign trajectory")
    ax.set_xlabel(r"noise level $\sigma$")
    ax.set_title(titles[p])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    if p == 0:
        ax.set_ylabel("MSE")

    slope = np.polyfit(np.log(sigmas), np.log(mse_wc), 1)[0]
    ratio = np.median(np.array(mse_wc) / np.array(mse_theory))
    print(
        f"p={p}: theory slope={2 * r / q:.3f}  WC-emp slope={slope:.3f}  "
        f"WC/theory median={ratio:.3f}  bias-calib={np.mean(bias_check):.4f}"
    )

plt.tight_layout()
plt.savefig(FIGURES / "exp1_monte_carlo.png", dpi=150, bbox_inches="tight")
plt.savefig(FIGURES / "exp1_monte_carlo.pdf", bbox_inches="tight")
with (RESULTS / "exp1_results.json").open("w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved exp1 outputs under {ROOT}")
