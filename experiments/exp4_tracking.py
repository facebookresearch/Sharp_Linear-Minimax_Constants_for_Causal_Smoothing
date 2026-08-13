# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Experiment 4 (REWRITTEN v2): practical tracking on genuinely IN-CLASS signals,
benchmarked against the OPTIMAL g*, on a COMMON scale.

Fixes over the previous version (reviewer feedback):
  * SIGNALS ARE IN THE HOLDER CLASS.  The old 'maneuver' (piecewise-constant
    velocity) and 'step_accel' (piecewise-constant acceleration) have Dirac-delta
    p-th derivatives -- NOT in X_{p+1}(D).  Here every signal is built by bounding
    the top derivative, s^{(p+1)}, by a common D and propagating the derivative
    state exactly, so all signals lie in X_3(D) with the SAME D.
  * COMMON, COMMENSURABLE SCALE.  All signals share one D and one sigma, so the
    empirical MSEs and the worst-case benchmark K_p* D^{2/q} sigma^{4(p+1)/q} are
    in the same units and can be plotted on one axis (old panels used D=1 bars vs
    signals with effective D ~ 1e3).
  * FAIR g* DISCRETIZATION.  The FIR taps of g* are projected onto the EXACT
    discrete moment constraints sum_j h_j (j)^m = delta_{m0} (m=0..p), so g*
    reproduces polynomials of degree <= p exactly in discrete time -- exactly as
    BB-type (by its recursion) and SG (by least squares) do.  Without this the cell-
    integral taps carry an O(dt) reproduction error that, on a high-derivative
    signal, dwarfs the sidelobe effect and unfairly penalizes g*.

The trajectory study is not a minimax test.  A filter tuned to one displayed
trajectory may beat g* on that trajectory.  The separate left panel reports
continuous-limit worst-case constants; the right panel reports only
oracle-tuned empirical MSE.

The comparison uses D=0.07, sigma_d=0.3, and dt=0.1.  A 360-sample warm-up
precedes 500 evaluated samples, and SNR is reported on that retained segment.
"""

import json
from collections.abc import Callable
from functools import cache
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
    savitzky_golay_continuous_constant,
)

np.random.seed(42)

paper_style.apply()

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# ---------- shared derivative bound and noise regime ----------
# The retained trajectory SNRs are 45--59 dB.  The resulting continuous
# worst-case-optimal memory is about 85 taps and lies inside the FIR grids below.
P = 2  # pos + vel + accel
D = 0.07  # common Holder bound ||s'''||_inf = D  (in-class)
sigma_d = 0.3
dt = 0.1
sigma_cont = sigma_d * np.sqrt(dt)
EVALUATION_SAMPLES = 500
MAX_FIR_MEMORY = 360
START = MAX_FIR_MEMORY
N = START + EVALUATION_SAMPLES
N_TRIALS = 40
freq = 1.0 / dt
q, r = 2 * P + 3, 2 * (P + 1)
FloatArray = NDArray[np.float64]

# p=2 worst-case shape-quality constants (from exp2/exp3)
K_OPT, K_BB = K_STAR[P], 2.554
K_SG = savitzky_golay_continuous_constant(P)

# ---------- optimal filter g* (fair, moment-exact discretization) ----------
_tap_cache = {}


@cache
def optimal_shape() -> CascadeFilter:
    return CascadeFilter(P, cells=120)


def gstar_taps(M: int) -> FloatArray:
    """Cell-integral taps of g* (shape) projected onto the EXACT discrete moment
    constraints sum_j h_j j^m = delta_{m0}, m=0..P, so g* reproduces degree-<=P
    polynomials exactly in discrete time (fair vs BB-type/SG)."""
    if M not in _tap_cache:
        _tap_cache[M] = optimal_shape().cell_taps(M, correct_moments=True)
    return _tap_cache[M]


def apply_fir(z: FloatArray, h: FloatArray) -> FloatArray:
    N = len(z)
    M = len(h)
    out = np.zeros(N)
    for n in range(N):
        k = min(M, n + 1)
        out[n] = np.dot(h[:k], z[n : n - k if n - k >= 0 else None : -1])
    return out


def bb_filter(z: FloatArray, alpha: float) -> FloatArray:
    """Apply the alpha-beta recursion and its p=2 small-gain extension."""
    N = len(z)
    out = np.zeros(N)
    beta = alpha**2 / (2 - alpha)
    gamma = alpha**3 / 6
    pos, vel, acc = z[0], 0.0, 0.0
    for n in range(N):
        pred = pos + vel + 0.5 * acc
        vp = vel + acc
        inn = z[n] - pred
        pos = pred + alpha * inn
        vel = vp + beta * inn
        acc += gamma * inn
        out[n] = pos
    return out


def sg_filter(z: FloatArray, W: int) -> FloatArray:  # causal Savitzky-Golay
    N = len(z)
    out = np.zeros(N)
    t = np.arange(W, dtype=float)
    X = np.column_stack([t**k for k in range(P + 1)])
    wts = np.linalg.lstsq(X, np.eye(W), rcond=None)[0][0, :]
    for n in range(N):
        out[n] = z[n] if n < W else np.dot(wts, z[n - W + 1 : n + 1][::-1])
    return out


def one_euro(
    z: FloatArray, mincut: float, beta_oe: float, dcut: float
) -> FloatArray:  # nonlinear/adaptive
    N = len(z)
    out = np.zeros(N)

    def smoothing_factor(fc: float) -> float:
        return 1.0 / (1.0 + (1.0 / (2 * np.pi * fc)) / (1.0 / freq))

    xh, dxh = z[0], 0.0
    for i in range(N):
        dx = (z[i] - z[i - 1]) * freq if i > 0 else 0.0
        dxh = smoothing_factor(dcut) * dx + (1 - smoothing_factor(dcut)) * dxh
        al = smoothing_factor(mincut + beta_oe * abs(dxh))
        xh = al * z[i] + (1 - al) * xh
        out[i] = xh
    return out


def kalman_filter(z: FloatArray, qq: float) -> FloatArray:
    """Steady-state-recursion Kalman filter for the discrete white-noise-jerk
    (nearly-constant-acceleration) model, state x=[pos,vel,acc].  Free parameter
    is the jerk PSD qq.  This is the recursive stochastic-model baseline."""
    N = len(z)
    out = np.zeros(N)
    F = np.array([[1.0, dt, dt * dt / 2.0], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
    Q = qq * np.array(
        [
            [dt**5 / 20.0, dt**4 / 8.0, dt**3 / 6.0],
            [dt**4 / 8.0, dt**3 / 3.0, dt**2 / 2.0],
            [dt**3 / 6.0, dt**2 / 2.0, dt],
        ]
    )
    H = np.array([[1.0, 0.0, 0.0]])
    R = sigma_d**2
    identity = np.eye(3)
    x = np.array([z[0], 0.0, 0.0])
    P = np.diag([1e3, 1e3, 1e3])
    for n in range(N):
        xp = F @ x
        Pp = F @ P @ F.T + Q
        S = (H @ Pp @ H.T)[0, 0] + R
        K = (Pp @ H.T) / S  # (3,1)
        x = xp + (K[:, 0]) * (z[n] - (H @ xp)[0])
        P = (identity - K @ H) @ Pp
        out[n] = x[0]
    return out


# ---------- in-class signals: bound s''' by D, integrate P+1 times ----------
def inclass_signal(kind: str, T_adv: float | None = None) -> FloatArray:
    """Sample a signal with bounded piecewise-constant top derivative."""
    t = (np.arange(N - 1) + 0.5) * dt
    if kind == "smooth":  # benign, low-frequency jerk
        j = np.sin(2 * np.pi * 0.12 * t)
    elif kind == "variable":  # richer multi-tone jerk
        j = (
            np.sin(2 * np.pi * 0.10 * t)
            + 0.6 * np.sin(2 * np.pi * 0.29 * t + 1.0)
            + 0.4 * np.sin(2 * np.pi * 0.50 * t + 2.0)
        )
    else:  # bang-bang jerk: saturates
        j = np.sign(np.sin(2 * np.pi * t / T_adv))  # the Holder bound (hardest)
    j = j / np.max(np.abs(j)) * D  # enforce ||s'''||_inf = D
    return integrate_piecewise_constant_top_derivative(j, dt, P + 1)


def mean_mse(
    make_out: Callable[[FloatArray], FloatArray],
    s: FloatArray,
    noises: list[FloatArray],
    start: int = START,
) -> float:
    """Mean MSE over the PRECOMPUTED, per-scenario noise realizations `noises`
    (a length-N_TRIALS list of length-N arrays) for a filter callable make_out(z).
    PAIRED comparison: every filter within a scenario is averaged over the SAME
    `noises`, so results no longer depend on evaluation order.  The warm-up is
    at least the longest candidate FIR memory."""
    e = 0.0
    for eps in noises:
        z = s + eps
        o = make_out(z)
        e += np.mean((o[start:] - s[start:]) ** 2)
    return e / len(noises)


# Worst-case-optimal memory.  These are support-normalized numerical metrics
# from the corrected p=2 cascade, not the old two-knot spline.
_x_shape, _g_shape = optimal_shape().grid(16001)
_phi_shape = peano_from_grid(_x_shape, _g_shape, P)
L1 = float(np.trapezoid(np.abs(_phi_shape), _x_shape))
L2 = float(np.trapezoid(_g_shape * _g_shape, _x_shape))
rho_star = (sigma_cont**2 * L2 / (2 * (P + 1) * (D * L1) ** 2)) ** (1.0 / q)
M_star = round(rho_star / dt)
T_adv = 2 * rho_star
print(
    f"regime: D={D}, sigma_d={sigma_d}, sigma_cont={sigma_cont:.6f}, "
    f"rho*={rho_star:.3f} (M*~{M_star} taps), adversarial period={T_adv:.2f}s"
)

# ---------- oracle-tuned best MSE per filter per scenario ----------
# Grids for the current regime: the optimal memory grows as SNR drops,
# so g*/SG lengths run to 360 taps and BB-type alpha / Kalman q reach smaller values;
# the reported comparison remains explicitly in-grid rather than a global tuning claim.
Ms = list(range(6, MAX_FIR_MEMORY + 1, 6))
alphas = np.logspace(-3.5, -0.05, 44)
Ws = list(range(4, MAX_FIR_MEMORY + 1, 6))
qs = np.concatenate(([0.0], np.logspace(-8, 4, 44)))  # includes deterministic limit
# 1-Euro tuned over all three params; speed-coef beta grid widened to 30 so the
# adaptive filter is not edge-limited (its best MSE plateaus well before that).
euro_mc = np.logspace(np.log10(0.02), np.log10(30), 15)
euro_be = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
euro_dc = [0.3, 1.0, 3.0, 10.0]
euros = [(mc, be, dc) for mc in euro_mc for be in euro_be for dc in euro_dc]

scen = [
    ("smooth", "Smooth (benign)", None),
    ("variable", "Variable (multi-tone)", None),
    ("bangbang", "Periodic bang-bang jerk", T_adv),
]

all_res = {}
plot_res: dict[str, dict[str, float]] = {}
edge = []
print(
    f"\n{'scenario':<22} {'SNR':>7} {'g*':>10} {'BB-type':>16} {'Kalman':>16} {'SG':>16} "
    f"{'1-Euro':>16}"
)
for key, title, Ta in scen:
    s = inclass_signal(key, Ta)
    # in-class SNR on the evaluated segment: SNR_dB = 10 log10(var(s[start:]) / sigma^2)
    snr_db = 10.0 * np.log10(np.var(s[START:]) / sigma_d**2)
    # PAIRED Monte-Carlo: pre-generate the N_TRIALS noise arrays ONCE per scenario
    # (from the fixed top-level seed 42), then average EVERY filter over these SAME
    # arrays.  This makes results independent of filter evaluation order and gives a
    # paired comparison across g*, BB-type, Kalman, SG, and 1-Euro.
    noises = [np.random.randn(N) * sigma_d for _ in range(N_TRIALS)]
    # tune each filter on the mean paired MSE (fair oracle: best expected MSE)
    g_by = {
        M: mean_mse(lambda z, M=M: apply_fir(z, gstar_taps(M)), s, noises) for M in Ms
    }
    a_by = {al: mean_mse(lambda z, al=al: bb_filter(z, al), s, noises) for al in alphas}
    k_by = {qv: mean_mse(lambda z, qv=qv: kalman_filter(z, qv), s, noises) for qv in qs}
    w_by = {W: mean_mse(lambda z, W=W: sg_filter(z, W), s, noises) for W in Ws}
    e_by = {pe: mean_mse(lambda z, pe=pe: one_euro(z, *pe), s, noises) for pe in euros}
    gM = min(g_by, key=lambda value: g_by[value])
    ba = min(a_by, key=lambda value: a_by[value])
    kq = min(k_by, key=lambda value: k_by[value])
    sW = min(w_by, key=lambda value: w_by[value])
    ep = min(e_by, key=lambda value: e_by[value])
    gs, bb, ka, sg, oe = g_by[gM], a_by[ba], k_by[kq], w_by[sW], e_by[ep]
    plot_res[key] = {"gstar": gs, "bb": bb, "kalman": ka, "sg": sg, "euro": oe}
    all_res[key] = {
        "snr_db": snr_db,
        "gstar": gs,
        "bb": bb,
        "kalman": ka,
        "sg": sg,
        "euro": oe,
        "bb_ratio": bb / gs,
        "kalman_ratio": ka / gs,
        "sg_ratio": sg / gs,
        "euro_ratio": oe / gs,
        "argmin": {
            "gM": gM,
            "alpha": float(ba),
            "q": float(kq),
            "sW": sW,
            "euro": [float(ep[0]), ep[1], ep[2]],
        },
    }
    for nm, v, grid in [
        ("g*", gM, Ms),
        ("BB-type", ba, list(alphas)),
        ("SG", sW, Ws),
        ("Kalman", kq, list(qs)),
    ]:
        is_kalman_domain_endpoint = nm == "Kalman" and np.isclose(v, 0.0)
        if not is_kalman_domain_endpoint and (v == grid[0] or v == grid[-1]):
            edge.append(f"{key}/{nm}")
    # 1-Euro's MSE plateaus in the speed-coef beta (its argmin sits at the beta grid
    # max only because the objective is flat there, not because it is grid-starved).
    # So instead of flagging the (immaterial) boundary, verify that pushing beta far
    # higher does not materially improve the MSE -- i.e. it is genuinely well-tuned.
    oe_hi = mean_mse(lambda z, ep=ep: one_euro(z, ep[0], 100.0, ep[2]), s, noises)
    if oe_hi < 0.99 * oe:
        edge.append(f"{key}/1E(beta-not-plateaued)")
    print(
        f"{title:<22} {snr_db:>5.1f}dB {gs:>8.5f} {bb:>9.5f}({bb / gs:.2f}x) {ka:>9.5f}({ka / gs:.2f}x) "
        f"{sg:>9.5f}({sg / gs:.2f}x) {oe:>9.5f}({oe / gs:.2f}x)"
    )
    # Report the lowest empirical MSE without interpreting it as a minimax order.
    mf_best = min(gs, bb, ka, sg, oe)
    if gs > mf_best + 1e-12:
        which = min(
            [("BB-type", bb), ("Kalman", ka), ("SG", sg), ("1-Euro", oe)],
            key=lambda t: t[1],
        )[0]
        print(
            f"    NOTE: {which} is lower than g* on '{key}' by "
            f"{100 * (gs - mf_best) / gs:.2f}%"
        )
print("edge warnings:", edge if edge else "none (all tuned within grid -> fair)")

# ---------- continuous-limit worst-case benchmark at the SAME D, sigma -----
wc = {"gstar": K_OPT, "bb": K_BB, "sg": K_SG}
wc_mse = {k: wc[k] * D ** (2 / q) * sigma_cont ** (2 * r / q) for k in wc}
all_res["_continuous_limit_worstcase_mse"] = dict(wc_mse, euro=None)
all_res["_regime"] = {
    "D": D,
    "sigma_d": sigma_d,
    "sigma_cont": sigma_cont,
    "dt": dt,
    "N": N,
    "warmup_samples": START,
    "evaluation_samples": EVALUATION_SAMPLES,
    "p": P,
    "rho_star": rho_star,
    "M_star": M_star,
    "N_trials": N_TRIALS,
}
with (RESULTS / "exp4_results.json").open("w") as f:
    json.dump(all_res, f, indent=2)
print(
    f"\ncontinuous-limit worst-case MSE (D={D}, sigma_cont={sigma_cont:.6f}): "
    f"g*={wc_mse['gstar']:.3f}  BB-type={wc_mse['bb']:.3f} (+{100 * (K_BB / K_OPT - 1):.0f}%)  "
    f"SG={wc_mse['sg']:.3f} (+{100 * (K_SG / K_OPT - 1):.0f}%)  1-Euro=inf"
)

# ---------- plot: continuous benchmark (left) + sampled tracking (right) -----
fig, axes = plt.subplots(1, 2, figsize=(paper_style.WIDTH_FULL, 4.2))
labels = [
    ("gstar", "Optimal $g^*$", "#2166AC"),
    ("bb", "BB-type", "#B2182B"),
    ("sg", "SG", "#4DAF4A"),
]

axW = axes[0]
_woff = max(wc_mse.values()) * 0.02  # label offset ~2% of axis
gap = {k: 100 * (wc_mse[k] - wc_mse["gstar"]) / wc_mse["gstar"] for k in wc_mse}
gtxt = {"gstar": "optimal", "bb": f"+{gap['bb']:.0f}%", "sg": f"+{gap['sg']:.0f}%"}
for i, (k, _lab, col) in enumerate(labels[:3]):
    axW.bar(i, wc_mse[k], 0.6, color=col)
    axW.text(
        i, wc_mse[k] + _woff, f"{wc_mse[k]:.3f}\n({gtxt[k]})", ha="center", fontsize=8.5
    )
axW.set_xticks(range(3))
axW.set_xticklabels(["$g^*$", "BB-type", "SG"])
axW.set_ylabel("MSE")
axW.set_ylim(0, max(wc_mse.values()) * 1.28)
axW.set_title("Continuous-time benchmark over $\\mathcal{X}_3(D)$\n$g^*$ optimal")
axW.grid(True, alpha=0.3, axis="y")

# Right panel: trajectory-specific MSE relative to g*.
axA = axes[1]
rlabels = [
    ("gstar", "Optimal $g^*$", "#2166AC"),
    ("bb", "BB-type", "#B2182B"),
    ("kalman", "Kalman", "#FF7F00"),
    ("sg", "SG", "#4DAF4A"),
]
x = np.arange(len(scen))
w = 0.15
nb = len(rlabels)
_sgmax = max(plot_res[s[0]]["sg"] / plot_res[s[0]]["gstar"] for s in scen)
YCAP = max(1.20, _sgmax * 1.10)
for i, (k, lab, col) in enumerate(rlabels):
    ratios = [plot_res[s[0]][k] / plot_res[s[0]]["gstar"] for s in scen]
    off = (i - (nb - 1) / 2.0) * w
    axA.bar(x + off, ratios, w, color=col, label=lab)
    if k != "gstar":
        for xi, rr in zip(x, ratios):
            axA.text(
                xi + off,
                rr + 0.004,
                f"{rr:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.0,
            )
axA.axhline(1.0, ls="--", color="#2166AC", lw=1.2, alpha=0.8)
axA.text(-0.5, 1.0, "$g^*$", fontsize=8, color="#2166AC", va="bottom")
axA.set_xticks(x)
axA.set_xticklabels([s[1].replace(" (", "\n(") for s in scen], fontsize=7.5)
axA.set_ylabel("MSE relative to $g^*$ (oracle-tuned per trajectory)")
axA.set_ylim(0, YCAP)
axA.set_title(
    "Sampled in-class tracking ($p=2$, $\\mathcal{X}_3(D)$)\n"
    "all filters oracle-tuned per trajectory"
)
axA.legend(fontsize=8.0, ncol=4, loc="upper center")
axA.grid(True, alpha=0.3, axis="y")

plt.tight_layout(w_pad=2.5)
plt.savefig(FIGURES / "exp4_tracking.png", dpi=150, bbox_inches="tight")
plt.savefig(FIGURES / "exp4_tracking.pdf", bbox_inches="tight")
print(f"\nSaved exp4 outputs under {ROOT}")
