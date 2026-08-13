# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shape constants K of the Benedict-Bordner filters.

Combines the state-space impulse response, Lyapunov variance, and taps-to-K
calculation. Running this file reproduces the BB entries of the
classical-filter table.

For order p the recipe is

  1. closed loop  A = (I - b c^T) F  from the gain alpha,
  2. impulse response  h_j = c A^j b,
  3. variance  V = sum_j h_j^2  (cross-checked against the discrete Lyapunov
     solution P = A P A^T + b b^T, V = c P c^T),
  4. Peano kernel  Phi = (p+1)-fold cumulative sum of  e_j = delta_{j0} - h_j,
     and  B = sum_j |Phi_j|,
  5. K = (2r+1) (2r)^{-2r/(2r+1)} B^{2/(2r+1)} V^{2r/(2r+1)},  r = p+1.

K is dilation invariant, so no sampling interval enters.  K(alpha) increases as
alpha decreases and converges to the constant of the continuous limiting shape;
that limit is the reported value.
"""

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

P_ORDERS = (0, 1, 2)
ALPHAS = (0.05, 0.02, 0.01, 0.005, 0.002)


def bb_filter(
    p: int, alpha: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Closed-loop state space (A, b, c) of the order-p Benedict-Bordner filter."""
    beta = alpha**2 / (2 - alpha)
    gamma = alpha**3 / 6
    if p == 0:
        F = np.array([[1.0]])
        b = np.array([alpha])
        c = np.array([1.0])
    elif p == 1:
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        b = np.array([alpha, beta])
        c = np.array([1.0, 0.0])
    elif p == 2:
        F = np.array([[1.0, 1.0, 0.5], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
        b = np.array([alpha, beta, gamma])
        c = np.array([1.0, 0.0, 0.0])
    else:
        raise ValueError(f"no Benedict-Bordner form for p={p}")
    return (np.eye(p + 1) - np.outer(b, c)) @ F, b, c


def impulse_response(p: int, alpha: float, n: int) -> np.ndarray:
    A, b, c = bb_filter(p, alpha)
    h = np.empty(n)
    state = b.copy()
    for j in range(n):
        h[j] = c @ state
        state = A @ state
    return h


def variance_lyapunov(p: int, alpha: float) -> float:
    """Exact steady-state output variance per unit input variance."""
    A, b, c = bb_filter(p, alpha)
    n = p + 1
    P = np.linalg.solve(np.eye(n * n) - np.kron(A, A), np.outer(b, b).reshape(-1))
    return float(c @ P.reshape(n, n) @ c)


def peano_kernel(h: np.ndarray, p: int) -> np.ndarray:
    """Discrete Peano kernel: (p+1)-fold cumulative sum of the error weights."""
    e = -h.copy()
    e[0] += 1.0
    for _ in range(p + 1):
        e = np.cumsum(e)
    return e


def shape_constant(p: int, alpha: float, n: int) -> dict:
    h = impulse_response(p, alpha, n)
    phi = peano_kernel(h, p)
    B = float(np.abs(phi).sum())
    V = float((h**2).sum())
    r = p + 1
    q = 2 * r + 1
    K = q * (2 * r) ** (-2 * r / q) * B ** (2 / q) * V ** (2 * r / q)
    return {
        "alpha": alpha,
        "n": n,
        "K": K,
        "B": B,
        "V": V,
        "V_lyapunov": variance_lyapunov(p, alpha),
        "dc_gain": float(h.sum()),
        "tail_frac": float(abs(phi[-1]) / B),
    }


def main() -> None:
    out = {}
    print(
        f"{'p':>2} {'alpha':>7} {'B=L1(Phi)':>16} {'V=sum h^2':>11} "
        f"{'K(alpha)':>10} {'tail':>9}"
    )
    for p in P_ORDERS:
        rows = [shape_constant(p, a, max(400_000, int(80 / a))) for a in ALPHAS]
        for row in rows:
            assert abs(row["dc_gain"] - 1.0) < 1e-9, "unit gain violated"
            assert abs(row["V"] - row["V_lyapunov"]) < 1e-9 * row["V"], (
                "variance disagrees with the Lyapunov solution"
            )
            print(
                f"{p:>2} {row['alpha']:>7.4f} {row['B']:>16.4f} {row['V']:>11.6f} "
                f"{row['K']:>10.4f} {row['tail_frac']:>9.1e}"
            )
        # K(alpha) = K_inf - c*alpha to leading order; Richardson on the last pair.
        k1, k2 = rows[-2]["K"], rows[-1]["K"]
        a1, a2 = rows[-2]["alpha"], rows[-1]["alpha"]
        k_inf = (k2 * a1 - k1 * a2) / (a1 - a2)
        out[str(p)] = {"sweep": rows, "K_limit": k_inf}
        print(f"{'':>2} {'limit':>7} {'':>16} {'':>11} {k_inf:>10.4f}\n")

    with (RESULTS / "bb_shape_constants.json").open("w") as fh:
        json.dump(out, fh, indent=1)
    print(f"Saved {RESULTS / 'bb_shape_constants.json'}")


if __name__ == "__main__":
    main()
