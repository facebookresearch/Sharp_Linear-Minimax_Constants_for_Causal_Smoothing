# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Numerical evaluators for the corrected optimal causal filters.

The sharp constants in this module are certified.  The filter shapes for
orders one and two are numerical cascade approximations.  Their infinitely
many switching intervals are truncated only after the omitted support is far
below double precision.
"""

from math import comb, factorial

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


K_STAR = {
    0: 1.100642416298,
    1: 1.7451509858523378,
    2: 2.4078312318151667,
    3: 3.0725144355627184,
    4: 3.7360360214557904,
}

TAIL_RATIO = {
    1: 0.24212137354815673,
    2: 0.5757361184208955,
}

# Ratios h_{j+1}/h_j on the connecting part of the p=2 orbit.
P2_RATIOS = np.array(
    [
        0.3485194720333346,
        0.6137445619288796,
        0.5677655229714218,
        0.5774287724521258,
        0.5753773890574188,
        0.5758121800671684,
        0.5757200358,
        0.5757397,
    ]
)


def _absolute_polynomial_integral(coefficients: FloatArray) -> float:
    """Integrate the absolute value of a polynomial over the unit interval."""
    roots = np.polynomial.polynomial.polyroots(coefficients)
    interior_roots = sorted(
        float(root.real)
        for root in roots
        if abs(float(root.imag)) < 1e-10 and 0.0 < float(root.real) < 1.0
    )
    antiderivative = np.zeros(len(coefficients) + 1)
    antiderivative[1:] = coefficients / np.arange(1, len(coefficients) + 1)
    endpoints = [0.0, *interior_roots, 1.0]
    return sum(
        abs(
            float(np.polynomial.polynomial.polyval(right, antiderivative))
            - float(np.polynomial.polynomial.polyval(left, antiderivative))
        )
        for left, right in zip(endpoints[:-1], endpoints[1:])
    )


def peano_l1_from_taps(taps: FloatArray, p: int) -> float:
    """Return the exact piecewise-polynomial Peano-kernel L1 norm.

    Tap ``j`` is located at lag ``j``.  On the open cell ``(i, i+1)``, only
    taps with ``j > i`` contribute, and the kernel is a degree-``p``
    polynomial.  Integrating between its real roots avoids a grid quadrature.
    """
    if p < 0:
        raise ValueError("p must be nonnegative")
    h = np.asarray(taps, dtype=float)
    total = 0.0
    for cell in range(len(h) - 1):
        distances = np.arange(1, len(h) - cell, dtype=float)
        future_taps = h[cell + 1 :]
        coefficients = np.array(
            [
                (-1.0) ** power
                * np.dot(future_taps, distances ** (p - power))
                / (factorial(power) * factorial(p - power))
                for power in range(p + 1)
            ]
        )
        total += _absolute_polynomial_integral(coefficients)
    return total


def shape_constant_from_norms(p: int, peano_l1: float, variance: float) -> float:
    """Return the scale-invariant shape constant from its two norms."""
    r = p + 1
    exponent_denominator = 2 * r + 1
    return (
        exponent_denominator
        * (2 * r) ** (-2 * r / exponent_denominator)
        * peano_l1 ** (2 / exponent_denominator)
        * variance ** (2 * r / exponent_denominator)
    )


def shape_constant_from_taps(taps: FloatArray, p: int) -> float:
    """Return the sampled-filter constant using the exact Peano L1 norm."""
    h = np.asarray(taps, dtype=float)
    return shape_constant_from_norms(
        p,
        peano_l1_from_taps(h, p),
        float(np.dot(h, h)),
    )


def savitzky_golay_continuous_constant(p: int) -> float:
    """Return the closed-form endpoint SG constant on a unit window."""
    peano_l1 = factorial(p + 1) / factorial(2 * p + 2)
    variance = float((p + 1) ** 2)
    return shape_constant_from_norms(p, peano_l1, variance)


def integrate_piecewise_constant_top_derivative(
    top_derivative: FloatArray, dt: float, order: int
) -> FloatArray:
    """Sample an order-``order`` integral with a zero initial derivative jet.

    Entry ``j`` of ``top_derivative`` is the constant value of the top
    derivative on interval ``[j * dt, (j + 1) * dt)``.  Exact Taylor state
    propagation returns one more signal sample than there are intervals.
    """
    if order < 1:
        raise ValueError("order must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")

    transition = np.zeros((order, order))
    for row in range(order):
        for column in range(row, order):
            transition[row, column] = dt ** (column - row) / factorial(column - row)
    input_gain = np.array(
        [dt ** (order - row) / factorial(order - row) for row in range(order)]
    )

    state = np.zeros(order)
    samples = np.empty(len(top_derivative) + 1)
    samples[0] = 0.0
    for interval, value in enumerate(np.asarray(top_derivative, dtype=float)):
        state = transition @ state + input_gain * value
        samples[interval + 1] = state[0]
    return samples


def interval_lengths(p: int, cells: int = 100) -> FloatArray:
    """Return positive switching lengths normalized to unit total support."""
    if p == 0:
        return np.array([1.0])
    rho = TAIL_RATIO[p]
    ratios = np.full(cells - 1, rho)
    if p == 2:
        n = min(len(P2_RATIOS), len(ratios))
        ratios[:n] = P2_RATIOS[:n]
    lengths = np.ones(cells)
    for j in range(1, cells):
        lengths[j] = lengths[j - 1] * ratios[j - 1]
    # Add the unrepresented geometric remainder to the last cell.  This keeps
    # the support exactly one without moving any visible early switch.
    tail = lengths[-1] * rho / (1.0 - rho) if p else 0.0
    lengths[-1] += tail
    return lengths / lengths.sum()


def _backward_states(p: int, lengths: FloatArray) -> list[tuple[FloatArray, float]]:
    """Build local Taylor states for f^(p+1)=+1,-1,+1,... and a flat end."""
    r = p + 1
    states: list[tuple[FloatArray, float]] = []
    right = np.zeros(r)
    for j in range(len(lengths) - 1, -1, -1):
        h = lengths[j]
        control = (-1.0) ** j
        left = np.empty(r)
        for k in range(r):
            left[k] = sum(
                right[m] * (-h) ** (m - k) / factorial(m - k) for m in range(k, r)
            )
            left[k] += control * (-h) ** (r - k) / factorial(r - k)
        states.insert(0, (left, control))
        right = left
    return states


class CascadeFilter:
    """Piecewise-polynomial approximation of the unit-support optimal shape."""

    def __init__(self, p: int, cells: int = 100) -> None:
        if p not in {0, 1, 2}:
            raise ValueError("shape evaluator currently supports p=0,1,2")
        self.p = p
        self.r = p + 1
        self.lengths = interval_lengths(p, cells)
        self.breaks = np.concatenate(([0.0], np.cumsum(self.lengths)))
        self.states = _backward_states(p, self.lengths)
        # A global sign is immaterial.  Normalize the filter mass to one.
        mass = self.integral_raw(0.0, 1.0)
        self.scale = 1.0 / mass

    def raw(self, x: FloatArray) -> FloatArray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        inside = (x >= 0.0) & (x <= 1.0)
        ids = np.searchsorted(self.breaks[1:], x[inside], side="left")
        ids = np.minimum(ids, len(self.lengths) - 1)
        vals = np.empty(np.count_nonzero(inside))
        xx = x[inside]
        for j in np.unique(ids):
            mask = ids == j
            u = xx[mask] - self.breaks[j]
            state, control = self.states[j]
            value = np.zeros_like(u)
            for k in range(self.r):
                value += state[k] * u**k / factorial(k)
            value += control * u**self.r / factorial(self.r)
            vals[mask] = value
        out[inside] = vals
        return out

    def __call__(self, x: FloatArray) -> FloatArray:
        return self.scale * self.raw(x)

    def integral_raw(self, lo: float, hi: float, order: int = 24) -> float:
        nodes, weights = np.polynomial.legendre.leggauss(order)
        total = 0.0
        for a, b in zip(self.breaks[:-1], self.breaks[1:]):
            left, right = max(lo, a), min(hi, b)
            if right <= left:
                continue
            x = (right - left) * nodes / 2.0 + (right + left) / 2.0
            total += (right - left) * np.dot(weights, self.raw(x)) / 2.0
        return total

    def cell_taps(self, cells: int, correct_moments: bool = True) -> FloatArray:
        """Integrate over cells, then impose exact discrete moments stably."""
        nodes, weights = np.polynomial.legendre.leggauss(12)
        edges = np.linspace(0.0, 1.0, cells + 1)
        taps = np.empty(cells)
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            x = (b - a) * nodes / 2.0 + (a + b) / 2.0
            taps[i] = (b - a) * np.dot(weights, self(x)) / 2.0
        if correct_moments:
            # Scale nodes to [0,1] to avoid a collapsing or badly scaled
            # Vandermonde system.  The correction is the minimum-norm one.
            t = np.arange(cells, dtype=float) / cells
            A = np.vstack([t**m for m in range(self.p + 1)])
            target = np.zeros(self.p + 1)
            target[0] = 1.0
            taps += A.T @ np.linalg.solve(A @ A.T, target - A @ taps)
        return taps

    def grid(self, n: int = 4001) -> tuple[FloatArray, FloatArray]:
        x = np.linspace(0.0, 1.0, n)
        return x, self(x)


def peano_from_grid(x: FloatArray, g: FloatArray, p: int) -> FloatArray:
    """Evaluate the Peano kernel by reverse cumulative trapezoid sums."""
    from scipy.integrate import cumulative_trapezoid

    phi = np.zeros_like(x)
    for ell in range(p + 1):
        integrand = g * x**ell
        tail = np.trapezoid(integrand, x) - cumulative_trapezoid(
            integrand, x, initial=0.0
        )
        phi += comb(p, ell) * (-x) ** (p - ell) * tail
    return phi / factorial(p)
