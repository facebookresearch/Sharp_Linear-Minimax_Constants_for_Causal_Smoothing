# Sharp Linear-Minimax Constants for Causal Smoothing

Reproduction code for the paper of the same title by Yonathan Murin and
Ali Özer Ercan (submitted to *IEEE Transactions on Signal Processing*). The
package separates three levels of evidence:

- the all-order filter structure and minimax formula are analytic results;
- the `p=0` constant is closed form and `p=1,...,4` are enclosed by exact
  rational certificates;
- filter shapes, terminal ratios, classical-filter comparisons, and tracking
  experiments are numerical.

The exact certificate does not assume that the optimizer has finitely many
knots. Numerical knot ratios are not certified. The sampled-filter study is
filterwise and is not a computation of the discrete minimax optimum.
The SG limits are evaluated in closed form, and sampled Peano-kernel norms are
integrated exactly on their piecewise-polynomial cells.
Experiments 1 and 4 generate continuous-class signals by exact state
propagation under bounded piecewise-constant top derivatives. The tracking
study discards a warm-up equal to its longest candidate FIR memory.

## Files

```text
Makefile                       reproduction targets (see below)
certificates/
  certify_finite_dual.py       exact rational primal-dual certificate
  certified_intervals.json     tracked machine-readable certificate output
experiments/
  optimal_cascade.py           shared numerical cascade evaluator
  paper_style.py               shared plotting style
  bb_shape_constants.py        BB/BB-type constants and checks
  exp1_monte_carlo.py          risk-law Monte Carlo experiment
  exp2_kp_scaling.py           certified-constant scaling plot
  exp3_filter_comparison.py    BB-family and Savitzky-Golay comparison
  exp4_tracking.py             trajectory-specific tracking experiment
  exp5_delta_sweep.py          sampled-filter finite-rate study
  plot_corrected_theory.py     impulse-response and knot-cascade figures
results/                       tracked reference JSON outputs
figures/                       generated locally by `make figures`; not tracked
tests/                         unit tests for the harness helpers
verify_results.py              tolerant JSON comparator used by `make verify`
figure_index.py                builds figures/index.html for `make figures`
check_solver.py                guards against a superseded finite-knot solver
```

`verify_results.py`, `figure_index.py`, and `check_solver.py` are reproduction
*harness* helpers (pure standard library); they are not part of the paper's
methods.

## Quick start

The `Makefile` wraps every step. From the repository root:

```bash
make setup         # create .venv and install the pinned requirements
make certificate   # verify + byte-check the exact-rational certificate (stdlib only)
make figures       # regenerate all paper figures + a comparison page
make test          # unit tests for the harness helpers (stdlib only)
make verify        # full non-destructive reproduction check
```

The reference outputs were generated with Python 3.12.13, NumPy 2.4.4,
Matplotlib 3.10.9, and SciPy 1.17.1. The certificate uses only the Python
standard library; the numerical experiments need the three packages pinned in
`requirements.txt`. `make setup` installs them into `.venv`, which the other
targets use automatically when present. All targets resolve their inputs and
outputs relative to the package, not the current directory.

Each target is described below, together with the plain commands it runs for
anyone who prefers not to use `make`.

## Exact certificate

```bash
make certificate
```

The certificate is pure Python and needs no third-party packages. The target
first runs the full verification for `p=1,...,4` — primal feasibility, dual
endpoint jets, moment reproduction, stationarity and complementarity defects,
Bernstein upper bounds, and the independent direct and modulus-derived intervals
for `K_p*` (any failed check aborts) — then byte-checks the JSON output against
the tracked certificate. It is read-only (it writes only to `/tmp`). Equivalent
plain commands:

```bash
python3 certificates/certify_finite_dual.py
python3 certificates/certify_finite_dual.py --json > /tmp/certified_intervals.json
diff -u certificates/certified_intervals.json /tmp/certified_intervals.json
```

An empty `diff` confirms the tracked certificate.

## Numerical results and figures

```bash
make figures
```

This regenerates every paper figure into `figures/` (overwriting it, so the
PNGs can be compared side by side with the manuscript PDF) and writes
`figures/index.html`, a one-scroll contact sheet with a caption per figure. The
figure scripts also write `results/*.json` as a side effect; to keep the tracked
`results/` pristine, `make figures` runs the scripts inside a throwaway working
copy and copies only the produced figures back into `figures/`. The paper's
figure *numbers* are not tracked here, so match each figure to the manuscript by
content.

To regenerate the tracked `results/*.json` and the figures together, run the
scripts directly (this writes into `results/` and `figures/` in place):

```bash
python3 -m experiments.bb_shape_constants
python3 -m experiments.exp1_monte_carlo
python3 -m experiments.exp2_kp_scaling
python3 -m experiments.exp3_filter_comparison
python3 -m experiments.exp4_tracking
python3 -m experiments.exp5_delta_sweep
python3 -m experiments.plot_corrected_theory
```

The first six commands regenerate the JSON files in `results/`; PDFs and PNGs
are not tracked because they are deterministic products of the source and
environment.

`exp1_monte_carlo.py` uses seed `0` and 400 trials at 13 noise levels.
`exp4_tracking.py` uses seed `42`, a 360-sample warm-up, 500 evaluated samples,
and 40 paired trials per trajectory. Its left panel is a continuous-limit
benchmark; its trajectory-specific errors are finite-sample in-grid oracle
comparisons, not minimax lower bounds.

## Verification

```bash
make verify
```

`make verify` is a full, non-destructive reproduction check that never modifies
any tracked file. It runs three of the four validation steps below in one
command:

1. `check-solver` — confirms no script imports a superseded finite-knot solver
   (the numerical results must come from the corrected infinite-cascade
   evaluator, `experiments/optimal_cascade.py`);
2. `certificate` — verifies and byte-checks the exact-rational certificate (it
   is platform-independent);
3. regenerates every experiment inside a temporary working copy and compares the
   regenerated `results/*.json` to the tracked ones with `verify_results.py`. The
   comparison is numerical: floats must agree within
   `|a - b| <= atol + rtol * |b|`, while integers, booleans, strings, and
   structure must match exactly.

The remaining step — comparing the generated figures to the manuscript — is
`make figures` (item 3 in the list below).

Most results reproduce cross-platform to a few units in the last place, so the
default tolerance is tight (`rtol=1e-8`, `atol=1e-12`). The one exception is the
p=2 Benedict-Bordner shape constants in `bb_shape_constants.json`: their `B` and
`K` come from a (p+1)-fold cumulative sum of a near-zero-sum error sequence over
a 400k-step impulse response, which is cancellation-heavy and reproduces
cross-platform only to `~1e-4`. That single file therefore gets its own looser
tolerance (`BB_RTOL=1e-4`, `BB_ATOL=1e-9`, the latter also treating the near-zero
`tail_frac` truncation diagnostic as zero) via `verify_results.py --file-tol`;
its extrapolated `K_limit` and every other experiment are held to the tight
default. The exact certificate is the rigorous reference for the constants;
`verify` is a cross-platform reproduction smoke test. Override any tolerance if
desired, e.g. `make verify RTOL=1e-6` or `make verify BB_RTOL=1e-3`. The tracked
`results/` and certificate are only ever read.

The solver guard is also available on its own as `make check-solver`.

The harness helpers have their own unit tests. They exercise the comparator, the
solver guard, and the contact-sheet builder rather than the paper's results:

```bash
make test        # or: PYTHONPATH=. python3 -m unittest discover -b -s tests -t tests
```

Several of these tests deliberately exercise the *failure* paths of the helpers,
so the helpers' own `FAIL`/`MISS` diagnostics are produced on purpose. `-b`
buffers them and replays them only for a test that actually fails, so a passing
run prints just the progress dots and `OK`.

## Paper Mapping

| Paper item | Generator | Reference output |
|---|---|---|
| Certified constants, `p=1,...,4` | `certify_finite_dual.py` | `certified_intervals.json` |
| BB-family constants | `bb_shape_constants.py` | `bb_shape_constants.json` |
| Monte Carlo slopes and ratios | `exp1_monte_carlo.py` | `exp1_results.json` |
| Constant-growth plot | `exp2_kp_scaling.py` | `exp2_results.json` |
| BB-family/SG comparison | `exp3_filter_comparison.py` | `exp3_results.json` |
| Tracking experiment | `exp4_tracking.py` | `exp4_results.json` |
| Sampled-filter convergence | `exp5_delta_sweep.py` | `exp5_results.json` |
| Filter and knot figures | `plot_corrected_theory.py` | generated figures |

## Validation

For a release candidate:

1. run the exact certificate and require an empty `diff` — `make certificate`;
2. rerun all numerical scripts and confirm the tracked JSON files reproduce
   within tolerance — `make verify` (regenerates in a temporary copy, leaving the
   tracked files untouched);
3. compare generated figures to the manuscript figures by rasterizing both at
   the same resolution, since PDF creation metadata need not be byte-identical —
   `make figures` writes `figures/index.html` for this;
4. confirm that no script imports the superseded finite-knot solver —
   `make check-solver` (also run as part of `make verify`).

## Citation

```bibtex
@unpublished{murin_ercan_causal_smoothing,
  author = {Murin, Yonathan and Ercan, Ali {\"O}zer},
  title  = {Sharp Linear-Minimax Constants for Causal Smoothing},
  note   = {Submitted to IEEE Transactions on Signal Processing},
  year   = {2026}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
