# Changelog

All notable changes to this project are documented in this file.

## 1.0.0

Initial public release, accompanying the paper ``Sharp Linear-Minimax Constants
for Causal Smoothing'' by Yonathan Murin and Ali Ozer Ercan.

### Added

- `certificates/` -- the exact rational primal-dual certificate enclosing the
  sharp constant `K_p*` for `p = 1, ..., 4`, and its tracked output
  `certified_intervals.json`.
- `experiments/` -- the five numerical experiments reported in the paper, the
  shared infinite-cascade evaluator, and the shared plotting style.
- `results/` -- tracked reference JSON for every experiment.
- `tests/` -- unit tests for the reproduction-harness helpers.
- `Makefile` -- a single entry point: `setup`, `certificate`, `figures`, `test`,
  `check-solver`, `verify`, `clean`.
- Reproduction harness: `verify_results.py` (tolerant JSON comparator),
  `check_solver.py` (guard against a superseded finite-knot solver), and
  `figure_index.py` (figure contact sheet).
