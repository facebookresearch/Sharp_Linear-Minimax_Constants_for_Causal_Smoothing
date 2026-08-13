# Convenience targets for the reproduction package.
# Everything runs from the repository root.

PYTHON ?= python3
VENV   ?= .venv
VPY    := $(VENV)/bin/python

# Every experiment module. exp1-exp5 and bb_shape_constants regenerate a
# results/*.json; exp1-exp5 and plot_corrected_theory also write a figure.
EXPERIMENTS := \
	bb_shape_constants \
	exp1_monte_carlo \
	exp2_kp_scaling \
	exp3_filter_comparison \
	exp4_tracking \
	exp5_delta_sweep \
	plot_corrected_theory

# The subset that produces the paper figures (bb_shape_constants has none).
FIGURE_MODULES := \
	exp1_monte_carlo \
	exp2_kp_scaling \
	exp3_filter_comparison \
	exp4_tracking \
	exp5_delta_sweep \
	plot_corrected_theory

.PHONY: setup certificate figures check-solver test verify clean

# Create .venv and install the pinned numerical dependencies.
setup:
	$(PYTHON) -m venv $(VENV)
	$(VPY) -m pip install --upgrade pip
	$(VPY) -m pip install -r requirements.txt

# --- Exact certificate (README "Exact Certificate") -------------------------
# Pure stdlib; needs no venv. First the full human-readable verification for
# p=1..4 -- primal feasibility, dual endpoint jets, moment reproduction,
# stationarity and complementarity defects, Bernstein upper bounds, and the
# independent direct and modulus-derived K_p* intervals; any failed check
# raises and aborts. Then the JSON output is byte-checked against the tracked
# certificates/certified_intervals.json. Read-only: writes only to /tmp.
certificate:
	$(PYTHON) certificates/certify_finite_dual.py
	$(PYTHON) certificates/certify_finite_dual.py --json > /tmp/certified_intervals.json
	diff -u certificates/certified_intervals.json /tmp/certified_intervals.json
	@echo "certificate verified and reproduced byte-for-byte"

# --- Figures (README "Validation" item 3) -----------------------------------
# Regenerate every paper figure into figures/ (overwriting it -- intended, so
# the PNGs can be compared side by side with the manuscript PDF), then build
# figures/index.html, a one-scroll contact sheet with a caption per figure.
# The figure scripts write results/*.json as a side effect, so to keep the
# tracked results/ pristine they run inside a throwaway working copy and only
# the produced figures are copied back into ./figures/. Needs the numerical
# dependencies (run `make setup` first).
figures:
	@RUN=$$( [ -x "$(VPY)" ] && echo "$(abspath $(VPY))" || echo "$(PYTHON)" ); \
		work=$$(mktemp -d); \
		cp -R certificates experiments results "$$work"/; \
		if ! ( cd "$$work" && for m in $(FIGURE_MODULES); do \
				echo ">> $$m"; $$RUN -m experiments.$$m || exit 1; \
			done ); then \
			rm -rf "$$work"; echo "figure generation failed -- see errors above"; exit 1; \
		fi; \
		mkdir -p figures; cp -f "$$work"/figures/* figures/; \
		rm -rf "$$work"; \
		$(PYTHON) figure_index.py; \
		echo "figures written to figures/ (tracked results/ untouched); open figures/index.html"

# --- No superseded finite-knot solver (README "Validation" item 4) ----------
# Confirms the numerical experiments import only the corrected infinite-cascade
# evaluator (experiments/optimal_cascade.py), never a finite-knot solver. Pure
# stdlib.
check-solver:
	$(PYTHON) check_solver.py

# --- Harness unit tests -----------------------------------------------------
# Unit tests for the three reproduction-harness helpers (verify_results,
# check_solver, figure_index). Pure stdlib; needs no venv. These cover the
# harness itself, not the paper's methods.
#
# -b buffers stdout during each test. Several tests deliberately drive the
# failure paths of the harness helpers, which print their own "FAIL"/"MISS"
# diagnostics; without buffering those land in the console and look like test
# failures even though the run passes. With -b they are captured and replayed
# only for a test that actually fails.
test:
	PYTHONPATH=. $(PYTHON) -m unittest discover -b -s tests -t tests

# --- Full non-destructive reproduction check --------------------------------
# Covers README "Validation" items 1, 2 and 4 in one command, without modifying
# any tracked file (item 3, the visual figure comparison, is `make figures`):
#   4. check-solver -- no finite-knot solver is imported;
#   1. certificate  -- the exact-rational certificate is verified and byte-checked;
#   2. experiments  -- every result is regenerated in a throwaway working copy
#      and compared to the tracked results/ within a numerical tolerance.
# The tracked results/ and certificate are only ever read.
#
# Numerical tolerances for step 2. Override on the command line, e.g.
# `make verify RTOL=1e-6`. Most results reproduce cross-platform to a few ULPs,
# so the default is tight. The one exception is the p=2 Benedict-Bordner shape
# constants: their (p+1)-fold cumulative sum over a 400k-step impulse response is
# cancellation-heavy, so B/K in bb_shape_constants.json reproduce only to ~1e-4
# (the extrapolated limits and the exact certificate remain the rigorous
# references). That file gets its own looser tolerance below; BB_ATOL=1e-9 also
# treats the near-zero tail_frac truncation diagnostic as zero.
RTOL    ?= 1e-8
ATOL    ?= 1e-12
BB_RTOL ?= 1e-4
BB_ATOL ?= 1e-9

verify: check-solver
	@$(MAKE) --no-print-directory certificate
	@RUN=$$( [ -x "$(VPY)" ] && echo "$(abspath $(VPY))" || echo "$(PYTHON)" ); \
		work=$$(mktemp -d); \
		cp -R certificates experiments results "$$work"/; \
		if ! ( cd "$$work" && for m in $(EXPERIMENTS); do \
				echo ">> $$m"; $$RUN -m experiments.$$m || exit 1; \
			done ); then \
			rm -rf "$$work"; echo "experiments failed -- see errors above"; exit 1; \
		fi; \
		echo "== comparing regenerated results to tracked results/ =="; \
		if $$RUN verify_results.py results "$$work/results" \
				--rtol $(RTOL) --atol $(ATOL) \
				--file-tol bb_shape_constants.json:$(BB_RTOL):$(BB_ATOL); then \
			echo "reproduced within tolerance (tracked files untouched)"; \
			rm -rf "$$work"; \
		else \
			echo "results differ beyond tolerance (tracked files were NOT modified)."; \
			rm -rf "$$work"; exit 1; \
		fi

clean:
	rm -rf figures $(VENV)
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
