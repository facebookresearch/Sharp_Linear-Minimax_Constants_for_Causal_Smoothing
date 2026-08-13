## Summary

What does this change do, and why?

## Effect on published results

Does this change any number or figure in the paper?

- [ ] No. `make verify` still passes against the tracked `results/`.
- [ ] Yes. Explain which values change and why, and update `results/` in the
      same pull request.

## Test plan

Please paste the output of the checks you ran:

- [ ] `make test` -- unit tests for the harness helpers
- [ ] `make check-solver` -- no superseded solver is imported
- [ ] `make certificate` -- the exact certificate reproduces byte-for-byte
- [ ] `make verify` -- full non-destructive reproduction check
