# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tolerant comparison of regenerated results against the tracked reference.

Part of the reproduction *harness* (invoked by ``make verify``); it is not part
of the paper's methods. The experiment scripts are seeded, so the pseudo-random
draws are bit-identical across platforms for a given NumPy version. Their
floating-point reductions still drift across platforms/BLAS builds: most
quantities to a few units in the last place, but the p=2 shape constants more
noticeably (their (p+1)-fold cumulative sum over a 400k-step impulse response is
cancellation-heavy, so B/K there reproduce only to ~1e-4). This comparator
therefore checks numbers numerically:

  * floats agree within ``|a - b| <= atol + rtol * |b|`` (absorbs that drift;
    the default atol also treats the near-zero tail_frac diagnostic as zero);
  * ints, bools, strings, ``null``, dict keys, and list lengths must match
    exactly (a change there signals a real difference, not round-off).

Exit status is 0 when everything is within tolerance, 1 otherwise. It only
*reads* both sides -- it never writes to either tree.

A tight default tolerance applies to every file; individual files can be given
a looser (or tighter) tolerance with ``--file-tol NAME:RTOL:ATOL`` (repeatable).
The p=2 shape constants in ``bb_shape_constants.json`` are the one file that
needs loosening -- see the Makefile.

Usage:
    python verify_results.py REF NEW [--rtol R] [--atol A]
        [--file-tol NAME:RTOL:ATOL ...] [--max-report N]

REF and NEW may be two directories (every ``*.json`` in REF is compared to the
same-named file in NEW) or two individual JSON files.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def compare(ref: Any, new: Any, rtol: float, atol: float, path: str = "") -> list[str]:
    """Recursively compare two decoded-JSON values; return a list of mismatches."""
    # bool is a subclass of int -- check it before the numeric branches.
    if isinstance(ref, bool) or isinstance(new, bool):
        if ref != new:
            return [f"{path or '<root>'}: bool {ref!r} != {new!r}"]
        return []

    if isinstance(ref, int) and isinstance(new, int):
        if ref != new:
            return [f"{path or '<root>'}: int {ref} != {new}"]
        return []

    if isinstance(ref, (int, float)) and isinstance(new, (int, float)):
        a, b = float(ref), float(new)
        if math.isnan(a) or math.isnan(b):
            if not (math.isnan(a) and math.isnan(b)):
                return [f"{path or '<root>'}: NaN mismatch {ref} vs {new}"]
            return []
        if math.isinf(a) or math.isinf(b):
            if a != b:
                return [f"{path or '<root>'}: inf mismatch {ref} vs {new}"]
            return []
        diff = abs(a - b)
        tol = atol + rtol * abs(b)
        if diff > tol:
            rel = diff / abs(b) if b else math.inf
            return [
                f"{path or '<root>'}: {a!r} vs {b!r} "
                f"(|delta|={diff:.3e} > tol={tol:.3e}, rel={rel:.3e})"
            ]
        return []

    if isinstance(ref, dict) and isinstance(new, dict):
        errors: list[str] = []
        for k in ref.keys() - new.keys():
            errors.append(
                f"{path}/{k}: key present in reference, missing in regenerated"
            )
        for k in new.keys() - ref.keys():
            errors.append(f"{path}/{k}: extra key in regenerated")
        for k in sorted(ref.keys() & new.keys(), key=str):
            errors += compare(ref[k], new[k], rtol, atol, f"{path}/{k}")
        return errors

    if isinstance(ref, list) and isinstance(new, list):
        if len(ref) != len(new):
            return [f"{path or '<root>'}: list length {len(ref)} != {len(new)}"]
        errors = []
        for i, (a, b) in enumerate(zip(ref, new)):
            errors += compare(a, b, rtol, atol, f"{path}[{i}]")
        return errors

    if ref is None or new is None:
        if ref is not new:
            return [f"{path or '<root>'}: null mismatch {ref!r} vs {new!r}"]
        return []

    if isinstance(ref, str) and isinstance(new, str):
        if ref != new:
            return [f"{path or '<root>'}: string differs"]
        return []

    return [
        f"{path or '<root>'}: type/value mismatch "
        f"({type(ref).__name__} {ref!r} vs {type(new).__name__} {new!r})"
    ]


def load(p: Path) -> Any:
    with open(p) as fh:
        return json.load(fh)


def parse_file_tol(specs: list[str]) -> dict[str, tuple[float, float]]:
    """Parse ``NAME:RTOL:ATOL`` overrides into ``{name: (rtol, atol)}``."""
    out: dict[str, tuple[float, float]] = {}
    for spec in specs:
        parts = spec.split(":")
        if len(parts) != 3:
            raise SystemExit(f"--file-tol must be NAME:RTOL:ATOL, got {spec!r}")
        name, rtol, atol = parts
        out[name] = (float(rtol), float(atol))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Tolerant JSON result comparison.")
    ap.add_argument("ref", help="reference file or directory (tracked results)")
    ap.add_argument("new", help="regenerated file or directory")
    ap.add_argument(
        "--rtol",
        type=float,
        default=1e-8,
        help="default relative tolerance (default 1e-8)",
    )
    ap.add_argument(
        "--atol",
        type=float,
        default=1e-12,
        help="default absolute tolerance (default 1e-12)",
    )
    ap.add_argument(
        "--file-tol",
        action="append",
        default=[],
        metavar="NAME:RTOL:ATOL",
        help="per-file tolerance override, by basename (repeatable)",
    )
    ap.add_argument(
        "--max-report", type=int, default=10, help="max mismatches shown per file"
    )
    args = ap.parse_args(argv)

    file_tol = parse_file_tol(args.file_tol)

    ref, new = Path(args.ref), Path(args.new)
    if ref.is_dir():
        names = sorted(p.name for p in ref.glob("*.json"))
        if not names:
            print(f"no *.json files found in {ref}", file=sys.stderr)
            return 1
        pairs = [(name, ref / name, new / name) for name in names]
    else:
        pairs = [(ref.name, ref, new)]

    failed = 0
    for label, rp, np_ in pairs:
        rtol, atol = file_tol.get(label, (args.rtol, args.atol))
        tol_note = f"rtol={rtol:g}, atol={atol:g}"
        if not np_.exists():
            print(f"FAIL  {label}: regenerated file missing ({np_})")
            failed += 1
            continue
        errs = compare(load(rp), load(np_), rtol, atol)
        if errs:
            failed += 1
            print(
                f"FAIL  {label} ({tol_note}): {len(errs)} mismatch(es) beyond tolerance"
            )
            for line in errs[: args.max_report]:
                print(f"        {line}")
            if len(errs) > args.max_report:
                print(f"        ... and {len(errs) - args.max_report} more")
        else:
            print(f"OK    {label} ({tol_note})")

    if failed:
        print(f"\n{failed} file(s) differ beyond tolerance.")
        return 1
    print("\nall results within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
