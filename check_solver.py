# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Confirm no script imports a superseded finite-knot solver.

Part of the reproduction *harness* (invoked by ``make verify`` and
``make check-solver``); it is README "Validation" item 4. The numerical results
must come from the corrected infinite-cascade evaluator
(``experiments/optimal_cascade.py``); an earlier finite-knot spline solver was
superseded and must not be reintroduced.

The check is structural, not name-based: it parses every module under
``experiments/`` and ``certificates/`` and asserts that the only *in-repo*
modules they import are on a small allow-list (the cascade evaluator and the
shared plotting style). Any other local import -- e.g. a reintroduced solver --
fails the check, and the corrected evaluator is required to actually be in use.
Pure standard library.

Exit status is 0 when the invariant holds, 1 otherwise.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGES = ("experiments", "certificates")

# In-repo modules the reproduction code is allowed to import from.
ALLOWED_LOCAL = {"optimal_cascade", "paper_style"}
# The corrected numerical evaluator that must be the one in use.
REQUIRED_EVALUATOR = "optimal_cascade"


def local_imports(tree: ast.AST) -> set[str]:
    """Top-level module names of every relative (in-repo) import in ``tree``."""
    found: set[str] = set()
    for node in ast.walk(tree):
        # Relative imports: `from . import paper_style`,
        # `from .optimal_cascade import X`, `from .pkg.sub import Y`.
        if isinstance(node, ast.ImportFrom) and node.level and node.level > 0:
            if node.module:
                found.add(node.module.split(".")[0])
            else:
                found.update(alias.name.split(".")[0] for alias in node.names)
    return found


def main() -> int:
    violations: list[str] = []
    evaluator_used = False

    for package in PACKAGES:
        pkg_dir = ROOT / package
        if not pkg_dir.is_dir():
            print(f"FAIL  expected package directory is missing: {package}/")
            return 1
        for path in sorted(pkg_dir.glob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for module in sorted(local_imports(tree)):
                if module == REQUIRED_EVALUATOR:
                    evaluator_used = True
                if module not in ALLOWED_LOCAL:
                    rel = path.relative_to(ROOT)
                    violations.append(
                        f"{rel} imports disallowed in-repo module '{module}'"
                    )

    if violations:
        print("FAIL  a superseded/unexpected solver appears to be imported:")
        for line in violations:
            print(f"        {line}")
        print(f"      allowed in-repo modules: {sorted(ALLOWED_LOCAL)}")
        return 1

    if not evaluator_used:
        print(
            f"FAIL  no experiment imports the corrected evaluator "
            f"'{REQUIRED_EVALUATOR}' -- the numerical results are not being "
            f"produced by the infinite-cascade solver."
        )
        return 1

    print(
        f"ok    no finite-knot solver imported; numerical results use "
        f"experiments/{REQUIRED_EVALUATOR}.py"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
