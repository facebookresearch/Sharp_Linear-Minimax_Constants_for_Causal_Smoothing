# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the finite-knot solver guard (reproduction harness)."""

from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import check_solver as cs


class LocalImportsTest(unittest.TestCase):
    def test_relative_from_import(self) -> None:
        tree = ast.parse("from .optimal_cascade import g\n")
        self.assertEqual(cs.local_imports(tree), {"optimal_cascade"})

    def test_relative_bare_import(self) -> None:
        tree = ast.parse("from . import paper_style\n")
        self.assertEqual(cs.local_imports(tree), {"paper_style"})

    def test_relative_subpackage_keeps_top_level_only(self) -> None:
        tree = ast.parse("from .pkg.sub import Y\n")
        self.assertEqual(cs.local_imports(tree), {"pkg"})

    def test_absolute_imports_ignored(self) -> None:
        tree = ast.parse("import os\nfrom numpy import array\n")
        self.assertEqual(cs.local_imports(tree), set())


class MainTest(unittest.TestCase):
    def _make_tree(self, root: Path, files: dict[str, str]) -> None:
        for rel, body in files.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body)

    def test_allowed_imports_pass(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            self._make_tree(
                root,
                {
                    "experiments/optimal_cascade.py": "def g() -> int:\n    return 1\n",
                    "experiments/exp1.py": (
                        "from .optimal_cascade import g\nfrom . import paper_style\n"
                    ),
                    "experiments/paper_style.py": "STYLE = {}\n",
                    "certificates/cert.py": "import json\n",
                },
            )
            with patch.object(cs, "ROOT", root):
                self.assertEqual(cs.main(), 0)

    def test_disallowed_solver_import_fails(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            self._make_tree(
                root,
                {
                    "experiments/optimal_cascade.py": "def g() -> int:\n    return 1\n",
                    "experiments/exp1.py": (
                        "from .optimal_cascade import g\nfrom .spline_solver import S\n"
                    ),
                    "certificates/cert.py": "import json\n",
                },
            )
            with patch.object(cs, "ROOT", root):
                self.assertEqual(cs.main(), 1)

    def test_evaluator_absent_fails(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            self._make_tree(
                root,
                {
                    "experiments/exp1.py": "from . import paper_style\n",
                    "experiments/paper_style.py": "STYLE = {}\n",
                    "certificates/cert.py": "import json\n",
                },
            )
            with patch.object(cs, "ROOT", root):
                self.assertEqual(cs.main(), 1)

    def test_missing_package_dir_fails(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            # experiments/ exists but certificates/ (in PACKAGES) is absent.
            self._make_tree(
                root,
                {"experiments/optimal_cascade.py": "def g() -> int:\n    return 1\n"},
            )
            with patch.object(cs, "ROOT", root):
                self.assertEqual(cs.main(), 1)
