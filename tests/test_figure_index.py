# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the figure contact-sheet builder (reproduction harness)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import figure_index as fi


class BuildHtmlTest(unittest.TestCase):
    def test_missing_figures_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            figs = Path(t) / "figures"
            figs.mkdir()
            first = fi.FIGURES_SPEC[0][0]
            (figs / first).write_bytes(b"\x89PNG")
            with patch.object(fi, "FIGURES", figs):
                index, missing = fi.build_html()
            self.assertTrue(index.exists())
            self.assertNotIn(first, missing)
            self.assertEqual(len(missing), len(fi.FIGURES_SPEC) - 1)
            html = index.read_text()
            self.assertIn(f'<img src="{first}"', html)
            self.assertIn("figure not found", html)

    def test_all_present_reports_no_missing(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            figs = Path(t) / "figures"
            figs.mkdir()
            for fname, _, _ in fi.FIGURES_SPEC:
                (figs / fname).write_bytes(b"\x89PNG")
            with patch.object(fi, "FIGURES", figs):
                index, missing = fi.build_html()
            self.assertEqual(missing, [])
            self.assertNotIn("figure not found", index.read_text())

    def test_main_nonzero_when_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            figs = Path(t) / "figures"
            figs.mkdir()
            with patch.object(fi, "FIGURES", figs):
                self.assertEqual(fi.main(), 1)

    def test_main_zero_when_complete(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            figs = Path(t) / "figures"
            figs.mkdir()
            for fname, _, _ in fi.FIGURES_SPEC:
                (figs / fname).write_bytes(b"\x89PNG")
            with patch.object(fi, "FIGURES", figs):
                self.assertEqual(fi.main(), 0)
