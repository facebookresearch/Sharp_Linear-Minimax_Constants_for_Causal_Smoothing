# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the tolerant JSON comparator (reproduction harness)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import verify_results as vr


class CompareScalarsTest(unittest.TestCase):
    def test_equal_ints(self) -> None:
        self.assertEqual(vr.compare(3, 3, rtol=0.0, atol=0.0), [])

    def test_int_mismatch(self) -> None:
        errs = vr.compare(3, 4, rtol=1e-8, atol=1e-12)
        self.assertEqual(len(errs), 1)
        self.assertIn("int", errs[0])

    def test_bool_mismatch(self) -> None:
        errs = vr.compare(True, False, rtol=0.0, atol=0.0)
        self.assertEqual(len(errs), 1)
        self.assertIn("bool", errs[0])

    def test_float_within_tol(self) -> None:
        self.assertEqual(vr.compare(1.0, 1.0 + 5e-4, rtol=1e-3, atol=0.0), [])

    def test_float_beyond_tol(self) -> None:
        errs = vr.compare(1.0, 1.0 + 1e-6, rtol=1e-8, atol=1e-12)
        self.assertEqual(len(errs), 1)
        self.assertIn("delta", errs[0])

    def test_rtol_scales_with_magnitude(self) -> None:
        # Same relative error passes at large magnitude, fails when atol/rtol tight.
        self.assertEqual(vr.compare(1e6, 1e6 + 1.0, rtol=1e-5, atol=0.0), [])
        self.assertEqual(len(vr.compare(1e6, 1e6 + 1.0, rtol=1e-8, atol=0.0)), 1)

    def test_nan_matches_nan(self) -> None:
        self.assertEqual(vr.compare(float("nan"), float("nan"), rtol=0.0, atol=0.0), [])

    def test_nan_vs_number(self) -> None:
        errs = vr.compare(float("nan"), 1.0, rtol=1.0, atol=1.0)
        self.assertEqual(len(errs), 1)
        self.assertIn("NaN", errs[0])

    def test_inf_matches_inf(self) -> None:
        self.assertEqual(vr.compare(float("inf"), float("inf"), rtol=0.0, atol=0.0), [])

    def test_inf_sign_mismatch(self) -> None:
        errs = vr.compare(float("inf"), float("-inf"), rtol=0.0, atol=0.0)
        self.assertEqual(len(errs), 1)
        self.assertIn("inf", errs[0])

    def test_string_equal_and_diff(self) -> None:
        self.assertEqual(vr.compare("a", "a", rtol=0.0, atol=0.0), [])
        self.assertIn("string", vr.compare("a", "b", rtol=0.0, atol=0.0)[0])

    def test_none_handling(self) -> None:
        self.assertEqual(vr.compare(None, None, rtol=0.0, atol=0.0), [])
        self.assertIn("null", vr.compare(None, 1, rtol=0.0, atol=0.0)[0])

    def test_type_mismatch(self) -> None:
        errs = vr.compare(1, "x", rtol=0.0, atol=0.0)
        self.assertEqual(len(errs), 1)
        self.assertIn("type/value mismatch", errs[0])


class CompareContainersTest(unittest.TestCase):
    def test_dict_missing_and_extra_keys(self) -> None:
        errs = vr.compare({"a": 1, "b": 2}, {"a": 1, "c": 3}, rtol=0.0, atol=0.0)
        joined = "\n".join(errs)
        self.assertIn("missing in regenerated", joined)
        self.assertIn("extra key in regenerated", joined)

    def test_dict_nested_value(self) -> None:
        errs = vr.compare({"a": {"b": 1}}, {"a": {"b": 2}}, rtol=0.0, atol=0.0)
        self.assertEqual(len(errs), 1)
        self.assertIn("/a/b", errs[0])

    def test_list_length_mismatch(self) -> None:
        errs = vr.compare([1, 2], [1, 2, 3], rtol=0.0, atol=0.0)
        self.assertEqual(len(errs), 1)
        self.assertIn("list length", errs[0])

    def test_list_elementwise(self) -> None:
        errs = vr.compare([1.0, 2.0], [1.0, 9.0], rtol=1e-9, atol=1e-12)
        self.assertEqual(len(errs), 1)
        self.assertIn("[1]", errs[0])


class ParseFileTolTest(unittest.TestCase):
    def test_valid(self) -> None:
        out = vr.parse_file_tol(["f.json:1e-4:1e-9"])
        self.assertEqual(out, {"f.json": (1e-4, 1e-9)})

    def test_invalid_raises(self) -> None:
        with self.assertRaises(SystemExit):
            vr.parse_file_tol(["missing_atol:1e-4"])


class MainTest(unittest.TestCase):
    def _write(self, d: Path, name: str, obj: object) -> None:
        (d / name).write_text(json.dumps(obj))

    def test_dir_pass(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            ref, new = Path(t) / "ref", Path(t) / "new"
            ref.mkdir()
            new.mkdir()
            self._write(ref, "a.json", {"x": 1.0})
            self._write(new, "a.json", {"x": 1.0 + 1e-10})
            self.assertEqual(vr.main([str(ref), str(new)]), 0)

    def test_dir_fail(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            ref, new = Path(t) / "ref", Path(t) / "new"
            ref.mkdir()
            new.mkdir()
            self._write(ref, "a.json", {"x": 1.0})
            self._write(new, "a.json", {"x": 2.0})
            self.assertEqual(vr.main([str(ref), str(new)]), 1)

    def test_missing_regenerated_file(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            ref, new = Path(t) / "ref", Path(t) / "new"
            ref.mkdir()
            new.mkdir()
            self._write(ref, "a.json", {"x": 1.0})
            self.assertEqual(vr.main([str(ref), str(new)]), 1)

    def test_empty_ref_dir(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            ref, new = Path(t) / "ref", Path(t) / "new"
            ref.mkdir()
            new.mkdir()
            self.assertEqual(vr.main([str(ref), str(new)]), 1)

    def test_per_file_tol_override(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            ref, new = Path(t) / "ref", Path(t) / "new"
            ref.mkdir()
            new.mkdir()
            self._write(ref, "loose.json", {"x": 1.0})
            self._write(new, "loose.json", {"x": 1.0 + 1e-4})
            # Tight default would fail; per-file override passes.
            self.assertEqual(
                vr.main([str(ref), str(new), "--file-tol", "loose.json:1e-3:1e-9"]),
                0,
            )
