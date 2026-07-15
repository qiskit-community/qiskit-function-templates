# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Executor Function Template unit tests: mitigation levels, option merging and validation.
"""

import unittest

from ddt import ddt, data
from pydantic import ValidationError

from options import Options, MAX_MITIGATION_LEVEL
from options.utils import merge_options

from .utils import dict_partially_equal


@ddt
class TestMitigationLevels(unittest.TestCase):
    """apply_mitigation_level produces the documented presets and caps at level 2."""

    def test_level_1_is_dd_only(self):
        """Level 1 enables dynamical decoupling and no twirling (keeps the default path local)."""
        applied = Options.apply_mitigation_level(1)
        self.assertTrue(applied["dynamical_decoupling"]["enable"])
        self.assertNotIn("twirling", applied)

    def test_level_2_adds_twirling(self):
        """Level 2 additionally enables gate and measurement twirling."""
        applied = Options.apply_mitigation_level(2)
        self.assertTrue(applied["dynamical_decoupling"]["enable"])
        self.assertTrue(applied["twirling"]["enable_gates"])
        self.assertTrue(applied["twirling"]["enable_measure"])

    def test_no_resilience_key(self):
        """Unlike the estimator template, there is no ZNE/PEC resilience preset."""
        applied = Options.apply_mitigation_level(MAX_MITIGATION_LEVEL)
        self.assertNotIn("resilience", applied)

    @data(0, 3, -1, 99)
    def test_invalid_level_raises(self, level):
        """Mitigation levels outside 1-2 are rejected."""
        with self.assertRaises(ValueError):
            Options.apply_mitigation_level(level)


@ddt
class TestOptionMerging(unittest.TestCase):
    """User options override mitigation-level defaults without disturbing siblings."""

    @data(
        {"twirling": {"enable_gates": False}},
        {"dynamical_decoupling": {"sequence_type": "XY4"}},
    )
    def test_overwrite_preserves_siblings(self, overwrite):
        """A nested override wins while unrelated fields survive."""
        base = Options.apply_mitigation_level(2)
        combined = merge_options(base, overwrite)
        self.assertTrue(dict_partially_equal(combined, overwrite))
        # DD stays enabled regardless of the twirling/DD sub-field override.
        self.assertTrue(combined["dynamical_decoupling"]["enable"])


class TestOptionExtraction(unittest.TestCase):
    """The transpilation and execution option splitters behave as documented."""

    def test_split_transpilation_and_execution(self):
        """optimization_level goes to transpilation; execution keeps the executor options."""
        parsed = Options(
            optimization_level=1,
            default_shots=512,
            dynamical_decoupling={"enable": True},
        ).model_dump(exclude_unset=True)

        transpilation = Options.get_transpilation_options(parsed)
        execution = Options.get_execution_options(parsed)

        self.assertEqual(transpilation, {"optimization_level": 1})
        self.assertEqual(execution["default_shots"], 512)
        self.assertNotIn("optimization_level", execution)
        self.assertNotIn("mitigation_level", execution)


class TestOptionValidation(unittest.TestCase):
    """The pydantic model rejects out-of-range and unknown fields."""

    def test_negative_shots_rejected(self):
        """default_shots must be positive."""
        with self.assertRaises(ValidationError):
            Options(default_shots=-10)

    def test_unknown_field_rejected(self):
        """Unknown option keys are forbidden by the strict model."""
        with self.assertRaises(ValidationError):
            Options(not_a_real_option=True)

    def test_optimization_level_range(self):
        """optimization_level is bounded to the transpiler's 0-3 range."""
        with self.assertRaises(ValidationError):
            Options(optimization_level=5)


if __name__ == "__main__":
    unittest.main()
