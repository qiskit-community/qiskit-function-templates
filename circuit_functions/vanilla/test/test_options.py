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
Circuit Function Template unit tests — mitigation levels and option merging.
"""

from ddt import ddt, data

from circuit_function_entrypoint import run_function
from options.utils import merge_options
from options import Options

from .base_test_case import BaseTemplateTestCase
from .utils import get_inputs, dict_partially_equal


@ddt
class TestOptions(BaseTemplateTestCase):
    """Verifies that mitigation level presets produce the correct option structure
    and that user overwrites are applied without disturbing unrelated fields."""

    @data(1, 2, 3)
    def test_mitigation_level(self, mit_level):
        """apply_mitigation_level returns the documented preset for each level,
        and run_function completes successfully with that level."""
        options = {"mitigation_level": mit_level}
        result = run_function(
            **get_inputs(backend_name=self._backend_name, options=options),
            testing_backend=self._testing_backend,
        )
        applied = Options.apply_mitigation_level(mit_level)

        self.assertIn("hw_results", result)
        self.assertTrue(applied["dynamical_decoupling"]["enable"])
        self.assertTrue(applied["twirling"]["enable_measure"])
        if mit_level >= 2:
            self.assertTrue(applied["twirling"]["enable_gates"])
        if mit_level >= 3:
            self.assertEqual(applied["resilience"]["zne"]["amplifier"], "pea")

    @data(
        # Overwrite a deeply-nested leaf — other sub-option fields must survive
        {"twirling": {"enable_gates": False}},
        # Overwrite another nested field — ZNE stays enabled
        {"resilience": {"zne": {"amplifier": "gate_folding"}}},
        # Disable ZNE mitigation entirely
        {"resilience": {"zne_mitigation": False}},
        # Overwrite a direct sub-option field — DD enable must survive
        {"dynamical_decoupling": {"sequence_type": "XY4"}},
    )
    def test_mitigation_overwrite(self, overwrite):
        """merge_options applies the user overwrite while preserving unrelated fields."""
        base = Options.apply_mitigation_level(3)
        merged = merge_options(base, overwrite)

        self.assertTrue(dict_partially_equal(merged, overwrite))
        self.assertTrue(merged["dynamical_decoupling"]["enable"])
