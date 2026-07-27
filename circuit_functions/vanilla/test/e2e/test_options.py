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

"""E2e tests for options — verifies mitigation options are accepted by the
serverless cluster and that invalid options cause the job to ERROR.

These tests run through the full serverless pipeline (not a fake backend),
so they confirm that options validation actually fires inside the running
function — not just client-side.

Note: job.result() returns {"hw_results": PrimitiveResult, "metadata": {...}}.
The circuit-function repo's result.metadata (job-level PrimitiveResult metadata)
is not directly accessible here; per-pub precision is checked via
hw_results[0].metadata["target_precision"] instead.
"""

from ddt import ddt, named_data
from qiskit_serverless.exception import QiskitServerlessException

from options import Options

from .base_e2e_test_case import BaseE2eTestCase
from ..utils import get_estimator_pub


@ddt
class TestE2eOptions(BaseE2eTestCase):
    """Verify options handling end-to-end through the live serverless pipeline."""

    def test_with_all_options(self):
        """Every option field is accepted; job reaches DONE; default_precision propagates to metadata."""
        options = {
            "optimization_level": 1,
            "default_precision": 0.1,
            "max_execution_time": 1800,
            "mitigation_level": 1,
            "dynamical_decoupling": {"enable": True, "sequence_type": "XX"},
            "twirling": {"enable_gates": False, "enable_measure": True},
            "resilience": {
                "measure_mitigation": True,
                "zne_mitigation": False,
                "zne": {
                    "amplifier": "pea",
                    "noise_factors": [1.0, 2.0],
                    "extrapolator": "linear",
                },
                "pec_mitigation": False,
                "pec": {"max_overhead": None},
            },
        }
        # Confirm coverage: every field in the model is present in the options dict
        func_model = Options(**options).model_dump(exclude_unset=False)
        self.assertDictEqual(func_model, options)

        job = self._func.run(
            backend_name=self._backend_name, pubs=[get_estimator_pub()], options=options
        )
        result = job.result()

        hw_results = result["hw_results"]
        actual_prec = hw_results[0].metadata["target_precision"]

        self.assertEqual(job.status(), "DONE")
        self.assertEqual(actual_prec, 0.1)

    @named_data(
        ("bad_optimization_level", {"optimization_level": 4}),
        ("bad_dd_seq", {"dynamical_decoupling": {"sequence_type": "YY"}}),
    )
    def test_invalid_options(self, options):
        """Invalid options cause the job to reach ERROR status; logs contain 'ValidationError'."""
        job = self._func.run(
            backend_name=self._backend_name, pubs=[get_estimator_pub()], options=options
        )

        with self.assertRaises(QiskitServerlessException):
            _ = job.result()

        self.assertEqual(job.status(), "ERROR")
        self.assertIn("ValidationError", job.logs())
