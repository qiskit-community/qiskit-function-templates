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
Executor Function Template unit tests: input validation and error handling.
"""

import unittest

from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit

from executor_entrypoint import run_function

from .base_test_case import BaseExecutorTestCase
from .utils import get_estimator_pub, get_sampler_pub


class TestValidation(BaseExecutorTestCase):
    """run_function rejects malformed inputs with clear errors."""

    def test_empty_pubs_raises(self):
        """No pubs is an error."""
        with self.assertRaises(ValueError):
            run_function(
                backend_name=self._backend_name, pubs=[], testing_backend=self._testing_backend
            )

    def test_invalid_mode_raises(self):
        """An unknown mode string is rejected."""
        with self.assertRaises(ValueError):
            run_function(
                backend_name=self._backend_name,
                pubs=[get_sampler_pub()],
                mode="wavefunction",
                testing_backend=self._testing_backend,
            )

    def test_missing_backend_name_raises(self):
        """An empty backend name is rejected."""
        with self.assertRaises(ValueError):
            run_function(
                backend_name="",
                pubs=[get_sampler_pub()],
                options={"default_shots": 64},
                testing_backend=self._testing_backend,
            )

    def test_sampler_shots_unspecified_raises(self):
        """Sampler mode needs shots either on the pub or as default_shots."""
        with self.assertRaises(ValueError):
            run_function(
                backend_name=self._backend_name,
                pubs=[get_sampler_pub()],
                mode="sampler",
                testing_backend=self._testing_backend,
            )

    def test_sampler_mismatched_shots_raises(self):
        """All sampler pubs must share the same shot count."""
        qc = random_circuit(2, 2, measure=True, seed=1)
        with self.assertRaises(ValueError):
            run_function(
                backend_name=self._backend_name,
                pubs=[(qc, None, 100), (qc, None, 200)],
                mode="sampler",
                testing_backend=self._testing_backend,
            )

    def test_estimator_empty_observables_raises(self):
        """An empty observable array is rejected in estimator mode."""
        circuit = random_circuit(2, 2, seed=42)
        with self.assertRaises(ValueError):
            run_function(
                backend_name=self._backend_name,
                pubs=[(circuit, [])],
                mode="estimator",
                options={"default_precision": 0.1},
                testing_backend=self._testing_backend,
            )

    def test_estimator_circuit_with_measurements_raises(self):
        """Estimator-mode circuits must not carry measurements."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.measure_all()
        with self.assertRaises(ValueError):
            run_function(
                backend_name=self._backend_name,
                pubs=[(circuit, "ZZ")],
                mode="estimator",
                options={"default_precision": 0.1},
                testing_backend=self._testing_backend,
            )

    def test_estimator_missing_precision_raises(self):
        """Estimator mode needs precision on the pub or as default_precision."""
        with self.assertRaises(ValueError):
            run_function(
                backend_name=self._backend_name,
                pubs=[get_estimator_pub()],
                mode="estimator",
                testing_backend=self._testing_backend,
            )


if __name__ == "__main__":
    unittest.main()
