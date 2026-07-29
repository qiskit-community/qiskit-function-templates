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
Executor Function Template unit tests: full run_function execution in both modes against a
fake backend, with the hardware submission simulated by FakeExecutor.
"""

import unittest

from ddt import ddt, data, named_data
import numpy as np

from qiskit.circuit.library import real_amplitudes
from qiskit.circuit.random import random_circuit
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.primitives.containers.primitive_result import PrimitiveResult
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector, random_pauli_list

from executor_entrypoint import run_function

from .base_test_case import BaseExecutorTestCase
from .utils import get_estimator_pub, get_inputs, get_sampler_pub


@ddt
class TestSamplerMode(BaseExecutorTestCase):
    """Sampler mode returns raw bit-array results, one PubResult per pub."""

    @data(1, 2, 3)
    def test_one_result_per_pub(self, num_pubs):
        """run_function returns exactly one PubResult per submitted sampler pub."""
        pubs = [get_sampler_pub()] * num_pubs
        result = run_function(
            **get_inputs(
                backend_name=self._backend_name,
                pubs=pubs,
                mode="sampler",
                options={"default_shots": 256},
            ),
            testing_backend=self._testing_backend,
        )
        hw_results = result["hw_results"]
        self.assertIsInstance(hw_results, PrimitiveResult)
        self.assertEqual(len(hw_results), num_pubs)

    def test_default_mode_is_sampler(self):
        """Omitting mode defaults to sampler semantics (bit-array data, no evs)."""
        result = run_function(
            backend_name=self._backend_name,
            pubs=[get_sampler_pub()],
            options={"default_shots": 256},
            testing_backend=self._testing_backend,
        )
        self.assertIsInstance(result["hw_results"], PrimitiveResult)
        self.assertNotIn("evs", result["hw_results"][0].data.keys())

    def test_parameterized_circuit_sweep_shape(self):
        """A parameter sweep of 3 sets produces a bit array with leading shape (3,)."""
        circuit = real_amplitudes(num_qubits=2, reps=1)
        circuit.measure_all()
        params = np.random.default_rng(0).uniform(size=(3, circuit.num_parameters))
        result = run_function(
            **get_inputs(
                backend_name=self._backend_name,
                pubs=[(circuit, params)],
                mode="sampler",
                options={"default_shots": 128},
            ),
            testing_backend=self._testing_backend,
        )
        data_bin = result["hw_results"][0].data
        register = next(iter(data_bin.keys()))
        self.assertEqual(data_bin[register].shape, (3,))

    def test_metadata_resource_usage_present(self):
        """The standard resource-usage timing blocks are reported."""
        result = run_function(
            **get_inputs(
                backend_name=self._backend_name, mode="sampler", options={"default_shots": 64}
            ),
            testing_backend=self._testing_backend,
        )
        usage = result["metadata"]["resources_usage"]
        self.assertIn("RUNNING: OPTIMIZING_FOR_HARDWARE", usage)
        self.assertIn("RUNNING: EXECUTING_QPU", usage)


@ddt
class TestEstimatorMode(BaseExecutorTestCase):
    """Estimator mode reconstructs expectation values client-side from the sampler path."""

    @data(1, 2, 3)
    def test_one_result_per_pub(self, num_pubs):
        """One PubResult with an evs array per submitted estimator pub."""
        pubs = [get_estimator_pub()] * num_pubs
        result = run_function(
            **get_inputs(
                backend_name=self._backend_name,
                pubs=pubs,
                mode="estimator",
                options={"default_precision": 0.1},
            ),
            testing_backend=self._testing_backend,
        )
        hw_results = result["hw_results"]
        self.assertIsInstance(hw_results, PrimitiveResult)
        self.assertEqual(len(hw_results), num_pubs)
        self.assertIn("evs", hw_results[0].data.keys())

    @named_data(
        ("string", "XX"),
        ("dict", {"XX": 0.1, "YY": 0.5}),
        ("sparse_pauli_op", SparsePauliOp.from_list([("ZI", 1), ("ZZ", 0.5)])),
        ("pauli", Pauli("IX")),
        (
            "nested_array",
            [
                [SparsePauliOp(random_pauli_list(2, 3, phase=False)) for _ in range(3)]
                for _ in range(2)
            ],
        ),
    )
    def test_observable_coercion_preserves_shape(self, observables):
        """Every observable format is coerced and the evs shape matches the observable array."""
        circuit = random_circuit(num_qubits=2, depth=2, seed=42)
        result = run_function(
            **get_inputs(
                backend_name=self._backend_name,
                pubs=[(circuit, observables)],
                mode="estimator",
                options={"default_precision": 0.2},
            ),
            testing_backend=self._testing_backend,
        )
        expected_shape = ObservablesArray.coerce(observables).shape
        self.assertEqual(np.asarray(result["hw_results"][0].data.evs).shape, expected_shape)

    def test_numerical_values_match_statevector(self):
        """Seeded reconstruction matches exact statevector expectation values."""
        circuit = real_amplitudes(num_qubits=2, reps=2)
        params = np.random.default_rng(3).uniform(size=circuit.num_parameters)
        observables = [
            SparsePauliOp.from_list([("XX", 1.0), ("ZZ", 0.5), ("IZ", 1.0)]),
            SparsePauliOp.from_list([("YI", 0.8), ("XZ", 1.0)]),
        ]
        result = run_function(
            **get_inputs(
                backend_name=self._backend_name,
                pubs=[(circuit, observables, params, 0.02)],
                mode="estimator",
            ),
            testing_backend=self._testing_backend,
        )
        evs = np.asarray(result["hw_results"][0].data.evs, dtype=float)
        bound = Statevector(circuit.assign_parameters(params))
        exact = np.array([bound.expectation_value(obs).real for obs in observables])
        np.testing.assert_allclose(evs, exact, atol=0.1)

    def test_precision_in_metadata(self):
        """The target precision is propagated to the result metadata."""
        result = run_function(
            **get_inputs(
                backend_name=self._backend_name,
                pubs=[(random_circuit(2, 2, seed=42), "ZZ", None, 0.05)],
                mode="estimator",
            ),
            testing_backend=self._testing_backend,
        )
        self.assertEqual(result["hw_results"][0].metadata["target_precision"], 0.05)

    def test_default_precision_propagates_to_metadata(self):
        """A precision supplied via default_precision (no pub precision) is recorded as
        target_precision."""
        result = run_function(
            **get_inputs(
                backend_name=self._backend_name,
                pubs=[(random_circuit(2, 2, seed=42), "ZZ")],
                mode="estimator",
                options={"default_precision": 0.2},
            ),
            testing_backend=self._testing_backend,
        )
        self.assertEqual(result["hw_results"][0].metadata["target_precision"], 0.2)

    def test_default_shots_is_a_floor_in_estimator_mode(self):
        """When both are set, default_shots floors the precision-derived shot count while the
        precision still drives (and is recorded)."""
        result = run_function(
            **get_inputs(
                backend_name=self._backend_name,
                pubs=[(random_circuit(2, 2, seed=42), "ZZ")],
                mode="estimator",
                options={"default_precision": 0.5, "default_shots": 4096},
            ),
            testing_backend=self._testing_backend,
        )
        # precision 0.5 -> 4 shots, floored up to 4096; precision still recorded as the target.
        self.assertEqual(result["hw_results"][0].metadata["target_precision"], 0.5)
        self.assertGreaterEqual(result["hw_results"][0].metadata["shots"], 4096)


class TestDryRun(BaseExecutorTestCase):
    """A dry run stops after optimization without producing hardware results."""

    def test_dry_run_skips_hardware(self):
        """dry_run returns metadata only, before any execution."""
        result = run_function(
            **get_inputs(
                backend_name=self._backend_name, mode="sampler", options={"default_shots": 64}
            ),
            testing_backend=self._testing_backend,
            dry_run=True,
        )
        self.assertNotIn("hw_results", result)
        self.assertIn("RUNNING: OPTIMIZING_FOR_HARDWARE", result["metadata"]["resources_usage"])


if __name__ == "__main__":
    unittest.main()
