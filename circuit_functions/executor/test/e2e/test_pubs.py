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

"""E2e tests for the executor's two modes through the live serverless pipeline.

job.result() returns {"hw_results": PrimitiveResult, "metadata": {...}} — all tests access
result["hw_results"]. Sampler mode returns raw bit-array data (no evs); estimator mode
reconstructs expectation values client-side, so those pub circuits must be unmeasured.
"""

import numpy as np
from ddt import ddt, data, named_data

from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import real_amplitudes
from qiskit.primitives.containers.primitive_result import PrimitiveResult
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.quantum_info import SparsePauliOp, random_pauli_list

from .base_e2e_test_case import BaseE2eTestCase
from ..utils import combine, get_sampler_pub


@ddt
class TestE2eSamplerPubs(BaseE2eTestCase):
    """Sampler-mode pubs return raw bit-array results through the live serverless pipeline."""

    @data(1, 2, 3)
    def test_one_result_per_pub(self, num_pubs):
        """run_function returns exactly one PubResult per submitted sampler pub (no evs)."""
        pubs = [get_sampler_pub()] * num_pubs

        job = self._func.run(
            backend_name=self._backend_name,
            pubs=pubs,
            mode="sampler",
            options={"default_shots": 256},
        )
        result = job.result()

        hw_results = result["hw_results"]
        self.assertEqual(job.status(), "DONE", f"job {job.job_id} failed: {result}")
        self.assertIsInstance(hw_results, PrimitiveResult)
        self.assertEqual(len(hw_results), num_pubs)
        self.assertNotIn("evs", hw_results[0].data.keys())

    def test_parameterized_sweep_shape(self):
        """A parameter sweep of 3 sets produces a bit array with leading shape (3,)."""
        circuit = real_amplitudes(num_qubits=2, reps=1)
        circuit.measure_all()
        params = np.random.uniform(size=(3, circuit.num_parameters))

        job = self._func.run(
            backend_name=self._backend_name,
            pubs=[(circuit, params)],
            mode="sampler",
            options={"default_shots": 128},
        )
        result = job.result()

        data_bin = result["hw_results"][0].data
        register = next(iter(data_bin.keys()))
        self.assertEqual(job.status(), "DONE", f"job {job.job_id} failed: {result}")
        self.assertEqual(data_bin[register].shape, (3,))


@ddt
class TestE2eEstimatorPubs(BaseE2eTestCase):
    """Estimator-mode pubs reconstruct expectation values through the live serverless pipeline."""

    @data(1, 2, 3)
    def test_min_parameters(self, num_pubs):
        """One PubResult with an evs array per submitted estimator pub."""
        circuit = random_circuit(num_qubits=2, depth=2, seed=42)
        observable = "Z" * circuit.num_qubits
        pubs = [(circuit, observable)] * num_pubs

        job = self._func.run(
            backend_name=self._backend_name,
            pubs=pubs,
            mode="estimator",
            options={"default_precision": 0.1},
        )
        result = job.result()

        hw_results = result["hw_results"]
        self.assertEqual(job.status(), "DONE", f"job {job.job_id} failed: {result}")
        self.assertIsInstance(hw_results, PrimitiveResult)
        self.assertEqual(len(hw_results), num_pubs)
        self.assertIn("evs", hw_results[0].data.keys())

    @data(
        [[1, 2, 3, 4]],
        [np.random.uniform(size=(4,))],
    )
    def test_parameterized_circuit(self, parameter_values):
        """evs length equals the number of parameter sets supplied."""
        circuit = real_amplitudes(num_qubits=2, reps=1)
        pubs = [(circuit, ["XX"], parameter_values)]

        job = self._func.run(
            backend_name=self._backend_name,
            pubs=pubs,
            mode="estimator",
            options={"default_precision": 0.1},
        )
        result = job.result()

        hw_results = result["hw_results"]
        bindings = BindingsArray.coerce({tuple(circuit.parameters): parameter_values})
        actual_len = len(hw_results[0].data.evs)

        self.assertEqual(job.status(), "DONE", f"job {job.job_id} failed: {result}")
        self.assertEqual(bindings.size, actual_len)

    @named_data(
        # Note: dict with Pauli keys skipped — serverless serialisation issue
        # (see https://github.com/Qiskit/qiskit-serverless/issues/1472)
        ("sp_multi", SparsePauliOp.from_list([("ZI", 1), ("ZZ", 0.5)])),
        ("sp_array", [SparsePauliOp("ZZ"), SparsePauliOp("ZI")]),
        (
            "sp_nested",
            [
                [SparsePauliOp(random_pauli_list(2, 3, phase=False)) for _ in range(3)]
                for _ in range(5)
            ],
        ),
    )
    def test_observable_type(self, observables):
        """Every observable format is coerced without loss: evs.size == obs_array.size."""
        pubs = [(random_circuit(num_qubits=2, depth=2, seed=42), observables)]

        job = self._func.run(
            backend_name=self._backend_name,
            pubs=pubs,
            mode="estimator",
            options={"default_precision": 0.2},
        )
        result = job.result()

        hw_results = result["hw_results"]
        obs_array = ObservablesArray.coerce(observables)
        actual_size = hw_results[0].data.evs.size

        self.assertEqual(job.status(), "DONE", f"job {job.job_id} failed: {result}")
        self.assertEqual(obs_array.size, actual_size)

    def test_multi_observables_params_zip(self):
        """Matching leading dims are zipped: 3 obs × 3 param sets → evs length 3."""
        paulis = ("II", "XX", "YY")
        circuit = real_amplitudes(num_qubits=2, reps=1)
        observables = [SparsePauliOp(p) for p in paulis]
        param_sets = np.random.uniform(size=(len(paulis), circuit.num_parameters))

        job = self._func.run(
            backend_name=self._backend_name,
            pubs=[(circuit, observables, param_sets)],
            mode="estimator",
            options={"default_precision": 0.2},
        )
        result = job.result()

        hw_results = result["hw_results"]
        actual_len = len(hw_results[0].data.evs)

        self.assertEqual(job.status(), "DONE", f"job {job.job_id} failed: {result}")
        self.assertEqual(len(paulis), actual_len)

    def test_multi_observables_params_product(self):
        """2-D obs array with 1-D params broadcasts as outer product → shape (3, 2)."""
        paulis = ("II", "XX", "YY")
        circuit = real_amplitudes(num_qubits=2, reps=1)
        observables = [[SparsePauliOp(p)] for p in paulis]
        num_param_sets = 2
        param_sets = np.random.uniform(size=(num_param_sets, circuit.num_parameters))

        job = self._func.run(
            backend_name=self._backend_name,
            pubs=[(circuit, observables, param_sets)],
            mode="estimator",
            options={"default_precision": 0.2},
        )
        result = job.result()

        hw_results = result["hw_results"]
        actual_shape = hw_results[0].data.shape
        expected_shape = (len(paulis), num_param_sets)

        self.assertEqual(job.status(), "DONE", f"job {job.job_id} failed: {result}")
        self.assertEqual(actual_shape, expected_shape)

    @combine(
        pub_precision=[None, 0.1],
        default_precision=[None, 0.2],
    )
    def test_precision(self, pub_precision, default_precision):
        """Pub-level precision overrides default; default propagates when the pub is unset.

        Estimator mode needs at least one precision source, so the all-unset combination is
        skipped rather than run (run_function would raise for it).
        """
        if pub_precision is None and default_precision is None:
            self.skipTest("estimator mode requires a precision source (pub or default_precision)")

        circuit = random_circuit(num_qubits=2, depth=2, seed=42)
        observable = "Z" * circuit.num_qubits
        options = {"default_precision": default_precision} if default_precision is not None else {}

        pubs = [(circuit, observable, None, pub_precision)]
        job = self._func.run(
            backend_name=self._backend_name,
            pubs=pubs,
            mode="estimator",
            options=options,
        )
        result = job.result()

        hw_results = result["hw_results"]
        expected_prec = pub_precision or default_precision
        actual_prec = hw_results[0].metadata.get("target_precision")

        self.assertEqual(job.status(), "DONE", f"job {job.job_id} failed: {result}")
        self.assertEqual(actual_prec, expected_prec)
