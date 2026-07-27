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
Circuit Function Template unit tests — full execution against a fake backend.
"""

import unittest
from unittest import mock
import sys

from ddt import ddt, data, named_data, unpack
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import real_amplitudes
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.primitive_result import PrimitiveResult
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.quantum_info import Pauli, SparsePauliOp, random_pauli_list
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

from circuit_function_entrypoint import run_function
from options import Options

from .base_test_case import BaseTemplateTestCase
from .utils import get_estimator_pub, get_inputs


@ddt
class TestRunFunction(BaseTemplateTestCase):
    """Exercises run_function end-to-end against a fake backend.

    All tests check that:
      - the return value is {"hw_results": PrimitiveResult, "metadata": {...}}
      - hw_results has the expected length / shape
      - numerical values (where seeded) match reference to within shot noise
    """

    # Basic execution
    @data(1, 2, 3)
    def test_returns_one_pub_result_per_pub(self, num_pubs):
        """run_function returns exactly one PubResult per submitted PUB."""
        pubs = [get_estimator_pub()] * num_pubs
        result = run_function(
            **get_inputs(backend_name=self._backend_name, pubs=pubs),
            testing_backend=self._testing_backend,
        )
        hw_results = result["hw_results"]

        self.assertIsInstance(hw_results, PrimitiveResult)
        self.assertEqual(len(hw_results), num_pubs)

    # Observable coercion
    @named_data(
        ("string", "XX"),
        ("dict", {"XX": 0.1, "YY": 0.5}),
        ("SparsePauliOp", SparsePauliOp.from_list([("ZI", 1), ("ZZ", 0.5)])),
        ("Pauli", Pauli("IX")),
        (
            "nested_array",
            [
                [SparsePauliOp(random_pauli_list(2, 3, phase=False)) for _ in range(3)]
                for _ in range(5)
            ],
        ),
    )
    def test_observable_coercion_preserves_size(self, observables):
        """Every observable format is coerced without loss: evs.size == obs_array.size."""
        pubs = [(random_circuit(num_qubits=2, depth=2, seed=42), observables)]
        result = run_function(
            **get_inputs(backend_name=self._backend_name, pubs=pubs),
            testing_backend=self._testing_backend,
        )
        hw_results = result["hw_results"]
        obs_array = ObservablesArray.coerce(observables)
        actual_size = hw_results[0].data.evs.size

        self.assertEqual(obs_array.size, actual_size)

    # Parameterized circuits
    @data(
        [[1, 2, 3, 4]],  # single parameter set
        [[1, 2, 3, 4], [1.2, 2.3, 3.4, 4.5]],  # two parameter sets
    )
    def test_parameterized_circuit_list_values(self, parameter_values):
        """List-style parameter values produce one evs entry per parameter set."""
        circuit = real_amplitudes(num_qubits=2, reps=1)
        pubs = [(circuit, ["XX"], parameter_values)]
        result = run_function(
            **get_inputs(backend_name=self._backend_name, pubs=pubs),
            testing_backend=self._testing_backend,
        )
        hw_results = result["hw_results"]
        bindings = BindingsArray.coerce({tuple(circuit.parameters): parameter_values})
        actual_len = len(hw_results[0].data.evs)

        self.assertEqual(bindings.size, actual_len)

    def test_parameterized_circuit_dict_values(self):
        """Dict-style {param: values} parameter binding is coerced correctly."""
        circuit = real_amplitudes(num_qubits=2, reps=1)
        parameter_values = {param: [1, 2, 3, 4] for param in circuit.parameters}
        pubs = [(circuit, ["XX"], parameter_values)]
        result = run_function(
            **get_inputs(backend_name=self._backend_name, pubs=pubs),
            testing_backend=self._testing_backend,
        )
        hw_results = result["hw_results"]
        bindings = BindingsArray.coerce(parameter_values)
        actual_len = len(hw_results[0].data.evs)

        self.assertEqual(bindings.size, actual_len)

    # Multiple observables × multiple parameter sets
    def test_multi_obs_params_zip_shape(self):
        """Matching leading dims are zipped: 3 obs × 3 param sets → evs shape (3,)."""
        paulis = ("II", "XX", "YY")
        circuit = real_amplitudes(num_qubits=2, reps=1)
        observables = [SparsePauliOp(p) for p in paulis]
        param_sets = np.random.uniform(size=(len(paulis), circuit.num_parameters))

        result = run_function(
            **get_inputs(
                backend_name=self._backend_name, pubs=[(circuit, observables, param_sets)]
            ),
            testing_backend=self._testing_backend,
        )
        hw_results = result["hw_results"]
        actual_len = len(hw_results[0].data.evs)

        self.assertEqual(len(paulis), actual_len)

    def test_multi_obs_params_product_shape(self):
        """2-D obs array with 1-D params broadcasts as outer product → shape (3, 2)."""
        paulis = ("II", "XX", "YY")
        circuit = real_amplitudes(num_qubits=2, reps=1)
        observables = [[SparsePauliOp(p)] for p in paulis]
        num_param_sets = 2
        param_sets = np.random.uniform(size=(num_param_sets, circuit.num_parameters))

        result = run_function(
            **get_inputs(
                backend_name=self._backend_name, pubs=[(circuit, observables, param_sets)]
            ),
            testing_backend=self._testing_backend,
        )
        hw_results = result["hw_results"]
        actual_shape = hw_results[0].data.shape
        expected_shape = (len(paulis), num_param_sets)

        self.assertEqual(actual_shape, expected_shape)

    # Numerical correctness (seeded AerSimulator)
    def test_numerical_values_match_reference(self):
        """Seeded AerSimulator: evs must match hardcoded reference values."""
        backend = AerSimulator(seed_simulator=42)

        def _check(_pubs, expected_evs):
            result = run_function(
                **get_inputs(backend_name=self._backend_name, pubs=_pubs),
                testing_backend=backend,
            )
            hw_results = result["hw_results"]
            self.assertEqual(len(hw_results), len(expected_evs))
            for pub_result, exp_list in zip(hw_results, expected_evs):
                evs = np.asarray(pub_result.data.evs, dtype=float)
                exp = np.asarray(exp_list, dtype=float)
                np.testing.assert_allclose(evs, exp)

        psi1 = real_amplitudes(num_qubits=2, reps=2)
        psi2 = real_amplitudes(num_qubits=2, reps=3)
        obs1 = SparsePauliOp.from_list([("II", 1), ("IZ", 2), ("XI", 3)])
        obs2 = SparsePauliOp.from_list([("IZ", 1)])
        obs3 = SparsePauliOp.from_list([("ZI", 1), ("ZZ", 1)])
        t1, t2, t3 = [0, 1, 1, 2, 3, 5], [0, 1, 1, 2, 3, 5, 8, 13], [1, 2, 3, 4, 5, 6]

        _check([(psi1, obs1, [t1])], [[1.56640625]])
        _check([(psi2, obs1, [t2])], [[2.99267578]])
        _check([(psi1, [obs2, obs3], t1)], [[-0.54931641, 0.05615234]])
        _check(
            [(psi1, [obs1, obs3], [t1, t3]), (psi2, obs2, [t2])],
            [[1.56640625, -1.10449219], [0.17041015625]],
        )

    # Precision routing
    @data(
        # (pub_precision, default_precision)  — pub-level overrides default
        (0.1, 0.2),
        # default propagates to metadata when pub precision is unset
        (None, 0.2),
    )
    @unpack
    def test_precision_precedence(self, pub_precision, default_precision):
        """Pub-level precision overrides default_precision; default propagates when pub is unset."""
        circuit = random_circuit(num_qubits=2, depth=2, seed=42)
        observable = "Z" * circuit.num_qubits
        pubs = [(circuit, observable, None, pub_precision)]
        result = run_function(
            **get_inputs(
                backend_name=self._backend_name,
                pubs=pubs,
                options={"default_precision": default_precision},
            ),
            testing_backend=self._testing_backend,
        )
        hw_results = result["hw_results"]
        expected_prec = pub_precision if pub_precision is not None else default_precision
        actual_prec = hw_results[0].metadata.get("target_precision")

        self.assertEqual(actual_prec, expected_prec)

    # All options accepted
    def test_all_options_accepted_and_precision_propagates(self):
        """Every option field is accepted; options round-trip; precision appears in metadata."""
        options = {
            "optimization_level": 1,
            "default_precision": 0.1,
            "max_execution_time": 300,
            "mitigation_level": 1,
            "dynamical_decoupling": {"enable": True, "sequence_type": "XX"},
            "twirling": {"enable_gates": True, "enable_measure": True},
            "resilience": {
                "measure_mitigation": True,
                "zne_mitigation": True,
                "zne": {"amplifier": "pea", "noise_factors": [1, 2], "extrapolator": "linear"},
                "pec_mitigation": False,
                "pec": {"max_overhead": None},
            },
        }
        func_model = Options(**options).model_dump(exclude_unset=False)
        self.assertDictEqual(func_model, options)

        result = run_function(
            **get_inputs(
                backend_name=self._backend_name, pubs=[get_estimator_pub()], options=options
            ),
            testing_backend=self._testing_backend,
        )
        hw_results = result["hw_results"]
        actual_prec = hw_results[0].metadata["target_precision"]

        self.assertEqual(actual_prec, 0.1)


# Per-pub precision propagation
@ddt
class TestPubPrecision(BaseTemplateTestCase):
    """Per-pub precision is forwarded to metadata; None entries are silently skipped."""

    @data([0.1], [0.1, 0.2], [0.1, None])
    def test_pub_precision(self, precisions):
        """Per-pub precision appears in metadata when set; None skips that pub's precision."""
        circuit = random_circuit(num_qubits=2, depth=2, seed=42)
        observable = "Z" * circuit.num_qubits
        pubs = [(circuit, observable, None, prec) for prec in precisions]
        result = run_function(
            **get_inputs(backend_name=self._backend_name, pubs=pubs),
            testing_backend=self._testing_backend,
        )
        hw_results = result["hw_results"]

        self.assertEqual(len(hw_results), len(precisions))
        for pub_result, target_prec in zip(hw_results, precisions):
            actual_prec = pub_result.metadata.get("target_precision")
            if target_prec is not None:
                self.assertEqual(actual_prec, target_prec)


# Logging
class TestLogging(BaseTemplateTestCase):
    """run_function calls the logger during normal execution."""

    def test_logging_level(self):
        """The serverless logger is called at least once — logging is not silenced."""
        qs_mock = sys.modules["qiskit_serverless"]
        mock_logger = qs_mock.get_logger.return_value
        mock_logger.reset_mock()

        run_function(
            **get_inputs(backend_name=self._backend_name, pubs=[get_estimator_pub()]),
            testing_backend=self._testing_backend,
        )

        calls = mock_logger.info.call_count + mock_logger.debug.call_count
        self.assertGreater(calls, 0)


# Instance kwarg routing
class TestInstance(unittest.TestCase):
    """instance= kwarg must be forwarded to get_runtime_service, not silently dropped."""

    def test_instance_kwarg_reaches_runtime_service(self):
        """Passing instance= causes get_runtime_service() to be called (not testing_backend path)."""
        circuit = QuantumCircuit(0, 0)
        pubs = [EstimatorPub(circuit, observables=ObservablesArray("X" * 0))]

        qs_mock = sys.modules["qiskit_serverless"]
        fake_service = mock.MagicMock()
        fake_service.backend.return_value = FakeManilaV2()
        qs_mock.get_runtime_service.return_value = fake_service

        _ = run_function(
            backend_name=FakeManilaV2().name,
            pubs=pubs,
            instance="h1/g1/p1",
        )

        qs_mock.get_runtime_service.assert_called()


if __name__ == "__main__":
    unittest.main()
