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
Circuit Function Template unit tests — input validation and rejection.
"""

from ddt import data, ddt
from pydantic import ValidationError

from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import efficient_su2
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit_ibm_runtime.fake_provider import FakeAlgiers
from qiskit_ibm_runtime.exceptions import IBMInputValueError

from circuit_function_entrypoint import run_function
from options import (
    MAX_MITIGATION_LEVEL,
    MAX_OPTIMIZATION_LEVEL,
    Options,
)

from .utils import get_inputs
from .base_test_case import BaseTemplateTestCase


# Options validation
@ddt
class TestOptionsValidation(BaseTemplateTestCase):
    """run_function raises ValidationError before touching hardware when options are malformed."""

    @data(
        -1,  # below minimum
        MAX_OPTIMIZATION_LEVEL + 1,  # above maximum
        "foo",  # wrong type entirely
    )
    def test_invalid_optimization_level(self, bad_level):
        """optimization_level outside [0, 3] or of wrong type → ValidationError."""
        with self.assertRaises(ValidationError):
            run_function(
                **get_inputs(
                    backend_name=self._backend_name, options={"optimization_level": bad_level}
                ),
                testing_backend=self._testing_backend,
            )

    @data(
        -1,  # below minimum
        MAX_MITIGATION_LEVEL + 1,  # above maximum
        "foo",  # wrong type entirely
    )
    def test_invalid_mitigation_level(self, bad_level):
        """mitigation_level outside [1, 3] or of wrong type → ValueError or ValidationError."""
        with self.assertRaises((ValueError, ValidationError)):
            run_function(
                **get_inputs(
                    backend_name=self._backend_name, options={"mitigation_level": bad_level}
                ),
                testing_backend=self._testing_backend,
            )

    def test_invalid_options_structure(self):
        """Passing nested-only keys at the top level is rejected — structure must be correct."""
        # pec_mitigation lives under resilience.pec_mitigation, not at the top level
        with self.assertRaises(ValidationError):
            run_function(
                **get_inputs(
                    backend_name=self._backend_name,
                    options={"pec_mitigation": True, "max_overhead": 100},
                ),
                testing_backend=self._testing_backend,
            )

    def test_estimator_option_not_in_schema(self):
        """A key valid for EstimatorOptions but absent from the Options schema is rejected."""
        with self.assertRaises(ValidationError):
            run_function(
                **get_inputs(backend_name=self._backend_name, options={"seed_estimator": 42}),
                testing_backend=self._testing_backend,
            )

    def test_default_options(self):
        """options=None is accepted and the function completes successfully."""
        result = run_function(
            **get_inputs(backend_name=self._backend_name, options=None),
            testing_backend=self._testing_backend,
        )
        self.assertIn("hw_results", result)

    def test_all_options_accepted(self):
        """All valid option fields can be set simultaneously without error."""
        options = {
            "default_precision": 0.01,
            "max_execution_time": 300,
            "mitigation_level": 1,
            "optimization_level": 1,
            "dynamical_decoupling": {"enable": True, "sequence_type": "XX"},
            "twirling": {"enable_gates": True, "enable_measure": True},
            "resilience": {
                "zne_mitigation": True,
                "pec_mitigation": False,
                "zne": {
                    "amplifier": "gate_folding",
                    "noise_factors": (1, 3),
                    "extrapolator": "linear",
                },
                "pec": {"max_overhead": None},
            },
        }
        result = run_function(
            **get_inputs(backend_name=self._backend_name, options=options),
            testing_backend=self._testing_backend,
        )
        self.assertIn("hw_results", result)

    def test_options_routing_transpiler_vs_estimator(self):
        """optimization_level is routed to the transpiler; default_precision stays for the estimator."""
        options = {"default_precision": 0.01, "optimization_level": 1}
        opts_dict = Options(**options).model_dump(exclude_unset=True)
        transpilation_options = Options.get_transpilation_options(opts_dict)

        self.assertDictEqual(transpilation_options, {"optimization_level": 1})
        self.assertNotIn("optimization_level", opts_dict)
        self.assertIn("default_precision", opts_dict)


# PUBs validation
class TestPubsValidation(BaseTemplateTestCase):
    """run_function rejects malformed PUBs before submitting to the backend."""

    def test_missing_observables(self):
        """A pub tuple with no observable is rejected with a clear 'length of pub' message."""
        circ = random_circuit(2, 2, measure=False)
        with self.assertRaisesRegex(ValueError, "length of pub"):
            run_function(
                **get_inputs(backend_name=self._backend_name, pubs=[(circ,)]),
                testing_backend=self._testing_backend,
            )

    def test_missing_circuit_params(self):
        """Submitting a parameterized circuit without parameter values is rejected."""
        circ = efficient_su2(2, reps=1)
        with self.assertRaisesRegex(ValueError, "does not match the number of parameters"):
            run_function(
                **get_inputs(backend_name=self._backend_name, pubs=[(circ, "Z" * circ.num_qubits)]),
                testing_backend=self._testing_backend,
            )

    def test_empty_pubs(self):
        """An empty pubs list is rejected immediately — at least one PUB is required."""
        with self.assertRaisesRegex(ValueError, "At least one PUB"):
            run_function(
                backend_name=self._backend_name,
                pubs=[],
                testing_backend=self._testing_backend,
            )

    def test_are_dynamic_circuits(self):
        """Circuits with classical control flow (if_else) are explicitly unsupported."""
        dynamic_circ = QuantumCircuit(3, 1)
        dynamic_circ.h(0)
        dynamic_circ.measure(0, 0)
        dynamic_circ.if_else((0, True), QuantumCircuit(3, 1), QuantumCircuit(3, 1), [0, 1, 2], [0])
        with self.assertRaisesRegex(ValueError, "Dynamic circuits are not supported."):
            run_function(
                **get_inputs(backend_name=self._backend_name, pubs=[(dynamic_circ, ["XXX"])]),
                testing_backend=self._testing_backend,
            )


# Backend validation
class TestBackendValidation(BaseTemplateTestCase):
    """run_function rejects invalid backend arguments before transpilation."""

    def test_missing_backend(self):
        """backend_name=None is rejected with a clear 'Invalid backend name value' message."""
        inputs = get_inputs(backend_name=self._backend_name)
        inputs["backend_name"] = None
        with self.assertRaisesRegex(ValueError, "Invalid backend name value"):
            run_function(**inputs, testing_backend=self._testing_backend)

    def test_backend_num_qubits(self):
        """A circuit that needs more qubits than the backend supports → IBMInputValueError."""
        backend = FakeAlgiers()
        num_qubits = backend.num_qubits + 1
        pubs = [
            EstimatorPub(
                random_circuit(num_qubits, depth=1),
                observables=ObservablesArray("X" * num_qubits),
            )
        ]
        with self.assertRaises(IBMInputValueError):
            run_function(
                **get_inputs(backend_name=backend.name, pubs=pubs),
                testing_backend=backend,
            )
