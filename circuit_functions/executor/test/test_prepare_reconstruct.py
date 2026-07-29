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
Unit tests for the estimator-mode math: prepare_estimator (observables -> measurement
circuits) and reconstruct_estimator (sampled counts -> expectation values), exercised for
real against a seeded Aer sampler and compared to exact statevector values.
"""

import unittest

from ddt import ddt, named_data
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import real_amplitudes
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector
from qiskit_aer.primitives import SamplerV2 as AerSampler

from executor_entrypoint import prepare_estimator, reconstruct_estimator, resolve_precision


def _to_operator(observable):
    """Coerce any supported observable format to a SparsePauliOp for the exact reference."""
    if isinstance(observable, SparsePauliOp):
        return observable
    if isinstance(observable, Pauli):
        return SparsePauliOp(observable)
    if isinstance(observable, dict):
        return SparsePauliOp.from_list(list(observable.items()))
    return SparsePauliOp(observable)


def _fixed_circuit() -> QuantumCircuit:
    """A small non-parametric circuit for structural prepare tests."""
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    return circuit


def _sample(circuits, shots, seed=1234):
    """Run prepared measurement circuits through a seeded Aer sampler."""
    return AerSampler(default_shots=shots, seed=seed).run([(c,) for c in circuits]).result()


def _run(pub):
    """prepare -> sample -> reconstruct, returning the reconstructed evs array.

    Mirrors the precision->shots resolution that run_function performs before sampling.
    """
    precision = resolve_precision([pub])
    shots = int(np.ceil(1.0 / (precision**2)))
    circuits, recon = prepare_estimator([pub])
    result = reconstruct_estimator(_sample(circuits, shots), recon, precision)
    return np.asarray(result[0].data.evs, dtype=float), recon


@ddt
class TestReconstruction(unittest.TestCase):
    """Reconstructed expectation values match exact statevector values within shot noise."""

    @named_data(
        ("string", "XX"),
        ("pauli", Pauli("YI")),
        ("multi_group", SparsePauliOp.from_list([("XX", 1.0), ("ZZ", 0.5), ("IZ", 1.0)])),
        ("dict", {"XZ": 0.7, "YY": 0.3}),
    )
    def test_matches_statevector(self, observable):
        """A single observable reconstructs to its exact expectation value."""
        circuit = real_amplitudes(num_qubits=2, reps=2)
        params = np.random.default_rng(1).uniform(size=circuit.num_parameters)
        pub = EstimatorPub.coerce((circuit, observable, [params], 0.02))

        evs, _ = _run(pub)
        exact = (
            Statevector(circuit.assign_parameters(params))
            .expectation_value(_to_operator(observable))
            .real
        )

        np.testing.assert_allclose(evs.ravel()[0], exact, atol=0.1)

    def test_broadcast_obs_and_params(self):
        """Observables (2,) broadcast against parameter sets (2,) give evs of shape (2,)."""
        circuit = real_amplitudes(num_qubits=2, reps=1)
        observables = [SparsePauliOp("ZZ"), SparsePauliOp.from_list([("XX", 1.0), ("YY", 0.5)])]
        params = np.random.default_rng(2).uniform(size=(2, circuit.num_parameters))
        pub = EstimatorPub.coerce((circuit, observables, params, 0.02))

        evs, recon = _run(pub)
        self.assertEqual(recon[0]["shape"], (2,))
        self.assertEqual(evs.shape, (2,))
        exact = [
            Statevector(circuit.assign_parameters(params[k])).expectation_value(observables[k]).real
            for k in range(2)
        ]
        np.testing.assert_allclose(evs, exact, atol=0.1)


class TestPrepare(unittest.TestCase):
    """Structural checks on prepare_estimator and precision resolution."""

    def test_commuting_terms_share_one_circuit(self):
        """Qubit-wise-commuting Paulis are measured together, not one circuit per term."""
        # ZZ, IZ, ZI all commute qubit-wise -> a single measurement circuit.
        operator = SparsePauliOp.from_list([("ZZ", 1), ("IZ", 1), ("ZI", 1)])
        pub = EstimatorPub.coerce((_fixed_circuit(), operator))
        circuits, recon = prepare_estimator([pub])
        self.assertEqual(len(circuits), 1)
        self.assertEqual(recon[0]["shape"], ())

    def test_anticommuting_bases_split(self):
        """Different measurement bases (X vs Z on a qubit) require separate circuits."""
        operator = SparsePauliOp.from_list([("ZZ", 1), ("XX", 1)])
        pub = EstimatorPub.coerce((_fixed_circuit(), operator))
        circuits, _ = prepare_estimator([pub])
        self.assertEqual(len(circuits), 2)


class TestResolvePrecision(unittest.TestCase):
    """resolve_precision (copied from the reference) drives the shot budget."""

    def test_pub_precision_returned(self):
        """The pub's precision is returned when set."""
        pub = EstimatorPub.coerce((_fixed_circuit(), "ZZ", None, 0.1))
        self.assertEqual(resolve_precision([pub]), 0.1)

    def test_none_when_unset(self):
        """No precision anywhere resolves to None (run_function then falls back to defaults)."""
        pub = EstimatorPub.coerce((_fixed_circuit(), "ZZ"))
        self.assertIsNone(resolve_precision([pub]))

    def test_mismatched_precision_raises(self):
        """Pubs must agree on precision."""
        pub_a = EstimatorPub.coerce((_fixed_circuit(), "ZZ", None, 0.1))
        pub_b = EstimatorPub.coerce((_fixed_circuit(), "ZZ", None, 0.2))
        with self.assertRaises(ValueError):
            resolve_precision([pub_a, pub_b])


if __name__ == "__main__":
    unittest.main()
