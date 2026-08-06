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

"""Fake-backend execution path: noisy *local* simulation, no credentials.

``backend='fake'`` runs ``EstimatorV2(mode=fake_backend)`` in-process (Aer
noise model), so it must work in a plain ``DynamicsFunction(**args).run()`` call
exactly like ``statevector`` — just noisier. These tests skip cleanly where
qiskit-aer is absent (e.g. a credential-free .venv), since the fake path needs
the Aer noise model.
"""

import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from ..source_files.source import execute as execute_stage
from ..source_files.source.app_function import DynamicsFunction

try:
    # The fake backend needs Aer's noise model; only its availability matters here.
    import qiskit_aer  # pylint: disable=unused-import

    HAS_AER = True
except ImportError:  # pragma: no cover
    HAS_AER = False


def _run(arguments: dict) -> dict:
    """Mirror what program.py does on the gateway: build + run, return the dict."""
    return DynamicsFunction(**arguments).run()


def _neel(n):
    """Néel product state as a circuit: X on even sites (|0101...> little-endian)."""
    qc = QuantumCircuit(n)
    for i in range(0, n, 2):
        qc.x(i)
    return qc


# Small chain keeps the noisy-simulation cost down while still exercising the
# transpile -> layout-remap -> EstimatorV2(local) path end to end.
_HEIS_6 = SparsePauliOp.from_sparse_list(
    [(p, [i, i + 1], 0.5) for i in range(5) for p in ("XX", "YY", "ZZ")], num_qubits=6
)
BASE_ARGS = {
    "t_steps": 4,
    "aqc_segments": [{"n_steps": 2, "ansatz_steps": 1}],
    "dt": 0.2,
    "hamiltonian": _HEIS_6,
    "initial_state": _neel(6),
}


@unittest.skipUnless(HAS_AER, "qiskit-aer is required for the fake backend's noise model")
class TestFakeBackendExecution(unittest.TestCase):
    """Running the full pipeline against a local noisy fake backend."""

    def test_fake_backend_runs_locally_no_credentials(self):
        """The fake path completes in-process with no IBM Quantum credentials."""
        result = _run({**BASE_ARGS, "backend": "fake"})
        self.assertEqual(result["metadata"]["execution_backend"], "fake")
        ev = np.array(result["expectation_values"])
        self.assertEqual(ev.shape, (5, 6))
        self.assertTrue(np.all(np.isfinite(ev)))

    def test_fake_backend_accepts_transpiler_options(self):
        """transpiler_options is splatted into generate_preset_pass_manager."""
        result = _run(
            {
                **BASE_ARGS,
                "backend": "fake",
                "transpiler_options": {"optimization_level": 1, "seed_transpiler": 7},
            }
        )
        ev = np.array(result["expectation_values"])
        self.assertEqual(ev.shape, (5, 6))
        self.assertTrue(np.all(np.isfinite(ev)))

    def test_fake_expectations_match_statevector_shape_and_are_noisy(self):
        """The fake series has the exact path's shape but is genuinely noisy."""
        sv = _run({**BASE_ARGS, "backend": "statevector"})
        fake = _run({**BASE_ARGS, "backend": "fake"})
        ev_sv = np.array(sv["expectation_values"])
        ev_fake = np.array(fake["expectation_values"])
        self.assertEqual(ev_fake.shape, ev_sv.shape)
        # noise perturbs the expectations away from exact, but not arbitrarily:
        # the two series should still be positively correlated.
        corr = np.corrcoef(ev_sv.ravel(), ev_fake.ravel())[0, 1]
        self.assertGreater(corr, 0.5)
        # genuinely noisy, not the exact path
        self.assertFalse(np.allclose(ev_sv, ev_fake))


@unittest.skipUnless(HAS_AER, "qiskit-aer is required for the fake backend's noise model")
class TestFakeBackendSelection(unittest.TestCase):
    """Choosing and sizing the local fake backend."""

    def test_default_fake_backend_is_sherbrooke(self):
        """With no name, the 127-qubit Sherbrooke snapshot is used."""
        opts = execute_stage.ExecutionOptions(backend="fake")
        backend = execute_stage._make_fake_backend(10, opts)
        self.assertEqual(backend.num_qubits, 127)

    def test_default_falls_back_to_generic_for_oversized_chain(self):
        """A chain larger than Sherbrooke auto-sizes a GenericBackendV2."""
        opts = execute_stage.ExecutionOptions(backend="fake")
        backend = execute_stage._make_fake_backend(200, opts)  # bigger than Sherbrooke
        self.assertGreaterEqual(backend.num_qubits, 200)

    def test_named_fake_backend_rejects_too_small(self):
        """An explicitly named fake that can't host the chain is an error."""
        opts = execute_stage.ExecutionOptions(backend="fake", backend_name="fake_manila")
        with self.assertRaises(ValueError):
            execute_stage._make_fake_backend(20, opts)  # FakeManila has 5 qubits
