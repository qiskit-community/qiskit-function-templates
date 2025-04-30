# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Hamiltonian Simulation Function Template unit tests.
"""

from itertools import chain
import unittest
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime.fake_provider import FakeFez

from ..hamiltonian_simulation import run_function


class TestHamiltonianSimulation(unittest.TestCase):
    """
    Test Hamiltonian Simulation
    """

    def setUp(self):
        super().setUp()
        np.random.seed(0)
        length = 50
        # Generate the edge list for this spin-chain
        edges = [(i, i + 1) for i in range(length - 1)]
        # Generate an edge-coloring so we can make hw-efficient circuits
        edges = edges[::2] + edges[1::2]
        # Generate random coefficients for our XXZ Hamiltonian
        coeffs = np.random.rand(length - 1) + 0.5 * np.ones(length - 1)
        self.hamiltonian = SparsePauliOp.from_sparse_list(
            chain.from_iterable(
                [
                    [
                        ("XX", (i, j), coeffs[i] / 2),
                        ("YY", (i, j), coeffs[i] / 2),
                        ("ZZ", (i, j), coeffs[i]),
                    ]
                    for i, j in edges
                ]
            ),
            num_qubits=length,
        )
        self.observable = SparsePauliOp.from_sparse_list(
            [("ZZ", (length // 2 - 1, length // 2), 1.0)], num_qubits=length
        )
        self.initial_state = QuantumCircuit(length)
        for i in range(length):
            if i % 2:
                self.initial_state.x(i)

    def test_run(self):
        """Test run_function"""

        out = run_function(
            initial_state=self.initial_state,
            hamiltonian=self.hamiltonian,
            observable=self.observable,
            backend_name="ibm_fez",
            estimator_options={},
            aqc_evolution_time=0.2,
            aqc_ansatz_num_trotter_steps=1,
            aqc_target_num_trotter_steps=32,
            remainder_evolution_time=0.2,
            remainder_num_trotter_steps=4,
            aqc_max_iterations=300,
            dry_run=True,
            testing_backend=FakeFez(),
        )
        with self.subTest("num params"):
            self.assertEqual(out.get("num_aqc_parameters"), 816)
        with self.subTest("starting fidelity"):
            self.assertAlmostEqual(out.get("aqc_starting_fidelity"), 0.9914, 4)
        with self.subTest("num iterations"):
            self.assertTrue(out.get("num_iterations") < 300)
        with self.subTest("final fidelity"):
            self.assertAlmostEqual(out.get("aqc_fidelity"), 0.999782, 4)
        with self.subTest("2q depth"):
            self.assertEqual(out.get("twoqubit_depth"), 33)
