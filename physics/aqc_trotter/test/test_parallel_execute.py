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

"""Parallel execution path (statevector/fake): parallel must equal sequential.

Opt-in ``parallel_sim=True`` fans the per-time-step PUB loop across all available
cores as worker processes (``execute._run_parallel``, a stdlib
``ProcessPoolExecutor``). For the exact ``statevector`` path the parallel output
must match the sequential output bit-for-bit — this guards that the fan-out is a
pure refactor of *what* is computed, only *where* it runs. One case drives the
full ``DynamicsFunction`` pipeline to confirm the ``parallel_sim`` input is
plumbed end-to-end.
"""

import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from ..source_files.source import build as build_stage
from ..source_files.source.app_function import DynamicsFunction
from ..source_files.source.execute import ExecutionOptions, run_pubs


def _circuits(n, count):
    """A few distinct, deterministic circuits (angles vary per circuit)."""
    out = []
    for k in range(count):
        qc = QuantumCircuit(n)
        for q in range(n):
            qc.ry(0.1 * (k + 1) * (q + 1), q)
        for q in range(n - 1):
            qc.cx(q, q + 1)
        out.append(qc)
    return out


def _pubs(n, count):
    """``count`` PUBs on ``n`` qubits, each measuring the default per-site Z list."""
    obs, _ = build_stage.build_observables(n)
    return [(qc, obs) for qc in _circuits(n, count)]


class TestParallelExecute(unittest.TestCase):
    """Process-pool fan-out must reproduce the sequential result exactly.

    The fan-out uses a stdlib ProcessPoolExecutor (no Ray), so this runs on the
    plain runtime with no cluster setup. ``subTest`` keeps the scenarios reported
    separately while sharing one test id.
    """

    def test_parallel_matches_sequential(self):
        """Fan-out changes only where the PUBs run, never what they evaluate to."""
        # count > cores exercises multi-circuit chunks (and intra-chunk ordering);
        # count < cores caps the chunks at n_pubs (one circuit each, no empties).
        for case, n, count in (("many circuits", 5, 20), ("few circuits", 4, 3)):
            with self.subTest(case=case):
                pubs = _pubs(n, count)
                seq = run_pubs(pubs, ExecutionOptions(backend="statevector", parallel_sim=False))
                par = run_pubs(pubs, ExecutionOptions(backend="statevector", parallel_sim=True))
                self.assertEqual(seq.shape, (count, n))
                self.assertEqual(par.shape, (count, n))
                # exact: statevector is deterministic
                np.testing.assert_array_equal(par, seq)

        with self.subTest(case="end to end"):
            ham = SparsePauliOp.from_sparse_list(
                [(p, [i, i + 1], 0.5) for i in range(3) for p in ("XX", "YY", "ZZ")], num_qubits=4
            )
            args = {
                "t_steps": 3,
                "aqc_segments": [{"n_steps": 1, "ansatz_steps": 1}],
                "dt": 0.2,
                "hamiltonian": ham,
                "backend": "statevector",
            }
            seq = DynamicsFunction(**args, parallel_sim=False).run()
            par = DynamicsFunction(**args, parallel_sim=True).run()
            np.testing.assert_allclose(
                np.array(par["expectation_values"]), np.array(seq["expectation_values"])
            )


class TestParallelSimNotFannedOut(unittest.TestCase):
    """`parallel_sim=True` requests that never fan out, so no pool is created."""

    def test_single_pub_not_fanned_out(self):
        """A single PUB is never parallelized, even with parallel_sim=True."""
        n = 4
        pubs = _pubs(n, 1)
        out = run_pubs(pubs, ExecutionOptions(backend="statevector", parallel_sim=True))
        self.assertEqual(out.shape, (1, n))
