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

"""Ray fan-out execution path (statevector/fake): parallel must equal sequential.

Opt-in ``parallel_sim=True`` fans the per-time-step PUB loop across all available
cores as Ray tasks (``execute._run_parallel``). For the exact ``statevector``
path the parallel output must match the sequential output bit-for-bit — this
guards that the fan-out is a pure refactor of *what* is computed, only *where*
it runs. One case drives the full ``DynamicsFunction`` pipeline to confirm the
``parallel_sim`` input is plumbed end-to-end.
"""

import os
import unittest
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from ..source_files.source import build as build_stage
from ..source_files.source.app_function import DynamicsFunction
from ..source_files.source.execute import ExecutionOptions, run_pubs

try:
    import ray

    HAS_RAY = True
except ImportError:  # pragma: no cover
    HAS_RAY = False

# Ray serialises the remote chunk function by reference, so each worker process
# re-imports the module it lives in. Workers don't inherit the driver's sys.path,
# so the repository root — the anchor for the
# ``physics.aqc_trotter.source_files.source.execute`` import path —
# has to be handed to them explicitly via the runtime environment.
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
_WORKER_PYTHONPATH = os.pathsep.join(p for p in (_REPO_ROOT, os.environ.get("PYTHONPATH", "")) if p)


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


@unittest.skipUnless(HAS_RAY, "ray is required for the parallel execution path")
class TestParallelExecute(unittest.TestCase):
    """Ray fan-out must reproduce the sequential result exactly.

    Every scenario shares one test method deliberately. ``stestr`` shards by test
    id, so a method per scenario hands this class to several runner processes at
    once, and each one's ``setUpClass`` starts its *own* head node — a bare
    ``ray.init()`` never joins an existing local cluster. Three of those booting
    together on a 3-core runner, alongside the AQC tests already saturating those
    cores, overloads the GCS and node startup times out. One id means one
    cluster; ``subTest`` keeps the scenarios reported separately.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Start the cluster here rather than letting `_run_parallel` do it, so the
        # workers get a runtime environment that can import the function package.
        # The dashboard is startup cost for something no test ever reads.
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            runtime_env={"env_vars": {"PYTHONPATH": _WORKER_PYTHONPATH}},
        )

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()
        super().tearDownClass()

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


@unittest.skipUnless(HAS_RAY, "ray is required for the parallel execution path")
class TestParallelSimNotFannedOut(unittest.TestCase):
    """`parallel_sim=True` requests that never reach Ray, so no cluster is needed."""

    def test_single_pub_not_fanned_out(self):
        """A single PUB is never parallelised, even with parallel_sim=True."""
        n = 4
        pubs = _pubs(n, 1)
        out = run_pubs(pubs, ExecutionOptions(backend="statevector", parallel_sim=True))
        self.assertEqual(out.shape, (1, n))
