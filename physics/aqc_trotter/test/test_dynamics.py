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

"""End-to-end correctness: AQC-compressed dynamics must match exact Trotter.

Runs the function on the statevector backend and compares the returned observable
time series against the *same* Trotter circuits evaluated exactly (no AQC). At
small n / low entanglement the AQC fidelity is ~1, so the two agree to tight
tolerance.

``program.py`` calls ``DynamicsFunction(**arguments).run()`` directly, so that is
what we exercise here.
"""

import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from ..source_files.source import build as build_stage
from ..source_files.source.app_function import DynamicsFunction
from ..source_files.source.hamiltonian import make_spec


def _run(arguments: dict) -> dict:
    """Mirror what program.py does on the gateway: build + run, return the dict."""
    return DynamicsFunction(**arguments).run()


def _neel(n):
    """Néel product state as a circuit: X on even sites (|0101...> little-endian)."""
    qc = QuantumCircuit(n)
    for i in range(0, n, 2):
        qc.x(i)
    return qc


N = 6
DT = 0.2
T_STEPS = 6
AQC_T_STEPS = 3
TERMS = [["XX", 0.5], ["YY", 0.5], ["ZZ", 0.5]]
INIT = _neel(N)


def _heis(n):
    """Heisenberg chain on n qubits from TERMS (XX+YY+ZZ at 0.5 on every bond)."""
    return SparsePauliOp.from_sparse_list(
        [(p, [i, i + 1], c) for i in range(n - 1) for p, c in TERMS], num_qubits=n
    )


def _exact_series():
    """The same Trotter circuits evaluated exactly (no AQC), as an (t+1, N) array."""
    spec = make_spec("pauli_1d_nn", N, operator=_heis(N))
    init = build_stage.prepare_initial_state(N, INIT)
    obs, _ = build_stage.build_observables(N)
    rows = []
    for k in range(T_STEPS + 1):
        qc = init.copy()
        qc.compose(spec.build_evolution(k, DT), inplace=True)
        sv = Statevector(qc)
        rows.append([float(np.real(sv.expectation_value(o))) for o in obs])
    return np.array(rows)


class TestDynamics(unittest.TestCase):
    """One statevector run of the full pipeline, inspected from several angles."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # The run is expensive (AQC optimisation), so it is shared by every test
        # in this class — none of them mutate the result.
        cls.result = _run(
            {
                "t_steps": T_STEPS,
                "aqc_segments": [{"n_steps": AQC_T_STEPS, "ansatz_steps": 1}],
                "dt": DT,
                "hamiltonian": _heis(N),
                "initial_state": INIT,
                "backend": "statevector",
            }
        )

    def test_output_shape_and_times(self):
        """The result carries one row per time step and one column per observable."""
        self.assertEqual(self.result["times"], [k * DT for k in range(T_STEPS + 1)])
        ev = np.array(self.result["expectation_values"])
        self.assertEqual(ev.shape, (T_STEPS + 1, N))
        self.assertEqual(self.result["observable_labels"], [f"Z_{i}" for i in range(N)])
        self.assertTrue(np.all(np.isfinite(ev)))

    def test_aqc_fidelities_high(self):
        """Every compressed step reports a high MPS fidelity."""
        fids = self.result["metadata"]["aqc_fidelities"]
        self.assertEqual(len(fids), AQC_T_STEPS)
        self.assertTrue(all(f > 0.95 for f in fids.values()))

    def test_circuit_stats_reports_every_step_and_never_worse(self):
        """Per-step depth/gate counts are reported and compression never regresses."""
        stats = self.result["metadata"]["circuit_stats"]
        self.assertEqual(sorted(stats), list(range(1, T_STEPS + 1)))
        for step in stats.values():
            for side in ("full_trotter", "aqc_trotter"):
                self.assertEqual(set(step[side]), {"depth_2q", "num_2q_gates"})
        # Compression shouldn't ever look worse than plain Trotter at the final step.
        final = stats[T_STEPS]
        self.assertLessEqual(final["aqc_trotter"]["depth_2q"], final["full_trotter"]["depth_2q"])
        self.assertLessEqual(
            final["aqc_trotter"]["num_2q_gates"], final["full_trotter"]["num_2q_gates"]
        )

    def test_matches_exact_trotter(self):
        """The compressed series tracks the exactly-evaluated Trotter series."""
        ev = np.array(self.result["expectation_values"])
        self.assertLess(np.max(np.abs(ev - _exact_series())), 0.05)

    def test_initial_values_are_neel(self):
        """t=0 row: Néel state "010101" -> <Z_i> alternates +1/-1."""
        ev = np.array(self.result["expectation_values"])
        row0 = ev[0]
        self.assertTrue(np.all(np.abs(np.abs(row0) - 1.0) < 1e-6))
        # strictly alternating
        self.assertTrue(np.allclose(np.abs(np.diff(np.sign(row0))), 2.0))


class TestDynamicsVariants(unittest.TestCase):
    """Runs that vary the observables or the compression plan."""

    def test_custom_observable_runs(self):
        """Caller-supplied Pauli strings become the output columns."""
        res = _run(
            {
                "t_steps": 3,
                "aqc_segments": [{"n_steps": 1, "ansatz_steps": 1}],
                "dt": 0.2,
                "hamiltonian": _heis(4),
                "observables": ["IIIZ", "IIZZ"],  # Pauli strings: Z_0 and Z_0 Z_1
                "backend": "statevector",
            }
        )
        ev = np.array(res["expectation_values"])
        self.assertEqual(ev.shape, (4, 2))
        self.assertEqual(res["observable_labels"], ["IIIZ", "IIZZ"])

    def test_two_segment_matches_exact_and_reports_plan(self):
        """Multi-ansatz compression still matches exact Trotter and reports its plan."""
        # 3 steps into a 1-layer ansatz, then 2 into a 2-layer ansatz (5 of 6
        # steps); the tail is plain Trotter.
        res = _run(
            {
                "t_steps": T_STEPS,
                "aqc_segments": [
                    {"n_steps": 3, "ansatz_steps": 1},
                    {"n_steps": 2, "ansatz_steps": 2},
                ],
                "dt": DT,
                "hamiltonian": _heis(N),
                "initial_state": INIT,
                "backend": "statevector",
            }
        )
        ev = np.array(res["expectation_values"])
        self.assertEqual(ev.shape, (T_STEPS + 1, N))
        # metadata gives a per-segment breakdown: plan + step range + params + fidelities
        segs = res["metadata"]["aqc_segments"]
        self.assertEqual([(s["n_steps"], s["ansatz_steps"]) for s in segs], [(3, 1), (2, 2)])
        self.assertEqual(segs[0]["steps"], [1, 2, 3])
        self.assertEqual(segs[1]["steps"], [4, 5])
        self.assertTrue(all(isinstance(s["n_params"], int) and s["n_params"] > 0 for s in segs))
        self.assertEqual(set(segs[0]["fidelities"]), {1, 2, 3})
        self.assertEqual(set(segs[1]["fidelities"]), {4, 5})
        self.assertTrue(all(f > 0.95 for s in segs for f in s["fidelities"].values()))
        self.assertEqual(res["metadata"]["aqc_compressed_steps"], 5)
        # flat aqc_fidelities kept (full series over all 5 compressed steps)
        fids = res["metadata"]["aqc_fidelities"]
        self.assertEqual(len(fids), 5)
        self.assertTrue(all(f > 0.95 for f in fids.values()))
        # and the compressed dynamics still track exact Trotter
        self.assertLess(np.max(np.abs(ev - _exact_series())), 0.05)
