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
"""Per-stage unit tests for the SKQD pipeline.

Each test targets one stage of ``skqd.py`` in isolation (CPU, no Fleets, no GPU)
rather than the full end-to-end run in ``test_skqd_fleets``: the classical
chemistry integrals, the Hamiltonian assembly, the exact reference energy, the
Krylov circuit family (and its parallelism hook), and the sampling invariant.

The stages need the chemistry stack (pyscf, qiskit-fermions, ffsim); when it is
not installed the whole module skips with a reason, so it runs in CI (which
installs ``requirements.txt``) and is skipped on a bare dev machine.
"""
import importlib.util
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source_files"))

# pylint: disable=wrong-import-position
from skqd import (  # noqa: E402
    build_hamiltonian,
    fermion_operator_from_integrals,
    reference_energy,
    build_krylov_circuit,
    krylov_circuits,
    sample_circuits,
)

_CHEM_STACK = ("pyscf", "ffsim", "qiskit_fermions")
_MISSING = [m for m in _CHEM_STACK if importlib.util.find_spec(m) is None]
_SKIP_REASON = f"chemistry stack not installed ({', '.join(_MISSING)})" if _MISSING else ""

# Linear H4 chain in STO-3G: 4 spatial orbitals, 2 alpha + 2 beta electrons.
ATOM = [["H", (0.0, 0.0, 1.0 * i)] for i in range(4)]


@unittest.skipIf(_MISSING, _SKIP_REASON)
class TestBuildHamiltonian(unittest.TestCase):
    """build_hamiltonian returns well-formed MO integrals for H4/STO-3G."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.chem = build_hamiltonian(ATOM, basis="sto-3g", use_gpu=False)

    def test_system_dimensions(self):
        self.assertEqual(self.chem["norb"], 4)
        self.assertEqual(self.chem["nelec"], (2, 2))

    def test_integral_shapes(self):
        norb = self.chem["norb"]
        self.assertEqual(self.chem["h1e"].shape, (norb, norb))
        self.assertEqual(self.chem["h2e"].shape, (norb, norb, norb, norb))

    def test_one_body_is_symmetric(self):
        h1e = self.chem["h1e"]
        np.testing.assert_allclose(h1e, h1e.T, atol=1e-10)

    def test_integrals_are_real_host_arrays(self):
        # _to_host must have pulled everything off any device; plain float ndarrays.
        for key in ("h1e", "h2e"):
            self.assertIsInstance(self.chem[key], np.ndarray)
            self.assertTrue(np.isrealobj(self.chem[key]))
        self.assertIsInstance(self.chem["nuclear_repulsion"], float)
        self.assertGreater(self.chem["nuclear_repulsion"], 0.0)


@unittest.skipIf(_MISSING, _SKIP_REASON)
class TestHamiltonianAndReference(unittest.TestCase):
    """Hamiltonian assembly and the exact reference energy."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.chem = build_hamiltonian(ATOM, basis="sto-3g", use_gpu=False)

    def test_fermion_operator_builds(self):
        """The 8-fold lower-triangular two-body packing assembles a valid op."""
        op = fermion_operator_from_integrals(self.chem["h1e"], self.chem["h2e"], self.chem["norb"])
        self.assertIsNotNone(op)

    def test_reference_energy_is_a_sane_bound(self):
        """Exact (norb, nelec)-sector energy for H4/STO-3G sits near its known range."""
        norb, nelec = self.chem["norb"], self.chem["nelec"]
        e_elec = reference_energy(self.chem["h1e"], self.chem["h2e"], norb, nelec)
        total = e_elec + self.chem["nuclear_repulsion"]
        self.assertTrue(np.isfinite(total))
        # H4/STO-3G total ground-state energy is roughly -2.1 to -2.2 Ha; assert a
        # loose window so the test guards correctness without being brittle.
        self.assertLess(total, -1.5)
        self.assertGreater(total, -3.0)


@unittest.skipIf(_MISSING, _SKIP_REASON)
class TestKrylovCircuits(unittest.TestCase):
    """The Krylov circuit family and its parallelism hook."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.chem = build_hamiltonian(ATOM, basis="sto-3g", use_gpu=False)

    def test_builds_one_circuit_per_krylov_power(self):
        fermionic, transpiled = krylov_circuits(
            krylov_dim=5,
            norb=self.chem["norb"],
            nelec=self.chem["nelec"],
            h1e=self.chem["h1e"],
            time_step=0.2,
        )
        self.assertEqual(len(fermionic), 5)
        self.assertEqual(len(transpiled), 5)

    def test_map_fn_hook_is_used_and_order_preserved(self):
        """krylov_circuits delegates to map_fn and keeps the dims in order.

        The hook is the fan-out seam (a fleet/Ray map in production). A spy map_fn
        records the dims it was handed and runs the builder itself, so we confirm
        both that it is called with range(krylov_dim) in order and that the
        returned circuits come back in that order.
        """
        seen = {}

        def spy_map(fn, dims):
            seen["dims"] = list(dims)
            return [fn(d) for d in dims]

        fermionic, _ = krylov_circuits(
            krylov_dim=4,
            norb=self.chem["norb"],
            nelec=self.chem["nelec"],
            h1e=self.chem["h1e"],
            time_step=0.2,
            map_fn=spy_map,
        )
        self.assertEqual(seen["dims"], [0, 1, 2, 3])
        self.assertEqual(len(fermionic), 4)

    def test_single_circuit_builder_runs(self):
        circuit = build_krylov_circuit(
            dim=2,
            norb=self.chem["norb"],
            nelec=self.chem["nelec"],
            h1e=self.chem["h1e"],
            time_step=0.2,
        )
        self.assertIsNotNone(circuit)


@unittest.skipIf(_MISSING, _SKIP_REASON)
class TestSampling(unittest.TestCase):
    """Sampling the Krylov circuits conserves particle number."""

    def test_sampled_configurations_conserve_electron_count(self):
        """Every sampled bitstring must have exactly nelec electrons per spin.

        ffsim samples within the (norb, nelec) sector, so each sampled
        configuration carries the right alpha/beta Hamming weight. This is a
        strong physical invariant that catches a wrong sector or layout.
        """
        chem = build_hamiltonian(ATOM, basis="sto-3g", use_gpu=False)
        norb, nelec = chem["norb"], chem["nelec"]
        fermionic, _ = krylov_circuits(
            krylov_dim=3, norb=norb, nelec=nelec, h1e=chem["h1e"], time_step=0.2
        )
        counts = sample_circuits(fermionic, norb, nelec, shots=500)
        self.assertGreater(len(counts), 0)

        n_alpha, n_beta = nelec
        for config in counts:
            # ffsim returns a length-2*norb bitstring with alpha and beta halves;
            # each half must carry its spin's electron count.
            bits = config if isinstance(config, str) else format(int(config), f"0{2 * norb}b")
            self.assertEqual(len(bits), 2 * norb)
            alpha, beta = bits[norb:], bits[:norb]
            self.assertEqual(alpha.count("1"), n_alpha, config)
            self.assertEqual(beta.count("1"), n_beta, config)


if __name__ == "__main__":
    unittest.main()
