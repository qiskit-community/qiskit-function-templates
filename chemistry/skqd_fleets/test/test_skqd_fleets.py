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
"""Tests for the SKQD Fleets Function Template.

Runs the SKQD pipeline locally on CPU (no Fleets, no GPU) for a small molecule
and checks that the recovered energy is a sane variational upper bound near the
exact reference. The entrypoint's ``source_files`` dir is added to the path so
the ``skqd`` module imports the same way it does inside the function container.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source_files"))

# pylint: disable=wrong-import-position
from skqd import (  # noqa: E402
    build_hamiltonian,
    reference_energy,
    krylov_circuits,
    sample_circuits,
    diagonalize,
)

# Linear H4 chain in STO-3G: small enough for a fast CPU test, correlated enough
# that SKQD is meaningful.
ATOM = [["H", (0.0, 0.0, 1.0 * i)] for i in range(4)]


class TestSKQDFleets(unittest.TestCase):
    """Smoke test of the SKQD pipeline on CPU."""

    def test_h4_energy_is_variational_upper_bound(self):
        """The SKQD energy should sit at or above the exact reference."""
        chem = build_hamiltonian(ATOM, basis="sto-3g", use_gpu=False)
        norb, nelec = chem["norb"], chem["nelec"]
        h1e, h2e, e_nuc = chem["h1e"], chem["h2e"], chem["nuclear_repulsion"]

        self.assertEqual(norb, 4)
        self.assertEqual(nelec, (2, 2))

        ref = reference_energy(h1e, h2e, norb, nelec) + e_nuc

        fermionic_circuits, _ = krylov_circuits(
            krylov_dim=5, norb=norb, nelec=nelec, h1e=h1e, time_step=0.2
        )
        counts = sample_circuits(fermionic_circuits, norb, nelec, shots=2000)
        self.assertGreater(len(counts), 0)

        result, _history = diagonalize(
            counts, h1e, h2e, norb, nelec, num_batches=3, max_iterations=3
        )
        energy = float(result.energy) + e_nuc

        # SQD is variational: energy >= exact reference (allow a small tolerance
        # for numerical noise).
        self.assertGreaterEqual(energy, ref - 1e-6)
        # And it should be in the right ballpark, not wildly off.
        self.assertLess(abs(energy - ref), 0.5)


if __name__ == "__main__":
    unittest.main()
