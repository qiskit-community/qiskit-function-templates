# This code is part of a Qiskit project.
#
# (C) Copyright IBM and Cleveland Clinic Foundation 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
SQD PCM Function Template unit tests.
"""

import unittest
from pathlib import Path

from qiskit_ibm_runtime.fake_provider import FakeHanoiV2

from ..source_files.sqd_pcm_entrypoint import run_function
from .data import test_molecule


class TestSQDPCM(unittest.TestCase):
    """
    Test SQD PCM with a sample molecule
    """

    def setUp(self):
        super().setUp()

        # The entrypoint no longer depends on Ray: on the Fleets runner it fans
        # the batches out across a local ProcessPoolExecutor, so the test can call
        # run_function directly without initializing a Ray cluster.
        cwd = Path.cwd()
        self.count_dict_name = cwd / "chemistry/sqd_pcm/test/data/water_mini_count_dict.txt"
        self.backend_name = None
        self.datafiles_name = test_molecule.FILE_NAME
        self.molecule = test_molecule.MOLECULE
        self.solvent_options = test_molecule.SOLVENT
        self.sqd_options = test_molecule.SQD

    def test_run(self):
        """Test run_function"""

        out = run_function(
            backend_name=self.backend_name,
            molecule=self.molecule,
            solvent_options=self.solvent_options,
            lucj_options={},
            sqd_options=self.sqd_options,
            testing_backend=FakeHanoiV2(),
            files_name=self.datafiles_name,
            count_dict_file_name=self.count_dict_name,
        )
        # Loose upper bound: this duration now includes ProcessPoolExecutor
        # startup, and under the "spawn" start method each worker cold-imports
        # PySCF, which dominates the runtime of this tiny (4-dim) problem. The
        # bound only guards against a hang/runaway, not solver speed.
        self.assertTrue(out["sci_solver_total_duration"] < 60)
        self.assertTrue(out["lowest_energy_value"] < -72)
        self.assertTrue(out["metadata"]["num_iterations_executed"] == 2)
