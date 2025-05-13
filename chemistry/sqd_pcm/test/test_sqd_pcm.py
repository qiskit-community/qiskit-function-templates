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

import ray

from qiskit_ibm_runtime.fake_provider import FakeHanoiV2

from ..source_files.sqd_pcm import run_function
from .data import test_molecule


class TestSQDPCM(unittest.TestCase):
    """
    Test SQD PCM with a sample molecule
    """

    def setUp(self):
        super().setUp()

        # mimick ray setup in serverless cluster
        cwd = Path.cwd()
        ray.init(runtime_env={"working_dir": cwd / "chemistry/sqd_pcm/source_files"})

        self.count_dict_name = cwd / "test/data/water_mini_count_dict.txt"
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
        # Review testing tolerance (high because of result variability)
        self.assertTrue(out["sci_solver_total_duration"] < 6)
        self.assertTrue(out["lowest_energy_value"] < -72)
