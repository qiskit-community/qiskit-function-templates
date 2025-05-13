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

from source_files.sqd_pcm import run_function
from .data import methylamine


class TestMethylamine(unittest.TestCase):
    """
    Test SQD PCM with methylamine molecule
    """

    def setUp(self):
        super().setUp()

        # mimick ray setup in serverless cluster
        cwd = Path.cwd()
        ray.init(runtime_env={"working_dir": cwd / "source_files"})

        self.count_dict_name = cwd / "test/data/methylamine_count_dict.txt"
        self.backend_name = None
        self.datafiles_name = methylamine.FILE_NAME
        self.molecule = methylamine.MOLECULE
        self.solvent_options = methylamine.SOLVENT
        self.sqd_options = {
            "sqd_iterations": 1,
            "number_of_batches": 2,
            "samples_per_batch": 2,
            "max_davidson_cycles": 2,
        }

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
        self.assertTrue(out["sci_solver_total_duration"] < 30)
        self.assertTrue(out["lowest_energy_value"] < -90)
