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
Hamiltonian Simulation Function Template unit tests.
"""

import unittest

from qiskit_ibm_runtime.fake_provider import FakeHanoiV2

from ..source_files.sqd_pcm import run_function


class TestSqdPcm(unittest.TestCase):
    """
    Test SQD PCM
    """

    def setUp(self):
        super().setUp()
        self.backend_name = "test_eagle_eu-de"
        self.datafiles_name = "methanol"

        # pylint: disable=consider-using-f-string
        self.molecule = {
            "atom": """
            O -0.04559 -0.75076 -0.00000;
            C -0.04844 0.65398 -0.00000;
            H 0.85330 -1.05128 -0.00000;
            H -1.08779 0.98076 -0.00000;
            H 0.44171 1.06337 0.88811;
            H 0.44171 1.06337 -0.88811
            """,
            "basis": "cc-pvdz",
            "spin": 0,
            "charge": 0,
            "verbosity": 0,
            "number_of_active_orb": 12,
            "number_of_active_alpha_elec": 7,
            "number_of_active_beta_elec": 7,
            "avas_selection": ["%d O %s" % (k, x) for k in [0] for x in ["2s", "2px", "2py", "2pz"]]
            + ["%d C %s" % (k, x) for k in [1] for x in ["2s", "2px", "2py", "2pz"]]
            + ["%d H 1s" % k for k in [2, 3, 4, 5]],
        }

        self.solvent_options = {
            "method": "IEF-PCM",
            "eps": 78.3553,  # value for water
        }

        self.lucj_options = {
            "dynamical_decoupling": True,
            "twirling": True,
            "number_of_shots": 200000,
            "optimization_level": 2,
        }

        self.sqd_options = {
            "sqd_iterations": 3,
            "number_of_batches": 10,
            "samples_per_batch": 1000,
            "max_davidson_cycles": 200,
        }

    def test_run(self):
        """Test run_function"""

        out = run_function(
            backend_name=self.backend_name,
            files_name=self.datafiles_name,
            molecule=self.molecule,
            solvent_options=self.solvent_options,
            lucj_options=self.lucj_options,
            sqd_options=self.sqd_options,
            testing_backend=FakeHanoiV2(),
        )

        print("OUTPUT:", out)
