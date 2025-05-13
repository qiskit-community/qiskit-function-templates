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
Methylamine molecule data for testing.
"""

FILE_NAME = "water_mini"

MOLECULE = {
    "atom": """
     O      0.00000       -0.00000        0.11641;
     H      0.00000        0.75141       -0.46564;
     H     -0.00000       -0.75141       -0.46564
    """,  # Must be specified
    "basis": "6-31G*",  # default is "sto-3g"
    "spin": 0,  # default is 0
    "charge": 0,  # default is 0
    "verbosity": 0,  # default is 0
    "number_of_active_orb": 4,  # Must be specified
    "number_of_active_alpha_elec": 2,  # Must be specified
    "number_of_active_beta_elec": 2,  # Must be specified
    "avas_selection": None,
}

SOLVENT = {
    "method": "IEF-PCM",  # other available methods are COSMO, C-PCM, SS(V)PE, see https://manual.q-chem.com/5.4/topic_pcm-em.html
    "eps": 78.3553,  # value for water
}

SQD = {
    "sqd_iterations": 1,
    "number_of_batches": 2,
    "samples_per_batch": 2,
    "max_davidson_cycles": 2,
}
