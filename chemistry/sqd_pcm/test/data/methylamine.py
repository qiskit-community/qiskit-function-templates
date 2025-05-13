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

FILE_NAME = "methylamine"

MOLECULE = {
    "atom": """
     N          -0.75263        0.00000       -0.14120;
     H          -1.13448       -0.79902        0.33988;
     H          -1.13448        0.79902        0.33988;
     C           0.69690        0.00000        0.00880;
     H           1.04449        0.00000        1.04915;
     H           1.11605       -0.88047       -0.48297;
     H           1.11605        0.88047       -0.48297
    """,  # Must be specified
    "basis": "cc-pvdz",  # default is "sto-3g"
    "spin": 0,  # default is 0
    "charge": 0,  # default is 0
    "verbosity": 0,  # default is 0
    "number_of_active_orb": 13,  # Must be specified
    "number_of_active_alpha_elec": 7,  # Must be specified
    "number_of_active_beta_elec": 7,  # Must be specified
    "avas_selection": ["%d N %s" % (k, x) for k in [0] for x in ["2s", "2px", "2py", "2pz"]]
    + ["%d C %s" % (k, x) for k in [3] for x in ["2s", "2px", "2py", "2pz"]]
    + ["%d H 1s" % k for k in [1, 2, 4, 5, 6]],
}

SOLVENT = {
    "method": "IEF-PCM",  # other available methods are COSMO, C-PCM, SS(V)PE, see https://manual.q-chem.com/5.4/topic_pcm-em.html
    "eps": 78.3553,  # value for water
}

LUCJ = {
    "initial_layout": [
        0,
        14,
        18,
        19,
        20,
        33,
        39,
        40,
        41,
        53,
        60,
        61,
        62,
        2,
        3,
        4,
        15,
        22,
        23,
        24,
        34,
        43,
        44,
        45,
        54,
        64,
    ],
    "dynamical_decoupling_choice": True,
    "twirling_choice": True,
    "number_of_shots": 200000,
    "optimization_level": 3,
}

SQD = {
    "sqd_iterations": 3,
    "number_of_batches": 10,
    "samples_per_batch": 1000,
    "max_davidson_cycles": 200,
}
