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
"""
Shared base test case providing a FakeManilaV2 backend for the unit tests.
"""

import unittest

from qiskit_ibm_runtime.fake_provider import FakeManilaV2


class BaseTemplateTestCase(unittest.TestCase):
    """Base class for template unit tests.

    Provides a shared FakeManilaV2 backend as ``self._testing_backend``,
    mirroring how ``BaseLocalTestCase`` in the circuit-function repo sets up
    ``LOCAL_TESTING`` mode for its ``CircuitFunction``-based tests.
    """

    @classmethod
    def setUpClass(cls):
        cls._backend_name = FakeManilaV2().name
        cls._testing_backend = FakeManilaV2()
        return super().setUpClass()
