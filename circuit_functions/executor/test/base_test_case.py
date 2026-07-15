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
Shared base test case: a FakeManilaV2 backend plus a patched-in FakeExecutor so run_function
executes end-to-end without a live backend.
"""

import unittest
from unittest import mock

from qiskit_ibm_runtime.fake_provider import FakeManilaV2

import executor_entrypoint

from .utils import FakeExecutor


class BaseExecutorTestCase(unittest.TestCase):
    """Base class for run_function-level tests.

    Provides a shared FakeManilaV2 backend as ``self._testing_backend`` and patches
    ``executor_entrypoint.Executor`` with ``FakeExecutor`` for the duration of each test, so
    the hardware submission is simulated locally with a seeded Aer sampler.
    """

    @classmethod
    def setUpClass(cls):
        cls._backend_name = FakeManilaV2().name
        cls._testing_backend = FakeManilaV2()
        return super().setUpClass()

    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(executor_entrypoint, "Executor", FakeExecutor)
        patcher.start()
        self.addCleanup(patcher.stop)
