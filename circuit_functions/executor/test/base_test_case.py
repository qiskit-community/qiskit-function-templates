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

from qiskit_ibm_runtime import SamplerOptions
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

import executor_entrypoint

from .utils import FakeExecutor


def _detect_prepare_broken():
    """Return (broken, reason). qiskit-ibm-runtime 0.50.0 ships a broken
    sampler_option_to_executor_options() (calls .model_dump() on options that lack
    it, and copies an Unset max_execution_time), so prepare() — and therefore the
    whole run_function path — cannot execute. Probe it once so the end-to-end tests
    skip with a clear reason on an affected runtime and re-enable automatically once
    a fixed release is installed."""
    try:
        from qiskit_ibm_runtime.options_models.converters import (
            sampler_option_to_executor_options,
        )

        sampler_option_to_executor_options(SamplerOptions())
    except ImportError:
        return False, ""  # older/newer layout; let the real tests run
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return True, (
            "qiskit-ibm-runtime 0.50.0 sampler->executor option converter is broken "
            f"({type(exc).__name__}: {exc}); the Executor sampler path cannot run. "
            "Re-enable once a fixed qiskit-ibm-runtime is released."
        )
    return False, ""


PREPARE_BROKEN, PREPARE_BROKEN_REASON = _detect_prepare_broken()


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
