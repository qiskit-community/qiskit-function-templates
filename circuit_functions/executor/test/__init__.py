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
Executor Function Template tests.

The entrypoint under ``source_files/`` imports its ``options`` sub-package and
``qiskit_serverless`` at module level. To keep the unit tests hermetic (no live serverless
cluster, no credentials) this package initializer:

1. puts ``source_files/`` on ``sys.path`` so the entrypoint's bare imports
   (``from executor_entrypoint import ...``, ``from options import ...``) resolve exactly as
   they do when the artifact runs on a cluster, and
2. registers a stub ``qiskit_serverless`` package so ``get_logger``, ``update_status``,
   ``Job`` and friends resolve at import time without a running gateway.

The stub is installed before the test modules import the entrypoint, so it is in place
regardless of test-discovery order.

Note that the executor itself is never run locally in these tests: the runtime
``Executor`` does not support local mode, so ``FakeExecutor`` in ``utils`` stands in for the
hardware submission and runs the program through a seeded Aer sampler instead.
"""

import os
import sys
import types
from unittest import mock

# 1. Make source_files importable (mirrors the artifact's runtime sys.path).
_SOURCE_FILES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "source_files")
if _SOURCE_FILES not in sys.path:
    sys.path.insert(0, _SOURCE_FILES)


# 2. Stub qiskit_serverless so the entrypoint imports without a live cluster.
class _QiskitServerlessException(Exception):
    """Stub matching qiskit_serverless.exception.QiskitServerlessException."""


class _MockJobStatus:
    """Mirrors the Job status constants used by run_function."""

    OPTIMIZING_HARDWARE = "OPTIMIZING_HARDWARE"
    WAITING_QPU = "WAITING_QPU"
    EXECUTING_QPU = "EXECUTING_QPU"
    POST_PROCESSING = "POST_PROCESSING"


# Only stub for hermetic unit runs. When QISKIT_FUNCTION_NAME is set we are running the live
# e2e suite (see test/e2e/base_e2e_test_case.py), which needs the real qiskit_serverless SDK
# to reach a deployed function — stubbing it here would shadow ServerlessClient with a mock.
if not os.environ.get("QISKIT_FUNCTION_NAME") and (
    "qiskit_serverless" not in sys.modules
    or isinstance(sys.modules["qiskit_serverless"], mock.MagicMock)
):
    _qs_mock = mock.MagicMock()
    _qs_mock.__path__ = []  # marks it as a package to the import machinery
    _qs_mock.get_logger.return_value = mock.MagicMock()
    _qs_mock.Job = _MockJobStatus
    _qs_mock.update_status = mock.MagicMock()
    _qs_mock.get_runtime_service = mock.MagicMock()
    _qs_mock.get_arguments = mock.MagicMock(return_value={})
    _qs_mock.save_result = mock.MagicMock()
    _qs_mock.QiskitServerlessException = _QiskitServerlessException

    _qs_exception_mod = types.ModuleType("qiskit_serverless.exception")
    # pylint: disable=attribute-defined-outside-init
    _qs_exception_mod.QiskitServerlessException = _QiskitServerlessException

    sys.modules["qiskit_serverless"] = _qs_mock
    sys.modules["qiskit_serverless.exception"] = _qs_exception_mod
