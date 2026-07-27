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
Base test case for the end-to-end tests against a live serverless cluster.
Base test case for the end-to-end tests against IBM Quantum Platform serverless.
"""

import os
import unittest

from qiskit_serverless import ServerlessClient

# python-dotenv is an optional local-dev convenience (loads a .env file). It is
# not a declared test dependency, so fall back to a no-op when it is absent —
# the e2e tests are skipped anyway unless the required env vars are set.
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover

    def load_dotenv(*_args, **_kwargs):
        """No-op fallback when python-dotenv is not installed."""
        return False


load_dotenv()

# The suite is skipped unless QISKIT_FUNCTION_NAME is set. Point it at a function you have
# deployed to IBM Quantum Platform (see the deploy-and-run notebook) with these env vars:
#
#   QISKIT_IBM_TOKEN     your IBM Quantum Platform API key
#   QISKIT_IBM_CRN       your instance CRN (crn:v1:...)
#   QISKIT_FUNCTION_NAME e.g. circuit_function_template
#   QISKIT_IBM_BACKEND   e.g. ibm_kingston
#
# The template run_function returns a dict:
#   {"hw_results": PrimitiveResult, "metadata": {...}}
#
# so job.result() gives that dict, not a bare PrimitiveResult. Every e2e test
# therefore accesses result["hw_results"] for pub data and result["metadata"]
# for timing info — mirroring the unit tests.


class BaseE2eTestCase(unittest.TestCase):
    """Base class for end-to-end tests against IBM Quantum Platform serverless.

    These tests require a deployed function and platform credentials. When
    ``QISKIT_FUNCTION_NAME`` is not set in the environment (the default in CI
    and for ``tox -ecircuit-vanilla``), the whole class is skipped so the
    unit-test run stays green without credentials.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Connect to IBM Quantum Platform and load the deployed function, or skip if unconfigured."""
        if not os.environ.get("QISKIT_FUNCTION_NAME"):
            raise unittest.SkipTest(
                "E2e tests require a function deployed to IBM Quantum Platform. Set "
                "QISKIT_IBM_TOKEN, QISKIT_IBM_CRN, QISKIT_FUNCTION_NAME and QISKIT_IBM_BACKEND "
                "(or add them to a .env file in the template directory) to run them."
            )

        # qiskit_ibm_catalog authenticates to IBM Quantum Platform with an API token and an
        # instance CRN, and loads the deployed function with .load(). Imported here (not at
        # module scope) so the unit-test run — which skips this class — never needs the SDK.
        # pylint: disable=import-outside-toplevel
        from qiskit_ibm_catalog import QiskitServerless

        cls._client = QiskitServerless(
            channel="ibm_quantum_platform",
            token=os.environ.get("QISKIT_IBM_TOKEN"),
            instance=os.environ.get("QISKIT_IBM_CRN"),
        )
        cls._func = cls._client.load(os.environ["QISKIT_FUNCTION_NAME"])
        cls._backend_name = os.environ["QISKIT_IBM_BACKEND"]

        return super().setUpClass()
