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
Shared helpers for the Executor Function Template tests: pub/input builders and a
``FakeExecutor`` test double that stands in for the runtime Executor (which cannot run
locally) by simulating the submitted quantum program with a seeded Aer sampler.
"""

from unittest import mock

from qiskit.circuit.random import random_circuit
from qiskit_aer.primitives import SamplerV2 as AerSampler


def get_sampler_pub() -> tuple:
    """Return a single sampler pub as a one-tuple ``(circuit,)`` with measurements."""
    circuit = random_circuit(num_qubits=2, depth=2, measure=True, seed=42)
    return (circuit,)


def get_estimator_pub() -> tuple:
    """Return a single estimator pub as a ``(circuit, observable)`` tuple (no measurements)."""
    circuit = random_circuit(num_qubits=2, depth=2, seed=42)
    observable = "Z" * circuit.num_qubits
    return (circuit, observable)


def get_inputs(backend_name=None, pubs=None, mode="sampler", options=None):
    """Build a run_function kwargs dict, filling sensible defaults for the given mode."""
    if pubs is None:
        pubs = [get_estimator_pub()] if mode == "estimator" else [get_sampler_pub()]
    return {
        "backend_name": backend_name or "fake_manila",
        "pubs": pubs,
        "mode": mode,
        "options": options,
    }


def dict_partially_equal(dict1: dict, dict2: dict) -> bool:
    """Return True if every key in dict2 exists in dict1 with the same value."""
    for key, val in dict2.items():
        if isinstance(val, dict):
            if not dict_partially_equal(dict1.get(key, {}), val):
                return False
        elif key not in dict1 or val != dict1[key]:
            return False
    return True


class _FakeJob:
    """Minimal RuntimeJobV2 stand-in wrapping an already-computed result."""

    def __init__(self, result):
        self._result = result

    def job_id(self) -> str:
        """Return a fixed fake job id."""
        return "fake-executor-job"

    def status(self) -> str:
        """Report the job as already complete."""
        return "DONE"

    def result(self):
        """Return the pre-computed result."""
        return self._result


class FakeExecutor:
    """Test double for qiskit_ibm_runtime.Executor.

    The real Executor cannot run in local mode, so this double simulates the submitted
    quantum program instead: each circuit item is run through a seeded Aer SamplerV2, which
    produces exactly the SamplerV2-style PrimitiveResult that the runtime's sampler_v2
    post-processor would return. Patch it over ``executor_entrypoint.Executor`` in tests.
    """

    # Fixed seed so numerical assertions are reproducible.
    seed = 1234

    def __init__(self, mode=None, options=None):  # pylint: disable=unused-argument
        # ``mode``/``options`` mirror the real Executor signature; the double ignores them.
        # ``options`` accepts arbitrary attribute assignment (job_tags, max_execution_time).
        self.options = mock.MagicMock()

    def run(self, program):
        """Simulate the program with a seeded Aer sampler and return a fake job.

        Only plain circuit items are supported. Twirling produces samplex (boxed-layer) items
        that the runtime executes on hardware; those cannot be simulated by a local sampler, so
        they are rejected here rather than silently mis-run.
        """
        pubs = []
        for item in program.items:
            if not hasattr(item, "circuit_arguments"):
                raise NotImplementedError(
                    "FakeExecutor only runs plain circuit items; twirling/samplex items "
                    "require real hardware."
                )
            pubs.append(
                (item.circuit,)
                if item.circuit_arguments is None
                else (item.circuit, item.circuit_arguments)
            )
        sampler = AerSampler(default_shots=program.shots, seed=self.seed)
        return _FakeJob(sampler.run(pubs).result())
