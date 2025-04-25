# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Circuit Function Template source code.
"""
from __future__ import annotations

from collections.abc import Iterable
import logging
import os
import traceback

import numpy as np

from qiskit.primitives.containers import EstimatorPubLike, PrimitiveResult
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.transpiler import generate_preset_pass_manager

from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
from qiskit_ibm_runtime.runtime_job_v2 import RuntimeJobV2

from qiskit_serverless import get_arguments, save_result

logger = logging.getLogger(__name__)


class CircuitFunction:
    """
    Circuit Function Template
    """

    def __init__(
        self,
        backend_name: str,
        pubs: Iterable[EstimatorPubLike],
        options: dict | None = None,
        instance: str | None = None,
    ) -> None:
        """This class encapsulates the initialization and execution of the circuit function.
        The template shows minimal input arguments that can be extended to fit custom implementations.

        Args:
            backend_name: Name of the backend to use.
            pubs: An iterable of pub-like (primitive unified bloc) objects, such as
                tuples ``(circuit, observables)`` or ``(circuit, observables, parameter_values)``.
            options: Input options.
            instance: The instance to use.

        Raises:
            ValueError: If input arguments are invalid.
        """
        logger.info("Input options are %s", options)
        logger.info("Backend used: %s", backend_name)
        logger.info("Instance used: %s", instance)

        self._jobs: list[RuntimeJobV2] = []

        # Validate and set input pubs
        if not pubs:
            raise ValueError("At least one PUB is required.")
        self._pubs = [EstimatorPub.coerce(pub) for pub in pubs]
        # Run pub validation: validate_estimator_pubs(self._pubs)

        # Validate and set input options
        self._set_options(options)

        # Validate and set input backend
        if not backend_name:
            raise ValueError(f"Invalid backend_name value {backend_name}.")
        if os.environ.get("LOCAL_TESTING", "false").lower() == "true":
            self._service = QiskitRuntimeService(channel="local")
            self._backend = self._service.backend(backend_name)
        else:
            self._service = QiskitRuntimeService(instance=instance)
            self._backend = self._service.backend(backend_name)

    def _set_options(self, options: dict | None = None) -> None:
        """Set options from the input."""
        options = options or {}
        self._transpilation_options = options.get("transpilation_options", None)
        self._execution_options = options.get("execution_options", None)
        # Here, additional options such as error mitigation options could be set

    def run(self) -> PrimitiveResult:
        """Execute the request."""

        # The circuit function encapsulates steps 2-4 of the Qiskit Pattern workflow:
        # https://docs.quantum.ibm.com/guides/intro-to-patterns

        # --
        # Step 2: Optimize
        # Transpile PUBs (circuits and observables) to match ISA
        circuits = [pub.circuit for pub in self._pubs]
        all_pubs_params = [pub.parameter_values.as_array() for pub in self._pubs]
        pass_manager = generate_preset_pass_manager(
            backend=self._backend,
            seed_transpiler=self._transpilation_options.get("seed_transpiler", None),
        )
        isa_circuits = pass_manager.run(circuits)

        isa_pubs = []

        for isa_circ, pub, params in zip(isa_circuits, self._pubs, all_pubs_params):
            isa_observables = np.array(pub.observables, copy=True)
            for ndi, obs in np.ndenumerate(isa_observables):
                isa_obs = obs.apply_layout(isa_circ.layout)
                isa_observables[ndi] = isa_obs
            isa_pub = (isa_circ, isa_observables, params, pub.precision)
            isa_pubs.append(EstimatorPub.coerce(isa_pub))

        # --
        # Step 3: Execute on Hardware
        # Initialize EstimatorV2
        estimator = EstimatorV2(mode=self._backend, options=self._execution_options)

        # Run
        job = estimator.run(pubs=isa_pubs)
        self._jobs.append(job)
        logger.info("Qiskit Runtime job %s submitted.", job.job_id())

        # In this case, the result is returned directly without post-processing,
        # but additional post-processing could be performed at this step
        # (Step 4 of the Qiskit Pattern)
        return job.result()


# This is the section where `CircuitFunction` is initialized and ran, it's
# boilerplate code and can be used without customization.
if __name__ == "__main__":
    # Use serverless helper function to extract input arguments,
    input_args = get_arguments()
    try:
        func = CircuitFunction(**input_args)
        # Use serverless function to save the results that
        # will be returned in the job.
        result = func.run()
        save_result(result)
    except Exception:
        save_result(traceback.format_exc())
        raise
