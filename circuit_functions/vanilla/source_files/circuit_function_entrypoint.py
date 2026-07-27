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
Generic Circuit Function Template source code.
"""

from collections.abc import Iterable, Mapping, Sequence

import time
import traceback

import numpy as np

from qiskit.circuit import ControlFlowOp
from qiskit.primitives.containers import EstimatorPubLike
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.providers import BackendV2
from qiskit.primitives.containers.observables_array import ObservableLike
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager

from options import Options, DEFAULT_MITIGATION_LEVEL, DEFAULT_OPTIMIZATION_LEVEL
from options.utils import merge_options
from typing import Union

from qiskit_ibm_runtime import EstimatorV2
from qiskit_ibm_runtime.exceptions import IBMInputValueError

from qiskit_serverless import (
    get_arguments,
    save_result,
    update_status,
    Job,
    get_runtime_service,
    get_logger,
)

logger = get_logger()

# Job tag for function
ESTIMATOR_JOB_TAG = "ibm/circuit-function-template"


# Helper functions
def validate_estimator_pubs(pubs: list[EstimatorPub]) -> None:
    """Validates the estimator pubs won't cause problems that can be caught client-side.

    Args:
        pubs: The list of pubs to validate

    Raises:
        ValueError: If any observable array is of size 0
    """

    for pub in pubs:
        if pub.observables.shape == (0,):
            raise ValueError("Empty observables array is not allowed")


def are_circuits_dynamic(pubs: Iterable[EstimatorPubLike]) -> bool:
    """Checks if the input pubs have dynamic circuits.
    Args:
        pubs: The list of pubs to validate.
    """

    for pub in pubs:
        circuit = pub.circuit
        for inst in circuit.data:
            if (
                isinstance(inst.operation, ControlFlowOp)
                or getattr(inst.operation, "condition", None) is not None
            ):
                return True
    return False


def validate_backend_size(pubs: list[EstimatorPub], backend: BackendV2) -> None:
    """Validates that the backend has sufficient number of qubits
     to accommodate each of the estimator pub separately

     Args:
        pubs: The list of pubs to validate
        backend: The backend to validate against

    Raises:
        IBMInputValueError: If any circuit is bigger than the number of qubits in the backend
    """
    for pub in pubs:
        if int(pub.circuit.num_qubits) > int(backend.num_qubits):
            raise IBMInputValueError(
                f"Circuit has {pub.circuit.num_qubits} qubits, which is greater than what the backend {backend.name} supports {backend.num_qubits}."
            )


def _convert_observable(observable: ObservableLike) -> Union[Pauli, SparsePauliOp]:
    """Convert observable to either a Pauli or SparsePauliOp."""
    if isinstance(observable, (Pauli, SparsePauliOp)):
        return observable
    if isinstance(observable, Mapping):
        return SparsePauliOp(list(observable.keys()), coeffs=list(observable.values()))
    return SparsePauliOp(observable)


# The function that actually runs the circuit
def run_function(
    backend_name: str,
    pubs: Iterable[EstimatorPubLike],
    options: dict | None = None,
    instance: str | None = None,
    **kwargs,
) -> dict:
    """
    Entry point to the Generic Circuit Function.

    Optimize, execute, and post-process estimator PUBs on hardware. Note that
    this is a template, thus we only show minimal input arguments so that they
    can be extended to fit custom implementations.

    Args:
        backend_name: Name of the backend to use.
        pubs: An iterable of pub-like objects that can be tuples such as
        ``(circuit, observables)`` or ``(circuit, observables, params)``.
        options: Input options. Recognized keys: ``transpilation_options`` (dict,
        forwarded to ``generate_preset_pass_manager``) and ``estimator_options``
        (dict, ``EstimatorV2`` options merged over defaults).
        instance: The instance to use

    Returns:
        A dictionary of results from hardware and metadata

    Raises:
        ValueError: If input arguments are invalid.
    """
    logger.info("Input options are: %s", options)
    logger.info("Backend used: %s", backend_name)
    logger.info("Instance used: %s", instance)

    # Parse kwargs for local testing
    dry_run = kwargs.get("dry_run", False)
    testing_backend = kwargs.get("testing_backend", None)

    # Preparation (validate and set)
    if not pubs:
        raise ValueError("At least one PUB needed.")
    coerced_pubs = [EstimatorPub.coerce(pub) for pub in pubs]
    validate_estimator_pubs(coerced_pubs)

    # Reject dynamic circuits
    # TODO: Check if they are still incompatible
    if are_circuits_dynamic(coerced_pubs):
        raise ValueError("Dynamic circuits are not supported.")

    # Validate and set input options. User provided options take precedence
    # over the defaults.
    options = options or {}
    applied_mit = Options.apply_mitigation_level(
        mitigation_level=options.get("mitigation_level", DEFAULT_MITIGATION_LEVEL)
    )
    combined = merge_options(applied_mit, options)

    # Validate the options are correct
    options_parsed: Options = Options(**combined)
    # Extract options for each step.
    options_dict = options_parsed.model_dump(exclude_unset=True)
    transpilation_options = Options.get_transpilation_options(options_dict)
    estimator_options = Options.get_estimator_options(options_dict)
    estimator_options = merge_options(
        estimator_options, {"environment": {"job_tags": [ESTIMATOR_JOB_TAG]}}
    )

    # Validate and set input backend
    if not backend_name:
        raise ValueError(f"Invalid backend name value {backend_name}")
    if testing_backend is None:
        # Initialize Qiskit Runtime Service
        logger.info("Starting runtime service")
        service = get_runtime_service()
        backend = service.backend(backend_name)
        logger.info(f"Backend: {backend.name}")
    else:
        backend = testing_backend
        logger.info(f"Testing backend: {backend.name}")

    # Validate whether or not the backend has enough qubits
    validate_backend_size(coerced_pubs, backend)

    output = {}

    # Step 1: Optimize
    # Transpile PUBs to match ISA
    start_optimizing = time.time()
    update_status(Job.OPTIMIZING_HARDWARE)

    circuits = [pub.circuit for pub in coerced_pubs]
    all_pubs_params = [pub.parameter_values.as_array() for pub in coerced_pubs]
    transpilation_options.setdefault("optimization_level", DEFAULT_OPTIMIZATION_LEVEL)
    pass_manager = generate_preset_pass_manager(backend=backend, **transpilation_options)
    isa_circuits = pass_manager.run(circuits)
    if not isinstance(isa_circuits, Sequence):
        isa_circuits = [isa_circuits]

    isa_pubs = []
    for isa_circ, pub, params in zip(isa_circuits, coerced_pubs, all_pubs_params):
        isa_observables = np.array(pub.observables, copy=True)
        for ndi, obs in np.ndenumerate(isa_observables):
            isa_obs = _convert_observable(obs).apply_layout(isa_circ.layout)
            isa_observables[ndi] = isa_obs
        isa_pub = (isa_circ, isa_observables, params, pub.precision)
        isa_pubs.append(EstimatorPub.coerce(isa_pub))

    end_optimizing = time.time()
    output["metadata"] = {
        "resources_usage": {
            "RUNNING: OPTIMIZING_FOR_HARDWARE": {"CPU_TIME": end_optimizing - start_optimizing}
        }
    }

    # Exit if dry run--don't take it to hardware
    if dry_run:
        logger.info("Exiting before Hardware")
        return output

    # Step 2: Execute on Hardware
    estimator = EstimatorV2(mode=backend, options=estimator_options)
    start_waiting_qpu = time.time()
    job = estimator.run(pubs=isa_pubs)
    logger.info("Qiskit runtime job %s", job.job_id())

    # Report job status
    while job.status() == "QUEUED":
        update_status(Job.WAITING_QPU)
        time.sleep(5)

    end_waiting_qpu = time.time()
    update_status(Job.EXECUTING_QPU)
    hw_results = job.result()
    end_executing_qpu = time.time()

    # Step 3: Post-processing
    start_pp = time.time()
    update_status(Job.POST_PROCESSING)
    end_pp = time.time()

    output["hw_results"] = hw_results
    output["metadata"]["resources_usage"]["RUNNING: WAITING_FOR_QPU"] = {
        "CPU_TIME": end_waiting_qpu - start_waiting_qpu,
    }
    output["metadata"]["resources_usage"]["RUNNING: EXECUTING_QPU"] = {
        "CPU_TIME": end_executing_qpu - end_waiting_qpu,
    }
    output["metadata"]["resources_usage"]["RUNNING: POST_PROCESSING"] = {
        "CPU_TIME": end_pp - start_pp,
    }
    return output


# run_function gets called here, meant to be boilerplate and used w/o customization
if __name__ == "__main__":
    input_args = get_arguments()

    try:
        func_result = run_function(**input_args)
        # Use the provided serverless function
        save_result(func_result)
    except Exception:
        save_result(traceback.format_exc())
        raise
