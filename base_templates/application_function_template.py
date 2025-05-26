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
Application Function Template source code.
"""
from typing import Any

import json
import logging
import traceback

from mergedeep import merge

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import EstimatorV2

from qiskit_serverless import get_arguments, save_result

logger = logging.getLogger(__name__)


def run_function(
    backend_name: str,
    arg_1: int,
    arg_2: dict,
    arg_3: bool | None,
    **kwargs,
) -> dict[str, Any]:
    """
    Main entry point for the workflow.

    This function encapsulates the end-to-end execution of the desired algorithm.
    Complex workflows should organize logic into well defined modules or classes
    that are then invoked within ``run_function``.

    Args:
        backend_name: Identifier for the target backend, required for all workflows that access
            IBM Quantum hardware.
        *args: Positional arguments specific to the workflow (in this example, a string, a
            dictionary and an optional boolean flag). The argument types must be serializable
            by qiskit-serverless.
        **kwargs: Optional keyword arguments to customize behavior (e.g., alternate paths
            for local testing). Use documented keywords where possible to maintain clarity.
            The argument types must be serializable by qiskit-serverless.

    Returns:
        The function should return the execution results as a dictionary with string keys.
        This is to ensure compatibility with ``qiskit_serverless.save_result``.
    """

    # Preparation Step: Input validation.
    # Do this at the top of the function definition so it fails early if any required
    # arguments are missing or invalid.

    # It's recommended to introduce nesting in the input arguments, this can be done using dictionaries:
    initial_state = arg_2.get("init_state", None)  # pylint: disable=unused-variable
    num_iter = arg_2.get("num_iter", 0)
    if num_iter < 0:
        raise ValueError(f"Incorrect number of iterations: {num_iter}, it must be greater than 0.")

    # Further dummy input handling
    if arg_3:
        num_qubits = arg_1
    else:
        num_qubits = 100

    # In this example, we contemplate two keyword arguments that can be defined for local testing:
    dry_run = kwargs.get("dry_run", False)
    testing_backend = kwargs.get("testing_backend", None)

    # --
    # Preparation Step: Qiskit Runtime & primitive configuration for execution on IBM Quantum Hardware.
    # In this example, the Qiskit Runtime Service initialization is skipped in
    # local testing mode, where we use the provided testing backend instead.
    # This testing backend will be a `FakeBackend`:
    # https://docs.quantum.ibm.com/migration-guides/local-simulators#fake-backends

    if testing_backend is None:
        # Initialize Qiskit Runtime Service
        logger.info("Starting runtime service")
        service = QiskitRuntimeService(channel="ibm_quantum")
        backend = service.backend(backend_name)
        logger.info("backend", backend)
    else:
        backend = testing_backend

    # The primitive of choice is the `EstimatorV2`:
    # https://docs.quantum.ibm.com/guides/get-started-with-primitives#get-started-with-primitives
    # You can configure different `EstimatorOptions` to control the parameters of the hardware
    # experiment.

    # This example sets default options:
    estimator_default_options = {
        # Add job tag to be able to track function usage
        "environment": {"job_tags": ["my_function"]},
        "resilience": {
            "measure_mitigation": True,
            "measure_noise_learning": {
                "num_randomizations": 512,
                "shots_per_randomization": 512,
            },
        },
        "twirling": {
            "enable_gates": True,
            "enable_measure": True,
            "num_randomizations": 300,
            "shots_per_randomization": 100,
            "strategy": "active",
        },
    }
    # And then merges them with user-provided options:
    estimator_options = merge(kwargs.get("estimator_options", {}), estimator_default_options)

    # When the function template is running, it is helpful to return
    # information in the logs by using print statements, so that you can better
    # evaluate the workload's progress. This example returns the estimator options.
    logger.info("estimator_options =", json.dumps(estimator_options, indent=4))

    # The options are then used together with the backend to initialize the Estimator:
    estimator = EstimatorV2(backend, options=estimator_options)

    # --
    # Preparation Step: Prepare a dictionary to hold all of the function template outputs.
    # Keys will be added to this dictionary throughout the workflow,
    # and it is returned at the end of the program.
    output = {}

    # Once the preparation steps are completed, the algorithm can be structured following a
    # Qiskit Pattern workflow:
    # https://docs.quantum.ibm.com/guides/intro-to-patterns

    # --
    # Step 1: Map
    # In this step, input arguments are used to construct relevant quantum circuits and operators
    # This is a dummy example:
    ansatz = QuantumCircuit(num_qubits)  # dummy circuit
    ansatz.measure_all()
    observable = SparsePauliOp.from_list(["Z"] * num_qubits)

    # --
    # Step 2: Optimize
    # Transpile PUBs (circuits and observables) to match ISA
    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=2)
    isa_circuit = pass_manager.run(ansatz)
    isa_observable = observable.apply_layout(isa_circuit.layout)

    isa_2qubit_depth = isa_circuit.depth(lambda x: x.operation.num_qubits == 2)
    logger.info("ISA circuit two-qubit depth:", isa_2qubit_depth)
    output["twoqubit_depth"] = isa_2qubit_depth

    # Exit now if dry run; don't execute on hardware
    if dry_run:
        logger.info("Exiting before hardware execution since `dry_run` is True.")
        return output

    # --
    # Step 3: Execute on Hardware
    # Submit the underlying Estimator job. Note that this is not the
    # actual function job.
    job = estimator.run([(isa_circuit, isa_observable)])
    logger.info("Job ID:", job.job_id())
    output["job_id"] = job.job_id()

    # Wait until job is complete
    hw_results = job.result()
    hw_results_dicts = [pub_result.data.__dict__ for pub_result in hw_results]

    # Save hardware results to serverless output dictionary
    output["hw_results"] = hw_results_dicts

    # --
    # Step 4: Post-process
    # Reorganize expectation values
    hw_expvals = [pub_result_data["evs"].tolist() for pub_result_data in hw_results_dicts]

    # Return expectation values in serializable format
    logger.info("Hardware expectation values", hw_expvals)
    output["hw_expvals"] = hw_expvals[0]
    return output


def set_up_logger(my_logger: logging.Logger, level: int = logging.INFO) -> None:
    """Logger setup to communicate logs through serverless."""

    log_fmt = "%(module)s.%(funcName)s:%(levelname)s:%(asctime)s: %(message)s"
    formatter = logging.Formatter(log_fmt)

    # Set propagate to `False` since handlers are to be attached.
    my_logger.propagate = False

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    my_logger.addHandler(stream_handler)
    my_logger.setLevel(level)


# This is the section where `run_function` is called, it's boilerplate code and can be used
# without customization.
if __name__ == "__main__":
    # Use serverless helper function to extract input arguments,
    input_args = get_arguments()

    # Allow to configure logging level
    logging_level = input_args.get("logging_level", logging.INFO)
    set_up_logger(logger, logging_level)

    try:
        func_result = run_function(**input_args)
        # Use serverless function to save the results that
        # will be returned in the job.
        save_result(func_result)
    except Exception:
        save_result(traceback.format_exc())
        raise
