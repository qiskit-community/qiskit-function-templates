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
Hamiltonian Simulation Function Template source code.
"""

import os
import datetime
import json
import logging
import time
import traceback
import numpy as np

from mergedeep import merge

from scipy.optimize import OptimizeResult, minimize

from qiskit import QuantumCircuit
from qiskit.synthesis import SuzukiTrotter
from qiskit.transpiler import generate_preset_pass_manager

from qiskit_addon_utils.problem_generators import generate_time_evolution_circuit
from qiskit_addon_aqc_tensor.ansatz_generation import generate_ansatz_from_circuit
from qiskit_addon_aqc_tensor.simulation import (
    tensornetwork_from_circuit,
    compute_overlap,
)
from qiskit_addon_aqc_tensor.simulation.quimb import QuimbSimulator
from qiskit_addon_aqc_tensor.objective import OneMinusFidelity

from qiskit_ibm_runtime import EstimatorV2 as Estimator

from qiskit_serverless import get_arguments, save_result, update_status, Job, get_runtime_service

# this variable is required to import quimb.tensor
os.environ["NUMBA_CACHE_DIR"] = "/data"
import quimb.tensor  # pylint: disable=wrong-import-position

logger = logging.getLogger(__name__)


def run_function(
    backend_name,
    hamiltonian,
    observable,
    initial_state=None,
    aqc_options=None,
    estimator_options={},
    **kwargs,
):
    """
    Entry point to the Hamiltonian Simulation Function.
    """

    # Preparation Step: Input validation.
    # Do this at the top of the function definition so it fails early if any required
    # arguments are missing or invalid.

    if aqc_options is not None:
        aqc_evolution_time = aqc_options.get("aqc_evolution_time", None)
        aqc_ansatz_num_trotter_steps = aqc_options.get("aqc_ansatz_num_trotter_steps", None)
        aqc_target_num_trotter_steps = aqc_options.get("aqc_target_num_trotter_steps", None)
        remainder_evolution_time = aqc_options.get("remainder_evolution_time", None)
        remainder_num_trotter_steps = aqc_options.get("remainder_num_trotter_steps", None)

        # Stop if this fidelity is achieved
        aqc_stopping_fidelity = aqc_options.get("aqc_stopping_fidelity", 1.0)
        # Stop after this number of iterations, even if stopping fidelity is not achieved
        aqc_max_iterations = aqc_options.get("aqc_max_iterations", 500)
    else:
        aqc_evolution_time = None
        aqc_ansatz_num_trotter_steps = None
        aqc_target_num_trotter_steps = None
        remainder_evolution_time = None
        remainder_num_trotter_steps = None
        aqc_stopping_fidelity = 1.0
        aqc_max_iterations = 500

    if initial_state is None:
        initial_state = QuantumCircuit(hamiltonian.num_qubits)

    # Parse kwargs for local testing
    dry_run = kwargs.get("dry_run", False)
    testing_backend = kwargs.get("testing_backend", None)

    # Preparation Step: Qiskit Runtime & primitive configuration for execution on IBM Quantum Hardware.
    if testing_backend is None:
        # Initialize Qiskit Runtime Service
        logger.info("Starting runtime service")
        service = get_runtime_service()
        backend = service.backend(backend_name)
        logger.info(f"Backend: {backend.name}")
    else:
        backend = testing_backend
        logger.info(f"Testing backend: {backend.name}")

    # Configure `EstimatorOptions`, to control the parameters of the hardware
    # experiment.
    # Set default options.
    estimator_default_options = {
        "resilience": {
            "measure_mitigation": True,
            "zne_mitigation": True,
            "zne": {
                "amplifier": "gate_folding",
                "noise_factors": [1, 2, 3],
                "extrapolated_noise_factors": list(np.linspace(0, 3, 31)),
                "extrapolator": ["exponential", "linear", "fallback"],
            },
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
        # Add job tag to be able to track function usage
        "environment": {"job_tags": ["hamsim_function"]},
    }
    # Merge with user-provided options
    merged_estimator_options = merge(estimator_options, estimator_default_options)

    # When the function template is running, it is helpful to return
    # information in the logs by using print statements, so that you can better
    # evaluate the workload's progress. This example returns the estimator options.
    logger.info(f"estimator_options = {json.dumps(merged_estimator_options, indent=4)}")

    # Initialize Estimator with options
    estimator = Estimator(backend, options=merged_estimator_options)

    # Perform parameter validation
    if not 0.0 < aqc_stopping_fidelity <= 1.0:
        raise ValueError(
            f"Invalid stopping fidelity: {aqc_stopping_fidelity}. ",
            "It must be a positive float no greater than 1.",
        )

    # Preparation Step: Prepare a dictionary to hold all of the function template outputs.
    # Keys will be added to this dictionary throughout the workflow,
    # and it is returned at the end of the program.
    output = {}

    # Beginning of the Qiskit Pattern
    # Step 1: Map
    # In this step, input arguments are used to construct relevant quantum circuits and operators

    start_mapping = time.time()
    update_status(Job.MAPPING)

    logger.info(f"Hamiltonian: {hamiltonian}")
    logger.info(f"Observable: {observable}")
    simulator_settings = QuimbSimulator(quimb.tensor.CircuitMPS, autodiff_backend="jax")

    # Construct the AQC target circuit
    aqc_target_circuit = initial_state.copy()
    if aqc_evolution_time:
        aqc_target_circuit.compose(
            generate_time_evolution_circuit(
                hamiltonian,
                synthesis=SuzukiTrotter(reps=aqc_target_num_trotter_steps),
                time=aqc_evolution_time,
            ),
            inplace=True,
        )

    # Construct matrix-product state representation of the AQC target state
    aqc_target_mps = tensornetwork_from_circuit(aqc_target_circuit, simulator_settings)
    logger.info(f"Target MPS maximum bond dimension: {aqc_target_mps.psi.max_bond()}")
    output["target_bond_dimension"] = aqc_target_mps.psi.max_bond()

    # Generate an ansatz and initial parameters from a Trotter circuit with fewer steps
    aqc_good_circuit = initial_state.copy()
    if aqc_evolution_time:
        aqc_good_circuit.compose(
            generate_time_evolution_circuit(
                hamiltonian,
                synthesis=SuzukiTrotter(reps=aqc_ansatz_num_trotter_steps),
                time=aqc_evolution_time,
            ),
            inplace=True,
        )
    aqc_ansatz, aqc_initial_parameters = generate_ansatz_from_circuit(aqc_good_circuit)
    logger.info(f"Number of AQC parameters: {len(aqc_initial_parameters)}")
    output["num_aqc_parameters"] = len(aqc_initial_parameters)

    # Calculate the fidelity of ansatz circuit vs. the target state, before optimization
    good_mps = tensornetwork_from_circuit(aqc_good_circuit, simulator_settings)
    starting_fidelity = abs(compute_overlap(good_mps, aqc_target_mps)) ** 2
    logger.info(f"Starting fidelity of AQC portion: {starting_fidelity}")
    output["aqc_starting_fidelity"] = starting_fidelity

    # Optimize the ansatz parameters by using MPS calculations
    def callback(intermediate_result: OptimizeResult):
        fidelity = 1 - intermediate_result.fun
        logger.info(f"{datetime.datetime.now()} Intermediate result: Fidelity {fidelity:.8f}")
        if intermediate_result.fun < stopping_point:
            raise StopIteration

    objective = OneMinusFidelity(aqc_target_mps, aqc_ansatz, simulator_settings)
    stopping_point = 1.0 - aqc_stopping_fidelity

    result = minimize(
        objective,
        aqc_initial_parameters,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": aqc_max_iterations},
        callback=callback,
    )
    if result.status not in (
        0,
        1,
        99,
    ):  # 0 => success; 1 => max iterations reached; 99 => early termination via StopIteration
        raise RuntimeError(f"Optimization failed: {result.message} (status={result.status})")
    logger.info(f"Done after {result.nit} iterations")
    output["num_iterations"] = result.nit
    aqc_final_parameters = result.x
    output["aqc_final_parameters"] = list(aqc_final_parameters)

    # Construct an optimized circuit for initial portion of time evolution
    aqc_final_circuit = aqc_ansatz.assign_parameters(aqc_final_parameters)

    # Calculate fidelity after optimization
    aqc_final_mps = tensornetwork_from_circuit(aqc_final_circuit, simulator_settings)
    aqc_fidelity = abs(compute_overlap(aqc_final_mps, aqc_target_mps)) ** 2
    logger.info(f"Fidelity of AQC portion: {aqc_fidelity}")
    output["aqc_fidelity"] = aqc_fidelity

    # Construct final circuit, with remainder of time evolution
    final_circuit = aqc_final_circuit.copy()
    if remainder_evolution_time:
        remainder_circuit = generate_time_evolution_circuit(
            hamiltonian,
            synthesis=SuzukiTrotter(reps=remainder_num_trotter_steps),
            time=remainder_evolution_time,
        )
        final_circuit.compose(remainder_circuit, inplace=True)
    end_mapping = time.time()

    # Step 2: Optimize
    # Transpile PUBs (circuits and observables) to match ISA

    start_optimizing = time.time()
    update_status(Job.OPTIMIZING_HARDWARE)

    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa_circuit = pass_manager.run(final_circuit)
    isa_observable = observable.apply_layout(isa_circuit.layout)

    isa_2qubit_depth = isa_circuit.depth(lambda x: x.operation.num_qubits == 2)
    logger.info(f"ISA circuit two-qubit depth: {isa_2qubit_depth}")
    output["twoqubit_depth"] = isa_2qubit_depth

    end_optimizing = time.time()

    output["metadata"] = {
        "resources_usage": {
            "RUNNING: MAPPING": {
                "CPU_TIME": end_mapping - start_mapping,
            },
            "RUNNING: OPTIMIZING_FOR_HARDWARE": {
                "CPU_TIME": end_optimizing - start_optimizing,
            },
        }
    }

    # Exit now if dry run; don't execute on hardware
    if dry_run:
        logger.info("Exiting before hardware execution since `dry_run` is True.")
        return output

    # Step 3: Execute on Hardware
    # Submit the underlying Estimator job. Note that this is not the
    # actual function job.
    start_waiting_qpu = time.time()
    job = estimator.run([(isa_circuit, isa_observable)])
    logger.info(f"Job ID: {job.job_id()}")
    output["job_id"] = job.job_id()
    while job.status() == "QUEUED":
        update_status(Job.WAITING_QPU)
        time.sleep(5)

    end_waiting_qpu = time.time()
    update_status(Job.EXECUTING_QPU)

    # Wait until job is complete
    hw_results = job.result()
    end_executing_qpu = time.time()
    hw_results_dicts = [pub_result.data.__dict__ for pub_result in hw_results]

    # Save hardware results to serverless output dictionary
    output["hw_results"] = hw_results_dicts

    # Reorganize expectation values
    start_pp = time.time()
    update_status(Job.POST_PROCESSING)
    hw_expvals = [pub_result_data["evs"].tolist() for pub_result_data in hw_results_dicts]
    end_pp = time.time()

    # Return expectation values in serializable format
    logger.info(f"Hardware expectation values: {hw_expvals}")
    output["hw_expvals"] = hw_expvals[0]
    output["metadata"]["resources_usage"]["RUNNING: WAITING_FOR_QPU"] = (
        {
            "CPU_TIME": end_waiting_qpu - start_waiting_qpu,
        },
    )
    output["metadata"]["resources_usage"]["RUNNING: EXECUTING_QPU"] = (
        {
            "QPU_TIME": end_executing_qpu - end_waiting_qpu,
        },
    )
    output["metadata"]["resources_usage"]["RUNNING: POST_PROCESSING"] = (
        {
            "CPU_TIME": end_pp - start_pp,
        },
    )
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
