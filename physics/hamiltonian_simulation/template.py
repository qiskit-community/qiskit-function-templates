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

import datetime
import json
import os
import traceback
import numpy as np

from mergedeep import merge
import quimb.tensor
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

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import EstimatorV2 as Estimator

from qiskit_serverless import get_arguments, save_result


# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# TODO: group inputs into categories
def run_function(
    backend_name,
    hamiltonian,
    observable,
    aqc_evolution_time,
    aqc_ansatz_num_trotter_steps,
    aqc_target_num_trotter_steps,
    remainder_evolution_time,
    remainder_num_trotter_steps,
    **kwargs,
):
    """
    Entry point to the Hamiltonian Simulation Function.
    """

    # Extract parameters from arguments.
    # Do this at the top of the program so it fails early if any required
    # arguments are missing or invalid.
    dry_run = kwargs.get("dry_run", False)

    # Stop if this fidelity is achieved
    aqc_stopping_fidelity = kwargs.get("aqc_stopping_fidelity", 1.0)
    # Stop after this number of iterations, even if stopping fidelity is not achieved
    aqc_max_iterations = kwargs.get("aqc_max_iterations", 500)
    initial_state = kwargs.get("initial_state", QuantumCircuit(hamiltonian.num_qubits))

    # Initialize Qiskit Runtime Service
    print("starting runtime service")
    service = QiskitRuntimeService(channel="ibm_quantum")
    backend = service.backend(backend_name)
    print("backend", backend)
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
    }
    # Merge with user-provided options
    estimator_options = merge(
        kwargs.get("estimator_options", {}), estimator_default_options
    )

    # When the function template is running, it is helpful to return
    # information in the logs by using print statements, so that you can better
    # evaluate the workload's progress. This example returns the estimator options.
    print("estimator_options =", json.dumps(estimator_options, indent=4))

    # Perform parameter validation
    if not 0.0 < aqc_stopping_fidelity <= 1.0:
        raise ValueError(
            f"Invalid stopping fidelity: {aqc_stopping_fidelity}. ",
            "It must be a positive float no greater than 1.",
        )

    # Prepare a dictionary to hold all of the function template outputs.
    # Keys will be added to this dictionary throughout the workflow,
    # and it is returned at the end of the program.
    output = {}

    # Step 1: Map
    os.environ["NUMBA_CACHE_DIR"] = "/data"
    print("Hamiltonian:", hamiltonian)
    print("Observable:", observable)
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
    print("Target MPS maximum bond dimension:", aqc_target_mps.psi.max_bond())
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
    print("Number of AQC parameters:", len(aqc_initial_parameters))
    output["num_aqc_parameters"] = len(aqc_initial_parameters)

    # Calculate the fidelity of ansatz circuit vs. the target state, before optimization
    good_mps = tensornetwork_from_circuit(aqc_good_circuit, simulator_settings)
    starting_fidelity = abs(compute_overlap(good_mps, aqc_target_mps)) ** 2
    print("Starting fidelity of AQC portion:", starting_fidelity)
    output["aqc_starting_fidelity"] = starting_fidelity

    # Optimize the ansatz parameters by using MPS calculations
    def callback(intermediate_result: OptimizeResult):
        fidelity = 1 - intermediate_result.fun
        print(f"{datetime.datetime.now()} Intermediate result: Fidelity {fidelity:.8f}")
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
        raise RuntimeError(
            f"Optimization failed: {result.message} (status={result.status})"
        )
    print(f"Done after {result.nit} iterations.")
    output["num_iterations"] = result.nit
    aqc_final_parameters = result.x
    output["aqc_final_parameters"] = list(aqc_final_parameters)

    # Construct an optimized circuit for initial portion of time evolution
    aqc_final_circuit = aqc_ansatz.assign_parameters(aqc_final_parameters)

    # Calculate fidelity after optimization
    aqc_final_mps = tensornetwork_from_circuit(aqc_final_circuit, simulator_settings)
    aqc_fidelity = abs(compute_overlap(aqc_final_mps, aqc_target_mps)) ** 2
    print("Fidelity of AQC portion:", aqc_fidelity)
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

    # Step 2: Optimize

    # Transpile PUBs (circuits and observables) to match ISA
    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa_circuit = pass_manager.run(final_circuit)
    isa_observable = observable.apply_layout(isa_circuit.layout)

    isa_2qubit_depth = isa_circuit.depth(lambda x: x.operation.num_qubits == 2)
    print("ISA circuit two-qubit depth:", isa_2qubit_depth)
    output["twoqubit_depth"] = isa_2qubit_depth

    # Exit now if dry run; don't execute on hardware
    if dry_run:
        print("Exiting before hardware execution since `dry_run` is True.")
        return output

    # Step 3: Execute on Hardware
    estimator = Estimator(backend, options=estimator_options)

    # Submit the underlying Estimator job. Note that this is not the
    # actual function job.
    job = estimator.run([(isa_circuit, isa_observable)])
    print("Job ID:", job.job_id())
    output["job_id"] = job.job_id()

    # Wait until job is complete
    hw_results = job.result()
    hw_results_dicts = [pub_result.data.__dict__ for pub_result in hw_results]

    # Save hardware results to serverless output dictionary
    output["hw_results"] = hw_results_dicts

    # Reorganize expectation values
    hw_expvals = [
        pub_result_data["evs"].tolist() for pub_result_data in hw_results_dicts
    ]

    # Return expectation values in serializable format
    print("Hardware expectation values", hw_expvals)
    output["hw_expvals"] = hw_expvals[0]
    return output


if __name__ == "__main__":
    input_args = get_arguments()
    try:
        func_result = run_function(**input_args)
        save_result(func_result)
    except Exception:
        save_result(traceback.format_exc())
        raise
