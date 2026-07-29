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
Executor Function Template source code.

This template exposes the Executor in two modes via a single run_function entry point:

* mode="sampler": run PUBs on the Executor and return the raw PrimitiveResult (bit arrays).
* mode="estimator": wrap the executor by expanding each observable into qubit-wise-commuting
  measurement circuits, running them through the sampler path, and reconstructing
  expectation values client-side.

"""

from collections.abc import Iterable, Sequence
from functools import lru_cache
import math
import time
import traceback

import numpy as np

from qiskit.circuit import ControlFlowOp, QuantumCircuit
from qiskit.primitives.containers import (
    EstimatorPubLike,
    SamplerPubLike,
    PrimitiveResult,
    PubResult,
    DataBin,
)
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager

from qiskit_ibm_runtime import Executor
from qiskit_ibm_runtime.exceptions import IBMInputValueError
from qiskit_ibm_runtime.executor_sampler.sampler import prepare as prepare_sampler_program
from qiskit_ibm_runtime.executor_sampler.utils import extract_shots_from_pubs
from qiskit_ibm_runtime.options_models.sampler_options import SamplerOptions

from options import Options, DEFAULT_MITIGATION_LEVEL, DEFAULT_OPTIMIZATION_LEVEL
from options.utils import merge_options

from qiskit_serverless import (
    get_arguments,
    save_result,
    update_status,
    Job,
    get_runtime_service,
    get_logger,
)

logger = get_logger()

# Job tag for the function.
EXECUTOR_JOB_TAG = "ibm/executor-function-template"

# Supported execution modes.
SAMPLER_MODE = "sampler"
ESTIMATOR_MODE = "estimator"

# The classical register created by QuantumCircuit.measure_all on the estimator-mode
# measurement circuits.
_MEAS_REGISTER = "meas"


def are_circuits_dynamic(pubs: Iterable) -> bool:
    """Return True if any pub circuit contains control flow (a dynamic circuit)."""
    for pub in pubs:
        circuit = pub.circuit
        for inst in circuit.data:
            if (
                isinstance(inst.operation, ControlFlowOp)
                or getattr(inst.operation, "condition", None) is not None
            ):
                return True
    return False


def validate_estimator_pubs(pubs: list[EstimatorPub]) -> None:
    """Validate estimator pubs before reconstruction.

    Raises:
        ValueError: If an observable array is empty or a circuit carries measurements
            (the measurement circuits are appended here, so the input must be unmeasured).
    """
    for pub in pubs:
        if pub.observables.shape == (0,):
            raise ValueError("Empty observables array is not allowed")
        if pub.circuit.num_clbits:
            raise ValueError(
                "Estimator-mode circuits must not contain measurements or classical bits; "
                "measurements are appended per commuting group during reconstruction."
            )


# resolve_precision and unbroadcast_index are copied verbatim from the reference
# qiskit_ibm_runtime.executor_estimator.utils module.
def resolve_precision(pubs: list[EstimatorPub], run_precision: float | None = None) -> float | None:
    """Resolve precision from multiple sources with clear precedence.

    Precedence order (highest to lowest):
    1. Individual pub precision (must be consistent across all pubs)
    2. run() method precision parameter (run_precision)

    Args:
        pubs: List of estimator pubs (may contain precision values).
        run_precision: Precision specified in run() method.

    Returns:
        The resolved precision value, or None if no precision is specified anywhere.

    Raises:
        IBMInputValueError: If pubs have different precision values.
    """
    # Extract precision from pubs
    pub_precisions = {pub.precision if pub.precision is not None else run_precision for pub in pubs}

    if len(pub_precisions) != 1:
        raise IBMInputValueError(
            f"All pubs must have the same precision. Found: {pub_precisions}"
            "(possibly via the run provided precision parameter)"
        )

    if (precision := next(iter(pub_precisions))) is not None and precision <= 0:
        raise IBMInputValueError("The precision value must be strictly greater than 0.")

    return precision


def unbroadcast_index(bc_index: tuple[int, ...], shape: tuple[int, ...]) -> tuple[int, ...]:
    """Index an array using an index from a compatible broadcasted shape.

    An ND-array ``arr`` is broadcastable to any shape ``bc_shape = (*pad_shape, *arr.shape)``.
    This function allows indexing ``arr`` using an ND-index from ``bc_shape`` and returns the
    index for ``arr`` that accesses the same value.

    Args:
        bc_index: An ND-index from a broadcasted shape.
        shape: The shape of the broadcasting compatible array to index.

    Returns:
        The equivalent un-broadcasted ND-index of the array with specified shape.
    """

    @lru_cache
    def _pad_broadcast_shape(shape: tuple[int, ...], ndims: int) -> tuple[int, ...]:
        # Pad a shape with trivial dimensions.
        shape_ndims = len(shape)
        pad = ndims - shape_ndims
        if pad > 0:
            return pad * (1,) + shape
        return shape

    shape_ndims = len(shape)
    if shape_ndims == 0:
        return ()

    pad_shape = _pad_broadcast_shape(shape, len(bc_index))
    bc_index = tuple(0 if dim == 1 else i for i, dim in zip(bc_index, pad_shape))
    return bc_index[-shape_ndims:]


def _observable_at(pub: EstimatorPub, index: tuple) -> SparsePauliOp:
    """Return the observable at ``index`` of the pub's broadcast shape as a SparsePauliOp."""
    element = pub.observables[unbroadcast_index(index, pub.observables.shape)]
    return SparsePauliOp.from_list(list(element.items()))


def _bound_circuit(pub: EstimatorPub, index: tuple) -> QuantumCircuit:
    """Bind the pub circuit at the parameter set for ``index`` of the broadcast shape."""
    params = pub.parameter_values
    if params.num_parameters == 0:
        return pub.circuit
    param_array = params.as_array()
    return pub.circuit.assign_parameters(param_array[unbroadcast_index(index, params.shape)])


def _group_measurement_pauli(group: SparsePauliOp) -> Pauli:
    """Return a representative Pauli whose per-qubit basis defines the group's measurement.

    All Paulis in a qubit-wise-commuting group agree (or are identity) on each qubit, so OR-ing
    their x/z masks yields the single measurement basis per qubit.
    """
    num_qubits = group.num_qubits
    x = np.zeros(num_qubits, dtype=bool)
    z = np.zeros(num_qubits, dtype=bool)
    for pauli in group.paulis:
        x |= pauli.x
        z |= pauli.z
    return Pauli((z, x))


def _diagonalizing_circuit(basis: Pauli) -> QuantumCircuit:
    """Build the basis-change circuit that rotates each Pauli axis onto the Z axis.

    X is diagonalized by H; Y by S-dagger followed by H; Z/I need no rotation. Applying this
    before measurement lets us read expectation values off the computational-basis counts.
    """
    circuit = QuantumCircuit(basis.num_qubits)
    for qubit in range(basis.num_qubits):
        x, z = bool(basis.x[qubit]), bool(basis.z[qubit])
        if x and not z:  # X
            circuit.h(qubit)
        elif x and z:  # Y
            circuit.sdg(qubit)
            circuit.h(qubit)
    return circuit


def _to_diagonal(pauli: Pauli) -> Pauli:
    """Map a Pauli to its diagonal (Z-on-support) form for post-rotation counts."""
    support = pauli.z | pauli.x
    return Pauli((support, np.zeros(pauli.num_qubits, dtype=bool)))


def prepare_estimator(pubs: Sequence[EstimatorPub]):
    """Expand estimator pubs into measurement circuits plus a reconstruction map.

    Each element of a pub's broadcast shape has its observable grouped into
    qubit-wise-commuting sets; every group becomes one measurement circuit (basis change +
    measure_all). These circuits are then fed through the shared sampler path.

    Args:
        pubs: The coerced estimator pubs.

    Returns:
        A tuple ``(circuits, recon)``: the flat list of measurement circuits to sample, and a
        per-pub list of reconstruction metadata mapping each output element to its circuits.
    """
    circuits: list[QuantumCircuit] = []
    recon: list[dict] = []

    for pub in pubs:
        entries: dict[tuple, list] = {}
        for index in np.ndindex(*pub.shape):
            operator = _observable_at(pub, index)
            base_circuit = _bound_circuit(pub, index)
            element_terms = []
            for group in operator.group_commuting(qubit_wise=True):
                basis = _group_measurement_pauli(group)
                meas_circuit = base_circuit.compose(_diagonalizing_circuit(basis))
                meas_circuit.measure_all()
                element_terms.append(
                    (
                        len(circuits),
                        [
                            (_to_diagonal(pauli), complex(coeff).real)
                            for pauli, coeff in zip(group.paulis, group.coeffs)
                        ],
                    )
                )
                circuits.append(meas_circuit)
            entries[index] = element_terms
        recon.append({"shape": pub.shape, "entries": entries})

    return circuits, recon


def reconstruct_estimator(
    sampler_result: PrimitiveResult,
    recon: list[dict],
    precision: float | None,
) -> PrimitiveResult:
    """Rebuild expectation values from the sampled measurement circuits.

    Each diagonal Pauli's expectation value is read off its measurement circuit's bit array and
    combined with its coefficient; the standard deviation uses the sampled-variance estimate
    ``(1 - <P>**2) / shots`` propagated through the coefficients.

    Args:
        sampler_result: The bit-array results, one entry per prepared measurement circuit.
        recon: The reconstruction metadata returned by prepare_estimator.
        precision: The resolved target precision, recorded in each pub's metadata.

    Returns:
        An estimator-style result with data.evs / data.stds shaped per pub.
    """
    pub_results: list[PubResult] = []
    for meta in recon:
        shape = meta["shape"]
        evs = np.zeros(shape, dtype=float)
        stds = np.zeros(shape, dtype=float)
        shots = None
        for index, element_terms in meta["entries"].items():
            value = 0.0
            variance = 0.0
            for circuit_id, diagonal_terms in element_terms:
                bit_array = sampler_result[circuit_id].data[_MEAS_REGISTER]
                shots = bit_array.num_shots
                for diagonal_pauli, coeff in diagonal_terms:
                    exp = float(bit_array.expectation_values(diagonal_pauli))
                    value += coeff * exp
                    variance += (coeff**2) * (1.0 - exp**2) / shots
            evs[index] = value
            stds[index] = math.sqrt(max(variance, 0.0))
        data = DataBin(evs=evs, stds=stds, shape=shape)
        pub_results.append(
            PubResult(data, metadata={"target_precision": precision, "shots": shots})
        )
    return PrimitiveResult(pub_results, metadata={})


def _to_sampler_options(execution_options: dict) -> SamplerOptions:
    """Map the template execution options onto the runtime SamplerOptions consumed by prepare."""
    mapped = {
        key: execution_options[key]
        for key in ("default_shots", "max_execution_time", "dynamical_decoupling", "twirling")
        if key in execution_options
    }
    sampler_options = SamplerOptions(**mapped)
    # runtime >=0.48 validates that the twirling flags are real booleans: prepare() raises
    # IBMInputValueError if either is left as None. The template only turns twirling on at
    # mitigation level 2; below that the flags are unset, so pin them to False explicitly to
    # keep the no-twirling path (plain circuit items) that None implied under runtime 0.47.
    if sampler_options.twirling.enable_gates is None:
        sampler_options.twirling.enable_gates = False
    if sampler_options.twirling.enable_measure is None:
        sampler_options.twirling.enable_measure = False
    return sampler_options


def _isa_sampler_pubs(sampler_pubs: list[SamplerPub], backend, optimization_level: int):
    """Transpile each sampler pub's circuit to the backend ISA, preserving params and shots."""
    pass_manager = generate_preset_pass_manager(
        backend=backend, optimization_level=optimization_level
    )
    isa_pubs = []
    for pub in sampler_pubs:
        isa_circuit = pass_manager.run(pub.circuit)
        isa_pubs.append(SamplerPub.coerce((isa_circuit, pub.parameter_values, pub.shots)))
    return isa_pubs


def run_function(
    backend_name: str,
    pubs: Iterable[SamplerPubLike | EstimatorPubLike],
    mode: str = SAMPLER_MODE,
    options: dict | None = None,
    instance: str | None = None,
    **kwargs,
) -> dict:
    """
    Entry point to the Executor Function.

    Optimize and execute PUBs on hardware through the Executor. In "sampler" mode the raw
    bit-array results are returned; in "estimator" mode the executor is wrapped to reconstruct
    expectation values client-side. As with the vanilla template we only show minimal input
    arguments so they can be extended.

    Args:
        backend_name: Name of the backend to use.
        pubs: An iterable of pub-like objects. For "sampler" mode these are sampler pubs such as
            ``(circuit,)`` or ``(circuit, parameter_values, shots)``; for "estimator" mode these
            are estimator pubs such as ``(circuit, observables)`` or
            ``(circuit, observables, parameter_values, precision)``.
        mode: Either "sampler" or "estimator".
        options: Input options. Recognized keys: ``optimization_level`` / ``mitigation_level``
            plus the executor execution options (``default_shots``, ``default_precision``,
            ``max_execution_time``, ``dynamical_decoupling``, ``twirling``).
        instance: The instance to use.

    Returns:
        A dictionary ``{"hw_results": PrimitiveResult, "metadata": {...}}``.

    Raises:
        ValueError: If input arguments are invalid.
    """
    logger.info("Execution mode: %s", mode)
    logger.info("Input options are: %s", options)
    logger.info("Backend used: %s", backend_name)
    logger.info("Instance used: %s", instance)

    if mode not in (SAMPLER_MODE, ESTIMATOR_MODE):
        raise ValueError(f"Invalid mode {mode!r}. Expected {SAMPLER_MODE!r} or {ESTIMATOR_MODE!r}.")

    # Parse kwargs used by the tests / dry runs.
    dry_run = kwargs.get("dry_run", False)
    testing_backend = kwargs.get("testing_backend", None)

    if not pubs:
        raise ValueError("At least one PUB is needed.")

    # User options take precedence over the mitigation-level defaults.
    options = options or {}
    applied_mit = Options.apply_mitigation_level(
        mitigation_level=options.get("mitigation_level", DEFAULT_MITIGATION_LEVEL)
    )
    combined = merge_options(applied_mit, options)
    options_parsed = Options(**combined)
    options_dict = options_parsed.model_dump(exclude_unset=True)
    transpilation_options = Options.get_transpilation_options(options_dict)
    execution_options = Options.get_execution_options(options_dict)
    optimization_level = transpilation_options.get("optimization_level", DEFAULT_OPTIMIZATION_LEVEL)

    # Validate and set input backend.
    if not backend_name:
        raise ValueError(f"Invalid backend name value {backend_name}")
    if testing_backend is None:
        logger.info("Starting runtime service")
        service = get_runtime_service()
        backend = service.backend(backend_name)
    else:
        backend = testing_backend
    logger.info("Backend: %s", backend.name)

    output: dict = {}

    # Step 2: Optimize -- coerce per mode, transpile to ISA, then build the program via prepare.
    start_optimizing = time.time()
    update_status(Job.OPTIMIZING_HARDWARE)

    recon = None
    precision = None
    if mode == ESTIMATOR_MODE:
        estimator_pubs = [EstimatorPub.coerce(pub) for pub in pubs]
        validate_estimator_pubs(estimator_pubs)
        if are_circuits_dynamic(estimator_pubs):
            raise ValueError("Dynamic circuits are not supported in estimator mode.")

        default_shots = execution_options.get("default_shots")
        default_precision = execution_options.get("default_precision")
        # Estimator mode is precision-driven: a pub precision wins, otherwise default_precision
        # sets the target. default_shots acts only as a floor on the derived shot count. The
        # resolved precision (which may come from default_precision) is recorded in the result
        # metadata as target_precision.
        precision = resolve_precision(estimator_pubs)
        if precision is None:
            precision = default_precision
        if precision is not None:
            shots = int(np.ceil(1.0 / (precision**2)))
            if default_shots is not None:
                shots = max(shots, int(default_shots))
        elif default_shots is not None:
            shots = int(default_shots)
        else:
            raise ValueError(
                "Estimator mode needs a precision (on the pub or via default_precision) "
                "or a default_shots value."
            )

        meas_circuits, recon = prepare_estimator(estimator_pubs)
        sampler_pubs = [SamplerPub.coerce(circuit) for circuit in meas_circuits]
    else:
        sampler_pubs = [SamplerPub.coerce(pub) for pub in pubs]
        shots = extract_shots_from_pubs(sampler_pubs, execution_options.get("default_shots"))

    isa_pubs = _isa_sampler_pubs(sampler_pubs, backend, optimization_level)
    program, executor_options = prepare_sampler_program(
        isa_pubs, _to_sampler_options(execution_options), default_shots=shots
    )

    end_optimizing = time.time()
    output["metadata"] = {
        "resources_usage": {
            "RUNNING: OPTIMIZING_FOR_HARDWARE": {"CPU_TIME": end_optimizing - start_optimizing}
        }
    }

    # Exit before hardware if this is a dry run.
    if dry_run:
        logger.info("Exiting before hardware")
        return output

    # Step 3: Execute on hardware through the Executor.
    executor = Executor(mode=backend, options=executor_options)
    executor.options.environment.job_tags = [EXECUTOR_JOB_TAG]

    start_waiting_qpu = time.time()
    job = executor.run(program)
    logger.info("Qiskit Runtime job %s submitted.", job.job_id())

    while job.status() == "QUEUED":
        update_status(Job.WAITING_QPU)
        time.sleep(5)

    end_waiting_qpu = time.time()
    update_status(Job.EXECUTING_QPU)
    sampler_result = job.result()
    end_executing_qpu = time.time()

    # Step 4: Post-process. Sampler returns raw samples; estimator reconstructs evs client-side.
    start_pp = time.time()
    update_status(Job.POST_PROCESSING)
    if mode == ESTIMATOR_MODE:
        hw_results = reconstruct_estimator(sampler_result, recon, precision)
    else:
        hw_results = sampler_result
    end_pp = time.time()

    output["hw_results"] = hw_results
    output["metadata"]["resources_usage"]["RUNNING: WAITING_FOR_QPU"] = {
        "CPU_TIME": end_waiting_qpu - start_waiting_qpu,
    }
    output["metadata"]["resources_usage"]["RUNNING: EXECUTING_QPU"] = {
        "QPU_TIME": end_executing_qpu - end_waiting_qpu,
    }
    output["metadata"]["resources_usage"]["RUNNING: POST_PROCESSING"] = {
        "CPU_TIME": end_pp - start_pp,
    }
    return output


# run_function gets called here; this is boilerplate and meant to be used without
# customization.
if __name__ == "__main__":
    input_args = get_arguments()

    try:
        func_result = run_function(**input_args)
        save_result(func_result)
    except Exception:
        save_result(traceback.format_exc())
        raise
