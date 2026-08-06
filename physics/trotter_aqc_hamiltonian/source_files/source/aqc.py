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

"""Stage 2 — AQC compression (classical, in-process).

Compresses the deep Trotter targets from stage 1 into a shallow parametrised
ansatz by maximising MPS-fidelity (``qiskit-addon-aqc-tensor`` + quimb). This is
the memory-dominant stage of the function: its footprint grows sharply with
``AQCOptions.max_bond``.

``cotengrust`` (a Rust contraction-path backend that quimb/cotengra pick up
automatically when installed) keeps that footprint manageable, so the stage runs
inline rather than needing a separate high-memory step.
:func:`check_cotengrust_engaged` reports whether it is present: if it is missing
the run continues on quimb's default, far more memory-heavy contraction path,
and a warning is logged and surfaced in the result metadata rather than failing
outright.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import numpy as np
import scipy.optimize
from qiskit import QuantumCircuit

from .build import mixed_circuit
from .hamiltonian import HamiltonianSpec
from ._serverless import get_logger

logger = get_logger(__name__)


def check_cotengrust_engaged(max_bond: int) -> bool:
    """Report whether the Rust contraction backend is available.

    ``cotengrust`` collapses the AQC contraction footprint and quimb/cotengra
    pick it up automatically when present. It is strongly recommended but not
    required: if it is missing the run falls back to quimb's default contraction
    path (far more memory-heavy at high ``max_bond``) and this logs a warning
    rather than failing.

    Returns ``True`` when cotengrust is available, ``False`` otherwise.
    """
    try:
        import cotengrust  # pylint: disable=unused-import
    except Exception:  # pylint: disable=broad-exception-caught
        msg = (
            f"cotengrust not available — falling back to quimb's default contraction "
            f"path, which is far more memory-heavy at max_bond={max_bond}. Install "
            f"cotengrust in the execution environment for the intended footprint."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)  # visible in the terminal/stderr
        logger.warning(msg)  # visible in job.logs()
        return False

    logger.info("cotengrust contraction backend is available and engaged.")
    return True


@dataclass
class AQCOptions:
    """Tunables for the AQC compression stage."""

    max_bond: int = 32
    cutoff: float = 1e-8
    autodiff_backend: str = "jax"
    # Compression plan: ordered list of ``(n_steps, ansatz_steps)`` segments. Each
    # segment compresses its ``n_steps`` consecutive time steps into an ansatz
    # generated from the ``ansatz_steps``-Trotter target; the total compressed count
    # is the sum of ``n_steps``. Deeper ansätze (larger ansatz_steps) hold fidelity
    # for the later, more-entangled steps. Default: one 1-step, 1-layer segment.
    segments: list = field(default_factory=lambda: [(1, 1)])
    fidelity_target: Optional[float] = None  # early-stop per step if set
    # Passed straight to scipy.optimize.minimize(objective, x0, **optimizer_settings).
    optimizer_settings: dict = field(
        default_factory=lambda: {"method": "L-BFGS-B", "jac": True, "options": {"maxiter": 300}}
    )


@dataclass
class AQCResult:
    """Compressed circuits, their parameters, and the fidelity achieved per step."""

    circuits: dict[int, QuantumCircuit]
    params: dict[int, np.ndarray]
    fidelities: dict[int, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)  # surfaced in result metadata


def compress(
    target_circuits: dict[int, QuantumCircuit],
    opts: AQCOptions,
) -> AQCResult:
    """Compress the leading time steps into per-segment shallow ansätze.

    ``opts.segments`` is an ordered list of ``(n_steps, ansatz_steps)``: each
    segment compresses its ``n_steps`` consecutive time steps into an ansatz
    generated from the ``ansatz_steps``-Trotter target. Steps are warm-started
    from the previous step's optimum *within a segment*, and reset to the fresh
    ansatz parameters at each segment boundary, since successive segments use
    different (deeper) ansätze.
    """
    from qiskit_addon_aqc_tensor import generate_ansatz_from_circuit
    from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity
    from qiskit_addon_aqc_tensor.simulation import tensornetwork_from_circuit
    from qiskit_addon_aqc_tensor.simulation.quimb import QuimbSimulator

    aqc_warnings: list[str] = []
    if not check_cotengrust_engaged(opts.max_bond):
        aqc_warnings.append(
            "cotengrust not available — AQC compression ran on quimb's default "
            f"contraction path (slower / far more memory at max_bond={opts.max_bond}). "
            "Install cotengrust in the execution environment for the intended footprint."
        )

    sim = QuimbSimulator(
        quimb_circuit_factory=partial(
            __import__("quimb.tensor", fromlist=["CircuitMPS"]).CircuitMPS,
            gate_opts={"cutoff": opts.cutoff, "max_bond": opts.max_bond},
        ),
        autodiff_backend=opts.autodiff_backend,
    )

    total = sum(n_steps for n_steps, _ in opts.segments)
    target_mps = {
        k: tensornetwork_from_circuit(target_circuits[k], sim) for k in range(1, total + 1)
    }

    circuits: dict[int, QuantumCircuit] = {}
    params: dict[int, np.ndarray] = {}
    fidelities: dict[int, float] = {}

    k = 0  # global time-step index across all segments
    n_segments = len(opts.segments)
    for seg_i, (n_steps, ansatz_steps) in enumerate(opts.segments):
        ansatz, init_params = generate_ansatz_from_circuit(
            target_circuits[ansatz_steps], qubits_initially_zero=True
        )
        x0 = np.array(init_params)  # reset at each segment boundary (ansatz changes)
        logger.info(
            "AQC segment %d/%d: %d step(s), ansatz seeded from step %d (%d params)",
            seg_i + 1,
            n_segments,
            n_steps,
            ansatz_steps,
            len(x0),
        )
        for _ in range(n_steps):
            k += 1
            obj = MaximizeStateFidelity(target_mps[k], ansatz, sim)

            kwargs = dict(opts.optimizer_settings)  # passthrough to scipy.optimize.minimize
            if opts.fidelity_target is not None:
                # our early-stop: end this step once the target fidelity is reached
                kwargs["callback"] = (
                    lambda x, _obj=obj: (1 - _obj.loss_function(x)[0]) >= opts.fidelity_target
                )

            result = scipy.optimize.minimize(obj.loss_function, x0, **kwargs)
            params[k] = result.x
            circuits[k] = ansatz.assign_parameters(result.x)
            fidelities[k] = float(1 - obj.loss_function(result.x)[0])
            x0 = result.x  # warm-start the next step within this segment
            logger.info(
                "AQC step %d/%d (segment %d): fidelity=%.4f", k, total, seg_i + 1, fidelities[k]
            )

    return AQCResult(circuits=circuits, params=params, fidelities=fidelities, warnings=aqc_warnings)


def assemble_all_circuits(
    aqc: AQCResult,
    n_compressed: int,
    time_steps: int,
    spec: HamiltonianSpec,
    dt: float,
) -> list[QuantumCircuit]:
    """AQC-compressed steps ``1..n_compressed`` + plain Trotter for the remainder."""
    all_circuits: list[QuantumCircuit] = []
    for k in range(1, n_compressed + 1):
        all_circuits.append(aqc.circuits[k])
    for k in range(1, time_steps - n_compressed + 1):
        all_circuits.append(mixed_circuit(spec, aqc.circuits[n_compressed], k, dt))
    return all_circuits
