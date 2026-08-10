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

"""Stage 1 — Build: classical spec -> circuits.

Initial-state preparation, the nearest-neighbor Trotter targets fed to AQC in
stage 2, and the measured observables.

Nothing here is model-specific: the time evolution comes from
:meth:`HamiltonianSpec.build_evolution`, so any 1D nearest-neighbor Pauli
Hamiltonian flows through unchanged.
"""
from __future__ import annotations

from typing import Optional

from qiskit import QuantumCircuit, transpile
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.quantum_info import SparsePauliOp

from .hamiltonian import HamiltonianSpec
from ._serverless import get_logger

logger = get_logger(__name__)

# 1q/2q gate set accepted by qiskit_quimb (AQC's MPS builder). Any prepared
# state circuit is translated to this so the AQC tensor-network path accepts it.
_QUIMB_BASIS = ["rx", "ry", "rz", "cx", "h"]


# ════════════════════════════════════════════════════════════════════════
#  Initial state
# ════════════════════════════════════════════════════════════════════════


def prepare_initial_state(n: int, circuit: Optional[QuantumCircuit]) -> QuantumCircuit:
    """Build the ``n``-qubit state-prep circuit.

    Default ``|0...0>`` when ``circuit`` is ``None``; otherwise a prepared
    :class:`~qiskit.circuit.QuantumCircuit` passed in natively — this is how a
    caller hands in a ground state it optimized itself (DMRG / VQE / …), and
    Qiskit Serverless QPY-serializes it over the wire.

    The result is transpiled to :data:`_QUIMB_BASIS` so the AQC MPS builder
    accepts it (qubit order preserved — no coupling map).
    """
    if circuit is None:
        qc = QuantumCircuit(n)
    else:
        qc = circuit
        if qc.num_qubits != n:
            raise ValueError(f"initial_state has {qc.num_qubits} qubits, expected {n}.")

    return transpile(qc, basis_gates=_QUIMB_BASIS, optimization_level=1)


# ════════════════════════════════════════════════════════════════════════
#  Trotter targets + observables
# ════════════════════════════════════════════════════════════════════════


def build_evolution_targets(
    spec: HamiltonianSpec,
    init_circuit: QuantumCircuit,
    dt: float,
    up_to_step: int,
) -> dict[int, QuantumCircuit]:
    """Raw (uncompressed) Trotter circuits: ``init + k`` steps for ``k = 1 .. up_to_step``.

    Doubles as the AQC compression stage's targets (:func:`aqc.compress` reads the
    per-segment ansatz seeds and the ``1..sum(n_steps)`` step targets it needs) and
    as the full-Trotter reference series for reporting (built up to ``t_steps``).
    """
    targets: dict[int, QuantumCircuit] = {}
    for k in range(1, up_to_step + 1):
        qc = init_circuit.copy()
        qc.compose(spec.build_evolution(k, dt), inplace=True)
        targets[k] = qc
    return targets


def circuit_stats(qc: QuantumCircuit) -> dict:
    """2q depth and 2q gate count of ``qc``.

    Circuits from :meth:`HamiltonianSpec.build_evolution` are already flattened to
    the ``rx``/``ry``/``rz``/``cx`` basis, so no decompose/transpile step is needed
    here (unlike a circuit still holding an opaque ``PauliEvolutionGate``).
    """
    return {
        "depth_2q": qc.depth(filter_function=lambda instr: instr.operation.num_qubits == 2),
        "num_2q_gates": qc.count_ops().get("cx", 0),
    }


def mixed_circuit(
    spec: HamiltonianSpec,
    aqc_circuit: QuantumCircuit,
    time_diff: int,
    dt: float,
) -> QuantumCircuit:
    """Concatenate ``time_diff`` extra plain Trotter steps onto an AQC circuit."""
    qc = aqc_circuit.copy()
    qc.compose(spec.build_evolution(time_diff, dt), inplace=True)
    return qc


def build_observables(n: int, observables=None) -> tuple[list[SparsePauliOp], list[str]]:
    """Observables to measure each time step, with human-readable labels.

    Accepts exactly what you pass to ``EstimatorV2`` as its ``observables`` argument —
    anything :meth:`ObservablesArray.coerce` understands: a
    :class:`~qiskit.quantum_info.SparsePauliOp` / ``Pauli`` / ``PauliList`` / Pauli
    string / ``{pauli: coeff}`` mapping, or a (nested) list of those. Each element is
    measured separately and becomes **one output column** (``None`` -> single-site
    ``Z`` on every qubit).

    A single unit-coefficient Pauli is labeled by its string; anything else is
    labeled ``obs_<i>``. Every observable must act on all ``n`` qubits.
    """
    if observables is None:
        obs = [SparsePauliOp("I" * (n - 1 - i) + "Z" + "I" * i) for i in range(n)]
        return obs, [f"Z_{i}" for i in range(n)]

    # Same parser EstimatorV2 uses; normalizes every element to a {pauli: coeff} dict.
    arr = ObservablesArray.coerce(observables)
    elements = arr.reshape((arr.size,)).tolist()  # flat list of {pauli: coeff} dicts

    obs, labels = [], []
    for i, element in enumerate(elements):
        op = SparsePauliOp.from_list(list(element.items()))
        if op.num_qubits != n:
            raise ValueError(f"observable {i}: acts on {op.num_qubits} qubits, expected {n}.")
        if len(element) == 1 and next(iter(element.values())) == 1.0:
            labels.append(next(iter(element)))  # single unit Pauli -> its string
        else:
            labels.append(f"obs_{i}")
        obs.append(op)
    return obs, labels
