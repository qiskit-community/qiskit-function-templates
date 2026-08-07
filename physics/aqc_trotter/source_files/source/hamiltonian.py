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

"""Hamiltonian specification: the only model-specific part of the pipeline.

A :class:`HamiltonianSpec` supplies the one thing the rest of the pipeline needs
that depends on the model — the ``n``-qubit time-evolution circuit, built with a
Trotter product formula (2nd-order Suzuki by default).

``build`` / ``aqc`` / ``execute`` consume only this interface, so supporting a
new model family means adding a subclass and registering it in ``_TIERS``;
nothing downstream changes.

Tiers
-----
``pauli_1d_nn`` : the only tier currently registered. The caller supplies the
                  Hamiltonian directly as a :class:`~qiskit.quantum_info.SparsePauliOp`
                  over ``n`` qubits. It is expected to be a 1D nearest-neighbor
                  Pauli operator (single-qubit fields + two-qubit couplings on
                  adjacent sites) so Trotter synthesis stays at most 2-qubit and
                  the AQC MPS builder accepts every gate. Heisenberg / XXZ /
                  disorder are all just different ``SparsePauliOp`` values.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp


# Supported PauliEvolutionGate synthesis methods, keyed by `trotter_options["method"]`.
# Limited to deterministic product formulas that emit <=2-qubit gates — what the AQC MPS
# builder + step-to-step warm-start continuation require. QDrift is excluded because
# it is randomized, MatrixExponential because it is a dense n-qubit unitary.
TROTTER_METHODS = ("lie", "suzuki")


# ════════════════════════════════════════════════════════════════════════
#  Spec interface
# ════════════════════════════════════════════════════════════════════════


class HamiltonianSpec:
    """Base interface consumed by build/aqc/execute."""

    tier: str = "base"
    n: int

    def build_evolution(self, t_steps: int, dt: float) -> QuantumCircuit:
        """``n``-qubit time-evolution circuit for ``t_steps`` Trotter steps (product
        formula / order set by the spec). One step advances physical time ``dt``
        (``exp(-i*dt*H)``). ``t_steps == 0`` returns an empty circuit.
        """
        raise NotImplementedError


@dataclass
class Pauli1DNNSpec(HamiltonianSpec):
    """1D nearest-neighbor Pauli Hamiltonian, supplied as a ``SparsePauliOp``.

    ``operator`` is the Hamiltonian over ``n`` qubits, given directly as a
    :class:`~qiskit.quantum_info.SparsePauliOp`. It is expected to be a 1D
    nearest-neighbor Pauli operator — single-qubit fields and two-qubit
    couplings on adjacent sites — so that 2nd-order Trotter synthesis produces
    at-most-2-qubit gates that AQC's MPS builder accepts. Heisenberg, XXZ and
    disorder / random-field models are all just different ``operator`` values,
    e.g. built with ``SparsePauliOp.from_sparse_list``.

    Time evolution is built generically with ``PauliEvolutionGate`` and a Trotter
    product formula; one Trotter step = ``exp(-i*dt*H)``. ``trotter_options`` selects
    the synthesis — ``{"method": "lie"|"suzuki", "synthesis_settings": {...}}`` (default
    2nd-order Suzuki). ``synthesis_settings`` is that method's constructor kwargs;
    ``reps`` is owned here (it equals the step count) and not accepted from callers.
    """

    n: int
    operator: SparsePauliOp
    # PauliEvolutionGate synthesis: {"method": "lie"|"suzuki", "synthesis_settings": {...}}.
    # synthesis_settings is the method-specific constructor kwargs; reps is set per call.
    trotter_options: dict = dc_field(
        default_factory=lambda: {"method": "suzuki", "synthesis_settings": {"order": 2}}
    )
    tier: str = dc_field(default="pauli_1d_nn", init=False)

    def __post_init__(self) -> None:
        if self.n < 2:
            raise ValueError("Pauli1DNNSpec requires n >= 2.")
        if self.operator.num_qubits != self.n:
            raise ValueError(f"operator has {self.operator.num_qubits} qubits, expected {self.n}.")

    # Universal 1q/2q basis that both qiskit_quimb (AQC's MPS builder) and the
    # hardware path accept. Suzuki synthesis otherwise emits an `r` gate for
    # single-qubit field terms, which qiskit_quimb rejects.
    _QUIMB_BASIS = ["rx", "ry", "rz", "cx"]

    def build_evolution(self, t_steps: int, dt: float) -> QuantumCircuit:
        from qiskit import transpile
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.synthesis import LieTrotter, SuzukiTrotter

        qc = QuantumCircuit(self.n)
        if t_steps <= 0:
            return qc

        synthesis_classes = {"lie": LieTrotter, "suzuki": SuzukiTrotter}
        method = self.trotter_options.get("method", "suzuki")
        if method not in synthesis_classes:
            raise ValueError(
                f"Unknown trotter_options method {method!r}. Supported: {sorted(synthesis_classes)}."
            )
        synthesis_settings = self.trotter_options.get("synthesis_settings", {})
        gate = PauliEvolutionGate(
            self.operator,
            time=dt * t_steps,  # physical time: one Trotter step advances dt
            synthesis=synthesis_classes[method](reps=t_steps, **synthesis_settings),
        )
        qc.append(gate, range(self.n))
        # Translate to a 1q/2q basis (no coupling map -> qubit indices preserved)
        # so AQC's generate_ansatz_from_circuit reads the 2q connectivity and the
        # MPS builder accepts every gate. NN terms keep gates at most 2-qubit.
        return transpile(qc, basis_gates=self._QUIMB_BASIS, optimization_level=1)


_TIERS = {"pauli_1d_nn": Pauli1DNNSpec}


def make_spec(tier: str, n: int, **params) -> HamiltonianSpec:
    """Factory: build a :class:`HamiltonianSpec` from a tier name + params."""
    if tier not in _TIERS:
        raise ValueError(f"Unknown Hamiltonian tier {tier!r}. Supported: {sorted(_TIERS)}.")
    return _TIERS[tier](n=n, **params)
