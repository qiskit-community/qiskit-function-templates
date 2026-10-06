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
"""Sample-based Krylov Quantum Diagonalization (SKQD) with qiskit-fermions.

This is a faithful, parameterized port of the qiskit-fermions SKQD getting-started
guide (``docs/guides/skqd.rst``) applied to a small molecular system built at
runtime with PySCF. It is deliberately split into the stages a Fleets function
would fan out or accelerate:

* :func:`build_hamiltonian` -- classical chemistry (PySCF) -> electronic-structure
  integrals -> ``FermionOperator`` (both a "position"/AO-like and a sparsifying
  rotated basis). The classical mean-field step is where ``gpu4pyscf`` would
  accelerate on a GPU profile.
* :func:`krylov_circuits` -- build + Jordan-Wigner-transpile the family of ``D``
  Krylov time-evolution circuits. These are independent per Krylov power -> the
  embarrassingly-parallel fan-out unit.
* :func:`sample_circuits` -- statevector-sample each circuit with ffsim (a real
  backend would run the transpiled circuits instead).
* :func:`diagonalize` -- the classical subspace eigensolve
  (``qiskit_addon_sqd``). This is the memory/compute-heavy step whose FCI-style
  matvecs scale combinatorially with the active space -- the reason to reach for
  a large-memory / GPU compute profile.

The GPU story: qiskit-fermions itself is CPU-only (its FCI matvec is a native
Rust kernel), so the acceleration here comes from the base image's public GPU
stack -- ``gpu4pyscf`` for the chemistry and (as a drop-in HPC backend, out of
scope for this first demo) ``fulqrum``/``qiskit-addon-sqd-hpc`` for the
diagonalization matvec.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np


@dataclass
class SKQDResult:
    """Structured result of an SKQD run (JSON/serialization friendly)."""

    energy: float
    reference_energy: float
    error: float
    krylov_dim: int
    num_configurations: int
    norb: int
    nelec: tuple[int, int]
    occupancies: list[float] = field(default_factory=list)
    energy_history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "energy": float(self.energy),
            "reference_energy": float(self.reference_energy),
            "error": float(self.error),
            "krylov_dim": int(self.krylov_dim),
            "num_configurations": int(self.num_configurations),
            "norb": int(self.norb),
            "nelec": [int(self.nelec[0]), int(self.nelec[1])],
            "occupancies": [float(x) for x in self.occupancies],
            "energy_history": [float(x) for x in self.energy_history],
        }


# --------------------------------------------------------------------------- #
# 1. Classical chemistry -> Hamiltonian (both bases)
# --------------------------------------------------------------------------- #
def build_hamiltonian(
    atom: Sequence[Sequence[Any]],
    basis: str = "sto-3g",
    use_gpu: bool = False,
    logger: Any | None = None,
) -> dict[str, Any]:
    """Run RHF for ``atom`` and return the electronic-structure integrals.

    Returns a dict with ``norb``, ``nelec``, the one-/two-body integrals in the
    (Hartree-Fock molecular-orbital) basis, and the nuclear repulsion. The MO
    basis already concentrates the ground state far better than the AO basis --
    the "sparsifying basis change" the SKQD guide performs by hand for the SIAM
    model. For a molecule, the HF MO basis plays that role.

    ``use_gpu=True`` runs the SCF with ``gpu4pyscf`` (present in the base image)
    when a device is attached; otherwise it uses CPU PySCF.
    """
    import pyscf
    import pyscf.ao2mo
    import pyscf.gto
    import pyscf.scf

    mol = pyscf.gto.Mole()
    mol.build(atom=list(atom), basis=basis, symmetry=False, verbose=0)

    scf = None
    if use_gpu:
        try:
            from gpu4pyscf import scf as gpu_scf  # noqa: F401

            scf = gpu_scf.RHF(mol).run()
            if logger:
                logger.info("SCF ran on GPU via gpu4pyscf.")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if logger:
                logger.info("gpu4pyscf SCF unavailable (%s); falling back to CPU.", exc)

    if scf is None:
        scf = pyscf.scf.RHF(mol).run()

    # gpu4pyscf mirrors the CPU pyscf API but keeps arrays on the DEVICE (cupy).
    # A cupy array refuses implicit numpy conversion:
    #   TypeError: Implicit conversion to a NumPy array is not allowed.
    #             Please use `.get()` to construct a NumPy array explicitly.
    # so we must call `.get()` BEFORE any numpy op (including np.asarray, which
    # itself triggers the guard). _to_host normalizes device -> host once, up
    # front; everything downstream is then plain numpy on the CPU.
    def _to_host(arr):
        return arr.get() if hasattr(arr, "get") else np.asarray(arr)

    mo_coeff = _to_host(getattr(scf, "mo_coeff"))

    norb = mo_coeff.shape[1]
    nelec = (int(mol.nelec[0]), int(mol.nelec[1]))

    # Build the integrals in the MO basis on the CPU. mol.intor / ao2mo are the
    # plain CPU pyscf routines (mol is a pyscf.gto.Mole even under gpu4pyscf), so
    # with a host-side mo_coeff the whole integral transform stays on host.
    hcore = np.asarray(mol.intor("int1e_kin") + mol.intor("int1e_nuc"))
    h1e = mo_coeff.T @ hcore @ mo_coeff
    eri = _to_host(pyscf.ao2mo.kernel(mol, mo_coeff))
    h2e = pyscf.ao2mo.restore(1, eri, norb)
    nuclear_repulsion = float(mol.energy_nuc())

    return {
        "norb": norb,
        "nelec": nelec,
        "h1e": np.ascontiguousarray(_to_host(h1e), dtype=float),
        "h2e": np.ascontiguousarray(_to_host(h2e), dtype=float),
        "nuclear_repulsion": nuclear_repulsion,
    }


def fermion_operator_from_integrals(h1e: np.ndarray, h2e: np.ndarray, norb: int):
    """Assemble a ``FermionOperator`` from MO integrals (spin-symmetric)."""
    from qiskit_fermions.operators import FermionOperator

    one_body = FermionOperator.from_1body_tril_spin_sym(h1e[np.tril_indices(norb)], norb)
    # Two-body integrals packed into 8-fold lower-triangular ordering.
    tril = np.tril_indices(norb)
    h2e_tril = h2e[tril[0], tril[1]][:, tril[0], tril[1]]
    s8 = h2e_tril[np.tril_indices(h2e_tril.shape[0])]
    two_body = FermionOperator.from_2body_tril_spin_sym(s8, norb)
    return (one_body + two_body).normal_ordered().simplify(atol=1e-12)


def reference_energy(h1e: np.ndarray, h2e: np.ndarray, norb: int, nelec: tuple[int, int]) -> float:
    """Exact ground-state energy of the (norb, nelec) sector via eigsh.

    Uses the FermionOperator's ``_linear_operator_`` protocol method -- a scipy
    ``LinearOperator`` backed by the native FCI matvec kernel. This is the CPU
    primitive whose cost scales as C(norb, n_a) * C(norb, n_b): the honest reason
    a large active space wants a large-memory / GPU compute profile.

    Note: we call ``operator._linear_operator_(norb, nelec)`` directly rather
    than the ``qiskit_fermions.linalg.linear_operator`` free function. The free
    function is a thin wrapper (`return operator._linear_operator_(...)`) added
    after the 0.2.0 release, so importing it fails on the pinned 0.2.0
    (`ImportError: cannot import name 'linear_operator'`). The protocol method
    ships in 0.2.0 (its README lists ``_linear_operator_``) and in newer versions
    alike, so calling it directly is version-safe.
    """
    import scipy.sparse.linalg

    hamiltonian = fermion_operator_from_integrals(h1e, h2e, norb)
    linop = hamiltonian._linear_operator_(norb, nelec)  # pylint: disable=protected-access
    evals, _ = scipy.sparse.linalg.eigsh(linop, k=1, which="SA")
    return float(evals[0])


# --------------------------------------------------------------------------- #
# 2. Krylov circuit family -- the fan-out unit
# --------------------------------------------------------------------------- #
def build_krylov_circuit(dim: int, norb: int, nelec: tuple[int, int], h1e: np.ndarray, time_step: float):
    """Build one untranspiled ``FermionicCircuit`` for Krylov power ``dim``.

    A reference Slater determinant followed by a ``dim``-step second-order
    Trotter product of ``e^{-i t H_1}`` (a single-particle OrbitalRotation).
    This minimal one-body-only evolution keeps the demo self-contained; the full
    guide additionally Trotterizes the two-body interaction term.
    """
    import scipy.linalg
    from qiskit_fermions.circuit import FermionicCircuit
    from qiskit_fermions.circuit.library import OrbitalRotation, PrepareSlaterDeterminant

    num_modes = 2 * norb
    occupation = [i < nelec[0] for i in range(norb)]
    reference = PrepareSlaterDeterminant(occupation, np.eye(norb, dtype=complex))
    full_exp_h1 = OrbitalRotation(scipy.linalg.expm(-1j * time_step * h1e))

    circuit = FermionicCircuit(num_modes)
    circuit.append(reference, range(norb))
    circuit.append(reference, range(norb, num_modes))
    for _ in range(dim):
        circuit.append(full_exp_h1, range(norb))
        circuit.append(full_exp_h1, range(norb, num_modes))
    return circuit


def krylov_circuits(
    krylov_dim: int,
    norb: int,
    nelec: tuple[int, int],
    h1e: np.ndarray,
    time_step: float,
    map_fn: Callable[[Callable, list], list] | None = None,
):
    """Build (and JW-transpile) the family of ``krylov_dim`` Krylov circuits.

    ``map_fn`` is the parallelism hook: pass a fleet/Ray fan-out to build the
    independent circuits concurrently. Defaults to a serial list comprehension.
    """
    from qiskit_fermions.transpiler.presets import generate_preset_jw_pass_manager

    pass_manager = generate_preset_jw_pass_manager(optimization_level=0)

    def _one(dim: int):
        fermionic = build_krylov_circuit(dim, norb, nelec, h1e, time_step)
        transpiled = pass_manager.run(fermionic)
        transpiled.measure_all()
        return fermionic, transpiled

    dims = list(range(krylov_dim))
    if map_fn is None:
        pairs = [_one(d) for d in dims]
    else:
        pairs = map_fn(_one, dims)
    fermionic_circuits = [p[0] for p in pairs]
    transpiled_circuits = [p[1] for p in pairs]
    return fermionic_circuits, transpiled_circuits


# --------------------------------------------------------------------------- #
# 3. Sampling (statevector; a real backend would execute the transpiled circuits)
# --------------------------------------------------------------------------- #
def sample_circuits(fermionic_circuits, norb: int, nelec: tuple[int, int], shots: int = 1000) -> Counter:
    """Statevector-sample each Krylov circuit with ffsim, pooling the counts."""
    import ffsim

    reference = ffsim.slater_determinant(norb, (list(range(nelec[0])), list(range(nelec[1]))))
    counts: Counter = Counter()
    for dim, fermionic_circuit in enumerate(fermionic_circuits):
        statevector = ffsim.apply_unitary(reference, fermionic_circuit, norb=norb, nelec=nelec)
        samples = ffsim.sample_state_vector(statevector, norb=norb, nelec=nelec, shots=shots, seed=dim)
        counts += Counter(samples)
    return counts


# --------------------------------------------------------------------------- #
# 4. Classical subspace diagonalization -- the heavy step
# --------------------------------------------------------------------------- #
def diagonalize(
    counts: Counter,
    h1e: np.ndarray,
    h2e: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    samples_per_batch: int = 100,
    num_batches: int = 3,
    max_iterations: int = 3,
    seed: int = 24,
) -> tuple[Any, list]:
    """Run the sample-based diagonalization (qiskit-addon-sqd).

    ``num_batches`` is the second parallelism axis: each batch is an independent
    subspace diagonalization, so on a large compute profile you would raise it
    to keep the workers busy (mirroring the sqd_pcm template's
    ``@distribute_task`` fan-out).
    """
    from qiskit.primitives import BitArray
    from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian

    bit_array = BitArray.from_counts(dict(counts))
    history: list = []
    result = diagonalize_fermionic_hamiltonian(
        h1e,
        h2e,
        bit_array,
        samples_per_batch=samples_per_batch,
        norb=norb,
        nelec=nelec,
        num_batches=num_batches,
        max_iterations=max_iterations,
        symmetrize_spin=True,
        callback=history.append,
        seed=np.random.default_rng(seed),
    )
    return result, history
