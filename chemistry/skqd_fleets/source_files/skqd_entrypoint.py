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
"""SKQD Fleets Function Template entrypoint.

Runs Sample-based Krylov Quantum Diagonalization (SKQD) for a small molecule,
built on ``qiskit-fermions`` for the quantum half. The four stages map onto the
Qiskit Pattern sub-statuses:

- **Map** (``MAPPING``): PySCF SCF -> MO integrals -> ``FermionOperator``; an
  exact reference energy via the native FCI matvec.
- **Optimize** (``OPTIMIZING_HARDWARE``): build + Jordan-Wigner-transpile the
  family of ``krylov_dim`` Krylov time-evolution circuits.
- **Execute** (``EXECUTING_QPU``): statevector-sample each circuit with ffsim.
- **Post-process** (``POST_PROCESSING``): the classical subspace diagonalization
  (``qiskit-addon-sqd``) -- the memory/compute-heavy step.

Designed to run on a Fleets GPU compute profile: the same code runs on CPU and
GPU, probing the worker at runtime (``gpu_check``) and using ``gpu4pyscf`` for
the classical chemistry only when a device is attached. See the README for the
tunable knobs.
"""
from __future__ import annotations

import time
import traceback

from qiskit_serverless import get_arguments, save_result, update_status, Job, get_logger

from skqd import (
    build_hamiltonian,
    reference_energy,
    krylov_circuits,
    sample_circuits,
    diagonalize,
)
from gpu_probe import gpu_check

logger = get_logger()

# Default demo molecule: a linear H4 chain in STO-3G. Small and fast, but
# correlated enough for a meaningful SKQD test. Override via `atom` / `basis`.
DEFAULT_ATOM = [["H", (0.0, 0.0, 1.0 * i)] for i in range(4)]


def run_function(
    atom: list | None = None,
    basis: str = "sto-3g",
    krylov_dim: int = 5,
    time_step: float = 0.2,
    shots: int = 1000,
    samples_per_batch: int = 100,
    num_batches: int = 3,
    max_iterations: int = 3,
    seed: int = 24,
    use_gpu_scf: bool = True,
    **kwargs,
) -> dict:
    """Run the SKQD pipeline and return the energy estimate and diagnostics.

    Args:
        atom: PySCF geometry ``[[symbol, (x, y, z)], ...]``. Defaults to linear H4.
        basis: Gaussian basis set (e.g. ``"sto-3g"``, ``"6-31g"``).
        krylov_dim: number of Krylov circuits ``D``.
        time_step: Trotter time step ``dt``.
        shots: measurement shots per Krylov circuit.
        samples_per_batch: configurations per SQD batch.
        num_batches: independent SQD batches per iteration (parallel fan-out width).
        max_iterations: SQD configuration-recovery iterations.
        seed: RNG seed for sampling and batching.
        use_gpu_scf: run the SCF on ``gpu4pyscf`` when a GPU is present.

    Returns:
        A dict with the SKQD ``energy``, the exact ``reference_energy``, the
        ``error``, sampling/GPU diagnostics, and per-stage ``timings``.
    """
    timings: dict[str, float] = {}
    t0 = time.time()
    atom = atom if atom is not None else DEFAULT_ATOM

    # Probe the worker: does this compute profile actually have a GPU?
    gpu = gpu_check()
    logger.info("GPU check: %s", gpu)
    on_gpu = gpu.get("device_count", 0) > 0

    # Step 1: Map -- classical chemistry + Hamiltonian.
    update_status(Job.MAPPING)
    t = time.time()
    chem = build_hamiltonian(atom, basis=basis, use_gpu=use_gpu_scf and on_gpu, logger=logger)
    norb, nelec = chem["norb"], chem["nelec"]
    h1e, h2e = chem["h1e"], chem["h2e"]
    logger.info("System: norb=%s nelec=%s basis=%s", norb, nelec, basis)
    ref_energy = reference_energy(h1e, h2e, norb, nelec) + chem["nuclear_repulsion"]
    timings["map"] = time.time() - t

    # Step 2: Optimize -- build + transpile the Krylov circuit family.
    update_status(Job.OPTIMIZING_HARDWARE)
    t = time.time()
    fermionic_circuits, _transpiled = krylov_circuits(krylov_dim, norb, nelec, h1e, time_step)
    timings["build_circuits"] = time.time() - t

    # Step 3: Execute -- sample the Krylov circuits (statevector here).
    update_status(Job.EXECUTING_QPU)
    t = time.time()
    counts = sample_circuits(fermionic_circuits, norb, nelec, shots=shots)
    timings["sample"] = time.time() - t
    logger.info("Sampled %s distinct configurations.", len(counts))

    # Step 4: Post-process -- the heavy classical subspace diagonalization.
    update_status(Job.POST_PROCESSING)
    t = time.time()
    result, history = diagonalize(
        counts,
        h1e,
        h2e,
        norb,
        nelec,
        samples_per_batch=samples_per_batch,
        num_batches=num_batches,
        max_iterations=max_iterations,
        seed=seed,
    )
    timings["diagonalize"] = time.time() - t

    import numpy as np

    sqd_energy = float(result.energy) + chem["nuclear_repulsion"]
    try:
        occupancies = [float(x) for x in np.sum(result.orbital_occupancies, axis=0)]
    except Exception:  # pylint: disable=broad-exception-caught
        occupancies = []
    energy_history = [float(min(batch, key=lambda r: r.energy).energy) for batch in history]

    logger.info(
        "SKQD energy: %.6f  reference: %.6f  error: %.2e",
        sqd_energy,
        ref_energy,
        abs(sqd_energy - ref_energy),
    )

    return {
        "energy": sqd_energy,
        "reference_energy": ref_energy,
        "error": abs(sqd_energy - ref_energy),
        "krylov_dim": krylov_dim,
        "num_configurations": len(counts),
        "norb": norb,
        "nelec": [nelec[0], nelec[1]],
        "occupancies": occupancies,
        "energy_history": energy_history,
        "gpu_check": gpu,
        "ran_on_gpu": on_gpu,
        "metadata": {
            "resources_usage": {
                "RUNNING: POST_PROCESSING": {"CPU_TIME": time.time() - t0},
            },
            "timings": timings,
        },
    }


if __name__ == "__main__":
    input_args = get_arguments()
    try:
        func_result = run_function(**input_args)
        save_result(func_result)
    except Exception:  # pylint: disable=broad-exception-caught
        save_result(traceback.format_exc())
        raise
