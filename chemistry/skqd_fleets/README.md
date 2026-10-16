# SKQD Fleets Function Template

A **Sample-based Krylov Quantum Diagonalization (SKQD)** chemistry function that runs on
IBM Code Engine **Fleets** and scales across the CPU/GPU compute-profile ladder. It is built
on [`qiskit-fermions`](https://github.com/Qiskit/qiskit-fermions) for the quantum half and
[`qiskit-addon-sqd`](https://github.com/Qiskit/qiskit-addon-sqd) for the classical solve.

> [!TIP]
> This is the first *GPU* Fleets template in this repository. It runs on a GPU compute profile
> when one is requested, and falls back to CPU transparently otherwise.

## About

SKQD estimates a molecule's ground-state energy by combining a quantum sampling step with a
classical diagonalization step. It prepares a reference state, evolves it for increasing
times to build a *Krylov* family of circuits, samples bitstrings from each, and diagonalizes
the Hamiltonian classically in the subspace those configurations span. The result is a
*variational upper bound* on the true ground-state energy: it sits at or above the exact
value and improves as the sampled subspace grows richer.

## Methodology

The function walks the four Qiskit Pattern stages:

| Stage | Sub-status | What happens | Library |
|---|---|---|---|
| **Map** | `MAPPING` | PySCF SCF → MO integrals → a `FermionOperator`; an exact reference energy via a native FCI matvec. | pyscf / gpu4pyscf, qiskit-fermions |
| **Optimize** | `OPTIMIZING_HARDWARE` | Build the `krylov_dim` Krylov `FermionicCircuit`s and Jordan-Wigner-transpile them to qubit circuits. | qiskit-fermions |
| **Execute** | `EXECUTING_QPU` | Statevector-sample each circuit (a real backend would execute the transpiled circuits). | ffsim |
| **Post-process** | `POST_PROCESSING` | The classical subspace diagonalization — the memory/compute-heavy step. | qiskit-addon-sqd |

The two heavy stages — building/sampling the `krylov_dim` circuits and the classical
diagonalization — both scale with the active space. The diagonalization does FCI-style
matrix-vector products on a state vector of dimension
`C(norb, n_alpha) * C(norb, n_beta)`, which grows combinatorially; that is what makes a
large-memory / GPU compute profile worthwhile on larger molecules. `qiskit-fermions` itself
is CPU-only (its FCI matvec is a native Rust kernel); the GPU acceleration in this template
comes from `gpu4pyscf` accelerating the classical chemistry in the **Map** stage.

## Function usage

Every input is optional; the defaults run a linear H₄ chain in STO-3G. The inputs fall into
three groups: **what** you compute, **how hard** you compute it, and **where** it runs.

### System — *what* you solve

| Parameter | Type | Default | Description |
|---|---|---|---|
| `atom` | list | linear H₄ (1.0 Å spacing) | PySCF geometry `[[symbol, (x, y, z)], ...]`. A bigger molecule means more orbitals and a combinatorially larger problem. |
| `basis` | str | `"sto-3g"` | Gaussian basis set. Larger sets (`6-31g`, `cc-pvdz`) are more accurate and much heavier — this is where `gpu4pyscf` starts to pay off. |

### Convergence / cost — *how hard* you solve it

| Parameter | Type | Default | Description |
|---|---|---|---|
| `krylov_dim` | int | `5` | Number of Krylov circuits `D`. A larger value gives a richer subspace and a lower (better) energy, at the cost of more and deeper circuits. |
| `time_step` | float | `0.2` | Trotter time step `dt`. A convergence knob to tune, not simply raise (larger steps span more of the spectrum but add Trotter error). |
| `shots` | int | `1000` | Measurement shots per Krylov circuit. More shots sample more distinct configurations. |
| `samples_per_batch` | int | `100` | Configurations per SQD batch — the main memory dial for the diagonalization. |
| `num_batches` | int | `3` | Independent SQD batches per iteration — the parallel fan-out width; raise it to keep a large profile busy. |
| `max_iterations` | int | `3` | SQD configuration-recovery iterations. |
| `seed` | int | `24` | RNG seed for sampling and batching. |
| `use_gpu_scf` | bool | `True` | Run the SCF on `gpu4pyscf` when a GPU is attached (else CPU PySCF). |

### Where it runs — the compute profile

`function_size` selects a compute profile from the function's `sizes_map` (set at upload):
`"s"` is a CPU baseline; `"m"`/`"l"`/`"xl"` add GPUs. The same uploaded code runs on all of
them — the function probes the worker at runtime and uses the GPU only when one is attached.

## Output

`run` returns a dict:

| Key | Description |
|---|---|
| `energy` | The SKQD ground-state estimate (Hartree). |
| `reference_energy` | Exact diagonalization in the same active space, for comparison. |
| `error` | `abs(energy - reference_energy)`. SKQD is variational, so `energy >= reference_energy`. |
| `num_configurations` | Distinct bitstrings sampled (subspace size before batching). |
| `norb`, `nelec` | The active space. |
| `occupancies` | Per-orbital average occupancy in the recovered ground state. |
| `energy_history` | Best energy per SQD iteration. |
| `ran_on_gpu`, `gpu_check` | Whether a GPU was attached and the stack loaded on it. |
| `metadata.timings` | Per-stage wall-clock. |

## Dependencies

Default:

```
qiskit-fermions
qiskit-addon-sqd
qiskit-serverless
qiskit-ibm-runtime
```

Custom:

```
ffsim
pyscf
gpu4pyscf   # for the GPU SCF path; provided by the Fleets GPU runtime image
```

## Citing this project

If you use SKQD, please cite the method:

- Yu, J. et al. *Quantum-Centric Algorithm for Sample-Based Krylov Diagonalization.*
  [arXiv:2501.09702](https://arxiv.org/abs/2501.09702)
- Robledo-Moreno, J. et al. *Chemistry beyond exact solutions on a quantum-centric
  supercomputer.* [arXiv:2405.05068](https://arxiv.org/abs/2405.05068)

## References

1. `qiskit-fermions`: https://github.com/Qiskit/qiskit-fermions
2. `qiskit-addon-sqd`: https://github.com/Qiskit/qiskit-addon-sqd
3. `ffsim`: https://github.com/qiskit-community/ffsim
