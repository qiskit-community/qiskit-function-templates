# AQC+Trotter Hamiltonian Dynamics Template

> [!TIP]
> [Download the AQC+Trotter Hamiltonian Dynamics template](https://github.com/qiskit-community/qiskit-function-templates/tree/main/physics/trotter_aqc_hamiltonian)
>
> [Deploy and run the template](https://github.com/qiskit-community/qiskit-function-templates/blob/main/physics/trotter_aqc_hamiltonian/deploy_and_run.ipynb)

## Overview

The function takes a state, a Hamiltonian and a list of observables, and returns how
those observables evolve in time — using AQC to keep the executed circuits shallow.
Swap the setup (pre) and the analysis (post) and the *same* core drives a different
experiment:

```
   PRE (your setup)            FUNCTION (this repo)                POST (your analysis)
 prepare a state         →  Trotter → AQC compress → execute   →  structure factors,
 (circuit / product)        (statevector / fake / runtime)         magnetization, transport,
 + optional local kick      → ⟨O⟩(t)                                quench dynamics, …
```

The function itself is experiment-agnostic: it consumes a Hamiltonian, a state and a
list of observables, and returns a time series. The physics-specific setup and analysis
stay on your side of that boundary.

### Why AQC

Plain Trotter evolution costs you circuit depth linear in the number of steps: reaching
step `k` means executing `k` copies of the per-step brickwork. On hardware that depth is
what eventually buries the signal in noise.

AQC ([approximate quantum compiling](https://qiskit.github.io/qiskit-addon-aqc-tensor/))
trades classical work for that depth. For each of the leading time steps, it classically
optimises a *fixed, shallow* parametrised ansatz until its state matches the deep Trotter
target, scoring the match as an MPS fidelity with a tensor-network simulator. The circuit
you actually execute at step `k` is then the shallow ansatz — its depth does not grow
with `k`.

The trade has two sides worth knowing before you commit:

- **You pay in classical time and memory.** Compression runs one optimisation per
  compressed step, and its memory footprint grows steeply with
  [`aqc_options.max_bond`](#aqc_options). This is the dominant classical cost of a run.
- **It only works while the target state stays MPS-representable.** Entanglement grows
  with time, so a given ansatz holds high fidelity for a while and then degrades. That
  is exactly what [`aqc_segments`](#aqc_segments) exists to manage — and why later steps
  need deeper ansätze, or plain Trotter.

Every run reports the per-step fidelities it achieved and the depth it saved
(`metadata.aqc_fidelities` and `metadata.circuit_stats`), so the trade is measurable
rather than assumed.

## Quickstart

1. **Deploy the function on your serverless account:** Run
   [`deploy_and_run.ipynb`](deploy_and_run.ipynb) top to bottom — authenticate, declare
   dependencies, upload the source. Deploy once, then run it as many times as you like.
2. **Adapt it:** the same notebook ends with a worked `fn.run(...)` call you can edit
   into your own experiment — swap the Hamiltonian, initial state and observables.

The rest of this README is reference: the [input contract](#function-usage),
[output shape](#output), and [execution backends](#execution-backends).

## Dependencies

Default (already in the managed serverless image):
```
qiskit
qiskit-serverless
qiskit-ibm-runtime
```

Custom (declared at upload time — see [`deploy_and_run.ipynb`](deploy_and_run.ipynb)):
```
qiskit-addon-aqc-tensor[quimb-jax]==0.3.1
qiskit-aer==0.17.2
cotengrust==0.2.0
```

The `[quimb-jax]` extra is what pulls in `quimb` and `jax`. `cotengrust` is the Rust
contraction-path backend quimb picks up automatically; without it, compression falls
back to a far more memory-heavy path and the run records a warning in
`metadata.warnings` rather than failing.

## Function usage

```python
from qiskit.quantum_info import SparsePauliOp

# Hamiltonian as a SparsePauliOp on n qubits (isotropic Heisenberg chain here)
n = 8
H = SparsePauliOp.from_sparse_list(
    [(p, [i, i + 1], 0.5) for i in range(n - 1) for p in ("XX", "YY", "ZZ")],
    num_qubits=n,
)
job = fn.run(t_steps=8, aqc_segments=[{"n_steps": 3, "ansatz_steps": 1}], hamiltonian=H)
result = job.result()
```

**Required keys: `hamiltonian`, `t_steps`, `aqc_segments`.** Everything else has a
default. The Hamiltonian's `num_qubits` fixes the chain length.

### Inputs

| parameter | type | valid values | required | default | description |
|-----------|------|--------------|:--------:|---------|-------------|
| `t_steps` | `int` | `>=1` | **yes** | — | Total Trotter steps. Evolves to `T = t_steps · dt` and reports every observable at each `t_k = k·dt` (`t_0` is the prepared state before any evolution). |
| `aqc_segments` | `list` of `{n_steps, ansatz_steps}` | non-empty; `sum(n_steps) ≤ t_steps` | **yes** | — | AQC compression plan. Each segment compresses its `n_steps` leading time steps into an ansatz generated from the `ansatz_steps`-Trotter target. `sum(n_steps)` steps are compressed; the rest run as plain Trotter ([detail below](#aqc_segments)). |
| `dt` | `float` | `>0` | no | `0.2` | Physical time advanced by one Trotter step (`exp(-i·dt·H)`). |
| `hamiltonian` | `SparsePauliOp` | — | **yes** | — | 1-D nearest-neighbour Pauli Hamiltonian, given as a `SparsePauliOp`. Its `num_qubits` sets the chain length — there is no separate `n` input ([detail below](#hamiltonian)). |
| `initial_state` | `QuantumCircuit` | — | no | `\|0…0⟩` | Prepared state to evolve; bake any local kick into this circuit ([detail below](#initial_state)). |
| `observables` | `EstimatorV2` observables \| `null` | — | no | single-site `Z` | Exactly what you pass `EstimatorV2` PUB as `observables` — `SparsePauliOp` / `Pauli` / `PauliList` / Pauli string / `{pauli: coeff}` / (nested) list; one observable per output column ([detail below](#observables)). |
| `trotter_options` | `object` | — | no | 2nd-order Suzuki | Trotter product-formula synthesis: `{"method": ..., "synthesis_settings": {...}}` ([detail below](#trotter_options)). |
| `aqc_options` | `object` | — | no | defaults | AQC compression tuning ([detail below](#aqc_options)). |
| `estimator_options` | `object` | — | no | DD/twirling/TREX stack | `EstimatorV2.options`, passed through as-is — any field configurable ([detail below](#estimator_options)). |
| `transpiler_options` | `object` | — | no | `{"optimization_level": 3}` | `generate_preset_pass_manager` kwargs (fake/runtime), passed through as-is — any field except `backend`/`target` ([detail below](#transpiler_options)). |
| `backend` | `str` | `statevector` \| `fake` \| `runtime` | no | `"runtime"` | Execution mode (see [Execution backends](#execution-backends)). Default is `runtime` (real hardware, needs credentials); pass `"statevector"` for an exact, no-credentials local run. |
| `backend_name` | `str` or `null` | — | no | `least_busy()` | IBM backend name for `runtime` (or a named fake); `null` selects the least-busy device. |
| `batches` | `int` | `>=1` | no | `1` | Split across N runtime jobs; `1` = one job holding every PUB (`runtime` backend only) |
| `parallel_sim` | `bool` | `true` \| `false` | no | `false` | Fan the local-sim execution (`statevector`/`fake`) across all available cores via Ray — each time-step circuit runs as its own task. `false` (default) runs them sequentially. No effect on `runtime`. |

<a id="aqc_segments"></a>
#### `aqc_segments`

An ordered list of segments. Each compresses its `n_steps` consecutive leading time steps
into one ansatz, generated from the `ansatz_steps`-Trotter target — so `ansatz_steps`
sets the ansatz *depth*, and `n_steps` sets how many time steps reuse it. Larger
`ansatz_steps` means more parameters, more optimisation time, and fidelity that holds
further out in time.

Steps are assigned to segments in order. `sum(n_steps)` must be `≤ t_steps`; any
remaining steps run as plain Trotter appended to the last compressed circuit. A
single-ansatz run is just one segment: `[{"n_steps": 5, "ansatz_steps": 1}]`.

**Picking the numbers.** There is no good universal default — the right plan depends on
how fast your model builds entanglement. The practical loop:

1. Start with one segment, `ansatz_steps: 1`, and `n_steps` set to however many steps you
   want compressed. Run it on `backend="statevector"` — it is exact, free and needs no
   credentials.
2. Read `metadata.aqc_fidelities` from the result. It reports the achieved fidelity per
   compressed step, and it falls as the step index grows.
3. Find the step where fidelity drops below your tolerance (~0.99 is a reasonable line).
   Split there: give the steps before it their own segment, and start a new segment at
   that point with a larger `ansatz_steps`.
4. Repeat until the fidelities across all segments hold. Then switch `backend` to `fake`
   or `runtime`.

`metadata.aqc_segments` echoes the plan back with each segment's step range, parameter
count and its own fidelities, so you can see exactly where a segment started to struggle.

**Cost.** Compression time scales with the number of compressed steps and with the
ansatz parameter count; memory is governed by [`aqc_options.max_bond`](#aqc_options).
Compressing every step is rarely worth it — the late steps are the expensive ones to fit
and the cheapest to just run as Trotter. Check `metadata.circuit_stats`, which reports
2-qubit depth and gate count for full Trotter versus the AQC circuit at every step, to
confirm the compression is actually buying you depth.

<a id="hamiltonian"></a>
#### `hamiltonian`
The Hamiltonian is a [`SparsePauliOp`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.SparsePauliOp) over `n` qubits, passed natively (Qiskit
Serverless serialises it over the wire). It should be a 1-D nearest-neighbour Pauli
operator — single-qubit fields and two-qubit couplings on adjacent sites — so that
2nd-order Trotter synthesis stays at most 2-qubit and AQC's MPS builder accepts every
gate.

Strings are Pauli/σ operators, not spin-½ — there is no implicit factor of ½.
`SparsePauliOp.from_sparse_list` is the convenient constructor; each entry is
`(pauli, [sites], coeff)`. A few ready models on `n` qubits:

```python
from qiskit.quantum_info import SparsePauliOp

# Heisenberg: XX+YY+ZZ at 0.5 on every nearest-neighbour bond
H = SparsePauliOp.from_sparse_list(
    [(p, [i, i + 1], 0.5) for i in range(n - 1) for p in ("XX", "YY", "ZZ")],
    num_qubits=n,
)

# XXZ (anisotropic): ZZ coupling differs from XX/YY
H = SparsePauliOp.from_sparse_list(
    [(p, [i, i + 1], c) for i in range(n - 1)
     for p, c in [("XX", 0.5), ("YY", 0.5), ("ZZ", 0.9)]],
    num_qubits=n,
)

# transverse-field Ising: ZZ coupling + uniform X field
H = SparsePauliOp.from_sparse_list(
    [("ZZ", [i, i + 1], 1.0) for i in range(n - 1)] + [("X", [i], 0.8) for i in range(n)],
    num_qubits=n,
)
```

Non-uniform chains: Because you build the operator directly, bond disorder or a
random-field model is just a different `SparsePauliOp` — vary the coefficient per bond
or per site:

```python
# n=6 random-field Heisenberg: uniform coupling, site-dependent Z field
h = [0.1, -0.2, 0.05, -0.1, 0.15, -0.05]                      # one value per site (len n)
H = SparsePauliOp.from_sparse_list(
    [(p, [i, i + 1], 0.5) for i in range(n - 1) for p in ("XX", "YY", "ZZ")]
    + [("Z", [i], h[i]) for i in range(n)],
    num_qubits=n,
)

# n=5 bond-disordered ZZ chain: coupling differs per bond
J = [1.0, 0.6, 1.2, 0.8]                                      # one value per bond (len n-1)
H = SparsePauliOp.from_sparse_list(
    [("ZZ", [i, i + 1], J[i]) for i in range(n - 1)], num_qubits=n,
)
```

<a id="initial_state"></a>
#### `initial_state`
A prepared [`QuantumCircuit`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.QuantumCircuit) passed natively (default `|0…0⟩` when omitted). This is
the direct form for Python callers handing in a ground state they optimised
themselves (DMRG / VQE / …); Qiskit Serverless QPY-serialises it over the wire.

The function never builds model-specific states for you. To start from a
computational-basis product state (e.g. a Néel state), build a tiny circuit of `X`
gates — `qc.x(i)` puts qubit `i` in `|1⟩`, and that same index `i` is what observables
refer to.

```python
from qiskit import QuantumCircuit

neel = QuantumCircuit(n)
for i in range(0, n, 2):        # X on even sites -> |0101…01⟩
    neel.x(i)
neel.rz(1.5708, n // 2)         # optional local kick: Z-rotation by π/2 at the centre
# → pass it in the call: fn.run(..., initial_state=neel)
```

<a id="observables"></a>
#### `observables`
Pass exactly what you'd give EstimatorV2 PUB as its `observables` argument  — anything
Qiskit's `ObservablesArray.coerce` accepts: a `SparsePauliOp`, [`Pauli`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.Pauli), [`PauliList`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.PauliList),
Pauli string, or `{pauli: coeff}` mapping, or a (nested) list of those. Each element is
measured separately and becomes one output column (`EstimatorV2` returns one
expectation value per element the same way).

Pauli strings follow Qiskit's ordering (leftmost char = highest qubit index) and cover
all `n` qubits. Labels: a single unit-coefficient Pauli is labelled by its string;
anything else (a multi-term operator) is labelled `obs_0`, `obs_1`, …. Use
`SparsePauliOp.from_sparse_list` for site-local operators or correlators without writing
the full-width string:

```python
from qiskit.quantum_info import SparsePauliOp

observables=["IIIZ", "IIZZ"]                                    # Pauli strings
observables=[{"IIIZ": 1.0}, {"IIZZ": 1.0}]                      # {pauli: coeff} dicts
observables=[                                                   # native, site-local
    SparsePauliOp.from_sparse_list([("Z",  [0],     1.0)], num_qubits=n),
    SparsePauliOp.from_sparse_list([("XX", [0, n-1], 0.5)], num_qubits=n),
]
```

Because each observable is one column, a single multi-term `SparsePauliOp` sums into one
column — pass a *list* when you want a separate series per operator.

<a id="trotter_options"></a>
#### `trotter_options`

The function evolves the state with a [`PauliEvolutionGate`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.library.PauliEvolutionGate), whose product-formula synthesis
you would normally construct yourself. In a plain Qiskit script, a 4th-order Suzuki–Trotter
evolution over `t_steps` steps of size `dt` looks like:

```python
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter

gate = PauliEvolutionGate(H, time=dt * t_steps, synthesis=SuzukiTrotter(order=4, reps=t_steps))
```

To get that *same* synthesis through the function, name the class with `method` and pass the
arguments you'd give its constructor as `synthesis_settings`:

```python
fn.run(..., trotter_options={"method": "suzuki", "synthesis_settings": {"order": 4}})
```

The function assembles the `PauliEvolutionGate` for you: it fills in `time` (`= dt × t_steps`)
and `reps` (`= the Trotter step count`, one repetition per `dt`), so those two are **not**
accepted in `synthesis_settings` — everything else the synthesis class takes is passed through.

| `method` | synthesis class | `synthesis_settings` (its constructor args) |
|----------|-----------------|----------------------------------------------|
| `"suzuki"` (default) | [`SuzukiTrotter`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.synthesis.SuzukiTrotter) | `order` (`2` default; `1` ≡ Lie–Trotter, higher even orders are deeper / more accurate), `cx_structure`, `insert_barriers`, … |
| `"lie"` | [`LieTrotter`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.synthesis.LieTrotter) | `cx_structure`, `insert_barriers`, … (first order; no `order`) |

Only these deterministic, ≤2-qubit product formulas are supported: the AQC MPS builder needs
≤2q gates and its step-to-step warm-start needs deterministic targets. Randomized `QDrift`
and dense `MatrixExponential` are therefore excluded.

```python
# first-order Lie–Trotter instead of the default 2nd-order Suzuki:
fn.run(..., trotter_options={"method": "lie"})
```

<a id="aqc_options"></a>
#### `aqc_options`

These knobs tune the [`qiskit-addon-aqc-tensor`](https://qiskit.github.io/qiskit-addon-aqc-tensor/) MPS-fidelity compression. Ansatz depth per step-range is set by [`aqc_segments`](#aqc_segments) (each segment's `ansatz_steps`), not here:

| field | default | meaning |
|-------|---------|---------|
| `max_bond` | `32` | MPS bond dimension cap during fidelity optimisation|
| `cutoff` | `1e-8` | MPS singular-value truncation cutoff |
| `autodiff_backend` | `"jax"` | gradient backend (`"jax"` or `"explicit"`) |
| `fidelity_target` | `null` | if set (0–1], early-stop each step once this fidelity is reached |
| `optimizer_settings` | `{"method": "L-BFGS-B", "jac": true, "options": {"maxiter": 300}}` | passed straight to [`scipy.optimize.minimize(objective, x0, **optimizer_settings)`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) — set `method`, `jac`, `tol`, `options` (e.g. `maxiter`), `bounds`, … |

`max_bond` is the main memory and time dial: it caps how much entanglement the MPS can
represent, so raising it improves fidelity on harder states at a steep cost. Raise it
only when `metadata.aqc_fidelities` says you need to, and keep `cotengrust` installed —
it is what keeps the contraction affordable at higher bond dimensions.

<a id="estimator_options"></a>
#### `estimator_options`

Applied on the `fake` / `runtime` paths. This is a plain `EstimatorV2.options` mapping
handed straight to the estimator, so any [`EstimatorOptions`](https://quantum.cloud.ibm.com/docs/api/qiskit-ibm-runtime/options-estimator-options) field is configurable —
not just the ones below. Omit it for the default mitigation stack; pass a dict to
replace it wholesale. The defaults are:

| field | default | meaning |
|-------|---------|---------|
| `twirling.enable_gates` | `true` | gate Pauli twirling on/off |
| `twirling.num_randomizations` | `1000` | Pauli-twirling randomizations |
| `twirling.shots_per_randomization` | `128` | shots per randomization |
| `dynamical_decoupling.enable` | `true` | dynamical decoupling on/off |
| `dynamical_decoupling.sequence_type` | `"XY4"` | DD pulse sequence |
| `resilience.measure_mitigation` | `true` | TREX measurement mitigation |

<a id="transpiler_options"></a>
#### `transpiler_options`

Kwargs handed straight to [`generate_preset_pass_manager(backend=…, **transpiler_options)`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.transpiler.generate_preset_pass_manager)
on the `fake` / `runtime` paths, so any preset-pass-manager keyword works — not just
`optimization_level`. `backend` and `target` are set by the execution path and rejected
here. Defaults to `{"optimization_level": 3}`.

| common field | default | meaning |
|-------|---------|---------|
| `optimization_level` | `3` | preset pass-manager level (0–3) |
| `seed_transpiler` | `null` | deterministic layout/routing |
| `layout_method` / `routing_method` | preset | e.g. `"sabre"` |
| `translation_method` | preset | gate-translation plugin |
| `approximation_degree` | `1.0` | 1q/2q synthesis approximation |

### Example

A prepared ground state with a local kick, compressed in two segments — the first 4 steps
into a shallow 1-layer ansatz, the next 2 into a deeper 2-layer ansatz that holds fidelity
for the later, more-entangled states — run on the noisy `fake` backend:

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

n = 10
gs = QuantumCircuit(n)
# ... prepare your ground state here (e.g. a DMRG/VQE circuit) ...
gs.rz(1.5708, 5)               # local Z-kick (π/2) at the chain centre, before evolution

H = SparsePauliOp.from_sparse_list(                                  # Heisenberg
    [(p, [i, i + 1], 0.5) for i in range(n - 1) for p in ("XX", "YY", "ZZ")],
    num_qubits=n,
)
job = fn.run(
    t_steps=20, dt=0.2,
    aqc_segments=[
        {"n_steps": 4, "ansatz_steps": 1},   # first 4 steps → 1-layer ansatz
        {"n_steps": 2, "ansatz_steps": 2},   # next 2 → 2-layer ansatz
    ],
    hamiltonian=H,
    initial_state=gs,
    # observables omitted -> default per-site Z
    backend="fake",
)
```

## Output

`fn.run(...)` hands back a job you can poll before the result is ready — `job.status()`
reports the stage the function is in, and `job.logs()` gives the per-stage detail. The
[deploy notebook](deploy_and_run.ipynb) walks through both, with the full status
lifecycle. A rejected input fails fast as a structured `ServerlessError` (code `4615`).

`job.result()` is a classical dict:

```python
{
  "times": [...],                  # length t_steps+1, t_k = k·dt (t=0 is the prepared state)
  "expectation_values": [[...]],   # shape (n_times, n_observables)
  "observable_labels": [...],      # e.g. ["Z_0", "ZZ_0_1"]
  "metadata": {
      "n", "t_steps", "dt", "tier",
      "aqc_compressed_steps": 5,   # total compressed steps (= sum of segment n_steps)
      "aqc_segments": [            # per segment: the plan + its own results
          {"n_steps": 3, "ansatz_steps": 1, "steps": [1, 2, 3], "n_params": 133,
           "fidelities": {1: ..., 2: ..., 3: ...}},
          {"n_steps": 2, "ansatz_steps": 2, "steps": [4, 5], "n_params": 245,
           "fidelities": {4: ..., 5: ...}},
      ],
      "execution_backend",
      "aqc_fidelities": {1: ..., 2: ...},   # flat per-step fidelity, full series (all compressed steps)
      "circuit_stats": {            # per-step 2q depth/gate-count, full Trotter vs AQC+Trotter
          1: {"full_trotter": {"depth_2q": ..., "num_2q_gates": ...},
              "aqc_trotter":  {"depth_2q": ..., "num_2q_gates": ...}},
          2: {...},
      },
      "warnings": [...],           # non-fatal notices, e.g. cotengrust-fallback
      "resource_usage": {          # per execution phase (CPU/QPU time)
          "RUNNING: OPTIMIZING_FOR_HARDWARE": {...},
          "RUNNING: WAITING_FOR_QPU": {...},
          "RUNNING: EXECUTING_QPU": {...},
      },
  },
}
```

`aqc_fidelities` and `circuit_stats` are the two to read first: together they tell you
whether the compression was faithful and whether it actually saved depth. See
[`aqc_segments`](#aqc_segments) for how to act on them.

## Execution backends

All three share the same code path and the same mitigation knobs; they differ only
in *where* circuits run. Select one with `backend=` in the call.

| backend | what it is | credentials | notes |
|---------|------------|-------------|-------|
| `statevector` | exact `StatevectorEstimator` | none | the exact reference path; runs in-process. Use it to tune `aqc_segments` before spending hardware time |
| `fake` | noisy local simulation on a Qiskit fake backend via `EstimatorV2(mode=fake_backend)` (Aer noise) | none | a faithful in-process rehearsal of the mitigated `runtime` path (DD/XY4 + Pauli twirling + TREX). Needs `qiskit-aer`. Defaults to 127-qubit `fake_sherbrooke` (real measured noise) |
| `runtime` *(default)* | the same mitigated `EstimatorV2` against a real IBM backend | **yes** | `backend_name` optional → `least_busy()`; split across N jobs with `batches` |

On the two local paths every time-step circuit is independent, so `parallel_sim=True`
fans them across all available cores via Ray (one task per chunk of circuits); the
default `False` runs them sequentially. It has no effect on `runtime`, which parallelises
via `batches` instead.

Mitigation on the `fake`/`runtime` paths — dynamical decoupling (XY4), gate Pauli
twirling, and measurement mitigation (TREX) — is configured via
[`estimator_options`](#estimator_options) and passed straight to `EstimatorV2.options` in
[`source/execute.py`](source_files/source/execute.py).

## Citing this project

If you use this template in your research, please cite it:

```bibtex
@software{trotter-aqc-hamiltonian-template,
  author = {TODO: template authors},
  title = {{AQC+Trotter Hamiltonian Dynamics Qiskit function template}},
  howpublished = {\url{https://github.com/qiskit-community/qiskit-function-templates/tree/main/physics/trotter_aqc_hamiltonian}},
  year = {2026}
}
```

The compression stage is built on the [Qiskit addon: AQC-Tensor](https://qiskit.github.io/qiskit-addon-aqc-tensor/);
if your work relies on the method, please also cite the addon and the references listed
in its documentation.
