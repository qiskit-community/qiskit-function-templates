# Circuit Function Template

> [Download the Circuit Function template](https://github.com/qiskit-community/qiskit-function-templates/tree/main/circuit_functions/vanilla)
>
> [Deploy and run the template](https://github.com/qiskit-community/qiskit-function-templates/blob/main/circuit_functions/vanilla/deploy_and_run_circuit_function.ipynb)

A Qiskit Function template that optimizes, executes, and post-processes
[Estimator PUBs](https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/qiskit-ibm-runtime.EstimatorV2)
on IBM Quantum hardware. It wraps the standard transpile → run → collect loop in a
serverless-ready entrypoint with configurable error mitigation, and is intended as a
starting point for building custom circuit-level functions.

---

## Contents

```
circuit_functions/vanilla/
├── source_files/
│   ├── circuit_function_entrypoint.py   # function entry point
│   └── options/                         # options sub-package
│       ├── options.py
│       ├── dynamical_decoupling_options.py
│       ├── resilience_options.py
│       ├── twirling_options.py
│       ├── zne_options.py
│       ├── pec_options.py
│       └── utils.py
├── test/
│   ├── test_run_function.py             # unit tests — full execution
│   ├── test_options.py                  # unit tests — mitigation levels
│   ├── test_validation.py               # unit tests — input rejection
│   └── e2e/                             # 17 tests, live cluster required
├── requirements.txt                     # runtime + test dependencies
├── deploy_and_run_circuit_function.ipynb
└── README.md                            # this file
```

---

## Function description

The function accepts one or more Estimator PUBs and:

1. **Optimizes** — transpiles each circuit to the target backend's ISA using
   `generate_preset_pass_manager`, applies the observable layout.
2. **Executes** — submits the ISA PUBs to `EstimatorV2` on the requested backend,
   reporting job status while queued.
3. **Post-processes** — collects results and returns them together with CPU-time
   metadata for each stage.

Error mitigation is controlled by a single `mitigation_level` (1–3) that sets
sensible defaults for dynamical decoupling, Pauli twirling, TREX, and ZNE. Every
default can be overridden via the `options` dict.

---

## Inputs

| Name | Type | Required | Description |
|---|---|---|---|
| `backend_name` | `str` | ✅ | Name of the IBM Quantum backend to target (e.g. `"ibm_kingston"`). |
| `pubs` | `list[EstimatorPubLike]` | ✅ | One or more PUBs, each a tuple `(circuit, observables)`, `(circuit, observables, parameter_values)`, or `(circuit, observables, parameter_values, precision)`. |
| `options` | `dict \| None` | ❌ | Optional override dict (see [Options](#options) below). Defaults to `None` (mitigation level 1). |
| `instance` | `str \| None` | ❌ | IBM Quantum CRN/instance string. Defaults to `None`. |

### PUB format

Each PUB is a tuple accepted by `EstimatorV2.run()`:

```python
(circuit, observables)
(circuit, observables, parameter_values)
(circuit, observables, parameter_values, precision)
```

- `circuit` — a `QuantumCircuit` (static circuits only; dynamic circuits with control
  flow are rejected).
- `observables` — a Pauli string, `SparsePauliOp`, `Pauli`, a dict `{pauli: coeff}`,
  or an array of any of the above.
- `parameter_values` — a numeric array whose last axis matches `circuit.num_parameters`.
- `precision` — per-PUB target precision (float, overrides `options.default_precision`).

---

## Options

Pass any of the following keys inside the `options` dict. Unset keys fall back to the
defaults implied by `mitigation_level`.

### Top-level keys

| Option | Type | Default | Description |
|---|---|---|---|
| `mitigation_level` | `int` (1–3) | `1` | Convenience knob that pre-sets DD, twirling, and ZNE defaults (see table below). |
| `optimization_level` | `int` (0–3) | `2` | Transpiler optimization level passed to `generate_preset_pass_manager`. |
| `default_precision` | `float > 0` | unset | Default target precision for every PUB that does not supply its own. |
| `max_execution_time` | `int > 0` | unset | Maximum wall-clock seconds for the Estimator job. |

### Mitigation level presets

| Level | Dynamical decoupling | Twirling (gates) | Twirling (measure) | TREX | ZNE | ZNE amplifier |
|---|---|---|---|---|---|---|
| **1** | ✅ | ❌ | ✅ | ✅ | ❌ | — |
| **2** | ✅ | ✅ | ✅ | ✅ | ✅ | `gate_folding` |
| **3** | ✅ | ✅ | ✅ | ✅ | ✅ | `pea` |

### Sub-option groups

Fine-grained overrides mirror `EstimatorV2` options and are passed nested inside
the `options` dict:

| Group | Key | Description |
|---|---|---|
| `dynamical_decoupling` | `enable` | Enable/disable DD. |
| `dynamical_decoupling` | `sequence_type` | DD sequence (e.g. `"XX"`, `"XY4"`). |
| `twirling` | `enable_gates` | Enable gate twirling. |
| `twirling` | `enable_measure` | Enable measurement twirling. |
| `resilience` | `measure_mitigation` | Enable TREX measurement mitigation. |
| `resilience` | `zne_mitigation` | Enable ZNE. |
| `resilience.zne` | `amplifier` | ZNE noise amplifier (`"gate_folding"`, `"pea"`, …). |
| `resilience.zne` | `noise_factors` | Noise amplification factors (list of floats ≥ 1). |
| `resilience.zne` | `extrapolator` | Extrapolation method(s) (`"linear"`, `"exponential"`, …). |
| `resilience.pec` | `max_overhead` | PEC max overhead bound. |

**Example — level 2 with gate-folding overridden to PEA:**

```python
options = {
    "mitigation_level": 2,
    "resilience": {"zne": {"amplifier": "pea"}},
}
```

---

## Output

`job.result()` returns a dict:

```python
{
    "hw_results": PrimitiveResult,   # EstimatorV2 output; index with result["hw_results"][i]
    "metadata": {
        "resources_usage": {
            "RUNNING: OPTIMIZING_FOR_HARDWARE": {"CPU_TIME": float},
            "RUNNING: WAITING_FOR_QPU":         {"CPU_TIME": float},
            "RUNNING: EXECUTING_QPU":           {"CPU_TIME": float},
            "RUNNING: POST_PROCESSING":         {"CPU_TIME": float},
        }
    }
}
```

Each `PubResult` in `hw_results` exposes `.data.evs` (expectation values) and
`.data.stds` (standard deviations). The result index `i` corresponds to PUB index
`i` in the input list.

**Dry-run mode** (local testing only — pass `dry_run=True` to `run_function`):
returns only the `metadata` dict with transpilation timing; no hardware job is
submitted.

---

## Deploy and run

The notebook [`deploy_and_run_circuit_function.ipynb`](deploy_and_run_circuit_function.ipynb)
walks through the full upload → run → inspect cycle with two worked examples:

1. **Multi-observable survey** — five Pauli observables on a parameterized
   `RealAmplitudes` ansatz swept over five random parameter sets.
2. **Mitigation comparison** — the same circuit run at mitigation levels 1, 2,
   and 3 to compare accuracy vs. cost.

### Quick start

```python
from qiskit_ibm_catalog import QiskitServerless, QiskitFunction

serverless = QiskitServerless(
    channel="ibm_quantum_platform",
    token="MY_TOKEN",
    instance="MY_CRN",
)

template = QiskitFunction(
    title="circuit_function_template",
    entrypoint="circuit_function_entrypoint.py",
    working_dir="./source_files/",
)
serverless.upload(template)
```

```python
from qiskit.circuit.library import real_amplitudes
import numpy as np

ansatz = real_amplitudes(num_qubits=4, reps=1)
params = np.random.default_rng(42).uniform(0, 2 * np.pi, size=(5, ansatz.num_parameters))
pubs = [(ansatz, "IIZZ", params)]

func = serverless.load("circuit_function_template")
job = func.run(backend_name="ibm_kingston", pubs=pubs)
result = job.result()
print(result["hw_results"][0].data.evs)
```

---

## Tests

Unit tests run locally against a fake backend — no IBM account or cluster needed.
From the repository root, use the template's `tox` environment (which installs the
dependencies and runs `stestr`):

```bash
tox -ecircuit-vanilla        # 46 unit tests; e2e tests skip without credentials
```

Or run `stestr` directly against the template's `test/` directory:

```bash
stestr --test-path circuit_functions/vanilla/test run
```

End-to-end tests require a deployed function and cluster credentials. They are
**skipped automatically** unless `QISKIT_FUNCTION_NAME` is set, so the commands
above stay green without credentials. To run them, export the required env vars
(or add them to a `.env` file in this template directory):

```bash
GATEWAY_URL=http://localhost:8000
GATEWAY_TOKEN=my_token
QISKIT_FUNCTION_NAME=provider/circuit_function_template
QISKIT_IBM_BACKEND=ibm_kingston
```

See [`test/RUNNING_TESTS.md`](test/RUNNING_TESTS.md) for the full test reference,
including per-file breakdowns and individual class shortcuts.

---

## Dependencies

Default:
```
qiskit-serverless
qiskit-ibm-runtime
```

Custom:
```
qiskit>=2.0.0
numpy>=1.26.0
pydantic>=2.0.0
```

Test only (listed in `requirements.txt` because CI installs it to build seeded
reference values; not needed to run the deployed function):
```
qiskit-aer
```

Optional, local dev only (never uploaded, not required for the tests):
```
python-dotenv   # loads a .env file for the e2e credentials
```
