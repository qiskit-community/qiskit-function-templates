# Running the Tests

All commands are run from the repository root:

```bash
cd qiskit-function-templates
```

Tests use [`stestr`](https://stestr.readthedocs.io/) (same as the other templates
in this repo). The quickest way to run them with the right dependencies is the
template's `tox` environment:

```bash
tox -ecircuit-vanilla
```

To run `stestr` directly (assumes `requirements-dev.txt` and the template's
`requirements.txt` are installed):

```bash
stestr --test-path circuit_functions/vanilla/test run
```

`stestr` selects tests by **regex over the dotted test id**, e.g.
`stestr ... run test.test_validation.TestBackendValidation`. Passing args through
`tox` works too: `tox -ecircuit-vanilla -- <regex>`.

---

## Test structure

```
test/
├── __init__.py              # bootstrap: puts source_files/ on sys.path + stubs qiskit_serverless
├── base_test_case.py        # BaseTemplateTestCase (setUpClass with FakeManilaV2 backend)
├── utils.py                 # get_estimator_pub, get_inputs, dict_partially_equal, combine, …
├── test_run_function.py     # 22 tests — full execution against a fake backend
├── test_options.py          #  7 tests — mitigation levels and merge_options
├── test_validation.py       # 17 tests — bad inputs rejected before hardware
└── e2e/
    ├── base_e2e_test_case.py  # reads credentials from env / .env; skips when unset
    ├── test_pubs.py           # 14 tests — pub shapes and precision through serverless
    └── test_options.py        #  3 tests — all-options accepted; invalid options → job ERROR
```

The unit tests import the entrypoint with bare imports
(`from circuit_function_entrypoint import ...`, `from options import ...`), exactly
as the artifact does when it runs on a cluster. `test/__init__.py` makes that work
by putting `source_files/` on `sys.path` and registering a stub `qiskit_serverless`
module, so no live cluster is needed for the unit run.

---

## Unit tests (no credentials needed)

Run against `FakeManilaV2` locally — no IBM account, no serverless cluster required.
The e2e tests skip automatically (see below), so a plain run gives the 46 unit tests:

```bash
stestr --test-path circuit_functions/vanilla/test run
```

**46 unit tests.** Breakdown: 22 in `test_run_function.py` + 7 in `test_options.py`
+ 17 in `test_validation.py`.

---

### `test_run_function.py` — full execution (22 tests)

Tests that `run_function` actually runs circuits and returns sensible
results against a fake backend.

The function returns `{"hw_results": PrimitiveResult, "metadata": {...}}`;
all tests access `result["hw_results"]` accordingly.

```bash
stestr --test-path circuit_functions/vanilla/test run test.test_run_function
```

#### `TestRunFunction` (17 tests)

| Test | What it checks |
|------|----------------|
| `test_returns_one_pub_result_per_pub` ×3 | 1, 2, 3 pubs each return a `PrimitiveResult` of the right length |
| `test_observable_coercion_preserves_size` ×5 | One case per observable type: string, dict, `SparsePauliOp`, `Pauli`, nested array |
| `test_parameterized_circuit_list_values` ×2 | Single and multi parameter-set list-style values; `bindings_array.size` matches `evs` length |
| `test_parameterized_circuit_dict_values` | Dict-style `{param: values}` parameter passing |
| `test_multi_obs_params_zip_shape` | 3 observables × 3 param sets zipped → evs length `3` |
| `test_multi_obs_params_product_shape` | 3 observables × 2 param sets outer-product → shape `(3, 2)` |
| **`test_numerical_values_match_reference`** | **Seeded `AerSimulator`: evs must match reference values exactly** |
| `test_precision_precedence` ×2 | Pub-level overrides default; default used when pub precision is unset |
| `test_all_options_accepted_and_precision_propagates` | All option fields accepted; options round-trip; `default_precision` in metadata |

#### `TestPubPrecision` (3 tests)

| Test | What it checks |
|------|----------------|
| `test_pub_precision` ×3 | `[0.1]`, `[0.1, 0.2]`, `[0.1, None]` — per-pub precision in metadata; `None` entry skipped |

#### `TestLogging` (1 test)

| Test | What it checks |
|------|----------------|
| `test_logging_level` | The serverless logger receives at least one `info` or `debug` call — logging is not silenced |

#### `TestInstance` (1 test)

| Test | What it checks |
|------|----------------|
| `test_instance_kwarg_reaches_runtime_service` | `instance=` kwarg triggers `get_runtime_service()` (stub) |

#### Shortcuts

```bash
# Numerical accuracy test only
stestr --test-path circuit_functions/vanilla/test run \
  test.test_run_function.TestRunFunction.test_numerical_values_match_reference

# Observable-type matrix only
stestr --test-path circuit_functions/vanilla/test run observable_coercion

# Pub precision only
stestr --test-path circuit_functions/vanilla/test run test.test_run_function.TestPubPrecision
```

---

### `test_options.py` — mitigation level logic (7 tests)

Tests that `Options.apply_mitigation_level` produces the right option
structure at each level and that `merge_options` correctly overwrites
individual fields.

```bash
stestr --test-path circuit_functions/vanilla/test run test.test_options
```

| Test | What it checks |
|------|----------------|
| `test_mitigation_level` ×3 | Level 1/2/3 each produce the correct keys (DD, twirling, ZNE, PEA); runs the function end-to-end |
| `test_mitigation_overwrite` ×4 | `twirling.enable_gates=False`; `zne.amplifier=gate_folding`; `zne_mitigation=False`; `dd.sequence_type=XY4` — each overwrites the target field while DD stays enabled |

---

### `test_validation.py` — input rejection (17 tests)

Tests that bad inputs are caught before any hardware is touched.

```bash
stestr --test-path circuit_functions/vanilla/test run test.test_validation
```

#### Options validation (11 tests — `TestOptionsValidation`)

| Test | Input | Expected |
|------|-------|----------|
| `test_invalid_optimization_level` ×3 | Below-min int (`-1`); above-max int (`4`); wrong type (`"foo"`) | `ValidationError` |
| `test_invalid_mitigation_level` ×3 | Below-min int (`-1`); above-max int (`4`); wrong type (`"foo"`) | `ValueError` or `ValidationError` |
| `test_invalid_options_structure` | `{"pec_mitigation": True, "max_overhead": 100}` (flat, not nested) | `ValidationError` |
| `test_estimator_option_not_in_schema` | `{"seed_estimator": 42}` (valid estimator key, not in `Options` schema) | `ValidationError` |
| `test_default_options` | `options=None` | No error |
| `test_all_options_accepted` | All valid fields set simultaneously | No error |
| `test_options_routing_transpiler_vs_estimator` | `optimization_level` + `default_precision` | `optimization_level` → transpiler dict; `default_precision` → estimator dict |

#### Pubs validation (4 tests — `TestPubsValidation`)

| Test | Bad input | Expected error |
|------|-----------|----------------|
| `test_missing_observables` | Pub tuple with only a circuit | `ValueError: length of pub` |
| `test_missing_circuit_params` | Parameterized circuit with no parameter values | `ValueError: does not match the number of parameters` |
| `test_empty_pubs` | `pubs=[]` | `ValueError: At least one PUB` |
| `test_are_dynamic_circuits` | Circuit with `if_else` control flow | `ValueError: Dynamic circuits are not supported` |

#### Backend validation (2 tests — `TestBackendValidation`)

| Test | Bad input | Expected error |
|------|-----------|----------------|
| `test_missing_backend` | `backend_name=None` | `ValueError: Invalid backend name value` |
| `test_backend_num_qubits` | Circuit with more qubits than the backend | `IBMInputValueError` |

```bash
# Run a single class
stestr --test-path circuit_functions/vanilla/test run test.test_validation.TestBackendValidation
stestr --test-path circuit_functions/vanilla/test run test.test_validation.TestPubsValidation
stestr --test-path circuit_functions/vanilla/test run test.test_validation.TestOptionsValidation
```

---

## E2e tests (credentials required)

Run against a live serverless cluster with a deployed function. The
function must already be uploaded before running these. When
`QISKIT_FUNCTION_NAME` is unset the whole e2e suite is **skipped**, so the unit
run above never touches a cluster.

### Prerequisites

Export the vars in your shell, or create a `.env` file in this template directory
(`circuit_functions/vanilla/`):

```bash
GATEWAY_URL=http://localhost:8000              # or your cluster URL
GATEWAY_TOKEN=my_token                         # your serverless token
QISKIT_FUNCTION_NAME=provider/my-function      # as returned by serverless.list()
QISKIT_IBM_BACKEND=ibm_kingston                # backend name passed to run_function
```

> `.env` loading uses `python-dotenv`, an optional local-dev convenience. Install
> it (`pip install python-dotenv`) if you want the `.env` file picked up
> automatically; otherwise export the vars in your shell.

> **Result shape reminder:** `job.result()` returns
> `{"hw_results": PrimitiveResult, "metadata": {...}}` — not a bare
> `PrimitiveResult`. All e2e tests access `result["hw_results"]`
> for pub data and `result["metadata"]` for timing info.

### Run all e2e tests

With the env vars set, a normal run picks the e2e tests up automatically. To run
only the e2e suite:

```bash
stestr --test-path circuit_functions/vanilla/test run test.e2e
```

**17 tests total** (14 in `test_pubs.py` + 3 in `test_options.py`).
Runtime depends on queue depth and hardware availability.

---

### `test_pubs.py` — pub shapes and precision (14 tests)

| Test | What it checks |
|------|----------------|
| `test_min_parameters` ×3 | 1, 2, 3 pubs → `hw_results` of matching length; job reaches `DONE` |
| `test_parameterized_circuit` ×2 | List-style param values; `bindings_array.size` matches `evs` length |
| `test_observable_type` ×3 | `SparsePauliOp` scalar, list, and nested list → `obs_array.size == evs.size` |
| `test_multi_observables_params_zip` | 3 obs × 3 param sets zipped → `len(evs) == 3` |
| `test_multi_observables_params_product` | 3 obs × 2 param sets outer-product → `data.shape == (3, 2)` |
| `test_precision` ×4 | `pub_precision` / `default_precision` combinations → correct `metadata["target_precision"]` |

```bash
stestr --test-path circuit_functions/vanilla/test run test.e2e.test_pubs
```

---

### `test_options.py` — options accepted / rejected end-to-end (3 tests)

| Test | What it checks |
|------|----------------|
| `test_with_all_options` | All option fields accepted; job reaches `DONE`; `target_precision == 0.1` in metadata |
| `test_invalid_options` ×2 | `optimization_level=4` and `sequence_type="YY"` each cause job to reach `ERROR`; logs contain `ValidationError` |

```bash
stestr --test-path circuit_functions/vanilla/test run test.e2e.test_options

# Just the error-path tests
stestr --test-path circuit_functions/vanilla/test run test.e2e.test_options.TestE2eOptions.test_invalid_options
```

---

## Known warnings (both harmless)

| Warning | Cause | Action needed |
|---------|-------|---------------|
| `IBMFractionalTranslationPlugin is deprecated` | Fires on every `FakeManilaV2()` construction — stale plugin in `qiskit-ibm-runtime ≥ 0.42` | None — upstream issue |
| `Options have no effect in local testing mode` | Fake backends are noiseless; mitigation options are silently ignored | None — expected for unit tests |
