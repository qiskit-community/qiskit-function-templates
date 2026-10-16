---
name: template-from-research-code
description: Guide a new contributor through turning their research code into a Qiskit Function template (entrypoint, requirements, test, deploy notebook) that matches this repo's conventions
---

# Research code → Qiskit Function template

Help a contributor turn a working piece of research code (a script, a notebook, a
small package) into a runnable Qiskit Function template that fits this repository.
Walk them through it conversationally — ask for what you need, scaffold the files,
and explain each step. Do not assume prior familiarity with Qiskit Serverless.

The goal is a self-contained template directory that another user can deploy to
Qiskit Serverless and run as-is, following the structure every runnable template
here already uses.

## Step 0 — Understand what they have

Ask the contributor for:

1. **Their research code** — the entry function or script, and any helper modules.
   Ask them to point you at the files or paste the core function.
2. **What it computes** — inputs (and which are quantum-backend-dependent), outputs,
   and the rough stages (classical prep → circuit build → hardware/simulator run →
   post-processing).
3. **Category** — does it belong under `chemistry/`, `physics/`, or
   `circuit_functions/`? Pick by scientific domain; `circuit_functions/` is for
   circuit-level utilities rather than application workflows.
4. **A short, kebab-or-snake template name** (e.g. `sqd_pcm`, `aqc_trotter`).

Read `base_templates/application_function_template.py` (application workflows) or
`base_templates/circuit_function_template.py` (circuit-level) as the blueprint, and
skim the closest existing template in the chosen category for a concrete example.

## Step 1 — Lay out the directory

Create `<category>/<name>/` with this structure (every runnable template has it):

```
<category>/<name>/
  __init__.py
  README.md
  requirements.txt
  deploy_and_run.ipynb
  source_files/
    __init__.py
    <name>_entrypoint.py        # defines run_function(...) + __main__
    <helper modules>.py          # the research code, refactored
  test/
    __init__.py
    test_<name>.py
    data/                        # small fixtures the test loads, if any
```

Keep the research code's helpers as sibling modules in `source_files/`; the
entrypoint imports them.

## Step 2 — Write the entrypoint

The heart of the transformation is reshaping the research code into a single
`run_function` plus a `__main__` block, in `source_files/<name>_entrypoint.py`.

### `run_function`

```python
def run_function(backend_name: str, <workflow args>, **kwargs) -> dict[str, Any]:
```

- **Signature**: `backend_name` first (identifier for the target backend), then the
  workflow's own arguments, then `**kwargs`. Every argument type must be
  **serializable by qiskit-serverless** (JSON-friendly: numbers, strings, lists,
  dicts, and Qiskit objects it knows how to encode). Prefer grouping related
  arguments into dicts (e.g. `solvent_options`, `sqd_options`) over long flat
  signatures.
- **Validate inputs at the top** so the job fails fast with a clear message.
- **Report progress** with `update_status(Job.MAPPING)`,
  `update_status(Job.OPTIMIZING_HARDWARE)`, etc. at the natural stage boundaries, so
  users watching the job see where it is.
- **Return a dict with string keys** — this is required for
  `qiskit_serverless.save_result`. Put numeric results, metadata, and timings in it.
- **Local-testing kwargs**: read `testing_backend = kwargs.get("testing_backend", None)`
  and `dry_run = kwargs.get("dry_run", False)` so the test suite can drive the
  function with a fake backend and skip hardware execution. Any file-path overrides
  the test needs (e.g. `count_dict_file_name`) also come in through kwargs.

Pull the backend from the runtime service when not testing:

```python
service = get_runtime_service()
backend = testing_backend or service.backend(backend_name)
```

### `__main__` plumbing

End the file with the standard serverless entry block (copy it verbatim and adjust
nothing but the function name):

```python
if __name__ == "__main__":
    input_args = get_arguments()
    try:
        func_result = run_function(**input_args)
        save_result(func_result)
    except Exception:
        save_result(traceback.format_exc())
        raise
    sys.exit(0)
```

Import the helpers from `qiskit_serverless`: `get_arguments`, `save_result`,
`update_status`, `Job`, `get_runtime_service`, `get_logger`. Use
`logger = get_logger()` and log at stage boundaries rather than printing.

### What to change vs. keep from the research code

- Keep the scientific core (the actual algorithm) in helper modules, imported by the
  entrypoint — don't inline everything into `run_function`.
- Replace any hardcoded local paths, interactive prompts, or `print` calls.
- Replace credential handling: the function receives its backend via
  `get_runtime_service()`, never a hardcoded token.
- If the code parallelizes, see **Parallelism** below.

## Step 3 — requirements.txt

Pin the scientific dependencies, matching the versions in the Fleets runtime image
so local testing runs against the same stack that executes the function. Model it on
the chosen category's existing template. Always include `qiskit-serverless` (and
`qiskit-ibm-runtime` if the function uses the runtime estimator/sampler paths). Put
test-only packages (e.g. `qiskit-aer` for a fake-backend path) in the same file with
a comment, since CI installs only this file before running the suite.

## Step 4 — test/test_<name>.py

Write a `unittest.TestCase` that imports `run_function` directly and calls it with a
**fake backend** and small inputs, with no network and no credentials:

```python
from ..source_files.<name>_entrypoint import run_function
from qiskit_ibm_runtime.fake_provider import FakeHanoiV2

out = run_function(
    backend_name=None,
    <small workflow args>,
    testing_backend=FakeHanoiV2(),
    <any file-path kwargs pointing at test/data fixtures>,
)
self.assertEqual(out["<key>"], <expected>)
```

Assert on the result dict's keys and on a loose correctness bound (e.g. an energy
below a known value). Keep fixtures tiny and store them under `test/data/`.

## Step 5 — deploy_and_run.ipynb

A short notebook showing a user how to deploy and run the function. Authentication
must use **placeholders, never real credentials**:

```python
from qiskit_ibm_catalog import QiskitServerless
token = "MY_TOKEN"
instance = "MY_INSTANCE"
serverless = QiskitServerless(token=token, instance=instance)
```

Then upload the function and run it with a small example. These templates are
**custom functions** — the uploader owns the whole definition, unlike a provider
function where the maintainer registers it. Upload with `QiskitFunction` and run the
returned handle:

```python
from qiskit_ibm_catalog import QiskitFunction

template = QiskitFunction(
    title="<name>_template",
    entrypoint="<name>_entrypoint.py",
    working_dir="./source_files/",   # everything here is uploaded
    runner="fleets",                 # Fleets runner (see Step 6 on sizing)
)
serverless.upload(template)

job = template.run(backend_name=..., <workflow args>)
print(job.job_id)
result = job.result()
```

If the notebook is committed with executed output, make sure no real token, CRN,
account id, or job id is left in a cell's source or output — replace any job id with
a placeholder.

## Step 6 — Sizes and compute profiles

**Most templates need nothing here** — uploading with `runner="fleets"` and no
`sizes_map` runs on a default profile, and that is fine for a first contribution
(the `aqc_trotter` template does exactly this). Add sizing only if the workload is
heavy enough to care about CPU/memory, or scales with a bigger machine (as a
parallel fan-out does — see **Parallelism**).

If you do add it, declare the catalog **at upload time** on the `QiskitFunction`
(these are custom functions, so the catalog lives in the function definition, not in
a separate deploy script):

```python
# Label -> compute profile. Labels are yours to choose; the right-hand side is a
# Fleets compute profile "<cpu>x<memory_GB>" (e.g. CPU-only profiles below).
SIZES_MAP = {
    "s": "8x32",     # 8 CPU,  32 GB
    "m": "16x128",   # 16 CPU, 128 GB
    "l": "48x768",   # 48 CPU, 768 GB
}

template = QiskitFunction(
    title="<name>_template",
    entrypoint="<name>_entrypoint.py",
    working_dir="./source_files/",
    runner="fleets",
    sizes_map=SIZES_MAP,
    default_size="s",
)
```

and pick a size per run:

```python
job = template.run(backend_name=..., <workflow args>, function_size="m")
```

Best practices:

- **Start small.** Make `default_size` the smallest profile that works, so an
  unqualified `run()` is cheap. Users opt into bigger machines explicitly with
  `function_size`.
- **Only offer sizes the workload can use.** More CPUs help only if the function
  actually parallelizes across them (the `_run_parallel` fan-out sizes itself from
  the cores the profile grants). More memory helps a bigger problem. Do not list an
  `"xl"` that behaves exactly like `"m"`.
- **A `function_size` must be a label in `sizes_map`, and `default_size` must too.**
  Keep labels short and lowercase (`"s"`/`"m"`/`"l"`/`"xl"`) matching the existing
  templates.
- **Profiles are `"<cpu>x<memory_GB>"`.** GPU profiles exist (used by GPU
  templates), but a profile that is not available for the deployment makes the run
  fail — if unsure which profiles you can use, ask on the project channel rather than
  guessing.
- **Document the sizes** in the README and show a representative one in the notebook.
  If the template benefits from scaling, a short size sweep (one run per label,
  reporting the timing from the result's metadata) is a strong addition — see
  `chemistry/sqd_pcm/compute_profile_benchmark.ipynb` for the pattern.

## Step 7 — README.md

A short README: what the function computes, its inputs/outputs, the sizes it offers
(if any), and a minimal usage snippet. Model the length and headings on the sibling
templates.

## Step 8 — Check it

Before handing back, confirm:

- `python -m black --check` passes on the new Python files (this repo pins
  `black[jupyter]~=24.1` in `requirements-dev.txt`; use that version — a newer black
  reformats differently).
- `pylint` is clean over the new files.
- The unit test runs and passes with the fake backend, no network.
- No secrets anywhere in the notebook (token/CRN/account/job-id).

Then point the contributor at `CONTRIBUTING.md` for opening the PR (there should be
an associated issue first, and the CLA bot will ask them to sign).

## Parallelism (only if the research code fans work out)

If the workflow runs independent chunks in parallel, use **Ray** directly
(`@ray.remote` + `ray.get`), started locally inside the entrypoint — the Fleets
selector image ships Ray. A few hard-won details for running inside a Fleets job:

- Give Ray a **short** temp dir: `tempfile.mkdtemp(prefix="ray-", dir="/tmp")`. Ray's
  session socket is an AF_UNIX path capped near 103 bytes, and the system default
  temp dir is too long on some platforms.
- Only a few dirs are writable: set `HOME` to that temp dir and
  `RAY_AUTH_MODE=disabled` (single-node in-job cluster).
- Ray serializes the remote function **by reference**, so workers re-import its
  module — pass the directory that makes its imports resolve on `PYTHONPATH` via
  `ray.init(runtime_env={"env_vars": {...}})`. Thread limits (OMP/OPENBLAS/MKL/...)
  go there too, so workers read them before importing numpy.
- Size the pool from `os.sched_getaffinity(0)` (the cgroup quota), not `os.cpu_count`.
- `ray.shutdown()` and remove the temp dir in a `finally`.

Model this on an existing parallel template in the repo rather than writing it from
scratch.
