---
name: managed-workflows-quickstart
description: Help a researcher whose local quantum-classical workflow (AQC-Tensor, QAOA, SQD, VQE, Trotter/MPS, error mitigation, or any Python code) has outgrown their machine run it on IBM Quantum Managed Workflows — size it roughly, wrap it as a function, deploy it, run it, and scale it up. For their own use; not for contributing a template to this repository.
when_to_use: The kernel dies, the process is killed, or it runs out of memory; they shrink the active space, qubits, bond dimension, depth or samples to make it fit; optimizer loops, sweeps or many instances take hours; a laptop stays open overnight or while QPU jobs queue; they ask whether a GPU or a bigger machine would help; they want to run their own code as a Qiskit Function, or choose sizes_map or function_size.
---

# Get a research workflow running on Managed Workflows

Help a researcher take a workflow that works on their laptop and run it on Managed Workflows: cloud-hosted single machines with up to 48 cores and 768 GB on CPU, or up to 8 GPUs. A job can run for up to 24 hours of execution (time in the queue does not count, and this can be extended), and it keeps running after the laptop closes.

This is for running their own code, not for contributing a template. Skip tests, linting, licence headers and repository layout unless they ask. Keep the code theirs: wrap it, don't rewrite it.

Work through it conversationally, in four steps. Do not assume they know Managed Workflows or Qiskit Functions.

## Step 1 — Understand the workflow and what they want to run

Ask for:

1. **Their code**: the entry function or script and its helper modules. Ask them to point at the files.
2. **What it computes**: inputs, outputs and the rough stages (classical prep → circuits → simulator or QPU → post-processing).
3. **The problems they want to run.** Label them `s`, `m`, `l` and optionally `xl` (the service accepts no other labels). Each one is a whole workflow: every argument that changes cost (molecule and active space, qubits, bond dimension, depth, samples, iterations), set together. They differ only in size, never in method. Propose them from the code and have the researcher confirm. Don't add bigger ones unless they ask what the ceiling is.
4. **What they already know about scaling**, e.g. how their method responds to more cores, a GPU or more memory, from papers or earlier runs. Use it; the probe below checks it rather than replacing it.

Watch for:

- **A simulator mode and a hardware mode**: size each on its own.
- **Exact checks** (a full FCI reference, an exact-unitary comparison, an exhaustive enumeration): these often grow faster than the method itself. Put them behind an off-by-default switch and size them separately.
- **Vendor functions** (`catalog.load("vendor/...").run(...)`): these already run remotely on compute the vendor sizes. Size only the researcher's own code around the call.

If it fits locally and finishes in a few minutes, say so. A laptop is fine for that.

## Step 2 — Write the formula, and check the code runs

Sizing is measured on Managed Workflows in Step 4, from small real jobs. Locally, you need two things first.

1. **A memory formula.** Find the stage that dominates and the largest objects it holds at its peak, and write their size as a formula in the problem arguments. `sizing_guide.md` shows the common shapes and how to count copies and workers. The formula supplies the *shape* of the growth; the measured runs supply the constant, so leave it rough. Copy `scale_probe.py` from this folder next to their code and put the formula in `MEMORY_GB`. Put the problems from Step 1 in `TIERS`, and optionally the relative amount of work in `TIME_UNITS`. `python scale_probe.py --estimate` evaluates the formula for every problem; treat it as a lower bound, not a sizing. Quote the printed numbers, and never work estimates or machine choices out in prose.
2. **The code runs.** Point `work` at their real code, never a stand-in written to imitate it. Set `SMOKE` to the smallest problem that still runs every stage in seconds (few qubits, one or two iterations, one batch), set `WORKERS_ARG` if their code takes a process or thread count, and run `python scale_probe.py --check` with the QPU mocked. Never submit to real hardware here. This catches import errors and wrong arguments before anything is uploaded, and records the run in `runs.json`. The check runs on the researcher's own machine, so it is capped: it runs in a child process group at the lowest priority, with BLAS threads at 1 and at most 2 workers, and kills the group, worker processes included, after 120 s or 4 GB (`LOCAL_CAPS`). It refuses to start when the machine is already busy, and refuses a tier the formula puts over the memory cap. If the check is stopped, make `SMOKE` smaller; do not raise the caps. Do not run their real problem locally to see how long it takes: sizing is what Step 4 measures, on Managed Workflows.

   **Keep the researcher's machine usable.** Every local run in this skill is a smoke test. Do not run larger sizes, core sweeps or profiling loops locally, and do not start a run while an earlier one is still going (check the load first). Code that uses `multiprocessing` must be run from a file whose work is under `if __name__ == "__main__":`. Run through `python -c`, stdin or a heredoc, on macOS and Windows every worker process fails to import the main module and the pool keeps restarting workers until the run is killed. If the researcher's machine is struggling already (fans, swapping, a kernel restarting), stop local runs and go to Step 3.

## Step 3 — Wrap it as a function and deploy it

Put their code in a folder (everything in it is uploaded, max 50 MB) with one entrypoint file next to their modules:

```
my_workflow/
  entrypoint.py        # run_function(...) + __main__
  <their modules>.py   # unchanged where possible
```

Copy `mw_telemetry.py` from this folder into the same folder. The entrypoint imports their code, reads the arguments, wraps the run in `telemetry()`, and returns a dict:

```python
import traceback
from qiskit_serverless import get_arguments, save_result, update_status, Job, get_runtime_service
from qiskit_serverless import install_uploaded_wheels

install_uploaded_wheels()                      # extra packages (see below); a no-op without wheels

from mw_telemetry import telemetry
from my_module import run_my_workflow          # their code


def run_function(backend_name=None, alloc_check=None, **args):
    backend = get_runtime_service().backend(backend_name) if backend_name else None
    result = None
    with telemetry(alloc_check=alloc_check) as t:  # peak memory, cores used, heartbeat in the logs
        update_status(Job.POST_PROCESSING)         # pick the label per stage: MAPPING, OPTIMIZING_HARDWARE,
        with t.stage("workflow"):                  # WAITING_QPU, EXECUTING_QPU, POST_PROCESSING
            result = run_my_workflow(backend=backend, **args)
    return {"result": result, "metadata": {"resources": t.summary}}


if __name__ == "__main__":
    try:
        save_result(run_function(**get_arguments()))
    except Exception:
        save_result(traceback.format_exc())
        raise
```

`t.summary` holds `peak_gb` (the job's own memory counter on Managed Workflows, so it includes worker processes), `start_gb`, `wall_s`, `cores_used`, `cores_granted`, the `function_size` and `compute_profile` the job was given, and per-stage time. `alloc_check` is only set by `mw_sweep.py --target` (Step 4); leave it in. A heartbeat prints time and memory to `job.logs()` every five minutes. Add more `t.stage(...)` blocks to see which stage dominates.

Jobs have no outbound network, so telemetry comes back through the job itself: `t.summary` in the result, `MW_TELEMETRY` JSON lines in the logs (`parse_logs(job.logs())` reads them), and a `telemetry-<run id>.json` file in the function's storage, named in `t.summary["telemetry_file"]`. To see runs in an OpenTelemetry backend, download the file with `mw.file_download(name, fn)` and run `python mw_telemetry.py replay <file>` on your own machine; it sends one trace per run to the collector set in the usual `OTEL_EXPORTER_OTLP_*` variables.

Keep these in mind:

- **`run_function` passes its arguments straight to their code.** The keyword arguments given to `fn.run(...)` must match their function's own parameters; only `function_size` is taken by the service.
- **Arguments and results must be JSON-friendly**: numbers, strings, lists, dicts, and the Qiskit objects the client knows how to encode. `result` must be too, so convert arrays with `.tolist()`.
- **Log progress** with `print(..., flush=True)` (or `t.log(...)`) once per iteration in long loops. With the heartbeat, a run that's slower than expected, or stuck, shows up in `job.logs()` early.
- **numba-based code** (quimb, for example) needs `env_vars={"NUMBA_CACHE_DIR": "/tmp/numba_cache"}` on `QiskitFunction`, because site-packages is read-only.
- **Use `/tmp` for scratch files.** `/function_user_data` is shared by all jobs of the function, if they need to pass files between jobs.
- **No tokens in the code.** The job gets its backend through `get_runtime_service()`.
- **If the code fans work out**, size the pool from the profile the job got (the vCPU count in `get_compute_profile()`, for example 48 for `48x768`), not `os.cpu_count()`. The job can run on a larger node than its profile, so `os.sched_getaffinity(0)` and `os.cpu_count()` can report more cores than the job is allocated. Set `OMP_NUM_THREADS` and similar to cores ÷ workers before numpy is imported. Keep pool creation and anything else with side effects (such as `install_uploaded_wheels()`) under `if __name__ == "__main__":`, so that worker processes started with spawn or forkserver don't run it again.

Install the client (`pip install "qiskit-ibm-catalog>=0.22"`, the Managed Workflows client). Save their account once (they paste their own API key and instance CRN; never write them into files), and list the machines this deployment offers:

```python
from qiskit_ibm_catalog import ManagedWorkflowClient, ManagedWorkflow

ManagedWorkflowClient.save_account(token="<API key>", instance="<CRN>", name="mw", overwrite=True)  # once
mw = ManagedWorkflowClient(name="mw")
print(mw.compute_profiles())
```

It prints one profile per line, CPU-only first, for example:

```
8x32            (8 vCPU / 32 GB)
16x128          (16 vCPU / 128 GB)
48x768          (48 vCPU / 768 GB)
20x320x2a100    (20 vCPU / 320 GB / 2a100)
160x1792x8h100  (160 vCPU / 1792 GB / 8h100)
```

The list depends on the deployment, so take the ids from the output, not from this example. Then upload with the machines from Step 2:

```python
fn = ManagedWorkflow(
    title="my-workflow",
    entrypoint="entrypoint.py",
    working_dir="./my_workflow/",
    runner="fleets",
    sizes_map={                                                # from the fit in Step 4
        "s": mw.compute_profile(cpu=8, memory=32),     # checked against the list
        "m": "16x128",                                         # or a profile id as a string
        "l": "48x768",
    },
    default_size="s",
)
mw.upload(fn)
fn = mw.load("my-workflow")
```

`compute_profile(...)` raises and prints the available profiles if nothing matches, and an upload naming an unregistered id is rejected the same way. Use a GPU profile only if *Would a GPU help?* in `sizing_guide.md` says so. Otherwise add a GPU run as a comparison in Step 4. Fleets is the default runner from qiskit-ibm-catalog 0.22; `runner="fleets"` is kept so older gateways get it too (Ray is deprecated).

**Extra packages.** The runtime image already ships Qiskit and many common packages (`mw.dependencies_versions()` lists them); anything else comes as a wheel you upload. `dependencies=` on `ManagedWorkflow` no longer installs anything (it is ignored and gives a `DeprecationWarning`), so don't use it.

1. Check what the image already has: `print(mw.dependency_report(["qiskit-aer==0.17.2", "quimb"]))`. Each requirement is shown as already present, to bring as a wheel, or clashing with the version the image ships. Bring only what it marks to bring, and use the image's version where it clashes.
2. Get a wheel for each one: `pip download <pkg>==<version> --no-deps --only-binary=:all: --platform manylinux2014_x86_64 --python-version <the image's Python> -d wheels/`, or `pip wheel --no-deps -w wheels/ <path>` for pure-Python code. The install runs with `--no-deps`, so add a wheel for every dependency the report doesn't show as present.
3. After `mw.upload(fn)`: `mw.add_wheels(fn, ["wheels/*.whl"])`. They stay in the function's storage across re-uploads; `mw.list_wheels(fn)` shows them, and to change one, `mw.remove_wheels(fn, [name])` then `add_wheels` the new build.
4. The entrypoint calls `install_uploaded_wheels()` before importing them (as in the template above). It installs into `/tmp/site` on every run, which adds a little startup time.

## Step 4 — Measure, then climb

Copy `mw_sweep.py` next to `scale_probe.py` and set `ACCOUNT`, `FUNCTION` and, if needed, `BASE` and `SWEEP_ARGS` in its edit block. It reads the machines from `mw.compute_profiles()`, submits the jobs, waits for them with `job.wait_for_final_state()`, collects `metadata["resources"]` into `runs.json` with each `job_id`, the profile the job reports and its `fleet_id`, and fits with `scale_probe.py`. Run it from the researcher's machine; job startup takes ~2–3 min.

**Measure small sizes first.**

```
python mw_sweep.py sweep
```

It starts from `BASE` (default: the smallest tier) and shrinks one argument at a time to ~¼ and ~½ of its formula memory, so every argument in `SWEEP_ARGS` gets its own small runs. All jobs go to `SWEEP_SIZE` at once. An argument the formula ignores is halved anyway, so a missing argument shows up. Check `BASE` fits `SWEEP_SIZE` (or set it smaller); each run should add memory well beyond imports and run for a few seconds or more.

It prints an estimate, a time, a starting machine and a fallback for every tier, then two checks:

- **Shape, per argument.** Along each argument, measured memory growth is compared with the formula's: "SHAPE WRONG along `norb` … grows 3× faster" means the formula weights that argument wrongly or misses an object that depends on it; memory that stays flat while the formula grows means the peak comes from something the formula leaves out. Fix `MEMORY_GB` with the researcher and run `python mw_sweep.py fit` before trusting larger tiers.
- **Cores.** If the largest run used far fewer cores than it was granted, pick machines for memory only; `cores_granted` can exceed the profile's vCPU count (96 on `48x768`). If it kept the cores busy, it suggests `python mw_sweep.py compare m 8x32 16x128`: the same tier on both machines at once, reporting the speedup for the extra cores and whether memory grew with them (worker pools sized from the core count use more memory on a bigger machine). Climb for cores only if it says so.

**Optionally, check the research size directly.** `python mw_sweep.py --target l` runs tier `l` as a job on the smallest machine with an allocation check: the first numpy array above `--cap-gb` (default 4) is recorded with its shape, size and code line, and not allocated. It takes about as long as the setup before the big array. It answers "which object dominates at research size" in minutes, and gives a lower bound on the peak: arrays created outside numpy's `zeros`/`empty`/`ones`/`full`, or several large arrays alive at once, are not counted. Use the result to check the formula; size from the fit. Never run it locally: below-cap arrays are still allocated.

**Then climb.**

```
python mw_sweep.py climb m l
```

It runs each tier on the machine the fit suggests, smallest first, adds each result to `runs.json` and refits before the next. A problem estimated above the largest machine still gets a try on the largest CPU profile (`48x768` today): estimates far out are crude and often high. On failure:

- **numpy `Unable to allocate X`**: X is read from the error, machines too small for it are skipped, and if X is above 768 GB it says so. Then work out with the researcher how to split or reshape the problem; *Bigger than one machine* in `sizing_guide.md` has the questions.
- **Killed with no traceback** (the machine ran out of memory): one retry on the next machine up, with the last heartbeat shown.
- **Any other error**: it stops and prints the traceback. Fix it with the researcher; never resubmit it unchanged.

`--from 16x128` starts the first tier on a given machine. Each profile it uses needs a size label in `sizes_map`; add one and re-upload under the same title if needed (existing jobs are unaffected).

Every number you report must come from a real job: quote its `job_id` and the `metadata["resources"]` it returned, as `mw_sweep.py` prints them. If an upload or submission fails, show the researcher the error and stop. Never estimate what a run would have measured, or fill in a result that wasn't returned.

If run time grows faster than expected, check the heartbeat in `job.logs()` before the next size. A 24-hour job that is 10× slower than planned should be split or extended, not resubmitted; the limit counts execution time, and `update_status(...)` calls do not reset it.

Keep going until something stops them; that is the interesting result. Then update `sizes_map` from what the runs showed, and keep `runs.json` with their code.
