---
name: managed-workflows-quickstart
description: Help a researcher whose local quantum-classical workflow (AQC-Tensor, QAOA, SQD, VQE, Trotter/MPS, error mitigation, or any Python code) has outgrown their machine run it on IBM Quantum Managed Workflows — size it roughly, wrap it as a function, deploy it, run it, and scale it up. For their own use; not for contributing a template to this repository.
when_to_use: The kernel dies, the process is killed, or it runs out of memory; they shrink the active space, qubits, bond dimension, depth or samples to make it fit; optimizer loops, sweeps or many instances take hours; a laptop stays open overnight or while QPU jobs queue; they ask whether a GPU or a bigger machine would help; they want to run their own code as a Qiskit Function, or choose sizes_map or function_size.
---

# Get a research workflow running on Managed Workflows

Help a researcher take a workflow that works on their laptop and run it on Managed Workflows: cloud-hosted single machines with up to 48 cores and 768 GB on CPU, or up to 8 GPUs. A job can run for up to 24 hours (this can be extended), and it keeps running after the laptop closes.

This is for running their own code, not for contributing a template. Skip tests, linting, licence headers and repository layout unless they ask. Keep the code theirs: wrap it, don't rewrite it.

Work through it conversationally, in four steps. Do not assume they know Qiskit Serverless.

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
2. **The code runs.** Point `work` at their real code, never a stand-in written to imitate it, and run `python scale_probe.py --check s` on the smallest problem with the QPU mocked. Never submit to real hardware here. This catches import errors and wrong arguments before anything is uploaded, and records the run in `runs.json`. If the researcher's largest problem also finishes in a few minutes within a few GB, a laptop is enough. Say so, and stop.

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

from mw_telemetry import telemetry
from my_module import run_my_workflow          # their code


def run_function(backend_name=None, **args):
    backend = get_runtime_service().backend(backend_name) if backend_name else None
    with telemetry() as t:                         # peak memory, cores used, heartbeat in the logs
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

`t.summary` holds `peak_gb` (the job's own memory counter on Managed Workflows, so it includes worker processes), `wall_s`, `cores_used`, `cores_granted` and per-stage time. A heartbeat prints time and memory to `job.logs()` every five minutes. Add more `t.stage(...)` blocks to see which stage dominates.

Jobs have no outbound network, so telemetry comes back through the job itself: `t.summary` in the result, `MW_TELEMETRY` JSON lines in the logs (`parse_logs(job.logs())` reads them), and a `telemetry-<run id>.json` file in the function's storage, named in `t.summary["telemetry_file"]`. To see runs in an OpenTelemetry backend, download the file with `serverless.file_download(name, fn)` and run `python mw_telemetry.py replay <file>` on your own machine; it sends one trace per run to the collector set in the usual `OTEL_EXPORTER_OTLP_*` variables.

Keep these in mind:

- **`run_function` passes its arguments straight to their code.** The keyword arguments given to `fn.run(...)` must match their function's own parameters; only `function_size` is taken by the service.
- **Arguments and results must be JSON-friendly**: numbers, strings, lists, dicts, and the Qiskit objects qiskit-serverless knows how to encode. `result` must be too, so convert arrays with `.tolist()`.
- **Log progress** with `print(..., flush=True)` (or `t.log(...)`) once per iteration in long loops. With the heartbeat, a run that's slower than expected, or stuck, shows up in `job.logs()` early.
- **numba-based code** (quimb, for example) needs `env_vars={"NUMBA_CACHE_DIR": "/tmp/numba_cache"}` on `QiskitFunction`, because site-packages is read-only.
- **Use `/tmp` for scratch files.** `/function_user_data` is shared by all jobs of the function, if they need to pass files between jobs.
- **No tokens in the code.** The job gets its backend through `get_runtime_service()`.
- **If the code fans work out**, size the pool from `os.sched_getaffinity(0)`, not `os.cpu_count()`. Set `OMP_NUM_THREADS` and similar to cores ÷ workers before numpy is imported.

Save their account once (they paste their own API key and instance CRN; never write them into files), then upload with the machines from Step 2:

```python
from qiskit_ibm_catalog import QiskitServerless, QiskitFunction

QiskitServerless.save_account(token="<API key>", instance="<CRN>", name="mw", overwrite=True)  # once
serverless = QiskitServerless(name="mw")

fn = QiskitFunction(
    title="my-workflow",
    entrypoint="entrypoint.py",
    working_dir="./my_workflow/",
    runner="fleets",
    sizes_map={"s": "8x32", "m": "16x128", "l": "48x768"},    # from the fit in Step 4
    default_size="s",
    dependencies=["qiskit-aer==0.17.2"],                       # extra packages, pinned
)
serverless.upload(fn)
fn = serverless.load("my-workflow")
```

Profiles available today:

| Profile | vCPU | Memory | GPU |
|---|---|---|---|
| `8x32` | 8 | 32 GB | — |
| `16x128` | 16 | 128 GB | — |
| `48x768` | 48 | 768 GB | — |
| `20x320x2a100` | 20 | 320 GB | 2 × A100 |
| `160x1792x8h100` | 160 | 1.7 TB | 8 × H100 |

A profile that isn't registered is rejected at upload with an error naming it. Use a GPU profile only if *Would a GPU help?* in `sizing_guide.md` says so. Otherwise add a GPU run as a comparison in Step 4.

## Step 4 — Measure, then climb

**Measure small sizes first.** Submit 2–3 problems smaller than the researcher's, in parallel, on `8x32` or `16x128`. Scale them down along every argument that changes cost, not just one. Pick sizes ~2–4× apart in formula memory, each large enough that it runs for a few seconds or more and adds memory well beyond imports:

```python
jobs = {name: fn.run(function_size="s", **problem) for name, problem in small_problems.items()}
print({name: (job.job_id, job.status()) for name, job in jobs.items()})   # startup takes ~2-3 min
runs = [{"problem": small_problems[name], "resources": job.result()["metadata"]["resources"]}
        for name, job in jobs.items()]
```

Save `runs` as `runs.json` and run `python scale_probe.py runs.json`. It fits memory as overhead + k × `MEMORY_GB` and time as a power of `TIME_UNITS`. It then prints an estimate, a time, a starting machine and a fallback for every tier. With three or more sizes it also tests the formula's shape: "SHAPE WRONG" means a middle run is far off the fitted line, so the formula missed a copy, a worker count or the dominant object. Fix it before trusting larger tiers.

Every number you report must come from a real job: quote its `job_id` and the `metadata["resources"]` it returned. If an upload, submission or result call fails, show the researcher the error and stop. Never estimate what a run would have measured, or fill in a result that wasn't returned.

`cores_granted` can be more than the profile's vCPU count (for example 96 on `48x768`); it counts the hardware threads the job can be scheduled on.

**Then climb.** Run the researcher's problems from smallest to largest on the machines the fit suggests. A problem estimated above the largest machine still gets a try on `48x768`: estimates far out are crude and often high. After each run:

- **Add it to `runs.json` and refit.** Each real run at a larger size improves the estimate for the next one; re-estimates from a nearby size have been within ~20%.
- **If it ran out of memory**, retry on the fallback machine. Only if `48x768` runs out of memory too, work out with the researcher how to split or reshape the problem; *Bigger than one machine* in `sizing_guide.md` has the questions.
- **If `cores_used` is far below `cores_granted`**, the extra cores came with the memory. That's fine, but don't climb for cores alone.
- **To compare machines on the same problem** (CPU vs GPU, or 8 vs 16 cores), run that problem with each size label whose profile you want to compare. Add a label to `sizes_map` and re-upload under the same title if you need another profile. Re-uploading replaces `sizes_map` and leaves existing jobs alone.
- **If run time grows faster than expected**, check the heartbeat in `job.logs()` before submitting the next size. A 24-hour job that is 10× slower than planned should be split or extended, not resubmitted.

Keep going until something stops them; that is the interesting result. Then update `sizes_map` from what the runs showed, and keep `runs.json` with their code.
