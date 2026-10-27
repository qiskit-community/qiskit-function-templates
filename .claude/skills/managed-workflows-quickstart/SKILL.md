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

## Step 2 — Rough sizing

The goal is a sensible first machine per problem, not a precise prediction. The real measurement comes from the cloud runs in Step 4.

1. **Estimate from the code.** Find the stage that dominates and the largest objects it holds at its peak. Write a memory formula in the problem arguments; `sizing_guide.md` shows the common shapes and how to count copies and workers. Evaluate it for every problem by running it: `python scale_probe.py --estimate` does this even when the code can't run locally. Quote the printed numbers, and never work estimates or machine choices out in prose.
2. **Check it with two small runs.** Copy `scale_probe.py` from this folder next to their code and edit its top block:
   - **`work`**: a generator that `yield`s after setup, so imports and one-time preparation aren't counted. It must call the researcher's real code, never a stand-in written to imitate it.
   - **`TIERS`**: the problems from Step 1.
   - **`MEMORY_GB`**: the formula.
   - **`TIME_UNITS`**: the relative amount of work per problem.
   - **`LOCAL_CHECKS`**: two problem sizes ~2–4× apart in formula memory. Each should run in a few minutes.

   Run it in their environment with the QPU mocked — never submit to real hardware while probing — and in the background if it takes a while.

   The probe measures memory and time across the whole process tree, the cores that actually help, and how much time could move to a GPU. It fits the formula to the two checks and prints, for each problem, an estimate, a time, a starting machine and a fallback. A constant that's off by a few × is fine, because the checks correct it. If the probe says the ratio drifts or the shape is wrong, the formula missed a copy, a worker count or the dominant object; fix it before trusting the bigger problems.

Present the result as the start of an experiment: "here is roughly what the bigger problems need; let's deploy and measure them on the real machines." Estimates far out tend to run high, so a problem estimated above the largest machine still gets a try on `48x768` (see Step 4).

## Step 3 — Wrap it as a function and deploy it

Put their code in a folder (everything in it is uploaded, max 50 MB) with one entrypoint file next to their modules:

```
my_workflow/
  entrypoint.py        # run_function(...) + __main__
  <their modules>.py   # unchanged where possible
```

The entrypoint imports their code, reads the arguments, reports progress, and returns a dict:

```python
import os, resource, sys, time, traceback
from qiskit_serverless import get_arguments, save_result, update_status, Job, get_runtime_service

from my_module import run_my_workflow          # their code


def _peak_gb():
    """Peak memory of the whole job: container counter on Linux, else largest process."""
    try:
        with open("/sys/fs/cgroup/memory.peak", encoding="utf-8") as fh:
            return round(int(fh.read()) / 1e9, 2)
    except (OSError, ValueError):
        unit = 1 if sys.platform == "darwin" else 1024
        return round(max(resource.getrusage(r).ru_maxrss
                         for r in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)) * unit / 1e9, 2)


def _cpu_s():
    return sum(resource.getrusage(r).ru_utime + resource.getrusage(r).ru_stime
               for r in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN))


def run_function(backend_name=None, **args):
    t0, c0, stages = time.perf_counter(), _cpu_s(), {}
    backend = get_runtime_service().backend(backend_name) if backend_name else None

    update_status(Job.POST_PROCESSING)                # or MAPPING, WAITING_QPU, ... per stage
    t = time.perf_counter()
    result = run_my_workflow(backend=backend, **args)
    stages["workflow_s"] = round(time.perf_counter() - t, 1)

    wall = time.perf_counter() - t0
    return {"result": result, "metadata": {"resources": {
        "peak_gb": _peak_gb(), "wall_s": round(wall, 1),
        "cores_used": round((_cpu_s() - c0) / wall, 1) if wall else None,
        "cores_granted": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
        "stages_s": stages}}}


if __name__ == "__main__":
    try:
        save_result(run_function(**get_arguments()))
    except Exception:
        save_result(traceback.format_exc())
        raise
```

Keep these in mind:

- **Arguments and results must be JSON-friendly**: numbers, strings, lists, dicts, and the Qiskit objects qiskit-serverless knows how to encode. `result` must be too, so convert arrays with `.tolist()`.
- **Log once per iteration** in long loops, with `print(..., flush=True)`. A run that's slower than expected then shows up in `job.logs()` early.
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
    sizes_map={"s": "8x32", "m": "16x128", "l": "48x768"},    # from the probe
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

## Step 4 — Run it and climb

Start with the smallest problem, then scale up one step at a time, using each run's real numbers to pick the next machine:

```python
job = fn.run(function_size="s", **problem_s)
print(job.job_id, job.status())       # INITIALIZING -> RUNNING: ... -> DONE (startup takes ~2-3 min)
print(job.logs())                     # per-iteration progress
out = job.result()
print(out["metadata"]["resources"])   # peak_gb, wall_s, cores_used, cores_granted, stages_s
```

After each run:

- **Correct the estimate.** Divide the measured `peak_gb` by the formula for that problem, and use the new ratio to re-estimate the next problem.
- **If it ran out of memory**, retry on the fallback machine. Only if `48x768` runs out of memory too, work out with the researcher how to split or reshape the problem; *Bigger than one machine* in `sizing_guide.md` has the questions.
- **If `cores_used` is far below `cores_granted`**, the extra cores came with the memory. That's fine, but don't climb for cores alone.
- **If run time grows faster than expected**, check the logs before submitting the next size. A 24-hour job that is 10× slower than planned should be split or extended, not resubmitted.

Keep going until something stops them; that is the interesting result. Then update `sizes_map` from what the runs showed.
