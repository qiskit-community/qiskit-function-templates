# Will it fit? Sizing a research workflow

Reference for Steps 2 and 4 of `SKILL.md`. Researchers usually come with one of a few questions — will the bigger problem fit in memory, how long will it take, would more cores or a GPU help, do I have to keep my laptop open. This guide helps answer them roughly, from what the researcher already knows and a few minutes of local runs, so they can pick a first machine and try the bigger problem on Managed Workflows. The real numbers come from a performance sweep after deploying (see **After deploying: the real sweep**). It also says plainly when the answer is "your laptop is fine".

## Start from what they know

Before measuring anything, ask how their algorithm scales with more cores, a GPU, and more memory. Many researchers know — from the method's paper, from earlier runs on a cluster, or from the library's documentation. If they do, use that as the formula and run the probe only to check it on their code at a small size. If they don't, work the formula out from the code together. Either way, the goal is a sensible first profile per tier, not a precise prediction.

## Estimate, then check

Ask what small, medium and large problems they actually want to run (labels `s`, `m`, `l`, and `xl` if needed — the service accepts no others). Each one is a whole workflow — every argument that changes cost (problem size, samples, iterations, depth), set together — and tiers differ only in size, never in method.

Then size them in two moves. A formula worked out from the code gets the shape of the growth right; two small real runs get the constant right and test the shape. Neither is enough alone: eyeballed numbers come out low, and curves fitted to small runs without a formula under-read exponential growth.

1. **Estimate from the code.** Answer the questions in **Finding what drives the cost**, write the dominant stage's memory as a formula in the tier arguments, and evaluate it for every tier by running the arithmetic — a few lines of Python — not by eye. Leave the constant rough; the checks set it.
2. **Check it with two small runs.** Put the formula in `MEMORY_GB` in `scale_probe.py` and two problem sizes in `LOCAL_CHECKS`, ~2–4× apart in formula memory and each running in a few minutes. Make the larger one big enough that the formula outweighs imports (a check whose formula gives less than ~0.05 GB measures only overhead). The probe fits measured memory as overhead + k × formula and time as a power of `TIME_UNITS`. A k of a few is normal — copies, workspace, library overhead. Make each check take at least a few seconds, or time cannot be fitted. What matters is that the ratio holds steady between the two checks; if it drifts, the formula has the wrong shape (a missed copy, a worker count, the wrong exponent) and the larger tiers will be off by more the further out they are. Recheck the formula with the researcher before trusting them.

How the check run is measured:

- **Setup is not counted.** Write `work` as a generator: do imports and one-time preparation, `yield`, then the work to be measured. Otherwise import time and JIT compilation hide the real work.
- **The whole process tree is measured.** Memory and CPU time are summed over the run and every process it starts, so worker pools, MPI ranks and compiled solvers (C++, Fortran, Julia) count. The CPU time also shows how many cores the run really used, and a short core sweep shows which counts help.
- **Failures are results.** A check that runs out of memory or time locally is reported with the reason. That is often why the researcher came, and it is the tier most worth running on a bigger machine.
- **Mock the QPU.** Use a fake backend, cached counts or fixed parameters. Never submit to real hardware while probing.
- **Time iterations, not whole runs, for optimizer loops** (AQC, QAOA, VQE). Have `work` set up and warm up the objective before the `yield`, then run a few iterations after it, and make `TIME_UNITS` count iterations × cost per iteration so the probe can scale the time to each tier.
- **Watch for libraries that cap their own memory.** Some solvers limit their working memory to a default (often a few GB) and trade the rest for time — the check runs then look memory-flat and slower than expected. If memory stops growing between checks while time grows faster, look for a max-memory setting, raise it for the checks, and set it to the profile's memory in the deployed function.
- **Leave exact checks out.** A full FCI reference, an exact-unitary comparison or an exhaustive enumeration usually grows faster than the method and becomes the largest cost at research scale. Measure the method with them switched off, and size them separately if the researcher wants them.

For each tier the probe prints the formula value, the fitted estimate, a rough time if `TIME_UNITS` is set, and a first profile plus a fallback: the smallest CPU profile that covers estimate × 1.3 and the helpful cores, and the next one up. Present these as a starting point to climb from, not a verdict.

Standard tools can cross-check or explain the probe's numbers when they are already installed; none is required. `memray run --native --follow-fork` followed by `memray stats --json` shows *which* allocation dominates the peak — useful when the formula is off. `py-spy record --subprocesses --idle --format raw` gives sampled stacks across workers (on macOS it needs sudo). On Linux, `systemd-run --user --scope -p MemoryAccounting=yes` then reading `memory.peak` gives the true peak of the whole process tree, and `/usr/bin/time -v` (`-l` on macOS) gives CPU time and the largest single process. For making the code faster rather than sizing it, point the researcher to Scalene.

## Rough memory formulas

A formula gives the *shape*: what grows, how fast, and what multiplies it. The constant in front is rough — copies, workspace and library overhead vary by code and version — and the check runs measure it. Write the shape from the code; do not tune the constant by hand.

| What dominates | Shape of peak memory |
|---|---|
| A state over n qubits | 16 B × 2^n (exact), × copies the code holds |
| A dense operator over n qubits | 16 B × 4^n, × copies |
| A dense N × N matrix | 8 B × N² (16 B if complex), × workspace for a factorization or eigensolve |
| A subspace built as a product of two index sets of size d (selected CI, SQD-like methods) | 8 B × d², × solver vectors and workspace (often tens), × subspaces held at once |
| A matrix-product state, n sites, bond dimension χ | 16–32 B × n × χ², × copies for gradients or environments |
| A list of K circuits or terms | K × size of one, × what is kept alongside (results, transpiled copies) |
| A sparse operator | nonzeros × (value + index bytes), where nonzeros = dimension × terms per row; × temporaries while it is built |
| A table or cache filled by a loop (dynamic programming, memoization) | entries × bytes per entry; in pure Python a dict or list entry costs ~100–200 B before its value |
| Results returned from parallel workers | the total once in the parent, plus again when they are concatenated or stacked |

Then work out what multiplies it, by walking through the moment of peak memory:

- **Count what is alive at the peak, not what is created.** A transform that returns a new array (`1 - x`, a copy, a cast, a concatenation) holds the old and the new together until the old one is dropped. Two or three live copies of the dominant object is typical.
- **Per-worker copies multiply; split work does not.** If each worker holds its own copy (its own state, subspace or model), memory grows with the worker count. If workers split a fixed total, total memory is roughly unchanged — but the parent often collects everything back and concatenates it, which adds one or two copies of the total.
- **Pure-Python data structures cost 10–30× the raw numbers.** Dicts, lists of tuples and big integers carry per-object overhead; a table of a billion small entries is ~100+ GB, not 8 GB.
- **Exponential drivers compound.** For 2^n or C(n, k) growth, a constant that is 2× off moves the answer by only one step in n, but a wrong base (2 vs 4, or a missed factor of n) is wrong by orders of magnitude at the target size. Get the base and the polynomial factor right from the code; let the checks set the constant.
- **Libraries may cap themselves.** Some solvers limit their own working memory and trade it for time, which hides the true shape (see below).

## Finding what drives the cost

The same questions work for any workflow, listed below or not:

0. **Which limit binds first — memory, time, or waiting on the QPU?** It sets the kind of advice. Memory-bound: a bigger machine is the lever, and caps or truncations are the knobs. Time-bound: cores and splitting into jobs are the levers, and the machine can stay small. QPU-bound: the smallest machine; the benefit is that the run survives the laptop closing.
1. **Which stage dominates at the sizes they care about?** Not at the tutorial's size — a stage that is trivial there (an exact check, a dense solve, a list of circuits) can take over at research scale. If it is not obvious, point `work` at one candidate stage at a time and compare.
2. **What makes that stage grow?** Qubits, orbitals, bond dimension, matrix size, number of circuits, samples, iterations. Take the effective size: if a cap, truncation or sampling limit binds first (a maximum subspace, a bond-dimension cap, a term budget, a sample count), that limit is the driver, not the nominal problem size. Check whether the cap is actually reached. If the driver is only known after running (bond dimension reached, terms kept, samples surviving post-selection), state the value the tier assumes as one of its arguments, and let the check run and the first cloud run correct it.

If memory grows with the number of workers (batches or circuits in flight at once), put the worker count in the tier's arguments and in the formula, and size each tier with the workers the profile you are about to pick would run. If that pushes the tier up a profile, cap the workers rather than climbing for memory the extra cores created.
3. **Which formula gives its memory?** Use the shapes above, and write it as a function of the tier arguments.
4. **Does that stage have a GPU path, and does it parallelize?** See **Would a GPU help?**; a loop over independent items parallelizes across cores almost perfectly.

These answers give `MEMORY_GB`, `TIME_UNITS` and so each tier's first profile.

### Common families at a glance

Starting points only — confirm against the researcher's code and the check runs.

| Family | Usually binds | Driver | Memory shape | Watch for |
|---|---|---|---|---|
| Tensor-network / MPS compilation and time evolution | time (optimizer loop), then memory | bond dimension actually reached; sites; iterations | MPS | a cap below what the target needs, or no cap; time iterations, not whole runs |
| Variational loops, simulated (QAOA, VQE) | memory | qubits; iterations | state | past ~32 qubits a full state stops fitting — approximate simulation or hardware |
| Variational loops on hardware | QPU | iterations, queue time | small | move the whole loop as one job if it runs for hours |
| Sample-based subspace methods (SQD, SKQD) | memory | the subspace cap or kept samples, whichever binds first — not orbitals alone; subspaces held at once | product subspace | exact references grow faster than the method itself |
| Krylov / exact-check workflows | memory of the check | qubits of the check | dense operator, state | size the method without its checks |
| Circuit cutting | time, then memory | number of cuts (circuit count grows exponentially in it) | circuit list | QPU sampling overhead grows faster still |
| Operator / Pauli propagation | memory and time | term budget × depth × qubits | term list | a fixed budget makes size sweeps look flat — sweep the budget |
| Kernels, QML | time | dataset size (circuits per kernel entry from the code) | circuit list, then the kernel matrix | in simulation, workers × state |
| Error correction, stabilizer simulation | time | variants × circuit size | small to moderate | parallel across variants |
| Transpilation sweeps | time | circuits × trials | small | perfectly parallel; limit native thread pools |

For a workflow not in the table, answer the four questions above and pick the closest formula.

## Would a GPU help?

`scale_probe.py` samples the call stack during the check run and reports the share of time spent in libraries that have a GPU version. That share caps the gain: if 60% of the time is portable, even an infinitely fast GPU gives at most 1 / 0.4 = 2.5×. A GPU is a good default for a tier when all of the following hold. When only some do — a smaller stage would run on a GPU but the stage that dominates at the largest tier would not — keep the tier on a CPU profile and add a GPU run to the sweep as a comparison (for example, `m` on both `16x128` and `24x120x1l40`). The sweep shows whether it pays off; that is an experiment worth running, not a reason to skip GPUs.

- **The share is high, and the GPU-portable stage is the one that grows** — at least ~0.6 on the check run, and the stage that dominates at the largest tier (from the formula) is that same stage.
- **The dominant stage has a GPU path** — dense linear algebra, large tensor contractions, mean-field chemistry and full-state simulation usually have one (often as a separate GPU build of the same library). Combinatorial and bit-string-heavy code, selected-CI solvers, configuration recovery, approximate-state simulators, transpilation and plain Python loops usually do not — for those, more cores or memory are what help. Check the library's documentation rather than assuming.
- **The arrays are large** — dense matrices around 2000×2000 or more, or big contractions. Small arrays run slower on a GPU because of transfer overhead.
- **The working set fits in one GPU** — 80 GB on an A100 or H100. Libraries use a single GPU unless the code shards explicitly, so a profile with several GPUs only helps code written to use them.

The probe cannot see inside compiled code: compute in Aer's C++ threads, in a subprocess, or in `@`/`np.dot` written directly in the researcher's code is reported as not portable or "n/a (compute outside Python)". In those cases decide from the library and the array sizes instead.

For example: *"most of the time is in a library with a GPU version and that share rises with size, so a GPU could help — compare one run on a GPU profile against a CPU run of the same problem before making it a tier's default."*

## Runs that take hours

If a problem takes more than ~30 minutes, or the researcher's real runs already do, the issue is often not the machine but keeping the run alive — a laptop left open overnight, runs restarted after sleep or a VPN drop, QPU jobs queued for hours, results collected over days, or `nohup`/`tmux` on a shared machine. Running it as a Managed Workflows function helps even when the problem fits locally: the workflow is persisted in the cloud, so it keeps running after the laptop closes, and `job.status()`, logs and `job.result()` work from anywhere — there is no save-and-restart logic to write.

A job can run for up to 24 hours; longer runs can be extended, or split into jobs (see below). A tier that fits in memory but looks longer than 24 hours is still a normal tier: give it the smallest profile that fits and note the time — longer runs are handled by time, not by a bigger machine. Rough runtimes before any cloud run are the least reliable number in the estimate, so let the first real run decide whether it needs an extension or a split. Use the smallest profile that fits, and:

- **Report progress.** Call `update_status(...)` at each stage boundary (`Job.MAPPING`, `Job.OPTIMIZING_HARDWARE`, `Job.WAITING_QPU`, `Job.EXECUTING_QPU`, `Job.POST_PROCESSING`) and log once per iteration, so a multi-hour job visibly makes progress.
- **Validate inputs first,** so a typo fails in seconds rather than at hour three.
- **Keep the profile small through long QPU waits.** A large machine waiting on the QPU queue sits idle.

For example: *"The medium problem fits your laptop (12 GB) but takes ~55 minutes on 8 cores. As a function it keeps running after you close the laptop. `8x32` is the smallest profile that fits; add `update_status` at each stage so you can follow it from `job.status()`."*

## Bigger than one machine

Every profile is a single machine — at most 48 cores and 768 GB on CPU, or 160 cores, 1.7 TB and 8 GPUs. When the rough memory estimate for a tier is more than that, treat it as a reason to try, not a verdict. Estimates that far out are crude and often high — workspace that shrinks relative to the problem, caps that bind, copies freed earlier than assumed — so suggest the researcher climb as usual and then run that tier on the largest profile anyway. If it fits, they have their result; if it runs out of memory, the last run shows exactly which stage hit the limit, and that is the moment to reshape.

If a real run on the largest profile runs out of memory, work out the reshaping together. The researcher knows which parts of their method can change without changing the science; ask before proposing anything:

- **Which stage is over the limit?** Use the last run's `stages` and the formula — often only one stage (the diagonalization, the subexperiment list, the exact reference) is the problem.
- **Are there pieces that are independent?** Batches, cut subexperiments, instances, time points or parameter values can run as separate jobs.
- **Is there an approximation they already trust?** Fewer kept samples, a truncation threshold, sampled instead of enumerated subexperiments, a sparse or iterative solver in place of a dense one.
- **Can the run stop and resume between stages?** A stage that writes its output can be the start of the next job.

Whatever they choose, check the reshaped version against a size that still fits in one piece, then keep climbing.

Splitting into jobs uses the function's user storage. Every job of the function can read and write it at `/function_user_data` inside a Fleets job (job-only scratch lives in `/job_user_data`), and the notebook reaches it with `serverless.file_upload`, `files` and `file_download`. Independent pieces can run as separate jobs that write their results there and combine them at the end:

- **SQD batches** — one job per batch or group of batches, writing each batch's energy and occupancies; a final job (or the notebook) combines them.
- **Circuit-cutting subexperiments** — generate once, run the subexperiments in parallel jobs, reconstruct at the end.
- **Instance and parameter sweeps** — one job per slice of the sweep.
- **Long chains** (AQC segments, outer self-consistency loops) — one job per stage, each reading the previous stage's output.

If the piece that must stay together still does not fit on the largest machine, the change has to come from the algorithm — smaller batches, fewer kept samples, a sparse or distributed solver. Agree that change with the researcher, then run the sweep again.

## After deploying: the real sweep

The local profile is rough by design; the deployed function is where the researcher gets real numbers. Once it is uploaded, encourage a scale-up sweep — the same incremental way they already work, just with bigger machines to climb onto:

- **Start with `s` on its profile**, then take the next bigger problem on the next profile up. Keep going until something stops them; that is the interesting result.
- **Read `metadata["resources"]` after each run** (see Step 3 of `SKILL.md`): `peak_gb`, `cores_used`, `cores_granted` and `stages`. Divide `peak_gb` by the formula for that problem to get a new ratio, re-estimate the next tier with it, and pick its profile from that — real numbers replace the local check one rung at a time.
- **If a run runs out of memory, retry on the fallback** rather than reshaping anything; that is what the fallback is for.
- **If `cores_used` is far below `cores_granted`**, the extra cores came with the memory, not because they help — that is fine, but no reason to climb for cores alone.
- **Add comparison runs where it is unclear** — the same problem on two neighbouring profiles, or a CPU and a GPU profile — to see what more cores or a GPU actually buy.
- **Update `sizes_map`** from what the runs showed, not from the local estimate.

A useful first experiment is the same problem on two or three profiles, comparing `wall_s` and `stages` from each result.

## When a laptop is enough

- **Short runs** — under ~10 minutes and fitting in memory. Starting a remote job takes longer than it saves.
- **Tight `Session` loops** (e.g. VQE calling Estimator every iteration) — each round trip must fit the ~60 s session window. Keep the loop local, or move the whole loop as one job.
- **Data that cannot leave the researcher's environment** — functions are cloud-hosted.
