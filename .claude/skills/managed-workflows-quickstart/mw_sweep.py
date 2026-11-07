# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Measure a deployed function on Managed Workflows, then size and climb its tiers.

Copy this file next to ``scale_probe.py`` (it reads ``TIERS``, ``MEMORY_GB`` and
``TIME_UNITS`` from there) and edit the block below. Run from your own machine::

    python mw_sweep.py sweep            # small jobs, one argument varied at a time
    python mw_sweep.py climb s m l      # run tiers smallest first, refit after each
    python mw_sweep.py climb m --from 8x32   # start tier m on a given machine
    python mw_sweep.py compare m 8x32 16x128   # same tier on two machines: what cores buy
    python mw_sweep.py --target l       # allocation check on tier l, as a job
                                        # (--cap-gb 4, --hits 1, --size s)
    python mw_sweep.py fit              # refit runs.json and print the advice again
    python mw_sweep.py collect          # wait for jobs a closed laptop or crash left behind
    python mw_sweep.py cancel           # cancel every job in jobs.json that has not ended

``sweep`` starts from ``BASE`` and makes it smaller along one argument at a time
(``SCALES``), so every argument in ``SWEEP_ARGS`` gets its own small runs. All jobs
are submitted at once and collected into ``runs.json``. The fit is
``scale_probe``'s, plus a shape test per argument: along each argument, measured
memory growth is compared with the growth the formula predicts, which catches an
argument the formula weights wrongly or leaves out.

``climb`` runs each tier on the machine the fit suggests and handles failure:

* numpy ``Unable to allocate X ...``: the size is read from the error, machines too
  small for it are skipped, and it says so when X is beyond the largest machine;
* killed with no traceback (out of memory): one retry on the next machine up;
* any other error: stop and show the traceback, with no retry.

Every job id is written to ``jobs.json`` as soon as the job is submitted, so a run
interrupted by a closed laptop or a crash can be picked up with ``collect`` (or
stopped with ``cancel``). Submission keeps at most ``MAX_ACTIVE`` jobs active and
waits when the service answers 429 (the active job limit), instead of failing
half-way. Waiting for a job gives up after ``JOB_TIMEOUT_S`` and cancels it, so a
stuck run does not keep using the instance's allocation for up to 24 hours.

Every number printed comes from a job: its id and ``metadata["resources"]`` are
kept in ``runs.json``. The entrypoint must pass ``alloc_check`` to ``telemetry()``
for ``--target`` (SKILL.md, Step 3). Machines are the CPU profiles the deployment
lists in ``mw.compute_profiles()``. Needs ``qiskit-ibm-catalog>=0.22``; older clients fall back to the
profiles in ``scale_probe``.
"""

import json
import os
import re
import sys
import time

import scale_probe as sp

# ============================ EDIT THIS BLOCK ============================

ACCOUNT = "mw"  # the name used with ManagedWorkflowClient.save_account
FUNCTION = "my-workflow"  # the title used at upload
BASE = None  # a problem to sweep down from; None uses the smallest tier in TIERS
SWEEP_ARGS = None  # arguments that change cost; None uses every numeric argument in BASE
SCALES = (0.25, 0.5)  # memory formula ratio to BASE for each small run, per argument
SWEEP_SIZE = "s"  # size label to run the small jobs on
MAX_PARALLEL = 6  # small jobs per sweep
MAX_ACTIVE = 10  # jobs active at once from this script (the service limit is 50 per user)
JOB_TIMEOUT_S = 2 * 3600  # waiting longer than this cancels the job; raise it for long tiers
POLL_S = 60

# ========================= machinery below this line =========================

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs.json")
JOBS = os.path.join(HERE, "jobs.json")
LOG = os.path.join(HERE, "mw_sweep.log")
_ALLOC_RE = re.compile(
    r"Unable to allocate ([\d.]+) (B|KiB|MiB|GiB|TiB|PiB) for an array with shape \(([^)]*)\)"
    r"(?: and data type (\w+))?"
)
_UNITS = {"B": 1, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "TiB": 2**40, "PiB": 2**50}
_KILL_HINTS = ("oomkilled", "out of memory", "exit code 137", "signal 9", "killed")


def _say(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def _load_runs():
    if not os.path.exists(RUNS):
        return []
    with open(RUNS, encoding="utf-8") as fh:
        return json.load(fh)


def _save_run(entry):
    runs = _load_runs()
    runs.append(entry)
    with open(RUNS + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(runs, fh, indent=1)
    os.replace(RUNS + ".tmp", RUNS)  # never leave a half-written runs.json


def _function():
    from qiskit_ibm_catalog import ManagedWorkflowClient  # pylint: disable=import-outside-toplevel

    mw = ManagedWorkflowClient(name=ACCOUNT)
    use_deployment_profiles(mw)
    fn = mw.load(FUNCTION)
    if fn is None:
        sys.exit(f"mw_sweep: function {FUNCTION!r} not found; upload it first (Step 3)")
    return fn


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def use_deployment_profiles(client):
    """Replace ``scale_probe.CPU_PROFILES`` with the deployment's CPU-only profiles.

    Keeps the built-in list when the client has no ``compute_profiles()`` or the call fails.
    """
    try:
        catalog = list(client.compute_profiles())
    except Exception as exc:  # pylint: disable=broad-except
        _say(f"compute_profiles() unavailable ({exc.__class__.__name__}); using built-in list")
        return sp.CPU_PROFILES
    cpu = []
    for prof in catalog:
        cores, mem = _num(getattr(prof, "cpu", None)), _num(getattr(prof, "memory", None))
        if getattr(prof, "gpu", None) or cores is None or mem is None:
            continue
        cpu.append((prof.compute_profile_id, int(cores), int(mem) if mem.is_integer() else mem))
    if cpu:
        sp.CPU_PROFILES = sorted(cpu, key=lambda p: (p[2], p[1]))
    return sp.CPU_PROFILES


def _profile_of(fn, label):
    sizes = {k.lower(): str(v) for k, v in (getattr(fn, "sizes_map", None) or {}).items()}
    return sizes.get(label.lower(), label)


def _label_for(fn, profile):
    """Size label in the function's sizes_map that maps to a profile, if any."""
    for label, prof in (getattr(fn, "sizes_map", None) or {}).items():
        if str(prof) == profile:  # a value can be a ComputeProfile; str() is its id
            return label
    return None


def _record(job, problem, payload, profile, **extra):
    """One runs.json entry; the profile the job reports wins over the sizes_map lookup."""
    entry = {
        "problem": problem,
        "resources": payload,
        "job_id": job.job_id,
        "profile": payload.get("compute_profile") or profile,
        **extra,
    }
    fleet = getattr(job, "fleet_id", None)
    if fleet:
        entry["fleet_id"] = fleet
    return entry


def _load_jobs():
    if not os.path.exists(JOBS):
        return {}
    with open(JOBS, encoding="utf-8") as fh:
        return json.load(fh)


def _note_job(job, **fields):
    """Record a job in jobs.json the moment it exists, and update it as it ends."""
    jobs = _load_jobs()
    entry = jobs.setdefault(job.job_id, {"submitted": time.time()})
    entry.update(fields)
    with open(JOBS + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(jobs, fh, indent=1)
    os.replace(JOBS + ".tmp", JOBS)


def _open_jobs():
    return [jid for jid, e in _load_jobs().items() if not e.get("final")]


def _is_limit_error(exc):
    return getattr(exc, "_code", None) == 429 or "active job limit" in str(exc).lower()


def submit(fn, problem, function_size, **meta):
    """``fn.run`` that keeps under MAX_ACTIVE, waits out a 429 and records the job id first."""
    waited = 0
    while len(_open_jobs()) >= MAX_ACTIVE:
        _refresh_open()
        if len(_open_jobs()) < MAX_ACTIVE:
            break
        if waited == 0:
            _say(f"  {MAX_ACTIVE} jobs active; waiting for one to end before submitting")
        time.sleep(POLL_S)
        waited += POLL_S
    for attempt in range(30):
        try:
            job = fn.run(function_size=function_size, **problem)
            break
        except Exception as exc:  # pylint: disable=broad-except
            if not _is_limit_error(exc) or attempt == 29:
                raise
            if attempt == 0:
                _say("  active job limit reached (429); retrying every minute. Jobs of other "
                     "workflows count too: `mw_sweep.py cancel` stops this script's jobs")
            time.sleep(POLL_S)
    _note_job(job, problem=problem, function_size=function_size, final=None, **meta)
    return job


def _refresh_open():
    """Mark jobs in jobs.json that have ended since we last looked."""
    from qiskit_ibm_catalog import ManagedWorkflowClient  # pylint: disable=import-outside-toplevel

    mw = ManagedWorkflowClient(name=ACCOUNT)
    for jid in _open_jobs():
        try:
            status = mw.job(jid).status()
        except Exception:  # pylint: disable=broad-except
            continue
        if status in ("DONE", "ERROR", "CANCELED"):
            _note_job(mw.job(jid), final=status)


def _wait(job):
    """Final status of a job, cancelling it if it is still not done after JOB_TIMEOUT_S."""
    started = _load_jobs().get(job.job_id, {}).get("submitted", time.time())
    remaining = max(60.0, JOB_TIMEOUT_S - (time.time() - started))
    if hasattr(job, "wait_for_final_state"):
        try:
            status = job.wait_for_final_state(timeout=remaining)  # backs off, retries short errors
        except TimeoutError:
            _say(f"  {job.job_id[:8]}: not done after {JOB_TIMEOUT_S / 3600:.1f} h; cancelling "
                 f"(raise JOB_TIMEOUT_S for long tiers): {job.cancel()}")
            status = job.wait_for_final_state(timeout=600)
        _note_job(job, final=status)
        return status
    deadline = time.time() + remaining
    failures = 0
    while True:
        if time.time() > deadline:
            _say(f"  {job.job_id[:8]}: not done after {JOB_TIMEOUT_S / 3600:.1f} h; cancelling")
            job.cancel()
            deadline = time.time() + 600
        try:
            status = job.status()
            failures = 0
        except Exception as exc:  # pylint: disable=broad-except
            failures += 1
            if failures >= 10:
                raise RuntimeError(f"cannot reach job {job.job_id}: {exc}") from exc
            status = None
        if status in ("DONE", "ERROR", "CANCELED"):
            _note_job(job, final=status)
            return status
        time.sleep(POLL_S)


# ------------------------------------------------------------- outcomes


def parse_alloc_error(text):
    """``(gib_requested, shape, dtype)`` from a numpy allocation error, else None."""
    found = _ALLOC_RE.findall(text or "")
    if not found:
        return None
    num, unit, shape, dtype = found[-1]
    dims = [int(d) for d in shape.replace(" ", "").split(",") if d]
    return float(num) * _UNITS[unit] / 1e9, dims, dtype or None


def outcome(job):  # pylint: disable=too-many-return-statements
    """Wait for a job; return ``(kind, payload)``.

    kind is ``ok`` (payload = resources), ``alloc`` (GB, shape, dtype, text),
    ``killed`` (text) or ``error`` (traceback text).
    """
    status = _wait(job)
    if status == "DONE":
        res = job.result()
        if isinstance(res, dict) and "resources" in (res.get("metadata") or {}):
            return "ok", res["metadata"]["resources"]
        return "error", f"job finished without metadata.resources: {str(res)[:2000]}"
    if status == "CANCELED":
        return "error", "job was canceled"
    try:
        text = job.error_message() or ""  # a killed job returns an empty dict here
    except Exception as exc:  # pylint: disable=broad-except
        text = f"(error_message failed: {exc})"
    text = text if isinstance(text, str) else json.dumps(text) if text else ""
    try:
        logs = job.logs() or ""
    except Exception:  # pylint: disable=broad-except
        logs = ""
    alloc = parse_alloc_error(text) or parse_alloc_error(logs)
    if alloc:
        return "alloc", (*alloc, text or logs[-3000:])
    if "Traceback" in text or "Traceback" in logs:
        return "error", text if "Traceback" in text else logs[-6000:]
    low = (text + logs[-3000:]).lower()
    if any(h in low for h in _KILL_HINTS) or not text.strip():
        return "killed", _kill_evidence(logs, text)
    return "error", text


def _kill_evidence(logs, text):
    """One line on how the run ended, from its telemetry lines in the logs."""
    tag = "MW_TELEMETRY "
    recs = []
    for line in logs.splitlines():
        if tag in line:
            try:
                recs.append(json.loads(line[line.index(tag) + len(tag) :]))
            except ValueError:
                pass
    if not recs:
        return f"no traceback and no telemetry in the logs {text.strip()[:200]}".strip()
    last = recs[-1]
    ended = any(r.get("event") == "end" for r in recs)
    return (
        f"no traceback; logs stop at '{last.get('message')}' "
        f"(peak so far {last.get('peak_gb')} GB){'' if not ended else ', after telemetry end'}"
    )


# ------------------------------------------------------------- sweep


def _numeric(problem):
    return [
        k for k, v in problem.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]


def _shrink(base, arg, ratio):
    """Smaller value of ``arg`` giving ~``ratio`` x BASE's formula memory (bisection)."""
    goal = ratio * sp.MEMORY_GB(base)
    lo, hi = 0.0, float(base[arg])
    for _ in range(60):
        mid = (lo + hi) / 2
        trial = {**base, arg: int(round(mid)) if isinstance(base[arg], int) else mid}
        if sp.MEMORY_GB(trial) < goal:
            lo = mid
        else:
            hi = mid
    value = int(round(hi)) if isinstance(base[arg], int) else hi
    value = max(value, 1) if isinstance(value, int) else value
    prob = {**base, arg: value}
    if sp.MEMORY_GB(prob) > 1.5 * goal:  # e.g. a constant term in the formula
        return None
    return prob


def plan_sweep():
    """BASE, then BASE shrunk along each argument; drops arguments the formula ignores."""
    base = BASE or next(iter(sp.TIERS.values()))
    args = SWEEP_ARGS or _numeric(base)
    plan, seen = [("base", base)], {json.dumps(base, sort_keys=True)}
    for arg in args:
        half = {
            **base,
            arg: max(base[arg] // 2, 1) if isinstance(base[arg], int) else base[arg] / 2,
        }
        if abs(sp.MEMORY_GB(half) - sp.MEMORY_GB(base)) < 1e-12:
            _say(f"  {arg}: not in MEMORY_GB, so swept as half of BASE; a change shows as SHAPE")
            plan.append((f"{arg}=x0.5", half))
            continue
        for ratio in SCALES:
            prob = _shrink(base, arg, ratio)
            if prob is None:
                _say(f"  {arg}: no value gives x{ratio} of BASE's formula memory; skipped")
                continue
            key = json.dumps(prob, sort_keys=True)
            if key not in seen:
                seen.add(key)
                plan.append((f"{arg}~x{ratio}", prob))
    return plan, args


def sweep():
    """Submit the small runs at once, collect them into runs.json, and fit."""
    plan, args = plan_sweep()
    fn = _function()
    _say(f"sweep: {len(plan)} jobs on {SWEEP_SIZE} ({_profile_of(fn, SWEEP_SIZE)})")
    jobs = []
    for name, problem in plan[:MAX_PARALLEL]:
        job = submit(fn, problem, SWEEP_SIZE, sweep=name)
        jobs.append((name, problem, job))
        _say(f"  {name:<14} {job.job_id}  {problem}  formula {sp.MEMORY_GB(problem):.3g} GB")
    if len(plan) > MAX_PARALLEL:
        _say(f"  {len(plan) - MAX_PARALLEL} more skipped: raise MAX_PARALLEL to run them")
    for name, problem, job in jobs:
        kind, payload = outcome(job)
        if kind == "ok":
            _save_run(_record(job, problem, payload, _profile_of(fn, SWEEP_SIZE), sweep=name))
            _say(
                f"  {name:<14} {job.job_id[:8]}  peak {payload['peak_gb']} GB "
                f"{payload['wall_s']} s  cores {payload['cores_used']}/{payload['cores_granted']}"
            )
        else:
            _say(f"  {name:<14} {job.job_id[:8]}  {kind.upper()}: {_short(payload)}")
    fit(args)


def _short(payload):
    if isinstance(payload, tuple):
        return f"asked for {payload[0]:,.1f} GB, shape {tuple(payload[1])}"
    return str(payload).strip().splitlines()[-1][:300] if str(payload).strip() else "(empty)"


# ------------------------------------------------------------- fit + advice


def shape_by_arg(runs, args):
    """Per argument: measured memory growth over the formula's growth, ~1 if it matches."""
    lines = []
    if not runs:
        return lines
    base_mem = sp._fit_memory(runs, [])  # pylint: disable=protected-access
    for arg in args:
        pts = []
        for run in (r for r in runs if arg in r["problem"]):
            others = {k: v for k, v in run["problem"].items() if k != arg}
            pts.append((json.dumps(others, sort_keys=True), run))
        groups = {}
        for key, run in pts:
            groups.setdefault(key, []).append(run)
        if not groups:
            continue
        best = max(groups.values(), key=len)
        best = sorted(best, key=lambda r, a=arg: r["problem"][a])
        if len(best) < 2 or best[0]["problem"][arg] == best[-1]["problem"][arg]:
            continue
        lo, hi = best[0], best[-1]
        start = [r["resources"].get("start_gb") for r in (lo, hi)]
        overhead = max(start) if None not in start else base_mem["base"]  # memory before the work
        overhead = max(min(overhead, 0.9 * lo["resources"]["peak_gb"]), 0.0)
        grow_meas = (hi["resources"]["peak_gb"] - overhead) / max(
            lo["resources"]["peak_gb"] - overhead, 1e-9
        )
        grow_form = sp.MEMORY_GB(hi["problem"]) / max(sp.MEMORY_GB(lo["problem"]), 1e-12)
        ratio = grow_meas / grow_form
        span = f"{arg} {lo['problem'][arg]} -> {hi['problem'][arg]}"
        flat = hi["resources"]["peak_gb"] - lo["resources"]["peak_gb"] < sp.MIN_ADDED_GB
        if flat and sp.MEMORY_GB(hi["problem"]) - sp.MEMORY_GB(lo["problem"]) < sp.MIN_ADDED_GB:
            lines.append(f"{span}: memory barely changes; runs too small to test this argument")
        elif flat:
            lines.append(
                f"SHAPE WRONG along {arg}: {span}, memory stays at {hi['resources']['peak_gb']} GB "
                f"but the formula grows x{grow_form:.2g}. The peak comes from an object the "
                f"formula leaves out (one that does not depend on {arg}, such as per-worker "
                "buffers); find it before trusting tiers."
            )
        elif 0.67 < ratio < 1.5:
            lines.append(f"{span}: memory grows x{grow_meas:.2g}, formula x{grow_form:.2g} - ok")
        else:
            word = "faster" if ratio > 1 else "slower"
            lines.append(
                f"SHAPE WRONG along {arg}: {span}, memory grows x{grow_meas:.2g} but the formula "
                f"x{grow_form:.2g} ({ratio:.2g}x {word}). The formula weights {arg} wrongly "
                "or misses an object that depends on it; fix MEMORY_GB before trusting tiers."
            )
    return lines


def core_advice(runs):
    """Core count to ask for, from the measured cores_used."""
    out = []
    ok = [r for r in runs if (r["resources"].get("wall_s") or 0) > 5]
    if not ok:
        return out
    run = max(ok, key=lambda r: r["resources"]["peak_gb"])
    big = run["resources"]
    used, granted = big.get("cores_used") or 1, big.get("cores_granted") or 1
    vcpus = {p[0]: p[1] for p in sp.CPU_PROFILES}  # granted can count hardware threads
    fits = [p[1] for p in sp.CPU_PROFILES if p[1] <= granted] or [granted]
    vcpu = vcpus.get(run.get("profile"), fits[-1])
    if used < 0.5 * min(granted, vcpu):
        out.append(
            f"cores: the largest run used {used:.1f} of {granted} granted - pick machines for "
            "memory; more cores will not speed this up"
        )
    else:
        names = [p[0] for p in sp.CPU_PROFILES]
        here = next((p[0] for p in sp.CPU_PROFILES if p[1] >= vcpu), names[-1])
        nxt = names[names.index(here) + 1] if names.index(here) + 1 < len(names) else None
        out.append(f"cores: the largest run kept {used:.1f} of {granted} busy - it can use cores")
        if nxt:
            out.append(
                f"  try one comparison: python mw_sweep.py compare <tier> {here} {nxt}. "
                "Climb for cores only if wall time drops clearly."
            )
    return out


def fit(args=None):
    """Fit runs.json with scale_probe, then add the per-argument shape test and core advice."""
    runs = [r for r in _load_runs() if (r.get("resources") or {}).get("peak_gb") is not None]
    if not runs:
        sys.exit("mw_sweep: no successful runs in runs.json")
    rows = sp.size(runs)
    args = args or SWEEP_ARGS or _numeric(runs[0]["problem"])
    extra = shape_by_arg(runs, args) + core_advice(runs)
    for line in extra:
        _say("  - " + line)
    return rows


# ------------------------------------------------------------- climb


def _next_profile(profile, need_gb=0.0):
    """Next CPU profile above ``profile`` with more than ``need_gb`` of memory."""
    names = [p[0] for p in sp.CPU_PROFILES]
    start = names.index(profile) + 1 if profile in names else 0
    for name, _, mem in sp.CPU_PROFILES[start:]:
        if mem > need_gb:  # the array alone; the rest of the run needs more, so climb again
            return name
    return None


def climb(labels, start=None):
    """Run tiers smallest first on the fitted machine; climb on out-of-memory."""
    fn = _function()
    names = [p[0] for p in sp.CPU_PROFILES]
    for label in labels:
        problem = sp.TIERS[label]
        row = {"estimate_gb": float("nan"), "first": names[0]}
        if any((r.get("resources") or {}).get("peak_gb") is not None for r in _load_runs()):
            row = {r["tier"]: r for r in fit()}[label]
        else:
            _say(f"{label}: no measured runs yet, starting on {names[0]}")
        profile = (
            row["first"] if row["first"] in names else names[0 if row["first"] == "laptop" else -1]
        )
        if start:
            profile, start = start, None
            _say(f"{label}: starting on {profile} as asked (--from)")
        retried = False
        while True:
            size_label = _label_for(fn, profile)
            if size_label is None:
                _say(f"{label}: no size label maps to {profile}; add one to sizes_map, re-upload")
                return
            job = submit(fn, problem, size_label, tier=label, profile=profile)
            _say(f"{label}: {job.job_id} on {profile} (estimate {row['estimate_gb']:.3g} GB)")
            kind, payload = outcome(job)
            if kind == "ok":
                _save_run(_record(job, problem, payload, profile, tier=label))
                _say(
                    f"{label}: peak {payload['peak_gb']} GB, {payload['wall_s']} s, cores "
                    f"{payload['cores_used']}/{payload['cores_granted']} - added, refitting"
                )
                break
            if kind == "alloc":
                gb, shape, dtype, _ = payload
                _say(f"{label}: numpy asked for {gb:,.1f} GB, shape {tuple(shape)} {dtype or ''}")
                profile = _next_profile(profile, gb)
                if profile is None:
                    _say(
                        f"{label}: one array of {gb:,.0f} GB is beyond the largest machine "
                        f"({sp.CPU_PROFILES[-1][2]} GB). Split or reshape the problem "
                        "(sizing_guide.md, 'Bigger than one machine')."
                    )
                    return
                _say(f"{label}: skipping to {profile}, the first machine that holds it")
                continue
            if kind == "killed" and not retried and _next_profile(profile):
                retried, profile = True, _next_profile(profile)
                _say(
                    f"{label}: killed with no traceback (likely out of memory): "
                    f"{_short(payload)}; one retry on {profile}"
                )
                continue
            if kind == "killed":
                _say(
                    f"{label}: killed again on {profile} with no traceback: {_short(payload)}. "
                    "Stopping; see 'Bigger than one machine' in sizing_guide.md."
                )
                return
            _say(f"{label}: {kind.upper()} on {profile}, stopping. Error:\n{payload}")
            return


# ------------------------------------------------------------- compare


def compare_verdict(run_a, run_b):
    """One line: speedup for the extra cores, and whether memory grew with them."""
    (p_a, r_a), (p_b, r_b) = run_a, run_b
    speedup = r_a["wall_s"] / max(r_b["wall_s"], 1e-9)
    cores = {p[0]: p[1] for p in sp.CPU_PROFILES}
    ratio = cores.get(p_b, 1) / cores.get(p_a, 1)
    if speedup >= 0.75 * ratio:
        verdict = "worth climbing for time"
    elif speedup >= 1.15:
        verdict = "modest: climb only if the time matters more than the machine"
    else:
        verdict = "not worth climbing for"
    line = f"{p_b} is {speedup:.2f}x faster than {p_a} for {ratio:.0f}x the cores - {verdict}"
    if r_b["peak_gb"] > 1.2 * r_a["peak_gb"]:
        line += (
            f"; memory grew with the cores ({r_a['peak_gb']} -> {r_b['peak_gb']} GB), so size "
            "each machine with its own worker count"
        )
    return line


def compare(label, profiles=None):
    """Run tier ``label`` on each profile at once; report time and cores from the jobs."""
    fn = _function()
    profiles = profiles or [p[0] for p in sp.CPU_PROFILES[:2]]
    problem = sp.TIERS[label]
    jobs = []
    for profile in profiles:
        size_label = _label_for(fn, profile)
        if size_label is None:
            _say(f"compare: no size label maps to {profile}; add one to sizes_map, re-upload")
            return None
        job = submit(fn, problem, size_label, tier=label, profile=profile, compare=True)
        jobs.append((profile, job))
        _say(f"compare {label}: {job.job_id} on {profile}")
    done = []
    for profile, job in jobs:
        kind, payload = outcome(job)
        if kind != "ok":
            _say(f"compare {label}: {kind.upper()} on {profile}: {_short(payload)}")
            continue
        _save_run(_record(job, problem, payload, profile, tier=label, compare=True))
        done.append((profile, payload))
        _say(
            f"compare {label}: {profile:<8} {payload['wall_s']} s, cores "
            f"{payload['cores_used']}/{payload['cores_granted']}, peak {payload['peak_gb']} GB"
        )
    if len(done) >= 2:
        _say("compare " + label + ": " + compare_verdict(done[0], done[-1]))
    return done


# ------------------------------------------------------------- --target


def target(label, cap_gb=4.0, max_hits=1, size_label=None):
    """Run tier ``label`` as a job with the allocation check on; report what it asks for."""
    fn = _function()
    problem = sp.TIERS[label]
    size_label = size_label or next(iter(getattr(fn, "sizes_map", None) or {"s": ""}))
    job = submit(fn, {**problem, "alloc_check": {"cap_gb": cap_gb, "max_hits": max_hits}},
                 size_label, tier=label, target=True)
    _say(f"--target {label}: {job.job_id} on {_profile_of(fn, size_label)}, cap {cap_gb} GB")
    kind, payload = outcome(job)
    if kind != "ok":
        _say(f"--target {label}: {kind.upper()}: {_short(payload)}")
        if kind == "error":
            _say(str(payload))
        return None
    check = payload.get("alloc_check")
    if not check:
        _say("--target: no alloc_check in the result; pass alloc_check to telemetry() (Step 3)")
        return None
    _say(f"--target {label} {problem}: {check['outcome']}, {payload['wall_s']} s")
    for i, hit in enumerate(check["hits"], 1):
        _say(
            f"  #{i} {hit['call']}{tuple(hit['shape'])} {hit['dtype']} = {hit['gb']:,.1f} GB "
            f"at {' < '.join(hit['sites'])}"
        )
    if check["hits"]:
        big = max(h["gb"] for h in check["hits"])
        first, fallback = sp._rungs(big * sp.HEADROOM, 1)  # pylint: disable=protected-access
        _say(
            f"  largest single array {big:,.1f} GB: the peak is at least this. Machine for it "
            f"alone: {first}{' (fallback ' + fallback + ')' if fallback else ''}"
        )
        if big > sp.CPU_PROFILES[-1][2]:
            _say("  beyond the largest machine: split or reshape before running it")
    else:
        _say(
            f"  no numpy array above {check['cap_gb']} GB; memory is in many smaller arrays "
            "or outside numpy - use the fit instead"
        )
    entry = {"problem": problem, "alloc_check": check, "job_id": job.job_id, "tier": label}
    with open(os.path.join(HERE, "target.json"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return check


def collect():
    """Wait for the jobs in jobs.json that have not ended, and add the finished ones to runs.json."""
    from qiskit_ibm_catalog import ManagedWorkflowClient  # pylint: disable=import-outside-toplevel

    mw = ManagedWorkflowClient(name=ACCOUNT)
    use_deployment_profiles(mw)
    seen = {r.get("job_id") for r in _load_runs()}
    pending = _open_jobs()
    _say(f"collect: {len(pending)} job(s) not ended in {JOBS}")
    for jid, meta in _load_jobs().items():
        if jid not in pending and (meta.get("final") != "DONE" or jid in seen):
            continue
        job = mw.job(jid)
        kind, payload = outcome(job)
        extra = {k: meta[k] for k in ("sweep", "tier", "compare") if k in meta}
        if kind == "ok" and jid not in seen and not meta.get("target"):
            _save_run(_record(job, meta.get("problem", {}), payload,
                              meta.get("profile") or meta.get("function_size"), **extra))
            _say(f"  {jid[:8]} {extra}: peak {payload['peak_gb']} GB, {payload['wall_s']} s - added")
        elif kind != "ok":
            _say(f"  {jid[:8]} {extra}: {kind.upper()}: {_short(payload)}")


def cancel_open():
    """Cancel every job this script submitted that has not ended."""
    from qiskit_ibm_catalog import ManagedWorkflowClient  # pylint: disable=import-outside-toplevel

    mw = ManagedWorkflowClient(name=ACCOUNT)
    for jid in _open_jobs():
        job = mw.job(jid)
        _say(f"cancel {jid}: {job.cancel()}")
        _note_job(job, final=job.status())


def main(argv):
    """Command-line entry."""
    if not argv:
        print(__doc__)
    elif argv[0] == "sweep":
        sweep()
    elif argv[0] == "fit":
        fit()
    elif argv[0] == "collect":
        collect()
    elif argv[0] == "cancel":
        cancel_open()
    elif argv[0] == "climb":
        start = argv[argv.index("--from") + 1] if "--from" in argv else None
        labels = [a for a in argv[1:] if a in sp.TIERS]
        climb(labels or list(sp.TIERS), start)
    elif argv[0] == "compare":
        compare(argv[1], argv[2:] or None)
    elif argv[0] == "--target":
        cap = float(argv[argv.index("--cap-gb") + 1]) if "--cap-gb" in argv else 4.0
        size_label = argv[argv.index("--size") + 1] if "--size" in argv else None
        hits = int(argv[argv.index("--hits") + 1]) if "--hits" in argv else 1
        target(argv[1], cap_gb=cap, max_hits=hits, size_label=size_label)
    else:
        sys.exit(f"mw_sweep: unknown command {argv[0]!r}\n{__doc__}")


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    main(sys.argv[1:])
