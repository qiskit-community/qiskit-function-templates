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

"""Fit a workflow's memory and time from measured runs, and pick a machine per tier.

Copy this file next to the research code and edit the block below. Three modes::

    python scale_probe.py --estimate        # formula only, nothing runs (a lower bound)
    python scale_probe.py --check           # run SMOKE once locally, capped: does the code run
    python scale_probe.py runs.json         # fit measured runs and size every tier

``runs.json`` is a list of ``{"problem": {...}, "resources": {...}}`` entries, where
``resources`` is the ``metadata["resources"]`` that ``mw_telemetry`` returns. The
runs normally come from small jobs on Managed Workflows (Step 3 of SKILL.md), so
they are measured on the same machines, libraries and memory accounting as the
real runs; ``--check`` appends its local run to the same file.

``--check`` is a smoke test, not a measurement: it runs ``SMOKE`` (or a tier, if it
fits the caps) in a child process group at the lowest priority, with BLAS threads
and ``WORKERS_ARG`` capped, and kills the whole group, worker processes included,
at ``LOCAL_CAPS["wall_s"]`` seconds or ``LOCAL_CAPS["gb"]`` GB. It refuses to start
on a machine that is already busy. Size tiers on Managed Workflows, not here.

The fit is ``peak = overhead + k * MEMORY_GB(problem)`` for memory and
``time = t0 * (TIME_UNITS / u0) ** a`` for time. The formula gives the shape, the
runs give the constants, and two or more runs at different sizes test the shape.
Standard library only (``mw_telemetry`` uses ``psutil`` if installed).
"""

import json
import math
import os
import sys
import time

# ============================ EDIT THIS BLOCK ============================


def work(**problem):
    """EDIT: run THEIR workflow (QPU mocked) on one problem, for --check (SMOKE by default).

    Import and call the researcher's own functions; never a stand-in that imitates
    them. Use the same entrypoint code the cloud jobs run, if there is one.
    """
    sys.path.insert(0, HERE)  # EDIT: folder that holds their modules
    from my_workflow import run  # pylint: disable=import-outside-toplevel,import-error

    return run(**problem)


# EDIT: the smallest problem that still runs every stage, in seconds; what --check runs.
SMOKE = {"n": 500, "n_vectors": 4, "workers": 1}
WORKERS_ARG = "workers"  # EDIT: argument that sets processes/threads, or None; --check caps it

TIERS = {  # labels must be s, m, l or xl; the problems the researcher wants to run
    "s": {"n": 4000, "n_vectors": 20, "workers": 1},
    "m": {"n": 20000, "n_vectors": 20, "workers": 4},
    "l": {"n": 60000, "n_vectors": 40, "workers": 8},
}


def MEMORY_GB(p):  # pylint: disable=invalid-name
    """EDIT: rough GB of the biggest object(s) for problem p - the shape, not the constant.

    Count what is alive at the peak and what multiplies it (sizing_guide.md). The
    measured runs set the constant, so leave copies and workspace rough.
    """
    # e.g. each worker holds a dense n x n matrix plus n_vectors work vectors
    return p["workers"] * 8 * (p["n"] ** 2 + p["n_vectors"] * p["n"]) / 1e9


TIME_UNITS = None  # optional: lambda p: relative amount of work, e.g. p["iterations"] * p["n"] ** 3

# ========================= machinery below this line =========================
# pylint: disable=not-callable

VALID_LABELS = ("S", "M", "L", "XL")
HERE = os.path.dirname(os.path.abspath(__file__))
ACCOUNT = None  # the name used with ManagedWorkflowClient.save_account; None = default account
# (id, vCPU, GB) of the deployment's CPU profiles, smallest first, read from
# ``ManagedWorkflowClient().compute_profiles()`` by load_profiles(); never hard-coded.
CPU_PROFILES = []
HEADROOM = 1.3
LAPTOP_GB = 12  # estimate x headroom at or below this fits a typical laptop
MIN_ADDED_GB = 0.05  # a run adding less than this over the overhead measures only imports
# --check runs on the researcher's own machine: keep it a smoke test that cannot take it over.
LOCAL_CAPS = {"wall_s": 120, "gb": 4.0, "workers": 2}
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
)


def _validate():
    """Fail fast with a clear message on a malformed edit block."""
    errors = []
    if not isinstance(SMOKE, dict) or not SMOKE:
        errors.append("SMOKE must be a dict of workflow arguments (a problem that runs in seconds)")
    if WORKERS_ARG is not None and not isinstance(WORKERS_ARG, str):
        errors.append("WORKERS_ARG must be an argument name or None")
    if not TIERS:
        errors.append("TIERS is empty")
    for label, problem in TIERS.items():
        if label.upper() not in VALID_LABELS:
            errors.append(f"tier label {label!r} must be one of s, m, l, xl")
        if not isinstance(problem, dict):
            errors.append(f"tier {label!r} must be a dict of workflow arguments")
            continue
        for name, fn in (("MEMORY_GB", MEMORY_GB), ("TIME_UNITS", TIME_UNITS)):
            if fn is None:
                continue
            try:
                value = fn(problem)
            except Exception as exc:  # pylint: disable=broad-except
                errors.append(f"{name} failed on tier {label!r}: {exc!r}")
                continue
            if not isinstance(value, (int, float)) or value < 0:
                errors.append(f"{name} must return a non-negative number, got {value!r}")
    if errors:
        sys.exit("scale_probe: fix the edit block:\n  - " + "\n  - ".join(errors))


def _fmt_t(seconds):
    if seconds is None:
        return "?"
    if seconds > 2 * 365 * 86400:
        return ">2 yr"
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 7200:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def _fmt_gb(gb):
    if gb is None:
        return "?"
    if gb > 1e6:
        return ">1 PB"
    return f"{gb:.2g} GB" if gb < 10 else f"{gb:,.0f} GB"


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_profiles(client=None):
    """Fill CPU_PROFILES from the deployment's ``compute_profiles()`` (CPU-only, smallest first)."""
    global CPU_PROFILES  # pylint: disable=global-statement
    if CPU_PROFILES:
        return CPU_PROFILES
    try:
        if client is None:
            from qiskit_ibm_catalog import (
                ManagedWorkflowClient,
            )  # pylint: disable=import-outside-toplevel

            client = ManagedWorkflowClient(name=ACCOUNT)
        catalog = list(client.compute_profiles())
    except Exception as exc:  # pylint: disable=broad-except
        sys.exit(
            "scale_probe: could not list compute profiles with ManagedWorkflowClient()."
            f"compute_profiles() ({exc.__class__.__name__}: {exc}). Machine choices need the "
            "deployment's list: install qiskit-ibm-catalog>=0.22 and save an account (SKILL.md, Step 3)."
        )
    cpu = []
    for prof in catalog:
        cores, mem = _num(getattr(prof, "cpu", None)), _num(getattr(prof, "memory", None))
        if getattr(prof, "gpu", None) or cores is None or mem is None:
            continue
        cpu.append((prof.compute_profile_id, int(cores), int(mem) if mem.is_integer() else mem))
    if not cpu:
        sys.exit("scale_probe: compute_profiles() returned no CPU profiles")
    CPU_PROFILES = sorted(cpu, key=lambda p: (p[2], p[1]))
    return CPU_PROFILES


def _rungs(need_gb, cores):
    """Smallest CPU profile that fits, plus the next one up as a fallback."""
    load_profiles()
    if need_gb <= LAPTOP_GB and cores <= 8:
        return "laptop", CPU_PROFILES[0][0]
    largest = CPU_PROFILES[-1]
    fits = [p for p in CPU_PROFILES if p[2] >= need_gb and p[1] >= min(cores, largest[1])]
    if not fits:
        return f"{largest[0]} (try it)", None
    first = CPU_PROFILES.index(fits[0])
    fallback = CPU_PROFILES[first + 1][0] if first + 1 < len(CPU_PROFILES) else None
    return fits[0][0], fallback


def _fit_memory(runs, notes):
    """Fit peak = overhead + k * formula over the runs; flag a wrong shape.

    Two runs fix the line exactly, so the shape can only be tested with three or more:
    a middle run far off the line means the formula has the wrong exponent or misses
    an object. With two, a negative or zero slope is the only shape signal.
    """
    pts = sorted(
        ((MEMORY_GB(r["problem"]), r["resources"]["peak_gb"]) for r in runs), key=lambda x: x[0]
    )
    if len(pts) < 2 or pts[-1][0] <= pts[0][0] or pts[-1][1] - pts[0][1] < MIN_ADDED_GB:
        form, peak = pts[-1]
        notes.append(
            "UNTESTED SHAPE: the runs don't span enough size to fit how memory grows. Run a "
            "second size ~2-4x larger in formula memory before trusting larger tiers."
        )
        return {"base": 0.0, "k": (peak / form) if form > 0 else 1.0}
    (f_lo, m_lo), (f_hi, m_hi) = pts[0], pts[-1]
    k = (m_hi - m_lo) / (f_hi - f_lo)
    base = m_hi - k * f_hi
    if k <= 0 or base < 0:
        k, base = m_hi / f_hi, 0.0
        notes.append("memory grows faster than the formula: using the largest run's ratio")
    notes.append(f"memory fit over {len(pts)} runs: {_fmt_gb(base)} overhead + {k:.2g} x formula")
    if f_hi < 1.8 * f_lo:
        notes.append("the runs are less than ~2x apart in formula memory: spread them out")
    for form, peak in pts[1:-1]:
        pred = base + k * form
        if not 0.67 < peak / max(pred, 1e-9) < 1.5:
            notes.append(
                f"SHAPE WRONG: a middle run is {peak / pred:.2f}x the fitted line - the formula "
                "has the wrong exponent or misses an object; fix it before trusting larger tiers"
            )
    if len(pts) == 2:
        notes.append("two runs fix the line; a third size would test the formula's shape")
    return {"base": base, "k": k}


def _fit_time(runs, notes):
    if TIME_UNITS is None:
        return None
    pts = sorted(
        (
            (TIME_UNITS(r["problem"]), r["resources"]["wall_s"])
            for r in runs
            if TIME_UNITS(r["problem"]) > 0 and r["resources"].get("wall_s", 0) > 1
        ),
        key=lambda x: x[0],
    )
    if not pts:
        notes.append("no run took over a second: time not fitted")
        return None
    (u1, t1), (u2, t2) = pts[0], pts[-1]
    exp = math.log(t2 / t1) / math.log(u2 / u1) if len(pts) > 1 and u2 > 1.5 * u1 else 1.0
    if len(pts) > 1:
        notes.append(f"time grows as TIME_UNITS^{exp:.2f} between the runs")
    else:
        notes.append("one timed run: tier times assume time grows exactly as TIME_UNITS")
    return (u2, t2, exp)


def size(runs):
    """Fit the measured runs and print a starting machine and fallback for every tier."""
    _validate()
    notes = []
    runs = [r for r in runs if (r.get("resources") or {}).get("peak_gb") is not None]
    if not runs:
        sys.exit("scale_probe: no runs with resources.peak_gb to fit")
    mem, tim = _fit_memory(runs, notes), _fit_time(runs, notes)
    cores_used = max(r["resources"].get("cores_used") or 1 for r in runs)
    granted = max(r["resources"].get("cores_granted") or 1 for r in runs)
    if granted >= 4 and cores_used < 0.5 * granted:
        notes.append(
            f"the runs used about {cores_used:.1f} of {granted} cores: extra cores come "
            "with the memory but do not speed this up"
        )
    rows = []
    for label, problem in TIERS.items():
        formula = MEMORY_GB(problem)
        est = mem["base"] + mem["k"] * formula
        seconds = None
        if tim:
            u_ref, t_ref, exp = tim
            seconds = t_ref * (TIME_UNITS(problem) / u_ref) ** exp
            if seconds > 20 * 3600:
                notes.append(f"tier {label}: ~{_fmt_t(seconds)}, near or past the 24 h job limit")
        first, fallback = _rungs(est * HEADROOM, round(cores_used))
        rows.append(
            {
                "tier": label,
                "problem": problem,
                "formula_gb": formula,
                "estimate_gb": est,
                "time_s": seconds,
                "first": first,
                "fallback": fallback,
            }
        )
    print(f"\n{'tier':<5}{'formula':<11}{'estimate':<11}{'time':<10}{'start on':<18}fallback")
    for row in rows:
        print(
            f"{row['tier']:<5}{_fmt_gb(row['formula_gb']):<11}{_fmt_gb(row['estimate_gb']):<11}"
            f"{_fmt_t(row['time_s']):<10}{row['first']:<18}{row['fallback'] or '-'}"
        )
    print(
        f"\nfrom {len(runs)} measured run(s); machines fit estimate x {HEADROOM}. Expect ~2x "
        "error until a run at a size close to the tier. notes:"
    )
    for note in dict.fromkeys(notes):
        print("  -", note)
    with open(os.path.join(HERE, "scale_probe.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {"tiers": rows, "notes": list(dict.fromkeys(notes))}, handle, indent=1, default=str
        )
    return rows


def _local_problem(label):
    """SMOKE, or a tier small enough for the caps, with the worker count capped."""
    problem = dict(SMOKE if label is None else TIERS[label])
    if label is not None and MEMORY_GB(problem) * HEADROOM > LOCAL_CAPS["gb"]:
        sys.exit(
            f"scale_probe: tier {label!r} needs at least {_fmt_gb(MEMORY_GB(problem) * HEADROOM)} by "
            f"the formula, over the {LOCAL_CAPS['gb']} GB local cap. Run it on Managed Workflows "
            "(mw_sweep.py); --check alone runs SMOKE."
        )
    if WORKERS_ARG and isinstance(problem.get(WORKERS_ARG), int):
        problem[WORKERS_ARG] = max(1, min(problem[WORKERS_ARG], LOCAL_CAPS["workers"]))
    return problem


def _group_gb(pid):
    """Resident memory of a process and all its descendants, or None without psutil."""
    try:
        import psutil  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None
    try:
        root = psutil.Process(pid)
        procs = [root, *root.children(recursive=True)]
    except psutil.Error:
        return 0.0
    total = 0
    for proc in procs:
        try:
            total += proc.memory_info().rss
        except psutil.Error:
            pass
    return total / 1e9


def _kill_group(proc):
    """Kill the child and every process it started (pool workers included)."""
    import signal  # pylint: disable=import-outside-toplevel

    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError):
        pass


def check(label, runs_file):
    """Run SMOKE (or a small tier) once in a capped child process group; append it to runs_file."""
    import subprocess  # pylint: disable=import-outside-toplevel
    import tempfile  # pylint: disable=import-outside-toplevel

    _validate()
    cpus = os.cpu_count() or 2
    if hasattr(os, "getloadavg") and os.getloadavg()[0] > cpus / 2:
        sys.exit(
            f"scale_probe: load average {os.getloadavg()[0]:.1f} on {cpus} CPUs; this machine is "
            "already busy (an earlier run still going?). Not starting a local check."
        )
    problem = _local_problem(label)
    env = {
        **os.environ,
        **{var: "1" for var in THREAD_VARS},
        "MW_LOCAL_WORKERS": str(LOCAL_CAPS["workers"]),
    }
    if _group_gb(os.getpid()) is None:
        print("note: psutil not installed, so the memory cap is not enforced; the time cap is")
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "summary.json")
        cmd = [sys.executable, os.path.abspath(__file__), "_child", json.dumps(problem), out]
        # own process group, so the whole tree (pool workers included) can be killed
        with subprocess.Popen(cmd, env=env, start_new_session=True) as proc:
            start, stopped = time.monotonic(), None
            while proc.poll() is None:
                elapsed = time.monotonic() - start
                gb = _group_gb(proc.pid)
                if elapsed > LOCAL_CAPS["wall_s"]:
                    stopped = f"over {LOCAL_CAPS['wall_s']} s"
                elif gb is not None and gb > LOCAL_CAPS["gb"]:
                    stopped = f"over {LOCAL_CAPS['gb']} GB ({gb:.1f} GB)"
                if stopped:
                    _kill_group(proc)
                    break
                time.sleep(0.5)
            proc.wait()
            _kill_group(proc)  # workers left behind by a child that exited early
        if stopped or not os.path.exists(out):
            why = stopped or f"exit code {proc.returncode}"
            sys.exit(
                f"scale_probe: --check stopped ({why}) on {problem}. Make SMOKE smaller until it "
                "runs in seconds; size larger problems on Managed Workflows (mw_sweep.py)."
            )
        with open(out, encoding="utf-8") as handle:
            summary = json.load(handle)
    print(json.dumps(summary))
    runs = []
    if os.path.exists(runs_file):
        with open(runs_file, encoding="utf-8") as handle:
            runs = json.load(handle)
    runs.append({"problem": problem, "resources": summary, "where": "local"})
    with open(runs_file, "w", encoding="utf-8") as handle:
        json.dump(runs, handle, indent=1)
    print(f"appended to {runs_file}; {len(runs)} run(s) recorded")


def _child(problem_json, out):
    """Inside the capped child: run work() under telemetry and write the summary."""
    if hasattr(os, "nice"):
        os.nice(19)
    sys.path.insert(0, HERE)
    from mw_telemetry import telemetry  # pylint: disable=import-outside-toplevel,import-error

    with telemetry(heartbeat_s=60) as tel:
        with tel.stage("work"):
            work(**json.loads(problem_json))
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(tel.summary, handle)


def estimate_only():
    """Evaluate MEMORY_GB for every tier without running anything."""
    _validate()
    print(f"{'tier':<5}{'formula':<11}{'x1.3':<11}{'start on':<20}fallback   (constant UNMEASURED)")
    for label, problem in TIERS.items():
        formula = MEMORY_GB(problem)
        first, fallback = _rungs(formula * HEADROOM, 1)
        print(
            f"{label:<5}{_fmt_gb(formula):<11}{_fmt_gb(formula * HEADROOM):<11}"
            f"{first:<20}{fallback or '-'}"
        )
    print(
        "LOWER BOUND, NOT A SIZING: the constant is unmeasured, and real copies, workspace and\n"
        "workers often add 2-30x. Measure small runs (Step 3 of SKILL.md) before choosing."
    )


if __name__ == "__main__":
    ARGS = sys.argv[1:]
    if ARGS[:1] == ["--estimate"]:
        estimate_only()
    elif ARGS[:1] == ["_child"]:
        _child(ARGS[1], ARGS[2])
    elif ARGS[:1] == ["--check"]:
        check(ARGS[1] if len(ARGS) > 1 else None, os.path.join(HERE, "runs.json"))
    elif ARGS:
        with open(ARGS[0], encoding="utf-8") as fh:
            size(json.load(fh))
    else:
        print(__doc__)
