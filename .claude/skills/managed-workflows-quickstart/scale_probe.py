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
    python scale_probe.py --check s         # run tier s once locally: does it run, and how big
    python scale_probe.py runs.json         # fit measured runs and size every tier

``runs.json`` is a list of ``{"problem": {...}, "resources": {...}}`` entries, where
``resources`` is the ``metadata["resources"]`` that ``mw_telemetry`` returns. The
runs normally come from small jobs on Managed Workflows (Step 3 of SKILL.md), so
they are measured on the same machines, libraries and memory accounting as the
real runs; ``--check`` appends its local run to the same file.

The fit is ``peak = overhead + k * MEMORY_GB(problem)`` for memory and
``time = t0 * (TIME_UNITS / u0) ** a`` for time. The formula gives the shape, the
runs give the constants, and two or more runs at different sizes test the shape.
Standard library only (``mw_telemetry`` uses ``psutil`` if installed).
"""

import json
import math
import os
import sys

# ============================ EDIT THIS BLOCK ============================


def work(**problem):
    """EDIT: run THEIR workflow (QPU mocked) on one problem, for --check.

    Import and call the researcher's own functions; never a stand-in that imitates
    them. Use the same entrypoint code the cloud jobs run, if there is one.
    """
    sys.path.insert(0, HERE)  # EDIT: folder that holds their modules
    from my_workflow import run  # pylint: disable=import-outside-toplevel,import-error

    return run(**problem)


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
CPU_PROFILES = [("8x32", 8, 32), ("16x128", 16, 128), ("48x768", 48, 768)]
HEADROOM = 1.3
LAPTOP_GB = 12  # estimate x headroom at or below this fits a typical laptop
MIN_ADDED_GB = 0.05  # a run adding less than this over the overhead measures only imports


def _validate():
    """Fail fast with a clear message on a malformed edit block."""
    errors = []
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


def _rungs(need_gb, cores):
    """Smallest CPU profile that fits, plus the next one up as a fallback."""
    if need_gb <= LAPTOP_GB and cores <= 8:
        return "laptop", CPU_PROFILES[0][0]
    fits = [p for p in CPU_PROFILES if p[2] >= need_gb and p[1] >= min(cores, 48)]
    if not fits:
        return "48x768 (try it)", None
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


def check(label, runs_file):
    """Run one tier locally under telemetry, print it, and append it to runs_file."""
    _validate()
    sys.path.insert(0, HERE)
    from mw_telemetry import telemetry  # pylint: disable=import-outside-toplevel,import-error

    problem = TIERS[label]
    with telemetry(heartbeat_s=60) as tel:
        with tel.stage("work"):
            work(**problem)
    print(json.dumps(tel.summary))
    runs = []
    if os.path.exists(runs_file):
        with open(runs_file, encoding="utf-8") as handle:
            runs = json.load(handle)
    runs.append({"problem": problem, "resources": tel.summary, "where": "local"})
    with open(runs_file, "w", encoding="utf-8") as handle:
        json.dump(runs, handle, indent=1)
    print(f"appended to {runs_file}; {len(runs)} run(s) recorded")


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
    elif ARGS[:1] == ["--check"]:
        check(ARGS[1] if len(ARGS) > 1 else next(iter(TIERS)), os.path.join(HERE, "runs.json"))
    elif ARGS:
        with open(ARGS[0], encoding="utf-8") as fh:
            size(json.load(fh))
    else:
        print(__doc__)
