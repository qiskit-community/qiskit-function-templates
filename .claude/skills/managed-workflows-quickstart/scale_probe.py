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

"""Check a memory formula against a local run and pick a first machine per tier.

Copy this file next to the research code, edit the block below, and run it with
the workflow's own Python environment (in the background if it takes a while)::

    python scale_probe.py > probe.log 2>&1 &

If the code cannot run locally, ``python scale_probe.py --estimate`` evaluates the
formula for every tier without running anything.

It runs the small problems in ``LOCAL_CHECKS`` for real, measuring memory and
CPU over the whole process tree (worker pools and compiled solvers count). It
fits measured memory as overhead + k x ``MEMORY_GB`` (one check gives k; two
give the overhead too and test how memory grows), applies that to every tier,
and suggests a starting profile plus a fallback. The formula supplies the
shape, the checks supply the constant, and the cloud runs are the real
measurement. Standard library only
(uses ``psutil`` if installed). Never contacts a QPU.
"""

# pylint: disable=too-many-lines,not-callable

import json
import math
import os
import resource
import subprocess
import sys
import threading
import time
import types

# ============================ EDIT THIS BLOCK ============================


def work(cores, **problem):  # pylint: disable=unused-argument
    """EDIT: run THEIR workflow (QPU mocked) on one problem - import and call the
    researcher's own functions; never write a stand-in that imitates them.

    Make it a generator so one-time setup is not measured::

        from my_workflow import build, solve   # setup
        h = build(**problem)
        yield                                  # everything after this is measured
        solve(h, n_workers=cores)
    """
    from my_workflow import run  # pylint: disable=import-outside-toplevel,import-error

    yield
    run(**problem)


TIERS = {  # labels must be s, m, l or xl; the problems the researcher wants to run
    "s": {"n": 4000, "n_vectors": 20, "workers": 1},
    "m": {"n": 20000, "n_vectors": 20, "workers": 4},
    "l": {"n": 60000, "n_vectors": 40, "workers": 8},
}


def MEMORY_GB(p):  # pylint: disable=invalid-name
    """EDIT: rough GB of the biggest object(s) for problem p - the shape, not the constant.

    Count what is alive at the peak and what multiplies it (sizing_guide.md). The
    check runs measure the constant, so leave copies and workspace rough.
    """
    # e.g. each worker holds a dense n x n matrix plus n_vectors work vectors
    return p["workers"] * 8 * (p["n"] ** 2 + p["n_vectors"] * p["n"]) / 1e9


TIME_UNITS = (
    None  # optional: lambda p: relative amount of work, e.g. p["max_iterations"] * 4 ** p["n"]
)
LOCAL_CHECKS = ["s"]  # 2 sizes, ~2-4x apart, each < a few minutes: tier labels or problem dicts
CORES = [1, 2, 4, 8]  # 1 is always included; values above this machine's cores are skipped
TIMEOUT_S = 300  # per run

# ========================= machinery below this line =========================

VALID_LABELS = ("S", "M", "L", "XL")
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
)
GPU_LIBS = (
    "/numpy/linalg/",
    "/scipy/linalg/",
    "einsumfunc",
    "/pyscf/scf/",
    "/pyscf/dft/",
    "/pyscf/df/",
    "/quimb/",
    "/cotengra/",
    "/opt_einsum/",
    "/cupy/",
    "/torch/",
)
AER = "/qiskit_aer/"
MARK = "@@SCALE_PROBE@@"
HERE = os.path.dirname(os.path.abspath(__file__))

try:
    import psutil  # pylint: disable=import-error
except ImportError:  # pragma: no cover - depends on the environment
    psutil = None


def _tree_usage(root):
    """Return (rss_bytes, cpu_seconds) summed over `root` and all descendants."""
    if psutil is not None:
        try:
            procs = [psutil.Process(root)]
            procs += procs[0].children(recursive=True)
        except psutil.Error:
            return None
        rss = cpu = 0.0
        for proc in procs:
            try:
                rss += proc.memory_info().rss
                times = proc.cpu_times()
                cpu += times.user + times.system
            except psutil.Error:
                pass
        return rss, cpu
    try:
        out = subprocess.run(
            ["ps", "-A", "-o", "pid=,ppid=,rss=,time="],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
    except OSError:
        return None
    rows, kids = {}, {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        pid, ppid = int(parts[0]), int(parts[1])
        rows[pid] = (int(parts[2]) * 1024, _ps_time(parts[3]))
        kids.setdefault(ppid, []).append(pid)
    if root not in rows:
        return None
    rss = cpu = 0.0
    stack = [root]
    while stack:
        pid = stack.pop()
        if pid in rows:
            rss += rows[pid][0]
            cpu += rows[pid][1]
        stack.extend(kids.get(pid, ()))
    return rss, cpu


def _ps_time(text):
    """Parse ps cumulative CPU time ([dd-]hh:mm:ss[.ff] or mm:ss.ff) to seconds."""
    days = 0
    if "-" in text:
        day, text = text.split("-", 1)
        days = int(day)
    total = 0.0
    for part in text.split(":"):
        total = total * 60 + float(part)
    return days * 86400 + total


class _Sampler(threading.Thread):
    """Poll a child's process tree; track peak RSS and CPU time per phase."""

    def __init__(self, pid, every=0.01):
        super().__init__(daemon=True)
        self.pid, self.every = pid, every
        self.stop_flag = threading.Event()
        self.lock = threading.Lock()
        self.phase = "setup"
        self.floor = 0.0
        self.peak = 0.0
        self.cpu_start = self.cpu_last = 0.0
        self.t_start = time.perf_counter()

    def mark_work(self):
        """Child reached the `yield`: reset the floor and CPU baseline."""
        with self.lock:
            self.phase = "work"
            self.floor = self.peak
            self.cpu_start = self.cpu_last
            self.t_start = time.perf_counter()

    def run(self):
        while not self.stop_flag.wait(self.every):
            usage = _tree_usage(self.pid)
            if usage is None:
                continue
            with self.lock:
                self.peak = max(self.peak, usage[0])
                self.cpu_last = max(self.cpu_last, usage[1])

    def result(self):
        """Peak RSS (bytes), RSS added after setup, and CPU and wall seconds of the work phase."""
        with self.lock:
            wall = time.perf_counter() - self.t_start
            return {
                "peak": self.peak,
                "added": max(self.peak - self.floor, 0.0),
                "cpu": max(self.cpu_last - self.cpu_start, 0.0),
                "wall": wall,
                "had_setup": self.phase == "work",
            }


def _gpu_sampler(stop, hits, every=0.002):
    """Count, per tick, whether any Python thread is inside a GPU-portable library."""
    me = threading.get_ident()
    while not stop.wait(every):
        frames = sys._current_frames()  # pylint: disable=protected-access
        portable = aer = busy = False
        for ident, frame in frames.items():
            if ident == me:
                continue
            files = []
            while frame is not None:
                files.append(frame.f_code.co_filename)
                frame = frame.f_back
            if any(lib in path for path in files for lib in GPU_LIBS):
                portable = True
            if any(AER in path for path in files):
                aer = True
            busy = busy or bool(files)
        hits["ticks"] += 1
        hits["portable"] += portable
        hits["aer"] += aer
        hits["busy"] += busy


def _maxrss():
    """Exact peak RSS (bytes) of this process and of its largest reaped child."""
    unit = 1 if sys.platform == "darwin" else 1024  # macOS reports bytes, Linux KiB
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit
    kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * unit
    return own, kids


def _child(problem, cores):
    """Run `work` once in this process and report timings; parent measures memory."""
    hits = {"ticks": 0, "portable": 0, "aer": 0, "busy": 0}
    stop = threading.Event()
    result = work(cores, **problem)
    generator = isinstance(result, types.GeneratorType)
    if generator:
        next(result, None)  # run setup up to the yield
    own0, _ = _maxrss()
    print(MARK + "work", flush=True)
    sampler = threading.Thread(target=_gpu_sampler, args=(stop, hits), daemon=True)
    sampler.start()
    start = time.perf_counter()
    try:
        if generator:
            for _ in result:
                pass
    finally:
        stop.set()
        sampler.join()
    seconds = time.perf_counter() - start
    own1, kids = _maxrss()
    ticks = max(hits["ticks"], 1)
    report = {
        "work_s": seconds,
        "rss_setup": own0,
        "rss_peak": own1,
        "rss_kids": kids,
        "generator": generator,
        "gpu_share": round(hits["portable"] / ticks, 2) if hits["ticks"] >= 10 else None,
        "aer_share": round(hits["aer"] / ticks, 2) if hits["ticks"] >= 10 else None,
    }
    print(MARK + "done" + json.dumps(report), flush=True)


def _run_once(problem, cores=1):
    """Run one problem in a fresh process; return measurements or {'error': ...}."""
    env = dict(os.environ, MW_PROFILE_CORES=str(cores))
    env.update({var: str(cores) for var in THREAD_VARS})
    cmd = [sys.executable, os.path.abspath(__file__), "--child", json.dumps(problem), str(cores)]
    with subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as proc:
        sampler = _Sampler(proc.pid)
        sampler.start()
        report, stdout_tail, deadline = None, [], time.monotonic() + TIMEOUT_S
        err_thread_out = []
        err_thread = threading.Thread(
            target=lambda: err_thread_out.append(proc.stderr.read()), daemon=True
        )
        err_thread.start()
        timed_out = False
        for line in proc.stdout:
            if line.startswith(MARK + "work"):
                sampler.mark_work()
            elif line.startswith(MARK + "done"):
                report = json.loads(line[len(MARK) + 4 :])
            else:
                stdout_tail = (stdout_tail + [line.rstrip()])[-3:]
            if time.monotonic() > deadline:
                timed_out = True
                break
        if timed_out:
            proc.kill()
        try:
            proc.wait(timeout=max(1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            proc.wait()
        sampler.stop_flag.set()
        sampler.join()
        err_thread.join(timeout=2)
    usage = sampler.result()
    stderr = (err_thread_out[0] if err_thread_out else "") or ""
    if timed_out:
        return {"error": f"timeout > {TIMEOUT_S}s", "kind": "timeout"}
    if proc.returncode != 0 or report is None:
        last = (stderr.strip().splitlines() or stdout_tail or ["no output"])[-1]
        if proc.returncode in (-9, 137) or "MemoryError" in stderr:
            return {"error": "killed (likely out of memory)", "kind": "oom", "stderr": stderr}
        return {"error": last[:200], "kind": "crash", "stderr": stderr}
    # rusage is exact for the probed process; the sampler adds what its descendants
    # (worker pools, compiled solvers) held, which rusage reports only for the largest
    # reaped child. The sampler's own reading of the main process is noisy (it can
    # land before or after a short peak), so it only adds the children's share.
    own_added = max(report["rss_peak"] - report["rss_setup"], 0)
    kids_seen = max(usage["peak"] - report["rss_peak"], 0)
    kids = max(kids_seen, report["rss_kids"])
    report.update(
        peak_gb=(report["rss_peak"] + kids) / 1024**3,
        added_gb=(own_added + kids) / 1024**3,
        cores_used=round(usage["cpu"] / usage["wall"], 1) if usage["wall"] > 0.05 else None,
    )
    return report


def _min_of(problem, cores=1, repeats=1):
    """Fastest of `repeats` runs (errors returned immediately)."""
    best = None
    for _ in range(max(1, repeats)):
        run = _run_once(problem, cores)
        if "error" in run:
            return run
        if best is None or run["work_s"] < best["work_s"]:  # pylint: disable=unsubscriptable-object
            best = run
    return best


CPU_PROFILES = [("8x32", 8, 32), ("16x128", 16, 128), ("48x768", 48, 768)]
HEADROOM = 1.3
LAPTOP_GB = 12  # estimate x headroom at or below this fits a typical laptop
MIN_FORMULA_GB = 0.05  # a check adding less than this after setup measures only overhead


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
    if not LOCAL_CHECKS:
        errors.append("LOCAL_CHECKS is empty: run at least the smallest tier locally")
    for check in LOCAL_CHECKS:
        if isinstance(check, str) and check not in TIERS:
            errors.append(f"LOCAL_CHECKS entry {check!r} is not a tier label")
        elif not isinstance(check, (str, dict)):
            errors.append(f"LOCAL_CHECKS entries must be tier labels or dicts, got {check!r}")
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


def _cores_sweep(problem, notes):
    """Measured speedup by core count on one problem; returns (helpful cores, speedups)."""
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    counts = sorted({1, *[c for c in CORES if c <= available]})
    if len(counts) == 1:
        return 1, {}  # nothing to compare; skip re-running the workload
    times = {}
    for cores in counts:
        run = _min_of(problem, cores, repeats=2)
        if "error" in run:
            notes.append(f"core sweep at {cores} cores failed: {run['error']}")
            continue
        times[cores] = run
    if 1 not in times:
        return 1, {}
    if times[1]["work_s"] < 1:
        notes.append("core sweep skipped: the work phase takes < 1 s, too short to time")
        return 1, {}
    speedup = {c: round(times[1]["work_s"] / r["work_s"], 2) for c, r in times.items()}
    helpful = max(c for c, s in speedup.items() if s / c >= 0.6)
    notes.append(f"speedup by cores (measured locally): {speedup}")
    return helpful, speedup


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


def _check_runs(notes):
    """Run LOCAL_CHECKS; return [(name, problem, run)] in order."""
    out = []
    for check in LOCAL_CHECKS:
        name, problem = (check, TIERS[check]) if isinstance(check, str) else ("check", check)
        run = _min_of(problem)
        out.append((name, problem, run))
        if "error" in run:
            notes.append(f"local check {name} failed: {run['error']}")
            print(run.get("stderr") or run["error"], file=sys.stderr)
    return out


def _calibrate(checks, notes):  # pylint: disable=too-many-locals
    """Fit measured memory = overhead + k * formula, and time = t0 * (units / u0) ** a.

    One usable check gives k only (overhead = setup memory); two or more also give
    the overhead and the time exponent; three or more test the formula's shape.
    """
    good = [(n, p, r) for n, p, r in checks if "error" not in r]
    if not good:
        return None
    good.sort(key=lambda c: MEMORY_GB(c[1]))
    if not good[-1][2].get("generator"):
        notes.append("`work` is not a generator: setup (imports, preparation) is counted as work")
    setup = max(r["peak_gb"] - r["added_gb"] for _, _, r in good)
    usable = [
        (MEMORY_GB(p), r["added_gb"], n)
        for n, p, r in good
        if MEMORY_GB(p) > 0 and r["added_gb"] >= MIN_FORMULA_GB
    ]
    model = {"k": 1.0, "overhead": setup, "time": None, "power": None}
    if not usable:
        notes.append(
            f"every check added less than {_fmt_gb(MIN_FORMULA_GB)} after setup: the runs only "
            "measure imports. Add a larger LOCAL_CHECKS problem (still < a few minutes) to "
            "test the "
            "formula. Using the formula as is."
        )
    elif len(usable) == 1:
        form, m, name = usable[0]
        model["k"] = max(m, 1e-6) / form
        notes.append(
            f"memory check on {name}: formula {_fmt_gb(form)}, measured {_fmt_gb(m)} -> ratio "
            f"{model['k']:.2g}. UNTESTED SHAPE: one check gives the constant, not how memory "
            "grows. Add a second check ~2-4x larger in formula memory before trusting the tiers."
        )
    else:
        (f1, m1, _), (f2, m2, _) = usable[0], usable[-1]
        k = (m2 - m1) / (f2 - f1) if f2 > f1 else m2 / f2
        c = m2 - k * f2
        if k <= 0 or c < -0.5 * m2:
            notes.append(
                "memory grows faster than the formula between the checks: the formula probably "
                "misses the dominant object or its exponent. Using the largest check's ratio."
            )
            k, c = m2 / f2, 0.0
        model["k"], model["overhead"] = k, setup + max(c, 0.0)
        notes.append(
            f"memory fit over {len(usable)} checks: {_fmt_gb(max(c, 0.0))} overhead + "
            f"{k:.2g} x formula (ratios {m1 / f1:.2g} -> {m2 / f2:.2g})"
        )
        if f2 < 1.8 * f1:
            notes.append(
                "the checks are less than ~2x apart in formula memory: too close to test the "
                "shape; spread them further"
            )
        drift = (m2 / f2) / (m1 / f1)
        if 0.67 <= drift < 0.85:
            notes.append(
                "the ratio falls between the checks: fixed costs are still a large share of "
                "them, so estimates for bigger tiers are probably HIGH - fine to try a smaller "
                "profile first and let the first real run decide"
            )
        if not 0.67 < drift < 1.5 and m1 > 0 and f2 > f1:
            alpha = math.log(m2 / m1) / math.log(f2 / f1)
            model["power"] = (f2, m2, alpha)
            notes.append(
                f"SHAPE WRONG: the ratio drifts {drift:.2f}x between the checks, so memory grows "
                f"as formula^{alpha:.2f}, not formula^1. Tier estimates below use that measured "
                "power instead of the formula's; fix the formula (a wrong exponent, or a factor "
                "that grows with size) and rerun - an empirical power from two points is rough."
            )
        for form, m, name in usable[1:-1]:
            pred = max(c, 0.0) + k * form
            if not 0.67 < m / max(pred, 1e-9) < 1.5:
                notes.append(f"check {name} is off the fit by {m / pred:.2f}x: recheck the formula")
    if not 1 / 3 <= model["k"] <= 3:
        notes.append(
            f"the formula is off by {max(model['k'], 1 / model['k']):.1f}x: fine if the shape is "
            "right (the fit corrects it), but check it does not miss a copy or a worker count"
        )
    if TIME_UNITS is not None:
        pts = [(TIME_UNITS(p), r["work_s"]) for _, p, r in good if TIME_UNITS(p) > 0]
        timed = [pt for pt in pts if pt[1] > 0.5]  # shorter runs are mostly noise
        if len(timed) < len(pts):
            notes.append(
                f"{len(pts) - len(timed)} check(s) ran < 0.5 s, too short to time: make the "
                "checks bigger for a time fit (memory and time both need checks that take a while)"
            )
        pts = timed
        if len(pts) == 1:
            notes.append(
                "only one check was long enough to time: tier times assume time grows exactly "
                "as TIME_UNITS, which is a guess"
            )
        if pts:
            (u1, t1), (u2, t2) = pts[0], pts[-1]
            exp = 1.0
            if len(pts) > 1 and u2 > 1.5 * u1:
                exp = math.log(t2 / t1) / math.log(u2 / u1)
                notes.append(f"time grows as TIME_UNITS^{exp:.2f} between the checks")
                if not 0.8 < exp < 1.2:
                    notes.append(
                        "time does not grow in proportion to TIME_UNITS: either TIME_UNITS misses "
                        "a factor, or fixed costs still dominate the checks. Using the fit, which "
                        "is the less certain the further the tiers are from the checks."
                    )
            model["time"] = (u2, t2, exp)
    return model


def _estimate_only():
    """Evaluate MEMORY_GB for every tier without running anything (code cannot run locally)."""
    _validate()
    print(f"{'tier':<5}{'formula':<11}{'x1.3':<11}{'start on':<20}fallback   (constant UNMEASURED)")
    for label, problem in TIERS.items():
        formula = MEMORY_GB(problem)
        first, fallback = _rungs(formula * HEADROOM, 1)
        print(
            f"{label:<5}{_fmt_gb(formula):<11}{_fmt_gb(formula * HEADROOM):<11}"
            f"{first:<20}{fallback or '-'}"
        )
    print("the constant is a guess until a check runs; real copies/workspace often add 2-30x")


def main():  # pylint: disable=too-many-locals
    """Run the local checks, scale the formula, and print a table with rungs."""
    _validate()
    notes = []
    checks = _check_runs(notes)
    model = _calibrate(checks, notes)
    if model is None:
        print("scale_probe: every local check failed; see the errors above.", file=sys.stderr)
        sys.exit(1)
    biggest = max((c for c in checks if "error" not in c[2]), key=lambda c: MEMORY_GB(c[1]))
    helpful, speedup = _cores_sweep(biggest[1], notes)
    gpu = biggest[2].get("gpu_share")
    aer = biggest[2].get("aer_share")
    if biggest[2].get("cores_used", 0) and biggest[2]["cores_used"] > 1.5 and not gpu:
        notes.append("GPU share n/a: compute appears to run outside Python (subprocess or C/C++)")
    if aer:
        notes.append(f"{aer:.0%} of time in qiskit_aer: GPU helps only with method='statevector'")
    measured = {n: r for n, _, r in checks if isinstance(n, str) and n in TIERS}

    rows = []
    for label, problem in TIERS.items():
        formula = MEMORY_GB(problem)
        if model["power"] is not None and formula > 0:
            f_ref, m_ref, alpha = model["power"]
            est = model["overhead"] + m_ref * (formula / f_ref) ** alpha
        else:
            est = model["overhead"] + model["k"] * formula
        row = {"tier": label, "problem": problem, "formula_gb": formula, "estimate_gb": est}
        run = measured.get(label)
        if run is not None and "error" in run:
            row["how"] = f"failed locally: {run['error']}"
        elif run is not None:
            row.update(estimate_gb=run["peak_gb"], how="measured")
        else:
            row["how"] = (
                f"measured power {model['power'][2]:.2f} of formula (shape wrong)"
                if model["power"] is not None
                else f"formula x {model['k']:.2g} + {_fmt_gb(model['overhead'])}"
            )
        if model["time"] is not None:
            u_ref, t_ref, exp = model["time"]
            seconds = t_ref * (TIME_UNITS(problem) / u_ref) ** exp / speedup.get(helpful, 1)
            row["time_s"] = run["work_s"] if run and "error" not in run else seconds
            if row["time_s"] > 20 * 3600:
                notes.append(
                    f"tier {label}: ~{_fmt_t(row['time_s'])}, near or past the 24 h job limit"
                )
        row["first"], row["fallback"] = _rungs(row["estimate_gb"] * HEADROOM, helpful)
        rows.append(row)

    print(
        f"\n{'tier':<5}{'formula':<11}{'estimate':<11}{'time':<10}"
        f"{'start on':<18}{'fallback':<10}how"
    )
    for row in rows:
        print(
            f"{row['tier']:<5}{_fmt_gb(row['formula_gb']):<11}{_fmt_gb(row['estimate_gb']):<11}"
            f"{_fmt_t(row.get('time_s')):<10}{row['first']:<18}"
            f"{row['fallback'] or '-':<10}{row['how']}"
        )
    for row in rows:
        if row["first"].endswith("(try it)"):
            budget = [(name, gb / HEADROOM) for name, _, gb in CPU_PROFILES]
            notes.append(
                f"tier {row['tier']} estimates above the largest profile - but the estimate may "
                "be high, so start it on 48x768 and see. To climb toward it, each profile fits "
                "a formula value up to about "
                + ", ".join(
                    f"{name}: {_fmt_gb(max(gb - model['overhead'], 0) / model['k'])}"
                    for name, gb in budget
                )
                + " (in MEMORY_GB units) - pick the problem sizes for each rung from that"
            )
    print(f"\nrungs fit estimate x {HEADROOM} and {helpful} helpful core(s). Problems:")
    for row in rows:
        print(f"  {row['tier']}: {row['problem']}")
    if gpu is not None:
        notes.append(f"share of time in GPU-portable libraries (local check): {gpu:.2f}")
    print("notes:")
    for note in dict.fromkeys(notes):
        print("  -", note)
    with open(os.path.join(HERE, "scale_probe.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "tiers": rows,
                "helpful_cores": helpful,
                "speedup": speedup,
                "notes": list(dict.fromkeys(notes)),
            },
            handle,
            indent=1,
            default=str,
        )


if __name__ == "__main__":  # guard required: worker pools re-import this file
    if len(sys.argv) > 1 and sys.argv[1] == "--estimate":
        _estimate_only()
    elif len(sys.argv) > 1 and sys.argv[1] == "--child":
        _child(json.loads(sys.argv[2]), int(sys.argv[3]))
    else:
        main()
