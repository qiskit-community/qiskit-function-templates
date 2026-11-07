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

"""Resource telemetry for a function: peak memory, cores used, time per stage.

Copy this file next to the entrypoint. Standard library only; uses ``psutil`` if
installed::

    from mw_telemetry import telemetry

    def run_function(**args):
        with telemetry() as t:
            with t.stage("prepare"):
                problem = build(**args)
            with t.stage("solve"):
                result = solve(problem)
        return {"result": result, "metadata": {"resources": t.summary}}

Peak memory is the job's own container counter on Linux (it includes every
worker process), else the process tree via ``psutil``, else the largest
process. A heartbeat thread prints time and memory every ``heartbeat_s``
seconds, so ``job.logs()`` shows progress even when one step takes hours.

Jobs on Managed Workflows have no outbound network, so the data leaves the job
through channels the service already has:

* ``t.summary`` in the result (``metadata["resources"]``);
* one ``MW_TELEMETRY {...}`` JSON line per event in the logs, read back with
  ``parse_logs(job.logs())``;
* a ``telemetry-<run id>.json`` file in the function's storage with every sample,
  fetched afterwards with ``serverless.file_download(name, function)``.

With ``telemetry(alloc_check={"cap_gb": 4, "max_hits": 3})`` the run becomes an
allocation check, which ``mw_sweep.py --target`` submits as a job: each numpy array-creation call
(``zeros``, ``empty``, ``ones``, ``full`` and their ``_like`` forms) above the cap is
recorded (shape, dtype, size, calling lines) and replaced by a tiny stand-in, and
the run stops after ``max_hits`` of them. It tells you which objects dominate
at research size without allocating them. It is a lower bound: arrays created
elsewhere (einsum, BLAS, C/C++ libraries) are not seen.

From your own machine, ``python mw_telemetry.py replay telemetry-<run id>.json``
sends a downloaded file (or saved logs) to an OpenTelemetry collector over
OTLP/HTTP, using the standard ``OTEL_EXPORTER_OTLP_*`` settings. Live export from
inside a run (set ``OTEL_EXPORTER_OTLP_ENDPOINT``) only works where a collector is
reachable, such as a laptop or an on-premises machine.
"""

import contextlib
import json
import math
import os
import resource
import sys
import threading
import time
import uuid

TAG = "MW_TELEMETRY "
_CGROUP = "/sys/fs/cgroup"
_JOB_STORAGE = "/function_user_data"  # the function's shared storage inside a job


def _read_gb(name):
    try:
        with open(os.path.join(_CGROUP, name), encoding="utf-8") as fh:
            return int(fh.read().split()[0]) / 1e9
    except (OSError, ValueError):
        return None


def _tree_gb():
    try:
        import psutil  # pylint: disable=import-outside-toplevel

        root = psutil.Process()
        total = 0
        for proc in [root, *root.children(recursive=True)]:
            try:
                total += proc.memory_info().rss
            except psutil.Error:
                pass
        return total / 1e9
    except ImportError:
        return None


def current_gb():
    """Memory in use now, whole job if the container counter is available."""
    value = _read_gb("memory.current")
    return value if value is not None else _tree_gb()


def _largest_process_gb():
    unit = 1 if sys.platform == "darwin" else 1024
    usages = (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)
    return max(resource.getrusage(r).ru_maxrss for r in usages) * unit / 1e9


def _cpu_s():
    usages = (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)
    return sum(resource.getrusage(r).ru_utime + resource.getrusage(r).ru_stime for r in usages)


def _cores_granted():
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count()


def _sizing():
    """Size label and compute profile the job was given (Fleets jobs), else None.

    Read from the variables behind ``qiskit_serverless.get_function_size()`` and
    ``get_compute_profile()``, so this also runs locally without qiskit-serverless.
    """
    return {
        "function_size": os.environ.get("ENV_JOB_FUNCTION_SIZE") or None,
        "compute_profile": os.environ.get("ENV_JOB_COMPUTE_PROFILE") or None,
    }


def _fmt(value):
    return round(value, 2) if value is not None else None


class AllocationStop(BaseException):
    """Raised inside the run once the allocation check has seen enough large arrays.

    A ``BaseException`` so that ``except Exception`` in the research code does not catch it.
    """


_ALLOC_CALLS = ("zeros", "empty", "ones", "full", "zeros_like", "empty_like", "ones_like")


class _AllocWatch:
    """Intercept large numpy array creation; record it and hand back a tiny stand-in."""

    def __init__(self, cap_gb, max_hits, on_hit):
        limit = _read_gb("memory.max")  # never let a below-cap array take more than half
        self.cap_gb = min(cap_gb, 0.5 * limit) if limit else cap_gb
        self.max_hits = max_hits
        self.on_hit = on_hit
        self.hits = []
        self._orig = {}
        self._np = None

    @staticmethod
    def _sites():
        frame, out = sys._getframe(1), []  # pylint: disable=protected-access
        while frame and len(out) < 4:
            name = frame.f_code.co_filename
            if "numpy" not in name and name != __file__:
                out.append(f"{os.path.basename(name)}:{frame.f_lineno}")
            frame = frame.f_back
        return out

    def _request(self, name, args, kw):
        np = self._np
        if name.endswith("_like"):
            proto = args[0] if args else kw.get("a")
            shape = kw.get("shape", getattr(proto, "shape", ()))
            dtype = kw.get("dtype", args[1] if len(args) > 1 else getattr(proto, "dtype", float))
        else:
            shape = args[0] if args else kw.get("shape", ())
            pos = 2 if name == "full" else 1
            dtype = kw.get("dtype", args[pos] if len(args) > pos else None)
        shape = tuple(int(d) for d in shape) if isinstance(shape, (tuple, list)) else (int(shape),)
        dtype = np.dtype(dtype if dtype is not None else float)
        return shape, dtype, math.prod(shape) * dtype.itemsize / 1e9

    def _wrap(self, name):
        orig = self._orig[name]

        def alloc(*args, **kw):
            try:
                shape, dtype, gb = self._request(name, args, kw)
            except (TypeError, ValueError):
                return orig(*args, **kw)
            if gb <= self.cap_gb:
                return orig(*args, **kw)
            hit = {"call": f"np.{name}", "shape": list(shape), "dtype": str(dtype)}
            hit.update({"gb": round(gb, 3), "sites": self._sites()})
            self.hits.append(hit)
            self.on_hit(hit, len(self.hits))
            if len(self.hits) >= self.max_hits:
                raise AllocationStop(f"{len(self.hits)} arrays above {self.cap_gb:.3g} GB")
            return self._orig["zeros"]((1,) * max(len(shape), 1), dtype=dtype)

        return alloc

    def install(self):
        """Patch numpy's array-creation calls."""
        import numpy as np  # pylint: disable=import-outside-toplevel

        self._np = np
        for name in _ALLOC_CALLS:
            self._orig[name] = getattr(np, name)
        for name in _ALLOC_CALLS:
            setattr(np, name, self._wrap(name))

    def uninstall(self):
        """Restore numpy."""
        for name, func in self._orig.items():
            setattr(self._np, name, func)


class _LiveOtel:
    """Optional live OTLP export, only when an endpoint is set and reachable."""

    def __init__(self, service):
        self.tracer = None
        self.providers = ()
        if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
            return
        try:
            # pylint: disable=import-outside-toplevel
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError:
            print("mw_telemetry: OTEL endpoint set but opentelemetry-sdk not installed", flush=True)
            return
        provider = TracerProvider(resource=Resource.create({"service.name": service}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        self.providers = (provider,)
        self.tracer = provider.get_tracer("mw_telemetry")

    def span(self, name):
        """A span for a stage, or a no-op when live export is off."""
        if self.tracer is None:
            return contextlib.nullcontext()
        return self.tracer.start_as_current_span(name)

    def flush(self):
        """Send what is buffered; an unreachable collector costs at most a few seconds."""
        for provider in self.providers:
            with contextlib.suppress(Exception):
                provider.force_flush(timeout_millis=5000)
                provider.shutdown()


class Telemetry:  # pylint: disable=too-many-instance-attributes
    """Collects the run's resources; ``summary`` and ``samples`` are JSON-friendly."""

    def __init__(self, heartbeat_s=300, service="managed-workflow", run_id=None, alloc_check=None):
        self.heartbeat_s = heartbeat_s
        self._watch = None
        if alloc_check:
            cap, hits = float(alloc_check.get("cap_gb", 4)), int(alloc_check.get("max_hits", 3))
            self._watch = _AllocWatch(cap, hits, self._on_hit)
        self.service = service
        self.run_id = run_id or os.environ.get("ENV_JOB_ID") or uuid.uuid4().hex[:12]
        self.stages = {}
        self.samples = []
        self.summary = {}
        self.file = None
        self._otel = _LiveOtel(service)
        self._stop = threading.Event()
        self._t0 = self._c0 = self._start_gb = None
        self._tree_peak = 0.0  # sampled process-tree peak, used without a container counter

    def _peak(self):
        container = _read_gb("memory.peak")
        if container is not None:
            return container
        return max(self._tree_peak, _largest_process_gb())

    def _source(self):
        if _read_gb("memory.peak") is not None:
            return "container"
        return "process tree (sampled)" if self._tree_peak else "largest process"

    def log(self, message="", event="note", **fields):
        """Print a readable line and a JSON line, and keep the sample for the file."""
        record = {
            "t": round(time.time(), 2),
            "event": event,
            "message": message,
            "mem_now_gb": _fmt(current_gb()),
            "peak_gb": _fmt(self._peak()),
            **fields,
        }
        self.samples.append(record)
        print(
            f"[{time.strftime('%H:%M:%S')}] {message} mem_now_gb={record['mem_now_gb']} "
            f"peak_gb={record['peak_gb']}",
            flush=True,
        )
        print(TAG + json.dumps(record), flush=True)

    @contextlib.contextmanager
    def stage(self, name):
        """Time a stage and record the peak memory reached by its end."""
        self.log(f"stage {name} start", event="stage_start", stage=name)
        start = time.perf_counter()
        with self._otel.span(name):
            try:
                yield
            finally:
                seconds = round(time.perf_counter() - start, 1)
                self.stages[name] = {"s": seconds, "peak_gb_at_end": _fmt(self._peak())}
                self.log(
                    f"stage {name} done in {seconds} s", event="stage_end", stage=name, s=seconds
                )

    def _on_hit(self, hit, count):
        self.log(
            f"large allocation #{count}: {hit['call']}{tuple(hit['shape'])} {hit['dtype']} = "
            f"{hit['gb']:,.1f} GB at {' < '.join(hit['sites'])}",
            event="large_allocation",
            **hit,
        )

    def _beat(self):
        last = time.perf_counter()
        sample = _read_gb("memory.peak") is None  # sample the tree only without a container counter
        while not self._stop.wait(0.2 if sample else self.heartbeat_s):
            if sample:
                self._tree_peak = max(self._tree_peak, _tree_gb() or 0.0)
            if time.perf_counter() - last >= self.heartbeat_s:
                self.log("heartbeat", event="heartbeat")
                last = time.perf_counter()

    def _write_file(self):
        """Write every sample to the function's storage (or the working directory)."""
        safe = "".join(c if c.isalnum() or c in "_.-" else "_" for c in self.run_id)
        name = f"telemetry-{safe}.json"
        payload = {
            "service": self.service,
            "run_id": self.run_id,
            "summary": self.summary,
            "samples": self.samples,
        }
        for folder in (_JOB_STORAGE, os.getcwd()):
            if not os.path.isdir(folder) or not os.access(folder, os.W_OK):
                continue
            path = os.path.join(folder, name)
            try:
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh)
                return path
            except OSError:
                continue
        return None

    def __enter__(self):
        self._t0, self._c0 = time.perf_counter(), _cpu_s()
        self._start_gb = current_gb()
        threading.Thread(target=self._beat, daemon=True).start()
        sizing = _sizing()
        self.log(
            f"start, cores granted {_cores_granted()}, "
            f"profile {sizing['compute_profile'] or 'unknown'}",
            event="start",
            cores_granted=_cores_granted(),
            **sizing,
        )
        if self._watch:
            self._watch.install()
            self.log(f"allocation check on, cap {self._watch.cap_gb:.3g} GB", event="alloc_check")
        return self

    def __exit__(self, *exc):
        self._stop.set()
        suppress = False
        if self._watch:
            self._watch.uninstall()
            if exc[0] is None:
                outcome = "finished"
            elif isinstance(exc[1], AllocationStop):
                outcome, suppress = f"stopped: {exc[1]}", True
            else:  # a stand-in array broke the code further on; the hits so far still count
                outcome = f"error after {len(self._watch.hits)} stand-in(s): {exc[0].__name__}"
                suppress = bool(self._watch.hits)
        wall = time.perf_counter() - self._t0
        self.summary = {
            "peak_gb": _fmt(self._peak()),
            "start_gb": _fmt(self._start_gb),
            "wall_s": round(wall, 1),
            "cores_used": round((_cpu_s() - self._c0) / wall, 1) if wall else None,
            "cores_granted": _cores_granted(),
            **_sizing(),
            "stages": self.stages,
            "source": self._source(),
        }
        if self._watch:
            self.summary["alloc_check"] = {
                "cap_gb": round(self._watch.cap_gb, 3),
                "outcome": outcome,
                "hits": self._watch.hits,
            }
        self.log("end", event="end", summary=self.summary)
        path = self._write_file()
        if path:
            self.file = os.path.basename(path)
            self.summary["telemetry_file"] = self.file
        self._otel.flush()
        return suppress


def telemetry(heartbeat_s=300, service="managed-workflow", run_id=None, alloc_check=None):
    """Context manager wrapping a run; see the module docstring."""
    return Telemetry(heartbeat_s, service, run_id, alloc_check)


def parse_logs(text):
    """Records from the ``MW_TELEMETRY`` lines in a job's logs, in order."""
    out = []
    for line in text.splitlines():
        idx = line.find(TAG)
        if idx >= 0:
            with contextlib.suppress(ValueError):
                out.append(json.loads(line[idx + len(TAG) :]))
    return out


def replay(source, service=None):
    """Send a telemetry file (or saved logs) to an OTLP collector from this machine.

    Each run becomes one trace: a root span for the run and one span per stage, with
    the memory samples as span events. Uses the standard ``OTEL_EXPORTER_OTLP_*``
    environment variables and needs ``opentelemetry-sdk`` and the OTLP HTTP exporter.
    """
    # pylint: disable=import-outside-toplevel
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    with open(source, encoding="utf-8") as fh:
        text = fh.read()
    try:
        data = json.loads(text)
        samples, run_id = data["samples"], data.get("run_id", "run")
        service = service or data.get("service")
    except (ValueError, KeyError, TypeError):
        samples, run_id = parse_logs(text), os.path.basename(source)
    if not samples:
        raise SystemExit(f"replay: no telemetry samples in {source}")
    provider = TracerProvider(
        resource=Resource.create({"service.name": service or "managed-workflow"})
    )
    provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
    tracer = provider.get_tracer("mw_telemetry.replay")
    ns = lambda t: int(t * 1e9)  # noqa: E731
    root = tracer.start_span(f"run {run_id}", start_time=ns(samples[0]["t"]))
    from opentelemetry import trace  # pylint: disable=import-outside-toplevel

    ctx = trace.set_span_in_context(root)
    open_stages = {}
    for rec in samples:
        attrs = {k: v for k, v in rec.items() if isinstance(v, (int, float, str)) and k != "t"}
        root.add_event(rec.get("event", "sample"), attributes=attrs, timestamp=ns(rec["t"]))
        if rec.get("event") == "stage_start":
            open_stages[rec["stage"]] = tracer.start_span(
                rec["stage"], context=ctx, start_time=ns(rec["t"])
            )
        elif rec.get("event") == "stage_end" and rec.get("stage") in open_stages:
            span = open_stages.pop(rec["stage"])
            span.set_attribute("peak_gb", rec.get("peak_gb") or 0.0)
            span.end(end_time=ns(rec["t"]))
        elif rec.get("event") == "end":
            for key, value in (rec.get("summary") or {}).items():
                if isinstance(value, (int, float, str)):
                    root.set_attribute(f"resources.{key}", value)
    root.end(end_time=ns(samples[-1]["t"]))
    provider.force_flush(timeout_millis=10000)
    provider.shutdown()
    return len(samples)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "replay":
        print(f"replayed {replay(sys.argv[2])} samples from {sys.argv[2]}")
    else:
        print(__doc__)
