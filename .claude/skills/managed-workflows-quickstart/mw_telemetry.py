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
installed and OpenTelemetry if ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set::

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
"""

import contextlib
import os
import resource
import sys
import threading
import time

_CGROUP = "/sys/fs/cgroup"


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


def peak_gb():
    """Peak memory so far: container counter, else largest process (lower bound)."""
    value = _read_gb("memory.peak")
    if value is not None:
        return value
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


class _Otel:
    """Optional OpenTelemetry export, only if an OTLP endpoint is configured."""

    def __init__(self, service):
        self.tracer = self.gauge = None
        if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
            return
        try:
            # pylint: disable=import-outside-toplevel
            from opentelemetry import metrics, trace
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError:
            print("mw_telemetry: OTEL endpoint set but opentelemetry-sdk not installed", flush=True)
            return
        res = Resource.create({"service.name": service})
        tracer_provider = TracerProvider(resource=res)
        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(tracer_provider)
        reader = PeriodicExportingMetricReader(OTLPMetricExporter(), export_interval_millis=30_000)
        meter_provider = MeterProvider(resource=res, metric_readers=[reader])
        metrics.set_meter_provider(meter_provider)
        self.providers = (tracer_provider, meter_provider)
        self.tracer = trace.get_tracer("mw_telemetry")
        meter = metrics.get_meter("mw_telemetry")
        meter.create_observable_gauge(
            "job.memory.current",
            unit="GB",
            callbacks=[lambda _opts: [metrics.Observation(current_gb() or 0.0)]],
        )
        meter.create_observable_gauge(
            "job.memory.peak",
            unit="GB",
            callbacks=[lambda _opts: [metrics.Observation(peak_gb() or 0.0)]],
        )

    def span(self, name):
        """An OTEL span for a stage, or a no-op when export is off."""
        if self.tracer is None:
            return contextlib.nullcontext()
        return self.tracer.start_as_current_span(name)

    def flush(self):
        """Send what is buffered before the job exits."""
        for provider in getattr(self, "providers", ()):
            with contextlib.suppress(Exception):
                provider.force_flush(timeout_millis=5000)  # an unreachable collector costs <=5 s
                provider.shutdown()


class Telemetry:
    """Collects the run's resources; ``summary`` is JSON-friendly."""

    def __init__(self, heartbeat_s=300, service="managed-workflow"):
        self.heartbeat_s = heartbeat_s
        self.stages = {}
        self.summary = {}
        self._otel = _Otel(service)
        self._stop = threading.Event()
        self._t0 = self._c0 = None
        self._tree_peak = 0.0  # sampled process-tree peak, used without a container counter

    def log(self, message=""):
        """Print a timestamped line with current and peak memory."""
        print(
            f"[{time.strftime('%H:%M:%S')}] {message} mem_now_gb={_fmt(current_gb())} "
            f"peak_gb={_fmt(peak_gb())}",
            flush=True,
        )

    @contextlib.contextmanager
    def stage(self, name):
        """Time a stage and record the peak memory reached by its end."""
        self.log(f"stage {name} start")
        start = time.perf_counter()
        with self._otel.span(name):
            try:
                yield
            finally:
                self.stages[name] = {
                    "s": round(time.perf_counter() - start, 1),
                    "peak_gb_at_end": _fmt(self._peak()),
                }
                self.log(f"stage {name} done in {self.stages[name]['s']} s")

    def _peak(self):
        container = _read_gb("memory.peak")
        if container is not None:
            return container
        return max(self._tree_peak, peak_gb())

    def _source(self):
        if _read_gb("memory.peak") is not None:
            return "container"
        return "process tree (sampled)" if self._tree_peak else "largest process"

    def _beat(self):
        last = time.perf_counter()
        sample = _read_gb("memory.peak") is None  # sample the tree only without a container counter
        while not self._stop.wait(0.2 if sample else self.heartbeat_s):
            if sample:
                self._tree_peak = max(self._tree_peak, _tree_gb() or 0.0)
            if time.perf_counter() - last >= self.heartbeat_s:
                self.log("heartbeat")
                last = time.perf_counter()

    def __enter__(self):
        self._t0, self._c0 = time.perf_counter(), _cpu_s()
        threading.Thread(target=self._beat, daemon=True).start()
        self.log(f"start, cores granted {_cores_granted()}")
        return self

    def __exit__(self, *exc):
        self._stop.set()
        wall = time.perf_counter() - self._t0
        self.summary = {
            "peak_gb": _fmt(self._peak()),
            "wall_s": round(wall, 1),
            "cores_used": round((_cpu_s() - self._c0) / wall, 1) if wall else None,
            "cores_granted": _cores_granted(),
            "stages": self.stages,
            "source": self._source(),
        }
        self.log("end")
        self._otel.flush()
        return False


def _fmt(value):
    return round(value, 2) if value is not None else None


def telemetry(heartbeat_s=300, service="managed-workflow"):
    """Context manager wrapping a run; see the module docstring."""
    return Telemetry(heartbeat_s=heartbeat_s, service=service)
