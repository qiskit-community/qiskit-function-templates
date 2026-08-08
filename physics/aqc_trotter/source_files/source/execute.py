# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Stage 3 — Execute: PUBs in, expectation values out.

A self-contained estimator: it accepts PUBs ``(circuit, observables)`` and
returns per-PUB expectation values. It holds no state from the earlier stages,
so the backend choice is the only thing that varies between runs.

Three backends, selected by :attr:`ExecutionOptions.backend`:

* ``statevector`` (default) — ``StatevectorEstimator``: exact, in-process, no
  credentials.
* ``fake`` — noisy simulation on a Qiskit fake backend via
  ``EstimatorV2(mode=fake_backend)`` (Aer noise model). Still in-process and
  credential-free, but it applies the same mitigation stack as ``runtime``
  (DD/XY4 + Pauli twirling + TREX). Defaults to the 127-qubit
  ``fake_sherbrooke`` (real measured noise); a different named fake or an
  auto-sized ``GenericBackendV2`` (for chains > 127 qubits) is also supported.
* ``runtime`` — the same mitigated ``EstimatorV2`` against a real backend via
  ``get_runtime_service()``, split across ``batches`` jobs. Requires IBM Quantum
  credentials and a gateway, so it is the one path the test suite cannot cover.
"""
from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

from ._serverless import get_logger, get_runtime_service

# Optional Ray fan-out for the in-process simulation paths (statevector / fake).
# Ray is used directly (ray.remote / ray.get) rather than through a wrapper. On
# the gateway it is already initialized; otherwise `_run_parallel` starts a
# per-core cluster on first use. If ray is not installed the parallel path
# disables itself and execution falls back to the sequential loop.
try:  # pragma: no cover - present wherever ray is installed
    import ray
except Exception:  # pylint: disable=broad-exception-caught  # pragma: no cover
    ray = None  # pylint: disable=invalid-name

logger = get_logger(__name__)

PubLike = tuple[QuantumCircuit, list[SparsePauliOp]]


@dataclass
class ExecutionOptions:
    """Backend selection and per-backend knobs for the execute stage."""

    backend: str = "statevector"  # "statevector" | "fake" | "runtime"
    backend_name: Optional[str] = None  # IBM backend (runtime) or named fake; None -> least_busy
    # kwargs for generate_preset_pass_manager(backend=..., **transpiler_options) on the
    # fake/runtime paths, passed through as-is (backend/target set by the exec path).
    transpiler_options: dict = field(default_factory=lambda: {"optimization_level": 3})
    batches: int = 1  # split pubs across N runtime jobs (1 = single job)
    # EstimatorV2.options applied on the fake + runtime paths (DD / twirling /
    # resilience / ...). An EstimatorOptions-shaped dict passed straight to
    # EstimatorV2(mode=..., options=...); {} keeps the estimator's own defaults.
    estimator_options: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)
    # optional status hook (runtime path only): called with "waiting" once the
    # job is queued, then "executing" once it leaves the queue and starts running.
    on_status: Optional[Callable[[str], None]] = None
    # Local-sim fan-out toggle (statevector/fake only). True -> split the
    # per-time-step PUBs across all cores Ray sees (chunked into Ray tasks);
    # False -> the sequential loop (default, unchanged behavior). Ignored when
    # Ray is absent or there is a single PUB. The runtime/QPU path is unaffected
    # (it parallelizes via `batches`).
    parallel_sim: bool = False


def _run_statevector(pubs: list[PubLike]) -> np.ndarray:
    """Exact expectation values for each PUB. Returns ``(n_pubs, n_obs)``."""
    estimator = StatevectorEstimator()
    rows = []
    for qc, observables in pubs:
        result = estimator.run([(qc, observables)]).result()
        rows.append(np.asarray(result[0].data.evs, dtype=float))
    return np.vstack(rows)


# Default fake backend: a snapshot of the 127-qubit IBM Sherbrooke device
# (heavy-hex, real measured noise). 127 qubits hosts any realistic chain, and
# the local Aer estimator only simulates the active qubits, so the full width is
# free. Chains larger than this fall back to a synthesized GenericBackendV2.
_DEFAULT_FAKE = "fake_sherbrooke"


def _make_fake_backend(n_qubits: int, opts: ExecutionOptions):
    """Build a local fake backend with at least ``n_qubits`` qubits.

    ``opts.backend_name`` selects a device-derived fake from
    ``qiskit_ibm_runtime.fake_provider`` (e.g. ``"fake_sherbrooke"``); an
    explicitly named one must have enough qubits. With no name we default to
    Sherbrooke (127 qubits, real noise) and only synthesize a ``GenericBackendV2``
    if the chain is larger than that — so any ``n`` works out of the box.
    """
    from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

    name = opts.backend_name or _DEFAULT_FAKE
    backend = FakeProviderForBackendV2().backend(name)
    if backend.num_qubits >= n_qubits:
        return backend

    if opts.backend_name:  # an explicit choice that doesn't fit is an error
        raise ValueError(
            f"Fake backend {opts.backend_name!r} has {backend.num_qubits} "
            f"qubits, fewer than the {n_qubits} the circuit needs. Pick a larger "
            "fake backend or omit backend_name to auto-size a GenericBackendV2."
        )

    # Default Sherbrooke can't host this (very large) chain: size a generic one.
    from qiskit.providers.fake_provider import GenericBackendV2

    logger.info(
        "Chain (%d qubits) exceeds %s (%d qubits); using a synthesized "
        "GenericBackendV2 instead.",
        n_qubits,
        name,
        backend.num_qubits,
    )
    return GenericBackendV2(num_qubits=n_qubits)


def _run_fake(pubs: list[PubLike], opts: ExecutionOptions) -> np.ndarray:
    """Noisy *local* execution on a fake backend. No credentials, no gateway.

    Mirrors the runtime mitigation knobs but runs in-process via
    ``EstimatorV2(mode=fake_backend)`` (Aer noise), so it works in a plain
    in-process ``DynamicsFunction(...).run()`` call exactly like ``statevector``.
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import EstimatorV2 as Estimator

    n_qubits = pubs[0][0].num_qubits
    backend = _make_fake_backend(n_qubits, opts)
    pm = generate_preset_pass_manager(backend=backend, **opts.transpiler_options)

    estimator = Estimator(mode=backend, options=opts.estimator_options)

    rows = []
    for qc, observables in pubs:
        isa = pm.run(qc)
        isa_obs = [ob.apply_layout(isa.layout) for ob in observables]
        result = estimator.run([(isa, isa_obs)]).result()
        rows.append(np.asarray(result[0].data.evs, dtype=float))
    return np.vstack(rows)


def _run_runtime(pubs: list[PubLike], opts: ExecutionOptions) -> np.ndarray:
    """Mitigated execution on real hardware, split across ``opts.batches`` jobs.

    More than one batch runs the jobs inside a single ``Batch``, so the group
    takes one queue wait. The default of a single batch submits a plain runtime
    job instead and creates no session.
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import Batch, EstimatorV2 as Estimator

    service = get_runtime_service()
    backend = (
        service.backend(opts.backend_name)
        if opts.backend_name
        else service.least_busy(operational=True, simulator=False)
    )

    pm = generate_preset_pass_manager(backend=backend, **opts.transpiler_options)
    circuits = [qc for qc, _ in pubs]
    isa_circuits = pm.run(circuits)
    # Observables are identical per pub here; remap each pub's onto its layout.
    isa_pubs = [
        (isa, [ob.apply_layout(isa.layout) for ob in observables])
        for isa, (_, observables) in zip(isa_circuits, pubs)
    ]

    n_pubs = len(isa_pubs)
    batches = max(1, min(opts.batches, n_pubs))
    bounds = np.array_split(np.arange(n_pubs), batches)

    rows: list[Optional[np.ndarray]] = [None] * n_pubs
    # A ``Batch`` is a server-side session (one ``POST /sessions``). It pays for
    # itself when several jobs can share a single queue wait, but for the
    # single-job default it is pure overhead, and one more call that can fail
    # before any circuit reaches the QPU. So batch only when there is more than
    # one job, and use plain job mode otherwise.
    batch_cm = Batch(backend=backend) if batches > 1 else nullcontext()
    with batch_cm as batch:
        # Pass the mode explicitly: the batch when batching, else the backend.
        mode = batch if batch is not None else backend
        estimator = Estimator(mode, options=opts.estimator_options)

        jobs = []
        for idx in bounds:
            sub = [isa_pubs[i] for i in idx]
            jobs.append((idx, estimator.run(sub)))

        # Distinguish the (often long) QPU queue wait from on-hardware execution:
        # signal "waiting" once submitted, then "executing" when the first job
        # leaves the queue. Any non-queue status ends the poll (fail-safe).
        if opts.on_status is not None:
            opts.on_status("waiting")
            first_job = jobs[0][1]
            while str(first_job.status()) in ("INITIALIZING", "QUEUED", "VALIDATING"):
                time.sleep(5)
            opts.on_status("executing")

        for idx, job in jobs:
            res = job.result()
            for local_i, global_i in enumerate(idx):
                # hardware layout reverses site order vs little-endian observable list
                rows[global_i] = np.asarray(res[local_i].data.evs, dtype=float)[::-1]
    return np.vstack(rows)


def _run_local_chunk(
    backend: str,
    sub_pubs: list[PubLike],
    backend_name: Optional[str],
    transpiler_options: dict,
    estimator_options: dict,
) -> np.ndarray:
    """Run one chunk of PUBs on a local-sim backend; returns ``(n_sub, n_obs)``.

    Self-contained (rebuilds its own estimator/backend) so it can run in a
    separate Ray worker process, and it reuses the exact sequential
    implementations — a chunk computes bit-for-bit what the sequential path
    would for those same PUBs.
    """
    if backend == "statevector":
        return _run_statevector(sub_pubs)
    opts = ExecutionOptions(
        backend="fake",
        backend_name=backend_name,
        transpiler_options=transpiler_options,
        estimator_options=estimator_options,
    )
    return _run_fake(sub_pubs, opts)


def _run_parallel(pubs: list[PubLike], opts: ExecutionOptions) -> np.ndarray:
    """Fan the per-PUB loop out across all available cores as Ray tasks, chunked.

    One task per chunk (not per PUB) so each task builds its estimator once and
    amortizes that setup over its circuits. Row order is preserved.
    """
    n_pubs = len(pubs)

    if not ray.is_initialized():
        # No-op on the gateway (Ray already up); locally starts a per-core
        # cluster. ignore_reinit_error guards against a concurrent init.
        ray.init(ignore_reinit_error=True)

    # Full available parallel capacity: one chunk per core Ray sees, capped at
    # the number of circuits (never more chunks than PUBs).
    n_cores = int(ray.cluster_resources().get("CPU", n_pubs))
    n_chunks = max(1, min(n_cores, n_pubs))
    groups = [g for g in np.array_split(np.arange(n_pubs), n_chunks) if len(g)]
    logger.info("Executing %d PUBs across %d Ray task(s) (%d cores).", n_pubs, len(groups), n_cores)

    remote = ray.remote(_run_local_chunk)
    refs = [
        remote.remote(
            opts.backend,
            [pubs[i] for i in g],
            opts.backend_name,
            opts.transpiler_options,
            opts.estimator_options,
        )
        for g in groups
    ]
    blocks = ray.get(refs)

    rows: list[Optional[np.ndarray]] = [None] * n_pubs
    for group, block in zip(groups, blocks):
        for local_i, global_i in enumerate(group):
            rows[global_i] = block[local_i]
    return np.vstack(rows)


def run_pubs(pubs: Iterable[PubLike], opts: ExecutionOptions) -> np.ndarray:
    """Execute PUBs and return an ``(n_pubs, n_obs)`` array of expectations."""
    pubs = list(pubs)
    if not pubs:
        raise ValueError("At least one PUB is required.")
    # Local-sim fan-out (opt-in via `parallel_sim`): the runtime path has its own
    # axis (`batches`) and is never fanned out here.
    if (
        opts.backend in ("statevector", "fake")
        and opts.parallel_sim
        and ray is not None
        and len(pubs) > 1
    ):
        logger.info(
            "Executing %d PUBs on %s in parallel (Ray fan-out).",
            len(pubs),
            opts.backend,
        )
        return _run_parallel(pubs, opts)
    if opts.backend == "statevector":
        logger.info("Executing %d PUBs on StatevectorEstimator (exact).", len(pubs))
        return _run_statevector(pubs)
    if opts.backend == "fake":
        logger.info(
            "Executing %d PUBs on fake backend %s (noisy, local).",
            len(pubs),
            opts.backend_name or _DEFAULT_FAKE,
        )
        return _run_fake(pubs, opts)
    if opts.backend == "runtime":
        logger.info(
            "Executing %d PUBs on runtime backend %s (mitigated).",
            len(pubs),
            opts.backend_name or "least_busy",
        )
        return _run_runtime(pubs, opts)
    raise ValueError(f"Unknown execution backend {opts.backend!r}.")


def measure_observables(
    circuits: list[QuantumCircuit],
    observables: list[SparsePauliOp],
    opts: ExecutionOptions,
) -> np.ndarray:
    """Measure every observable on every circuit; return ``(n_circuits, n_obs)``.

    One PUB per time-step circuit, each measuring the full observable list.
    """
    pubs = [(qc, observables) for qc in circuits]
    return run_pubs(pubs, opts)
