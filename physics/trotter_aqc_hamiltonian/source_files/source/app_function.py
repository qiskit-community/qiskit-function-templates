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

"""Application function: AQC-compressed Hamiltonian dynamics with mitigated readout.

Experiment-agnostic. Classical inputs (a 1D nearest-neighbour Pauli Hamiltonian,
a prepared initial state, evolution steps, observables, options) -> classical
output (the time series of each observable's expectation value). The
Hamiltonian's ``num_qubits`` fixes the system size; there is no separate size
input.

:class:`InputModel` defines and validates the input contract;
:class:`DynamicsFunction` orchestrates the stages:

    prepare state -> Trotter -> AQC compress -> execute -> expectations
"""
from __future__ import annotations

import time
from copy import deepcopy
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from . import build as build_stage
from . import aqc as aqc_stage
from . import execute as execute_stage
from .hamiltonian import TROTTER_METHODS, make_spec
from ._serverless import (
    Job,
    ServerlessError,
    get_logger,
    update_status,
)

logger = get_logger(__name__)


# ════════════════════════════════════════════════════════════════════════
#  Options models (pydantic)
# ════════════════════════════════════════════════════════════════════════


class _Base(BaseModel):
    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, arbitrary_types_allowed=True
    )


# Default settings handed straight to ``scipy.optimize.minimize(objective, x0,
# **optimizer_settings)`` in the AQC compression stage. Passed through as-is, so any
# minimize keyword (method, jac, tol, options, bounds, ...) is configurable; only the
# objective (fidelity loss) and x0 (ansatz params) are fixed by the function.
DEFAULT_OPTIMIZER_SETTINGS: dict = {"method": "L-BFGS-B", "jac": True, "options": {"maxiter": 300}}


class AQCSegment(_Base):
    """One compression segment.

    Its ``n_steps`` consecutive leading time steps are compressed into an ansatz
    generated from the ``ansatz_steps``-Trotter target. Deeper ansätze (larger
    ansatz_steps) hold fidelity for the later, more-entangled steps, so a plan
    typically compresses the first steps with a shallow ansatz and switches to a
    deeper one for the steps after that.
    """

    n_steps: int = Field(gt=0)
    ansatz_steps: int = Field(gt=0)


class AQCOptionsModel(_Base):
    """Tunables for the AQC compression stage."""

    max_bond: int = Field(gt=0, default=32)
    cutoff: float = Field(gt=0, default=1e-8)
    autodiff_backend: str = "jax"
    fidelity_target: Optional[float] = Field(
        default=None, gt=0, le=1
    )  # our early-stop, not a scipy arg
    # scipy.optimize.minimize kwargs for the fidelity optimisation; passed through as-is.
    optimizer_settings: dict = Field(default_factory=lambda: deepcopy(DEFAULT_OPTIMIZER_SETTINGS))


# Default ``EstimatorV2.options`` (the mitigation stack) used on the ``fake``/
# ``runtime`` paths when the caller passes no ``estimator_options``. This is an
# ``EstimatorOptions``-shaped dict handed straight to
# ``EstimatorV2(mode=..., options=...)``; a caller-supplied dict overrides it
# wholesale, so any EstimatorOptions field is configurable without a parallel
# pydantic model to maintain here.
DEFAULT_ESTIMATOR_OPTIONS: dict = {
    "dynamical_decoupling": {"enable": True, "sequence_type": "XY4"},
    "twirling": {
        "enable_gates": True,
        "num_randomizations": 1000,
        "shots_per_randomization": 128,
    },
    "resilience": {"measure_mitigation": True},  # TREX readout mitigation
}

# Default transpiler options handed to ``generate_preset_pass_manager(backend=..., **opts)``
# on the fake/runtime paths. Passed through as-is, so any preset-pass-manager keyword
# (seed_transpiler, routing_method, approximation_degree, ...) is configurable; ``backend``
# and ``target`` are owned by the execution path and rejected. (The pass manager's own
# default optimization_level is 2; we pin 3.)
DEFAULT_TRANSPILER_OPTIONS: dict = {"optimization_level": 3}

# PauliEvolutionGate synthesis config for the time evolution:
#   {"method": "lie" | "suzuki", "synthesis_settings": {<method-specific kwargs>}}
# ``method`` picks the (deterministic, <=2q) product formula; ``synthesis_settings`` is that
# method's constructor kwargs (JSON-serialisable — order, cx_structure, insert_barriers, ...).
# ``reps`` is owned by the function (= Trotter step count) and rejected. Default 2nd-order Suzuki.
DEFAULT_TROTTER_OPTIONS: dict = {"method": "suzuki", "synthesis_settings": {"order": 2}}


class InputModel(_Base):
    """The function's input contract."""

    t_steps: int = Field(gt=0)
    dt: float = Field(gt=0, default=0.2)
    # SuzukiTrotter synthesis kwargs (order, ...) for the time evolution; passed through
    # as-is. Default 2nd order; `reps` is set to the step count and rejected in _check.
    trotter_options: dict = Field(default_factory=lambda: deepcopy(DEFAULT_TROTTER_OPTIONS))
    # AQC compression plan: an ordered list of segments, each compressing its
    # ``n_steps`` leading time steps into an ansatz generated from the
    # ``ansatz_steps``-Trotter target. The total compressed count (sum of n_steps)
    # must be <= t_steps; remaining steps run as plain Trotter. A single-ansatz run
    # is just one segment, e.g. [{"n_steps": 5, "ansatz_steps": 1}].
    aqc_segments: list[AQCSegment] = Field(min_length=1)
    # 1-D nearest-neighbour Pauli Hamiltonian over the chain — required. Its
    # ``num_qubits`` fixes the system size; there is no separate ``n`` input.
    hamiltonian: SparsePauliOp
    initial_state: Optional[QuantumCircuit] = (
        None  # None -> |0...0>; else a prepared QuantumCircuit
    )
    # None -> per-site Z. Otherwise exactly what EstimatorV2 takes as its `observables`
    # argument (SparsePauliOp / Pauli / PauliList / Pauli string / {pauli: coeff} dict,
    # or a (nested) list of those); one observable -> one output column.
    observables: Optional[Any] = None
    backend: str = "runtime"  # "statevector" | "fake" | "runtime"
    backend_name: Optional[str] = (
        None  # runtime IBM backend / named fake; None -> least_busy on runtime
    )
    batches: int = Field(gt=0, default=1)  # split pubs across N runtime jobs (1 = single job)
    # Fan the local-sim execution (statevector/fake) across all available cores
    # via Ray. False (default) = sequential. No effect on the runtime backend.
    parallel_sim: bool = False
    # generate_preset_pass_manager kwargs for the fake/runtime paths; passed through
    # as-is (defaults to optimization_level 3). `backend`/`target` are set by the
    # execution path and rejected in _check.
    transpiler_options: dict = Field(default_factory=lambda: deepcopy(DEFAULT_TRANSPILER_OPTIONS))
    # EstimatorV2.options for the fake/runtime paths; passed through as-is. An
    # EstimatorOptions-shaped dict (defaults to the DD/twirling/TREX stack above).
    estimator_options: dict = Field(default_factory=lambda: deepcopy(DEFAULT_ESTIMATOR_OPTIONS))
    # AQC compression tuning (MPS bond/cutoff, optimizer, fidelity_target); optional.
    aqc_options: AQCOptionsModel = Field(default_factory=AQCOptionsModel)

    @model_validator(mode="after")
    def _check(self):
        if self.hamiltonian.num_qubits < 2:
            raise ValueError("hamiltonian must act on >= 2 qubits (1-D nearest-neighbour chain).")
        reserved = {"backend", "target"} & set(self.transpiler_options)
        if reserved:
            raise ValueError(
                f"transpiler_options may not set {sorted(reserved)} — the execution backend owns those."
            )
        trotter = self.trotter_options
        unknown = set(trotter) - {"method", "synthesis_settings"}
        if unknown:
            raise ValueError(
                f"trotter_options may only contain 'method' and 'synthesis_settings', got "
                f"extra {sorted(unknown)} — put method-specific kwargs under 'synthesis_settings'."
            )
        if trotter.get("method", "suzuki") not in TROTTER_METHODS:
            raise ValueError(
                f"trotter_options['method'] must be one of {sorted(TROTTER_METHODS)}, "
                f"got {trotter.get('method')!r}."
            )
        reserved_synth = {"reps", "time"} & set(trotter.get("synthesis_settings", {}))
        if reserved_synth:
            raise ValueError(
                f"trotter_options['synthesis_settings'] may not set {sorted(reserved_synth)} — "
                "the function owns the evolution time (dt × step count) and the repetition "
                "count (= step count)."
            )
        opt_settings = self.aqc_options.optimizer_settings
        reserved_opt = {"fun", "x0"} & set(opt_settings)
        if reserved_opt:
            raise ValueError(
                f"aqc_options.optimizer_settings may not set {sorted(reserved_opt)} — the "
                "objective (fidelity loss) and initial parameters are fixed by the function."
            )
        if self.aqc_options.fidelity_target is not None and "callback" in opt_settings:
            raise ValueError(
                "aqc_options.optimizer_settings may not set 'callback' when fidelity_target "
                "is set — fidelity_target's early-stop uses the optimiser's callback slot."
            )
        total_aqc = sum(seg.n_steps for seg in self.aqc_segments)
        if total_aqc > self.t_steps:
            raise ValueError(
                f"aqc_segments compress {total_aqc} steps in total, which must be <= "
                f"t_steps ({self.t_steps})."
            )
        for i, seg in enumerate(self.aqc_segments):
            if seg.ansatz_steps > self.t_steps:
                raise ValueError(
                    f"aqc_segments[{i}].ansatz_steps ({seg.ansatz_steps}) must be <= "
                    f"t_steps ({self.t_steps}) — it seeds the ansatz from that Trotter target."
                )
        if self.backend not in ("statevector", "fake", "runtime"):
            raise ValueError("backend must be 'statevector', 'fake', or 'runtime'.")
        return self


# ════════════════════════════════════════════════════════════════════════
#  Function
# ════════════════════════════════════════════════════════════════════════


class DynamicsFunction:
    """End-to-end AQC-compressed Hamiltonian dynamics with mitigated readout."""

    def __init__(self, **arguments) -> None:
        try:
            self.cfg = InputModel(**arguments)
        except Exception as exc:  # pydantic ValidationError or ValueError
            raise ServerlessError(
                code="4615",
                message=f"Invalid input: {exc}",
                details={
                    "solution": (
                        "Check your arguments against the input contract "
                        "(required: hamiltonian, t_steps, aqc_segments)."
                    )
                },
            ) from exc
        self._jobs: list = []  # tracked runtime jobs (runtime backend only)

    # -- orchestration ----------------------------------------------------
    def run(self) -> dict:
        """Run the full pipeline and return the serialisable result dictionary."""
        cfg = self.cfg

        operator = cfg.hamiltonian
        n = operator.num_qubits  # system size is set by the Hamiltonian
        spec = make_spec("pauli_1d_nn", n, operator=operator, trotter_options=cfg.trotter_options)

        # --- Stage 1+2: build + AQC compression (classical optimisation) ---
        t_opt0 = time.time()
        update_status(Job.OPTIMIZING_HARDWARE)

        init = build_stage.prepare_initial_state(n, cfg.initial_state)

        # Built to cfg.t_steps (not just the compressed count): also serves as the full,
        # uncompressed-Trotter reference series for metadata.circuit_stats below.
        targets = build_stage.build_evolution_targets(spec, init, cfg.dt, cfg.t_steps)
        total_aqc = sum(seg.n_steps for seg in cfg.aqc_segments)  # compressed step count
        aqc_opts = aqc_stage.AQCOptions(
            max_bond=cfg.aqc_options.max_bond,
            cutoff=cfg.aqc_options.cutoff,
            autodiff_backend=cfg.aqc_options.autodiff_backend,
            segments=[(seg.n_steps, seg.ansatz_steps) for seg in cfg.aqc_segments],
            fidelity_target=cfg.aqc_options.fidelity_target,
            optimizer_settings=cfg.aqc_options.optimizer_settings,
        )
        aqc_res = aqc_stage.compress(targets, aqc_opts)
        evolved = aqc_stage.assemble_all_circuits(aqc_res, total_aqc, cfg.t_steps, spec, cfg.dt)
        # t=0 is the prepared state (no evolution); then steps 1..t_steps.
        all_circuits = [init] + evolved
        observables, obs_labels = build_stage.build_observables(n, cfg.observables)
        t_opt1 = time.time()

        # --- Stage 3: execute -------------------------------------------
        # On runtime the status tracks the QPU job: WAITING_QPU while it sits in
        # the queue, then EXECUTING_QPU once it starts running (both signalled
        # from the execute stage). Local sim backends have no queue, so they mark
        # EXECUTING_QPU directly.
        t_exec0 = time.time()

        def _on_exec_status(stage: str) -> None:
            update_status(Job.WAITING_QPU if stage == "waiting" else Job.EXECUTING_QPU)

        exec_opts = execute_stage.ExecutionOptions(
            backend=cfg.backend,
            backend_name=cfg.backend_name,
            transpiler_options=cfg.transpiler_options,
            batches=cfg.batches,
            estimator_options=cfg.estimator_options,
            parallel_sim=cfg.parallel_sim,
            on_status=_on_exec_status if cfg.backend == "runtime" else None,
        )
        if cfg.backend != "runtime":
            update_status(Job.EXECUTING_QPU)
        evs = execute_stage.measure_observables(all_circuits, observables, exec_opts)
        t_exec1 = time.time()

        # --- Stage 4: assemble result -----------------------------------
        update_status(Job.POST_PROCESSING)
        times = [k * cfg.dt for k in range(cfg.t_steps + 1)]
        # Per-segment AQC breakdown: each segment's plan plus the global step indices
        # it covers, its ansatz parameter count (shared across its steps), and its
        # per-step fidelities. Steps are assigned to segments in order, matching
        # compress(). The flat `aqc_fidelities` below keeps the full-series view.
        aqc_segments_meta = []
        _step = 0
        for seg in cfg.aqc_segments:
            seg_steps = list(range(_step + 1, _step + seg.n_steps + 1))
            _step += seg.n_steps
            aqc_segments_meta.append(
                {
                    "n_steps": seg.n_steps,
                    "ansatz_steps": seg.ansatz_steps,
                    "steps": seg_steps,
                    "n_params": len(aqc_res.params[seg_steps[0]]),
                    "fidelities": {k: aqc_res.fidelities[k] for k in seg_steps},
                }
            )
        qpu_key = "QPU_TIME" if cfg.backend == "runtime" else "CPU_TIME"
        # Per-step depth/gate-count: full (uncompressed) Trotter vs the AQC+Trotter
        # circuit actually executed, so the caller can quantify the compression saving.
        circuit_stats_by_step = {
            k: {
                "full_trotter": build_stage.circuit_stats(targets[k]),
                "aqc_trotter": build_stage.circuit_stats(all_circuits[k]),
            }
            for k in range(1, cfg.t_steps + 1)
        }
        result = {
            "times": times,
            "expectation_values": evs.tolist(),
            "observable_labels": obs_labels,
            "metadata": {
                "n": n,
                "t_steps": cfg.t_steps,
                "dt": cfg.dt,
                "tier": spec.tier,
                "aqc_compressed_steps": total_aqc,
                "aqc_segments": aqc_segments_meta,
                "execution_backend": cfg.backend,
                "aqc_fidelities": aqc_res.fidelities,  # flat, full-series (all compressed steps)
                "circuit_stats": circuit_stats_by_step,
                "warnings": aqc_res.warnings,  # e.g. cotengrust-fallback notice
                "resource_usage": {
                    "RUNNING: OPTIMIZING_FOR_HARDWARE": {"CPU_TIME": t_opt1 - t_opt0},
                    "RUNNING: WAITING_FOR_QPU": {"CPU_TIME": 0.0},
                    "RUNNING: EXECUTING_QPU": {qpu_key: t_exec1 - t_exec0},
                },
            },
        }
        logger.info(
            "Done. %d observables x %d times; AQC fids=%s; aqc_compress=%.2fs execute=%.2fs",
            len(obs_labels),
            len(times),
            {k: round(v, 4) for k, v in aqc_res.fidelities.items()},
            t_opt1 - t_opt0,
            t_exec1 - t_exec0,
        )
        return result

    # -- termination ------------------------------------------------------
    def cleanup(self) -> None:
        """Cancel any runtime jobs this run still has in flight."""
        for job in self._jobs:
            try:
                job.cancel()
            # Best-effort cleanup: a job that cannot be cancelled (already
            # finished, transport error, ...) must not mask the original failure.
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("Unable to cancel runtime job: %s", exc)

    def signal_handler(self, signum, frame):  # pragma: no cover
        """Cancel in-flight jobs when the gateway signals termination."""
        del frame  # part of the signal-handler signature, unused
        logger.info("Received signal %s, shutting down.", signum)
        self.cleanup()
