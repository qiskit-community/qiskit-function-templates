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

"""Per-module unit checks for the AQC-dynamics function."""

import sys
import unittest
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError
from qiskit import QuantumCircuit
from qiskit.quantum_info import PauliList, SparsePauliOp, Statevector

from ..source_files.source import build as build_stage
from ..source_files.source.app_function import InputModel
from ..source_files.source.aqc import check_cotengrust_engaged
from ..source_files.source.hamiltonian import Pauli1DNNSpec, make_spec


def _nn_op(n, terms, fields=()):
    """Uniform 1D NN SparsePauliOp: each (pauli, coeff) coupling on every bond,
    each (pauli, coeff) field on every site."""
    sl = [(p, [i, i + 1], c) for i in range(n - 1) for p, c in terms]
    sl += [(p, [i], c) for i in range(n) for p, c in fields]
    return SparsePauliOp.from_sparse_list(sl, num_qubits=n)


# A valid 6-qubit 1D-NN Hamiltonian; the input's num_qubits now fixes the system size.
_H6 = _nn_op(6, [("ZZ", 1.0)])
_HEIS_TERMS = [("XX", 0.5), ("YY", 0.5), ("ZZ", 0.5)]


class TestHamiltonianSpec(unittest.TestCase):
    """Hamiltonian spec construction and Trotter evolution."""

    def test_make_spec_dispatch(self):
        """The factory returns the tier's spec class."""
        spec = make_spec("pauli_1d_nn", 6, operator=_nn_op(6, _HEIS_TERMS))
        self.assertIsInstance(spec, Pauli1DNNSpec)
        self.assertEqual(spec.tier, "pauli_1d_nn")

    def test_build_evolution_is_one_or_two_qubit_and_physical_time(self):
        """Evolution stays at most 2-qubit; zero steps gives an empty circuit."""
        spec = Pauli1DNNSpec(n=6, operator=_nn_op(6, _HEIS_TERMS))
        qc = spec.build_evolution(2, dt=0.3)
        self.assertEqual(qc.num_qubits, 6)
        self.assertGreater(qc.size(), 0)
        self.assertTrue(all(inst.operation.num_qubits <= 2 for inst in qc.data))
        self.assertEqual(spec.build_evolution(0, dt=0.3).size(), 0)

    def test_trotter_method_selects_synthesis(self):
        """`method` picks the product formula, and both stay AQC-compatible."""
        op = _nn_op(4, _HEIS_TERMS)
        lie = Pauli1DNNSpec(
            n=4, operator=op, trotter_options={"method": "lie", "synthesis_settings": {}}
        ).build_evolution(2, 0.2)
        suz = Pauli1DNNSpec(
            n=4,
            operator=op,
            trotter_options={"method": "suzuki", "synthesis_settings": {"order": 2}},
        ).build_evolution(2, 0.2)
        self.assertGreater(lie.size(), 0)
        self.assertGreater(suz.size(), 0)
        # stays AQC-compatible
        self.assertTrue(all(inst.operation.num_qubits <= 2 for inst in lie.data))
        # 2nd-order Suzuki does more work per step than 1st-order Lie
        self.assertGreater(suz.size(), lie.size())

    def test_pauli_spec_rejects_wrong_num_qubits(self):
        """The operator width must equal n, and n must be at least 2."""
        with self.assertRaises(ValueError):
            Pauli1DNNSpec(n=4, operator=_nn_op(5, [("ZZ", 1.0)]))  # operator nq (5) != n (4)
        with self.assertRaises(ValueError):
            Pauli1DNNSpec(n=1, operator=SparsePauliOp("Z"))  # n < 2


class TestInitialState(unittest.TestCase):
    """Initial-state preparation."""

    def test_initial_state_default_is_all_zero(self):
        """No circuit given -> |0...0>."""
        qc = build_stage.prepare_initial_state(4, None)
        self.assertEqual(qc.num_qubits, 4)
        sv = Statevector(qc).data
        expected = np.zeros(2**4)
        expected[0] = 1.0
        np.testing.assert_allclose(np.abs(sv), expected, atol=1e-8)

    def test_initial_state_circuit_form(self):
        """A prepared circuit is preserved up to global phase; width is checked."""
        # Native QuantumCircuit hand-off (what serverless QPY-deserialises on the worker).
        src = QuantumCircuit(3)
        src.h(0)
        src.cx(0, 1)
        src.rz(0.5, 2)
        qc = build_stage.prepare_initial_state(3, src)
        np.testing.assert_allclose(
            np.abs(np.vdot(Statevector(src).data, Statevector(qc).data)), 1.0, atol=1e-6
        )
        with self.assertRaises(ValueError):
            build_stage.prepare_initial_state(3, QuantumCircuit(2))  # wrong nq


class TestObservables(unittest.TestCase):
    """Observable parsing and labelling."""

    def test_observables_default_per_site_z(self):
        """None -> single-site Z on every qubit."""
        obs, labels = build_stage.build_observables(5)
        self.assertEqual(len(obs), 5)
        self.assertEqual(labels, [f"Z_{i}" for i in range(5)])
        self.assertTrue(all(isinstance(o, SparsePauliOp) and o.num_qubits == 5 for o in obs))

    def test_observables_accepts_estimator_forms(self):
        """Anything EstimatorV2 takes as `observables` is accepted."""
        # Pauli strings, SparsePauliOp, {pauli: coeff} dicts, PauliList — one
        # observable (column) per element.
        obs, labels = build_stage.build_observables(4, ["IIIZ", "IIZZ"])
        self.assertEqual(labels, ["IIIZ", "IIZZ"])  # single unit Pauli -> its string
        obs, labels = build_stage.build_observables(4, PauliList(["IIIZ", "IIZZ"]))
        self.assertEqual(labels, ["IIIZ", "IIZZ"])
        obs, labels = build_stage.build_observables(4, [{"IIIZ": 1.0}, {"IIZZ": 1.0}])
        self.assertEqual(labels, ["IIIZ", "IIZZ"])
        self.assertTrue(all(isinstance(o, SparsePauliOp) and o.num_qubits == 4 for o in obs))

    def test_observables_single_and_multiterm(self):
        """A bare observable is one column; a multi-term one is summed into one."""
        obs, labels = build_stage.build_observables(4, "IIIZ")
        self.assertEqual(len(obs), 1)
        self.assertEqual(labels, ["IIIZ"])
        # Multi-term observable (SparsePauliOp or dict) -> one column, obs_<i>.
        obs, labels = build_stage.build_observables(4, SparsePauliOp(["ZZII", "IIXX"], [1.0, 0.5]))
        self.assertEqual(len(obs), 1)
        self.assertEqual(labels, ["obs_0"])
        _, labels = build_stage.build_observables(4, [{"ZZII": 1.0, "IIXX": 0.5}, "IIIZ"])
        self.assertEqual(labels, ["obs_0", "IIIZ"])

    def test_observables_validation(self):
        """Wrong-width and non-observable inputs are rejected."""
        with self.assertRaises(ValueError):
            build_stage.build_observables(4, [SparsePauliOp("ZZ")])  # wrong width (2 != 4)
        with self.assertRaises(ValueError):
            build_stage.build_observables(4, ["ZZ"])  # Pauli string wrong width
        with self.assertRaises(Exception):
            build_stage.build_observables(4, [123])  # not an observable


class TestEvolutionTargets(unittest.TestCase):
    """Trotter target construction and circuit statistics."""

    def test_build_evolution_targets_shapes(self):
        """One target per step, each deeper than the last."""
        spec = Pauli1DNNSpec(n=6, operator=_nn_op(6, _HEIS_TERMS))
        init = build_stage.prepare_initial_state(6, None)
        targets = build_stage.build_evolution_targets(spec, init, dt=0.2, up_to_step=3)
        self.assertEqual(sorted(targets), [1, 2, 3])
        self.assertTrue(all(c.num_qubits == 6 for c in targets.values()))
        self.assertGreater(targets[3].size(), targets[1].size())

    def test_circuit_stats_counts_2q_depth_and_gates(self):
        """2q depth/count ignore 1q gates."""
        # 3 cx gates in series, plus a 1q gate that counts toward neither metric.
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        stats = build_stage.circuit_stats(qc)
        self.assertEqual(stats, {"depth_2q": 3, "num_2q_gates": 3})


class TestCotengrustCheck(unittest.TestCase):
    """The optional Rust contraction backend check."""

    def test_cotengrust_present_returns_true(self):
        """cotengrust is installed in the test env; the check reports it engaged."""
        self.assertIs(check_cotengrust_engaged(max_bond=32), True)

    def test_cotengrust_missing_falls_back(self):
        """Absent cotengrust is a soft fallback: warn and return False, no raise."""
        # The caller also logs it and surfaces it in the result metadata.
        # sys.modules[name] = None makes `import cotengrust` raise ImportError.
        with patch.dict(sys.modules, {"cotengrust": None}):
            with self.assertWarnsRegex(UserWarning, "cotengrust not available"):
                self.assertIs(check_cotengrust_engaged(max_bond=32), False)


class TestInputModel(unittest.TestCase):
    """Input contract validation."""

    @staticmethod
    def _args(**overrides):
        """A minimal valid argument set, with per-test overrides."""
        args = {
            "hamiltonian": _H6,
            "t_steps": 4,
            "aqc_segments": [{"n_steps": 2, "ansatz_steps": 1}],
        }
        args.update(overrides)
        return args

    def test_input_rejects_segments_sum_gt_t_steps(self):
        """The compressed step total may not exceed t_steps."""
        with self.assertRaises(ValidationError):
            InputModel(**self._args(t_steps=5, aqc_segments=[{"n_steps": 8, "ansatz_steps": 1}]))
        with self.assertRaises(ValidationError):  # two segments summing past t_steps
            InputModel(
                **self._args(
                    t_steps=5,
                    aqc_segments=[
                        {"n_steps": 3, "ansatz_steps": 1},
                        {"n_steps": 3, "ansatz_steps": 2},
                    ],
                )
            )

    def test_input_rejects_ansatz_steps_gt_t_steps(self):
        """ansatz_steps seeds the ansatz from a Trotter target, which must exist."""
        with self.assertRaises(ValidationError):
            InputModel(**self._args(t_steps=5, aqc_segments=[{"n_steps": 3, "ansatz_steps": 8}]))

    def test_input_rejects_empty_or_nonpositive_segments(self):
        """At least one segment, and every count must be positive."""
        with self.assertRaises(ValidationError):
            InputModel(**self._args(t_steps=5, aqc_segments=[]))
        with self.assertRaises(ValidationError):
            InputModel(**self._args(t_steps=5, aqc_segments=[{"n_steps": 0, "ansatz_steps": 1}]))
        with self.assertRaises(ValidationError):
            InputModel(**self._args(t_steps=5, aqc_segments=[{"n_steps": 2, "ansatz_steps": 0}]))

    def test_input_segments_accepted_single_and_multi(self):
        """Both a single segment and an ordered multi-segment plan parse."""
        single = InputModel(
            **self._args(t_steps=5, aqc_segments=[{"n_steps": 3, "ansatz_steps": 1}])
        )
        self.assertEqual([(s.n_steps, s.ansatz_steps) for s in single.aqc_segments], [(3, 1)])
        multi = InputModel(
            **self._args(
                t_steps=10,
                aqc_segments=[
                    {"n_steps": 6, "ansatz_steps": 1},
                    {"n_steps": 4, "ansatz_steps": 2},
                ],
            )
        )
        self.assertEqual(
            [(s.n_steps, s.ansatz_steps) for s in multi.aqc_segments], [(6, 1), (4, 2)]
        )

    def test_input_runtime_backend_name_optional(self):
        """backend_name is optional for runtime (resolved to least_busy at run time)."""
        cfg = InputModel(**self._args(backend="runtime"))
        self.assertEqual(cfg.backend, "runtime")
        self.assertIsNone(cfg.backend_name)

    def test_input_rejects_unknown_backend(self):
        """Only the three known execution backends are accepted."""
        with self.assertRaises(ValidationError):
            InputModel(**self._args(backend="qpu"))

    def test_input_requires_hamiltonian(self):
        """A Trotterization function is meaningless without a Hamiltonian."""
        # It is required, there is no separate `n`, and it must act on >= 2 qubits.
        args = self._args()
        args.pop("hamiltonian")
        with self.assertRaises(ValidationError):
            InputModel(**args)
        with self.assertRaises(ValidationError):
            InputModel(**self._args(n=6))  # `n` is no longer accepted
        with self.assertRaises(ValidationError):
            InputModel(**self._args(hamiltonian=SparsePauliOp("Z")))  # 1-qubit

    def test_input_defaults_ok(self):
        """The documented defaults are what the model produces."""
        cfg = InputModel(**self._args())
        self.assertEqual(cfg.backend, "runtime")
        self.assertEqual(cfg.hamiltonian.num_qubits, 6)  # system size read off the Hamiltonian
        self.assertEqual(cfg.aqc_options.max_bond, 32)  # AQC tuning is its own object
        # optimizer_settings is a scipy.optimize.minimize passthrough
        self.assertEqual(
            cfg.aqc_options.optimizer_settings,
            {"method": "L-BFGS-B", "jac": True, "options": {"maxiter": 300}},
        )
        # transpiler_options (pass-manager kwargs) and batches are top-level knobs
        self.assertEqual(cfg.transpiler_options, {"optimization_level": 3})
        self.assertEqual(
            cfg.trotter_options, {"method": "suzuki", "synthesis_settings": {"order": 2}}
        )
        self.assertEqual(cfg.batches, 1)
        # estimator_options is an EstimatorOptions-shaped dict passed to EstimatorV2
        self.assertEqual(cfg.estimator_options["twirling"]["shots_per_randomization"], 128)
        self.assertEqual(cfg.estimator_options["dynamical_decoupling"]["sequence_type"], "XY4")
        self.assertIs(cfg.estimator_options["resilience"]["measure_mitigation"], True)
        self.assertIsNone(cfg.initial_state)
        self.assertIsNone(cfg.observables)

    def test_input_passthrough_options_and_reserved_keys(self):
        """Passthrough dicts accept any field except the ones the function owns."""
        # batches is a validated top-level field.
        with self.assertRaises(ValidationError):
            InputModel(**self._args(batches=0))
        cfg = InputModel(
            **self._args(
                batches=3,
                transpiler_options={
                    "optimization_level": 1,
                    "seed_transpiler": 42,
                    "routing_method": "sabre",
                },
                estimator_options={"default_shots": 4096, "resilience_level": 2},
            )
        )
        self.assertEqual(cfg.batches, 3)
        self.assertEqual(cfg.transpiler_options["routing_method"], "sabre")
        self.assertEqual(cfg.estimator_options, {"default_shots": 4096, "resilience_level": 2})
        # backend/target are owned by the execution path, not transpiler_options.
        with self.assertRaises(ValidationError):
            InputModel(**self._args(transpiler_options={"backend": "x"}))
        with self.assertRaises(ValidationError):
            InputModel(**self._args(transpiler_options={"target": "x"}))

    def test_trotter_options_method_and_reserved(self):
        """`method` selects the product formula; its kwargs live in synthesis_settings."""
        cfg = InputModel(
            **self._args(
                trotter_options={
                    "method": "suzuki",
                    "synthesis_settings": {"order": 4, "insert_barriers": True},
                }
            )
        )
        self.assertEqual(cfg.trotter_options["method"], "suzuki")
        self.assertEqual(cfg.trotter_options["synthesis_settings"]["order"], 4)
        # lie is accepted (LieTrotter takes no order, so synthesis_settings may be omitted).
        InputModel(**self._args(trotter_options={"method": "lie"}))
        # unknown method rejected (qdrift not supported yet).
        with self.assertRaises(ValidationError):
            InputModel(**self._args(trotter_options={"method": "qdrift"}))
        # method-specific kwargs must live under 'synthesis_settings', not at the top level.
        with self.assertRaises(ValidationError):
            InputModel(**self._args(trotter_options={"method": "suzuki", "order": 2}))
        # reps (= step count) and time (= dt x step count) are owned by the function,
        # so both are rejected inside synthesis_settings.
        with self.assertRaises(ValidationError):
            InputModel(
                **self._args(
                    trotter_options={"method": "suzuki", "synthesis_settings": {"reps": 5}}
                )
            )
        with self.assertRaises(ValidationError):
            InputModel(
                **self._args(
                    trotter_options={"method": "suzuki", "synthesis_settings": {"time": 1.0}}
                )
            )

    def test_aqc_optimizer_settings_passthrough_and_reserved(self):
        """optimizer_settings is a scipy.optimize.minimize passthrough."""
        cfg = InputModel(
            **self._args(
                aqc_options={
                    "optimizer_settings": {
                        "method": "CG",
                        "jac": True,
                        "tol": 1e-9,
                        "options": {"maxiter": 50},
                    }
                }
            )
        )
        self.assertEqual(cfg.aqc_options.optimizer_settings["method"], "CG")
        self.assertEqual(cfg.aqc_options.optimizer_settings["tol"], 1e-9)
        self.assertEqual(cfg.aqc_options.optimizer_settings["options"]["maxiter"], 50)
        # the objective and initial parameters are fixed by the function -> fun/x0 rejected.
        with self.assertRaises(ValidationError):
            InputModel(**self._args(aqc_options={"optimizer_settings": {"fun": "x"}}))
        with self.assertRaises(ValidationError):
            InputModel(**self._args(aqc_options={"optimizer_settings": {"x0": [0.0]}}))
        # fidelity_target's early-stop owns the callback slot -> rejected only when both set.
        with self.assertRaises(ValidationError):
            InputModel(
                **self._args(
                    aqc_options={
                        "fidelity_target": 0.99,
                        "optimizer_settings": {"callback": lambda *_: None},
                    }
                )
            )
        # without fidelity_target, a user callback passes straight through.
        cfg = InputModel(
            **self._args(aqc_options={"optimizer_settings": {"callback": lambda *_: None}})
        )
        self.assertTrue(callable(cfg.aqc_options.optimizer_settings["callback"]))
