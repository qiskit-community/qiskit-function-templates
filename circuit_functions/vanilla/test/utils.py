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
"""
Shared helpers for the Circuit Function Template tests: pub/input builders,
dict comparison, and a Cartesian-product ``@data`` decorator.
"""

import itertools

from ddt import data, unpack
from qiskit.circuit.random import random_circuit


def get_estimator_pub() -> tuple:
    """Return a single pub as a (circuit, observable) tuple."""
    circuit = random_circuit(num_qubits=2, depth=2, seed=42)
    observable = "Z" * circuit.num_qubits
    return (circuit, observable)


def get_inputs(backend_name=None, pubs=None, options=None):
    """Build a run_function kwargs dict, filling sensible defaults."""
    return {
        "backend_name": backend_name or "fake_manila",
        "pubs": pubs or [get_estimator_pub()],
        "options": options,
    }


def dict_partially_equal(dict1: dict, dict2: dict) -> bool:
    """Return True if every key in dict2 exists in dict1 with the same value."""
    for key, val in dict2.items():
        if isinstance(val, dict):
            if not dict_partially_equal(dict1.get(key, {}), val):
                return False
        elif key not in dict1 or val != dict1[key]:
            return False
    return True


class Case(dict):
    """A single ddt test case; carries ``__doc__``/``__name__`` for readable ids."""


def generate_cases(docstring, name=None, **kwargs):
    """Combine kwargs in Cartesian product and return a list of Case objects."""
    ret = []
    for values in itertools.product(*kwargs.values()):
        case = Case(zip(kwargs.keys(), values))
        if docstring is not None:
            setattr(case, "__doc__", docstring.format(**case))
        if name is not None:
            setattr(case, "__name__", name.format(**case))
        ret.append(case)
    return ret


def combine(**kwargs):
    """Decorator that expands kwargs as a Cartesian-product @data."""

    def deco(func):
        return data(*generate_cases(docstring=func.__doc__, **kwargs))(unpack(func))

    return deco
