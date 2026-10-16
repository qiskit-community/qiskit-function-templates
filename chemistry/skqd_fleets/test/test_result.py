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
"""Unit tests for SKQDResult serialization.

``SKQDResult.to_dict`` is what feeds ``qiskit_serverless.save_result``, so the
result must be plain JSON-friendly types: Python ``float``/``int``, ``nelec`` as
a list (not a tuple), and empty lists for the optional fields when unset. Pure
logic -- no chemistry stack needed.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source_files"))

# pylint: disable=wrong-import-position
from skqd import SKQDResult  # noqa: E402


class TestSKQDResultToDict(unittest.TestCase):
    """to_dict coerces to serialization-friendly types."""

    def _make(self, **overrides):
        kwargs = dict(
            energy=-2.1,
            reference_energy=-2.2,
            error=0.1,
            krylov_dim=5,
            num_configurations=12,
            norb=4,
            nelec=(2, 2),
        )
        kwargs.update(overrides)
        return SKQDResult(**kwargs)

    def test_nelec_is_a_two_element_list(self):
        """nelec is stored as a tuple but must serialize as a JSON list."""
        d = self._make().to_dict()
        self.assertIsInstance(d["nelec"], list)
        self.assertEqual(d["nelec"], [2, 2])

    def test_scalar_types_are_native_python(self):
        d = self._make().to_dict()
        for key in ("energy", "reference_energy", "error"):
            self.assertIsInstance(d[key], float)
        for key in ("krylov_dim", "num_configurations", "norb"):
            self.assertIsInstance(d[key], int)

    def test_optional_lists_default_to_empty(self):
        d = self._make().to_dict()
        self.assertEqual(d["occupancies"], [])
        self.assertEqual(d["energy_history"], [])

    def test_list_fields_are_coerced_to_floats(self):
        d = self._make(occupancies=[1, 0], energy_history=[-2, -2.1]).to_dict()
        self.assertTrue(all(isinstance(x, float) for x in d["occupancies"]))
        self.assertTrue(all(isinstance(x, float) for x in d["energy_history"]))

    def test_round_trips_through_json(self):
        """The whole dict must be JSON-serializable (save_result requirement)."""
        import json

        d = self._make(occupancies=[0.5], energy_history=[-2.0]).to_dict()
        self.assertEqual(json.loads(json.dumps(d)), d)


if __name__ == "__main__":
    unittest.main()
