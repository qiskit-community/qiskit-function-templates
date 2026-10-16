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
"""Unit tests for the GPU capability probe.

These exercise only the pure environment/filesystem logic (cache-dir selection
and the probe's result contract), so they run on any worker -- including the
CPU-only CI runner, where ``cupy``/``gpu4pyscf`` are absent and the probe must
report ``device_count == 0`` without raising.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source_files"))

# pylint: disable=wrong-import-position
import gpu_probe  # noqa: E402


_CACHE_VARS = ("CUPY_CACHE_DIR", "NUMBA_CACHE_DIR", "MPLCONFIGDIR", "XDG_CACHE_HOME")


class TestConfigureCacheDirs(unittest.TestCase):
    """configure_cache_dirs redirects library caches to a writable location."""

    def setUp(self):
        super().setUp()
        # Snapshot and clear the cache env vars so each test starts clean; restore
        # on teardown so tests do not leak environment into one another.
        self._saved = {var: os.environ.pop(var, None) for var in _CACHE_VARS}

    def tearDown(self):
        for var in _CACHE_VARS:
            os.environ.pop(var, None)
            if self._saved[var] is not None:
                os.environ[var] = self._saved[var]
        super().tearDown()

    def test_falls_back_to_tmp_without_function_mount(self):
        """With no writable /function_user_data, caches live under /tmp."""
        # The CI/test machine has no /function_user_data mount, so the fallback
        # branch is what runs here.
        info = gpu_probe.configure_cache_dirs()
        self.assertEqual(info["cache_base"], "/tmp")
        for var in _CACHE_VARS:
            self.assertTrue(info[var].startswith("/tmp"), f"{var}={info[var]}")

    def test_sets_all_four_cache_vars_in_environ(self):
        """Every cache var is exported so child libs (cupy/numba/mpl) see it."""
        gpu_probe.configure_cache_dirs()
        for var in _CACHE_VARS:
            self.assertIn(var, os.environ)
            self.assertTrue(os.path.isdir(os.environ[var]), f"{var} not created")

    def test_explicit_override_is_respected(self):
        """An env var set before the call wins (setdefault semantics)."""
        os.environ["CUPY_CACHE_DIR"] = "/tmp/skqd-test-explicit-cupy"
        info = gpu_probe.configure_cache_dirs()
        self.assertEqual(info["CUPY_CACHE_DIR"], "/tmp/skqd-test-explicit-cupy")

    def test_idempotent(self):
        """Calling twice yields the same paths (safe to call before each import)."""
        first = gpu_probe.configure_cache_dirs()
        second = gpu_probe.configure_cache_dirs()
        self.assertEqual(first, second)


class TestGpuCheck(unittest.TestCase):
    """gpu_check never raises and reports a CPU-only worker as device_count 0."""

    def test_reports_no_device_on_cpu_only_worker(self):
        result = gpu_probe.gpu_check()
        # On a CPU-only machine (no CUDA driver), cupy either fails to import or
        # finds no device; either way device_count is 0 and nothing raises.
        self.assertEqual(result["device_count"], 0)
        self.assertFalse(result["cupy_import"])
        self.assertFalse(result["gpu4pyscf_import"])

    def test_always_includes_the_contract_keys(self):
        result = gpu_probe.gpu_check()
        for key in ("cupy_import", "device_count", "gpu4pyscf_import", "cache_dirs"):
            self.assertIn(key, result)

    def test_has_gpu_is_false_on_cpu_only_worker(self):
        self.assertFalse(gpu_probe.has_gpu())


if __name__ == "__main__":
    unittest.main()
