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
"""GPU capability probe.

``qiskit-fermions`` has no GPU code path of its own (its FCI matvec is a native
Rust kernel that runs on CPU). The GPU acceleration in this demo comes from the
*base image*'s public GPU stack -- ``cupy-cuda12x`` and ``gpu4pyscf-cuda12x`` --
which the ``fleet-selector`` image bakes in and Code Engine's NVIDIA container
runtime lights up on a GPU compute profile.

This module mirrors the layered probe from ``fulqrum-gpu-function``: cheapest
check first, each step independently pass/fail, so the result localizes any
problem. It never raises -- a CPU-only worker simply reports
``device_count == 0`` and the caller falls back to the CPU path.

Cache directories: cupy/numba compile kernels and cache them under ``~/.<name>``
by default. The base image runs as uid 1000 with no writable HOME (HOME=/),
so those caches must be redirected or the import fails with
``PermissionError: '/.cupy'``. :func:`configure_cache_dirs` points them at the
function's writable COS mount (``/function_user_data``) when present -- so the
CUDA kernel cache persists across jobs and the first job's compile is reused --
and falls back to ``/tmp`` otherwise (local smoke test, or a job without that
mount). It MUST run before ``import cupy`` because cupy reads
``CUPY_CACHE_DIR`` at import time.
"""
from __future__ import annotations

import os

# The function-scoped writable mount (chown 1000:1000 in the base image);
# persists across jobs of the function. Job-scoped alternative: /job_user_data.
FUNCTION_USER_DATA = "/function_user_data"


def _writable(path: str) -> bool:
    return os.path.isdir(path) and os.access(path, os.W_OK)


def configure_cache_dirs() -> dict:
    """Redirect cupy/numba/matplotlib caches to a writable dir.

    Prefers the persistent function mount (``/function_user_data``); falls back
    to ``/tmp``. Returns the chosen base dir and per-library paths (for logging).
    Idempotent and safe to call before every cupy import.
    """
    base = FUNCTION_USER_DATA if _writable(FUNCTION_USER_DATA) else "/tmp"
    cache = os.path.join(base, ".cache")
    dirs = {
        "CUPY_CACHE_DIR": os.path.join(base, ".cupy"),
        "NUMBA_CACHE_DIR": os.path.join(base, ".numba"),
        "MPLCONFIGDIR": os.path.join(base, ".mpl"),
        "XDG_CACHE_HOME": cache,
    }
    for var, path in dirs.items():
        # Only set if unset, so an explicit env override still wins.
        os.environ.setdefault(var, path)
        try:
            os.makedirs(os.environ[var], exist_ok=True)
        except OSError:
            pass
    return {"cache_base": base, **{k: os.environ[k] for k in dirs}}


def gpu_check() -> dict:
    """Probe the base image's GPU stack.

    Returns a dict with, cheapest step first:

    1. ``cupy_import`` -- ``import cupy`` (the CUDA array library ships in the
       base image via ``cupy-cuda12x``).
    2. ``device_count`` -- ``cupy.cuda.runtime.getDeviceCount()``; ``> 0`` only
       when a GPU + driver are actually attached (a GPU compute profile),
       ``0`` on a CPU-only worker.
    3. ``gpu4pyscf_import`` -- ``import gpu4pyscf`` (GPU-accelerated PySCF, used
       for the classical mean-field/integral step on larger basis sets).

    All failures are caught and reported in ``error`` so this never breaks the
    surrounding job.
    """
    result: dict = {
        "cupy_import": False,
        "device_count": 0,
        "gpu4pyscf_import": False,
    }
    # Must run before importing cupy: it reads CUPY_CACHE_DIR at import time,
    # and the default ~/.cupy is unwritable as uid 1000 (HOME=/).
    result["cache_dirs"] = configure_cache_dirs()
    try:
        import cupy  # noqa: F401

        result["cupy_import"] = True
        result["cupy_version"] = getattr(cupy, "__version__", "unknown")

        # Calls into CUDA: number of visible devices (0 on a CPU-only worker).
        result["device_count"] = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as exc:  # pylint: disable=broad-exception-caught
        result["error"] = f"{type(exc).__name__}: {exc}"

    try:
        import gpu4pyscf  # noqa: F401

        result["gpu4pyscf_import"] = True
        result["gpu4pyscf_version"] = getattr(gpu4pyscf, "__version__", "unknown")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        result["gpu4pyscf_error"] = f"{type(exc).__name__}: {exc}"

    return result


def has_gpu() -> bool:
    """True when a CUDA device is actually attached to this worker."""
    return gpu_check().get("device_count", 0) > 0
