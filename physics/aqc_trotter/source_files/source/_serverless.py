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

"""Compatibility layer over ``qiskit_serverless``.

The rest of the package imports ``update_status``, ``get_logger``,
``get_runtime_service``, ``save_result``, ``send_error`` and ``ServerlessError``
from here rather than from ``qiskit_serverless`` directly.

Each symbol is bound **independently**: the genuine one is used whenever the
installed ``qiskit_serverless`` exposes it, and an inert, standard-logging-based
stub is used otherwise. Binding per symbol rather than all-or-nothing means one
missing top-level export cannot force *every* helper to its stub — in particular
``update_status`` stays the real gateway callback whenever it exists, so job
sub-status reporting is never silently downgraded to a no-op.
"""
from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ── Fallback stubs ──────────────────────────────────────────────────────────
# Used only for the symbols the installed qiskit_serverless does not expose (see
# the per-symbol binding below). When qiskit_serverless is fully present, none of
# these are bound and behavior is exactly the genuine gateway one.


class _StubServerlessError(Exception):
    """Stand-in for :class:`qiskit_serverless.ServerlessError`.

    Mirrors the structured ``code``/``message``/``details`` shape so error
    handling code is identical on- and off-gateway.
    """

    def __init__(
        self, message: str = "", code: str | None = None, details: dict | None = None
    ) -> None:
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(f"[{code}] {message} :: {self.details}")


class _StubJob:  # noqa: D401 - status-name constants only
    """Status constants matching the serverless ``Job`` enum names."""

    OPTIMIZING_HARDWARE = "OPTIMIZING_HARDWARE"
    WAITING_QPU = "WAITING_QPU"
    EXECUTING_QPU = "EXECUTING_QPU"
    POST_PROCESSING = "POST_PROCESSING"


def _stub_get_logger(name: str | None = None):
    return logging.getLogger(name or "aqc-dynamics-function")


def _stub_update_status(status) -> None:
    logging.getLogger("aqc-dynamics-function").info("status -> %s", status)


def _stub_send_error(code=None, message="", exception=None, args=None) -> None:
    logging.getLogger("aqc-dynamics-function").warning(
        "send_error code=%s message=%s exception=%s args=%s",
        code,
        message,
        exception,
        args,
    )


def _stub_save_result(result) -> None:
    del result  # the stub discards it; only the gateway persists results
    logging.getLogger("aqc-dynamics-function").info("save_result called (no-op off-gateway)")


def _stub_get_runtime_service(*args, **kwargs):
    raise ServerlessError(
        code="9001",
        message=(
            "get_runtime_service() is unavailable: qiskit_serverless is not "
            "installed / not running inside a gateway. The 'runtime' execution "
            "backend requires the deployed environment; the 'statevector' and "
            "'fake' backends run in-process and need no gateway."
        ),
        details={"solution": "Run with backend='statevector' or backend='fake'."},
    )


# ── Per-symbol binding ──────────────────────────────────────────────────────
# HAS_SERVERLESS reflects whether qiskit_serverless imports at all; each helper is
# then bound to the genuine symbol when present, else to its stub above.
try:
    import qiskit_serverless as _qs  # type: ignore

    HAS_SERVERLESS = True
except Exception:  # pylint: disable=broad-exception-caught  # pragma: no cover
    # qiskit_serverless not installed at all, or unimportable in this environment.
    _qs = None  # pylint: disable=invalid-name
    HAS_SERVERLESS = False


def _bind(name: str, stub):
    return getattr(_qs, name, stub) if _qs is not None else stub


Job = _bind("Job", _StubJob)
ServerlessError = _bind("ServerlessError", _StubServerlessError)
update_status = _bind("update_status", _stub_update_status)
send_error = _bind("send_error", _stub_send_error)
save_result = _bind("save_result", _stub_save_result)
get_runtime_service = _bind("get_runtime_service", _stub_get_runtime_service)

# The real ``get_logger()`` takes no arguments; our call sites pass ``__name__``.
# Adapt it to drop the name so the same source runs both ways.
if _qs is not None and hasattr(_qs, "get_logger"):
    _real_get_logger = _qs.get_logger

    def get_logger(name: str | None = None):  # type: ignore
        """Gateway ``get_logger()``, adapted to accept and drop a module name."""
        del name
        return _real_get_logger()

else:
    get_logger = _stub_get_logger


__all__ = [
    "HAS_SERVERLESS",
    "Job",
    "ServerlessError",
    "get_logger",
    "get_runtime_service",
    "save_result",
    "send_error",
    "update_status",
]
