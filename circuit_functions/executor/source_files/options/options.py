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

"""Options for the Executor Function Template.

These options mirror the shape of the vanilla circuit-function options, but are scoped to
what a raw Executor supports. Notably there is no resilience (ZNE/PEC) sub-model: those are
estimator-server-side techniques and have no meaning for a raw sampler executor. The
mitigation levels here therefore top out at dynamical decoupling and twirling.
"""

from typing import Union

from pydantic import Field

from .utils import BaseOptionsModel, UnsetType, Unset, merge_options
from .dynamical_decoupling_options import DynamicalDecouplingOptions
from .twirling_options import TwirlingOptions

MIN_MITIGATION_LEVEL: int = 1
# ZNE/PEC (levels beyond twirling) are estimator-server-side and unavailable on a raw
# sampler executor, so the executor template caps mitigation at level 2.
MAX_MITIGATION_LEVEL: int = 2
DEFAULT_MITIGATION_LEVEL: int = 1

MIN_OPTIMIZATION_LEVEL: int = 0
MAX_OPTIMIZATION_LEVEL: int = 3
DEFAULT_OPTIMIZATION_LEVEL: int = 2


class Options(BaseOptionsModel):
    """IBM executor function options."""

    default_shots: Union[UnsetType, int] = Field(gt=0, default=Unset)
    r"""The default number of shots to use when a sampler PUB does not specify its own.

    Also used by the ``"estimator"`` mode as a floor: the precision-derived shot count is the
    primary driver there (see ``default_precision``)."""

    default_precision: Union[UnsetType, float] = Field(gt=0, default=Unset)
    r"""The default target precision for ``"estimator"`` mode, converted to a shot budget via
    ``shots = ceil(1 / precision**2)``."""

    max_execution_time: Union[UnsetType, int] = Field(gt=0, default=Unset)
    r"""Maximum execution time in seconds."""

    mitigation_level: int = Field(
        ge=MIN_MITIGATION_LEVEL,
        le=MAX_MITIGATION_LEVEL,
        default=DEFAULT_MITIGATION_LEVEL,
    )
    r"""How much resilience to build against errors."""

    optimization_level: Union[int, UnsetType] = Field(
        ge=MIN_OPTIMIZATION_LEVEL,
        le=MAX_OPTIMIZATION_LEVEL,
        default=DEFAULT_OPTIMIZATION_LEVEL,
    )
    r"""How much optimization to perform on the circuits."""

    dynamical_decoupling: DynamicalDecouplingOptions = Field(
        default_factory=DynamicalDecouplingOptions
    )
    r"""Suboptions for dynamical decoupling."""

    twirling: TwirlingOptions = Field(default_factory=TwirlingOptions)
    r"""Suboptions for Pauli twirling. Enabling twirling routes the program through the
    executor's samplex (boxed-layer) path, which only executes on real hardware."""

    @staticmethod
    def apply_mitigation_level(
        mitigation_level: int = DEFAULT_MITIGATION_LEVEL,
    ) -> dict:
        """Apply the appropriate options based on the mitigation level.

        .. list-table:: ``mitigation_level=1``: dynamical decoupling only
            :header-rows: 1
            :widths: 30 70
            * - Option Path
              - Value
            * - ``options.dynamical_decoupling.enable``
              - ``True``
        .. list-table:: ``mitigation_level=2``: DD + gate & measurement twirling
            :header-rows: 1
            :widths: 30 70
            * - Option Path
              - Value
            * - ``options.dynamical_decoupling.enable``
              - ``True``
            * - ``options.twirling.enable_gates``
              - ``True``
            * - ``options.twirling.enable_measure``
              - ``True``

        Unlike the estimator-based vanilla template there is no ZNE/PEC level: a raw sampler
        executor cannot amplify or cancel noise on its own, so the highest level simply adds
        twirling. Level 1 stays twirling-free so the default path builds plain circuit items
        (twirling produces samplex items that only run on hardware).

        Args:
            mitigation_level: The mitigation level to set options from.

        Return:
            A new options dictionary.
        """
        if mitigation_level not in range(MIN_MITIGATION_LEVEL, MAX_MITIGATION_LEVEL + 1):
            raise ValueError(
                f"Invalid mitigation level {mitigation_level}. "
                f"Valid range is {MIN_MITIGATION_LEVEL}-{MAX_MITIGATION_LEVEL}"
            )

        level1_methods = {
            "dynamical_decoupling": {"enable": True},
        }
        level2_extra = {
            "twirling": {"enable_gates": True, "enable_measure": True},
        }

        if mitigation_level >= 1:
            mit_methods = level1_methods
        if mitigation_level >= 2:
            mit_methods = merge_options(mit_methods, level2_extra)

        return mit_methods

    @staticmethod
    def get_transpilation_options(options: dict) -> dict:
        """Extract options for the transpiler."""
        return {"optimization_level": options.pop("optimization_level", DEFAULT_OPTIMIZATION_LEVEL)}

    @staticmethod
    def get_execution_options(options: dict) -> dict:
        """Extract the options consumed while building and running the quantum program.

        The transpiler-only and preset-selector keys are removed; what remains
        (default_shots, default_precision, max_execution_time, dynamical_decoupling,
        twirling) drives program construction and the Executor submission.
        """
        for del_key in ("optimization_level", "mitigation_level"):
            options.pop(del_key, None)
        return options
