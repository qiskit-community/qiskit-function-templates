# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from typing import Union

from pydantic import Field

from qiskit_ibm_runtime.options import EstimatorOptions

from .utils import BaseOptionsModel, UnsetType, Unset
from .dynamical_decoupling_options import (
    DynamicalDecouplingOptions,
)
from .twirling_options import TwirlingOptions
from .resilience_options import ResilienceOptions
from .utils import merge_options

MIN_MITIGATION_LEVEL: int = 1
MAX_MITIGATION_LEVEL: int = 3
DEFAULT_MITIGATION_LEVEL: int = 1

MIN_OPTIMIZATION_LEVEL: int = 0
MAX_OPTIMIZATION_LEVEL: int = 3
DEFAULT_OPTIMIZATION_LEVEL: int = 2


class Options(BaseOptionsModel):
    """IBM circuit function options."""

    default_precision: Union[UnsetType, float] = Field(gt=0, default=Unset)
    r"""The default precision to use."""

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
    r"""Suboptions for Pauli twirling."""

    resilience: ResilienceOptions = Field(default_factory=ResilienceOptions)
    r"""Advanced resilience options to fine tune the resilience strategy."""

    @staticmethod
    def apply_mitigation_level(
        mitigation_level: int = DEFAULT_MITIGATION_LEVEL,
    ) -> dict:
        """Apply the appropriate options based on the mitigation level.

        .. list-table:: ``mitigation_level=1``: DD + measurement twirling + TREX
            :header-rows: 1
            :widths: 30 70
            * - Option Path
              - Value
            * - ``options.dynamical_decoupling.enable``
              - ``True``
            * - ``options.twirling.enable_measure``
              - ``True``
            * - ``options.resilience.measure_mitigation``
              - ``True``
        .. list-table:: ``mitigation_level=2``: DD + Twirling + TREX + ZNE via gate-folding
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
            * - ``options.resilience.measure_mitigation``
              - ``True``
            * - ``options.resilience.zne_mitigation``
              - ``True``
        .. list-table:: ``mitigation_level=3``: DD + Twirling + TREX + ZNE via PEA
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
            * - ``options.resilience.measure_mitigation``
              - ``True``
            * - ``options.resilience.zne_mitigation``
              - ``True``
            * - ``options.resilience.zne.amplifier``
              - ``pea``

        Args:
            mitigation_level: The mitigation level to set options from.

        Return:
            A new options instance.
        """
        if mitigation_level not in range(MIN_MITIGATION_LEVEL, MAX_MITIGATION_LEVEL + 1):
            raise ValueError(
                f"Invalid mitigation level {mitigation_level}. "
                f"Valid range is {MIN_MITIGATION_LEVEL}-{MAX_MITIGATION_LEVEL}"
            )

        level1_methods = {
            "dynamical_decoupling": {"enable": True},
            "twirling": {"enable_measure": True},
            "resilience": {"measure_mitigation": True},
        }
        level2_exrta = {
            "twirling": {"enable_gates": True},
            "resilience": {"zne_mitigation": True},
        }
        level3_extra = {"resilience": {"zne": {"amplifier": "pea"}}}

        if mitigation_level >= 1:
            mit_methods = level1_methods
        if mitigation_level >= 2:
            mit_methods = merge_options(mit_methods, level2_exrta)
        if mitigation_level >= 3:
            mit_methods = merge_options(mit_methods, level3_extra)

        return mit_methods

    @staticmethod
    def get_transpilation_options(options: dict) -> dict:
        """Extract options for the transpiler."""
        return {"optimization_level": options.pop("optimization_level", DEFAULT_OPTIMIZATION_LEVEL)}

    @staticmethod
    def get_estimator_options(options: dict) -> dict:
        """Extract options for the estimator."""
        for del_key in ("optimization_level", "mitigation_level"):
            options.pop(del_key, None)
        # Validate the options
        EstimatorOptions(**options)
        return options
