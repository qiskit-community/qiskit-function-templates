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

"""Zero noise extrapolation mitigation options."""

from typing import Literal, Union
from collections.abc import Sequence

from pydantic import field_validator, model_validator

from .utils import BaseOptionsModel, UnsetType, Unset, skip_unset_validation

ExtrapolatorType = Literal[
    "linear",
    "exponential",
    "double_exponential",
    "polynomial_degree_1",
    "polynomial_degree_2",
    "polynomial_degree_3",
    "polynomial_degree_4",
    "polynomial_degree_5",
    "polynomial_degree_6",
    "polynomial_degree_7",
]


class ZneOptions(BaseOptionsModel):
    """Zero noise extrapolation mitigation options."""

    amplifier: Union[
        UnsetType,
        Literal["gate_folding", "gate_folding_front", "gate_folding_back", "pea"],
    ] = Unset
    r"""Which technique to use for amplifying noise."""

    noise_factors: Union[UnsetType, Sequence[float]] = Unset
    r"""Noise factors to use for noise amplification."""

    extrapolator: Union[UnsetType, ExtrapolatorType, Sequence[ExtrapolatorType]] = Unset
    r""""Extrapolator(s) to try (in order) for extrapolating to zero noise."""

    def _default_noise_factors(self) -> Sequence[float]:
        return (1, 1.5, 2, 2.5, 3) if self.amplifier == "pea" else (1, 3, 5)

    @classmethod
    def _default_extrapolator(cls) -> Sequence[ExtrapolatorType]:
        return ("exponential", "linear")

    @field_validator("noise_factors")
    @classmethod
    @skip_unset_validation
    def _validate_zne_noise_factors(cls, factors: Sequence[float]) -> Sequence[float]:
        """Validate noise_factors."""
        if any(i < 1 for i in factors):
            raise ValueError("noise_factors` option value must all be >= 1")
        return factors

    @model_validator(mode="after")
    def _validate_options(self) -> "ZneOptions":
        """Check that there are enough noise factors for all extrapolators."""
        noise_factors = (
            self.noise_factors if self.noise_factors != Unset else self._default_noise_factors()
        )
        extrapolator = (
            self.extrapolator if self.extrapolator != Unset else self._default_extrapolator()
        )

        required_factors = {
            "linear": 2,
            "exponential": 2,
            "double_exponential": 4,
        }
        for idx in range(1, 8):
            required_factors[f"polynomial_degree_{idx}"] = idx + 1

        extrapolators: Sequence = (
            [extrapolator]  # type: ignore[assignment]
            if isinstance(extrapolator, str)
            else extrapolator
        )
        for extrap in extrapolators:
            if len(noise_factors) < required_factors[extrap]:  # type: ignore[arg-type]
                raise ValueError(
                    f"{extrap} requires at least {required_factors[extrap]} noise_factors"
                )
        return self
