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

"""Resilience options."""

from typing import Union

from pydantic import Field

from .utils import Unset, UnsetType, BaseOptionsModel
from .zne_options import ZneOptions
from .pec_options import PecOptions


class ResilienceOptions(BaseOptionsModel):
    """Resilience option."""

    measure_mitigation: Union[UnsetType, bool] = Unset
    r"""measure_mitigation: Whether to enable measurement error mitigation method."""

    zne_mitigation: Union[UnsetType, bool] = Unset
    r"""Whether to turn on Zero Noise Extrapolation error mitigation method."""

    zne: ZneOptions = Field(default_factory=ZneOptions)
    r"""Additional zero noise extrapolation mitigation options."""

    pec_mitigation: Union[UnsetType, bool] = Unset
    r"""Whether to turn on Probabilistic Error Cancellation error mitigation method."""

    pec: PecOptions = Field(default_factory=PecOptions)
    r"""pec: Additional probabalistic error cancellation mitigation options."""
