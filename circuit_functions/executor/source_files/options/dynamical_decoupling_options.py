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

"""Options for dynamical decoupling."""

from typing import Union, Literal

from .utils import BaseOptionsModel, UnsetType, Unset


class DynamicalDecouplingOptions(BaseOptionsModel):
    """Options for dynamical decoupling (DD)."""

    enable: Union[bool, UnsetType] = Unset
    r"""Whether to enable dynamical decoupling."""

    sequence_type: Union[Literal["XX", "XpXm", "XY4"], UnsetType] = Unset
    r""""Which dynamical decoupling sequence to use."""
