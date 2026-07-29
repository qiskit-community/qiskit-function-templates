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

"""Twirling options."""

from typing import Union

from .utils import BaseOptionsModel, UnsetType, Unset


class TwirlingOptions(BaseOptionsModel):
    """Twirling options.

    When either field is enabled, these options are forwarded to
    ``qiskit_ibm_runtime.executor_sampler``'s ``prepare``, which boxes each circuit (inserting
    BoxOp layers) and builds per-randomization samplex items. Those samplex items execute only
    on real hardware -- the local ``FakeExecutor`` used in the tests runs plain circuit items
    only and rejects samplex items.
    """

    enable_gates: Union[bool, UnsetType] = Unset
    r"""Whether to apply 2-qubit Clifford gate twirling."""

    enable_measure: Union[bool, UnsetType] = Unset
    r"""Whether to enable twirling of measurements."""
