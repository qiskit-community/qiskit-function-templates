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

import copy
from typing import Any, ClassVar
from collections.abc import Callable
import functools

from pydantic import ConfigDict, BaseModel


class BaseOptionsModel(BaseModel):
    """Common model class for options."""

    model_config = ConfigDict(
        strict=True,
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )


class UnsetType:
    """Class used to represent an unset field."""

    _instance: ClassVar["UnsetType"]

    def __repr__(self) -> str:
        return "Unset"

    def __new__(cls) -> "UnsetType":
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __bool__(self) -> bool:
        return False


Unset = UnsetType()


def merge_options(old_options: dict, new_options: dict) -> dict:
    """Merge new_options into old_options.

    Args:
        old_options: Old options to be updated with new_options.
        new_options: New options to merge.

    Returns:
        Merged dictionary.

    Raises:
        TypeError: if input type is invalid.
    """

    def _update_options(old: dict, new: dict) -> None:
        if not new:
            return

        # Update values of existing keys
        for key, val in old.items():
            if key in new.keys():
                if isinstance(val, dict):
                    _update_options(val, new.pop(key))
                else:
                    old[key] = new.pop(key)

        # Add new keys.
        for key in list(new.keys()):
            old[key] = new.pop(key)

    combined = copy.deepcopy(old_options)

    if not new_options:
        return combined
    new_options_copy = copy.deepcopy(new_options)

    _update_options(combined, new_options_copy)

    return combined


def skip_unset_validation(func: Callable) -> Callable:
    """Decorator used to skip unset value"""

    @functools.wraps(func)
    def wrapper(cls: Any, val: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(val, UnsetType):
            return val
        return func(cls, val, *args, **kwargs)

    return wrapper
