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

"""Qiskit Serverless entry point — source-upload path.

This is the program the managed Qiskit Serverless gateway runs when the function
is invoked. It follows the standard serverless source-program contract:

    get_arguments()  ->  do work  ->  save_result()

The gateway executes this script top-to-bottom: read the arguments, run the
function, save the result. All of the function logic lives in
``DynamicsFunction`` (``source/app_function.py``), so the *same* ``arguments``
dict always produces the *same* result.

This directory is what gets uploaded as the function's ``working_dir``::

    QiskitFunction(
        title="aqc-dynamics-function",
        entrypoint="program.py",
        working_dir="./source_files/",
        dependencies=[...],            # see requirements.txt
    )
"""
import os
import signal
import tempfile

# Redirect numba's JIT cache to a writable directory. The managed worker's
# site-packages is read-only and there is no writable ~/.cache, so quimb's
# cache=True njit functions (e.g. quimb.core.threading_choose_num_blocks) fail
# with "cannot cache function: no locator available". Set before numba/quimb are
# imported (both load lazily, after this line). Honor an explicit override.
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "numba_cache"))

# pylint: disable=wrong-import-position
# Both imports have to follow the NUMBA_CACHE_DIR line above: importing the
# function package pulls in quimb, which reads the cache directory at import time.
from qiskit_serverless import get_arguments, save_result

from source.app_function import DynamicsFunction

# 1. Read the classical `arguments` dict the client passed to `fn.run(**arguments)`.
arguments = get_arguments()

# 2. Build + run the function. This installs a SIGTERM handler that cancels
#    in-flight runtime jobs if the serverless job is stopped (matters for
#    backend="runtime").
func = DynamicsFunction(**arguments)
signal.signal(signal.SIGTERM, func.signal_handler)
result = func.run()

# 3. Persist the classical result dict; the client fetches it via job.result().
save_result(result)
