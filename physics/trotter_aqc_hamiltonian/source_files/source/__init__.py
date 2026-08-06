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

"""AQC-compressed Hamiltonian dynamics — an experiment-agnostic Qiskit application function.

Trotterize the time evolution of any 1-D nearest-neighbour Pauli Hamiltonian,
compress the leading steps with AQC (tensor-network MPS fidelity), and execute
with error mitigation, returning the expectation value of each observable at
each time step.

The stages are one module each, and run in this order:

* :mod:`hamiltonian` — the Hamiltonian spec and its Trotter evolution circuits
* :mod:`build` — initial state, Trotter targets, observables
* :mod:`aqc` — compression of the leading Trotter targets into shallow ansätze
* :mod:`execute` — PUBs to expectation values on the selected backend

:mod:`app_function` validates the input contract and orchestrates them.
"""
