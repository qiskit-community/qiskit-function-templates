# This code is part of a Qiskit project.
#
# (C) Copyright IBM and Cleveland Clinic Foundation 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
SQD-PCM Function Template source code.
"""

from typing import Any

import os
import json
import logging
import time
import traceback
import numpy as np
from pathlib import Path
import ffsim

from pyscf import gto, scf, mcscf, ao2mo, tools, cc
from pyscf.lib import chkfile
from pyscf.mcscf import avas
from pyscf.solvent import pcm

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.primitives import BackendSamplerV2

from qiskit_addon_sqd.counts import counts_to_arrays
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.fermion import bitstring_matrix_to_ci_strs
from qiskit_addon_sqd.subsampling import postselect_and_subsample

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit_serverless import get_arguments, save_result, distribute_task, get

from .solve_solvent import solve_solvent

logger = logging.getLogger(__name__)


def run_function(
    backend_name: str,
    files_name: str,
    molecule: dict,
    solvent_options: dict,
    lucj_options: dict,
    sqd_options: dict,
    **kwargs,  # pylint: disable=unused-argument
) -> dict[str, Any]:
    """
    Main entry point for the SQD-PCM (Polarizable Continuum Model) workflow.

    This function encapsulates the end-to-end execution of the algorithm.

    Args:
        backend_name: Identifier for the target backend, required for all
            workflows that access IBM Quantum hardware.

        files_name: Must be specified. Name without the file extension.
            This name will be assigned for checkpoint, fcidump, and count_dictionary files
            as well as the name of the SQD results folder containing the
            "bitstring_matrix" and "sci_vector" for each batch.

        molecule: dictionary with molecule information:
            - "atom" (str): required field, follows pyscf specification for atomic geometry.
                For example, for methanol the value would be::

                    '''
                    O -0.04559 -0.75076 -0.00000;
                    C -0.04844 0.65398 -0.00000;
                    H 0.85330 -1.05128 -0.00000;
                    H -1.08779 0.98076 -0.00000;
                    H 0.44171 1.06337 0.88811;
                    H 0.44171 1.06337 -0.88811;
                    '''

            - "number_of_active_orb" (int): required field
            - "number_of_active_alpha_elec" (int): required field
            - "number_of_active_beta_elec" (int): required field
            - "basis" (str): optional field, default is "sto-3g"
            - "verbosity" (int): optional field, default is 0
            - "charge" (int): optional field, default is 0
            - "spin" (int): optional field, default is 0
            - "avas_selection" (list[str] | None): optional field, default is None

        solvent_options: dictionary with solvent options information:
            - "method" (str): required field. Method for computing solvent reaction field
                for the PCM. Accepted values are: "IEF-PCM", "COSMO",
                "C-PCM", "SS(V)PE", see https://manual.q-chem.com/5.4/topic_pcm-em.html
            - "eps" (float): required field. Dielectric constant of the solvent in the PCM.

        lucj_options: dictionary with lucj options information:
            - "optimization_level" (int): optional field, default is 2
            - "initial_layout" (list[int]): optional field, default is None
            - "dynamical_decoupling" (bool): optional field, default is True
            - "twirling" (bool): optional field, default is True
            - "number_of_shots" (int): optional field, default is 10000

        sqd_options: dictionary with sqd options information:
            - "sqd_iterations" (int): required field.
            - "number_of_batches" (int): required field.
            - "samples_per_batch" (int): required field.
            - "max_davidson_cycles" (int): required field.

        **kwargs
            Optional keyword arguments to customize behavior (e.g., alternate paths
            for local testing). Use documented keywords where possible to maintain clarity.
            The argument types must be serializable by qiskit-serverless.

    Returns:
        The function should return the execution results as a dictionary with string keys.
        This is to ensure compatibility with ``qiskit_serverless.save_result``.
    """

    # Preparation Step: Input validation.
    # Do this at the top of the function definition so it fails early if any required
    # arguments are missing or invalid.

    Path("./output_sqd_pcm").mkdir(exist_ok=True)
    datafiles_name = "./output_sqd_pcm/" + files_name

    # Molecule parsing
    # Required:
    geo = molecule["atom"]
    num_active_orb = molecule["number_of_active_orb"]
    num_active_alpha = molecule["number_of_active_alpha_elec"]
    num_active_beta = molecule["number_of_active_beta_elec"]
    # Optional:
    input_basis = molecule.get("basis", "sto-3g")
    input_verbosity = molecule.get("verbosity", 0)
    input_charge = molecule.get("charge", 0)
    input_spin = molecule.get("spin", 0)
    myavas = molecule.get("avas_selection", None)

    # Solvent options parsing
    myeps = solvent_options["eps"]
    mymethod = solvent_options["method"]

    # LUCJ options parsing
    opt_level = lucj_options.get("optimization_level", 2)
    initial_layout = lucj_options.get("initial_layout", None)
    use_dd = lucj_options.get("dynamical_decoupling", True)
    use_twirling = lucj_options.get("twirling", True)
    num_shots = lucj_options.get("number_of_shots", True)

    # SQD options parsing
    iterations = sqd_options["sqd_iterations"]
    n_batches = sqd_options["number_of_batches"]
    samples_per_batch = sqd_options["samples_per_batch"]
    max_davidson_cycles = sqd_options["max_davidson_cycles"]

    # kwarg parsing (local testing)
    testing_backend = kwargs.get("testing_backend", None)

    # --
    # Preparation Step: Qiskit Runtime & primitive configuration for
    # execution on IBM Quantum Hardware.

    if testing_backend is None:
        # Initialize Qiskit Runtime Service
        print("Starting runtime service")
        service = QiskitRuntimeService(channel="ibm_quantum")
        backend = service.backend(backend_name)
        print("backend", backend)

        # Set up sampler and corresponding options
        sampler = SamplerV2(backend)
        sampler.options.dynamical_decoupling.enable = use_dd
        sampler.options.twirling.enable_measure = False
        sampler.options.twirling.enable_gates = use_twirling
        sampler.options.default_shots = num_shots
    else:
        backend = testing_backend
        print(f"testing backend: {backend.name}")

        # Set up backend sampler.
        # This doesn't allow running with twirling and dd
        sampler = BackendSamplerV2(backend=testing_backend)

    # Once the preparation steps are completed, the algorithm can be structured following a
    # Qiskit Pattern workflow:
    # https://docs.quantum.ibm.com/guides/intro-to-patterns

    # --
    # Step 1: Map
    # In this step, input arguments are used to construct relevant quantum circuits and operators

    # Initialize the molecule object (pyscf)
    print("Initializing molecule object")
    mol = gto.Mole()
    mol.build(
        atom=geo,
        basis=input_basis,
        verbose=input_verbosity,
        charge=input_charge,
        spin=input_spin,
        symmetry=False,
    )  # Not tested for symmetry calculations

    cm = pcm.PCM(mol)
    cm.eps = myeps
    cm.method = mymethod

    mf = scf.RHF(mol).PCM(cm)
    # Generation of checkpoint file for the solute and solvent
    # which will be used reused in all subsequent sections
    checkpoint_file_name = str(datafiles_name + ".chk")
    mf.chkfile = checkpoint_file_name
    mf.kernel()

    # Read-in the information about the molecule
    mol = chkfile.load_mol(checkpoint_file_name)

    # Read-in RHF data
    scf_result_dic = chkfile.load(checkpoint_file_name, "scf")
    mf = scf.RHF(mol)
    mf.__dict__.update(scf_result_dic)

    # LUCJ uses isolated solute
    mf.kernel()

    # Initialize orbital selection based on user input
    if myavas is not None:
        orbs = myavas
        avas_out = avas.AVAS(mf, orbs, with_iao=True)
        avas_out.kernel()
        ncas, nelecas = (avas_out.ncas, avas_out.nelecas)
    else:
        ncas = num_active_orb
        nelecas = (
            num_active_alpha,
            num_active_beta,
        )

    # LUCJ Step:
    # Generate active space
    mc = mcscf.CASCI(mf, ncas=ncas, nelecas=nelecas)
    if myavas is not None:
        mc.mo_coeff = avas_out.mo_coeff
    mc.batch = None
    # Reliable and most convinient way to do the CCSD on only the active space
    # is to create the FCIDUMP file and then run the CCSD calculation only on the
    # orbitals stored in the FCIDUMP file.

    h1e_cas, ecore = mc.get_h1eff()
    h2e_cas = ao2mo.restore(1, mc.get_h2eff(), mc.ncas)

    fcidump_file_name = str(datafiles_name + ".fcidump.txt")
    tools.fcidump.from_integrals(
        fcidump_file_name,
        h1e_cas,
        h2e_cas,
        ncas,
        nelecas,
        nuc=ecore,
        ms=0,
        orbsym=[1] * ncas,
    )

    print("Performing CCSD")
    # Read FCIDUMP and perform CCSD on only active space
    mf_as = tools.fcidump.to_scf(fcidump_file_name)
    mf_as.kernel()

    mc_cc = cc.CCSD(mf_as)
    mc_cc.kernel()
    mc_cc.t1
    t2 = mc_cc.t2

    n_reps = 2
    norb = ncas

    if myavas is not None:
        nelec = (int(nelecas / 2), int(nelecas / 2))
    else:
        nelec = nelecas

    alpha_alpha_indices = [(p, p + 1) for p in range(norb - 1)]
    alpha_beta_indices = [(p, p) for p in range(0, norb, 4)]

    print("Same spin orbital connections:", alpha_alpha_indices)
    print("Opposite spin orbital connections:", alpha_beta_indices)

    # Construct LUCJ op
    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2, n_reps=n_reps, interaction_pairs=(alpha_alpha_indices, alpha_beta_indices)
    )
    # Construct circuit
    qubits = QuantumRegister(2 * norb, name="q")
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)
    circuit.measure_all()

    # --
    # Step 2: Optimize
    # Transpile circuits to match ISA

    pass_manager = generate_preset_pass_manager(
        optimization_level=opt_level,
        backend=backend,
        initial_layout=initial_layout,
    )

    pass_manager.pre_init = ffsim.qiskit.PRE_INIT
    transpiled = pass_manager.run(circuit)

    print("Optimization level ", opt_level, transpiled.count_ops(), transpiled.depth())

    two_q_depth = transpiled.depth(lambda x: x[0].name == "ecr")
    print("Two-qubit gate depth:", two_q_depth)

    # --
    # Step 3: Execute on Hardware
    # Submit the underlying Sampler job. Note that this is not the
    # actual function job.

    # Submit the LUCJ job
    job = sampler.run([transpiled])
    print(f">>> Job ID: {job.job_id()}")
    print(f">>> Job Status: {job.status()}")

    # Wait until job is complete
    result = job.result()
    pub_result = result[0]
    counts = pub_result.data.meas.get_counts()

    # Save the count dictionary into a file
    count_dict_file_name = str(datafiles_name + ".count_dict.txt")
    with open(count_dict_file_name, "w") as output:
        print(counts, file=output)

    # --
    # Step 4: Post-process

    # SQD-PCM section
    start = time.time()

    # Orbitals, electron, and spin initialization
    num_orbitals = ncas
    if myavas is not None:
        num_elec_a = num_elec_b = int(nelecas / 2)
    else:
        num_elec_a, num_elec_b = nelecas
    spin_sq = input_spin

    # read LUCJ samples from count_dict
    count_dict_file_name = str(datafiles_name + ".count_dict.txt")
    with open(count_dict_file_name, "r") as file:
        count_dict_string = file.read().replace("\n", "")

    counts_dict = json.loads(count_dict_string.replace("'", '"'))

    # Convert counts into bitstring and probability arrays
    bitstring_matrix_full, probs_arr_full = counts_to_arrays(counts_dict)

    # Assing checkpoint file name based on "datafiles_name" user input
    checkpoint_file_name = str(datafiles_name + ".chk")

    # We set qiskit_serverless to explicitly reserve 1 cpu per thread, as
    # the task is CPU-bound and might degrade in performance when sharing
    # a core at scale (this might not be the case with smaller examples)
    @distribute_task(target={"cpu": 1})
    def solve_solvent_parallel(
        batches,
        myeps,
        mysolvmethod,
        myavas,
        num_orbitals,
        spin_sq,
        max_davidson,
        checkpoint_file,
    ):
        return solve_solvent(  # sqd for pyscf
            batches,
            myeps,
            mysolvmethod,
            myavas,
            num_orbitals,
            spin_sq=spin_sq,
            max_davidson=max_davidson,
            checkpoint_file=checkpoint_file,
        )

    # Code below checks if the folder with name matching "datafiles_name" already exists.
    # If folder does not exist yet, the code block below creates it.
    def create_directory_if_not_exists(path):
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
        else:
            print(f"Directory already exists: {path}")

    folder_name = str(molecule["datafiles_name"] + "_results")
    create_directory_if_not_exists(folder_name)

    e_hist = np.zeros((iterations, n_batches))  # energy history
    s_hist = np.zeros((iterations, n_batches))  # spin history
    g_solv_hist = np.zeros((iterations, n_batches))  # g_solv history
    occupancy_hist = []
    avg_occupancy = None

    for i in range(iterations):
        print(f"Starting configuration recovery iteration {i}")
        # On the first iteration, we have no orbital occupancy information from the
        # solver, so we begin with the full set of noisy configurations.
        if avg_occupancy is None:
            bs_mat_tmp = bitstring_matrix_full
            probs_arr_tmp = probs_arr_full

        # If we have average orbital occupancy information, we use it to refine the full
        # set of noisy configurations
        else:
            bs_mat_tmp, probs_arr_tmp = recover_configurations(
                bitstring_matrix_full, probs_arr_full, avg_occupancy, num_elec_a, num_elec_b
            )

        # Create batches of subsamples. We post-select here to remove configurations
        # with incorrect hamming weight during iteration 0, since no config recovery was performed.
        batches = postselect_and_subsample(
            bs_mat_tmp,
            probs_arr_tmp,
            hamming_right=num_elec_a,
            hamming_left=num_elec_b,
            samples_per_batch=samples_per_batch,
            num_batches=n_batches,
        )

        # Run eigenstate solvers in a loop. This loop should be parallelized for larger problems.
        e_tmp = np.zeros(n_batches)
        s_tmp = np.zeros(n_batches)
        g_solvs_tmp = np.zeros(n_batches)
        occs_tmp = []
        coeffs = []

        res1 = []
        for j in range(n_batches):
            strs_a, strs_b = bitstring_matrix_to_ci_strs(batches[j])
            print(f"Batch {j} subspace dimension: {len(strs_a) * len(strs_b)}")

            res1.append(
                solve_solvent_parallel(
                    batches[j],
                    myeps,
                    mymethod,
                    myavas,
                    num_orbitals,
                    spin_sq=spin_sq,
                    max_davidson=max_davidson_cycles,
                    checkpoint_file=checkpoint_file_name,
                )
            )

        res = get(res1)

        for j in range(n_batches):
            energy_sci, coeffs_sci, avg_occs, spin, g_solv = res[j]
            e_tmp[j] = energy_sci
            s_tmp[j] = spin
            g_solvs_tmp[j] = g_solv
            occs_tmp.append(avg_occs)
            coeffs.append(coeffs_sci)

            # These lines of code save sci_vector and bitsring_matrix for all batches at
            # all of the iterations
            # np.savetxt(
            #     folder_name + "/address_list-for-iter-" + str(i) + "-batch-" + str(j) + ".txt",
            #     batches[j],
            # )
            # np.savetxt(
            #     folder_name + "/c-for-iter-" + str(i) + "-batch-" + str(j) + ".txt",
            #     coeffs_sci.amplitudes,
            # )

        # Combine batch results
        avg_occupancy = tuple(np.mean(occs_tmp, axis=0))

        # Track optimization history
        e_hist[i, :] = e_tmp
        s_hist[i, :] = s_tmp
        g_solv_hist[i, :] = g_solvs_tmp
        occupancy_hist.append(avg_occupancy)

        lowest_e_batch_index = np.argmin(e_hist[i, :])

        # These lines of code save sci_vector and bitsring_matrix only for
        # lowest energy batch on each iteration
        coeffs_sci_batch = coeffs[lowest_e_batch_index]
        np.savetxt(
            folder_name
            + "/bitstring_matrix_for_lowest_energy_batch_"
            + str(lowest_e_batch_index)
            + "_at_SQD_iteration_"
            + str(i)
            + ".txt",
            batches[lowest_e_batch_index],
        )
        np.savetxt(
            folder_name
            + "/sci_vector_for_lowest_energy_batch_"
            + str(lowest_e_batch_index)
            + "_at_SQD_iteration_"
            + str(i)
            + ".txt",
            coeffs_sci_batch.amplitudes,
        )

        print("Lowest energy batch: " + str(lowest_e_batch_index))
        print("Lowest energy value: " + str(np.min(e_hist[i, :])))
        print("Corresponding g_solv value: " + str(g_solv_hist[i, lowest_e_batch_index]))
        print("-----------------------------------")

        end = time.time()
        duration = end - start
        print("SCI_solver totally takes:", duration, "seconds")

        output = {
            "total_energy_hist": e_hist,
            "spin_squared_value_hist": s_hist,
            "solvation_free_energy_hist": g_solv_hist,
            "occupancy_hist": occupancy_hist,
            "lowest_energy_batch": lowest_e_batch_index,
            "lowest_energy_value": str(np.min(e_hist[i, :])),
            "solvation_free_energy": str(g_solv_hist[i, lowest_e_batch_index]),
            "sci_solver_total_duration": duration,
        }

        return output


def set_up_logger(my_logger: logging.Logger, level: int = logging.INFO) -> None:
    """Logger setup to communicate logs through serverless."""

    log_fmt = "%(module)s.%(funcName)s:%(levelname)s:%(asctime)s: %(message)s"
    formatter = logging.Formatter(log_fmt)

    # Set propagate to `False` since handlers are to be attached.
    my_logger.propagate = False

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    my_logger.addHandler(stream_handler)
    my_logger.setLevel(level)


# This is the section where `run_function` is called, it's boilerplate code and can be used
# without customization.
if __name__ == "__main__":
    # Use serverless helper function to extract input arguments,
    input_args = get_arguments()

    # Allow to configure logging level
    logging_level = input_args.get("logging_level", logging.INFO)
    set_up_logger(logger, logging_level)

    try:
        func_result = run_function(**input_args)
        # Use serverless function to save the results that
        # will be returned in the job.
        save_result(func_result)
    except Exception:
        save_result(traceback.format_exc())
        raise
