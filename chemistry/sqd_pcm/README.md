# SQD IEF-PCM Template

[Download SQD IEF-PCM template](https://ibm.biz/sqd-pcm-template)

### Table of contents

* [About](#about)
* [Methodology](#methodology)
* [Workflow sections](#workflow-sections)
* [Dependencies](#dependencies)
* [References](#references)

----------------------------------------------------------------------------------------------------
### About

This template, developed in collaboration with the Cleveland Clinic Foundation, encapsulates a workflow for quantum-centric implicit solvent simulations [1] of ground state energy in molecular systems. These simulations are based on the sample-based quantum diagonalization (SQD) method [2-6] and the integral equation formalism polarizable continuum model (IEF-PCM) of solvent [7]. Inclusion of the solute−solvent interactions in simulations of electronic structure is critical for biochemical and medical applications, but it was not previously available within the formalism of SQD method. The SQD IEF-PCM technique allowed for this critical feature in quantum-centric simulations of electronic structure in molecular systems. Similarly to standard SQD [2-3], the SQD IEF-PCM technique can be run on current quantum computers and it was shown to scale to over 50 qubits [1]. The SQD IEF-PCM workflow is enabled through the interface between the Qiskit Addon: SQD [8] and [Solvent module](https://pyscf.org/user/solvent.html) of PySCF [9]. The local unitary cluster Jastrow (LUCJ) ansatz is defined with FFSIM package [10]. 

----------------------------------------------------------------------------------------------------
### Methodology

The SQD IEF-PCM simulations start in a similar manner to the standard SQD method where the LUCJ quantum circuit is executed to sample a set of computational basis states from the probability distribution of the molecular system in the gas phase. The parametrization of the LUCJ ansatz is derived from a classical gas-phase restricted closed-shell coupled cluster singles and doubles (CCSD) calculations, which is a standard strategy in SQD studies [2].

 During the execution of LUCJ ansatz, the quantum computer introduces the noise, producing the noise-corrupted samples with broken particle-number and spin-z symmetries. To restore the particle-number and spin-z symmetries of these noise-corrupted samples the SQD employs an iterative self-consistent configuration recovery (S-CORE) procedure. The S-CORE utilizes a fixed set of computational basis states sampled from a quantum computer and an approximation to the ground-state occupation number distribution to flip the entries of the computational basis states, which is followed by the projection and diagonalization of the Hamiltonian in the subspace spanned by the samples corrected with S-CORE procedure. 

The key difference between SQD and SQD IEF-PCM lays within the diagonalization step. While the standard SQD utilizes the Hamiltonian of the solute in gas phase, the SQD IEF-PCM introduces the solute−solvent interaction potential in the diagonalization step.  It is important to note that in the case of SQD IEF-PCM the average orbital occupancies obtained during each iteration of the S-CORE step are calculated in the presence of a solute−solvent interaction potential. Hence, even though the initial electron configuration distribution is obtained from the gas-phase LUCJ ansatz, in addition to recovering the particle number, the S-CORE procedure brings the final electron configuration distribution closer to the true electron configuration distribution in solvent.

----------------------------------------------------------------------------------------------------
### Workflow sections

The SQD IEF-PCM workflow as described above is split into the three sections:

***1. Input section***

This section takes as an input the geometry of the molecule, selected active space, solvation model, LUCJ options, and SQD options. Subsequently it produces the [PySCF Checkpoint file](https://github.com/pyscf/pyscf.github.io/blob/master/examples/misc/02-chkfile.py) which contains the Hartree-Fock (HF) IEF-PCM data. This data will be used in the SQD portion of the workflow. For LUCJ portion of the workflow the input section also generates the gas-phase HF data which is stored internally in [PySCF FCIDUMP format](https://github.com/pyscf/pyscf.github.io/blob/master/examples/tools/01-fcidump.py). 

***2. LUCJ section***

It takes as the input the information from the HF gas-phase simulation as well as the definition of the active space. Importantly, it also utilizes the user-defined  information from the input section concerning the [error suppresion](https://docs.quantum.ibm.com/guides/configure-error-suppression), [number of shots](https://docs.quantum.ibm.com/api/qiskit/0.43/qiskit.primitives.Sampler), [circuit transpiler optimization level](https://docs.quantum.ibm.com/guides/set-optimization), and [the qubit layout](https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.TranspileLayout).

It generates one-electron and two-electron integrals within the defined active space. The integrals are then used to perform classical CCSD calculations, which give us t2 amplitudes that we use for parametrization of the LUCJ circuit. The LUCJ calculations return the bitstrings for each measurement, where these bitstrings correspond to electron configurations of the studied system.

***3. SQD section***

It takes as the input the [PySCF Checkpoint file](https://github.com/pyscf/pyscf.github.io/blob/master/examples/misc/02-chkfile.py) containing the HF IEF-PCM information, the bitstrings representing the electron configurations predicted by LUCJ, as well as the user-defined SQD options selected in the input section. As an output it produces the SQD IEF-PCM total energy of the lowest energy batch as well as the corresponding solvation free energy. 


## Dependencies

Default:
```
qiskit-ibm-runtime
qiskit-serverless
````

Custom:
```
ffsim==0.0.54
pyscf==2.9.0
qiskit_addon_sqd==0.10.0
```

----------------------------------------------------------------------------------------------------
### References

[1] Danil Kaliakin, Akhil Shajan, Fangchun Liang, and Kenneth M. Merz Jr. [Implicit Solvent Sample-Based Quantum Diagonalization](https://pubs.acs.org/doi/10.1021/acs.jpcb.5c01030), The Journal of Physical Chemistry B, 2025, DOI: 10.1021/acs.jpcb.5c01030

[2] Javier Robledo-Moreno, et al., [Chemistry Beyond Exact Solutions on a Quantum-Centric Supercomputer](https://arxiv.org/abs/2405.05068), arXiv:2405.05068 [quant-ph].

[3] Jeffery Yu, et al., [Quantum-Centric Algorithm for Sample-Based Krylov Diagonalization](https://arxiv.org/abs/2501.09702), arXiv:2501.09702 [quant-ph].

[4] Keita Kanno, et al., [Quantum-Selected Configuration Interaction: classical diagonalization of Hamiltonians in subspaces selected by quantum computers](https://arxiv.org/abs/2302.11320), arXiv:2302.11320 [quant-ph].

[5] Kenji Sugisaki, et al., [Hamiltonian simulation-based quantum-selected configuration interaction for large-scale electronic structure calculations with a quantum computer](https://arxiv.org/abs/2412.07218), arXiv:2412.07218 [quant-ph].

[6] Mathias Mikkelsen, Yuya O. Nakagawa, [Quantum-selected configuration interaction with time-evolved state](https://arxiv.org/abs/2412.13839), arXiv:2412.13839 [quant-ph].

[7] Herbert, John M. [Dielectric continuum methods for quantum chemistry. WIREs Computational Molecular Science](https://wires.onlinelibrary.wiley.com/doi/10.1002/wcms.1519), 2021, 11, 1759-0876.

[8] Saki, A. A.; Barison, S.; Fuller, B.; Garrison, J. R.; Glick, J. R.; Johnson, C.; Mezzacapo, A.; Robledo-Moreno, J.; Rossmannek, M.; Schweigert, P. et al. Qiskit addon: sample-based quantum diagonalization, 2024; https://github.com/Qiskit/qiskit-addon-sqd

[9] Asun, Q.; Zhang, X.; Banerjee, S.; Bao, P.; Barbry, M.; Blunt, N. S.; Bogdanov, N. A.; Booth, G. H.; Chen, J.; Cui, Z.-H. PySCF: Python-based Simulations of Chemistry Framework, 2025; https://github.com/pyscf/pyscf

[10] Kevin J. Sung; et al., FFSIM: Faster simulations of fermionic quantum circuits, 2024. https://github.com/qiskit-community/ffsim
