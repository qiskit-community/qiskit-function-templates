# SQD-PCM Template

This template implementation, developed in collaboration with the Cleveland Clinic Foundation, encapsulates
an implementation of the SQD-PCM method from [1] to perform quantum-centric simulations of ground 
state energies in molecular systems. 

The workflow consists of 3 sections:

### 1. Input section

This section takes as an input the geometry of the molecule, selected orbitals, solvation model, LUCJ options, and SQD options. It defines the active space within the system.

Generates an intermediate "checkpoint" file (chk) containing the HF IEF-PCM information which will be used in the SQD portion of the workflow. For LUCJ portion of the code it also generates the gas-phase HF data. 

### 2. LUCJ section

It takes as the input the information from the HF gas-phase simulation as well as the definition of the active space. Importantly it also utilizes the user input (concerning error mitigation, number of shots, and etc.) that we initially defined in the input section.

It generates one-electron and two-electron integrals within the defined active space. The integrals are then used to perform CCSD calculations, which give us t2 amplitudes that we parametrize the LUCJ circuit with. The LUCJ calculations return the bitstrings for each measurement.

### 3. SQD section

It takes as the input the "checkpoint" file (chk) containing the HF IEF-PCM information; the `count_dict` containing the electron configurations predicted by LUCJ in form of the bitstrings; as well as the user-defined options selected in input section.

As an output it produces the SQD IEF-PCM total energy of the lowest energy batch as well as the corresponding solvation free energy. 

**References**

[1] https://arxiv.org/html/2502.10189v1

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