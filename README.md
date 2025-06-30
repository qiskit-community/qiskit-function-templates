![Platform](https://img.shields.io/badge/%F0%9F%92%BB%20Platform-Linux%20%7C%20macOS-informational)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-%E2%89%A5%202.0%20-%20%236133BD?logo=Qiskit)](https://github.com/Qiskit/qiskit)
<br />
[![License](https://img.shields.io/github/license/Qiskit/qiskit-addon-aqc-tensor?label=License)](LICENSE.txt)

# Qiskit Function Templates

### Table of contents

* [About](#about)
* [Contributing](#contributing)
* [License](#license)

----------------------------------------------------------------------------------------------------

### About

This repository hosts Qiskit Function templates and template implementations powered by [`qiskit`](https://github.com/Qiskit/qiskit) and [`qiskit-serverless`](https://github.com/Qiskit/qiskit-serverless).

#### Application Template Implementations

Qiskit Function *template implementations* are realistic examples of how to build utility-scale application workflows with [Qiskit](https://docs.quantum.ibm.com/guides) and [Qiskit addons](https://docs.quantum.ibm.com/guides/addons). These are prebuilt workflows that you can customize for your own research needs instead of starting from scratch. To simplify execution and parallelization, they are easily deployable to [Qiskit Serverless](https://docs.quantum.ibm.com/guides/serverless).

Implementations are organized by application area, which currently includes:

- `physics`
- `chemistry`

Each implementation is contained in a directory in the corresponding application area. For example, the Hamiltonian simulation function is contained in [`physics/hamiltonian_simulation`](https://github.com/qiskit-community/qiskit-function-templates/blob/main/physics/hamiltonian_simulation). The directory might contain additional files that can be uploaded to the serverless environment for the function execution.

For convenience, we provide direct download links for each application template directory:

- [Download Hamiltonian Simulation template](https://ibm.biz/ham-sim-template)
- [Download SQD IEF-PCM template](https://ibm.biz/sqd-pcm-template)

#### Base Templates

Want to build your own custom [Qiskit Function](https://docs.quantum.ibm.com/guides/functions)? The *base templates* are references you can use to kickstart the Qiskit Function development process with best-practices in interface development, code formatting, unit testing, and more. The repository itself is exposed as a [GitHub template](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-template-repository) so that the CI/CD implementation can be easily extended to match your projects.

There are currently two templates: a circuit function template and an application function template.

<!-- <img src="tools/image.png" alt="image" width="500"/> -->

----------------------------------------------------------------------------------------------------

### Documentation

All documentation is available at [TBD].

----------------------------------------------------------------------------------------------------

### Contributing

The source code is available [on GitHub](https://github.com/Qiskit/qiskit-function-templates).

The developer guide is located at [CONTRIBUTING.md](https://github.com/Qiskit/qiskit-function-templates/blob/main/CONTRIBUTING.md)
in the root of this project's repository.
By participating, you are expected to uphold Qiskit's [code of conduct](https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md).

We use [GitHub issues](https://github.com/Qiskit/qiskit-function-templates/issues/new/choose) for tracking requests and bugs.

----------------------------------------------------------------------------------------------------

### License

[Apache License 2.0](LICENSE.txt)
