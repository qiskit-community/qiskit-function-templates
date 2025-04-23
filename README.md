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

This repository hosts Qiskit Function templates and template implementations powered by [`qiskit`](https://github.com/Qiskit/qiskit) and [`qiskit-serverless`](https://github.com/Qiskit/qiskit-serverless) that you can use as reference for developing custom [Qiskit Functions](https://docs.quantum.ibm.com/guides/functions), or to explore how to leverage specific function implementations in your own project.

#### Templates
Qiskit Function *templates* allow to kickstart the [Qiskit Function](https://docs.quantum.ibm.com/guides/functions) development process with best-practices in interface development, code formatting, unit testing, and more. The repository itself is exposed as a [GitHub template](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-template-repository), so the CI/CD implementation can be easily extended by implementers to match new projects.

There are currently two templates: a circuit function template and an application function template.

<img src="image.png" alt="image" width="500"/>

#### Template Implementations
Qiskit Function *template implementations* show different realizations of the application function template above that leverage [Qiskit Addons](https://docs.quantum.ibm.com/guides/addons) to run industry-relevant application workflows. These implementations are structured by application area, currently:

- `physics`
- `chemistry`

Each implementation is contained in a directory in the corresponding application area, for example, for the hamiltonian simulation function: `physics/hamiltonian_simulation/template.py`. The directory might contain additional files that can be uploaded to the serverless environment for the function execution.

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