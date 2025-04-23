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

The Qiskit Function Templates are a collection of open-source quantum application implementations powered by [`qiskit`](https://github.com/Qiskit/qiskit) and [`qiskit-serverless`](https://github.com/Qiskit/qiskit-serverless) that can be used as reference for future Qiskit Function implementations.

The repository serves two purposes:

1. It shows how to leverage combinations of Qiskit ecosystem tools (The Qiskit SDK, the Qiskit Serverless package, and different Qiskit Addons) to enable industry-relevant application workflows.
2. It kickstarts the Qiskit Function development process by sharing best-practices in interface development, testing, code formatting, CI/CD implementation... The repository is a GitHub template, which allows to easily use as reference for new repositories. 

The package is structured by application area, currently:

- `physics`
- `chemistry`

Each template serverless entry point is contained within a directory in the corresponding application area, for example, for the hamiltonian simulation template: `physics/hamiltonian_simulation/template.py`. The directory might contain additional files that can be uploaded to the serverless environment for the function execution.

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