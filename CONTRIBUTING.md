# Contributing

We appreciate all kinds of help, so thank you! 🎉

You can contribute in many ways to this project.

## Issue reporting

This is a good point to start, when you find a problem please add
it to the [issue tracker](https://github.com/Qiskit/qiskit-function-templates/issues).
The ideal report should include the steps to reproduce it.

## Doubts solving

To help less advanced users is another wonderful way to start. You can
help us close some opened issues. This kind of ticket should be
labeled as `question`.

## Improvement proposal

If you have an idea for a new feature, please open a ticket labeled as
`enhancement`. If you could also add a piece of code with the idea
or a partial implementation, that would be awesome.

## Repository structure

This repository is organized into two main categories:

### Base Templates

Located in the [`base_templates/`](base_templates/) directory, these are **reference templates**
that provide a starting point for building custom Qiskit Functions. They demonstrate best practices
for interface development, code formatting, and structure:

- [`application_function_template.py`](base_templates/application_function_template.py) - Template for building application-level workflows
- [`circuit_function_template.py`](base_templates/circuit_function_template.py) - Template for building circuit-level functions

**Important**: Base templates are **not runnable** on their own. They serve as blueprints and
reference implementations. Currently, they do not have unit tests, but we welcome contributions
to improve testability (see [Testing base templates](#testing-base-templates) below).

### Runnable Template Implementations

These are **complete, self-contained implementations** organized by application area:

- **[`physics/`](physics/)** - Physics-related function templates
  - [`hamiltonian_simulation/`](physics/hamiltonian_simulation/) - Hamiltonian simulation implementation
  
- **[`chemistry/`](chemistry/)** - Chemistry-related function templates
  - [`sqd_pcm/`](chemistry/sqd_pcm/) - SQD IEF-PCM implementation

Each runnable template includes:
- Complete source code in `source_files/`
- Unit tests in `test/`
- `requirements.txt` with dependencies
- `README.md` with documentation
- Jupyter notebook (`deploy_and_run.ipynb`) demonstrating usage

These templates can be deployed to Qiskit Serverless and executed as-is, or customized for
specific research needs.

## Contributor License Agreement

We'd love to accept your code! Before we can, we have to get a few legal
requirements sorted out. By signing a Contributor License Agreement (CLA), we
ensure that the community is free to use your contributions.

When you contribute to the Qiskit Function Templates project with a new pull request,
a bot will evaluate whether you have signed the CLA. If required, the bot will
comment on the pull request, including a link to accept the agreement. The
[individual CLA](https://qiskit.org/license/qiskit-cla.pdf) document is
available for review as a PDF.

**Note**:
> If you work for a company that wants to allow you to contribute your work,
> then you'll need to sign a [corporate CLA](https://qiskit.org/license/qiskit-corporate-cla.pdf)
> and email it to us at qiskit@us.ibm.com.

## Good first contributions

You are welcome to contribute wherever in the code you want to, of course, but
we recommend taking a look at the "good first issue" label into the issues and
pick one. We would love to mentor you!

## Doc

Review the parts of the documentation regarding the new changes and update it
if it's needed.

## Pull requests

We use [GitHub pull requests](https://help.github.com/articles/about-pull-requests)
to accept contributions.

While not required, opening a new issue about the bug you're fixing or the
feature you're working on before you open a pull request is an important step
in starting a discussion with the community about your work. The issue gives us
a place to talk about the idea and how we can work together to implement it in
the code. It also lets the community know what you're working on, and if you
need help, you can reference the issue when discussing it with other community
and team members.

If you've written some code but need help finishing it, want to get initial
feedback on it prior to finishing it, or want to share it and discuss prior
to finishing the implementation, you can open a *Draft* pull request and prepend
the title with the **\[WIP\]** tag (for Work In Progress). This will indicate
to reviewers that the code in the PR isn't in its final state and will change.
It also means that we will not merge the commit until it is finished. You or a
reviewer can remove the [WIP] tag when the code is ready to be fully reviewed for merging.

### Before working on a pull request

1. **Ensure there is an issue**: Before starting work on a pull request, please ensure
   there is an associated issue. If one doesn't exist, create one to discuss the
   proposed changes with the maintainers.

2. **Check for existing work**: Search existing pull requests to make sure someone
   else isn't already working on the same thing.

3. **Discuss your approach**: For significant changes, discuss your implementation
   approach in the issue before starting work.

### Pull request checklist

When submitting a pull request and you feel it is ready for review,
please ensure that:

1. The code follows the code style of the project. For convenience, you can
   execute `tox -elint` locally, which will print out any style violations.
   See the [Code style](#code-style) section for more details.

2. The documentation has been updated accordingly. In particular, if a function
   or class has been modified during the PR, please update the docstring
   accordingly. See the [Documentation](#documentation) section for more details.

3. Your contribution passes the existing tests, and if developing a new feature,
   that you have added new tests that cover those changes. See the
   [Tests](#tests) section for more details.

4. You add a new line to the `CHANGELOG.md` file, in the `Unreleased` section,
   with a description of your change. If the `Unreleased` section doesn't exist,
   please create it.

5. All contributors have signed the CLA.

## Setting up the development environment

### Installation from source

We recommend using [Python virtual environments](https://docs.python.org/3/tutorial/venv.html)
to cleanly separate Qiskit from other applications and improve your experience.

The simplest way to use environments is by using the `venv` module, included with most Python installations.
To create a virtual environment, run:

```bash
python3 -m venv venv
```

Where `venv` is the name of the virtual environment. This will create a new folder with that name
that contains the files for the virtual environment.

To activate the environment on Linux or macOS, run:

```bash
source venv/bin/activate
```

Or on Windows:

```bash
venv\Scripts\activate.bat
```

Once the virtual environment is activated, you can install the development requirements:

```bash
pip install -r requirements-dev.txt
```

To install specific function template requirements, navigate to the template directory and install:

```bash
# For Hamiltonian Simulation
pip install -r physics/hamiltonian_simulation/requirements.txt

# For SQD PCM
pip install -r chemistry/sqd_pcm/requirements.txt
```

## Code style

The code style is enforced by [Black](https://black.readthedocs.io/en/stable/) and
[Pylint](https://pylint.pycqa.org/en/latest/). You can check your code by running:

```bash
tox -elint
```

This will run both Black and Pylint on the codebase. If there are any style violations,
Black will report them, and you can fix them by running:

```bash
tox -eblack
```

This will automatically format the code to match the project's style guidelines.

### Code style guidelines

- Line length is limited to 100 characters (Black) or 105 characters (Pylint)
- Use descriptive variable names
- Add docstrings to all public functions and classes
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use type hints where appropriate

## Tests

New features should be tested to ensure they work as expected and to prevent
future regressions. To run the tests, you can use `tox`:

```bash
# Run tests for Hamiltonian Simulation
tox -ehamsim

# Run tests for SQD PCM
tox -esqdpcm
```

### Writing tests

- Place test files in the `test/` directory within each function template
- Name test files with the prefix `test_`
- Use descriptive test function names that explain what is being tested
- Include both positive and negative test cases
- Test edge cases and error conditions
- Use appropriate assertions to verify expected behavior

## Documentation

Documentation is an important part of the project. When adding new features or
modifying existing ones, please update the documentation accordingly.

### Docstring format

We use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
Here's an example:

```python
def example_function(param1: int, param2: str) -> bool:
    """Brief description of the function.

    More detailed description if needed. This can span multiple lines
    and provide additional context about the function's behavior.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of the return value

    Raises:
        ValueError: Description of when this error is raised
    """
    pass
```

### README files

Each function template should have a `README.md` file that includes:

- Overview of the function
- Installation instructions
- Usage examples
- References to relevant papers or documentation

## Development workflow

### Contributing to runnable templates

When working on physics or chemistry templates:

1. **Fork the repository**: Create your own fork of the repository on GitHub

2. **Clone your fork**: Clone your fork to your local machine
   ```bash
   git clone https://github.com/YOUR_USERNAME/qiskit-function-templates.git
   cd qiskit-function-templates
   ```

3. **Create a branch**: Create a new branch for your changes
   ```bash
   git checkout -b my-feature-branch
   ```

4. **Install dependencies**: Install the specific template's requirements
   ```bash
   # For Hamiltonian Simulation
   pip install -r requirements-dev.txt
   pip install -r physics/hamiltonian_simulation/requirements.txt
   
   # For SQD PCM
   pip install -r requirements-dev.txt
   pip install -r chemistry/sqd_pcm/requirements.txt
   ```

5. **Make your changes**: Implement your changes in the template's `source_files/` directory

6. **Add tests**: Add or update tests in the template's `test/` directory

7. **Test your changes**: Run the tests to ensure everything works
   ```bash
   tox -elint
   tox -ehamsim  # or tox -esqdpcm for chemistry templates
   ```

8. **Update documentation**: Update the template's `README.md` and any relevant docstrings

9. **Commit your changes**: Commit your changes with a descriptive commit message
   ```bash
   git add .
   git commit -m "Add feature X to hamiltonian_simulation"
   ```

10. **Push to your fork**: Push your changes to your fork on GitHub
    ```bash
    git push origin my-feature-branch
    ```

11. **Open a pull request**: Go to the original repository and open a pull request
    from your branch

### Contributing to base templates

When improving base templates:

1. Follow steps 1-3 above to fork and create a branch

2. **Make your changes**: Edit files in the `base_templates/` directory
   - Ensure changes follow best practices and are well-documented
   - Consider how changes will affect users who copy these templates

3. **Test your changes**: Run linting to ensure code quality
   ```bash
   tox -elint
   ```

4. **Document your changes**: Update docstrings and add comments explaining the pattern

5. Follow steps 9-11 above to commit and create a pull request

### Adding a new template

To contribute a new runnable template:

1. Create a new directory under `physics/` or `chemistry/` (or propose a new category)

2. Structure your template following the existing pattern:
   ```
   your_template/
   ├── README.md
   ├── requirements.txt
   ├── deploy_and_run.ipynb
   ├── source_files/
   │   ├── __init__.py
   │   └── your_entrypoint.py
   └── test/
       ├── __init__.py
       └── test_your_template.py
   ```

3. Add a test environment to `tox.ini`:
   ```ini
   [testenv:yourtemplate]
   install_command = pip install -c{toxinidir}/constraints.txt -U {opts:} {packages:}
   setenv =
     VIRTUAL_ENV={envdir}
     LANGUAGE=en_US
     LC_ALL=en_US.utf-8
   deps =
       -r{toxinidir}/requirements-dev.txt
       -r{toxinidir}/category/your_template/requirements.txt
   commands =
     stestr --test-path category/your_template/test run {posargs}
   ```

4. Follow the contribution workflow above to submit your new template

## Code of Conduct

This project adheres to the Qiskit [Code of Conduct](https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.

## Questions?

If you have questions or need help, please:

- Open an issue on [GitHub](https://github.com/Qiskit/qiskit-function-templates/issues)
- Join the Qiskit Slack workspace
- Reach out to the maintainers

Thank you for contributing to Qiskit Function Templates! 🚀
