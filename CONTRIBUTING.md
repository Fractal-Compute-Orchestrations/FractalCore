# Contributing to FractalCore

First off, thank you for considering contributing to FractalCore! It's people like you that make FractalCore such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by the [FractalCore Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues as it might be a known problem. When you are creating a bug report, please include as many details as possible:

* **Use a clear and descriptive title** for the issue to identify the problem.
* **Describe the exact steps which reproduce the problem** in as many details as possible.
* **Provide specific examples to demonstrate the steps**.
* **Describe the behavior you observed after following the steps** and point out what exactly is the problem with that behavior.
* **Explain which behavior you expected to see instead and why.**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* **Use a clear and descriptive title** for the issue to identify the suggestion.
* **Provide a step-by-step description of the suggested enhancement** in as many details as possible.
* **Explain why this enhancement would be useful** to most FractalCore users.

### Pull Requests

* Fill in [the pull request template](.github/PULL_REQUEST_TEMPLATE.md).
* Follow the Python style guide (PEP 8).
* Ensure the CI tests pass.
* Include tests for any new features or bug fixes.
* Document any changes to the API or configuration.

## Styleguides

### Python Styleguide

All Python code should be formatted with `black` and linted with `flake8`.

```bash
# Format code
black .

# Lint code
flake8 .
```

### Git Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

## Development Setup

1. Fork the repository.
2. Clone your fork: `git clone https://github.com/your-username/FractalCore.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate it: `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
5. Install dependencies: `pip install -r requirements.txt`
6. Run the server: `python src/fractal_server/server.py`
