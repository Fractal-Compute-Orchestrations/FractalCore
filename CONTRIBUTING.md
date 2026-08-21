# Contributing to FractalCore

Thank you for contributing to FractalCore. This document outlines coding standards, development workflows, and pull request procedures.

---

## Code of Conduct

All contributors are expected to adhere to the [FractalCore Code of Conduct](CODE_OF_CONDUCT.md).

---

## Development Workflow

### 1. Environment Setup

```bash
# 1. Clone your fork
git clone https://github.com/your-username/FractalCore.git
cd FractalCore

# 2. Initialize virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install development and test dependencies
pip install -r requirements.txt
pip install pytest black flake8
```

### 2. Code Quality and Formatting Standards

FractalCore enforces strict automated linting and formatting standards. Before opening a pull request, all checks must pass locally:

- **Formatting (Black)**: Code must be formatted with `black`:
  ```bash
  black .
  black --check .
  ```
- **Linting (Flake8)**: Code must pass syntax and complexity checks:
  ```bash
  flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
  flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
  ```
- **Documentation Standard**: All markdown files must be technical, precise, and completely free of emojis.

### 3. Running Unit Tests

```bash
pytest tests/
```

---

## Pull Request Guidelines

1. Create a feature branch from `main`: `git checkout -b feature/your-feature-name`.
2. Maintain PEP 8 style conventions and add type annotations where applicable.
3. Include unit or integration tests for new features and bug fixes.
4. Ensure all CI checks pass.
5. Write concise, imperative commit messages (`feat: add asynchronous FedAvg queue`).
