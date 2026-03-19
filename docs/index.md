# c108

Curated core Python utilities with zero dependencies for introspection, formatting,
CLI, IO/streams, filesystem, validation, networking, numerics, and sentinels.

- **License**: MIT
- **Audience**: Python developers who prefer small, practical APIs

[![Docs](https://img.shields.io/badge/docs-readthedocs.io-blue)](https://c108.readthedocs.io/)
![Python Versions](https://img.shields.io/badge/python-3.10–3.14-green)
[![Codecov](https://codecov.io/gh/C108-Lab/c108/branch/main/graph/badge.svg)](https://codecov.io/gh/C108-Lab/c108)

## Installation

Install from PyPI:

```
shell
pip install c108
```

Install the latest from GitHub (main branch):

```
shell
pip install git+https://github.com/C108-Lab/c108.git
```

## Modules

- [c108.abc](api/abc.md) – Runtime introspection and type-validation utilities
- **c108.cli** – CLI helpers
- **c108.collections** – BiDirectionalMap collection
- **c108.dataclasses** – dataclasses tools
- **c108.dictify** – serialization utilities
- **c108.display** – value with units of measurement display
- **c108.formatters** – formatting utilities for development and debugging
- **c108.io** – streaming and chunking helpers (StreamingFile, etc.)
- **c108.json** – safe JSON file read/write/update with optional atomic operations
- **c108.network** – timeout estimators
- **c108.numeric** – std_numeric convertor
- **c108.os** – low-level filesystem/path helpers
- **c108.scratch** – scratch & temp file utilities
- **c108.sentinels** – sentinel types
- **c108.shutil** – high-level file utilities
- **c108.tools** – miscellaneous helpers
- **c108.unicode** – unicode text formatters
- **c108.utils** – shared utils
- **c108.validators** – common validation utilities

## Features

C108-Lab packages are:

- **Curated** – Centrally developed and maintained for consistency
- **Production-ready** – Thoroughly tested and documented
- **Dependency-conscious** – Core package stays lightweight; extra features and heavy deps live in sub-packages
- **Community-friendly** – Issues and feature requests are welcome

`c108` has **no external dependencies**, standard library only.

## Community & Contributing

While we don't accept pull requests, we warmly welcome:

- 🐛 **Bug reports**
- ✨ **Feature requests**
- 📖 **Documentation feedback**
- ❓ **Usage questions**

Please open an issue on GitHub for any of the above.

## Releases

- Tagged releases on GitHub
- PyPI is the source of truth
- conda-forge feedstock tracks PyPI

## License

[MIT License (full text)](https://github.com/C108-Lab/c108/blob/main/LICENSE)

## Development Reference

### Commands 🖥️

#### **1. Create dev environment locally**

```bash
uv venv                            # creates .venv
uv sync --extra dev                # sync with dev environment with optional ML and Scientific deps
uv sync --extra test --extra tools # sync with basic dev environment, no ML or Scientific deps
```

#### **2. Fromat** with `ruff`

```shell
ruff format c108 tests
```

#### **3. Run Tests** with `uv run COMMAND`

Unit tests only (the subset used in CI):

```bash
pytest
```

Integration tests only (run locally):

```bash
pytest -m "integration"
```

Specific integration module:

```shell
pytest tests/integration/test_numeric.py
```

Unit and Integration tests:

```bash
pytest -m "integration or not integration"
```

Doctests:

```bash
pytest --xdoctest c108
```

#### **4. Build and publish**

```bash
# Build wheel + sdist via Hatchling
uv build
# Publish to PyPI; secrets handled by CI
uv publish --token ${{ secrets.PYPI_TOKEN }}
```

### Test Structure ✅

- **Unit tests** (fast, minimal deps): live in `tests/` and are always run by CI.
- **Integration tests** (optional, heavy deps): live in `tests/integration/` and cover interactions with external
  packages such as NumPy, Pandas, PyTorch, TensorFlow, JAX, Astropy, and SymPy.

All integration tests use `pytest.importorskip()`,
automatically **skipped** if a dependency is missing.

### Test Dependencies

Integration tests use optional third‑party packages that are **not** required
by the core test suite:

| Package    | Supported Types            |
|------------|----------------------------|
| Astropy    | Physical `Quantity` types  |
| JAX        | DeviceArray scalars        |
| NumPy      | Numeric scalars and arrays |
| Pandas     | Nullable scalars/Series    |
| PyTorch    | Tensor dtypes              |
| SymPy      | Symbolic numeric support   |
| TensorFlow | Tensor dtypes              |

Install only what you need, for example:

```shell
pip install numpy pandas
```

### Continuous Integration

GitHub Actions runs only unit tests for performance and reliability.

Integration tests are intended for local verification before releasing major versions
or dependency interface changes.
