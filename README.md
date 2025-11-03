# c108

Curated core Python utilities with minimal dependencies — CLI, IO/streams, filesystem, validation, packaging/versioning,
markdown helpers, math, networking, tempfile, and small tools. Heavier integrations (YAML, Rich UI, Git, GCP, etc.) live
in optional package extras.

- **License**: MIT
- **Audience**: Python developers who prefer small, practical APIs

## Installation

```shell
# Core only (minimal dependencies)
pip install c108
```

Optional integrations are provided as Extension Packages to keep the core lean.

## Modules

- **c108.abc** – Runtime introspection and type-validation utilities
- **c108.cli** – CLI helpers
- **c108.collections** – BiDirectionalMap collection
- **c108.dictify** – serialization utilities
- **c108.display** – value with units of measurement display
- **c108.io** – stream and chunking helpers (StreamingFile, etc.)
- **c108.json** – safe JSON file read/write/update with optional atomic operations
- **c108.network** – bandwidth/time estimates helpers
- **c108.numeric** – std_numeric convertor
- **c108.os** – low-level filesystem/path helpers
- **c108.scratch** – scratch & temp file utilities
- **c108.sentinels** – sentinel types
- **c108.shutil** – high-level file utilities
- **c108.tools** – misc utility helpers
- **c108.unicode** – unicode text formatters
- **c108.utils** – shared utils
- **c108.validators** – common validation utilities
- **c108.zip** – tar/zip helpers

## Extension Packages

- **🚧 In progress**  

<!-- 

## Extension Packages

- **c108-gcp** – Google Cloud Platform utilities
- **c108-rich** – Rich formatting helpers
- **c108-yaml** – YAML utilities

```bash
# YAML Features
pip install c108-yaml
```
--> 


## Features

C108-Lab packages are:

- **Curated** – Centrally developed and maintained for consistency
- **Production-ready** – Thoroughly tested and documented
- **Dependency-conscious** – Core package stays lightweight; extra features and heavy deps live in sub-packages
- **Community-friendly** – Issues and feature requests are welcome

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

MIT License, see [full text](LICENSE).

## Developer & Test Notes

### Test Structure

- **Unit tests** (fast, minimal deps): live in `tests/` and are always run by CI.
- **Integration tests** (optional, heavy deps): live in `tests/integration/` and cover interactions with external
  packages such as NumPy, Pandas, PyTorch, TensorFlow, JAX, Astropy, and SymPy.

### Test Dependencies

Integration tests use optional third‑party packages that are **not** required 
by the core package test suite:

| Package      | Supported Types            |
|--------------|----------------------------|
| Astropy      | Physical `Quantity` types  |
| JAX          | DeviceArray scalars        |
| NumPy        | Numeric scalars and arrays |
| Pandas       | Nullable scalars/Series    |
| PyTorch      | Tensor dtypes              |
| SymPy        | Symbolic numeric support   |
| TensorFlow   | Tensor dtypes              |

Install only what you need, for example:

```shell
pip install numpy pandas
```

All integrations use `pytest.importorskip()`, automatically **skipped** 
if a dependency is missing.

### Running Tests

Unit tests (CI default):
```shell
pytest -m "not integration"
```

Integration tests only:
```shell
pytest -m integration
```

Full suite of tests:
```shell
pytest
```

Specific integration module:
```shell
pytest tests/integration/test_numeric.py
```

### Continuous Integration

GitHub Actions runs only unit tests for performance and reliability.
Integration tests are intended for local verification before releasing major versions or dependency interface changes.
