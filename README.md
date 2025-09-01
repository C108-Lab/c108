
# c108

Curated core Python utilities with minimal dependencies — CLI, IO/streams, filesystem, validation, packaging/versioning, markdown helpers, math, networking, tempfile, and small tools. Heavier integrations (YAML, Rich UI, Git, GCP, etc.) live in optional packages/extras.

- **License**: MIT
- **Audience**: Python developers who prefer small, practical APIs

## Installation

```bash
# Core only (minimal dependencies)
pip install c108
```

Optional integrations are provided as separate packages to keep core lean:

- **YAML**: `c108-yaml`
- **Rich/Console UX**: `c108-rich`
- **GCP**: `c108-gcp`

See each extension package for details.

## Modules

- **c108.abc** – lightweight utilities (container/sequence/dict helpers, simple type checks)
- **c108.cli** – CLI helpers: clify, cli_multiline
- **c108.io** – stream and chunking helpers (StreamingFile, etc.)
- **c108.markdown** – simple Markdown parsing helpers
- **c108.math** – small math utilities
- **c108.network** – bandwidth/time estimates helpers
- **c108.os** – filesystem/path helpers (documented platform specifics where relevant)
- **c108.pack** – version/packaging helpers (PEP 440, semantic/numbered checks)
- **c108.tempfile** – temp file utilities
- **c108.tools** – misc utility helpers
- **c108.units** – units of measurement utilities
- **c108.validators** – common validation utilities
- **c108.zip** – zip/tar helpers

## Design Philosophy

C108-Lab packages are:
- **Curated** – Centrally developed and maintained for consistency
- **Production-ready** – Thoroughly tested and documented
- **Dependency-conscious** – Core package stays lightweight; extra features and heavy deps live in sub-packages
- **Community-friendly** – Issues and feature requests are welcome

## Community & Contributing

While we don't accept pull requests, we actively welcome:
- 🐛 **Bug reports** 
- ✨ **Feature requests**
- 📖 **Documentation feedback**
- ❓ **Usage questions**

Please open an issue on GitHub for any of the above.

## Extensions

```bash
# YAML features
pip install c108-yaml

# Rich console UX
pip install c108-rich

# GCP utilities
pip install c108-gcp
```

## Releases

- Tagged releases on GitHub
- PyPI is the source of truth
- conda-forge feedstock tracks PyPI

## License

MIT License. See [LICENSE](LICENSE) for full details.