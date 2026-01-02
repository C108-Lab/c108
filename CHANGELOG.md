# Changelog

## [0.2.6] - 2026-01-02

### 🤖 CI/CD

- remove verbose flag from PyPI publish step (main) (76ade4f)

## [0.2.4] - 2026-01-02

### 🤖 CI/CD

- grant write permission for coverage badge updates (main) (541fbc3)

## [0.2.3] - 2026-01-02

### ♻️ Refactor

- simplify option handling and streamline core functions (46c938a)
- replace FmtStyle enum with Style Literal (#108) (88f1bbe)
- standardize `style` defaults and enhance recursion control (8e290b6)
- add `label_primitives` option and enhance formatting functions (883a39e)
- replace `_fmt_is_textual` with `_is_textual` and add set handling improvements (cd4f086)

### ✨ Features

- add reference tracking and cycle detection options and params (540d065)
- add reference tracking & cycle detection (#feat/dictify) (bb5bf0d)
- tweak preset limits for compact/debug/logger (#108) (e3051d0)
- introduce `TestFmtOptions` class (4983c12)

### 🎨 Styling

- simplify _fmt_type_value signature (#feat/formatters) (0bb5545)

### 🐛 Bug Fixes

- rename _fmt_more_token to _fmt_ellipsis (#42) (fdd7384)
- update `fmt_sequence` and `fmt_set` to standardize set formatting (0e4bd63)

### 📖 Documentation

- update changelog for version 0.2.2 (e073e3f)
- switch to dynamic coverage badge and update badge config (a4c93fe)
- switch to dynamic coverage badge and update badge config (7efe827)
- switch to dynamic coverage badge and update badge config (9002d5e)
- remove commented-out extension packages section (#docs) (eb9c724)
- clarify class name injection in mapping objects (db4f20e)
- add reference-tracking and cycle-detection docs (#dictify) (b5328e1)
- clarify docstring and rename _fmt_type_value (#108) (55aa126)
- fix grammar in docstring for merge and mergeable (#108) (040a5a3)

### 📦 Build

- relax upper bounds and bump min versions (83cd18f)
- remove static badge assets and switch to `toml-cli` (18f6a01)
- switch coverage badge to toml-cli color thresholds (main) (7d5b18e)

### 🤖 CI/CD

- disable doctest step temporarily (#108) (d5db20c)
- update release workflow for PyPI Trusted Publisher and improved badge handling (43cfc37)
- comment out doctest step in release workflow (6b97e72)
- comment-out doctest step (953f055)

### 🧪 Testing

- relax match pattern for boolean error message (#dictify) (352f58d)
- add tests for `label_primitives` parameter and truncation logic (6b1f9fa)
- add set dispatch (1f7537c)

## [0.2.2] - 2025-12-05

### 🎨 Styling

- add emoji groups, drop HTML comments (b2b6851)

### 🔧 Other

- cliff.toml update (cefc7ab)
- cliff.toml clean up (977d5dd)

## [pre-rewrite-docs] - 2025-12-05

### ♻️ Refactor

- move badges deps into extras; tidy docs (af0c067)

### 📖 Documentation

- README Coverage (768f446)
- Badges (5e3ab71)
- Badges in README.md (346330d)
- README.md fix (09e1c78)
- fix badge image paths in README (ae7023d)
- badge generation reference (e833d5d)
- update badge URLs to point to main branch (bbd5c4e)

## [0.2.1] - 2025-12-04


### 📚 Docs

- clean up sections in README (#0.2.1) (f36e8a5)

## [0.2.0] - 2025-12-04

### 📚 Docs

- docstrings for DictifyOptions and core_dictify() (a3da517)
- docs for inject_*_metadata options (b3933c4)
- docstrings for TrimmedMeta, SizeMeta (6500812)
- docs up (af29bd1)
- docs drafts (4411da0)
- docs (59b497a)
- docs (ea00db5)
- docs with trim/precision rules (a3496ed)
- docs formatting pipeline (9e8bda8)
- docs formatting pipeline in creators (273cac6)
- docs in creators (2e25670)
- docs (e6618c7)
- docs unicode.py (966947e)
- docs numeric.py (4d62f81)
- docs trim_digits() (3b189ce)
- docs (d660545)
- docs (5d354eb)
- docs (3e2499d)
- docs examples (9c4fd73)
- docs abc.py dictify.py (4522afb)
- docs os.py (bbe28b5)

### 🧪 Testing

- test names nad docstrings (08cf839)
- tests review (1fd2307)
- tests renamed (5908d30)
- test_overflow_format_unitflex (5ba8adb)
- test_display.py clean up (5c4cff8)
- test_display.py postpone astropy (e26eaea)
- test_display.py fix Factory tests (e76553a)
- test_display.py (acd56d6)
- test_supports_empty_or_whitespace_key (8250cf4)
- tests (5d8952d)
- tests (c49bdff)
- test_numeric.py fix (64c259a)
- test_numeric.py fix (a0212b2)
- test_network.py refactored (766d159)
- test-release.yml workflow (b3883d7)
- test-release.yml matrix (e258347)
- test-release.yml codecov upd (207872e)
- test-release.yml codecov v5 (4e36685)
- test-release.yml Coverage fix (9193519)
- test-release.yml comments (b01f9f2)
- test-matrix, test-core (f86cc31)
- test-matrix, test-core (f746bd0)
- test-matrix fix (e2f9664)
- test-release.yml up (3359f26)
- test-release.yml for publish (5803937)
- test-release.yml (fa3ce2a)
- test-release.yml (620b928)
- test-release.yml fix (fda3a91)
- test-release.yml stable versions (a218055)

### 🐛 Bug Fixes

- fix imports (0d12b9d)
- fixed sort_keys and class name injection (a52acc2)
- fixed typecheck (b77ed8a)
- fix include_* flags (38d71a5)
- fix def tests (dc7adcc)
- fix abc.py def tests (2b0b4a5)
- fixed DictifyOptions() presets (2e29c03)
- fixed walrus precedence (a962393)
- fix zip.py (cfb9a51)
- fixed unit exponents and mode (4cbdd4d)
- fix tensorflow booleans (3a0754c)
- fixed astropy tests expectations (75b6748)
- fixed astropy tests for bool and arrays reject (34728bc)
- fix tests, _is_units_value() (8479abd)
- fix validate_type() strict mode (9869d95)
- fix for 3.10+ (0e90b8b)
- fixed Examples display.py (15a998f)
- fix trim_digits (fb73b4c)
- fix trim_digits (7a266b8)
- fixed 'standard' (a055b79)
- fix validators.py (b9b967c)
- fix err messages (a864a19)
- fix ObjectInfo (7d585ca)
- fix coverage uploads (df2d582)
