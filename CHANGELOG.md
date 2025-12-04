# Changelog

## [0.2.0] - 2025-12-04

### 📚 Documentation

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

### 💼 Other

- GH write permissions (9781625)

- job names (5c98288)

- Clarify audience and license in README

Updated README to clarify audience and license information. (a3f687f)

- Clean up extension packages section

Removed in-progress section from README. (07e1324)

- ReadTheDocs confgured (3be0505)

- mkdocs up (5933c77)

- mkdocs.yml fix (ecb744a)

- README.md, index.md (6424016)

- typos fixed (5bbb21d)

- mkdocs.yml up (9929915)

- mkdocs.yml up (5629e2d)

- pyproject.toml desc (96545de)

- API docs (aa1a9ad)

- rm stubs docs (2ed9bdf)

- mkdocs.yml (6c54288)

- Docs ref (16a0cbe)

- Docs ref, Py versions (cd30272)

- Docs up (8739497)

- rm dunders from Docs (61b1762)

- rm privates from Docs (0600463)

- clean README.md (dd117a5)

- clean README.md (e908dc8)

- pyproject.toml licence (978d5bf)

- README.md licence (fc58203)

- Docs Licence (3eeb526)

- Docs Licence (a7f281f)

- release.yml + test.yml (e4a1c36)

- release.yml + test.yml up (468ddd1)

- ObjectInfo, deep_sizeof edge cases tests (ca96a79)

- *.cov in gitignore (9a9a022)

- Sort test_abc.py (cda98e9)

- Methods rename in abc.py (50cd2a8)

- isbuiltin update (b091a1f)

- TestValidateParamTypes, TestSearchAttrs updates (c5bbfdf)

- TestValidateTypes* edge cases tests (4fc9749)

- TestClify edge cases (bb0e489)

- mkdocs.yml privates and dunders up (4c5026c)

- BiDirectionalMap type hints, tests update (360d94d)

- mergeable fix (829e051)

- mergeable fix (6225cbb)

- dataclasses.py mergeable InitVar checks (cf76d6f)

- MetaMixin.to_dict() (220b26d)

- MetaMixin.to_dict() fixed (e472c0e)

- Fix docs (d32d038)

- README.md typos fixed (00debf8)

- Add tests for Meta edge cases and note validation TODO (8165f02)

- Comment out optional badge and install snippets (08cd9d8)

- Implement __post_init__ validation for DictifyOptions fields (e36034b)

- Add tests for MetaOptions edge cases and inject convenience flags (d62f93c)

- Add tests for DictifyOptions edge cases and expand behaviors. (ae5bf8c)

- Add comprehensive tests for _iterable_to_mutable and _mapping_to_dict (aab36a4)

- Add comprehensive tests for merge and CLI modules (237939c)

- Extend mergeable tests; move sample.py to examples (1bd9a98)

- display.py clean up (a304875)

- formatters.py test cases for truncation, no-message, and traceback handling (15efaa4)

- Test_GetChunkSize for io.py (10b615b)

- std_numeric refactored + tests (5ebd488)

- pytest update -m "not integration" (1c428ee)

- pytest-rerunfailures moved to test section (7135619)

- network.py tests (0ec47d9)

- numeric.py tests for edge cases and error handling (8cdab4b)

- numeric.py tests for edge cases and error handling (baa7b9e)

- shutil.py test cases (3df96a9)

- Remove "not integration" filter for unit tests in CI (29d5bd6)

- Drop redundant doctest coverage uploads and unify job names (f3f25d6)

- Update Codecov badge URL to include unit flag filtering (2eeaf21)

- Add terminal coverage report and restrict upload to Python 3.12 (4316238)

- test_dictify.py refactored utility classes to nested (2a28fdc)

- .gitignore (d249d57)

- LICENSE (80b91b5)

- README.md (16d3bc4)

- README.md (0e5640c)

- up (17c464d)

- up README.md (4b36d3d)

- up README.md (a0a23b7)

- up README.md (19862bc)

- up typo (0d29edb)

- up typo (37a8dbc)

- rm Ext Packages (5f97047)

- rm Ext Packages (e2dbff5)

- rm Ext Packages (3fd9776)

- up (82f804c)

- up modules (561ab78)

- Ver 25.70.0 (eb4c311)

- as_dict(), minor updates (ffc1551)

- collections, dictify, tools (fc97746)

- is_builtin() (5feb22f)

- TestClassName (48c4cbd)

- up (d5ce361)

- TestAttrIsProperty (926ef7f)

- TestAttrsSearch (e1d979a)

- attrs_eq_names() (00e0604)

- attrs_eq_names() (1c2866c)

- attrs_eq_names() (e03f820)

- deep_sizeof() (ef6885c)

- deep_sizeof() (6e111e1)

- remove_extra_attrs() (662e71f)

- ObjectInfo (865b21c)

- TestObjectInfo (6ceb345)

- ObjectInfo (318cc85)

- ObjectInfo(fq_name:bool) (d102580)

- ObjectInfo docstrings (7bfb2ac)

- class_name docstring (f753287)

- typos (95dbc0a)

- hints (f67b5d7)

- ObjectInfo raises (a6585d0)

- ObjectInfo raises (8de52f0)

- fmt_* in tools.py (1982e22)

- max_repr (0840539)

- TestFmormatSuite cases (1492e8b)

- fmt_value() for broken __repr__? docstring (c50695b)

- TestFmtSequence suite (d362d2b)

- TestFmtSequence full suite (e74e6bf)

- sort methods in tools.py (fec293e)

- FmtSequence full suite (c144f80)

- fmt_sequence upd (212658b)

- fmt_value edge cases (2e71bf8)

- raise formatted (aaf3ce5)

- raise formatted (31ee79c)

- attrs_eq_names docs (4b013bb)

- BiDirectionalMap API (7dc1b5c)

- BiDirectionalMap fmt messages (c0a2772)

- dictify.py hook_mode (5e7bd3d)

- clen up (ee6387a)

- random.py test cases (a1c3b1e)

- scratch.py allocate sparce implemented (5b2952a)

- scratch.py allocate sparce implemented (aa0cff3)

- clify docstring (6dcb000)

- fmt_mappinng API (e13ff4f)

- .gitignore (b944866)

- fmt_exception, fmt_any (318056a)

- listify (52efbcf)

- clify (317e2b6)

- fmt_any() docs (406fc23)

- fmt_* sanity checks (2fdbadb)

- rm print_method (4c1bf6d)

- remove obj_to_str (8726c8a)

- ObjectInfo tests (89a6242)

- to_ascii multi-type (8b5ccec)

- get_caller_name with tests (4048536)

- sort methods (3457637)

- sequence_get and tests (a3b0d84)

- sequence_get docs (aa691ce)

- fmt_type implemented (7cda9d7)

- match=r"(?i)" for tests (221959d)

- list_get (bc8f918)

- list_get tests (4295e9d)

- todo-s (175e483)

- dict_get (af75042)

- dict_set method (0731dbd)

- dict_set tests (0eb85d7)

- TestFmt* style (31336c8)

- listify update (2879141)

- as_ascii (5925164)

- as_ascii (efeb37e)

- imports from collections.abc (fa63f83)

- __version__ (4eff8ab)

- ObjectInfo (49bb1c0)

- ObjectInfo (a5dd6b3)

- ObjectInfo tests (082ba28)

- ObjectInfo tests (3241e68)

- clean tests (9a0a58a)

- attrs_eq_names & tests (d6c55df)

- attr_is_property (8c60138)

- attrs_search tests update (8e02466)

- class_name() update + tests (7b5bd4b)

- deep_sizeof() update (5800955)

- deep_sizeof() tests (dc64a9f)

- is_builtin tests (fddc5eb)

- remove_extra_attrs parametrized tests (f132e66)

- imports fix (8796470)

- error msg format fix (91d3f5a)

- clify error msgs (801f6f9)

- clify clean API (902fdc2)

- cli_multiline and tests (dc5be68)

- up cli* (2020ddc)

- up cli*, fmt_* errors (48ebb16)

- BiDirectionalMao (1023941)

- BiDirectionalMap fix for duplicates (112ee63)

- fmt_* for errors (1741fc4)

- dictify.py fmt_* for errors (ea499c7)

- ToDictOptions, to_dict new API (eac9d6f)

- core_to_dict update (e91b1c5)

- core_dictify API and DOCs (4e427fd)

- core_dictify API and DOCs major update (f129e85)

- up (4538ef9)

- core_dictify up docstring and add tests (7008f34)

- serialize_* comments (dc4bc44)

- units.py (d0f99aa)

- dictify API and docs (486a8de)

- core_dictify API review (10c5268)

- core_dictify API review (bdc7970)

- core_dictify API, __fn_terminal() (49b4333)

- core_dictify API fixed (f850c6c)

- class name injection (d17ef55)

- class name injection and sort_keys (d9f2756)

- _count_positional_args (983a165)

- core_dictify depth, class name include/inject tests (a48d64e)

- core_dictify type_handlers (3616597)

- core_dictify docstring (86ad07a)

- _get_type_handler() (fcf7615)

- DictifyOptions.add_type_handler() (b56f9f3)

- never_filter types update (9ca3e6e)

- _is_skip_type, _implements_iter (d6d888c)

- .gitignore (1e55ff8)

- __trim_extra_items() (5461b51)

- __trim_extra_items() islice (26310b5)

- mv methods outer (879f405)

- up fn_* handlers (2aebf60)

- sort module methods (13b276f)

- up (3e632b2)

- _proc_<collection_type> processors (90eb179)

- _proc_trimme_<collection_type> processors, _process_collection up (46882be)

- DictifyOptions methods up (93f5a24)

- DictifyOptions class methods (eda3ed0)

- DictifyOptions type_handlers property (d6e1f47)

- up (d1b5b2e)

- process_keys, _process_key() (32329bd)

- _process_key() (04ae564)

- _proc_dict* (b66581c)

- _process_key ok (ca0dff1)

- rm __process key, inject_*_metadata options (030acfb)

- DictifyMeta, keys handling reviewed (5e373f0)

- SizeMeta, TrimmedMeta, TypeMeta (1f33393)

- SizeMeta (39c43d8)

- TrimmedMeta (6c12ea7)

- TrimmedMeta setter (e9613f8)

- TrimmedMeta, SizeMeta sanity checks (d8400cf)

- clean up (a318022)

- fully_qualified_name() todo (01e6b15)

- TypeMeta (d4d5596)

- TypeMeta (e42306c)

- MetaMixin, SizeMeta, TrimmedMeta (25dd8b3)

- TrimMeta, TypeMeta (f01442f)

- MetaMixin tests (f99fd42)

- SizeMeta tests (6837440)

- TrimMeta tests (c2311df)

- TypeMeta tests (c9b55dc)

- TrinNeta, DictifyMeta (f17e57a)

- DictifyMeta.VERSION (7361c4a)

- DictifyMeta docsrting (581bb2a)

- DictifyMeta docs (5c98aad)

- TrimMeta (88bc6bf)

- TrimMeta clen up (da5dee5)

- SizeMeta, TrimMeta, TypeMeta frozen (1cc177e)

- clean up tests (1e050ca)

- deep_sizeof TODO (b7e153b)

- move _default_type_handlers() (6d24646)

- DictifyMeta tests (e1ed0cf)

- attrs_eq_names + tests (2c3b2f6)

- DictifyMetaOptions (4814141)

- MetaInjectOptions (03c38eb)

- MetaInjectOptions.injected property (4c7043e)

- MetaInjectionOptions, names (9d4e7be)

- DictifyOptions doctstring for Meta Data Injection (42aa52a)

- __dictify__ key (5b8b475)

- DictifyMeta.type, _make_metadata(), _inject_metadata() (22ddb89)

- _inject_metadata() (77560f9)

- include_none_attrs (ae245fc)

- _make_metadata() (7612372)

- DictifyMeta.to_dict() (0a850fa)

- MetaInjectionOptions.sizes_enabled (e15f2ae)

- serial_dictify() (99bf2d3)

- _make_metadata() rm was_trimmed, auto cals (12b4a7e)

- core_dictify() meta strategy in docs (f0c5e64)

- TODO-s (e239dc5)

- *_sized_iterable rename (a38e05a)

- *_sized_iterable up (bdd49ec)

- _process* methods (6afa0df)

- rm post-trimming processors (3eb52e4)

- _proc_dict_like() robust (39bbd18)

- clean keys (0898542)

- clean keys (cfc8ca0)

- TypeConversionOptions, _proc_sequence for named tuple (9973278)

- _proc_sequence type_opt fix (6b7add4)

- remove bad trimming logic (8363f7d)

- create/inject_meta (aa4e652)

- _proc for views (162ef28)

- SizeMeta.from_object() and tests (fb205cb)

- TrimMeta.from_objects() and tests (eae5698)

- TypeMeta.from_objects() and tests (0efbb8e)

- TypeMeta.from_objects() and tests (4baf255)

- TypeMeta.from_objects() and tests logic update (4f28e1e)

- TypeMeta tests (9fa9f75)

- DictifyMeta.from_objects() tests (102c955)

- mv DictifyMeta.from_objects() tests (f473341)

- to_mutable(), inject_meta() (2bf852a)

- to_mutable() memoryview (3d51d19)

- _is_sized_kv_iterable() (f3f89a5)

- _is_sized_kv_iterable() (0ecb6e6)

- Handlers = field(default_factory=Handlers) (97f4f9e)

- Processing Handlers docs (850f17f)

- DictifyOptions.basic() .debug() .logging() .serial() (1f4ee36)

- DictifyOptions.basic() .debug() .logging() .serial() docs (6ce4a23)

- TrimMeta to support unknown length (generators et al) (78b6520)

- TrimMeta + tests support unsized iterables (551faca)

- _process_iterable() (9082fc0)

- _process_iterable clean up (593d300)

- _iterable_to_mutable, _shallow_to_mutable (513e041)

- expand(), inject_meta(), to_mutable() (ab2b82d)

- sorting and trimming (d4640a0)

- inject_meta() update (a979126)

- inject_meta(), core_dictify(), expand() docstrings (ad0aed8)

- ClassNameOptions (5dcd6c3)

- clean up unused methods (e518959)

- expand() sort and trim (0e9cd8c)

- dictify() (3e20760)

- ClassNameOptions propagated (79dcc1a)

- ClassNameOptions (3d92255)

- TODOs (c382a92)

- MetaInjectionOptions in_to_dict, in_expand (625f018)

- TrimMeta.from_objects() (b66070a)

- _get_from_to_dict() meta inject (fc1117e)

- Tests inject_meta (ea64608)

- expand () list | dict (b3705a3)

- MetaOptions.merge() (5208b9c)

- ClassNameOptions.merge() (af38b96)

- DictifyOptions.merge() (a12522a)

- Tests for DictifyOptions.merge() (c96bf69)

- C108 Sentinels + tests (ee5e87d)

- UNSET, UnsetType (a44f77c)

- ClassNameOptions.merge() (fe92d16)

- core_dictify() tests up finished (ff55a98)

- expand() docs (6ab6323)

- expand() fixed iterables (e86c32c)

- _handle_memoty_view fix (fc69d2a)

- dictify_core renamed (10d1daa)

- TODO-s (effb084)

- Handlers (b710239)

- inject_meta() docs (8b71d05)

- dictify_core rm fn_* args (633a382)

- dictify_core docs (1ec5586)

- Meta class (eb4d0c0)

- max_str_len (8fc0275)

- dictify() docs (731c3fd)

- dictify() docs (537727d)

- TypeMeta.from_object (b725f03)

- Meta.from_object() (b068cde)

- expand() (4112fa1)

- optimize imports (ed16793)

- clean up defensive validations/typechecks (aef4453)

- clean up typechecks (18f2505)

- abc.py validations clean up (93bb412)

- abc.py validations clean up (20bbb99)

- cli.py messages (937e02f)

- core_dicttify() validators (132a974)

- expand() clen up (af4abcc)

- dictify.py clean up (020bc96)

- dictify_core() with max_items max_str_len max_bytes None (ee9f2ec)

- TestDictify suite (587bbca)

- dictify_core docs (144e971)

- dictify_core docs (40eead9)

- dictify() docstring (acb6bed)

- default type handlers added (7552693)

- default type handlers tests (2d78aed)

- default_type_handlers() up (8213763)

- up (ccf059f)

- clean up dictify.py (6ade664)

- up (5435369)

- module docstring (d40fa6d)

- attrs_search() update (ac284d2)

- ObjectInfo updates (a5b986b)

- ObjectInfo fix (6e23a01)

- ObjectInfo tests (637ba4d)

- module docs (9a69da1)

- module docs (6546f2e)

- StreamingFile clean up (a409922)

- StreamingFile docs (40d8827)

- StreamingFile tests (2efec8c)

- rm markdown.py (33d046f)

- network.py clean up, upd (bf442c0)

- StreamingFile(io.BufferedIOBase) thread safe (0d5cb82)

- StreamingFile(io.BufferedIOBase) tests (b10ce6e)

- StreamingFile(io.BufferedIOBase) tests for thread safety (2b5c52d)

- network.py major update (2dfd51c)

- network.py sort, polish (b61072c)

- network.py up (4a826a8)

- methods sorted (c62d7f9)

- transfer_estimates() clean up (4691a5f)

- network.py tests (95a504c)

- up defaults (418a9b3)

- os.py and tests (4e5e1fc)

- c108.os.clean_dir (a5a7847)

- random_factor() up (3f7fcc0)

- minor fix cli.py (03d6942)

- sort methods (286a99f)

- up (d06d9d8)

- _attrs_is_property moved (fe0a32f)

- deep_sizeof upgraded (b01a162)

- search_attrs() (75e06b5)

- search_attrs() (bce465f)

- search_attrs() docstring (82f4e61)

- minor up (d160b7a)

- clean up modules pack random scratch (7befdc6)

- scratch.py (d6e1fac)

- atomic_open() (fcc9d76)

- json.py (bccb7c3)

- read_json() tests (e9e8c61)

- write_json() tests (3666e49)

- update_json() tests (990468f)

- update_json() tests (04ee210)

- up imports in os.py (e130c06)

- up os.py (af106f3)

- io.py docstring (696488c)

- shutil and os split (93e53d6)

- copy_file(0 (2638dec)

- copy_file() docs (0e47c8f)

- imports up (24d9aa2)

- shutil imports (0fa83aa)

- scratch.py temp_dir() docs and tests (0c08989)

- backup_file() args merged to name_format (7202607)

- backup_file() comments (758c082)

- up formatting (be3c915)

- rm list_get() (7520c5f)

- rm print_title() (bdf6529)

- fmt_* docs and get_caller_name docs (c58b832)

- tools.py module docs (3bc0e83)

- sentinels.py module docs (85b9059)

- sentinels.py module docs (3beca32)

- TestNumUnitsDEMO (ccddc19)

- units.py multi-fix docs and edge cases (ce5fe99)

- units.py plural_units (6e12e8b)

- units.py privates (8b94314)

- units.py frozen NumberUnit, .si_fixed() (9015c11)

- rm si_unit from NumberUnit (aa6a129)

- mv NumberUnit to DisplayValue (d9ba0b8)

- display.py module docs (edbe6e3)

- display.py exponents and _is_finite() (c44a35a)

- stdlib infinite numerics support (92f3cf3)

- trimmed_digits() fix, .display_digits (8b0dcbf)

- creator methods base_fixed plain si_flex (78b114b)

- _get_plural_units() wrapper (34ef723)

- _validate_si_prefixes* (d293e97)

- mv constants to DisplayConf (bb171fb)

- _validate_prefixes_multipliers (c1ca885)

- _validate_prefixes_multipliers (c89368a)

- unit_prefixes (961ccb9)

- _validate_prefixes_and_multipliers() (d8b3b1f)

- unicode.py to_sup() to_sub() with tests (c83ec25)

- disp_power() (c56dfea)

- _disp_power() (e75926b)

- unicode.py (cd8148c)

- _disp_power() up (130ea34)

- _disp_power() up (dbbb7f2)

- sort lines (ce617fc)

- BIN_SCALE_*, SI_SCALE_* constants (07c704c)

- .scale_base/step (7e3d898)

- numeric.py with tests (27419e2)

- display.py wrapper in _std_numeric() (ffeef42)

- display.py docs (8ba8753)

- _disp_power() formats (6ef6531)

- *_exp TODO-s (eac987c)

- .scale (1f43f95)

- _validate_scale_type() (d74f74b)

- _validate_unit_prefixes_map() (3df56e2)

- SI_PREFIXES_3N, SI_PREFIXES (2db93bf)

- updated BiDirectionalMap instantiation from another BiDirectionalMap instance (8518d54)

- _validate_unit_prefixes_raise() (01f1c84)

- _unit_exp _mult_exp mulifix (5f566f5)

- _validate_unit_plurals() (0d7c7fa)

- _validate_*() other fields (74ee1ec)

- __post_init__() sorted (5160b09)

- mult_* rename (dc3508d)

- impl _disp_power() usage (288c164)

- MultSymbol.X (d8660eb)

- up (0361ce9)

- overflow tolerance (c27f396)

- tolerance up (b21a0e8)

- total_exponent handling (d1120ef)

- DisplayMode.FIXED (c3721e6)

- sorted SI prefixes (09ee23c)

- _validate_scale_type() (cad2c16)

- _validate_unit_exp* validators + tests (87b6b14)

- _auto_mult_exponent() with tests - bin+decimal (03727f6)

- _auto_mult_exponent() with tests - bin+decimal (3e003ad)

- _auto_unit_exponent() + tests (925ebec)

- bog scale gaps handling docs (b1098bc)

- _residual_exponent (376a77c)

- DisplaySymbols, TODOs clean up (4f9f731)

- up comments (fedfec2)

- intro comments (9f4f4fc)

- overflow predicates + tests (9b49760)

- overflow/underflow formatting (05bfee5)

- DisplayValue docs (1770f09)

- TestDisplayValueMode (b4b3feb)

- TestDisplayValueAsStr (efcf2a6)

- .units (8fb49b8)

- .number (13da7cc)

- .number (15642af)

- DisplaySymbols (cd4882d)

- _over_unit_str() (8320142)

- .number formatting (dfd4386)

- .normalized (1cdeecc)

- _normalized_number() (f1c1147)

- trimmed_round() with tests (37e3b3f)

- TestDisplayValueNormalized (3eaa92b)

- DisplayValue props (bfc418a)

- separator (d214884)

- separator up (71d51c7)

- DisplaySymbols frozen (18def08)

- DisplaySymbols.mult (81fd02d)

- ref_value docs (04399ea)

- DisplayFlow, DisplayFormat (1eea4de)

- DisplayScale, scale.base, scale.step, tests (8b796a0)

- scale.type up (b71542e)

- DisplayFlow + tests for overflows/underflows (e8f6a11)

- DisplayFlow docs (b3aee19)

- DisplayFlow docs (f15fb0b)

- DisplayValue docs (38be9ee)

- DisplayMode docs (e3ab5da)

- DisplaySymbols docs (fdd9312)

- MultSymbol docs (9276036)

- DisplayConf docs (417bf73)

- TODOs (ba86412)

- DisplayConf docs (3a09e1d)

- DisplayFormat docs and tests (c0d24e8)

- TestDisplayFormat.merge() (c918ff0)

- DisplayScale docs (20d6365)

- DisplayFormat .ascii .unicode (ce7235c)

- DisplayFormat.mult_exp() (babbcd7)

- up comments (3181282)

- DisplayValue.unit_prefix (ac53c8c)

- TestDisplaySymbols (ab1eecc)

- TestDisplayFlow (9aaef31)

- TestDisplayFormat update (cb270ca)

- DisplaySymbols sort (675b465)

- TestDisplayScale suite (0bbb48e)

- DisplayValue tests, flow, format, scale up (a4828e1)

- DisplayValue tests, flow, format, scale up (5444db0)

- TestNumpyNumericSupport (bbb757d)

- TestStdNumNumpyNumericSupport (3a45bed)

- TestStdNumPandasNumericSupport (d5b6788)

- TestStdNumPandasNumericSupport (40e9201)

- std_numeric fixed for Series (1bd10a0)

- std_numeric fixed for Series (2ed0b4f)

- PyTorch tests (f5e9a9d)

- jax, tensorflow deps (b591000)

- JAX tests (5a74812)

- JAX tests 64x comments (7fe9f60)

- TestDisplayValueIntegration (92ff375)

- Astropy float_to_int (aaee277)

- Heuristics float>>int partial support - Fractiopnal, Decimal, Astropy (a9914e6)

- SymPy tests (e92d330)

- DisplayValue tests clean up (1a6b603)

- integration tests (20097df)

- README for tests (94e0973)

- integration/test_io.py (252b3e2)

- gitignore (55954e1)

- display.py updates (b061684)

- dictify.py TODO (b4ec3cc)

- dictify.py TODO (b383573)

- clean upp print() (7998b72)

- si_fixed() si_flex() factories (9734e99)

- base_fixed() params (d2c316a)

- factories API (3fdf9f2)

- TestDisplayValueFactoryBaseFixed (5b82925)

- TestDisplayValueFactorySIFixed (4deb6af)

- TestDisplayValueFactorySIFlex (9363ac5)

- merge tests (27bece2)

- factories with scale, overflow, format + tests (a0cdfa3)

- factories docs (2f5b123)

- base_fixed() examples (8517838)

- Overflow Formatting docs (d214fe4)

- factories docs (fc3d62a)

- factories docs (5851614)

- DisplayValue.merge() .to_str() and tests (b7e7657)

- attrs_search() refactored (e2c8c07)

- validate_types() (ef2337e)

- validate_types() up (e09ce45)

- validate_types() fix for None values (7cd76e6)

- validate_types() fix with tests and docs (2788774)

- fmt_value() params (33fb7af)

- fmt_type() applied (c5fe337)

- bool = True (7e3f9af)

- validate_types(), validate_para_types(), @valid_param_types (a0b8cd0)

- validate_param_types() docs (70fab88)

- valid_param_types() docs (69e07f2)

- exclude_self docs (37bfb62)

- module docs abc.py validators.py (6070d65)

- _validate_obj_type messages (aaa8a7d)

- sorted methods (6dbd97d)

- complex Union docs, tests (dafc34b)

- valid_param_type tests (397945d)

- 🚀 Performance (e32bb33)

- 🚀 Performance (92d9617)

- abc.py doctest ok (032f4ca)

- DisplayValue.normalize for int (894b85f)

- DisplayValue __post_init__ (91c75cc)

- type validation failed (c34c33d)

- _is_close_to_one() (b435fe3)

- _is_close_to_one() (a8def59)

- normalized() (ada7a7e)

- DisplayValue.normalized preserving precision (c4e9ce2)

- DisplayValue.normalized preserving precision (2386cee)

- Test_MultiplyPreservingPrecision + fix huge int passed to DisplayValue (ddcf801)

- isnan fix (f9d28d6)

- Any (49607e9)

- trimmed_round() Examples (8d1e081)

- MultSymbol (39d99a6)

- MultSymbol docs (79891fb)

- MultSymbol docs (8f69309)

- DisplayConf sort (4397366)

- DisplayFlow docs (c3ecec0)

- DisplayFormat examples (7950105)

- DisplayScale ex (32487de)

- sentinels.py if* helpers (a59c7fb)

- _if_sentinel() refactored, tests (558e916)

- dictify.py with ifunset() (bf92eca)

- sentinels.py naming fix (5050930)

- DisplaySymbols.merge() and tests (78614ae)

- DisplaySymbols.merge() and tests (ec7dd1d)

- DisplaySymbols.merge() examples (4611e1c)

- TODO (247f721)

- sentinels.py SentinelBase (b498f20)

- create_sentinel_type and SentinelBase fixed (c6cf2e1)

- removed pickling support (4c96e3c)

- quotes (5bc409a)

- sentinels.py Examples (85631d3)

- up (03f36c7)

- format (b956b8a)

- format (f95e9e9)

- format (f7d55cd)

- ruff format (219488d)

- sentinels.py module-level docstrings (2a85733)

- tools.py Examples (0a8012b)

- unicode.py Examples (bd2ad98)

- utils.py Examples (b358a41)

- README.md (69dc8f4)

- utils.py docs (589781e)

- tools.py docs (261686f)

- validators.py error formatters (9a3ddae)

- validators.py error format (680c667)

- validate_language_code() and tests (af6e71a)

- validate_language_code() messages (582a6ff)

- validate* format (4b832ac)

- validate_language() examples (5d76c9f)

- Schemes and validate_uri() (fe157ee)

- Sorted Schemes (cb80ba8)

- ClassGetter, @classgetter (6d97b82)

- ClassGetter, @classgetter tests (305aec5)

- Schemes.all with @classgetter (48e2a78)

- validators.py all() with @classgetter (6df65a4)

- "invalid" messages (88e8d70)

- secure messages (bb14fe8)

- sort classes (899b1b5)

- validate_uri() tests for AWS, Azure, GCP (213062f)

- validate_uri() tests for AWS, Azure, GCP (6d9be7f)

- up (2c6f4e2)

- valudate_uri for MLFlow models and runs (3ae4718)

- valudate_uri for Mongo and Neo4j (fdc4dc3)

- valudate_uri for Vector DB tests (d1135c9)

- TestValidateCategorical (c4a27c9)

- validators.py Integration tests (4266fe4)

- up (aa121f9)

- validate_categorical() *, (ee85ea9)

- multi-API kwargs fix (7ec3ead)

- validators modules structure (106c734)

- validators Examples fix (b242743)

- up (8fffebc)

- zip.py mv (bc85c5a)

- README.md up (7fb6557)

- README.md sorted (81486e1)

- TODO fmt_* (5893dd2)

- tools.py todos (b55af14)

- validate_shape(), TestValidateShapePandas (6b234b5)

- validate_shape() allow_scalar removed, Dummy tests (889d8f7)

- validate_shape() tests (ee7d7ff)

- TestValidateShape unit tests (9ecfc7a)

- _get_shape* (c0b897b)

- rm zip.py (b08e52d)

- std_numeric() up (ec78fcc)

- formatters.py refactored (b9a3f84)

- formatters.py refactored (fea7cdd)

- abc.py fix error messages (69c6d59)

- cli.py fix error messages (e46f594)

- collections.py clean error messages (b735d3a)

- validate_categorical (bd24dd9)

- pixi ENV (6ebda77)

- numeric.py (9d9f086)

- rm pixi (dbf8629)

- add pyproject.toml, uv (7ab554e)

- class_name() + tests (b67c2e2)

- formatting (b804f8b)

- class_name() + tests (425c141)

- unicode.py docs (b796906)

- network.py constants (00e07c2)

- network.py constants (4dbffe8)

- opts (ba77e73)

- dictify.py opts (6208a38)

- dictify.py opts (947171a)

- opts (a0d18c4)

- ifnotnone wrapper (2fbd362)

- is UNSET, is not MISSING (bc3658f)

- ClassGetter, @classgetter (0e8feaf)

- ClassGetter, @classgetter docs (c2c5618)

- ClassGetter, @classgetter docs (d7875ca)

- pytest.ini (9e4a7a9)

- network.py (0357369)

- fmt_type (ba8c335)

- TransferOptions.__post_init__ (c892b59)

- network.py (e7d1ba7)

- network.py (fc4ec48)

- dataclasses @mergable (2091429)

- dataclasses @mergable (f7aaeee)

- dataclasses @mergable stubs (ca8fcda)

- SampleMergeable (03ba5a4)

- mergeable.py stub (41fafd3)

- merge.py stub (28a254b)

- ObjectInfo frozen (0369043)

- TestMergeStubs suite (1abb86f)

- mergeable.py + merge.py docs (4f03028)

- mergeable.py + merge.py docs (6359230)

- SampleMerge, SampleMergeShort (0006abd)

- mergeable docs (d04dbbb)

- mergeable docs (0835432)

- stabs/samples.py (020c214)

- merge.py docs (3ef9c6a)

- README.md structure (1c79207)

- README.md uv sync (981d649)

- TransferOptions test (a8f6cfc)

- TestTransferOptionsFactories (8b9566b)

- TestTransferTimeout clean up (2c1334d)

- abc.py doctest (fc8721e)

- cli.py docs (d6a570a)

- dataclasses.py docs (0f85599)

- TestSearchAttrsSorting (0fa12db)

- sort order fix (251bdb3)

- uv upd (d7a4709)

- abc.py doctest (ab42c38)

- abc.py doctest (a21b706)

- rm @formatter:... (bd1ac37)

- speed_unit fix (4dfa742)

- Meta.from_object* (22e71e4)

- Meta.from_object* (d24c52a)

- io.py (a1a492d)

- network.py doctest (bd44815)

- numeric.py doctest (48e1ca6)

- scratch.py doctest (a2d6d23)

- backup_file doctest (7b85646)

- clean_dir, copy_file doctests (c48710b)

- copy_file docs (c58c138)

- shutil.py xdoctest (19798d5)

- LICENCE (6436148)

- Licence format (6694854)

- README.md intro (8f15e31)

- README.md clean up (a61062a)

- README.md clean up (8d5152d)

- README.md intro (bb2cd2d)

- README.md install (00d0103)

- ruff format (4515dbe)

- pyproject.toml for ruff (ba67c37)

- README.md Install (6cb7b6a)

- README naming (94c99e2)

- Ruff action (7bc8024)

- up (2fa8865)

- up (0503712)

- pyproject.toml (9d154b7)

- Self compatibility shim (a7d415f)

- Self fixed (0f6e4e6)

- IO fix (38757d9)

- validate_shape() docs (9d88076)

- explicit --cov-report=xml:coverage.xml (61a2ad3)

- explicit --cov-report=xml:coverage.xml (cf5f8dd)

- Check coverage.xml file (3a2dbbe)

- Codecov badge (da8ff23)

- up (208f5bf)

- up (a02a143)

- Python test matrix (a12ea9d)

- README.md badges (4a78152)

- README.md badges (a575639)

- README.md badges (7871896)

- 2x coverage uploads (715e31d)

- 2x coverage uploads (b764b5e)

- Coverage tag (7b37cba)

- Coverage tag (9331f00)

- CodeQL workflow (5ed0eda)

- actions up v4 (f563831)

- CodeQL, SECURITY.md (5ee3ce7)

- SECURITY.md (3dbe41c)

- tagged/non-tagged (ae2a84b)

- Release 0.1.0 (a7cd16b)

- refactored display.py with ifset() (2a38685)
