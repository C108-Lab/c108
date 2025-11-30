"""Test suite for c108.stubs.mergeable module."""

import ast
import textwrap
from pathlib import Path
import pytest

from c108.stubs import mergeable


class TestMergeableStubs:
    """Core tests for mergeable stub generator."""

    def test_generate_merge_stub_basic(self, tmp_path: Path):
        """Generate stub for simple dataclass with CLI sentinel."""
        fields_info = [("x", "int"), ("y", "str")]
        decorator_config = {"sentinel": None, "include": None, "exclude": None}
        stub = mergeable.generate_merge_stub(
            "Point", fields_info, decorator_config, cli_sentinel="UNSET", include_docs=True
        )
        assert "def merge" in stub
        assert "x: int = UNSET" in stub  # Uses CLI sentinel since decorator has None
        assert "y: str = UNSET" in stub
        assert "Point" in stub
        assert "raise NotImplementedError" in stub

    def test_generate_merge_stub_decorator_sentinel(self, tmp_path: Path):
        """Generate stub using sentinel from decorator config."""
        fields_info = [("timeout", "int"), ("retries", "int")]
        decorator_config = {"sentinel": "None", "include": None, "exclude": None}
        stub = mergeable.generate_merge_stub(
            "Config", fields_info, decorator_config, cli_sentinel="UNSET", include_docs=True
        )
        assert "timeout: int = None" in stub  # Uses decorator sentinel
        assert "retries: int = None" in stub
        assert "UNSET" not in stub  # CLI sentinel not used

    def test_extract_mergeable_dataclass_only(self, tmp_path: Path):
        """Extract only classes with both @dataclass and @mergeable decorators."""
        file_path = tmp_path / "sample.py"
        file_path.write_text(
            textwrap.dedent(
                """
                from dataclasses import dataclass
                from c108.dataclasses import mergeable

                @dataclass
                class PlainConfig:
                    timeout: int = 30

                @mergeable(sentinel=None, include=['host'])
                @dataclass
                class DatabaseConfig:
                    host: str = "localhost"
                    port: int = 5432
                """
            )
        )
        info = mergeable.extract_dataclass_info(file_path)
        assert len(info) == 1  # Only DatabaseConfig should be found
        class_name, fields, config = info[0]
        assert class_name == "DatabaseConfig"
        assert config["sentinel"] == "None"
        assert config["include"] == ["host"]

    @pytest.mark.parametrize(
        "decorator_sentinel,cli_sentinel,expected_in_stub",
        [
            pytest.param(None, "UNSET", "UNSET", id="no_decorator_sentinel_uses_cli"),
            pytest.param("None", "UNSET", "None", id="decorator_sentinel_overrides_cli"),
            pytest.param("MISSING", "DEFAULT", "MISSING", id="custom_decorator_sentinel"),
        ],
    )
    def test_sentinel_precedence(
        self, decorator_sentinel: str | None, cli_sentinel: str, expected_in_stub: str
    ):
        """Test that decorator sentinel takes precedence over CLI sentinel."""
        fields_info = [("field", "str")]
        decorator_config = {"sentinel": decorator_sentinel, "include": None, "exclude": None}
        stub = mergeable.generate_merge_stub(
            "TestClass",
            fields_info,
            decorator_config,
            cli_sentinel=cli_sentinel,
            include_docs=False,
        )
        assert f"field: str = {expected_in_stub}" in stub

    def test_main_no_mergeable_classes(self, tmp_path: Path, capsys):
        """Run main() on file with no @mergeable classes."""
        file_path = tmp_path / "data.py"
        file_path.write_text(
            textwrap.dedent(
                """
                from dataclasses import dataclass

                @dataclass
                class User:
                    name: str
                    age: int
                """
            )
        )
        args = type(
            "Args",
            (),
            {
                "files": [str(file_path)],
                "sentinel": "UNSET",
                "no_docs": False,
                "no_color": True,
                "output": None,
            },
        )()
        mergeable.main(args)
        out, err = capsys.readouterr()
        assert "No @mergeable dataclasses found" in out


class TestMergeableExtented:
    def test_generate_stub_private_include_and_config_docs(self):
        """Ensure private fields can be included and config docs rendered."""
        class_name = "Sample"
        fields_info = [
            ("_hidden", "int"),
            ("public", "str"),
            ("excluded", "float"),
        ]
        decorator_config = {
            "sentinel": "CUSTOM_SENTINEL",
            "include": ["_hidden", "public"],
            "exclude": ["excluded"],
        }

        stub = mergeable.generate_merge_stub(
            class_name,
            fields_info,
            decorator_config,
            cli_sentinel="CLI_SENTINEL",
            include_docs=True,
        )

        # Private field only kept when explicitly included
        assert "_hidden: int = CUSTOM_SENTINEL" in stub
        # Field not in include list is dropped
        assert "excluded: float" not in stub
        # Use decorator sentinel, not CLI sentinel
        assert "CUSTOM_SENTINEL" in stub
        assert "CLI_SENTINEL" not in stub
        # Config notes are present in docstring
        assert "Only these fields can be merged: _hidden, public" in stub
        assert "These fields are excluded: excluded" in stub

    def test_generate_stub_no_fields_returns_skip_comment(self):
        """Return skip comment when no mergeable fields remain."""
        class_name = "Empty"
        fields_info = [("field", "int")]
        decorator_config = {
            "sentinel": None,
            "include": ["nonexistent"],
            "exclude": None,
        }

        result = mergeable.generate_merge_stub(
            class_name,
            fields_info,
            decorator_config,
            cli_sentinel="UNSET",
            include_docs=True,
        )

        assert "SKIPPING: No mergeable fields found for Empty" in result

    def test_generate_stub_no_docs_uses_simple_docstring(self):
        """Use minimal docstring when include_docs is false."""
        class_name = "NoDocs"
        fields_info = [("value", "int")]
        decorator_config = {"sentinel": None, "include": None, "exclude": None}

        stub = mergeable.generate_merge_stub(
            class_name,
            fields_info,
            decorator_config,
            cli_sentinel="UNSET",
            include_docs=False,
        )

        assert '"""Create new NoDocs with updated fields."""' in stub

    @pytest.mark.parametrize(
        "decorator,expected",
        [
            pytest.param(ast.Name(id="dataclass"), True, id="name"),
            pytest.param(
                ast.Attribute(value=ast.Name(id="mod"), attr="dataclass"),
                True,
                id="attr",
            ),
            pytest.param(
                ast.Call(func=ast.Name(id="dataclass"), args=[], keywords=[]),
                True,
                id="call-name",
            ),
            pytest.param(
                ast.Call(
                    func=ast.Attribute(value=ast.Name(id="m"), attr="dataclass"),
                    args=[],
                    keywords=[],
                ),
                True,
                id="call-attr",
            ),
            pytest.param(ast.Name(id="something_else"), False, id="negative"),
        ],
    )
    def test_is_dataclass_decorator_variants(self, decorator, expected):
        """Detect dataclass decorator in multiple AST forms."""
        assert mergeable._is_dataclass_decorator(decorator) is expected

    @pytest.mark.parametrize(
        "decorator,expected",
        [
            pytest.param(ast.Name(id="mergeable"), True, id="name"),
            pytest.param(
                ast.Attribute(value=ast.Name(id="m"), attr="mergeable"),
                True,
                id="attr",
            ),
            pytest.param(
                ast.Call(func=ast.Name(id="mergeable"), args=[], keywords=[]),
                True,
                id="call-name",
            ),
            pytest.param(
                ast.Call(
                    func=ast.Attribute(value=ast.Name(id="pkg"), attr="mergeable"),
                    args=[],
                    keywords=[],
                ),
                True,
                id="call-attr",
            ),
            pytest.param(ast.Name(id="other"), False, id="negative"),
        ],
    )
    def test_is_mergeable_decorator_variants(self, decorator, expected):
        """Detect mergeable decorator in multiple AST forms."""
        assert mergeable._is_mergeable_decorator(decorator) is expected

    def test_extract_mergeable_config_no_parens_defaults(self):
        """Return default config for bare mergeable decorator."""
        decorator = ast.Name(id="mergeable")

        config = mergeable._extract_mergeable_config(decorator)

        assert config == {"sentinel": None, "include": None, "exclude": None}

    def test_extract_mergeable_config_with_exclude_list(self):
        """Parse include and exclude lists from decorator call."""
        decorator = ast.Call(
            func=ast.Name(id="mergeable"),
            args=[],
            keywords=[
                ast.keyword(arg="sentinel", value=ast.Name(id="UNSET")),
                ast.keyword(
                    arg="include",
                    value=ast.List(
                        elts=[ast.Constant("a"), ast.Constant("b")],
                        ctx=ast.Load(),
                    ),
                ),
                ast.keyword(
                    arg="exclude",
                    value=ast.List(
                        elts=[ast.Constant("x"), ast.Constant("y")],
                        ctx=ast.Load(),
                    ),
                ),
            ],
        )

        config = mergeable._extract_mergeable_config(decorator)

        assert config["sentinel"] == "UNSET"
        assert config["include"] == ["a", "b"]
        assert config["exclude"] == ["x", "y"]

    def test_extract_fields_from_class_includes_assign_without_type(self):
        """Collect fields both with and without type hints."""
        class_node = ast.ClassDef(
            name="Sample",
            bases=[],
            keywords=[],
            body=[
                ast.AnnAssign(
                    target=ast.Name(id="typed", ctx=ast.Store()),
                    annotation=ast.Name(id="int", ctx=ast.Load()),
                    value=None,
                    simple=1,
                ),
                ast.Assign(
                    targets=[ast.Name(id="untyped", ctx=ast.Store())],
                    value=ast.Constant(1),
                ),
            ],
            decorator_list=[],
        )

        fields = mergeable._extract_fields_from_class(class_node)

        assert ("typed", "int") in fields
        assert ("untyped", "Any") in fields

    def test_ast_to_string_complex_nodes(self):
        """Convert complex AST expressions to readable strings."""
        # Attribute
        attr = ast.Attribute(value=ast.Name(id="mod"), attr="Name")
        # Subscript
        sub = ast.Subscript(
            value=ast.Name(id="list"),
            slice=ast.Name(id="int"),
            ctx=ast.Load(),
        )
        # Union with |
        union = ast.BinOp(
            left=ast.Name(id="A"),
            op=ast.BitOr(),
            right=ast.Name(id="B"),
        )
        # Tuple
        tup = ast.Tuple(
            elts=[ast.Name(id="X"), ast.Name(id="Y")],
            ctx=ast.Load(),
        )
        # Non-str constant
        const = ast.Constant(42)
        # Fallback using ast.unparse
        fallback = ast.UnaryOp(op=ast.USub(), operand=ast.Constant(1))

        assert mergeable._ast_to_string(attr) == "mod.Name"
        assert mergeable._ast_to_string(sub) == "list[int]"
        assert mergeable._ast_to_string(union) == "A | B"
        assert mergeable._ast_to_string(tup) == "(X, Y)"
        assert mergeable._ast_to_string(const) == "42"
        assert mergeable._ast_to_string(fallback) == "-1"

    def test_extract_dataclass_info_syntax_error_raises_value_error(self, tmp_path):
        """Raise value error when parsing invalid Python file."""
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def broken(")

        with pytest.raises(ValueError, match=r"(?i).*Could not parse.*bad\.py.*"):
            mergeable.extract_dataclass_info(bad_file)

    def test_main_file_not_found_reports_error_and_no_dataclasses(self, capsys):
        """Report not found files and print no-dataclasses message."""

        class Args:
            no_color = True
            files = ["nonexistent_file_xyz.py"]
            sentinel = "UNSET"
            no_docs = False
            output = None

        mergeable.main(Args())

        captured = capsys.readouterr()
        assert "Error: File nonexistent_file_xyz.py not found" in captured.err
        assert "No @mergeable dataclasses found in specified files" in captured.out

    def test_main_with_valid_file_generates_stub_to_stdout(self, tmp_path, capsys):
        """Generate stubs for mergeable dataclasses to standard output."""
        src = tmp_path / "model.py"
        src.write_text(
            "from dataclasses import dataclass\n"
            "UNSET = object()\n"
            "\n"
            "@dataclass\n"
            "@mergeable(sentinel=UNSET, include=['field1'], exclude=['field2'])\n"
            "class Sample:\n"
            "    field1: int\n"
            "    field2: int\n"
        )

        class Args:
            no_color = False
            files = [str(src)]
            sentinel = "CLI_UNSET"
            no_docs = False
            output = None

        mergeable.main(Args())

        captured = capsys.readouterr()
        # When pygments is missing, a tip is printed
        if "# Tip: Install pygments for syntax highlighting" in captured.out:
            assert "# Tip: Install pygments for syntax highlighting" in captured.out
        # Stub header and method should be present
        assert f"# Stubs for {src}" in captured.out
        assert "def merge(self, *," in captured.out
        assert "Only these fields can be merged: field1" in captured.out
        assert "These fields are excluded: field2" in captured.out

    def test_main_with_output_file_writes_results(self, tmp_path, capsys):
        """Write generated stubs to output file when requested."""
        src = tmp_path / "model_out.py"
        src.write_text(
            "from dataclasses import dataclass\n"
            "UNSET = object()\n"
            "\n"
            "@dataclass\n"
            "@mergeable\n"
            "class SampleOut:\n"
            "    a: int\n"
        )

        out_file = tmp_path / "stubs.txt"

        class Args:
            no_color = True
            files = [str(src)]
            sentinel = "CLI_SENTINEL"
            no_docs = True
            output = str(out_file)

        mergeable.main(Args())

        captured = capsys.readouterr()
        assert f"Stubs written to {out_file}" in captured.out

        content = out_file.read_text()
        assert "# Stubs for" in content
        # With no_docs=True, use compact docstring
        assert '"""Create new SampleOut with updated fields."""' in content
