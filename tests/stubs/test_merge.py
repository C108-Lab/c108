"""
Test suite for c108.stubs.merge module.
"""

# Standard Library -----------------------------------------------------------------------------------------------------
import ast
import textwrap
import tempfile
import types

from pathlib import Path

import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.stubs import merge


class TestMergeStubs:
    """Core tests for merge implementation generator."""

    def test_generate_merge_implementation_basic(self, tmp_path: Path):
        """Generate implementation for simple dataclass with UNSET sentinel."""
        fields_info = [("x", "int"), ("y", "str")]
        implementation = merge.generate_merge_implementation(
            "Point", fields_info, sentinel="UNSET", include_docs=True
        )
        assert "def merge" in implementation
        assert "x: int = UNSET" in implementation
        assert "y: str = UNSET" in implementation
        assert "ifnotunset(x, default=self.x)" in implementation
        assert "ifnotunset(y, default=self.y)" in implementation
        assert "return Point(x=x, y=y)" in implementation

    def test_generate_merge_implementation_different_sentinel(self, tmp_path: Path):
        """Generate implementation using different sentinel and wrapper."""
        fields_info = [("timeout", "int"), ("retries", "int")]
        implementation = merge.generate_merge_implementation(
            "Config", fields_info, sentinel="None", include_docs=True
        )
        assert "timeout: int = None" in implementation
        assert "retries: int = None" in implementation
        assert "ifnotnone(timeout, default=self.timeout)" in implementation
        assert "ifnotnone(retries, default=self.retries)" in implementation

    def test_extract_dataclass_info_specific_classes(self, tmp_path: Path):
        """Extract info for specific dataclass names only."""
        file_path = tmp_path / "sample.py"
        file_path.write_text(
            textwrap.dedent(
                """
                from dataclasses import dataclass

                @dataclass
                class User:
                    name: str
                    age: int

                @dataclass
                class Config:
                    timeout: int = 30
                    retries: int = 3
                """
            )
        )
        info = merge.extract_dataclass_info(file_path, target_classes=["Config"])
        assert len(info) == 1  # Only Config should be found
        class_name, fields = info[0]
        assert class_name == "Config"
        field_names = [f[0] for f in fields]
        assert "timeout" in field_names
        assert "retries" in field_names

    def test_extract_dataclass_info_all_classes(self, tmp_path: Path):
        """Extract info for all dataclasses when target_classes is None."""
        file_path = tmp_path / "sample.py"
        file_path.write_text(
            textwrap.dedent(
                """
                from dataclasses import dataclass

                @dataclass
                class User:
                    name: str

                @dataclass
                class Config:
                    timeout: int = 30
                """
            )
        )
        info = merge.extract_dataclass_info(file_path, target_classes=None)
        assert len(info) == 2  # Both classes should be found
        class_names = [class_name for class_name, _ in info]
        assert "User" in class_names
        assert "Config" in class_names

    def test_extract_dataclass_info_missing_class(self, tmp_path: Path):
        """Raise error when target class not found."""
        file_path = tmp_path / "sample.py"
        file_path.write_text(
            textwrap.dedent(
                """
                from dataclasses import dataclass

                @dataclass
                class User:
                    name: str
                """
            )
        )
        with pytest.raises(ValueError, match=r"(?i).*not found.*NonExistent"):
            merge.extract_dataclass_info(file_path, target_classes=["NonExistent"])

    @pytest.mark.parametrize(
        "sentinel,expected_wrapper",
        [
            pytest.param("UNSET", "ifnotunset", id="unset_sentinel"),
            pytest.param("None", "ifnotnone", id="none_sentinel"),
            pytest.param("MISSING", "ifnotmissing", id="missing_sentinel"),
            pytest.param("DEFAULT", "ifnotdefault", id="default_sentinel"),
        ],
    )
    def test_sentinel_wrapper_mapping(self, sentinel: str, expected_wrapper: str):
        """Test that sentinels map to correct wrapper functions."""
        fields_info = [("field", "str")]
        implementation = merge.generate_merge_implementation(
            "TestClass", fields_info, sentinel=sentinel, include_docs=False
        )
        assert f"field: str = {sentinel}" in implementation
        assert f"{expected_wrapper}(field, default=self.field)" in implementation

    def test_list_dataclasses(self, tmp_path: Path):
        """List all dataclass names in a file."""
        file_path = tmp_path / "sample.py"
        file_path.write_text(
            textwrap.dedent(
                """
                from dataclasses import dataclass

                class RegularClass:
                    pass

                @dataclass
                class User:
                    name: str

                @dataclass
                class Config:
                    timeout: int = 30
                """
            )
        )
        dataclass_names = merge.list_dataclasses(file_path)
        assert dataclass_names == ["Config", "User"]  # Should be sorted

    def test_main_missing_arguments(self, tmp_path: Path, capsys):
        """Run main() without --classes or --all arguments."""
        file_path = tmp_path / "data.py"
        file_path.write_text(
            textwrap.dedent(
                """
                from dataclasses import dataclass

                @dataclass
                class User:
                    name: str
                """
            )
        )
        args = type(
            "Args",
            (),
            {
                "files": [str(file_path)],
                "sentinel": "UNSET",
                "classes": None,
                "all": False,
                "exclude_private": True,
                "no_docs": False,
                "no_color": True,
                "output": None,
            },
        )()

        with pytest.raises(SystemExit):
            merge.main(args)

        out, err = capsys.readouterr()
        assert "Must specify either --classes" in err

    def test_main_conflicting_arguments(self, tmp_path: Path, capsys):
        """Run main() with both --classes and --all arguments."""
        file_path = tmp_path / "data.py"
        file_path.write_text("# empty file")
        args = type(
            "Args",
            (),
            {
                "files": [str(file_path)],
                "sentinel": "UNSET",
                "classes": ["User"],
                "all": True,
                "exclude_private": True,
                "no_docs": False,
                "no_color": True,
                "output": None,
            },
        )()

        with pytest.raises(SystemExit):
            merge.main(args)

        out, err = capsys.readouterr()
        assert "Cannot use both --classes and --all" in err


class TestMergeExtra:
    def test_unsupported_sentinel_raise(self):
        """Raise on unsupported sentinel."""
        with pytest.raises(ValueError, match=r"(?i).*unsupported sentinel.*"):
            merge.generate_merge_implementation(
                "C",
                [("a", "int")],
                sentinel="BAD_SENTINEL",
            )

    def test_exclude_private_skips_all(self):
        """Skip when only private fields present and exclusion enabled."""
        result = merge.generate_merge_implementation(
            "SecretClass",
            [("_secret", "int")],
            exclude_private=True,
        )
        assert "SKIPPING: No mergeable fields found for SecretClass" in result

    def test_include_and_docs_and_wrapper(self):
        """Include filter appears in docstring and correct wrapper used."""
        impl = merge.generate_merge_implementation(
            "MyClass",
            [("a", "int"), ("b", "str")],
            sentinel="UNSET",
            include=["a"],
            exclude_private=False,
            include_docs=True,
        )
        # wrapper for UNSET is ifnotunset
        assert "ifnotunset(" in impl
        # config note about include should be present
        assert "Only these fields can be merged: a" in impl
        # ensure only a present as parameter
        assert "a: int = UNSET" in impl
        assert "b: str = UNSET" not in impl

    def test_no_docs_short_docstring(self):
        """Omit detailed docs when include_docs is False."""
        impl = merge.generate_merge_implementation(
            "Other",
            [("x", "float")],
            sentinel="UNSET",
            include=None,
            exclude_private=False,
            include_docs=False,
        )
        assert "Create new Other with merged fields" in impl

    def test_extract_dataclass_info_parse_error(self):
        """Raise ValueError when file cannot be parsed."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.py"
            p.write_text("def :\n")
            with pytest.raises(ValueError, match=r"(?i).*could not parse.*"):
                merge.extract_dataclass_info(p)

    def test_extract_dataclass_info_missing_target(self):
        """Raise when specified target_classes are not found."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "one.py"
            p.write_text(
                textwrap.dedent(
                    """
                    from dataclasses import dataclass

                    @dataclass
                    class Present:
                        a: int
                    """
                )
            )
            with pytest.raises(ValueError, match=r"(?i).*dataclasses.*not.*found.*"):
                # ask for a missing class name to trigger the error
                merge.extract_dataclass_info(p, target_classes=["Missing"])

    def test_list_and_extract_various_decorators_and_annotations(self):
        """List dataclasses and extract a variety of annotation AST nodes."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "many.py"
            p.write_text(
                textwrap.dedent(
                    """
                    from dataclasses import dataclass
                    import typing as t

                    @dataclass
                    class A:
                        x: "StrType"
                        y: t.List[int]
                        z: int | str
                        pair: (int, str)
                        nn = 5

                    @dataclass()
                    class B:
                        b: float

                    @t.dataclass
                    class C:
                        c: t.Dict[str, int]

                    @t.dataclass()
                    class D:
                        d: int
                    """
                )
            )
            names = merge.list_dataclasses(p)
            assert names == sorted(["A", "B", "C", "D"])

            infos = merge.extract_dataclass_info(p, target_classes=None)
            info_map = {name: fields for name, fields in infos}

            # Check class A fields and various annotation string forms
            assert "A" in info_map
            a_fields = dict(info_map["A"])
            # x was a string literal annotation
            assert a_fields["x"] == "StrType"
            # y should show the attribute + subscript form (t.List[int])
            assert a_fields["y"] == "t.List[int]"
            # z uses | union syntax
            assert a_fields["z"] == "int | str"
            # pair is a Tuple AST
            assert a_fields["pair"] == "(int, str)"
            # nn was a plain assignment without annotation => Any
            assert a_fields["nn"] == "Any"

    def test_list_syntax_error_raises(self):
        """Raise ValueError for invalid python when listing dataclasses."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "broken.py"
            p.write_text("class\n")
            with pytest.raises(ValueError, match=r"(?i).*could not parse.*"):
                merge.list_dataclasses(p)

    def test_generate_merge_invalid_sentinel(self):
        """Raise ValueError for unsupported sentinel."""
        with pytest.raises(ValueError, match=r"(?i).*unsupported sentinel.*"):
            merge.generate_merge_implementation("X", [("a", "int")], sentinel="BAD")

    @pytest.mark.parametrize(
        "fields,include,exclude,exclude_private,expected_sub",
        [
            pytest.param([("_a", "int")], None, None, True, "SKIPPING", id="private_skip"),
            pytest.param([("a", "int")], ["b"], None, False, "SKIPPING", id="include_skip"),
            pytest.param([("a", "int")], None, ["a"], False, "SKIPPING", id="exclude_skip"),
        ],
    )
    def test_generate_merge_skip_conditions(
        self, fields, include, exclude, exclude_private, expected_sub
    ):
        """Return skip message when no mergeable fields."""
        result = merge.generate_merge_implementation(
            "C", fields, include=include, exclude=exclude, exclude_private=exclude_private
        )
        assert expected_sub in result

    def test_generate_merge_with_docs_and_config(self):
        """Generate merge with docs and config info."""
        fields = [("a", "int"), ("b", "str")]
        result = merge.generate_merge_implementation(
            "C",
            fields,
            include=["a"],
            exclude=["b"],
            exclude_private=True,
            include_docs=True,
        )
        assert "Only these fields can be merged" in result
        assert "These fields are excluded" in result
        assert "Private fields" in result

    def test_generate_merge_without_docs(self):
        """Generate merge without docs."""
        fields = [("a", "int")]
        result = merge.generate_merge_implementation("C", fields, include_docs=False)
        assert '"""Create new C with merged fields."""' in result

    def test_ast_to_string_constant_non_str(self):
        """Convert constant non-string to repr."""
        node = ast.Constant(value=123)
        assert merge._ast_to_string(node) == "123"

    def test_ast_to_string_attribute_and_subscript(self):
        """Convert attribute and subscript nodes."""
        attr = ast.Attribute(value=ast.Name(id="x"), attr="y")
        sub = ast.Subscript(value=ast.Name(id="List"), slice=ast.Name(id="int"))
        assert merge._ast_to_string(attr) == "x.y"
        assert merge._ast_to_string(sub) == "List[int]"

    def test_ast_to_string_union_and_tuple(self):
        """Convert union and tuple nodes."""
        node_union = ast.BinOp(left=ast.Name(id="A"), op=ast.BitOr(), right=ast.Name(id="B"))
        node_tuple = ast.Tuple(elts=[ast.Name(id="A"), ast.Name(id="B")])
        assert merge._ast_to_string(node_union) == "A | B"
        assert merge._ast_to_string(node_tuple) == "(A, B)"

    def test_ast_to_string_fallback_unparse(self):
        """Fallback to ast.unparse for unknown node."""
        node = ast.BinOp(left=ast.Constant(1), op=ast.Add(), right=ast.Constant(2))
        result = merge._ast_to_string(node)
        assert "1" in result and "2" in result

    def test_is_dataclass_decorator_variants(self):
        """Recognize various dataclass decorator forms."""
        assert merge._is_dataclass_decorator(ast.Name(id="dataclass"))
        assert merge._is_dataclass_decorator(
            ast.Attribute(value=ast.Name(id="x"), attr="dataclass")
        )
        call = ast.Call(func=ast.Name(id="dataclass"), args=[], keywords=[])
        assert merge._is_dataclass_decorator(call)
        call_attr = ast.Call(
            func=ast.Attribute(value=ast.Name(id="x"), attr="dataclass"), args=[], keywords=[]
        )
        assert merge._is_dataclass_decorator(call_attr)
        assert not merge._is_dataclass_decorator(ast.Constant(1))

    def test_extract_fields_from_class_annassign_and_assign(self):
        """Extract annotated and assigned fields."""
        class_node = ast.ClassDef(name="C", bases=[], keywords=[], body=[], decorator_list=[])
        ann = ast.AnnAssign(
            target=ast.Name(id="a"), annotation=ast.Name(id="int"), value=None, simple=1
        )
        assign = ast.Assign(targets=[ast.Name(id="b")], value=ast.Constant(1))
        class_node.body = [ann, assign]
        result = merge._extract_fields_from_class(class_node)
        assert ("a", "int") in result and ("b", "Any") in result

    def test_extract_dataclass_info_missing_class(self, tmp_path):
        """Raise ValueError for missing target class."""
        code = "@dataclass\nclass A:\n a:int"
        p = tmp_path / "f.py"
        p.write_text(code)
        with pytest.raises(ValueError, match=r"(?i).*not found.*"):
            merge.extract_dataclass_info(p, target_classes=["B"])

    def test_extract_dataclass_info_parse_error(self, tmp_path):
        """Raise ValueError for syntax error."""
        p = tmp_path / "bad.py"
        p.write_text("class A(: pass")
        with pytest.raises(ValueError, match=r"(?i).*could not parse.*"):
            merge.extract_dataclass_info(p)

    def test_list_dataclasses_and_parse_error(self, tmp_path):
        """List dataclasses and handle parse error."""
        code = "@dataclass\nclass A:\n a:int"
        p = tmp_path / "f.py"
        p.write_text(code)
        assert merge.list_dataclasses(p) == ["A"]
        bad = tmp_path / "bad.py"
        bad.write_text("class A(: pass")
        with pytest.raises(ValueError):
            merge.list_dataclasses(bad)

    def test_main_no_args(self, capsys):
        """Exit when no args provided."""
        args = types.SimpleNamespace(classes=[], all=False)
        with pytest.raises(SystemExit):
            merge.main(args)
        out = capsys.readouterr().err
        assert "Must specify" in out

    def test_main_both_args(self, capsys):
        """Exit when both classes and all provided."""
        args = types.SimpleNamespace(classes=["A"], all=True)
        with pytest.raises(SystemExit):
            merge.main(args)
        out = capsys.readouterr().err
        assert "Cannot use both" in out

    def test_main_file_not_found(self, tmp_path, capsys):
        """Handle missing file gracefully."""
        args = types.SimpleNamespace(
            classes=["A"],
            all=False,
            files=[str(tmp_path / "nofile.py")],
            sentinel="UNSET",
            exclude_private=True,
            no_docs=False,
            output=None,
            no_color=True,
        )
        merge.main(args)
        err = capsys.readouterr().err
        assert "not found" in err

    def test_main_all_no_dataclasses(self, tmp_path, capsys):
        """Handle --all with no dataclasses."""
        p = tmp_path / "f.py"
        p.write_text("class A: pass")
        args = types.SimpleNamespace(
            classes=[],
            all=True,
            files=[str(p)],
            sentinel="UNSET",
            exclude_private=True,
            no_docs=False,
            output=None,
            no_color=True,
        )
        merge.main(args)
        err = capsys.readouterr().err
        assert "No dataclasses found" in err

    def test_main_all_with_dataclass(self, tmp_path, capsys):
        """Generate merge for all dataclasses."""
        code = "from dataclasses import dataclass\n@dataclass\nclass A:\n a:int"
        p = tmp_path / "f.py"
        p.write_text(code)
        args = types.SimpleNamespace(
            classes=[],
            all=True,
            files=[str(p)],
            sentinel="UNSET",
            exclude_private=True,
            no_docs=False,
            output=None,
            no_color=True,
        )
        merge.main(args)
        out = capsys.readouterr().out
        assert "Implementation for A" in out or "Merge implementations" in out

    def test_main_classes_with_output_file(self, tmp_path, capsys):
        """Write output to file when output specified."""
        code = "from dataclasses import dataclass\n@dataclass\nclass A:\n a:int"
        p = tmp_path / "f.py"
        p.write_text(code)
        out_file = tmp_path / "out.txt"
        args = types.SimpleNamespace(
            classes=["A"],
            all=False,
            files=[str(p)],
            sentinel="UNSET",
            exclude_private=True,
            no_docs=False,
            output=str(out_file),
            no_color=True,
        )
        merge.main(args)
        assert out_file.exists()
        content = out_file.read_text()
        assert "Implementation for A" in content

    def test_main_valueerror_and_available_classes(self, tmp_path, capsys):
        """Handle ValueError and list available dataclasses."""
        code = "from dataclasses import dataclass\n@dataclass\nclass A:\n a:int"
        p = tmp_path / "f.py"
        p.write_text(code)
        args = types.SimpleNamespace(
            classes=["B"],
            all=False,
            files=[str(p)],
            sentinel="UNSET",
            exclude_private=True,
            no_docs=False,
            output=None,
            no_color=True,
        )
        merge.main(args)
        err = capsys.readouterr().err
        assert "Available dataclasses" in err or "Error:" in err

    def test_main_no_output_lines(self, tmp_path, capsys):
        """Print no implementations generated when nothing produced."""
        code = "from dataclasses import dataclass\n@dataclass\nclass A:\n a:int"
        p = tmp_path / "f.py"
        p.write_text(code)
        args = types.SimpleNamespace(
            classes=["A"],
            all=False,
            files=[str(p)],
            sentinel="BAD_SENTINEL",
            exclude_private=True,
            no_docs=False,
            output=None,
            no_color=True,
        )
        merge.main(args)
        out = capsys.readouterr().out
        assert "No implementations generated" in out or "Error generating" in out
