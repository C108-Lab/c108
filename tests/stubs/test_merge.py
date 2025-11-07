"""
Test suite for c108.stubs.merge module.
"""

import io
import sys
import textwrap
from pathlib import Path
import pytest

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
