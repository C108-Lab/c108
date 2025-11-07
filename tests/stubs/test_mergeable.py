"""Test suite for c108.stubs.mergeable module."""

import io
import sys
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
