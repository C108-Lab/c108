#
# C108 - IO Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import os
import threading
from typing import Callable

# Third Party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.json import read_json, write_json, update_json

# Tests ----------------------------------------------------------------------------------------------------------------

import builtins
import json as py_json
from typing import Any

class TestReadJson:
    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param({"a": 1, "b": [1, 2]}, id="dict"),
            pytest.param([1, 2, 3], id="list"),
        ],
    )
    def test_read_ok(self, tmp_path, payload: Any) -> None:
        """Read valid JSON and return parsed data."""
        p = tmp_path / "data.json"
        with p.open("w", encoding="utf-8") as f:
            py_json.dump(payload, f)

        result = read_json(str(p), default={"fallback": True}, encoding="utf-8")
        assert result == payload

    def test_missing_default(self, tmp_path) -> None:
        """Return default when file is missing."""
        p = tmp_path / "missing.json"
        sentinel = object()

        result = read_json(str(p), default=sentinel, encoding="utf-8")
        assert result is sentinel

    def test_invalid_json_default(self, tmp_path) -> None:
        """Return default when JSON is invalid."""
        p = tmp_path / "bad.json"
        with p.open("w", encoding="utf-8") as f:
            f.write("{ invalid json")

        default_list: list[int] = []
        result = read_json(str(p), default=default_list, encoding="utf-8")
        assert result is default_list

    def test_oserror_propagates(self, tmp_path, monkeypatch) -> None:
        """Propagate OSError when file is unreadable."""
        p = tmp_path / "unreadable.json"
        p.write_text("{}", encoding="utf-8")

        orig_open = builtins.open

        def fake_open(file, mode="r", encoding=None, *args, **kwargs):
            if str(file) == str(p):
                raise PermissionError("blocked by policy")
            return orig_open(file, mode, encoding=encoding, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", fake_open)

        with pytest.raises(OSError, match=r"(?i).*blocked by policy.*"):
            read_json(str(p), default={"unused": True}, encoding="utf-8")

    def test_invalid_path_type(self) -> None:
        """Raise TypeError when path is not path-like."""
        with pytest.raises(TypeError, match=r"(?i).*os\.PathLike.*"):
            read_json(123, default=None, encoding="utf-8")  # type: ignore[arg-type]

    def test_custom_encoding(self, tmp_path) -> None:
        """Support custom encoding when reading file."""
        p = tmp_path / "latin.json"
        content = {"text": "café"}
        with p.open("w", encoding="latin-1") as f:
            py_json.dump(content, f, ensure_ascii=False)

        result = read_json(str(p), default={}, encoding="latin-1")
        assert result == content


import json
from pathlib import Path


class TestWriteJson:
    @pytest.mark.parametrize(
        "data, indent, atomic, ensure_ascii, encoding",
        [
            pytest.param({"a": 1, "b": [1, 2]}, 4, True, False, "utf-8", id="pretty-atomic"),
            pytest.param({"a": 1, "b": [1, 2]}, 4, False, False, "utf-8", id="pretty-non-atomic"),
            pytest.param({"x": "y", "n": None}, None, True, False, "utf-8", id="compact-atomic"),
            pytest.param({"x": "y", "n": None}, None, False, False, "utf-8", id="compact-non-atomic"),
        ],
    )
    def test_content_matches_dump(self, tmp_path: Path, data, indent, atomic, ensure_ascii, encoding):
        """Write JSON and match json.dumps output with trailing newline."""
        file_path = tmp_path / "out.json"
        write_json(
            path=str(file_path),
            data=data,
            indent=indent,
            atomic=atomic,
            encoding=encoding,
            ensure_ascii=ensure_ascii,
        )
        content = file_path.read_text(encoding=encoding)
        expected = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii) + "\n"
        assert content == expected

    def test_unicode_written_when_not_escaped(self, tmp_path: Path):
        """Write Unicode characters when ensure_ascii is False."""
        file_path = tmp_path / "unicode.json"
        data = {"name": "François", "emoji": "👍"}
        write_json(
            path=str(file_path),
            data=data,
            indent=2,
            atomic=False,
            encoding="utf-8",
            ensure_ascii=False,
        )
        content = file_path.read_text(encoding="utf-8")
        assert "François" in content
        assert "👍" in content
        assert "\\u00e9" not in content  # 'é' should not be escaped

    def test_unicode_escaped_when_ascii_true(self, tmp_path: Path):
        """Escape non-ASCII characters when ensure_ascii is True."""
        file_path = tmp_path / "ascii_safe.json"
        data = {"name": "François", "emoji": "👍"}
        write_json(
            path=str(file_path),
            data=data,
            indent=2,
            atomic=True,
            encoding="utf-8",
            ensure_ascii=True,
        )
        content = file_path.read_text(encoding="utf-8")
        assert "François" not in content
        assert "👍" not in content
        # Validate typical escape sequences appear
        assert "\\u00e7" in content or "\\u00C7" in content
        assert "\\ud83d\\udc4d" in content.lower()  # surrogate pair for 👍

    def test_invalid_path_type(self):
        """Raise TypeError for non path-like path argument."""
        with pytest.raises(TypeError, match=r"(?i).*path must be str or os\.PathLike.*"):
            write_json(
                path=123,  # type: ignore[arg-type]
                data={"x": 1},
                indent=2,
                atomic=True,
                encoding="utf-8",
                ensure_ascii=False,
            )

    def test_negative_indent_value(self, tmp_path: Path):
        """Raise ValueError for negative indent."""
        file_path = tmp_path / "out.json"
        with pytest.raises(ValueError, match=r"(?i).*indent must be non-negative.*"):
            write_json(
                path=str(file_path),
                data={"x": 1},
                indent=-1,
                atomic=False,
                encoding="utf-8",
                ensure_ascii=False,
            )

    def test_not_serializable_type(self, tmp_path: Path):
        """Raise TypeError for non-JSON-serializable data."""
        file_path = tmp_path / "bad.json"
        with pytest.raises(TypeError, match=r"(?i).*not json serializable.*"):
            write_json(
                path=str(file_path),
                data={1, 2, 3},  # sets are not JSON-serializable
                indent=2,
                atomic=False,
                encoding="utf-8",
                ensure_ascii=False,
            )

    def test_accept_pathlike(self, tmp_path: Path):
        """Accept pathlib.Path as path-like and write file."""
        file_path = tmp_path / "pathlike.json"
        data = {"ok": True}
        write_json(
            path=file_path,  # pass PathLike directly
            data=data,
            indent=2,
            atomic=True,
            encoding="utf-8",
            ensure_ascii=False,
        )
        content = file_path.read_text(encoding="utf-8")
        assert content.endswith("\n")
        assert json.loads(content) == data

    def test_overwrite_existing_file(self, tmp_path: Path):
        """Overwrite existing file content with new data."""
        file_path = tmp_path / "overwrite.json"
        old_data = {"old": 1}
        new_data = {"new": [1, 2, 3]}
        write_json(
            path=str(file_path),
            data=old_data,
            indent=2,
            atomic=False,
            encoding="utf-8",
            ensure_ascii=False,
        )
        write_json(
            path=str(file_path),
            data=new_data,
            indent=2,
            atomic=True,
            encoding="utf-8",
            ensure_ascii=False,
        )
        content = file_path.read_text(encoding="utf-8")
        assert json.loads(content) == new_data
