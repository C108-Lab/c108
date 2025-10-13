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
