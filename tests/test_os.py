#
# C108 - os Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import datetime as dt
import errno
import os
from pathlib import Path

# Third Party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
import c108.os as c108_os
from c108.os import atomic_open, tail_file


# Tests ----------------------------------------------------------------------------------------------------------------


def _freeze_utc_now(monkeypatch: pytest.MonkeyPatch, fixed: dt.datetime) -> None:
    """Patch c108.os.datetime.now(...) to return fixed UTC datetime."""
    if fixed.tzinfo is None:
        fixed = fixed.replace(tzinfo=dt.timezone.utc)

    class FixedDateTime(dt.datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            if tz is None:
                return fixed.replace(tzinfo=None)
            return fixed.astimezone(tz)

    monkeypatch.setattr("c108.os.datetime", FixedDateTime, raising=True)


@pytest.fixture()
def populated_dir(tmp_path: Path) -> Path:
    """Create a directory with mixed contents (files and nested subdirectories)."""
    root = tmp_path / "root"
    root.mkdir()
    # Top-level files
    (root / "a.txt").write_text("A")
    (root / "b.log").write_text("B")
    # Nested directory with files
    sub = root / "sub"
    sub.mkdir()
    (sub / "c.dat").write_text("C")
    (sub / "d.bin").write_bytes(b"\x00\x01")
    # Deeper nesting
    deep = sub / "deep"
    deep.mkdir()
    (deep / "e.txt").write_text("E")
    return root


@pytest.fixture()
def src_file(tmp_path: Path) -> Path:
    """Create a temporary source file with initial content."""
    p = tmp_path / "config.txt"
    p.write_text("alpha")
    return p


class TestAtomicOpen:
    def test_text_atomic_write_renames_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Write text atomically and ensure final file content replaces target."""
        path = tmp_path / "text.txt"
        original_content = "old"
        new_content = "new content"
        path.write_text(original_content, encoding="utf-8")

        calls = {"count": 0}
        real_replace = c108_os.os.replace

        def spy_replace(
            src: str | os.PathLike[str], dst: str | os.PathLike[str]
        ) -> None:
            calls["count"] += 1
            return real_replace(src, dst)

        monkeypatch.setattr(c108_os.os, "replace", spy_replace)

        with atomic_open(
            path=path,
            mode="w",
            encoding="utf-8",
            newline="\n",
            temp_dir=None,
            overwrite=True,
            fsync=False,
        ) as f:
            f.write(new_content)

        assert path.read_text(encoding="utf-8") == new_content
        assert calls["count"] == 1

    def test_binary_atomic_write_bytes(self, tmp_path: Path) -> None:
        """Write bytes atomically in binary mode."""
        path = tmp_path / "data.bin"
        payload = b"\x00\x01abc\xff"

        with atomic_open(
            path=path,
            mode="wb",
            temp_dir=None,
            overwrite=True,
            fsync=False,
        ) as f:
            f.write(payload)

        assert path.read_bytes() == payload

    def test_overwrite_false_blocks_existing(self, tmp_path: Path) -> None:
        """Raise FileExistsError when target exists with overwrite disabled."""
        path = tmp_path / "existing.txt"
        path.write_text("keep", encoding="utf-8")

        with pytest.raises(FileExistsError, match=r"(?i).*(exist|already).*"):
            with atomic_open(
                path=path,
                mode="w",
                encoding="utf-8",
                newline="\n",
                temp_dir=None,
                overwrite=False,
                fsync=False,
            ) as f:
                f.write("should not write")

        assert path.read_text(encoding="utf-8") == "keep"

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permissions semantics required")
    def test_preserve_permissions_on_overwrite(self, tmp_path: Path) -> None:
        """Preserve original permissions when overwriting existing file."""
        path = tmp_path / "perm.txt"
        path.write_text("orig", encoding="utf-8")
        original_mode = 0o640
        os.chmod(path, original_mode)

        with atomic_open(
            path=path,
            mode="w",
            encoding="utf-8",
            newline="\n",
            temp_dir=None,
            overwrite=True,
            fsync=False,
        ) as f:
            f.write("updated")

        after_mode = Path(path).stat().st_mode & 0o777
        assert after_mode == original_mode, (
            f"Expected mode {oct(original_mode)}, got {oct(after_mode)}"
        )
        assert path.read_text(encoding="utf-8") == "updated"

    @pytest.mark.parametrize(
        "bad_mode",
        [
            pytest.param("a", id="append"),
            pytest.param("w+", id="read_write"),
            pytest.param("wtb", id="wtb"),
            pytest.param("rb", id="rb"),
            pytest.param("rt", id="rt"),
            pytest.param("", id="empty"),
            pytest.param("wb+", id="wb_plus"),
            pytest.param("x", id="x"),
            pytest.param("xt", id="xt"),
            pytest.param("xb", id="xb"),
        ],
        ids=[
            "append",
            "read_write",
            "wtb",
            "rb",
            "rt",
            "empty",
            "wb_plus",
            "x",
            "xt",
            "xb",
        ],
    )
    def test_invalid_mode_rejected(self, tmp_path: Path, bad_mode: str) -> None:
        """Reject unsupported modes with ValueError."""
        path = tmp_path / "invalid_mode.txt"
        with pytest.raises(ValueError, match=r"(?i).*mode.*"):
            with atomic_open(
                path=path,
                mode=bad_mode,
                encoding="utf-8",
                newline="\n",
                temp_dir=None,
                overwrite=True,
                fsync=False,
            ):
                pass

    def test_temp_dir_on_different_fs_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Raise OSError when temp_dir is on different filesystem."""
        path = tmp_path / "cross.txt"
        temp_dir = tmp_path / "tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        def fake_replace(
            src: str | os.PathLike[str], dst: str | os.PathLike[str]
        ) -> None:
            raise OSError(errno.EXDEV, "Invalid cross-device link")

        monkeypatch.setattr(c108_os.os, "replace", fake_replace)

        with pytest.raises(OSError, match=r"(?i).*cross.*"):
            with atomic_open(
                path=path,
                mode="w",
                encoding="utf-8",
                newline="\n",
                temp_dir=temp_dir,
                overwrite=True,
                fsync=False,
            ) as f:
                f.write("data")

        assert not path.exists()

    def test_fsync_when_enabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Call fsync before rename when fsync enabled."""
        path = tmp_path / "fsync.txt"
        calls: list[int] = []
        real_fsync = c108_os.os.fsync

        def spy_fsync(fd: int) -> None:
            calls.append(fd)
            return real_fsync(fd)

        monkeypatch.setattr(c108_os.os, "fsync", spy_fsync)

        with atomic_open(
            path=path,
            mode="w",
            encoding="utf-8",
            newline="\n",
            temp_dir=None,
            overwrite=True,
            fsync=True,
        ) as f:
            f.write("with fsync")

        assert len(calls) >= 1
        assert all(isinstance(fd, int) for fd in calls)
        assert path.read_text(encoding="utf-8") == "with fsync"

    def test_error_during_write_rolls_back(self, tmp_path: Path) -> None:
        """Do not replace target when exception occurs inside context."""
        path = tmp_path / "rollback.txt"
        original = "safe"
        path.write_text(original, encoding="utf-8")

        with pytest.raises(RuntimeError, match=r"(?i).*boom.*"):
            with atomic_open(
                path=path,
                mode="w",
                encoding="utf-8",
                newline="\n",
                temp_dir=None,
                overwrite=True,
                fsync=False,
            ) as f:
                f.write("partial")
                raise RuntimeError("boom")

        assert path.read_text(encoding="utf-8") == original

    @pytest.mark.parametrize(
        "newline,expected",
        [
            pytest.param("", "A\nB\n", id="no_translation"),
            pytest.param("\n", "A\nB\n", id="lf"),
            pytest.param("\r", "A\rB\r", id="cr"),
            pytest.param("\r\n", "A\r\nB\r\n", id="crlf"),
        ],
        ids=["no_translation", "lf", "cr", "crlf"],
    )
    def test_newline_translation_variants(
        self, tmp_path: Path, newline: str, expected: str
    ) -> None:
        """Honor provided newline options in text mode."""
        path = tmp_path / f"nl_{len(newline)}.txt"
        content = "A\nB\n"

        with atomic_open(
            path=path,
            mode="w",
            encoding="utf-8",
            newline=newline,
            temp_dir=None,
            overwrite=True,
            fsync=False,
        ) as f:
            f.write(content)

        assert path.read_bytes() == expected.encode("utf-8")

    def test_temp_dir_usage(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Create temp file in provided temp_dir."""
        path = tmp_path / "tempdir.txt"
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir(parents=True, exist_ok=True)

        recorded_dirs: list[Path] = []

        real_mkstemp = c108_os.tempfile.mkstemp
        real_namedtemp = c108_os.tempfile.NamedTemporaryFile

        def spy_mkstemp(*args, **kwargs):
            d = kwargs.get("dir", None)
            if d is not None:
                recorded_dirs.append(Path(d))
            return real_mkstemp(*args, **kwargs)

        def spy_namedtemp(*args, **kwargs):
            d = kwargs.get("dir", None)
            if d is not None:
                recorded_dirs.append(Path(d))
            return real_namedtemp(*args, **kwargs)

        monkeypatch.setattr(c108_os.tempfile, "mkstemp", spy_mkstemp)
        monkeypatch.setattr(c108_os.tempfile, "NamedTemporaryFile", spy_namedtemp)

        with atomic_open(
            path=path,
            mode="w",
            encoding="utf-8",
            newline="\n",
            temp_dir=custom_dir,
            overwrite=True,
            fsync=False,
        ) as f:
            f.write("x")

        assert recorded_dirs, "No tempfile creation was observed"
        assert all(d.resolve() == custom_dir.resolve() for d in recorded_dirs)
        assert path.read_text(encoding="utf-8") == "x"

    @pytest.mark.parametrize(
        "use_binary",
        [
            pytest.param(False, id="text_mode"),
            pytest.param(True, id="binary_mode"),
        ],
        ids=["text_mode", "binary_mode"],
    )
    def test_pathlike_support(self, tmp_path: Path, use_binary: bool) -> None:
        """Accept PathLike for path and temp_dir arguments."""
        target_dir = tmp_path / "dir"
        target_dir.mkdir()
        path = target_dir / ("p.bin" if use_binary else "p.txt")
        temp_dir = tmp_path / "t"
        temp_dir.mkdir()

        if use_binary:
            payload_b = b"\x10\x11\x12"
            with atomic_open(
                path=path,
                mode="wb",
                temp_dir=temp_dir,
                overwrite=True,
                fsync=False,
            ) as f:
                f.write(payload_b)
            assert path.read_bytes() == payload_b
        else:
            payload_s = "hello pathlike"
            with atomic_open(
                path=path,
                mode="w",
                encoding="utf-8",
                newline="\n",
                temp_dir=temp_dir,
                overwrite=True,
                fsync=False,
            ) as f:
                f.write(payload_s)
            assert path.read_text(encoding="utf-8") == payload_s

    def test_invalid_newline_rejected(self, tmp_path: Path) -> None:
        """Reject invalid newline value with ValueError."""
        path = tmp_path / "bad_nl.txt"
        with pytest.raises(ValueError, match=r"(?i).*newline.*"):
            with atomic_open(
                path=path,
                mode="w",
                encoding="utf-8",
                newline="invalid",
                temp_dir=None,
                overwrite=True,
                fsync=False,
            ):
                pass

    @pytest.mark.skipif(
        os.name == "nt",
        reason="Windows permission model differs; use POSIX for reliable denial",
    )
    def test_permission_errors_propagate(self, tmp_path: Path) -> None:
        """Propagate permission denied when directory not writable."""
        ro_dir = tmp_path / "ro"
        ro_dir.mkdir()
        try:
            os.chmod(ro_dir, 0o555)
            path = ro_dir / "f.txt"
            with pytest.raises(PermissionError, match=r"(?i).*(permission|denied).*"):
                with atomic_open(
                    path=path,
                    mode="w",
                    encoding="utf-8",
                    newline="\n",
                    temp_dir=None,
                    overwrite=True,
                    fsync=False,
                ) as f:
                    f.write("x")
        finally:
            os.chmod(ro_dir, 0o755)

    def test_directory_autocreated(self, tmp_path: Path) -> None:
        """Parent directories are created automatically if missing."""
        missing_dir = tmp_path / "missing" / "nested"
        path = missing_dir / "file.txt"

        assert not missing_dir.exists()

        with atomic_open(path) as f:
            f.write("content")

        assert path.exists()
        assert path.read_text() == "content"


class TestTailFile:
    def test_small_text_tail(self, tmp_path: Path) -> None:
        """Return last lines for a small text file."""
        p = tmp_path / "small.txt"
        lines = [f"line{i}\n" for i in range(1, 6)]
        p.write_text("".join(lines), encoding="utf-8")
        out = tail_file(p, n=3, encoding="utf-8", errors="strict")
        assert out == "".join(lines[-3:])

    def test_large_text_tail(self, tmp_path: Path) -> None:
        """Return last lines for a large text file efficiently."""
        p = tmp_path / "large.txt"
        # Create >8KB content to trigger large-file path
        lines = [("X" * 100) + f"_{i}\n" for i in range(120)]
        p.write_text("".join(lines), encoding="utf-8")
        out = tail_file(p, n=10, encoding="utf-8", errors="strict")
        assert out == "".join(lines[-10:])

    @pytest.mark.parametrize(
        ("encoding", "expected"),
        [
            pytest.param("utf-8", "", id="text-empty"),
            pytest.param(None, b"", id="bytes-empty"),
        ],
    )
    def test_n_zero(
        self, tmp_path: Path, encoding: str | None, expected: str | bytes
    ) -> None:
        """Return empty result when n is zero."""
        # Note: function returns early; file need not exist
        p = tmp_path / "anything.txt"
        out = tail_file(p, n=0, encoding=encoding, errors="strict")
        assert out == expected

    def test_negative_n(self) -> None:
        """Raise on negative n."""
        with pytest.raises(ValueError, match=r"(?i).*non-negative.*"):
            tail_file("irrelevant.txt", n=-1, encoding="utf-8", errors="strict")

    def test_missing_path(self, tmp_path: Path) -> None:
        """Raise on nonexistent path."""
        p = tmp_path / "missing.txt"
        with pytest.raises(FileNotFoundError, match=r"(?i).*file not found.*"):
            tail_file(p, n=1, encoding="utf-8", errors="strict")

    def test_directory_path(self, tmp_path: Path) -> None:
        """Raise on directory path."""
        d = tmp_path / "dir"
        d.mkdir()
        with pytest.raises(IsADirectoryError, match=r"(?i).*directory.*"):
            tail_file(d, n=1, encoding="utf-8", errors="strict")

    def test_binary_mode(self, tmp_path: Path) -> None:
        """Return bytes in binary mode."""
        p = tmp_path / "data.bin"
        content = b"aa\nbb\ncc\n"
        p.write_bytes(content)
        out = tail_file(p, n=2, encoding=None, errors="strict")
        assert out == b"bb\ncc\n"

    def test_no_trailing_newline(self, tmp_path: Path) -> None:
        """Handle last line without trailing newline."""
        p = tmp_path / "no_newline.txt"
        p.write_text("a\nb\nc", encoding="utf-8")
        out = tail_file(p, n=2, encoding="utf-8", errors="strict")
        assert out == "b\nc"

    def test_n_exceeds_total(self, tmp_path: Path) -> None:
        """Return all lines when n exceeds count."""
        p = tmp_path / "few.txt"
        lines = ["one\n", "two\n", "three\n"]
        p.write_text("".join(lines), encoding="utf-8")
        out = tail_file(p, n=10, encoding="utf-8", errors="strict")
        assert out == "".join(lines)

    def test_empty_file(self, tmp_path: Path) -> None:
        """Return empty for empty file."""
        p = tmp_path / "empty.txt"
        p.write_text("", encoding="utf-8")
        out = tail_file(p, n=5, encoding="utf-8", errors="strict")
        assert out == ""

    def test_encoding_strict_raises(self, tmp_path: Path) -> None:
        """Raise UnicodeDecodeError on invalid bytes with strict errors."""
        p = tmp_path / "bad.txt"
        # Include invalid UTF-8 bytes in a middle line
        bad = b"ok1\nbad:\xff\xfe\nok2\n"
        p.write_bytes(bad)
        with pytest.raises(UnicodeDecodeError, match=r"(?i).*decode.*"):
            tail_file(p, n=3, encoding="utf-8", errors="strict")

    def test_encoding_replace(self, tmp_path: Path) -> None:
        """Replace invalid bytes when errors='replace'."""
        p = tmp_path / "bad_replace.txt"
        p.write_bytes(b"ok1\nbad:\xff\xfe\nok2\n")
        out = tail_file(p, n=3, encoding="utf-8", errors="replace")
        assert out == "ok1\nbad:��\nok2\n"

    def test_encoding_ignore(self, tmp_path: Path) -> None:
        """Ignore invalid bytes when errors='ignore'."""
        p = tmp_path / "bad_ignore.txt"
        p.write_bytes(b"ok1\nbad:\xff\xfe\nok2\n")
        out = tail_file(p, n=3, encoding="utf-8", errors="ignore")
        assert out == "ok1\nbad:\nok2\n"

    def test_encoding_backslashreplace(self, tmp_path: Path) -> None:
        """Escape invalid bytes when errors='backslashreplace'."""
        p = tmp_path / "bad_backslash.txt"
        p.write_bytes(b"ok1\nbad:\xff\xfe\nok2\n")
        out = tail_file(p, n=3, encoding="utf-8", errors="backslashreplace")
        assert out == "ok1\nbad:\\xff\\xfe\nok2\n"
