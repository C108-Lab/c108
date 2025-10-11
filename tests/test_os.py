#
# C108 - OS Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import datetime as dt
import importlib.util
import sys
from pathlib import Path

# Third Party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.os import backup_file, clean_dir, tail_file


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


class TestBackupFile:
    def test_create_in_dest_dir(self, src_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Create backup in explicit destination with deterministic timestamp."""
        dest_dir = tmp_path / "backups"
        dest_dir.mkdir(parents=True, exist_ok=True)
        fixed = dt.datetime(2024, 10, 11, 14, 30, 22, tzinfo=dt.timezone.utc)
        _freeze_utc_now(monkeypatch, fixed)

        name_format = "{stem}.{timestamp}{suffix}"
        time_format = "%Y%m%d-%H%M%S"
        exist_ok = False

        backup_path = backup_file(
            path=str(src_file),
            dest_dir=str(dest_dir),
            name_format=name_format,
            time_format=time_format,
            exist_ok=exist_ok,
        )
        expected = (dest_dir / "config.20241011-143022.txt").resolve()

        assert backup_path == expected
        assert expected.exists()
        assert expected.read_text() == "alpha"

    def test_invalid_placeholder_raises(self, src_file: Path, tmp_path: Path):
        """Reject invalid placeholders in name_format."""
        dest_dir = tmp_path / "d"
        dest_dir.mkdir()
        with pytest.raises(ValueError, match=r"(?i).*invalid placeholder.*"):
            backup_file(
                path=str(src_file),
                dest_dir=str(dest_dir),
                name_format="{stemm}.{timestamp}{suffix}",
                time_format="%Y%m%d-%H%M%S",
                exist_ok=False,
            )

    def test_missing_source_raises(self, tmp_path: Path):
        """Raise when source file is missing."""
        missing = tmp_path / "missing.txt"
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()
        with pytest.raises(FileNotFoundError, match=r"(?i).*source file not found.*"):
            backup_file(
                path=str(missing),
                dest_dir=str(dest_dir),
                name_format="{stem}.{timestamp}{suffix}",
                time_format="%Y%m%d-%H%M%S",
                exist_ok=False,
            )

    def test_path_is_directory_raises(self, tmp_path: Path):
        """Raise when path points to a directory."""
        src_dir = tmp_path / "src_dir"
        src_dir.mkdir()
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()
        with pytest.raises(IsADirectoryError, match=r"(?i).*directory, not a file.*"):
            backup_file(
                path=str(src_dir),
                dest_dir=str(dest_dir),
                name_format="{stem}.{timestamp}",
                time_format="%Y%m%d",
                exist_ok=False,
            )

    def test_dest_dir_not_exist_raises(self, src_file: Path, tmp_path: Path):
        """Raise when destination directory does not exist."""
        dest_dir = tmp_path / "does_not_exist"
        with pytest.raises(NotADirectoryError, match=r"(?i).*destination directory not found.*"):
            backup_file(
                path=str(src_file),
                dest_dir=str(dest_dir),
                name_format="{stem}.{timestamp}{suffix}",
                time_format="%Y%m%d-%H%M%S",
                exist_ok=False,
            )

    def test_dest_dir_not_a_directory_raises(self, src_file: Path, tmp_path: Path):
        """Raise when destination path is not a directory."""
        not_a_dir = tmp_path / "file_target"
        not_a_dir.write_text("not a dir")
        with pytest.raises(NotADirectoryError, match=r"(?i).*not a directory.*"):
            backup_file(
                path=str(src_file),
                dest_dir=str(not_a_dir),
                name_format="{stem}.{timestamp}{suffix}",
                time_format="%Y%m%d-%H%M%S",
                exist_ok=False,
            )

    def test_backup_exists_exist_ok_false_raises(self, src_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Raise when backup exists and exist_ok is false."""
        dest_dir = tmp_path / "d"
        dest_dir.mkdir()
        fixed = dt.datetime(2024, 1, 2, 3, 4, 5, tzinfo=dt.timezone.utc)
        _freeze_utc_now(monkeypatch, fixed)
        name_format = "{stem}.{timestamp}{suffix}"
        time_format = "%Y%m%d-%H%M%S"

        first = backup_file(
            path=str(src_file),
            dest_dir=str(dest_dir),
            name_format=name_format,
            time_format=time_format,
            exist_ok=True,
        )
        assert first.exists()

        with pytest.raises(FileExistsError, match=r"(?i).*backup file already exists.*"):
            backup_file(
                path=str(src_file),
                dest_dir=str(dest_dir),
                name_format=name_format,
                time_format=time_format,
                exist_ok=False,
            )

    def test_backup_exists_exist_ok_true_overwrites(self, src_file: Path, tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch):
        """Overwrite existing backup when exist_ok is true."""
        dest_dir = tmp_path / "d2"
        dest_dir.mkdir()
        fixed = dt.datetime(2024, 6, 7, 8, 9, 10, tzinfo=dt.timezone.utc)
        _freeze_utc_now(monkeypatch, fixed)
        name_format = "{stem}.{timestamp}{suffix}"
        time_format = "%Y%m%d-%H%M%S"

        backup_path = backup_file(
            path=str(src_file),
            dest_dir=str(dest_dir),
            name_format=name_format,
            time_format=time_format,
            exist_ok=True,
        )
        assert backup_path.exists()
        assert backup_path.read_text() == "alpha"

        src_file.write_text("beta")
        overwritten = backup_file(
            path=str(src_file),
            dest_dir=str(dest_dir),
            name_format=name_format,
            time_format=time_format,
            exist_ok=True,
        )
        assert overwritten == backup_path
        assert backup_path.read_text() == "beta"

    @pytest.mark.parametrize(
        "name_format, expected_name",
        [
            pytest.param(
                "{timestamp}_{name}",
                "20241011-143022_config.txt",
                id="timestamp_prefix_fullname",
            ),
            pytest.param(
                "bak-{stem}{suffix}",
                "bak-config.txt",
                id="no_timestamp_custom_prefix",
            ),
        ],
    )
    def test_name_format_variants(
            self,
            src_file: Path,
            tmp_path: Path,
            monkeypatch: pytest.MonkeyPatch,
            name_format: str,
            expected_name: str,
    ):
        """Honor various valid name_format patterns."""
        dest_dir = tmp_path / "var"
        dest_dir.mkdir()
        fixed = dt.datetime(2024, 10, 11, 14, 30, 22, tzinfo=dt.timezone.utc)
        _freeze_utc_now(monkeypatch, fixed)

        backup_path = backup_file(
            path=str(src_file),
            dest_dir=str(dest_dir),
            name_format=name_format,
            time_format="%Y%m%d-%H%M%S",
            exist_ok=True,
        )
        assert backup_path.name == expected_name

    def test_time_format_custom(self, src_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Apply custom time_format in filename."""
        dest_dir = tmp_path / "custom"
        dest_dir.mkdir()
        fixed = dt.datetime(2024, 10, 11, 0, 0, 0, tzinfo=dt.timezone.utc)
        _freeze_utc_now(monkeypatch, fixed)

        backup_path = backup_file(
            path=str(src_file),
            dest_dir=str(dest_dir),
            name_format="{stem}.{timestamp}{suffix}",
            time_format="%Y-%m-%d",
            exist_ok=True,
        )
        assert backup_path.name == "config.2024-10-11.txt"


class TestCleanDir:
    def test_remove_nested_content(self, populated_dir: Path):
        """Remove all contents recursively and leave root directory empty."""
        clean_dir(populated_dir, missing_ok=False, ignore_errors=False)
        assert populated_dir.exists() and populated_dir.is_dir()
        assert list(populated_dir.iterdir()) == []

    def test_preserve_directory(self, populated_dir: Path):
        """Preserve the directory itself after cleaning."""
        before_stat = populated_dir.stat()
        clean_dir(populated_dir, missing_ok=False, ignore_errors=False)
        after_stat = populated_dir.stat()
        assert populated_dir.exists() and populated_dir.is_dir()
        assert list(populated_dir.iterdir()) == []
        # Inode may be available on POSIX; if present, it should be the same directory.
        assert before_stat.st_ino == after_stat.st_ino if hasattr(before_stat, "st_ino") else True

    def test_missing_dir_missing_ok_false_raises(self, tmp_path: Path):
        """Raise when directory is missing and missing_ok is false."""
        missing = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError, match=r"(?i).*(doesn't exist|not found|no such file).*"):
            clean_dir(missing, missing_ok=False, ignore_errors=False)

    def test_missing_dir_missing_ok_true_succeeds(self, tmp_path: Path):
        """Succeed silently when directory is missing and missing_ok is true."""
        missing = tmp_path / "gone"
        clean_dir(missing, missing_ok=True, ignore_errors=False)
        assert not missing.exists()

    def test_path_not_directory_raises(self, tmp_path: Path):
        """Raise when path exists but is not a directory."""
        not_dir = tmp_path / "file.txt"
        not_dir.write_text("content")
        with pytest.raises(NotADirectoryError, match=r"(?i).*not a directory.*"):
            clean_dir(not_dir, missing_ok=False, ignore_errors=False)

    def test_ignore_errors_true_continues_on_file_unlink_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Continue cleaning when a file deletion fails and ignore_errors is true."""
        root = tmp_path / "r"
        root.mkdir()
        keep = root / "keep.txt"
        keep.write_text("x")
        gone = root / "gone.txt"
        gone.write_text("y")

        original_unlink = Path.unlink

        def failing_unlink(self: Path, missing_ok: bool = False) -> None:  # type: ignore[override]
            if self == keep:
                raise OSError("simulated unlink failure")
            return original_unlink(self)

        monkeypatch.setattr(Path, "unlink", failing_unlink, raising=True)

        clean_dir(root, missing_ok=False, ignore_errors=True)

        assert not gone.exists()
        assert keep.exists()
        assert [p.name for p in root.iterdir()] == ["keep.txt"]

    def test_ignore_errors_false_propagates_file_unlink_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Propagate deletion error when ignore_errors is false."""
        root = tmp_path / "r2"
        root.mkdir()
        failme = root / "fail.txt"
        ok = root / "ok.txt"
        failme.write_text("1")
        ok.write_text("2")

        original_unlink = Path.unlink

        def failing_unlink(self: Path, missing_ok: bool = False) -> None:  # type: ignore[override]
            if self == failme:
                raise OSError("simulated unlink failure")
            return original_unlink(self)

        monkeypatch.setattr(Path, "unlink", failing_unlink, raising=True)

        with pytest.raises(OSError, match=r"(?i).*simulated unlink failure.*"):
            clean_dir(root, missing_ok=False, ignore_errors=False)

        assert failme.exists()
        assert any(p.exists() for p in [failme, ok])

    def test_ignore_errors_true_continues_on_subdir_rmdir_failure(self, tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch):
        """Continue when directory removal fails and ignore_errors is true."""
        root = tmp_path / "r3"
        root.mkdir()
        sub_ok = root / "ok"
        sub_ok.mkdir()
        (sub_ok / "a").write_text("a")
        sub_fail = root / "fail"
        sub_fail.mkdir()
        (sub_fail / "b").write_text("b")

        # Patch the call sites inside c108.os so that removal of sub_fail fails regardless
        # of whether clean_dir uses shutil.rmtree or os.rmdir.
        import c108.os as os_mod  # type: ignore

        original_rmtree = os_mod.shutil.rmtree
        original_rmdir = os_mod.os.rmdir

        def failing_rmtree(path, *args, **kwargs):  # type: ignore[no-redef]
            if Path(path) == sub_fail:
                raise OSError("simulated rmtree failure")
            return original_rmtree(path, *args, **kwargs)

        def failing_rmdir(path, *args, **kwargs):  # type: ignore[no-redef]
            if Path(path) == sub_fail:
                raise OSError("simulated rmdir failure")
            return original_rmdir(path, *args, **kwargs)

        monkeypatch.setattr("c108.os.shutil.rmtree", failing_rmtree, raising=True)
        monkeypatch.setattr("c108.os.os.rmdir", failing_rmdir, raising=True)

        clean_dir(root, missing_ok=False, ignore_errors=True)

        # The failing subdir should remain (possibly empty), ok subdir should be gone
        assert (root / "ok").exists() is False
        assert (root / "fail").exists() is True

    def test_symlink_to_file_removed_only_link(self, tmp_path: Path):
        """Remove symlink entry while preserving the target file."""
        root = tmp_path / "r4"
        root.mkdir()
        target_dir = tmp_path / "outside"
        target_dir.mkdir()
        target = target_dir / "t.txt"
        target.write_text("T")
        link = root / "link.txt"

        try:
            link.symlink_to(target)
        except OSError as e:
            pytest.skip(f"Symlink not permitted on this platform: {e}")

        assert link.is_symlink()
        clean_dir(root, missing_ok=False, ignore_errors=False)

        assert not link.exists()
        assert target.exists() and target.read_text() == "T"
        assert list(root.iterdir()) == []

    def test_symlink_to_dir_removed_only_link(self, tmp_path: Path):
        """Remove symlink to directory while preserving the target directory."""
        root = tmp_path / "r5"
        root.mkdir()
        target = tmp_path / "actual_dir"
        target.mkdir()
        (target / "f.txt").write_text("F")
        link = root / "dirlink"

        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError as e:
            pytest.skip(f"Symlink not permitted on this platform: {e}")

        assert link.is_symlink()
        clean_dir(root, missing_ok=False, ignore_errors=False)

        assert not link.exists()
        assert target.exists() and (target / "f.txt").exists()
        assert list(root.iterdir()) == []

    def test_idempotent_multiple_calls(self, populated_dir: Path):
        """Allow repeated cleaning without error."""
        clean_dir(populated_dir, missing_ok=False, ignore_errors=False)
        clean_dir(populated_dir, missing_ok=False, ignore_errors=False)
        assert list(populated_dir.iterdir()) == []

    def test_return_none(self, populated_dir: Path):
        """Return None explicitly."""
        result = clean_dir(populated_dir, missing_ok=False, ignore_errors=False)
        assert result is None


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
    def test_n_zero(self, tmp_path: Path, encoding: str | None, expected: str | bytes) -> None:
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
