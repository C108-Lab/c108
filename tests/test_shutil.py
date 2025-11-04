#
# C108 - shutil Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import datetime as dt
from pathlib import Path

# Third Party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.shutil import backup_file, clean_dir, find_files


# Tests ----------------------------------------------------------------------------------------------------------------


def _freeze_utc_now(monkeypatch: pytest.MonkeyPatch, fixed: dt.datetime) -> None:
    """Patch c108.shutil.datetime.now(...) to return fixed UTC datetime."""
    if fixed.tzinfo is None:
        fixed = fixed.replace(tzinfo=dt.timezone.utc)

    class FixedDateTime(dt.datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            if tz is None:
                return fixed.replace(tzinfo=None)
            return fixed.astimezone(tz)

    monkeypatch.setattr("c108.shutil.datetime", FixedDateTime, raising=True)


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
    def test_create_in_dest_dir(
        self, src_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Create backup in explicit destination with deterministic timestamp."""
        dest_dir = tmp_path / "backups"
        dest_dir.mkdir(parents=True, exist_ok=True)
        fixed = dt.datetime(2024, 10, 11, 14, 30, 22, tzinfo=dt.timezone.utc)
        _freeze_utc_now(monkeypatch, fixed)

        name_format = "{stem}.{timestamp}{suffix}"
        exist_ok = False

        backup_path = backup_file(
            path=str(src_file),
            dest_dir=str(dest_dir),
            name_format=name_format,
            exist_ok=exist_ok,
        )
        expected = (dest_dir / "config.20241011-143022.txt").resolve()

        assert backup_path == expected

    def test_invalid_placeholder_raises(self, src_file: Path, tmp_path: Path):
        """Reject invalid placeholders in name_format."""
        dest_dir = tmp_path / "d"
        dest_dir.mkdir()
        with pytest.raises(ValueError, match=r"(?i).*invalid placeholder.*"):
            backup_file(
                path=str(src_file),
                dest_dir=str(dest_dir),
                name_format="{stemm}.{timestamp}{suffix}",
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
                name_format="{stem}.{timestamp:%Y%m%d}",
                exist_ok=False,
            )

    def test_dest_dir_not_exist_raises(self, src_file: Path, tmp_path: Path):
        """Raise when destination directory does not exist."""
        dest_dir = tmp_path / "does_not_exist"
        with pytest.raises(
            NotADirectoryError, match=r"(?i).*destination directory not found.*"
        ):
            backup_file(
                path=str(src_file),
                dest_dir=str(dest_dir),
                name_format="{stem}.{timestamp}{suffix}",
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
                exist_ok=False,
            )

    def test_backup_exists_exist_ok_false_raises(
        self, src_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Raise when backup exists and exist_ok is false."""
        dest_dir = tmp_path / "d"
        dest_dir.mkdir()
        fixed = dt.datetime(2024, 1, 2, 3, 4, 5, tzinfo=dt.timezone.utc)
        _freeze_utc_now(monkeypatch, fixed)
        name_format = "{stem}.{timestamp}{suffix}"

        first = backup_file(
            path=str(src_file),
            dest_dir=str(dest_dir),
            name_format=name_format,
            exist_ok=True,
        )
        assert first.exists()

        with pytest.raises(
            FileExistsError, match=r"(?i).*backup file already exists.*"
        ):
            backup_file(
                path=str(src_file),
                dest_dir=str(dest_dir),
                name_format=name_format,
                exist_ok=False,
            )

    def test_backup_exists_exist_ok_true_overwrites(
        self, src_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Overwrite existing backup when exist_ok is true."""
        dest_dir = tmp_path / "d2"
        dest_dir.mkdir()
        fixed = dt.datetime(2024, 6, 7, 8, 9, 10, tzinfo=dt.timezone.utc)
        _freeze_utc_now(monkeypatch, fixed)
        name_format = "{stem}.{timestamp}{suffix}"

        backup_path = backup_file(
            path=str(src_file),
            dest_dir=str(dest_dir),
            name_format=name_format,
            exist_ok=True,
        )
        assert backup_path.exists()
        assert backup_path.read_text() == "alpha"

        src_file.write_text("beta")
        overwritten = backup_file(
            path=str(src_file),
            dest_dir=str(dest_dir),
            name_format=name_format,
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
            exist_ok=True,
        )
        assert backup_path.name == expected_name

    def test_time_format_custom(
        self, src_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Apply custom time_format in filename."""
        dest_dir = tmp_path / "custom"
        dest_dir.mkdir()
        fixed = dt.datetime(2024, 10, 11, 0, 0, 0, tzinfo=dt.timezone.utc)
        _freeze_utc_now(monkeypatch, fixed)

        backup_path = backup_file(
            path=str(src_file),
            dest_dir=str(dest_dir),
            name_format="{stem}.{timestamp:%Y-%m-%d}{suffix}",
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
        assert (
            before_stat.st_ino == after_stat.st_ino
            if hasattr(before_stat, "st_ino")
            else True
        )

    def test_missing_dir_missing_ok_false_raises(self, tmp_path: Path):
        """Raise when directory is missing and missing_ok is false."""
        missing = tmp_path / "does_not_exist"
        with pytest.raises(
            FileNotFoundError, match=r"(?i).*(doesn't exist|not found|no such file).*"
        ):
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

    def test_ignore_errors_true_continues_on_file_unlink_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
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

    def test_ignore_errors_false_propagates_file_unlink_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
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

    def test_ignore_errors_true_continues_on_subdir_rmdir_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Continue when directory removal fails and ignore_errors is true."""
        root = tmp_path / "r3"
        root.mkdir()
        sub_ok = root / "ok"
        sub_ok.mkdir()
        (sub_ok / "a").write_text("a")
        sub_fail = root / "fail"
        sub_fail.mkdir()
        (sub_fail / "b").write_text("b")

        # Patch the call sites inside c108.shutil so that removal of sub_fail fails regardless
        # of whether clean_dir uses shutil.rmtree or os.rmdir.
        import c108.shutil as os_mod  # type: ignore

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

        monkeypatch.setattr("c108.shutil.shutil.rmtree", failing_rmtree, raising=True)
        monkeypatch.setattr("c108.shutil.os.rmdir", failing_rmdir, raising=True)

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


# Python

from pathlib import Path
import os, re
import stat
import pytest

import os
import stat
from pathlib import Path
from typing import Callable, Iterator

import pytest


class TestFindFiles:
    """Test suite for find_files function (12 test methods, ~35 parametrized cases)."""

    @pytest.mark.parametrize(
        "path_type,expected_exc,match_msg",
        [
            pytest.param(
                "missing", FileNotFoundError, "does not exist", id="missing-path"
            ),
            pytest.param(
                "file", NotADirectoryError, "not a directory", id="file-not-dir"
            ),
            pytest.param(
                None,
                TypeError,
                "argument should be a str or an os.PathLike",
                id="none-type",
            ),
            pytest.param(
                123,
                TypeError,
                "argument should be a str or an os.PathLike",
                id="int-type",
            ),
        ],
    )
    def test_path_validation_errors(
        self, tmp_path: Path, path_type, expected_exc, match_msg
    ):
        """Raise appropriate errors for invalid path inputs."""
        if path_type == "missing":
            target = tmp_path / "nope"
        elif path_type == "file":
            target = tmp_path / "f.txt"
            target.write_text("x")
        else:
            target = path_type  # None or 123

        with pytest.raises(expected_exc, match=rf"(?i).*{re.escape(match_msg)}.*"):
            list(find_files(target, "*"))

    @pytest.mark.parametrize(
        "files,pattern,expected",
        [
            pytest.param(
                ["a.py", "b.txt", "test_a.py"],
                "*.py",
                {"a.py", "test_a.py"},
                id="ext-wildcard",
            ),
            pytest.param(
                ["test_1.py", "test_a.py", "prod.py"],
                "test_*",
                {"test_1.py", "test_a.py"},
                id="prefix-wildcard",
            ),
            pytest.param(
                [".env", ".hidden.txt", "visible.txt"],
                "[!.]*.txt",
                {"visible.txt"},
                id="negated-class",
            ),
            pytest.param([], "*", set(), id="empty-dir"),
        ],
    )
    def test_pattern_matching_filename_only(
        self, tmp_path: Path, files, pattern, expected
    ):
        """Pattern matches filename only (not path), using fnmatch syntax."""
        for f in files:
            (tmp_path / f).write_text("x")
        results = {p.name for p in find_files(tmp_path, pattern)}
        assert results == expected

    def test_empty_pattern_raises_error(self, tmp_path: Path):
        """Empty pattern raises ValueError."""
        (tmp_path / "file.txt").write_text("x")
        with pytest.raises(ValueError, match="Pattern cannot be empty"):
            list(find_files(tmp_path, ""))

    @pytest.mark.parametrize(
        "structure,pattern,exclude,expected",
        [
            pytest.param(
                {"a.py", "b.pyc", "sub/__pycache__/x.pyc"},
                "*",
                ["*.pyc"],
                {"a.py"},
                id="exclude-extension-anywhere",
            ),
            pytest.param(
                {"src/main.py", "tests/test_main.py", "docs/api.py"},
                "*.py",
                ["tests"],
                {"src/main.py", "docs/api.py"},
                id="exclude-dir-name-anywhere",
            ),
            pytest.param(
                {".git/config", "src/.hidden", "main.py"},
                "*",
                [".*"],
                {"main.py"},
                id="exclude-hidden-dotfiles",
            ),
            pytest.param(
                {"a.py", "sub/a.py", "sub/tests/a.py"},
                "*.py",
                ["tests"],
                {"a.py", "sub/a.py"},
                id="exclude-component-in-path",
            ),
            pytest.param(
                {"keep.txt", "skip.txt", "sub/keep.txt"},
                "*.txt",
                ["skip.txt"],
                {"keep.txt", "sub/keep.txt"},
                id="exclude-by-filename-not-pattern",
            ),
        ],
    )
    def test_exclude_patterns_match_relpath_components(
        self, tmp_path: Path, structure, pattern, exclude, expected
    ):
        """Exclude patterns match against relative path and any path component."""
        for rel in structure:
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x")
        got = {
            str(p.relative_to(tmp_path))
            for p in find_files(tmp_path, pattern, exclude=exclude)
        }
        assert got == expected

    @pytest.mark.parametrize(
        "structure,max_depth,expected",
        [
            pytest.param(
                {"f0.py", "d1/f1.py", "d1/d2/f2.py", "d1/d2/d3/f3.py"},
                0,
                {"f0.py"},
                id="depth-0-root-only",
            ),
            pytest.param(
                {"f0.py", "d1/f1.py", "d1/d2/f2.py"},
                1,
                {"f0.py", "d1/f1.py"},
                id="depth-1",
            ),
            pytest.param(
                {"f0.py", "d1/f1.py", "d1/d2/f2.py"},
                2,
                {"f0.py", "d1/f1.py", "d1/d2/f2.py"},
                id="depth-2",
            ),
            pytest.param(
                {"f0.py", "d1/f1.py", "d1/d2/f2.py"},
                None,
                {"f0.py", "d1/f1.py", "d1/d2/f2.py"},
                id="depth-none-unlimited",
            ),
            pytest.param(
                {"d1/d2/d3/d4/d5/deep.py"},
                10,
                {"d1/d2/d3/d4/d5/deep.py"},
                id="depth-very-deep",
            ),
        ],
    )
    def test_max_depth_traversal_limits(
        self, tmp_path: Path, structure, max_depth, expected
    ):
        """Respect max_depth parameter (0=root only, None=unlimited)."""
        for rel in structure:
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x")
        got = {
            str(p.relative_to(tmp_path))
            for p in find_files(tmp_path, "*.py", max_depth=max_depth)
        }
        assert got == expected

    @pytest.mark.parametrize(
        "scenario,follow_symlinks,expected_count",
        [
            pytest.param(
                "two-links-one-target", False, 3, id="no-follow-all-entries"
            ),  # Changed: 1 → 3
            pytest.param("two-links-one-target", True, 1, id="follow-dedupe-by-inode"),
            pytest.param("self-loop-dir", True, 1, id="self-loop-detected"),
            pytest.param("two-node-cycle", True, 1, id="cycle-detected"),
            pytest.param("deep-chain", True, 1, id="chain-resolved"),
        ],
    )
    def test_symlink_handling_and_loop_detection(
        self, tmp_path: Path, scenario, follow_symlinks, expected_count
    ):
        """Handle symlinks correctly: deduplicate by inode, detect loops."""
        if scenario == "two-links-one-target":
            target = tmp_path / "file.txt"
            target.write_text("x")
            (tmp_path / "link1").symlink_to(target)
            (tmp_path / "link2").symlink_to(target)
            pattern = "*"
        elif scenario == "self-loop-dir":
            d = tmp_path / "d1"
            d.mkdir()
            (d / "x.txt").write_text("x")
            (d / "loop").symlink_to(d, target_is_directory=True)
            pattern = "*.txt"
        elif scenario == "two-node-cycle":
            d1 = tmp_path / "d1"
            d2 = tmp_path / "d2"
            d1.mkdir()
            d2.mkdir()
            (d1 / "a.txt").write_text("x")
            (d1 / "link").symlink_to(d2, target_is_directory=True)
            (d2 / "link").symlink_to(d1, target_is_directory=True)
            pattern = "*.txt"
        else:
            # deep-chain
            d1 = tmp_path / "d1"
            d2 = tmp_path / "d2"
            d3 = tmp_path / "d3"
            d1.mkdir()
            d2.mkdir()
            d3.mkdir()
            (d1 / "link").symlink_to(d2, target_is_directory=True)
            (d2 / "link").symlink_to(d3, target_is_directory=True)
            (d3 / "t.txt").write_text("x")
            pattern = "*.txt"

        results = list(find_files(tmp_path, pattern, follow_symlinks=follow_symlinks))
        count = sum(1 for _ in results)
        assert count == expected_count

    @pytest.mark.parametrize(
        "include_dirs,pattern,structure,expected_files,expected_dirs",
        [
            pytest.param(
                False,
                "*",  # Changed: "cache*" → "*"
                {"files": {"a.py"}, "dirs": {"cache", "sub/cache_tmp"}},
                {"a.py"},
                set(),
                id="dirs-not-included-default",
            ),
            pytest.param(
                True,
                "cache*",
                {"files": {"a.py"}, "dirs": {"cache", "sub/cache_tmp"}},
                set(),
                {"cache", "sub/cache_tmp"},
                id="dirs-included-when-enabled",
            ),
            pytest.param(
                True,
                "*.py",
                {"files": {"main.py"}, "dirs": {"tests"}},
                {"main.py"},
                set(),
                id="dirs-filtered-by-pattern",
            ),
        ],
    )
    def test_include_dirs_option(
        self,
        tmp_path: Path,
        include_dirs,
        pattern,
        structure,
        expected_files,
        expected_dirs,
    ):
        """Control whether directories matching pattern are yielded."""
        for d in structure["dirs"]:
            (tmp_path / d).mkdir(parents=True, exist_ok=True)
        for f in structure["files"]:
            p = tmp_path / f
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x")

        results = list(find_files(tmp_path, pattern, include_dirs=include_dirs))
        files = {str(p.relative_to(tmp_path)) for p in results if p.is_file()}
        dirs = {str(p.relative_to(tmp_path)) for p in results if p.is_dir()}
        assert files == expected_files
        assert dirs == expected_dirs

    @pytest.mark.parametrize(
        "structure,pattern,exclude,predicate_desc,expected",
        [
            pytest.param(
                {"keep.txt": "x", "empty.txt": ""},
                "*.txt",
                [],
                "size>0",
                {"keep.txt"},
                id="predicate-size-filter",
            ),
            pytest.param(
                {"src/keep.py": "x", "tests/skip.py": "x"},
                "*.py",
                ["tests"],
                "always-true",
                {"src/keep.py"},
                id="predicate-after-exclude",
            ),
            pytest.param(
                {"a.py": "x", "b.py": "y", "c.txt": "z"},
                "*",
                [],
                "endswith-py",
                {"a.py", "b.py"},
                id="predicate-custom-logic",
            ),
        ],
    )
    def test_predicate_filtering(
        self, tmp_path: Path, structure, pattern, exclude, predicate_desc, expected
    ):
        """Apply predicate after pattern/exclude; predicate receives Path objects."""
        for rel, content in structure.items():
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)

        if predicate_desc == "size>0":
            pred = lambda p: p.is_file() and p.stat().st_size > 0
        elif predicate_desc == "always-true":
            pred = lambda p: True
        else:
            pred = lambda p: p.name.endswith(".py")

        got = {
            str(p.relative_to(tmp_path))
            for p in find_files(tmp_path, pattern, exclude=exclude, predicate=pred)
        }
        assert got == expected

    @pytest.mark.skipif(os.getuid() == 0, reason="Cannot test permissions as root")
    @pytest.mark.parametrize(
        "unreadable_type",
        [
            pytest.param("directory", id="unreadable-dir-skipped"),
            pytest.param("file", id="unreadable-file-skipped"),
        ],
    )
    def test_permission_errors_silently_skipped(self, tmp_path: Path, unreadable_type):
        """Skip entries with PermissionError without raising."""
        readable_dir = tmp_path / "readable"
        readable_dir.mkdir()
        (readable_dir / "ok.py").write_text("x")

        if unreadable_type == "directory":
            blocked = tmp_path / "blocked"
            blocked.mkdir()
            (blocked / "secret.py").write_text("x")
            blocked.chmod(0)
        else:
            blocked = tmp_path / "secret.py"
            blocked.write_text("x")
            blocked.chmod(0)

        try:
            results = {
                str(p.relative_to(tmp_path)) for p in find_files(tmp_path, "*.py")
            }
            assert results == {"readable/ok.py"}
        finally:
            # restore perms so tmp cleanup works
            if blocked.exists():
                blocked.chmod(stat.S_IRWXU)

    @pytest.mark.parametrize(
        "structure,pattern,exclude,max_depth,expected",
        [
            pytest.param(
                {"a.py", "b.pyc", "sub/c.py", "sub/__pycache__/d.pyc"},
                "*.py",
                ["*.pyc", "__pycache__"],
                None,
                {"a.py", "sub/c.py"},
                id="common-python-filters",
            ),
            pytest.param(
                {"src/a.py", "tests/test_a.py", "tests/unit/test_b.py"},
                "*.py",
                ["tests"],
                1,
                {"src/a.py"},
                id="depth-with-exclude",
            ),
            pytest.param(
                {".git/a", "src/.env", "src/main.py", "dist/pkg.whl"},
                "*",
                [".*", "dist"],
                None,
                {"src/main.py"},
                id="multi-exclude-patterns",
            ),
        ],
    )
    def test_combined_filters_realistic_scenarios(
        self, tmp_path: Path, structure, pattern, exclude, max_depth, expected
    ):
        """Test realistic combinations of pattern, exclude, depth."""
        for rel in structure:
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            # choose content extension-agnostic
            if p.suffix:
                p.write_text("x")
            else:
                # might be a directory path if last component has no suffix
                # ensure it's a file unless it is meant as a directory container
                p.write_text("x")
        got = {
            str(p.relative_to(tmp_path))
            for p in find_files(tmp_path, pattern, exclude=exclude, max_depth=max_depth)
        }
        assert got == expected

    @pytest.mark.parametrize(
        "files,pattern,exclude_param,expected",
        [
            pytest.param(["a.py"], "*", None, {"a.py"}, id="exclude-none"),
            pytest.param(["a.py"], "*", [], {"a.py"}, id="exclude-empty-list"),
        ],
    )
    def test_exclude_none_vs_empty_list_equivalent(
        self, tmp_path: Path, files, pattern, exclude_param, expected
    ):
        """Exclude None and [] behave identically."""
        for f in files:
            (tmp_path / f).write_text("x")
        got = {p.name for p in find_files(tmp_path, pattern, exclude=exclude_param)}
        assert got == expected

    def test_returns_path_objects_relative_or_absolute(self, tmp_path: Path):
        """Return Path objects; type (relative/absolute) matches input path type."""
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.py").write_text("x")

        # absolute
        abs_results = list(find_files(tmp_path, "*.py"))
        assert all(p.is_absolute() for p in abs_results)

        # relative: call with relative path
        cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            rel_results = list(find_files(Path("."), "*.py"))
            assert all(not str(p).startswith(str(tmp_path)) for p in rel_results)
            # ensure they are Paths
            assert all(isinstance(p, Path) for p in rel_results)
        finally:
            os.chdir(cwd)

    def test_pattern_exclude_interaction_edge_case(self, tmp_path: Path):
        """File matches pattern but is in excluded directory: should not yield."""
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test.py").write_text("x")
        results = list(find_files(tmp_path, "*.py", exclude=["tests"]))
        assert results == []
