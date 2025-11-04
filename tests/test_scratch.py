#
# C108 - Scratch Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import os
import re
from pathlib import Path

# Third-party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.scratch import temp_dir


class TestTempDir:
    """Test suite for temp_dir context manager."""

    @pytest.mark.parametrize(
        "name_format,validation",
        [
            pytest.param(
                "tmp-{random}",
                lambda name: name.startswith("tmp-"),
                id="default_format",
            ),
            pytest.param(
                "prefix-{random}-suffix",
                lambda name: name.startswith("prefix-") and name.endswith("-suffix"),
                id="prefix_suffix",
            ),
            pytest.param(
                "worker-{pid}-{random}",
                lambda name: f"-{os.getpid()}-" in name,
                id="pid_placeholder",
            ),
            pytest.param(
                "build-{timestamp}-{random}",
                lambda name: re.match(r"^build-\d{8}-\d{6}-.+$", name) is not None,
                id="timestamp_default",
            ),
            pytest.param(
                "test-{timestamp:%Y%m%d}-{random}",
                lambda name: re.match(r"^test-\d{8}-.+$", name) is not None,
                id="timestamp_custom",
            ),
        ],
    )
    def test_name_format_placeholders(self, name_format, validation):
        """Verify all placeholders are correctly replaced in directory name."""
        with temp_dir(name_format=name_format) as p:
            assert isinstance(p, Path)
            assert validation(p.name)

    @pytest.mark.parametrize(
        "name_format,expected_error_substring",
        [
            pytest.param(
                "no-random-here",
                r"must\s*contain\s*\{random\}",
                id="missing_random",
            ),
            pytest.param(
                "{random}-{unknown}",
                r"Unknown\s+placeholders",
                id="unknown_placeholder",
            ),
            pytest.param(
                "",
                r"name_format\s+cannot\s+be\s+empty",
                id="empty_string",
            ),
        ],
    )
    def test_name_format_validation_errors(self, name_format, expected_error_substring):
        """Verify invalid name_format raises ValueError with descriptive message."""
        with pytest.raises(ValueError, match=rf"(?i).*{expected_error_substring}.*"):
            with temp_dir(name_format=name_format):
                pass

    def test_returns_path_and_cleans_up_after_exit(self):
        """Verify returns Path instance, directory exists in context, and is removed after."""
        captured_path: Path
        with temp_dir(name_format="tmp-{random}") as p:
            captured_path = p
            assert isinstance(p, Path)
            assert p.exists()
            assert p.is_dir()
        assert not captured_path.exists()

    def test_delete_false_preserves_directory(self, tmp_path: Path):
        """Verify directory persists when delete=False."""
        with temp_dir(
            parent=tmp_path.as_posix(), name_format="keep-{random}", delete=False
        ) as p:
            created = p
            assert created.exists()
        assert created.exists()
        for child in created.glob("**/*"):
            if child.is_file():
                child.unlink()
        created.rmdir()

    @pytest.mark.parametrize(
        "parent_type",
        [
            pytest.param("str", id="str_path"),
            pytest.param("Path", id="pathlib_path"),
        ],
    )
    def test_custom_parent_directory(self, parent_type):
        """Verify directory is created under custom parent as str or PathLike."""
        base = Path(os.getcwd()) / "tmp_test_parent"
        base.mkdir(exist_ok=True)
        try:
            parent_arg: str | Path
            if parent_type == "str":
                parent_arg = base.as_posix()
            elif parent_type == "Path":
                parent_arg = base
            else:
                raise AssertionError("Unexpected parent_type")

            with temp_dir(parent=parent_arg, name_format="x-{random}") as p:
                assert p.parent == base
                assert p.exists()
        finally:
            if base.exists():
                for child in sorted(base.glob("**/*"), reverse=True):
                    if child.is_file():
                        try:
                            child.unlink()
                        except OSError:
                            pass
                    elif child.is_dir():
                        try:
                            child.rmdir()
                        except OSError:
                            pass
                try:
                    base.rmdir()
                except OSError:
                    pass

    def test_files_and_subdirs_cleaned_up_recursively(self):
        """Verify nested files and directories are removed during cleanup."""
        with temp_dir(name_format="deep-{random}") as p:
            nested_dir = p / "a" / "b" / "c"
            nested_dir.mkdir(parents=True, exist_ok=True)
            f1 = nested_dir / "file.txt"
            f1.write_text("data", encoding="utf-8")
            assert f1.exists()
            assert nested_dir.exists()
            (p / "root.txt").write_text("root", encoding="utf-8")
            assert (p / "root.txt").exists()
        assert not p.exists()

    def test_concurrent_calls_create_unique_directories(self):
        """Verify multiple concurrent calls generate distinct directory paths."""
        paths: list[Path] = []
        with (
            temp_dir(name_format="con-{random}") as p1,
            temp_dir(name_format="con-{random}") as p2,
        ):
            paths = [p1, p2]
            assert p1 != p2
            assert p1.exists() and p2.exists()
        assert all(not p.exists() for p in paths)

    def test_exception_propagates_and_cleanup_still_occurs(self):
        """Verify cleanup happens even when exception is raised inside context."""
        caught_path: Path | None = None

        class Boom(Exception):
            pass

        with pytest.raises(Boom, match=r"(?i).*boom.*"):
            with temp_dir(name_format="err-{random}") as p:
                caught_path = p
                assert p.exists()
                raise Boom("boom happened")

        assert caught_path is not None
        assert not caught_path.exists()
