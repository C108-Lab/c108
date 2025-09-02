#
# C108 - Scratch Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import os
from pathlib import Path

# Third-party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.scratch import allocate_file, allocated_file, temp_dir


class TestUtils:
    def test_temp_dir(self):
        with temp_dir(prefix="test_") as d:
            assert isinstance(d, Path)
            assert d.exists() and d.is_dir()
            # create something inside to ensure cleanup removes contents too
            p = d / "touch.txt"
            p.write_text("ok")
            assert p.exists()
            saved = d

        assert not saved.exists(), "Directory should have been removed"

    @pytest.mark.parametrize(
        "size,unit,expected",
        [(108, "B", 108),
         (1, "kB", 1024),
         ],
    )
    def test_allocate_file_logical_size(self, size, unit, expected):
        # Use a temp directory to avoid clutter
        with temp_dir(prefix="files_") as d:
            fp = allocate_file(path=d, size=size, unit=unit)
            assert fp.exists() and fp.is_file()
            # Logical size should match what we asked for
            assert os.path.getsize(fp) == expected

    def test_allocated_deletes_on_exit(self):
        with allocated_file(size=128, unit="KB") as fp:
            assert fp.exists() and fp.is_file()
            path_copy = fp
        # file should be removed after context exit
        assert not path_copy.exists()
