#
# C108 - Scratch Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import os
from pathlib import Path

# Third-party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.scratch import temp_dir


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
