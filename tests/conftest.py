#
# Pytest Fixtures
#

# Third-party ----------------------------------------------------------------------------------------------------------
import pytest
import pathlib
from typing import Callable

# Local ----------------------------------------------------------------------------------------------------------------
from c108.scratch import allocate_file

FILE_SIZE = 10 * 1024

# Fixtures -------------------------------------------------------------------------------------------------------------

@pytest.fixture
def temp_file(tmp_path: pathlib.Path):
    """Fixture to create a temporary file with specified size and content."""

    def _create_file(size: int = 0, content: bytes = b"\x00") -> pathlib.Path:
        file_path = tmp_path / "test.dat"
        data = (content * (size // len(content) + 1))[:size]
        file_path.write_bytes(data)
        return file_path

    return _create_file
