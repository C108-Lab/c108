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


@pytest.fixture
def temp_file_known(tmp_path) -> str:
    """Create a temporary file with known content and return its path."""
    p = tmp_path / "data.bin"
    # Fill with deterministic pattern
    block = (b"0123456789ABCDEF" * 64)  # 1024 bytes per block
    repeats = FILE_SIZE // len(block)
    remainder = FILE_SIZE % len(block)
    with open(p, "wb") as f:
        for _ in range(repeats):
            f.write(block)
        if remainder:
            f.write(block[:remainder])
    return str(p)
