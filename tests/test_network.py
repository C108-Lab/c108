#
# C108 - Network Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import pytest
from pathlib import Path

# Local ----------------------------------------------------------------------------------------------------------------
from c108.network import transfer_timeout


# Tests ----------------------------------------------------------------------------------------------------------------

class TestUtils:
    pass


@pytest.mark.parametrize(
    "size,speed,expected_timeout",
    [(1, 10, 10),
     (10, 10, 10),
     ],
)
def test_transfer_timeout(temp_file, size, speed, expected_timeout):
    file_path = temp_file(size=size)
    timeout = transfer_timeout(file_path, speed=speed, base_timeout=10, safety_factor=10)
    print()
    print(f"File       : {Path(file_path).name}")
    print(f"Speed, Mbit: {speed}")
    print(f"Timeout, s : {timeout}")
    print(f"Expected Timeout, s : {expected_timeout}")
