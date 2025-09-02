#
# Pytest Fixtures
#

# Third-party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.scratch import allocate_file


# Fixtures -------------------------------------------------------------------------------------------------------------

@pytest.fixture
def temp_file(tmp_path):
    """
    A pytest fixture that wraps the main allocate_file function to work with pytest's tmp_path.

    This provides a consistent API with the main allocate_file function while leveraging
    pytest's automatic temporary directory management.
    """

    def _create_temp_file(name: str = "test_file", size: int = 0, unit: str = "B"):
        return allocate_file(path=tmp_path, name=name, size=size, unit=unit)

    return _create_temp_file
