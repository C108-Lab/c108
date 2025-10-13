#
# C108 Temp File Tools
#

# Standard library -----------------------------------------------------------------------------------------------------
import os.path
import tempfile
from pathlib import Path
from contextlib import contextmanager
from typing import Iterator


# Methods --------------------------------------------------------------------------------------------------------------


@contextmanager
def temp_dir(suffix: str = None, prefix: str = None, dir: str | os.PathLike[str] = None,
             ignore_cleanup_errors: bool = False, *, delete: bool = True) -> Iterator[Path]:
    """
    Context manager that provides a Path object to a temporary directory.
    The directory and its contents are automatically removed upon exiting the 'with' block.
    """
    with tempfile.TemporaryDirectory(
            suffix=suffix, prefix=prefix, dir=dir,
            ignore_cleanup_errors=ignore_cleanup_errors,
            delete=delete) as tmpdir_str:
        yield Path(tmpdir_str)


# Private methods ------------------------------------------------------------------------------------------------------
