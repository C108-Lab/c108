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

def allocate_file(
        path: str | os.PathLike[str] | None = None,
        name: str = "temp_file",
        size: int | float = 0,
        unit: str = "B",
        add_suffix: bool = True,
        sparce: bool = True
) -> Path:
    """
    Creates a temporary file of a specified size.

    Args:
        path: Directory path or path-like where to create the file. If None, uses system temp directory.
              If 'path' looks like a filename under an existing directory, uses its parent as dir and
              the last component as the base filename.
        name: Base name for the file (without extension)
        size: Size of the file (non-negative)
        unit: Unit for the size. Supported: "B", "kB", "MB", "GB" (case-insensitive)
        sparce: Whether to create a sparse file if the platform supports it; if False, zero fills file in chunks.

    Returns:
        Path: Absolute path to the created file
    """
    # Determine directory and base filename input
    tmp_dir = Path(tempfile.gettempdir())
    p = Path(path) if path is not None else tmp_dir

    # Detection of dir vs file-like path
    if p.exists() and p.is_dir():
        # Looks like an existing directory
        dir_path = p
        base_name = name
    elif p.parent.exists() and p.parent.is_dir():
        # Parent exists -> treat last component as filename
        dir_path = p.parent
        base_name = p.stem or p.name
    elif p.is_absolute() and len(p.parts) > 1:
        # Absolute path with at least one parent -> treat last component as filename
        dir_path = p.parent
        base_name = p.stem or p.name
    else:
        dir_path = p
        base_name = name

    # Ensure directory exists (final guard)
    if not dir_path.exists():
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError):
            dir_path = tmp_dir

    # Validate and convert size
    if size < 0:
        raise ValueError("size must be non-negative")

    unit_upper = unit.upper()
    # Use integer byte count; allow float inputs by rounding down
    size_bytes = int(size)
    size_desc: str

    if unit_upper == "KB":
        size_bytes = int(size * 1024)
        size_desc = f"{size}kB"
    elif unit_upper == "MB":
        size_bytes = int(size * 1024 * 1024)
        size_desc = f"{size}MB"
    elif unit_upper == "GB":
        size_bytes = int(size * 1024 * 1024 * 1024)
        size_desc = f"{size}GB"
    elif unit_upper == "B":
        size_bytes = int(size)
        size_desc = f"{int(size)}B"
    else:
        raise ValueError(f"Unsupported unit: {unit}. Use 'B', 'kB', 'MB', or 'GB'")

    # Sanitize base_name to avoid path separators leaking in
    base_name = base_name.replace(os.sep, "_").replace("/", "_").strip() or "allocate_file"

    # Construct final filename with optional size suffix and .tmp extension
    final_name = base_name + (f"_{size_desc}.tmp" if add_suffix else "")
    file_path = (dir_path / final_name)

    # Write the file with specified size
    with open(file_path, "wb") as f:
        if sparce:
            # Efficient allocation; should create sparse files on modern POSIX filesystems
            f.truncate(size_bytes)
        else:
            # Non-sparse allocation by writing zero bytes
            _allocate_non_sparse(f, size_bytes)

    return file_path.resolve()


@contextmanager
def allocated_file(
        path: str | os.PathLike[str] | None = None,
        name: str = "temp_file",
        size: int | float = 0,
        unit: str = "B",
        add_suffix: bool = True,
        sparce: bool = True
) -> Iterator[Path]:
    """
    Context manager that creates a size-allocated temporary file and deletes it on exit.
    Yields the pathlib.Path to the created file.
    """
    p = allocate_file(path=path, name=name,
                      size=size, unit=unit, add_suffix=add_suffix,
                      sparce=sparce)
    try:
        yield p
    finally:
        try:
            p.unlink()
        except FileNotFoundError:
            pass
        except PermissionError:
            # On Windows or when the file is still open, the caller must close handles first.
            pass


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


def _allocate_non_sparse(f, size_bytes: int):
    """
    Write data to an open file f, make it a non-sparse file filled with zero bytes of length `size_bytes`.
    Writes in `chunk_size` blocks to avoid large memory use and fsyncs at the end.
    """
    size = int(size_bytes)
    if size == 0:
        f.write(b"")
        return

    chunk_size = 128 * 1024 * 1024  # 128 MB
    chunk = b"\0" * min(chunk_size, size)
    written = 0
    while written < size:
        to_write = min(len(chunk), size - written)
        if to_write == len(chunk):
            f.write(chunk)
        else:
            f.write(chunk[:to_write])
        written += to_write
