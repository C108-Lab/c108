"""
High-level, robust utilities for common file and directory operations.
"""

# Standard library -----------------------------------------------------------------------------------------------------
import json
import os
import shutil
import tempfile

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from string import Formatter
from typing import IO, Literal, Iterator


# Methods --------------------------------------------------------------------------------------------------------------

@contextmanager
def atomic_open(
        path: str | os.PathLike[str],
        mode: Literal["w", "wt", "wb"] = "w",
        *,
        encoding: str = "utf-8",
        newline: str | None = None,
        temp_dir: str | os.PathLike[str] | None = None,
        overwrite: bool = True,
        fsync: bool = True
) -> Iterator[IO[str] | IO[bytes]]:
    """
    Open a file for atomic writing via temporary file and rename.

    Creates a temporary file, yields it for writing, then atomically renames it to the target path on success.
    This prevents partial writes if the operation is interrupted. The temporary file is created in the same
    directory as the target (or a specified temp directory) to ensure the rename operation is atomic.

    Args:
        path: Target file path. Accepts string or any PathLike object.
        mode: File mode for writing. Must be one of:
            - 'w' or 'wt': Text mode
            - 'wb': Binary mode
        encoding: Text encoding to use. Only applies in text mode.
        newline: Newline handling for text mode. Only applies in text mode:
            - None: Universal newlines mode. "\n" on input becomes platform-native line ending on output
            - "": No newline translation
            - "\n", "\r", or "\r\n": Force specific line endings

        temp_dir: Directory for temporary file. If None, uses the same directory as the target file. Must be on
            the same filesystem as the target for atomic rename to work. Created automatically if it doesn't exist.
        overwrite: If False, raises FileExistsError if target file already exists. If True,
            allows overwriting existing files.
        fsync: If True, calls fsync() before renaming to ensure data is written to disk. Provides
            durability guarantees against system crashes but may impact performance.

    Yields:
        IO[str] | IO[bytes]: File handle for writing. Type depends on mode:
            - Text modes ("w", "wt"): IO[str]
            - Binary mode ("wb"): IO[bytes]

    Raises:
        ValueError: If mode is not one of "w", "wt", or "wb".
        FileExistsError: If target file exists and overwrite=False.
        OSError: If temp_dir is on a different filesystem than target, or if other I/O errors occur
            during file operations.

    Note:
        The function preserves the original file's permissions if the target file already exists. For new files,
        system default permissions apply.

        The rename operation (os.replace) is atomic on both Unix and Windows, ensuring that the target file
        is never partially written or corrupted, even if the process is interrupted.

    Examples:
        Basic usage with string path:

        >>> with atomic_open("config.json") as f:
        ...     json.dump({"key": "value"}, f)

        Using Path object:

        >>> from pathlib import Path
        >>> with atomic_open(Path("config.json")) as f:
        ...     json.dump({"key": "value"}, f)

        Binary mode for writing bytes:

        >>> with atomic_open("data.bin", mode="wb") as f:
        ...     f.write(b"\\x00\\x01\\x02")

        Force Unix line endings in text mode:

        >>> with atomic_open("unix.txt", newline="\\n") as f:
        ...     f.write("Line 1\\nLine 2\\n")

        Prevent overwriting existing files:

        >>> with atomic_open("important.txt", overwrite=False) as f:
        ...     f.write("New content")  # Raises FileExistsError if file exists

        Use custom temporary directory:

        >>> with atomic_open("data.txt", temp_dir="/tmp") as f:
        ...     f.write("Content")
    """
    # Validate mode
    if mode not in ("w", "wt", "wb"):
        raise ValueError(f"Invalid mode: {mode!r}. Must be 'w', 'wt', or 'wb'")

    # Normalize path to string for os operations
    target_path = Path(os.fspath(path))

    # Check if target exists when overwrite=False
    if not overwrite and target_path.exists():
        raise FileExistsError(f"File exists: {target_path}")

    # Determine temp directory (same as target by default for atomic rename)
    if temp_dir is None:
        tmp_dir = target_path.parent
    else:
        tmp_dir = Path(os.fspath(temp_dir))

    # Ensure temp directory exists
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Determine if binary mode
    is_binary = mode == "wb"

    # Get target file permissions if it exists (to preserve them)
    try:
        original_stat = target_path.stat()
        original_mode = original_stat.st_mode
    except FileNotFoundError:
        original_mode = None

    # Create temporary file with appropriate prefix
    # Use delete=False so we control cleanup and can rename
    fd, temp_path_str = tempfile.mkstemp(
        prefix=f".{target_path.name}.",
        suffix=".tmp",
        dir=tmp_dir,
        text=not is_binary
    )

    temp_path = Path(temp_path_str)
    temp_file = None
    success = False

    try:
        # Close the low-level file descriptor and open with proper mode
        os.close(fd)

        # Open temp file with requested parameters
        if is_binary:
            temp_file = open(temp_path, mode="wb")
        else:
            temp_file = open(
                temp_path,
                mode="w",
                encoding=encoding,
                newline=newline
            )

        # Preserve original file permissions if they exist
        if original_mode is not None:
            os.chmod(temp_path, original_mode)

        # Yield file handle to user
        yield temp_file

        # If we get here, writing succeeded
        temp_file.close()
        temp_file = None

        # Optionally fsync for durability
        if fsync:
            # Reopen to fsync, then close
            with open(temp_path, "rb" if is_binary else "r") as f:
                os.fsync(f.fileno())

        # Atomic rename (os.replace is atomic on both Unix and Windows)
        os.replace(temp_path, target_path)
        success = True

    finally:
        # Clean up temp file if something went wrong
        if temp_file is not None:
            try:
                temp_file.close()
            except Exception:
                pass

        # Remove temp file if rename didn't happen
        if not success and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


def backup_file(
        path: str | os.PathLike[str],
        dest_dir: str | os.PathLike[str] | None = None,
        name_format: str = "{stem}.{timestamp}{suffix}",
        time_format: str = "%Y%m%d-%H%M%S",
        exist_ok: bool = False,
) -> Path:
    """
    Creates a timestamped backup copy of a file.

    Timestamps use UTC to ensure unambiguous, sortable filenames across
    timezones and DST transitions.

    Args:
        path: Path to the file to be backed up.
        dest_dir: Directory where backup will be created. If None, uses the source
            file's directory. Directory must exist.
        name_format: Format string for backup filename. Available placeholders:
            - {stem}: Filename without extension (e.g., "config")
            - {suffix}: File extension including dot (e.g., ".txt")
            - {name}: Full filename (e.g., "config.txt")
            - {timestamp}: Formatted UTC timestamp using time_format
        time_format: strftime format string for UTC timestamp (e.g., "20241011-143022").
        exist_ok: If False, raises FileExistsError when backup file already exists.
            If True, overwrites existing backup.

    Returns:
        Path: Absolute path to the created backup file.

    Raises:
        FileNotFoundError: If source file does not exist.
        NotADirectoryError: If dest_dir is specified but does not exist or is not
            a directory.
        IsADirectoryError: If path points to a directory (only files are supported).
        FileExistsError: If backup file already exists and exist_ok=False.
        ValueError: If name_format contains invalid placeholders.
        PermissionError: If lacking read permission on source file or write
            permission on destination directory.
        OSError: If backup operation fails due to disk space, I/O errors, or other
            OS-level issues.

    Examples:
        >>> backup_file("config.txt")
        Path('/path/to/config.20241011-143022.txt')

        >>> backup_file("data.json", dest_dir="/backups", name_format="{timestamp}_{name}")
        Path('/backups/20241011-143022_data.json')

        >>> backup_file("log.txt", time_format="%Y-%m-%d")
        Path('/path/to/log.2024-10-11.txt')
    """
    source = Path(path).resolve()

    # Validate source file exists and is a file
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")
    if not source.is_file():
        raise IsADirectoryError(f"Path is a directory, not a file: {source}")

    # Determine destination directory
    if dest_dir is None:
        backup_dir = source.parent
    else:
        backup_dir = Path(dest_dir).resolve()
        if not backup_dir.exists():
            raise NotADirectoryError(f"Destination directory not found: {backup_dir}")
        if not backup_dir.is_dir():
            raise NotADirectoryError(
                f"Destination path is not a directory: {backup_dir}"
            )

    # Validate name_format placeholders
    valid_placeholders = {"stem", "suffix", "name", "timestamp"}
    format_placeholders = {
        field_name
        for _, field_name, _, _ in Formatter().parse(name_format)
        if field_name is not None
    }
    invalid_placeholders = format_placeholders - valid_placeholders
    if invalid_placeholders:
        raise ValueError(
            f"Invalid placeholder(s) in name_format: {invalid_placeholders}. "
            f"Valid placeholders: {valid_placeholders}"
        )

    # Build backup filename
    timestamp = datetime.now(timezone.utc).strftime(time_format)
    backup_name = name_format.format(
        stem=source.stem,
        suffix=source.suffix,
        name=source.name,
        timestamp=timestamp,
    )
    backup_path = backup_dir / backup_name

    # Check if backup already exists
    if backup_path.exists() and not exist_ok:
        raise FileExistsError(f"Backup file already exists: {backup_path}")

    # Perform backup using shutil.copy2 (preserves metadata)
    # This can raise: PermissionError, OSError (disk full, I/O error, etc.)
    shutil.copy2(source, backup_path)

    return backup_path


def clean_dir(
        path: str | os.PathLike[str], *,
        missing_ok: bool = False,
        ignore_errors: bool = False,
) -> None:
    """
    Removes all contents from a directory, leaving the directory empty.

    Recursively deletes all files, subdirectories, and symlinks within
    the directory, but preserves the directory itself (including its
    permissions and metadata).

    Args:
        path: Directory to empty.
        missing_ok: If False, raises FileNotFoundError if directory doesn't exist.
            If True, silently succeeds if directory is missing.
        ignore_errors: If False, raises exceptions on deletion failures.
            If True, silently continues when individual items can't be deleted.

    Raises:
        FileNotFoundError: If path doesn't exist (when missing_ok=False).
        NotADirectoryError: If path exists but is not a directory.
        PermissionError: If lacking permission to delete contents (when ignore_errors=False).
        OSError: If deletion fails for other reasons (when ignore_errors=False).

    Examples:
        >>> clean_dir("/tmp/cache")
        >>> clean_dir("/tmp/cache", missing_ok=True)  # Safe if dir doesn't exist
        >>> clean_dir("/tmp/cache", ignore_errors=True)  # Continue on errors
    """
    dir_path = Path(path)

    # Handle missing directory
    if not dir_path.exists():
        if missing_ok:
            return
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    # Validate it's a directory
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {dir_path}")

    # Remove all contents
    for item in dir_path.iterdir():
        try:
            if item.is_dir() and not item.is_symlink():
                # Directory (not a symlink to a directory)
                shutil.rmtree(item)
            else:
                # File or symlink (including symlinks to directories)
                item.unlink()
        except Exception:
            if not ignore_errors:
                raise


def tail_file(
        path: str | os.PathLike[str],
        n: int = 10,
        *,
        encoding: str | None = "utf-8",
        errors: str = "strict",
) -> str | bytes:
    """
    Return the last n lines from a text or binary file.

    Efficiently reads large files by seeking from the end rather than
    reading the entire file into memory.

    Args:
        path: Path to the file.
        n: Number of lines to return from end of file. If n is 0, returns
            empty string/bytes. If n exceeds total lines, returns all lines.
        encoding: Text encoding (e.g., 'utf-8', 'latin-1'). If None, returns bytes.
        errors: How to handle encoding errors. Common values:
            - 'strict': Raise UnicodeDecodeError (default)
            - 'ignore': Skip invalid bytes silently
            - 'replace': Replace invalid bytes with � (U+FFFD)
            - 'backslashreplace': Replace with \\xNN escape sequences

    Returns:
        String containing the last n lines (if encoding specified) or bytes
        (if encoding is None). Returns empty string/bytes if n=0.

    Raises:
        FileNotFoundError: If file doesn't exist.
        IsADirectoryError: If path is a directory.
        PermissionError: If lacking read permission.
        ValueError: If n is negative.
        UnicodeDecodeError: If encoding fails and errors='strict'.

    Examples:
        >>> tail_file("server.log", n=5)
        'line1\\nline2\\nline3\\nline4\\nline5\\n'

        >>> tail_file("server.log", n=0)
        ''

        >>> tail_file("data.bin", encoding=None, n=3)
        b'line1\\nline2\\nline3\\n'

        >>> tail_file("corrupt.txt", n=1, errors='replace')
        'Hello�World\\n'  # � indicates decoding error
    """
    file_path = Path(path)

    # Validate inputs
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

    if n == 0:
        return b"" if encoding is None else ""

    # Validate file exists and is a file
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise IsADirectoryError(f"Path is a directory, not a file: {file_path}")

    # Open file in binary mode for efficient seeking
    with open(file_path, "rb") as f:
        # Get file size
        f.seek(0, os.SEEK_END)
        file_size = f.tell()

        # Empty file
        if file_size == 0:
            return b"" if encoding is None else ""

        # For small files, just read everything
        if file_size < 8192:  # 8KB threshold
            f.seek(0)
            content = f.read()
            lines = content.splitlines(keepends=True)
            result_bytes = b"".join(lines[-n:])
        else:
            # For large files, read backwards in chunks
            result_bytes = _tail_large_file(f, n, file_size)

    # Handle encoding
    if encoding is None:
        return result_bytes
    else:
        return result_bytes.decode(encoding=encoding, errors=errors)


def _tail_large_file(f, n: int, file_size: int) -> bytes:
    """
    Efficiently read last n lines from a large file by seeking backwards.

    Args:
        f: Open file object in binary mode, positioned at end.
        n: Number of lines to read.
        file_size: Total size of file in bytes.

    Returns:
        bytes: Last n lines as bytes.
    """
    chunk_size = 8192  # 8KB chunks
    lines_found = 0
    buffer = b""
    position = file_size

    # Read backwards in chunks until we find n lines
    while position > 0 and lines_found < n:
        # Determine how much to read
        read_size = min(chunk_size, position)
        position -= read_size

        # Seek and read chunk
        f.seek(position)
        chunk = f.read(read_size)

        # Prepend to buffer
        buffer = chunk + buffer

        # Count newlines in buffer
        lines_found = buffer.count(b"\n")

        # If we found enough lines, we can stop
        if lines_found >= n:
            break

    # Split into lines and take last n
    lines = buffer.splitlines(keepends=True)

    # Handle edge case: if file doesn't end with newline, last line won't have one
    # but splitlines still treats it as a line, which is correct
    return b"".join(lines[-n:])
