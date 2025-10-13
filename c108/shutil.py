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


# yourpackage/shutil.py
"""
File operation utilities mirroring stdlib shutil with progress support.
"""
import os
import shutil
from pathlib import Path
from typing import Callable, Union
from .io import StreamingFile, DEFAULT_CHUNK_SIZE


def copy_file(
        src: Union[str, bytes, os.PathLike[str], int],
        dst: Union[str, bytes, os.PathLike[str], int],
        *,
        callback: Union[Callable[[int, int], None], None] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        follow_symlinks: bool = True,
        preserve_metadata: bool = True,
        overwrite: bool = True
) -> Path:
    """
    Copy file with optional progress reporting.

    Copies a file from src to dst, optionally tracking progress via callback.
    Similar to shutil.copy2() but with progress tracking support for large files.

    Args:
        src: Source file path (string, bytes, PathLike object, or file descriptor).
        dst: Destination path (string, bytes, PathLike object, or file descriptor).
            Can be a file path or directory. If directory, the file is copied
            into it using the source filename.
        callback: Optional progress callback function.
            Signature: callback(bytes_written: int, total_bytes: int) -> None
            Called after each chunk is written to destination. Not called on empty files.
        chunk_size: Size in bytes for each copy chunk. Defaults to 8MB.
            Larger chunks mean faster copies but less frequent progress updates.
            Set to 0 to use file_size (single chunk, minimal progress updates).
        follow_symlinks: If True, copies the file content that symlink points to.
            If False, creates a new symlink at dst pointing to the same target.
        preserve_metadata: If True, preserves file metadata (timestamps, permissions).
            Similar to shutil.copy2(). If False, only copies content like shutil.copy().
        overwrite: If False, raises FileExistsError if destination file exists.
            If True, overwrites existing files.

    Returns:
        Path: Absolute path to the destination file.

    Raises:
        ValueError: If src and dst are the same file, or if chunk_size is negative.
        FileExistsError: If destination exists and overwrite=False.
        IsADirectoryError: If src is a directory (only files supported).

    Notes:
        - For files under ~1MB, progress callback overhead may exceed copy time.
          Consider callback=None for small files.
        - The function creates parent directories of dst if they don't exist.
        - When dst is a directory, behavior matches shutil.copy: the file is
          copied into the directory with its original basename.
        - Symlink handling matches shutil.copy2 behavior by default.
        - Empty files (0 bytes) are copied without calling the callback.
        - Progress tracking reports bytes written to destination, which accurately
          reflects copy progress.
        - All other exceptions (FileNotFoundError, PermissionError, OSError, etc.)
          are propagated from underlying operations (Path.stat(), open(), StreamingFile, etc.).

    Examples:
        Basic copy with progress:

        >>> def progress(current, total):
        ...     print(f"Copying: {current}/{total} bytes ({current/total*100:.1f}%)")
        ...
        >>> copy_file("large_video.mp4", "backup/", callback=progress)
        Path('/absolute/path/to/backup/large_video.mp4')

        Copy to specific filename without progress:

        >>> copy_file("data.csv", "archive/data_backup.csv")
        Path('/absolute/path/to/archive/data_backup.csv')

        Prevent overwriting existing files:

        >>> copy_file("config.json", "prod/config.json", overwrite=False)
        # Raises FileExistsError if prod/config.json exists

        Copy with custom chunk size (faster, less frequent updates):

        >>> copy_file("huge.bin", "backup/", chunk_size=64*1024*1024)  # 64MB chunks

        Copy without preserving metadata:

        >>> copy_file("file.txt", "copy.txt", preserve_metadata=False)

        Handle symlinks explicitly:

        >>> # Copy symlink as symlink (don't follow)
        >>> copy_file("link.txt", "copy_link.txt", follow_symlinks=False)
    """
    # Convert to Path objects for consistent handling
    src = Path(src)
    dst = Path(dst)

    # Validation - raises ValueError (our exception)
    if chunk_size < 0:
        raise ValueError(f"chunk_size must be non-negative, got {chunk_size}")

    # Resolve source (respecting follow_symlinks)
    if follow_symlinks:
        src_resolved = src.resolve()
    else:
        src_resolved = src

    # Check if source is a directory - raises IsADirectoryError (our exception)
    # Do this before checking existence to provide clearer error message
    if src_resolved.is_dir():
        raise IsADirectoryError(f"Source is a directory, not a file: {src}")

    # Path.exists(), Path.stat() may raise FileNotFoundError, PermissionError (propagated)

    # Handle symlinks when follow_symlinks=False
    if not follow_symlinks and src.is_symlink():
        link_target = os.readlink(src)
        if dst.is_dir():
            dst = dst / src.name
        if dst.exists():
            if not overwrite:
                raise FileExistsError(f"Destination already exists: {dst}")
            dst.unlink()
        os.symlink(link_target, dst)
        return dst.resolve()

    # Determine actual destination path
    if dst.is_dir():
        dst = dst / src.name

    # Check if source and destination are the same - raises ValueError (our exception)
    try:
        if src_resolved.samefile(dst):
            raise ValueError(f"Source and destination are the same file: {src}")
    except FileNotFoundError:
        # dst doesn't exist yet, which is fine
        pass

    # Check overwrite setting - raises FileExistsError (our exception)
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {dst}")

    # Create destination parent directory if needed
    # Path.mkdir() may raise PermissionError, OSError (propagated)
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Get source file size for progress tracking
    # Path.stat() may raise PermissionError, OSError (propagated)
    file_size = src_resolved.stat().st_size

    # Handle empty files quickly without overhead
    if file_size == 0:
        dst.touch()
        if preserve_metadata:
            shutil.copystat(src_resolved, dst)
        return dst.resolve()

    # Perform the copy with progress tracking on destination write
    # StreamingFile may raise ValueError, PermissionError, OSError (propagated)
    with open(src_resolved, 'rb') as src_f:
        with StreamingFile(
                dst,
                'wb',
                callback=callback,
                chunk_size=chunk_size,
                expected_size=file_size
        ) as dst_f:
            while True:
                # Read chunks and let StreamingFile handle progress on writes
                read_size = chunk_size if chunk_size > 0 else file_size
                chunk = src_f.read(read_size)

                if not chunk:
                    break

                # StreamingFile.write() tracks progress automatically
                dst_f.write(chunk)

    # Preserve metadata if requested
    # shutil.copystat() may raise PermissionError, OSError (propagated)
    if preserve_metadata:
        shutil.copystat(src_resolved, dst)

    return dst.resolve()