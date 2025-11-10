"""
High-level, robust utilities for common file and directory operations.
"""

# Standard library -----------------------------------------------------------------------------------------------------
import os
import shutil

from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from string import Formatter
from typing import Callable, Iterator

# Local ----------------------------------------------------------------------------------------------------------------
from .io import StreamingFile


# Methods --------------------------------------------------------------------------------------------------------------


def backup_file(
    path: str | os.PathLike[str],
    dest_dir: str | os.PathLike[str] | None = None,
    name_format: str = "{stem}.{timestamp}{suffix}",
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
            - {timestamp}: UTC timestamp (e.g., "20250101-143010")
            - {timestamp:fmt}: Formatted UTC timestamp using strftime syntax in fmt
            - {pid}: Process ID
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
        ValueError: If name_format contains invalid placeholders or invalid strftime
            format in timestamp.
        PermissionError: If lacking read permission on source file or write
            permission on destination directory.
        OSError: If backup operation fails due to disk space, I/O errors, or other
            OS-level issues.

    Examples:
        >>> backup_file(file_txt)                   # doctest: +SKIP
        Path('/path/to/config.20250101-143010.txt')

        >>> backup_file("data.json", dest_dir="/backups", name_format="{timestamp}_{name}")  # doctest: +SKIP
        Path('/backups/20250101-143010_data.json')

        >>> backup_file("log.txt", name_format="{stem}.{timestamp:%Y-%m-%d}{suffix}")        # doctest: +SKIP
        Path('/path/to/log.2025-01-01.txt')

        >>> backup_file("app.log", name_format="{timestamp:%Y%m%d}_{pid}_{name}")            # doctest: +SKIP
        Path('/path/to/20250101_12345_app.log')

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
            raise NotADirectoryError(f"Destination path is not a directory: {backup_dir}")

    # Validate name_format placeholders and build format values
    valid_placeholders = {"stem", "suffix", "name", "timestamp", "pid"}
    format_placeholders = {
        field_name.split(":")[0]  # Extract base name before format spec
        for _, field_name, _, _ in Formatter().parse(name_format)
        if field_name is not None
    }
    invalid_placeholders = format_placeholders - valid_placeholders
    if invalid_placeholders:
        raise ValueError(
            f"Invalid placeholder(s) in name_format: {invalid_placeholders}. "
            f"Valid placeholders: {valid_placeholders}"
        )

    # Process timestamp placeholders manually
    now_utc = datetime.now(timezone.utc)
    processed_format = name_format

    # Find all timestamp placeholders with their format specs
    import re

    timestamp_pattern = r"\{timestamp(?::([^}]+))?\}"

    def replace_timestamp(match):
        format_spec = match.group(1)
        if format_spec:
            try:
                return now_utc.strftime(format_spec)
            except ValueError as e:
                raise ValueError(f"Invalid strftime format '{format_spec}': {e}")
        else:
            # Default timestamp format
            return now_utc.strftime("%Y%m%d-%H%M%S")

    try:
        processed_format = re.sub(timestamp_pattern, replace_timestamp, processed_format)
    except ValueError:
        raise  # Re-raise ValueError from strftime

    # Build backup filename with remaining placeholders
    try:
        backup_name = processed_format.format(
            stem=source.stem,
            suffix=source.suffix,
            name=source.name,
            pid=os.getpid(),
        )
    except (KeyError, ValueError) as e:
        raise ValueError(f"Error formatting backup filename: {e}")

    backup_path = backup_dir / backup_name

    # Check if backup already exists
    if backup_path.exists() and not exist_ok:
        raise FileExistsError(f"Backup file already exists: {backup_path}")

    # Perform backup using shutil.copy2 (preserves metadata)
    # This can raise: PermissionError, OSError (disk full, I/O error, etc.)
    shutil.copy2(source, backup_path)

    return backup_path


def clean_dir(
    path: str | os.PathLike[str],
    *,
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
        >>> clean_dir("/tmp/cache")                                 # doctest: +SKIP
        >>> clean_dir("/tmp/cache", missing_ok=True)                # doctest: +SKIP
        >>> clean_dir("/tmp/cache", ignore_errors=True)             # doctest: +SKIP
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


def copy_file(
    source: str | os.PathLike[str],
    dest: str | os.PathLike[str],
    *,
    callback: Callable[[int, int], None] | None = None,
    chunk_size: int = 8 * 1024 * 1024,
    follow_symlinks: bool = True,
    preserve_metadata: bool = True,
    overwrite: bool = True,
) -> Path:
    """
    Copy file with optional progress tracking support.

    Similar to shutil.copy2() but with progress tracking via callback for large files.

    Args:
        source: Source file path (string or PathLike object).
        dest: Destination path (string or PathLike object).
            Can be a file path or directory. If directory, the file is copied
            into it using the source filename.
        callback: Optional progress callback function.
            Signature: callback(bytes_written: int, total_bytes: int) -> None
            Called after each chunk is written to destination. Not called on empty files.
        chunk_size: Size in bytes for each copy chunk. Defaults to 8 MB.
            Larger chunks mean faster copies but less frequent progress updates.
            Set to 0 to use file_size (single chunk, minimal progress updates).
        follow_symlinks: If True, copies the file content that symlink points to.
            If False, creates a new symlink at dest pointing to the same target.
        preserve_metadata: If True, preserves file metadata (timestamps, permissions).
            Similar to shutil.copy2(). If False, only copies content like shutil.copy().
        overwrite: If False, raises FileExistsError if destination file exists.
            If True, overwrites existing files.

    Returns:
        Path: Absolute path to the destination file.

    Raises:
        ValueError: If source and dest are the same file, or if chunk_size is negative.
        FileExistsError: If destination exists and overwrite=False.
        IsADirectoryError: If source is a directory (only files supported).
        Exception: Types propagated from Path.stat(), open(), StreamingFile, and shutil.copystat():
            FileNotFoundError, PermissionError, OSError, and other I/O exceptions.

    Notes:
        - For files under ~1MB, progress callback overhead may exceed copy time.
          Consider callback=None for small files.
        - The function creates parent directories of dest if they don't exist.
        - When dest is a directory, behavior matches shutil.copy: the file is
          copied into the directory with its original basename.
        - Symlink handling matches shutil.copy2 behavior by default.
        - Empty files (0 bytes) are copied without calling the callback.
        - Progress tracking reports bytes written to destination, which accurately
          reflects copy progress.

    Examples:
        Basic copy with progress:

        >>> def progress(current, total):
        ...     print(f"Copying: {current}/{total} bytes ({current/total*100:.1f}%)")
        ...
        >>> copy_file("large_video.mp4", "backup/", callback=progress)      # doctest: +SKIP
        Path('/absolute/path/to/backup/large_video.mp4')

        Copy to specific filename without progress:

        >>> copy_file("data.csv", "archive/data_backup.csv")               # doctest: +SKIP
        Path('/absolute/path/to/archive/data_backup.csv')

        Prevent overwriting existing files:

        >>> copy_file("config.json", "prod/config.json", overwrite=False)  # doctest: +SKIP
        # Raises FileExistsError if prod/config.json exists

        Copy with custom chunk size (faster, less frequent updates):

        >>> copy_file("huge.bin", "backup/", chunk_size=64*1024*1024)       # doctest: +SKIP

        Copy without preserving metadata:

        >>> copy_file("file.txt", "copy.txt", preserve_metadata=False)      # doctest: +SKIP

        Handle symlinks explicitly:

        >>> # Copy symlink as symlink (don't follow)
        >>> copy_file("link.txt", "copy_link.txt", follow_symlinks=False)   # doctest: +SKIP
    """
    # Convert to Path objects for consistent handling
    source = Path(source)
    dest = Path(dest)

    # Validation - raises ValueError (our exception)
    if chunk_size < 0:
        raise ValueError(f"chunk_size must be non-negative, got {chunk_size}")

    # Resolve source (respecting follow_symlinks)
    if follow_symlinks:
        source_resolved = source.resolve()
    else:
        source_resolved = source

    # Check if source is a directory - raises IsADirectoryError (our exception)
    # Do this before checking existence to provide clearer error message
    if source_resolved.is_dir():
        raise IsADirectoryError(f"Source is a directory, not a file: {source}")

    # Path.exists(), Path.stat() may raise FileNotFoundError, PermissionError (propagated)

    # Handle symlinks when follow_symlinks=False
    if not follow_symlinks and source.is_symlink():
        link_target = os.readlink(source)
        if dest.is_dir():
            dest = dest / source.name
        if dest.exists():
            if not overwrite:
                raise FileExistsError(f"Destination already exists: {dest}")
            dest.unlink()
        os.symlink(link_target, dest)
        return dest.resolve()

    # Determine actual destination path
    if dest.is_dir():
        dest = dest / source.name

    # Check if source and destination are the same - raises ValueError (our exception)
    try:
        if source_resolved.samefile(dest):
            raise ValueError(f"Source and destination are the same file: {source}")
    except FileNotFoundError:
        # dest doesn't exist yet, which is fine
        pass

    # Check overwrite setting - raises FileExistsError (our exception)
    if dest.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {dest}")

    # Create destination parent directory if needed
    # Path.mkdir() may raise PermissionError, OSError (propagated)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Get source file size for progress tracking
    # Path.stat() may raise PermissionError, OSError (propagated)
    file_size = source_resolved.stat().st_size

    # Handle empty files quickly without overhead
    if file_size == 0:
        dest.touch()
        if preserve_metadata:
            shutil.copystat(source_resolved, dest)
        return dest.resolve()

    # Perform the copy with progress tracking on destination write
    # StreamingFile may raise ValueError, PermissionError, OSError (propagated)
    with open(source_resolved, "rb") as source_f:
        with StreamingFile(
            dest,
            "wb",
            callback=callback,
            chunk_size=chunk_size,
            expected_size=file_size,
        ) as dest_f:
            while True:
                # Read chunks and let StreamingFile handle progress on writes
                read_size = chunk_size if chunk_size > 0 else file_size
                chunk = source_f.read(read_size)

                if not chunk:
                    break

                # StreamingFile.write() tracks progress automatically
                dest_f.write(chunk)

    # Preserve metadata if requested
    # shutil.copystat() may raise PermissionError, OSError (propagated)
    if preserve_metadata:
        shutil.copystat(source_resolved, dest)

    return dest.resolve()


def find_files(
    path: str | os.PathLike[str],
    pattern: str = "*",
    *,
    exclude: list[str] | None = None,
    max_depth: int | None = None,
    follow_symlinks: bool = False,
    include_dirs: bool = False,
    predicate: Callable[[Path], bool] | None = None,
) -> Iterator[Path]:
    """
    Find files recursively with glob-style patterns.

    A more flexible alternative to glob.glob() and Path.rglob() with support
    for exclusion patterns, depth control, and custom filtering.

    Args:
        path: Root directory to search. Must exist and be a directory.
        pattern: Glob pattern matching against filename only.
            Uses fnmatch syntax: "*", "*.py", "test_*", "[!.]*.txt"
        exclude: Simple glob patterns to exclude, matching against the
            relative path from the search root. Uses fnmatch syntax.
            Common patterns:
            - "*.pyc", "*.pyo" - Compiled Python files anywhere
            - "__pycache__", ".git" - Directories anywhere
            - ".*" - Hidden files/directories (names starting with .)
            - "tests" - Anything named "tests" at any depth
            Note: Does NOT support ** recursive wildcard (gitignore syntax).
            When a directory is excluded, its entire subtree is skipped.
            For complex path-based exclusions, use predicate with pathspec.
        max_depth: Maximum directory depth relative to path.
            - None (default): Unlimited depth
            - 0: Only files directly in path
            - 1: path and immediate subdirectories
        follow_symlinks: If True, follow symbolic links. Default False.
        include_dirs: If True, yield directories that match pattern and are
            not excluded. Default False (files only).
        predicate: Optional callable for custom filtering. Called with each
            Path after pattern/exclude matching. Return True to include.

    Returns:
        Iterator[Path]: Paths to matching files (and directories if include_dirs=True).

    Raises:
        FileNotFoundError: If path does not exist.
        NotADirectoryError: If path exists but is not a directory.
        Exception: Types propagated from os.scandir: ``PermissionError``, ``OSError``

    Examples:
        Basic usage:

        >>> # Find all Python files
        >>> list(find_files("src", "*.py"))                         # doctest: +SKIP
        [Path('src/main.py'), Path('src/utils.py'), Path('src/tests/test_main.py')]

        >>> # Exclude by name (matches anywhere in tree)
        >>> list(find_files("src", "*.py", exclude=["test_*"]))     # doctest: +SKIP
        [Path('src/main.py'), Path('src/utils.py')]

        >>> # Exclude directories (skips entire subtree)
        >>> list(find_files(".", "*.py", exclude=["__pycache__", ".git", "venv"]))      # doctest: +SKIP

        >>> # Common Python project exclusions
        >>> PYTHON_IGNORE = [
        ...     ".*",           # Hidden files/dirs
        ...     "*.pyc",        # Compiled files
        ...     "__pycache__",  # Cache dirs
        ...     "*.egg-info",   # Package metadata
        ...     "dist",         # Distribution
        ...     "build",        # Build output
        ...     "venv",         # Virtual envs
        ...     ".venv",
        ... ]
        >>> list(find_files(".", "*.py", exclude=PYTHON_IGNORE))                        # doctest: +SKIP

        >>> # Limit search depth
        >>> list(find_files("src", "*.py", max_depth=0))                                # doctest: +SKIP
        [Path('src/main.py'), Path('src/utils.py')]

        >>> # Include directories
        >>> list(find_files("src", "cache*", include_dirs=True, exclude=[".*"]))        # doctest: +SKIP
        [Path('src/cache'), Path('src/tests/cache_temp')]

        Loading exclusions from files:

        >>> def load_ignore_patterns(filepath: str) -> list[str]:
        ...     '''Load exclusion patterns from file (one per line).'''
        ...     return [
        ...         line.strip()
        ...         for line in Path(filepath).read_text().splitlines()
        ...         if line.strip() and not line.startswith('#')
        ...     ]
        >>>
        >>> # File format (simple, not gitignore):
        >>> # # Python build artifacts
        >>> # *.pyc
        >>> # __pycache__
        >>> # .pytest_cache
        >>> patterns = load_ignore_patterns('.buildignore')             # doctest: +SKIP
        >>> list(find_files(".", "*.py", exclude=patterns))             # doctest: +SKIP

        Gitignore-style exclusions with pathspec:

        >>> # For gitignore syntax (**, negation !, trailing /), use pathspec
        >>> import pathspec                                                     # doctest: +SKIP
        >>>
        >>> # Simple .gitignore usage
        >>> with open('.gitignore') as f:                                       # doctest: +SKIP
        ...     spec = pathspec.PathSpec.from_lines('gitwildmatch', f)          # doctest: +SKIP
        >>> root = Path('.').resolve()
        >>> files = find_files(                                                 # doctest: +SKIP
        ...     ".",
        ...     "*",
        ...     predicate=lambda p: not spec.match_file(str(p.relative_to(root)))
        ... )

        Advanced filtering with predicates:

        >>> # Regex matching
        >>> import re
        >>> pattern = re.compile(r"test_.*py$")
        >>> list(find_files("tests", "*.py", predicate=lambda p: pattern.search(p.name)))       # doctest: +SKIP

        >>> # File size filter
        >>> large_files = find_files(
        ...     "data",
        ...     "*",
        ...     exclude=[".*"],
        ...     predicate=lambda p: p.stat().st_size > 1_000_000
        ... )

        >>> # Modification time filter
        >>> from datetime import datetime, timedelta
        >>> recent = datetime.now() - timedelta(days=7)
        >>> recent_logs = find_files(                                                           # doctest: +SKIP
        ...     "logs",
        ...     "*.log",
        ...     exclude=["*.gz", "archived"],
        ...     predicate=lambda p: datetime.fromtimestamp(p.stat().st_mtime) > recent
        ... )

        >>> # Multiple conditions
        >>> def is_recent_python_file(p: Path) -> bool:
        ...     if p.suffix != '.py':
        ...         return False
        ...     age = datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)
        ...     size = p.stat().st_size
        ...     return age.days < 30 and size > 100 and size < 100_000
        >>>
        >>> list(find_files("src", "*", predicate=is_recent_python_file))                      # doctest: +SKIP

    Notes:
        - Both pattern and exclude use fnmatch syntax: *, ?, [abc], [!abc]
        - Pattern matches against filename only
        - Exclude patterns match against relative path from search root
        - Exclude does NOT support ** (gitignore recursive wildcard)
        - For gitignore-style patterns (**, !, trailing /), use pathspec
          library with predicate parameter (see examples above)
        - When a directory matches exclude, entire subtree is skipped
        - Predicate called after pattern/exclude for efficiency
        - Permission errors on directories are skipped silently
        - Symlink loops detected and skipped when follow_symlinks=True

    See Also:
        pathlib.Path.rglob(): Simpler recursive globbing with ** support
        fnmatch.fnmatch(): Pattern matching function used for pattern/exclude
        pathspec library: Full gitignore syntax support for predicates
    """
    # Validate and normalize input path
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")

    # Validate pattern
    if not pattern:
        raise ValueError("Pattern cannot be empty")

    # Normalize exclude patterns
    exclude_patterns = exclude if exclude is not None else []

    # Track visited inodes (device, inode) to detect symlink loops and duplicates
    visited_inodes: set[tuple[int, int]] = set()

    def _is_excluded(rel_path: Path) -> bool:
        """Check if path matches any exclude pattern."""
        path_str = str(rel_path).replace(os.sep, "/")

        for pattern_str in exclude_patterns:
            # Match against full relative path
            if fnmatch(path_str, pattern_str):
                return True
            # Also check each path component (for patterns like "*.pyc" or "__pycache__")
            if fnmatch(rel_path.name, pattern_str):
                return True
            # Check any parent directory names
            for part in rel_path.parts:
                if fnmatch(part, pattern_str):
                    return True
        return False

    def _walk(current: Path, depth: int) -> Iterator[Path]:
        """Recursively walk directory tree."""
        # Check depth limit
        if max_depth is not None and depth > max_depth:
            return

        # Handle symlinks - track by inode to detect loops
        if follow_symlinks and current.is_symlink():
            try:
                stat_info = current.stat()
                inode_key = (stat_info.st_dev, stat_info.st_ino)

                # Check if we've already visited this inode
                if inode_key in visited_inodes:
                    return  # Loop detected, skip
                visited_inodes.add(inode_key)
            except (OSError, RuntimeError):
                # Can't stat or circular symlink
                return

        # Try to scan directory
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        entry_path = Path(entry.path)

                        # Calculate relative path from root
                        try:
                            rel_path = entry_path.relative_to(root)
                        except ValueError:
                            # Path is not relative to root (shouldn't happen)
                            continue

                        # Check exclusions - skip early if excluded
                        if _is_excluded(rel_path):
                            continue

                        # Check if it's a directory
                        is_dir = entry.is_dir(follow_symlinks=follow_symlinks)

                        if is_dir:
                            # Recurse into directories
                            yield from _walk(entry_path, depth + 1)

                            # Yield directory if requested and matches pattern
                            if include_dirs and fnmatch(entry.name, pattern):
                                if predicate is None or predicate(entry_path):
                                    yield entry_path
                        else:
                            # Check if file matches pattern
                            if fnmatch(entry.name, pattern):
                                # Check file accessibility using os.access
                                # This respects actual permissions including ownership
                                if not os.access(entry_path, os.R_OK):
                                    continue  # File not readable, skip it

                                # For files, track inode to avoid yielding duplicates through symlinks
                                if follow_symlinks:
                                    try:
                                        stat_info = entry_path.stat()
                                        inode_key = (stat_info.st_dev, stat_info.st_ino)
                                        if inode_key in visited_inodes:
                                            continue  # Already yielded this file
                                        visited_inodes.add(inode_key)
                                    except OSError:
                                        continue  # Can't stat, skip

                                if predicate is None or predicate(entry_path):
                                    yield entry_path

                    except (PermissionError, OSError):
                        # Skip individual entries we can't access
                        continue

        except (PermissionError, OSError):
            # Skip directories we can't read
            return

    # Start walking from root at depth 0
    yield from _walk(root, 0)
