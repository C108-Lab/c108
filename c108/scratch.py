"""
Utilities for creating and managing temporary directories with customizable names

Provides temp_dir(), a context manager for secure temporary directories with
timestamp/PID/randomized naming.
"""

# Why this wrapper: smooth Path-first usage (no manual conversion from str), clear, validated name formatting,
# and helpful options (parent dir, delete flag, ignore cleanup errors) that address
# stdlib TemporaryDirectory ergonomics.

# Standard library -----------------------------------------------------------------------------------------------------
import os
import re
import tempfile
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


# Methods --------------------------------------------------------------------------------------------------------------


@contextmanager
def temp_dir(
    *,
    parent: str | os.PathLike[str] | None = None,
    name_format: str = "tmp-{random}",
    delete: bool = True,
    ignore_cleanup_errors: bool = False,
) -> Iterator[Path]:
    """
    Context manager that provides a Path object to a temporary directory.

    The directory is created when entering the context and automatically removed
    (along with its contents) when exiting, unless delete=False.

    Args:
        parent: Optional parent directory where temp dir will be created.
            If None, uses the system default temp directory.
        name_format: Format string for directory name. Available placeholders:
            - {random}: Random characters for uniqueness (e.g., "x7k2p9qr")
            - {timestamp}: UTC timestamp (e.g., "20250101-143010")
            - {timestamp:fmt}: Formatted UTC timestamp using strftime syntax in FMT, e.g. ``%Y%m%d``
            - {pid}: Process ID
        delete: If True (default), remove the directory on context exit.
        ignore_cleanup_errors: If True, suppress errors during cleanup when delete=True.

    Yields:
        Path: A pathlib.Path object pointing to the temporary directory.

    Raises:
        OSError: From tempfile.TemporaryDirectory if directory creation fails.
        PermissionError: From tempfile.TemporaryDirectory if lacking permissions.
        ValueError: If name_format contains invalid placeholders or format syntax.

    Examples:
        Basic usage:
        >>> with temp_dir() as tmp:
        ...     config_file = tmp / "config.json"
        ...     n_bytes = config_file.write_text('{"key": "value"}')
        ...     assert config_file.exists()

        Timestamped directory:
        >>> with temp_dir(name_format="build-{timestamp:%Y%m%d}-{random}") as tmp:
        ...     # Creates: /tmp/build-20251014-x7k2p9qr
        ...     pass

        Process-specific directory:
        >>> with temp_dir(name_format="worker-{pid}-{random}") as tmp:
        ...     # Creates: /tmp/worker-12345-x7k2p9qr
        ...     pass

        Custom parent directory:
        >>> with temp_dir(parent="/tmp") as tmp:
        ...     assert tmp.parent == Path("/tmp")

        Preserve for debugging:
        >>> with temp_dir(delete=False, name_format="debug-{timestamp}-{random}") as tmp:
        ...     debug_log = tmp / "debug.log"
        ...     n_bytes = debug_log.write_text("Debug info")
        >>> assert debug_log.exists()

    Note:
        The temporary directory is created in a secure manner with appropriate
        permissions (0o700 on Unix-like systems). Timestamps use UTC to ensure
        unambiguous, sortable filenames. The {random} placeholder ensures
        uniqueness even with identical timestamps or PIDs.
    """
    # Parse the name_format to extract prefix and suffix around {random}
    prefix, suffix = _temp_dir_parse_name_fmt(name_format)

    # Create the temporary directory with parsed prefix/suffix
    with tempfile.TemporaryDirectory(
        prefix=prefix,
        suffix=suffix,
        dir=parent,
        delete=delete,
        ignore_cleanup_errors=ignore_cleanup_errors,
    ) as tmpdir_str:
        yield Path(tmpdir_str)


def _temp_dir_parse_name_fmt(name_format: str) -> tuple[str, str]:
    """
    Parse name_format string and return (prefix, suffix) for tempfile.TemporaryDirectory.

    Replaces placeholders with their values and splits around the random component.

    Args:
        name_format: Format string with placeholders like {random}, {timestamp}, {pid}.

    Returns:
        Tuple of (prefix, suffix) strings for tempfile.TemporaryDirectory.

    Raises:
        ValueError: If name_format contains invalid placeholders or is malformed.
    """
    if not name_format:
        raise ValueError("name_format cannot be empty")

    # Replace non-random placeholders first
    result = name_format

    # Replace {pid}
    result = result.replace("{pid}", str(os.getpid()))

    # Replace {timestamp} with optional format specifier
    # Pattern: {timestamp} or {timestamp:%Y%m%d-%H%M%S}
    timestamp_pattern = r"\{timestamp(?::([^}]+))?\}"

    def replace_timestamp(match):
        format_spec = match.group(1)
        if format_spec:
            # Custom format like {timestamp:%Y%m%d}
            try:
                return datetime.now(timezone.utc).strftime(format_spec)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid timestamp format '{format_spec}': {e}") from e
        else:
            # Default format
            return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    result = re.sub(timestamp_pattern, replace_timestamp, result)

    # Check if {random} exists - it's required for uniqueness
    if "{random}" not in result:
        raise ValueError(
            f"name_format must contain {{random}} placeholder for uniqueness. Got: '{name_format}'"
        )

    # Check for any remaining unprocessed placeholders
    remaining_placeholders = re.findall(r"\{([^}]+)\}", result)
    if remaining_placeholders and remaining_placeholders != ["random"]:
        unknown = [p for p in remaining_placeholders if p != "random"]
        raise ValueError(
            f"Unknown placeholders in name_format: {unknown}. "
            f"Valid placeholders: random, timestamp, timestamp:FORMAT, pid"
        )

    # Split on {random} to get prefix and suffix
    parts = result.split("{random}", maxsplit=1)

    if len(parts) != 2:
        # This shouldn't happen given the check above, but handle it anyway
        raise ValueError(f"Invalid name_format: multiple {{random}} placeholders found")

    prefix = parts[0] if parts[0] else None
    suffix = parts[1] if parts[1] else None

    return prefix, suffix


# Private methods ------------------------------------------------------------------------------------------------------
