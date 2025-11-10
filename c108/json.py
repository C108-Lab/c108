"""
Utilities for safe JSON file read/write/update with sensible defaults and optional atomic operations.
"""

# Standard library -----------------------------------------------------------------------------------------------------
import json
import os

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

# Local ----------------------------------------------------------------------------------------------------------------
from c108.os import atomic_open

# Methods --------------------------------------------------------------------------------------------------------------

T = TypeVar("T")


def read_json(
    path: str | os.PathLike[str], *, default: T = None, encoding: str = "utf-8"
) -> Any | T:
    """
    Read JSON from file with graceful error handling.

    Reads and parses a JSON file, returning a default value if the file doesn't exist or contains invalid JSON.
    This provides a safer alternative to raw json.load() for configuration files and optional data sources.

    Args:
        path: Path to the JSON file to read.
        default: Value to return if file is missing or contains invalid JSON. Defaults to None.
        encoding: Text encoding to use when reading the file. Defaults to "utf-8".

    Returns:
        Parsed JSON data (typically dict or list), or the default value if reading fails.

    Raises:
        OSError: If file exists but cannot be read due to permissions or I/O errors (not FileNotFoundError).
        TypeError: If path is not a valid path-like object.

    Examples:
        >>> config = read_json("config.json", default={})
        >>> k = config.get("api_key", "default-key")

        >>> # Type-safe with explicit default
        >>> settings: dict[str, Any] = read_json(Path("settings.json"), default={})

        >>> # Returns None if file missing
        >>> cache = read_json("cache.json")
        >>> if cache is None:
        ...     print("No cache found")
        No cache found

        >>> # Custom encoding for legacy files
        >>> data = read_json("legacy.json", encoding="latin-1", default=[])
    """
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("path must be str or os.PathLike")

    try:
        with open(path, "r", encoding=encoding) as f:
            return json.load(f)

    except FileNotFoundError:
        return default

    except json.JSONDecodeError:
        return default


def write_json(
    path: str | os.PathLike[str],
    data: Any,
    *,
    indent: int = 2,
    atomic: bool = True,
    encoding: str = "utf-8",
    ensure_ascii: bool = False,
) -> None:
    """
    Write JSON to file with safe defaults and optional atomic write.

    Writes data to a JSON file with sensible formatting defaults. Supports atomic writes to prevent data corruption
    if the process is interrupted mid-write (e.g., power loss, SIGKILL).

    Args:
        path: Destination path for the JSON file.
        data: Python object to serialize to JSON. Must be JSON-serializable (dict, list, str, int, float, bool, None).
        indent: Number of spaces for indentation. Use None for compact output. Defaults to 2.
        atomic: If True, write to a temporary file then atomically rename to target path. This prevents corruption
            but may not preserve file metadata (permissions, extended attributes). Defaults to True.
        encoding: Text encoding to use when writing the file. Defaults to "utf-8".
        ensure_ascii: If True, escape non-ASCII characters. If False, write Unicode directly. Defaults to False.

    Raises:
        TypeError: If data is not JSON-serializable; if path is not a valid path-like object.
        OSError: If file cannot be written due to permissions, disk space, or I/O errors.
        ValueError: If indent is negative.

    Examples:
        >>> import os, tempfile, json
        >>> tmp = tempfile.gettempdir()
        >>> path_json = os.path.join(tmp, "example.json")
        >>>
        >>> write_json(path_json, {"debug": True, "timeout": 30})
        >>> with open(path_json, encoding="utf-8") as f:
        ...     json.load(f)
        {'debug': True, 'timeout': 30}

        >>> # Compact output for space-constrained environments
        >>> write_json(path_json, {"a": 1, "b": 2}, indent=None)

        >>> # ASCII-safe output for legacy systems
        >>> write_json(path_json, {"name": "François"}, ensure_ascii=True)

        >>> os.remove(path_json)
    """
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("path must be str or os.PathLike")

    if indent is not None and indent < 0:
        raise ValueError("indent must be non-negative")

    if atomic:
        with atomic_open(path, mode="w", encoding=encoding) as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
            # Add trailing newline for POSIX compliance
            f.write("\n")
    else:
        with open(path, "w", encoding=encoding) as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
            # Add trailing newline for POSIX compliance
            f.write("\n")


def update_json(
    path: str | os.PathLike[str],
    updater: Callable[[Any], Any] | None = None,
    *,
    key: str | None = None,
    value: Any = None,
    default: Any = None,
    encoding: str = "utf-8",
    indent: int = 2,
    atomic: bool = True,
    ensure_ascii: bool = False,
    create_parents: bool = True,
) -> None:
    """Read JSON, apply transformation, and write back atomically.

    Supports two modes of operation:
        1. Function mode: Apply a transformation function to the entire data structure
        2. Key mode: Update a value at a specific key path using dot notation

    The entire operation is atomic if atomic=True, preventing partial updates.

    Args:
        path: Path to the JSON file to update.
        updater: Callable that transforms the entire data structure (current: Any) -> Any.
            Mutually exclusive with key parameter.
        key: Dot-separated key path for updating nested values (e.g., "database.host" or
            "server.settings.port"). Mutually exclusive with updater parameter.
            Use simple keys like "count" for top-level updates.
        value: New value to set at the key path. Required when key is provided, ignored otherwise.
        default: Value to use if file is missing or contains invalid JSON. Defaults to None.
        encoding: Text encoding for reading and writing. Defaults to "utf-8".
        indent: Number of spaces for indentation in output. Use None for compact. Defaults to 2.
        atomic: If True, write atomically to prevent corruption. Defaults to True.
        ensure_ascii: If True, escape non-ASCII characters. Defaults to False.
        create_parents: If True, automatically create missing parent dicts in nested key paths.
            If False, raises KeyError when intermediate keys don't exist. Only applies in key mode.
            Defaults to True.

    Raises:
        ValueError: If both updater and key are provided, or if neither is provided, or if
            key is provided without value.
        TypeError: If updater returns non-JSON-serializable data, or if key mode is used
            but root data is not a dict, or if nested path encounters non-dict intermediate values.
        KeyError: If nested key path references missing keys and create_parents=False.
        OSError: If file cannot be read or written due to permissions or I/O errors.
        Exception: Any exception raised by the updater function is propagated to caller.

    Examples:
        >>> import os, tempfile
        >>> from datetime import datetime, timezone
        >>> tmp = tempfile.gettempdir()
        >>> file_json = os.path.join(tmp, "example_update.json")
        >>> write_json(file_json, {})

        >>> # Basic key mode - update top-level keys:
        >>> update_json(file_json, key="last_run", value=datetime.now(timezone.utc).isoformat(), default={})
        >>> update_json(file_json, key="count", value=42, default={})

        >>> # Nested key mode - update deeply nested values:
        >>> update_json(file_json, key="database.host", value="localhost", default={})
        >>> update_json(file_json, key="server.port", value=8080, default={})
        >>> update_json(file_json, key="ui.theme.colors.primary", value="#007bff", default={})

        >>> # Deep nesting with automatic parent creation:
        >>> update_json(file_json, key="features.experimental.beta", value=True, default={})
        >>> # Result: {"features": {"experimental": {"beta": True}}}

        >>> # Strict mode - fail if intermediate keys don't exist:
        >>> update_json(
        ...     file_json,
        ...     key="database.host",
        ...     value="localhost",
        ...     create_parents=False,
        ...     default={}
        ... )  # Raises KeyError if "database" key is missing

        >>> # Function mode - complex transformations:
        >>> update_json(
        ...     file_json,
        ...     updater=lambda cfg: {**cfg, "last_modified": datetime.now(timezone.utc).isoformat()},
        ...     default={}
        ... )

        >>> # Increment a counter with function mode:
        >>> update_json(
        ...     file_json,
        ...     updater=lambda data: {"count": data.get("count", 0) + 1},
        ...     default={}
        ... )

        >>> # Append to a list:
        >>> def add_entry(data):
        ...     entries = data if isinstance(data, list) else []
        ...     entries.append({"id": len(entries), "value": "new"})
        ...     return entries
        >>> update_json(file_json, updater=add_entry, default=[])

        >>> # Reset file to dict for next examples
        >>> write_json(file_json, {})

        >>> # Update nested counter (comparing both modes):
        >>> # Key mode - simpler for direct updates
        >>> update_json(file_json, key="stats.visits", value=100, default={})

        >>> # Function mode - needed for increments
        >>> def increment_visits(data):
        ...     if "stats" not in data:
        ...         data["stats"] = {}
        ...     data["stats"]["visits"] = data["stats"].get("visits", 0) + 1
        ...     return data
        >>> update_json(file_json, updater=increment_visits, default={})

        >>> # Error case: Cannot use both modes
        >>> update_json(file_json, lambda x: x, key="foo", value="bar")
        Traceback (most recent call last):
        ...
        ValueError: specify either updater or key, not both

        >>> os.remove(file_json)
    """
    # Validate mutually exclusive modes
    if updater is not None and key is not None:
        raise ValueError("specify either updater or key, not both")

    if updater is None and key is None:
        raise ValueError("must specify either updater or key")

    # Validate key mode arguments
    if key is not None and value is None:
        raise ValueError("value is required when key is provided")

    # Read current data
    data = read_json(path, default=default, encoding=encoding)

    # Apply transformation based on mode
    if updater is not None:
        # Function mode: apply transformation function
        data = updater(data)
    else:
        # Key mode: update value at key path
        if not isinstance(data, dict):
            raise TypeError(f"Cannot set key on non-dict type: {type(data).__name__}")

        # Parse key path
        keys = key.split(".")

        # Navigate to parent of target key
        current = data
        for i, k in enumerate(keys[:-1]):
            if k not in current:
                if create_parents:
                    current[k] = {}
                else:
                    raise KeyError(f"Key '{k}' not found in path '{'.'.join(keys[: i + 1])}'")

            if not isinstance(current[k], dict):
                raise TypeError(
                    f"Cannot traverse through non-dict at key '{k}': found {type(current[k]).__name__}"
                )

            current = current[k]

        # Set the final key
        final_key = keys[-1]
        current[final_key] = value

    # Write back to file
    write_json(
        path,
        data,
        indent=indent,
        atomic=atomic,
        encoding=encoding,
        ensure_ascii=ensure_ascii,
    )
