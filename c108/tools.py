"""
Miscelenious helpers for ACII, dicts and sequences.

**Dictionary navigation** (dict_*): Safe traversal and modification of nested
dictionaries using dot-notation or key sequences. Handles missing keys and
intermediate path creation with consistent error handling.

**Defensive collection utilities**: Normalize inputs into predictable types
(listify), safely access sequences (sequence_get), and retrieve call stack
information for debugging contexts (get_caller_name).
"""

# Standard library -----------------------------------------------------------------------------------------------------

import collections.abc as abc
from inspect import stack
from typing import (
    Any,
    Mapping,
    Sequence,
    Callable,
    TypeVar,
)

from c108.formatters import fmt_any, fmt_type, fmt_value

# Classes --------------------------------------------------------------------------------------------------------------
T = TypeVar("T")


# Methods --------------------------------------------------------------------------------------------------------------
def as_ascii(
    s: str | bytes | bytearray, replacement: str | bytes | None = None
) -> str | bytes | bytearray:
    """
    Convert a string-like object to ASCII by replacing non-ASCII characters and preserving object type.

    This function processes each character/byte in the input and replaces any
    non-ASCII value (code point or byte value >= 128) with the specified
    replacement. The return type matches the input type.

    Args:
        s: The input str, bytes, or bytearray to sanitize.
        replacement: The character or byte to use for replacement.
                     None translates to '_' for str and b'_' for bytes/bytearray.
                     Must be a single ASCII character/byte.

    Returns:
        A new object of the same type as the input (str, bytes, or bytearray)
        containing only ASCII characters/bytes.

    Raises:
        TypeError: If the input `s` is not a str, bytes, or bytearray, or if
                   `replacement` has an incompatible type.
        ValueError: If `replacement` is not a single ASCII character/byte.

    Examples:
        >>> # Process a standardstring
        >>> as_ascii("Hello, 世界!")
        'Hello, __!'

        >>> # Process a UTF-8 encoded byte string with a custom replacement
        >>> euro_price_bytes = "Price: 100€".encode('utf-8')
        >>> euro_price_bytes
        b'Price: 100\\xe2\\x82\\xac'
        >>> as_ascii(euro_price_bytes, replacement=b'?')
        b'Price: 100???'

        >>> # Process a mutable bytearray
        >>> data = bytearray(b'caf\\xc3\\xa9') # bytearray for 'café'
        >>> as_ascii(data)
        bytearray(b'caf__')
    """
    if isinstance(s, str):
        # Handle string input
        if replacement is None:
            replacement = "_"
        if not isinstance(replacement, str):
            raise TypeError(f"replacement for str input must be str, not {fmt_type(replacement)}")
        if len(replacement) != 1:
            raise ValueError("replacement must be a single character")
        if ord(replacement) >= 128:
            raise ValueError("replacement character must be ASCII")

        return "".join(replacement if ord(char) >= 128 else char for char in s)

    elif isinstance(s, (bytes, bytearray)):
        # Handle bytes and bytearray input
        if replacement is None:
            replacement = b"_"
        if not isinstance(replacement, bytes):
            raise TypeError(
                f"replacement for bytes input must be bytes, not {fmt_type(replacement)}"
            )
        if len(replacement) != 1:
            raise ValueError("replacement must be a single byte")

        # The replacement byte's value must be < 128
        if replacement[0] >= 128:
            raise ValueError("replacement byte must be ASCII (< 128)")

        new_bytes = (replacement[0] if byte >= 128 else byte for byte in s)

        if isinstance(s, bytearray):
            return bytearray(new_bytes)
        else:  # bytes
            return bytes(new_bytes)

    else:
        raise TypeError(f"Input must be str, bytes, or bytearray, not {fmt_type(s)}")


def dict_get(
    source: dict | Mapping,
    key: str | Sequence[str],
    default: Any = None,
    *,
    separator: str = ".",
) -> Any:
    """
    Get a value from a nested dictionary using dot-separated keys or a sequence of keys.

    Args:
        source: The dictionary or mapping to search in
        key: Either a dot-separated string ('a.b.c') or sequence of keys ['a', 'b', 'c']
        default: Value to return if the key path is not found
        separator: Character used to split string keys (default: '.')

    Returns:
        The value at the specified key path, or default if not found

    Raises:
        TypeError: If source is not a dict or Mapping
        ValueError: If key is empty or invalid

    Examples:
        >>> data = {'user': {'profile': {'name': 'John'}}}
        >>> dict_get(data, 'user.profile.name')
        'John'

        >>> dict_get(data, ['user', 'profile', 'name'])
        'John'

        >>> dict_get(data, 'user.missing', 'default')
        'default'
    """
    if not isinstance(source, (dict, abc.Mapping)):
        raise TypeError(f"source must be dict or Mapping, got {type(source).__name__}")

    # Handle key parameter - string or sequence
    if isinstance(key, str):
        keys = key.split(separator)
    elif isinstance(key, abc.Sequence) and not isinstance(key, (str, bytes)):
        keys = list(key)
        if not keys:
            raise ValueError("key sequence cannot be empty")
    else:
        raise TypeError(f"key must be str or sequence, got {type(key).__name__}")

    # Navigate through the nested structure
    current = source
    for k in keys:
        if not isinstance(current, (dict, abc.Mapping)):
            return default
        if k not in current:
            return default
        current = current[k]

    return current


def dict_set(
    dest: dict | abc.MutableMapping,
    key: str | Sequence[str],
    value: Any,
    *,
    separator: str = ".",
    create_missing: bool = True,
) -> None:
    """
    Set a value in a nested dictionary using dot-separated keys or a sequence of keys.

    Args:
        dest: The dictionary or mutable mapping to modify
        key: Either a dot-separated string ('a.b.c') or sequence of keys ['a', 'b', 'c']
        value: The value to set at the specified key path
        separator: Character used to split string keys (default: '.')
        create_missing: If True, creates intermediate dictionaries as needed (default: True)

    Raises:
        TypeError: If target is not a dict or MutableMapping
        ValueError: If key is empty or invalid
        KeyError: If create_missing=False and intermediate keys don't exist
        TypeError: If intermediate value exists but is not a dict/MutableMapping

    Examples:
        >>> data = {}
        >>> dict_set(data, 'user.profile.name', 'John')
        >>> data
        {'user': {'profile': {'name': 'John'}}}

        >>> dict_set(data, ['user', 'profile', 'age'], 30)
        >>> data
        {'user': {'profile': {'name': 'John', 'age': 30}}}

        >>> dict_set(data, 'user.email', 'john@example.com')
        >>> data
        {'user': {'profile': {'name': 'John', 'age': 30}, 'email': 'john@example.com'}}
    """
    if not isinstance(dest, (dict, abc.MutableMapping)):
        raise TypeError(f"dest must be dict or MutableMapping, got {fmt_type(dest)}")

    # Handle key parameter - string or sequence
    if isinstance(key, str):
        if not key.strip():
            raise ValueError("key string cannot be empty")
        keys = key.split(separator)
    elif isinstance(key, Sequence) and not isinstance(key, (str, bytes)):
        keys = list(key)
        if not keys:
            raise ValueError("key sequence cannot be empty")
    else:
        raise TypeError(f"key must be str or sequence, got {fmt_type(key)}")

    # Navigate to the parent of the dest key
    current = dest
    for k in keys[:-1]:
        if k not in current:
            if not create_missing:
                raise KeyError(
                    f"intermediate key '{fmt_any(k)}' not found and create_missing=False"
                )
            current[k] = {}
        elif not isinstance(current[k], (dict, abc.MutableMapping)):
            raise TypeError(f"cannot traverse through non-dict value at key {fmt_any(current[k])}")
        current = current[k]

    # Set the final value
    current[keys[-1]] = value


def get_caller_name(depth: int = 1) -> str:
    """
    Retrieve the function name from the call stack at a specified depth.

    **⚠️ PERFORMANCE WARNING:** This function uses `inspect.stack()`, which is
    computationally expensive. Avoid using in performance-critical code paths,
    tight loops, or frequently-called functions. Use only for debugging and
    logging contexts where the overhead is acceptable.

    Args:
        depth: Stack depth to inspect. `1` returns the immediate caller,
            `2` returns the caller's caller, etc. Must be ≥ 1.

    Returns:
        The qualified name of the function at the specified stack depth.

    Raises:
        ValueError: If depth < 1.
        TypeError: If depth is not an integer.
        IndexError: If the call stack is shallower than the requested depth.

    Examples:
        >>> def inner():
        ...     return get_caller_name()
        >>> def outer():
        ...     return inner()
        >>> outer()
        'inner'

        >>> def trace_caller():
        ...     print(f"Called by: {get_caller_name(1)}")
        ...     print(f"Called by (2 levels up): {get_caller_name(2)}")
    """
    if not isinstance(depth, int):
        raise TypeError(f"stack depth must be an integer, but got {fmt_type(depth)}")
    if depth < 1:
        raise ValueError(f"stack depth must be 1 or greater, but got {fmt_value(depth)}")

    # stack()[0] is the frame for get_caller_name itself.
    # stack()[1] corresponds to depth=1 (the immediate caller).
    # So we access the stack at the given depth.
    try:
        # stack() returns a list of FrameInfo objects
        # FrameInfo(frame, filename, lineno, function, code_context, index)
        return stack()[depth][3]
    except IndexError as e:
        raise IndexError(
            f"call stack is not deep enough to access frame at depth {fmt_value(depth)}."
        ) from e


def listify(
    x: object, as_type: type | Callable | None = None, mapping_mode: str = "items"
) -> list[object]:
    """
    Convert input into a list with predictable rules, optionally performing as_type conversion for items.

    Behavior:
    - Atomic treatment for text/bytes:
      - str, bytes, bytearray are NOT expanded character/byte-wise; they become [value].
    - Mappings (dict, etc.):
      - mapping_mode="items": Extract (key, value) tuples (default)
      - mapping_mode="keys": Extract keys only
      - mapping_mode="values": Extract values only
      - mapping_mode="atomic": Treat mapping as single item [mapping]
    - Other iterables:
      - Any other Iterable is expanded into a list of its items.
    - Non-iterables:
      - Wrapped as a single-element list: [x].
    - Conversion:
      - If as_type is provided, it is applied to each resulting item (or the single wrapped x).

    Examples:
    - listify("abc") -> ["abc"]
    - listify([1, 2, "3"]) -> [1, 2, "3"]
    - listify({"a": 1, "b": 2}) -> [("a", 1), ("b", 2)]  # items (default)
    - listify({"a": 1, "b": 2}, mapping_mode="keys") -> ["a", "b"]
    - listify({"a": 1, "b": 2}, mapping_mode="values") -> [1, 2]
    - listify({"a": 1, "b": 2}, mapping_mode="atomic") -> [{"a": 1, "b": 2}]

    Args:
        x: Value to normalize into a list.
        as_type: Optional type or callable used to convert each item.
        mapping_mode: How to handle mappings - "items" (default), "keys", "values", or "atomic"

    Returns:
        List of items, optionally converted.

    Raises:
        ValueError: If conversion via as_type fails for any item or invalid mapping_mode.
    """
    # Handle mappings explicitly
    if isinstance(x, abc.Mapping):
        if mapping_mode == "items":
            items = list(x.items())
        elif mapping_mode == "keys":
            items = list(x.keys())
        elif mapping_mode == "values":
            items = list(x.values())
        elif mapping_mode == "atomic":
            items = [x]
        else:
            raise ValueError(
                f"Invalid mapping_mode: {mapping_mode}. Must be 'items', 'keys', 'values', or 'atomic'"
            )
    # Handle atomic text/bytes
    elif isinstance(x, (str, bytes, bytearray)):
        items = [x]
    # Handle other iterables
    elif isinstance(x, abc.Iterable):
        items = list(x)
    # Handle non-iterables
    else:
        items = [x]

    # Apply conversion if specified
    if as_type is not None:
        try:
            return [as_type(item) for item in items]
        except Exception as e:
            raise ValueError(f"Conversion to {as_type} failed for item in {items}") from e

    return items


def sequence_get(seq: Sequence[T] | None, index: int | None, default: Any = None) -> T | Any:
    """
    Safely get an item from a sequence with default fallback.

    This function provides a robust way to access sequence elements, supporting
    both positive and negative indices (e.g., -1 for the last item).

    Returns the item at the specified index, or the default value if:
    - The sequence is None
    - The index is None
    - The index is out of bounds (negative indices supported)

    Args:
        seq: The sequence to access, or None
        index: The index to retrieve, or None. Supports negative indexing
        default: Value to return when item cannot be accessed

    Returns:
        The item at the specified index, or the default value

    Raises:
        TypeError: If seq is not a Sequence or None, or index is not int or None

    Examples:
        >>> sequence_get([1, 2, 3], 0)  # First element
        1
        >>> sequence_get([1, 2, 3], 1)  # Second element
        2
        >>> sequence_get([1, 2, 3], -1)  # Last element
        3
        >>> sequence_get([1, 2, 3], 5, "missing")  # Out of bounds
        'missing'
        >>> sequence_get([], 0, "empty_seq")  # Empty sequence
        'empty_seq'
    """
    if seq is not None and not isinstance(seq, abc.Sequence):
        raise TypeError(f"expected Sequence or None, got {fmt_type(seq)}")

    if index is not None and not isinstance(index, int):
        raise TypeError(f"expected int or None for index, got {fmt_type(index)}")

    if seq is None or index is None:
        return default

    try:
        return seq[index]
    except IndexError:
        return default
