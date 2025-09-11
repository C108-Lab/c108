#
# C108 Tools & Utilites
#

# Standard library -----------------------------------------------------------------------------------------------------

import collections.abc as abc
from enum import Enum, unique
from inspect import stack
from itertools import islice
from typing import Any, Iterable, Iterator, Mapping, Sequence, Tuple, Callable, overload, TypeVar


# Local ----------------------------------------------------------------------------------------------------------------


# Classes --------------------------------------------------------------------------------------------------------------

@unique
class FmtStyle(str, Enum):
    """
    Enumerates all supported fmt_* formatter styles as unique string values.

    Members are str subclasses, so they can be used anywhere a plain style
    string is expected (e.g., passing directly to your formatting functions).
    """
    ASCII = "ascii"
    UNICODE_ANGLE = "unicode-angle"
    EQUAL = "equal"
    PAREN = "paren"
    COLON = "colon"


T = TypeVar('T')


# Methods --------------------------------------------------------------------------------------------------------------


@overload
def as_ascii(s: str, replacement: str = '_') -> str: ...


@overload
def as_ascii(s: bytes, replacement: bytes = b'_') -> bytes: ...


@overload
def as_ascii(s: bytearray, replacement: bytes = b'_') -> bytearray: ...


def as_ascii(s: str | bytes | bytearray, replacement: str | bytes | None = None) -> str | bytes | bytearray:
    """Convert a string-like object to ASCII by replacing non-ASCII characters and preserving object type.

    This function processes each character/byte in the input and replaces any
    non-ASCII value (code point or byte value >= 128) with the specified
    replacement. The return type matches the input type.

    Args:
        s: The input str, bytes, or bytearray to sanitize.
        replacement: The character or byte to use for replacement.
                     Defaults to '_' for str and b'_' for bytes/bytearray.
                     Must be a single ASCII character/byte.

    Returns:
        A new object of the same type as the input (str, bytes, or bytearray)
        containing only ASCII characters/bytes.

    Raises:
        TypeError: If the input `s` is not a str, bytes, or bytearray, or if
                   `replacement` has an incompatible type.
        ValueError: If `replacement` is not a single ASCII character/byte.

    Examples:
        >>> # Process a standard string
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
            replacement = '_'
        if not isinstance(replacement, str):
            raise TypeError(f"replacement for str input must be str, not {fmt_type(replacement)}")
        if len(replacement) != 1:
            raise ValueError("replacement must be a single character")
        if ord(replacement) >= 128:
            raise ValueError("replacement character must be ASCII")

        return ''.join(replacement if ord(char) >= 128 else char for char in s)

    elif isinstance(s, (bytes, bytearray)):
        # Handle bytes and bytearray input
        if replacement is None:
            replacement = b'_'
        if not isinstance(replacement, bytes):
            raise TypeError(f"replacement for bytes input must be bytes, not {fmt_type(replacement)}")
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


def fmt_any(
        obj: Any, *,
        style: str = "ascii",
        max_items: int = 8,
        max_repr: int = 120,
        depth: int = 2,
        include_traceback: bool = False,
        ellipsis: str | None = None,
) -> str:
    """Format any object for debugging, logging, and exception messages.

    Main entry point for formatting arbitrary Python objects with robust handling
    of edge cases like broken __repr__, recursive objects, and chained exceptions.
    Intelligently routes to specialized formatters based on object type while
    maintaining consistent styling and graceful error handling.

    Args:
        obj: Any Python object to format.
        style: Display style - "ascii" (default), "unicode-angle", "equal", etc.
        max_items: For collections, max items to show before truncating.
        max_repr: Maximum length of individual reprs before truncation.
        depth: Maximum recursion depth for nested structures.
        include_traceback: For exceptions, whether to include location info.
        ellipsis: Custom truncation token. Auto-selected per style if None.

    Returns:
        Formatted string with appropriate structure for the object type.

    Dispatch Logic:
        - BaseException → fmt_exception() (with optional traceback)
        - Mapping → fmt_mapping() (dicts, OrderedDict, etc.)
        - Sequence (non-text) → fmt_sequence() (lists, tuples, etc.)
        - All others → fmt_value() (atomic values, custom objects)

    Examples:
        >>> fmt_any({"key": "value"})
        "{<str: 'key'>: <str: 'value'>}"
        >>> fmt_any([1, 2, 3])
        "[<int: 1>, <int: 2>, <int: 3>]"
        >>> fmt_any(ValueError("bad input"))
        "<ValueError: bad input>"
        >>> fmt_any("simple string")
        "<str: 'simple string'>"
        >>> fmt_any(42)
        "<int: 42>"

    Notes:
        - Text-like sequences (str, bytes) are treated as atomic values
        - Safe for unknown object types in error handling contexts
        - Preserves specialized behavior of each formatter

    See Also:
        fmt_exception, fmt_mapping, fmt_sequence, fmt_value: Specialized formatters
    """
    # Priority 1: Exceptions get special handling
    if isinstance(obj, BaseException):
        return fmt_exception(
            obj,
            style=style,
            max_repr=max_repr,
            include_traceback=include_traceback,
            ellipsis=ellipsis,
        )

    # Priority 2: Mappings (dict, OrderedDict, etc.)
    if isinstance(obj, abc.Mapping):
        return fmt_mapping(
            obj,
            style=style,
            max_items=max_items,
            max_repr=max_repr,
            depth=depth,
            ellipsis=ellipsis,
        )

    # Priority 3: Sequences (but not text-like ones)
    if isinstance(obj, abc.Sequence) and not _fmt_is_textual(obj):
        return fmt_sequence(
            obj,
            style=style,
            max_items=max_items,
            max_repr=max_repr,
            depth=depth,
            ellipsis=ellipsis,
        )

    # Priority 4: Everything else (atomic values, text, custom objects)
    return fmt_value(
        obj,
        style=style,
        max_repr=max_repr,
        ellipsis=ellipsis,
    )


def fmt_exception(
        exc: BaseException, *,
        style: str = "ascii",
        max_repr: int = 120,
        include_traceback: bool = False,
        ellipsis: str | None = None,
) -> str:
    """Format exceptions with optional traceback context for debugging and logging.

    Provides robust formatting of exception objects with type-message pairs,
    optional traceback location info, and consistent styling. Handles edge cases
    like empty messages, broken __str__ methods, and chained exceptions gracefully.

    Args:
        exc: Exception instance to format.
        style: Formatting style - "ascii" (<>), "unicode-angle" (⟨⟩), "equal" (=).
        max_repr: Maximum length before truncation (only applies to message, not type).
        include_traceback: Whether to include traceback location info.
        ellipsis: Custom truncation marker (defaults based on style).

    Returns:
        Formatted exception string like "<ValueError: message>" with type always preserved.

    Examples:
        >>> fmt_exception(ValueError("bad input"))
        '<ValueError: bad input>'
        >>> fmt_exception(RuntimeError())
        '<RuntimeError>'
        >>> fmt_exception(ValueError("very long message"), max_repr=20)
        '<ValueError: very...>'

    Notes:
        - Exception type name is NEVER truncated for reliability
        - Only the message portion is subject to max_repr truncation
        - Broken __str__ methods are handled with fallback formatting
        - Traceback info shows function name and line number when requested
    """

    # Check if it's BaseException - if not, delegate to fmt_value
    if not isinstance(exc, BaseException):
        return fmt_value(exc, style=style, max_repr=max_repr, ellipsis=ellipsis)

    # Get exception type name
    exc_type = type(exc).__name__

    # Get exception message safely
    try:
        exc_msg = str(exc)
    except Exception:
        exc_msg = "<repr failed>"

    # Determine ellipsis based on style
    if ellipsis is None:
        ellipsis = "…" if style == "unicode-angle" else "..."

    # Build base format based on style - TYPE NAME IS NEVER TRUNCATED
    if exc_msg:
        if style == "equal":
            # Calculate space available for message
            base_length = len(exc_type) + 1  # "ValueError="
            if max_repr > 0 and base_length + len(exc_msg) > max_repr:
                available_for_msg = max_repr - base_length - len(ellipsis)
                if available_for_msg > 0:
                    truncated_msg = exc_msg[:available_for_msg]
                    base_format = f"{exc_type}={truncated_msg}{ellipsis}"
                else:
                    base_format = f"{exc_type}={ellipsis}"
            else:
                base_format = f"{exc_type}={exc_msg}"
        elif style == "unicode-angle":
            # Calculate space available for message
            base_length = len(exc_type) + 4  # "⟨ValueError: ⟩"
            if max_repr > 0 and base_length + len(exc_msg) > max_repr:
                available_for_msg = max_repr - base_length - len(ellipsis)
                if available_for_msg > 0:
                    truncated_msg = exc_msg[:available_for_msg]
                    base_format = f"⟨{exc_type}: {truncated_msg}{ellipsis}⟩"
                else:
                    base_format = f"⟨{exc_type}: {ellipsis}⟩"
            else:
                base_format = f"⟨{exc_type}: {exc_msg}⟩"
        else:  # ascii
            # Calculate space available for message
            base_length = len(exc_type) + 4  # "<ValueError: >"
            if max_repr > 0 and base_length + len(exc_msg) > max_repr:
                available_for_msg = max_repr - base_length - len(ellipsis)
                if available_for_msg > 0:
                    truncated_msg = exc_msg[:available_for_msg]
                    base_format = f"<{exc_type}: {truncated_msg}{ellipsis}>"
                else:
                    base_format = f"<{exc_type}: {ellipsis}>"
            else:
                base_format = f"<{exc_type}: {exc_msg}>"
    else:
        # No message - just show type
        if style == "equal":
            base_format = exc_type
        elif style == "unicode-angle":
            base_format = f"⟨{exc_type}⟩"
        else:  # ascii
            base_format = f"<{exc_type}>"

    # Add traceback location if requested
    if include_traceback:
        try:
            tb = exc.__traceback__
            if tb:
                # Get the last frame from the traceback
                while tb.tb_next:
                    tb = tb.tb_next
                frame = tb.tb_frame
                filename = frame.f_code.co_filename
                function_name = frame.f_code.co_name
                line_number = tb.tb_lineno

                # Extract module name from filename
                import os
                module_name = os.path.splitext(os.path.basename(filename))[0]

                location = f" at {module_name}.{function_name}:{line_number}"

                # For equal style, append differently
                if style == "equal":
                    base_format = f"{base_format}{location}"
                else:
                    # Insert before closing bracket/angle
                    base_format = base_format[:-1] + location + base_format[-1]
        except Exception:
            # If traceback extraction fails, continue without it
            pass

    return base_format


def fmt_mapping(
        mp: Any, *,
        style: str = "ascii",
        max_items: int = 8,
        max_repr: int = 120,
        depth: int = 2,
        ellipsis: str | None = None,
) -> str:
    """Robustly format mappings and other Python objects for display.

    This function formats any Python object for debugging or logging, providing
    detailed, truncated formatting for mappings (like dicts) and delegating
    to `fmt_value` for all other types. This makes it a safe, all-purpose
    formatter for potentially unknown data structures in error contexts.

    Args:
        mp: The object to format. Mappings are formatted as `{key: value}`
            pairs, while all other types are passed to `fmt_value`.
        style: Display style - "ascii" (default), "unicode-angle", "equal", etc.
        max_items: For mappings, the max key-value pairs to show before truncating.
        max_repr: Maximum length of individual key/value reprs before truncation.
        depth: Maximum recursion depth for nested structures within a mapping.
        ellipsis: Custom truncation token. Auto-selected per style if None.

    Returns:
        A formatted string representation of the object. For mappings, the
        format is `'{<type: key_repr>: <type: value_repr>...}'`. For other
        types, the format is determined by `fmt_value`.

    Notes:
        - Non-mapping types are formatted directly by `fmt_value`.
        - Preserves insertion order for modern dicts.
        - Keys and values within a mapping are formatted using `fmt_value`.
        - Broken `__repr__` methods in keys or values are handled gracefully.

    Examples:
        >>> # Standard mapping formatting
        >>> fmt_mapping({"name": "Alice", "age": 30})
        "{<str: 'name'>: <str: 'Alice'>, <str: 'age'>: <int: 30>}"

        >>> # Truncation of a large mapping
        >>> fmt_mapping({i: i**2 for i in range(10)}, max_items=3)
        "{<int: 0>: <int: 0>, <int: 1>: <int: 1>, <int: 2>: <int: 4>...}"

        >>> # Graceful handling of non-mapping types
        >>> fmt_mapping("a simple string")
        "<str: 'a simple string'>"
        >>> import datetime
        >>> fmt_mapping(datetime.date(2025, 9, 4))
        "<date: 2025-09-04>"

    See Also:
        fmt_value: The underlying formatter for individual values and non-mappings.
        fmt_sequence: Formats sequences/iterables with similar robustness.
    """
    # Check if it's actually a mapping - if not, delegate to fmt_value
    if not isinstance(mp, abc.Mapping):
        return fmt_value(mp, style=style, max_repr=max_repr, ellipsis=ellipsis)

    # Support mappings without reliable len by sampling
    items_iter: Iterator[Tuple[Any, Any]] = iter(mp.items())
    sampled = list(islice(items_iter, max_items + 1))
    had_more = len(sampled) > max_items
    if had_more:
        sampled = sampled[:max_items]

    parts: list[str] = []
    for k, v in sampled:
        k_str = fmt_value(k, style=style, max_repr=max_repr, ellipsis=ellipsis)
        if depth > 0 and not _fmt_is_textual(v):
            if isinstance(v, abc.Mapping):
                v_str = fmt_mapping(
                    v,
                    style=style,
                    max_items=max_items,
                    max_repr=max_repr,
                    depth=depth - 1,
                    ellipsis=ellipsis,
                )
            elif isinstance(v, abc.Sequence):
                v_str = fmt_sequence(
                    v,
                    style=style,
                    max_items=max_items,
                    max_repr=max_repr,
                    depth=depth - 1,
                    ellipsis=ellipsis,
                )
            else:
                v_str = fmt_value(v, style=style, max_repr=max_repr, ellipsis=ellipsis)
        else:
            v_str = fmt_value(v, style=style, max_repr=max_repr, ellipsis=ellipsis)
        parts.append(f"{k_str}: {v_str}")

    more = _fmt_more_token(style, ellipsis) if had_more else ""
    return "{" + ", ".join(parts) + more + "}"


def fmt_sequence(
        seq: Iterable[Any], *,
        style: str = "ascii",
        max_items: int = 8,
        max_repr: int = 120,
        depth: int = 2,
        ellipsis: str | None = None,
) -> str:
    """
    Format a sequence elementwise for debugging, logging, and exception messages.

    Intended for robust display of lists, tuples, and other iterables in error contexts.
    Preserves literal shape ([] for lists, () for tuples) while handling problematic
    elements, deep nesting, and large sequences gracefully with configurable limits.

    Args:
        seq: Any iterable (list, tuple, set, generator, etc.) to format.
        style: Display style - "ascii" (safest, default), "unicode-angle", "equal", "paren", "colon".
        max_items: Maximum elements to show before truncating. Conservative default of 6.
        max_repr: Maximum length of individual element reprs before truncation.
        depth: Maximum recursion depth for nested structures. 0 treats nested objects as atomic.
        ellipsis: Custom truncation token. Auto-selected per style if None.

    Returns:
        Formatted string like "[<int: 1>, <str: 'hello'>...]" (list) or "(<int: 1>,)" (singleton tuple).

    Notes:
        - Preserves container literal syntax: [] for lists, () for tuples, etc.
        - str/bytes/bytearray are treated as atomic (not decomposed into characters).
        - Nested sequences/mappings are recursively formatted up to 'depth' levels.
        - Singleton tuples show trailing comma for Python literal accuracy.
        - Broken __repr__ methods in elements are handled gracefully.
        - Non-iterable inputs fall back to fmt_value for atomic formatting.
        - Conservative defaults prevent overwhelming exception messages.

    Examples:
        >>> fmt_sequence([1, "hello", [2, 3]])
        '[<int: 1>, <str: \'hello\'>, [<int: 2>, <int: 3>]]'
        >>> fmt_sequence(range(10), max_items=3)
        '[<int: 0>, <int: 1>, <int: 2>...]'
        >>> fmt_sequence("text")  # Strings are atomic
        '<str: \'text\'>'

    See Also:
        fmt_value: Format individual elements with the same robustness guarantees.
        fmt_mapping: Format mappings with similar nesting support.
    """
    # Check if it's Iterable - if not, delegate to fmt_value
    if not isinstance(seq, abc.Iterable):
        return fmt_value(seq, style=style, max_repr=max_repr, ellipsis=ellipsis)

    if _fmt_is_textual(seq):
        # Treat text-like as a scalar value, not a sequence of characters
        return fmt_value(seq, style=style, max_repr=max_repr, ellipsis=ellipsis)

    # Choose delimiters by common concrete types; fallback to []
    open_ch, close_ch = "[", "]"
    is_tuple = isinstance(seq, tuple)
    if is_tuple:
        open_ch, close_ch = "(", ")"
    elif isinstance(seq, list):
        open_ch, close_ch = "[", "]"
    elif isinstance(seq, set):
        open_ch, close_ch = "{", "}"  # Though sets are unordered
    elif isinstance(seq, frozenset):
        # frozenset displays differently but we'll use {} for consistency
        open_ch, close_ch = "{", "}"

    items, had_more = _fmt_head(seq, max_items)

    parts: list[str] = []
    for x in items:
        # Recurse into nested structures one level at a time
        if depth > 0 and not _fmt_is_textual(x):
            if isinstance(x, abc.Mapping):
                parts.append(
                    fmt_mapping(
                        x,
                        style=style,
                        max_items=max_items,
                        max_repr=max_repr,
                        depth=depth - 1,
                        ellipsis=ellipsis,
                    )
                )
                continue
            if isinstance(x, abc.Sequence):
                parts.append(
                    fmt_sequence(
                        x,
                        style=style,
                        max_items=max_items,
                        max_repr=max_repr,
                        depth=depth - 1,
                        ellipsis=ellipsis,
                    )
                )
                continue
        parts.append(fmt_value(x, style=style, max_repr=max_repr, ellipsis=ellipsis))

    more = _fmt_more_token(style, ellipsis) if had_more else ""
    # Singleton tuple needs a trailing comma for Python literal accuracy
    tail = "," if is_tuple and len(parts) == 1 and not more else ""
    return f"{open_ch}" + ", ".join(parts) + more + f"{tail}{close_ch}"


def fmt_type(
        obj: Any, *,
        style: str = "ascii",
        max_repr: int = 120,
        ellipsis: str | None = None,
        show_module: bool = False,
) -> str:
    """Format type information for debugging, logging, and exception messages.

    Provides consistent formatting of type information for both type objects and
    instances. Complements the other fmt_* functions by focusing specifically on
    type display with optional module qualification and the same robust error handling.

    Args:
        obj: Any Python object or type to extract type information from.
        style: Display style - "ascii" (default), "unicode-angle", "equal", etc.
        max_repr: Maximum length before truncation (applies to full type name).
        ellipsis: Custom truncation token. Auto-selected per style if None.
        show_module: Whether to include module name (e.g., "builtins.int" vs "int").

    Returns:
        Formatted type string like "<type: int>" or "⟨type: MyClass⟩".

    Logic:
        - If obj is a type object → format the type itself
        - If obj is an instance → format type(obj)
        - Module qualification controlled by show_module parameter
        - Graceful handling of broken __name__ attributes

    Examples:
        >>> fmt_type(42)
        '<type: int>'
        >>> fmt_type(int)
        '<type: int>'
        >>> fmt_type(ValueError("test"))
        '<type: ValueError>'
        >>> fmt_type([], show_module=True)
        '<type: builtins.list>'
        >>> fmt_type(CustomClass(), style="unicode-angle")
        '⟨type: CustomClass⟩'

    Notes:
        - Consistent with other fmt_* functions in style and error handling
        - Type name truncation preserves readability in error contexts
        - Module information helps distinguish between similarly named types
    """
    # Determine the type to format
    if isinstance(obj, type):
        # obj is already a type
        target_type = obj
    else:
        # obj is an instance, get its type
        target_type = type(obj)

    # Get type name safely
    try:
        type_name = target_type.__name__
    except AttributeError:
        # Fallback for objects without __name__
        type_name = str(target_type)

    # Add module qualification if requested
    if show_module:
        try:
            module_name = target_type.__module__
            if module_name and module_name != 'builtins':
                type_name = f"{module_name}.{type_name}"
        except AttributeError:
            # If __module__ is missing, continue with just the type name
            pass

    # Apply truncation if needed
    ellipsis_token = _fmt_more_token(style, ellipsis)
    truncated_name = _fmt_truncate(type_name, max_repr, ellipsis=ellipsis_token)

    # Format as a type-value pair using existing infrastructure
    return _fmt_format_pair("type", truncated_name, style)


def fmt_value(
        x: Any, *,
        style: str = "ascii",
        max_repr: int = 120,
        ellipsis: str | None = None,
) -> str:
    """
    Format a single value as a type–value pair for debugging, logging, and exception messages.

    Intended for robust display of arbitrary values in error contexts where safety and
    readability matter more than perfect fidelity. Handles edge cases like broken __repr__,
    recursive objects, and extremely long representations gracefully.

    Args:
        x: Any Python object to format.
        style: Display style - "ascii" (safest, default), "unicode-angle", "equal", "paren", "colon".
        max_repr: Maximum length of the value's repr before truncation. Generous default of 120.
        ellipsis: Custom truncation token. Auto-selected per style if None ("..." for ASCII, "…" for Unicode).

    Returns:
        Formatted string like "<int: 42>" (ASCII) or "⟨str: 'hello'⟩" (Unicode-angle).

    Notes:
        - For quoted reprs (strings), ellipsis is placed outside quotes for clarity.
        - ASCII style escapes inner ">" to avoid conflicts with wrapper brackets.
        - Broken __repr__ methods are handled gracefully with fallback formatting.
        - Designed for exception messages and logs where robustness trumps perfect formatting.

    Examples:
        >>> fmt_value(42)
        '<int: 42>'
        >>> fmt_value("hello world", max_repr=8)
        "<str: 'hello'...>"
        >>> fmt_value([1, 2, 3], style="unicode-angle")
        '⟨list: [1, 2, 3]⟩'

    See Also:
        fmt_sequence: Format sequences/iterables elementwise with nesting support.
        fmt_mapping: Format mappings with key-value pairs and nesting support.
    """
    t = type(x).__name__
    ellipsis_token = _fmt_more_token(style, ellipsis)

    # Defensive repr() call - handle broken __repr__ methods gracefully
    try:
        base_repr = repr(x)
    except Exception as e:
        # Fallback for broken __repr__: show type and exception info
        exc_type = type(e).__name__
        base_repr = f"<{t} object (repr failed: {exc_type})>"

    # Apply ASCII escaping BEFORE truncation so custom ellipsis isn't escaped
    if style == "ascii":
        base_repr = base_repr.replace(">", "\\>")

    r = _fmt_truncate(base_repr, max_repr, ellipsis=ellipsis_token)
    return _fmt_format_pair(t, r, style)


def dict_get(source: dict | Mapping,
             key: str | Sequence[str],
             default: Any = None,
             *,
             separator: str = ".") -> Any:
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
        if not key.strip():
            raise ValueError("key string cannot be empty")
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


def dict_set(dest: dict | abc.MutableMapping,
             key: str | Sequence[str],
             value: Any,
             *,
             separator: str = '.',
             create_missing: bool = True) -> None:
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
                raise KeyError(f"intermediate key '{fmt_any(k)}' not found and create_missing=False")
            current[k] = {}
        elif not isinstance(current[k], (dict, abc.MutableMapping)):
            raise TypeError(f"cannot traverse through non-dict value at key {fmt_any(current[k])}")
        current = current[k]

    # Set the final value
    current[keys[-1]] = value


def get_caller_name(depth: int = 1) -> str:
    """Gets the name of a function from the call stack.

    This function inspects the call stack to retrieve the name of a function
    at a specified depth. It is primarily useful for debugging, logging, and
    creating context-aware messages. By default, it returns the name of the
    immediate caller.

    Args:
        depth (int): The desired depth in the call stack. `1` refers to the
            immediate caller, `2` to the caller's caller, and so on.
            Defaults to 1.

    Returns:
        str: The name of the function at the specified stack depth.

    Raises:
        IndexError: If the call stack is not deep enough for the given depth.
        TypeError: If `depth` is not an integer.
        ValueError: If `depth` is less than 1.

    Warning:
        This function depends on `inspect.stack()`, which can be
        computationally expensive. Its use in performance-sensitive code or
        frequent calls (like in tight loops) is discouraged.

    Examples:
        def low_level_func():
            # Gets the name of the function that called it.
            caller_name = get_caller_name()
            print(f"low_level_func was called by: {caller_name}")

        def mid_level_func():
            low_level_func()

        # Calling mid_level_func() will print:
        # "low_level_func was called by: mid_level_func"
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
        raise IndexError(f"call stack is not deep enough to access frame at depth {fmt_value(depth)}.") from e


def list_get(lst: list[T] | None, index: int | None, default: Any = None) -> T | Any:
    """
    Safely get an item from a list with default fallback.

    This method supports both positive and negative indices (e.g., -1 for the last item).

    Returns the item at the specified index, or the default value if:
    - The list is None
    - The index is None
    - The index is out of bounds (negative indices supported)

    Args:
        lst: The list to access, or None
        index: The index to retrieve, or None. Supports negative indexing
        default: Value to return when item cannot be accessed

    Returns:
        The item at the specified index, or the default value

    Raises:
        TypeError: If lst is not a list or None, or index is not int or None

    Examples:
        >>> list_get([1, 2, 3], 0)  # First element
        1
        >>> list_get([1, 2, 3], 1)  # Second element
        2
        >>> list_get([1, 2, 3], -1)  # Last element
        3
        >>> list_get([1, 2, 3], 5, "missing")  # Out of bounds
        'missing'
        >>> list_get([], 0, "empty_list")  # Empty list
        'empty_list'

    Notes:
        - Delegates to sequence_get for implementation consistency
        - Supports all standard list indexing including negative indices
        - Type-safe with generic return type matching list element type
    """
    if lst is not None and not isinstance(lst, list):
        raise TypeError(f"expected list or None, got {fmt_type(lst)}")
    if index is not None and not isinstance(index, int):
        raise TypeError(f"expected int or None for index, got {fmt_type(index)}")

    return sequence_get(lst, index, default=default)


def listify(x: object, as_type: type | Callable | None = None,
            mapping_mode: str = "items") -> list[object]:
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
    from collections.abc import Mapping, Iterable

    # Handle mappings explicitly
    if isinstance(x, Mapping):
        if mapping_mode == "items":
            items = list(x.items())
        elif mapping_mode == "keys":
            items = list(x.keys())
        elif mapping_mode == "values":
            items = list(x.values())
        elif mapping_mode == "atomic":
            items = [x]
        else:
            raise ValueError(f"Invalid mapping_mode: {mapping_mode}. Must be 'items', 'keys', 'values', or 'atomic'")
    # Handle atomic text/bytes
    elif isinstance(x, (str, bytes, bytearray)):
        items = [x]
    # Handle other iterables
    elif isinstance(x, Iterable):
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


def listify_OLD(x: object, as_type: type | Callable | None = None) -> list[object]:
    """
    Convert input into a list with predictable rules, optionally performing as_type conversion for items.

    Behavior:
    - Atomic treatment for text/bytes:
      - str, bytes, bytearray are NOT expanded character/byte-wise; they become [value].
    - Iterables:
      - Any other Iterable is expanded into a list of its items (note: dict iterates over keys).
    - Non-iterables:
      - Wrapped as a single-element list: [x].
    - Conversion:
      - If as_type is provided, it is applied to each resulting item (or the single wrapped x).
      - as_type may be a type (e.g., int) or any callable (e.g., a function, lambda, or functools.partial).
      - If conversion fails for any element, ValueError is raised with context and the original exception as the cause.

    Examples:
    - listify("abc") -> ["abc"]
    - listify([1, 2, "3"]) -> [1, 2, "3"]
    - listify((1, "2"), as_type=str) -> ["1", "2"]
    - listify(b"bytes") -> [b"bytes"]
    - listify({"a": 1}) -> ["a"]  # dict iterates over keys

    Args:
        x: Value to normalize into a list.
        as_type: Optional type or callable used to convert each item.

    Returns:
        List of items, optionally converted.

    Raises:
        ValueError: If conversion via as_type fails for any item.
    """

    def _convert(v: object) -> object:
        if as_type is None:
            return v
        try:
            return as_type(v)  # type: ignore[misc,call-arg]
        except Exception as e:
            raise ValueError(f"failed to convert value {fmt_any(v)} using provided as_type") from e

    if isinstance(x, (str, bytes, bytearray)):
        return [_convert(x)]

    if isinstance(x, abc.Iterable):
        return [_convert(e) for e in x]

    return [_convert(x)]


def print_title(title,
                prefix: str = "------- ",
                suffix: str = " -------",
                start: str = "\n",
                end: str = "\n"):
    """
    Prints a formatted title to the console.

    Args:
        title (str): The main title string to be printed.
        prefix (str, optional): A string to prepend to the title. Defaults to "------- ".
        suffix (str, optional): A string to append to the title. Defaults to " -------".
        start (str, optional): A string to print before the entire formatted title. Defaults to "\n".
        end (str, optional): A string to print after the entire formatted title. Defaults to "\n".
    """
    print(f"{start}{prefix}{title}{suffix}{end}", end="")


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


# Private Methods ------------------------------------------------------------------------------------------------------

def _fmt_truncate(s: str, max_len: int, ellipsis: str = "…") -> str:
    """
    Truncate s to at most max_len visible characters before appending the ellipsis.

    Behavior:
    - If s fits within max_len, return s unchanged.
    - If s looks like a quoted repr (starts/ends with the same ' or "), preserve the quotes
      and place the ellipsis outside the closing quote.
    - For unquoted strings, keep at least one character, then append the full ellipsis.

    Note: The ellipsis is appended in full (not counted against max_len), to match display needs.
    """
    if max_len <= 0:
        return ""
    if len(s) <= max_len:
        return s

    # Quoted repr (e.g., "'abc'" or '"abc"')
    if len(s) >= 2 and s[0] in ("'", '"') and s[-1] == s[0]:
        # Conservative inner budget: reserve space for quotes and punctuation;
        # ensure at least 1 inner char remains.
        inner_budget = max(1, max_len - 4)
        inner = s[1:1 + inner_budget]
        # Always place ellipsis outside quotes for consistency
        return f"{s[0]}{inner}{s[0]}{ellipsis}"

    # General case: keep at least one character, then append ellipsis
    keep = max(1, max_len)
    return s[:keep] + ellipsis


def _fmt_format_pair(type_name: str, value_repr: str, style: str) -> str:
    """Combine a type name and a repr into a single display token according to style."""
    if style == FmtStyle.ASCII:
        # ASCII: angle-bracket wrapper, inner value already escaped if needed
        return f"<{type_name}: {value_repr}>"
    if style == FmtStyle.UNICODE_ANGLE:
        return f"⟨{type_name}: {value_repr}⟩"
    if style == FmtStyle.EQUAL:
        return f"{type_name}={value_repr}"
    if style == FmtStyle.PAREN:
        return f"{type_name}({value_repr})"
    if style == FmtStyle.COLON:
        return f"{type_name}: {value_repr}"
    # default: ascii-like without escaping
    return f"<{type_name}: {value_repr}>"


def _fmt_more_token(style: str, more_token: str | None = None) -> str:
    """Decide which 'more' token to use (ellipsis vs custom)."""
    if more_token is not None:
        return more_token
    return "..." if style == "ascii" else "…"


def _fmt_is_textual(x: Any) -> bool:
    return isinstance(x, (str, bytes, bytearray))


def _fmt_head(iterable: Iterable[Any], n: int) -> Tuple[list[Any], bool]:
    """Take up to n items and indicate whether there were more items."""
    it = iter(iterable)
    buf = list(islice(it, n + 1))
    if len(buf) <= n:
        return buf, False
    return buf[:n], True
