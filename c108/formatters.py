"""
Robust formatting utilities for development and debugging.

Type-aware formatters for debugging, logging, and exception messages.
All formatters handle broken __repr__, recursive objects,
and edge cases gracefully with consistent styling across ASCII/Unicode output.
The fmt_any() function intelligently dispatches to specialized formatters.
"""

# Standard library -----------------------------------------------------------------------------------------------------

import collections.abc as abc
from collections import abc as abc
from enum import Enum, unique
from itertools import islice
from typing import (
    Any,
    Iterable,
    Iterator,
    Tuple,
    TypeVar,
)

from c108.utils import class_name


# Classes --------------------------------------------------------------------------------------------------------------


# Methods --------------------------------------------------------------------------------------------------------------


# Private Methods ------------------------------------------------------------------------------------------------------
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


def fmt_any(
    obj: Any,
    *,
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
        '[<int: 1>, <int: 2>, <int: 3>]'

        >>> fmt_any(ValueError("bad input"))
        '<ValueError: bad input>'

        >>> fmt_any("simple string")
        "<str: 'simple string'>"

        >>> fmt_any(42)
        '<int: 42>'

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
    exc: Any,
    *,
    style: str = "ascii",
    max_repr: int = 120,
    include_traceback: bool = False,
    ellipsis: str | None = None,
) -> str:
    """Format exceptions with automatic fallback for non-exception types.

    Provides robust formatting of exception objects with type-message pairs,
    optional traceback location info, and consistent styling. Non-exception
    inputs are gracefully handled by delegating to `fmt_value`, making this
    function safe to use in error contexts where object types may be uncertain.

    Args:
        exc: The object to format. Exceptions are formatted as type-message pairs,
            while all other types delegate to `fmt_value`.
        style: Formatting style - "ascii" (<>), "unicode-angle" (⟨⟩), "equal" (=).
        max_repr: Maximum length before truncation (only applies to message, not type).
        include_traceback: Whether to include traceback location info (exceptions only).
        ellipsis: Custom truncation marker (defaults based on style).

    Returns:
        Formatted string. For exceptions: "<ValueError: message>" with type always preserved.
        For non-exceptions: delegated to `fmt_value`.

    Notes:
        - Non-exception types automatically fall back to `fmt_value` (no exceptions raised)
        - Exception type name is NEVER truncated for reliability
        - Only the message portion is subject to max_repr truncation
        - Broken __str__ methods are handled with fallback formatting
        - Traceback info shows function name and line number when requested
        - Handles edge cases: empty messages, chained exceptions, broken __str__

    Examples:
        >>> # Standard exception formatting
        >>> fmt_exception(ValueError("bad input"))
        '<ValueError: bad input>'

        >>> # Empty message
        >>> fmt_exception(RuntimeError())
        '<RuntimeError>'

        >>> # Message truncation (type preserved)
        >>> fmt_exception(ValueError("very long message"), max_repr=21)
        '<ValueError: very...>'

        >>> # Automatic fallback for non-exceptions (no error)
        >>> fmt_exception("not an exception")
        "<str: 'not an exception'>"
        >>> fmt_exception(42)
        '<int: 42>'

    See Also:
        fmt_value: The underlying formatter for non-exception types.
        fmt_any: Main dispatcher that routes exceptions to this function.
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
    mp: Any,
    *,
    style: str = "ascii",
    max_items: int = 8,
    max_repr: int = 120,
    depth: int = 2,
    ellipsis: str | None = None,
) -> str:
    """Format mapping for display with automatic fallback for non-mapping types.

    Formats mapping objects (dicts, OrderedDict, etc.) for debugging or logging.
    Non-mapping inputs are gracefully handled by delegating to `fmt_value`,
    making this function safe to use in error contexts where object types
    may be uncertain.

    Args:
        mp: The object to format. Mappings are formatted as `{key: value}`
            pairs, while all other types delegate to `fmt_value`.
        style: Display style - "ascii" (default), "unicode-angle", "equal", etc.
        max_items: For mappings, the max key-value pairs to show before truncating.
        max_repr: Maximum length of individual key/value reprs before truncation.
        depth: Maximum recursion depth for nested structures within a mapping.
        ellipsis: Custom truncation token. Auto-selected per style if None.

    Returns:
        Formatted string. For mappings: `'{<type: key>: <type: value>...}'`.
        For non-mappings: delegated to `fmt_value`.

    Notes:
        - Non-mapping types automatically fall back to `fmt_value` (no exceptions)
        - Preserves insertion order for modern dicts
        - Keys and values are formatted using `fmt_value`
        - Broken `__repr__` methods are handled gracefully

    Examples:
        >>> # Standard mapping formatting
        >>> fmt_mapping({"name": "Alice", "age": 30})
        "{<str: 'name'>: <str: 'Alice'>, <str: 'age'>: <int: 30>}"

        >>> # Truncation of large mappings
        >>> fmt_mapping({i: i**2 for i in range(10)}, max_items=3)
        '{<int: 0>: <int: 0>, <int: 1>: <int: 1>, <int: 2>: <int: 4>...}'

        >>> # Automatic fallback for non-mappings (no error)
        >>> fmt_mapping("a simple string")
        "<str: 'a simple string'>"
        >>> fmt_mapping(42)
        '<int: 42>'

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
    seq: Iterable[Any],
    *,
    style: str = "ascii",
    max_items: int = 8,
    max_repr: int = 120,
    depth: int = 2,
    ellipsis: str | None = None,
) -> str:
    """Format sequence for display with automatic fallback for non-iterable types.

    Formats iterables (lists, tuples, sets, generators) for debugging or logging.
    Non-iterable inputs and text-like sequences (str, bytes, bytearray) are
    gracefully handled by delegating to `fmt_value`, making this function safe
    to use in error contexts where object types may be uncertain.

    Args:
        seq: The object to format. Iterables are formatted elementwise,
            while non-iterables and text types delegate to `fmt_value`.
        style: Display style - "ascii" (safest, default), "unicode-angle", "equal", "paren", "colon".
        max_items: Maximum elements to show before truncating. Default of 8.
        max_repr: Maximum length of individual element reprs before truncation.
        depth: Maximum recursion depth for nested structures. 0 treats nested objects as atomic.
        ellipsis: Custom truncation token. Auto-selected per style if None.

    Returns:
        Formatted string like "[<int: 1>, <str: 'hello'>...]" (list) or
        "(<int: 1>,)" (singleton tuple). Non-iterables delegated to `fmt_value`.

    Notes:
        - Non-iterable types automatically fall back to `fmt_value` (no exceptions)
        - str/bytes/bytearray are treated as atomic (not decomposed into characters)
        - Preserves container literal syntax: [] for lists, () for tuples, etc.
        - Nested sequences/mappings are recursively formatted up to 'depth' levels
        - Singleton tuples show trailing comma for Python literal accuracy
        - Broken __repr__ methods in elements are handled gracefully

    Examples:
        >>> fmt_sequence([1, "hello", [2, 3]])
        "[<int: 1>, <str: \'hello\'>, [<int: 2>, <int: 3>]]"

        >>> fmt_sequence(range(10), max_items=3)
        '[<int: 0>, <int: 1>, <int: 2>...]'

        >>> # Automatic fallback for text (treated as atomic)
        >>> fmt_sequence("text")
        "<str: 'text'>"

        >>> # Automatic fallback for non-iterables (no error)
        >>> fmt_sequence(42)
        '<int: 42>'

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
    obj: Any,
    *,
    style: str = "ascii",
    max_repr: int = 120,
    ellipsis: str | None = None,
    fully_qualified: bool = False,
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
        fully_qualified: Whether to include module name (e.g., "builtins.int" vs "int").

    Returns:
        Formatted type string like "<type: int>" or "⟨type: MyClass⟩".

    Logic:
        - If obj is a type object → format the type itself
        - If obj is an instance → format type(obj)
        - Module qualification controlled by fully_qualified parameter
        - Graceful handling of broken __name__ attributes

    Examples:
        >>> fmt_type(42)
        '<int>'

        >>> fmt_type(int)
        '<int>'

        >>> fmt_type(ValueError("test"))
        '<ValueError>'

        >>> class CustomClass:
        ...     pass
        >>> fmt_type(CustomClass(), style="unicode-angle")
        '⟨CustomClass⟩'

    Notes:
        - Consistent with other fmt_* functions in style and error handling
        - Type name truncation preserves readability in error contexts
        - Module information helps distinguish between similarly named types
    """
    # get type name with robust edge cases
    type_name = class_name(obj, fully_qualified=fully_qualified, fully_qualified_builtins=False)

    # Apply truncation if needed
    ellipsis_token = _fmt_more_token(style, ellipsis)
    truncated_name = _fmt_truncate(type_name, max_repr, ellipsis=ellipsis_token)

    # Format as a type-no-value string
    return _fmt_format_pair(truncated_name, "", style=style, show_value=False)


def fmt_value(
    obj: Any,
    *,
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
        obj: Any Python object to format.
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

        >>> fmt_value("hello world", max_repr=9)
        "<str: 'hello'...>"

        >>> fmt_value([1, 2, 3], style="unicode-angle")
        '⟨list: [1, 2, 3]⟩'

        >>> fmt_value({1: 111, 2: 222, 3: 333}, style="unicode-angle")
        '⟨dict: {1: 111, 2: 222, 3: 333}⟩'

    See Also:
        fmt_sequence: Format sequences/iterables elementwise with nesting support.
        fmt_mapping: Format mappings with key-value pairs and nesting support.
    """
    t = type(obj).__name__
    ellipsis_token = _fmt_more_token(style, ellipsis)

    # Defensive repr() call - handle broken __repr__ methods gracefully
    try:
        base_repr = repr(obj)
    except Exception as e:
        # Fallback for broken __repr__: show type and exception info
        exc_type = type(e).__name__
        base_repr = f"<{t} object (repr failed: {exc_type})>"

    # Apply ASCII escaping BEFORE truncation so custom ellipsis isn't escaped
    if style == "ascii":
        base_repr = base_repr.replace(">", "\\>")

    r = _fmt_truncate(base_repr, max_repr, ellipsis=ellipsis_token)
    return _fmt_format_pair(t, r, style=style, show_value=True)


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
        inner = s[1 : 1 + inner_budget]
        # Always place ellipsis outside quotes for consistency
        return f"{s[0]}{inner}{s[0]}{ellipsis}"

    # General case: keep at least one character, then append ellipsis
    keep = max(1, max_len)
    return s[:keep] + ellipsis


def _fmt_format_pair(
    type_name: str,
    value_repr: str,
    *,
    style: str = FmtStyle.ASCII,
    show_value: bool = True,  # TODO for <type>
) -> str:
    """Combine a type name and a repr into a single display token according to style."""
    if style == FmtStyle.ASCII:
        return f"<{type_name}{f': {value_repr}' if show_value else ''}>"
    if style == FmtStyle.UNICODE_ANGLE:
        return f"⟨{type_name}{f': {value_repr}' if show_value else ''}⟩"
    if style == FmtStyle.EQUAL:
        return f"{type_name}{f'={value_repr}' if show_value else ''}"
    if style == FmtStyle.PAREN:
        return f"{type_name}" + (f"({value_repr})" if show_value else "")
    if style == FmtStyle.COLON:
        return f"{type_name}{f': {value_repr}' if show_value else ''}"
    else:
        # Gracefully fallback to ASCII if invalid style provided by user
        return f"<{type_name}{f': {value_repr}' if show_value else ''}>"


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
