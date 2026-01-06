"""
Robust formatting utilities for development and debugging.

Type-aware formatters for debugging, logging, and exception messages.
All formatters handle broken __repr__, recursive objects,
and edge cases gracefully with consistent styling across ASCII/Unicode output.
The fmt_any() function intelligently dispatches to specialized formatters.
"""

# TODO Formatters default to common JSON-friendly representation with equals-style
#      and safe zero depth of recursion

# TODO Clean Signatures
#
# def fmt_any(obj: Any, *, opts: FmtOptions | None = None) -> str:
#     """All formatting controlled via opts"""
#
# def fmt_exception(exc: Any, *, opts: FmtOptions | None = None) -> str:
#     """All formatting controlled via opts"""
#
# def fmt_mapping(mp: Any, *, opts: FmtOptions | None = None) -> str:
#     """All formatting controlled via opts"""
#
# def fmt_sequence(seq: Iterable[Any], *, opts: FmtOptions | None = None) -> str:
#     """All formatting controlled via opts"""
#
# def fmt_set(st: AbstractSet[Any], *, opts: FmtOptions | None = None) -> str:
#     """All formatting controlled via opts"""
#
# def fmt_type(obj: Any, *, opts: FmtOptions | None = None, fully_qualified: bool = False) -> str:
#     """fully_qualified is method-specific, rest via opts"""
#
# def fmt_value(obj: Any, *, opts: FmtOptions | None = None) -> str:
#     """All formatting controlled via opts"""


# Standard library -----------------------------------------------------------------------------------------------------

import collections.abc as abc
import reprlib

from dataclasses import dataclass, field, replace
from itertools import islice
from typing import (
    AbstractSet,
    Any,
    Iterable,
    Iterator,
    Literal,
    Tuple,
)

from pygments.lexer import default

from c108.sentinels import UNSET, ifnotunset
from c108.typing import validate_types
from c108.utils import Self, class_name

PRIMITIVE_TYPES = (
    type(None),
    bool,  # Comes before int (is subclass of int)
    int,
    float,
    complex,
    str,
    bytes,
    type(Ellipsis),  # EllipsisType (...)
    type(NotImplemented),  # NotImplementedType
)

# Classes --------------------------------------------------------------------------------------------------------------

Style = Literal["angle", "arrow", "braces", "colon", "equal", "paren", "repr", "unicode-angle"]


@dataclass(frozen=True)
class FmtOptions:
    """Formatting options for fmt_* functions.

    Controls display style, repr behavior, and formatting preferences.
    Immutable for safe sharing across recursive calls.

    Attributes:
        deduplicate_types: Deduplicate type labels for identical types.
        fully_qualified: Whether to include FQN type names.
        include_traceback: Include exception traceback info.
        label_primitives: Whether to show type labels for int, float, str, bytes, etc.
        repr: reprlib.Repr instance controlling collection formatting and limits.
            Used both for truncation (maxlist, maxdict, maxlevel) and for
            delegating built-in collection formatting when appropriate.
        style: Display style for type-value pairs: "angle" | "colon" | "equal" | "paren" | "repr" | "unicode-angle"

    Note:
        While FmtOptions itself is frozen, the 'repr' field is not. Avoid mutating 'repr' attributes directly.
        Prefer to use merge() or factory methods to create new instances with updated configuration.

    Examples:
        >>> # Use defaults
        >>> opts = FmtOptions()
        >>> fmt_any(data, opts=opts)

        >>> # Custom repr config
        >>> r = reprlib.Repr()
        >>> r.maxdict = 3
        >>> r.maxlevel = 2
        >>> opts = FmtOptions(repr=r, style='unicode-angle')
        >>> fmt_any(data, opts=opts)

        >>> # Create variants with replace()
        >>> debug_opts = opts.replace(label_primitives=True)
    """

    deduplicate_types: bool = False
    fully_qualified: bool = False
    include_traceback: bool = False
    label_primitives: bool = False
    repr: reprlib.Repr = field(
        default_factory=lambda: _fmt_repr(max_items=6, max_depth=6, max_str=120)
    )
    style: Style = "repr"

    def __post_init__(self):
        object.__setattr__(self, "deduplicate_types", bool(self.deduplicate_types))
        object.__setattr__(self, "fully_qualified", bool(self.fully_qualified))
        object.__setattr__(self, "include_traceback", bool(self.include_traceback))
        object.__setattr__(self, "label_primitives", bool(self.label_primitives))

        if not isinstance(self.repr, reprlib.Repr):
            raise TypeError(f"reprlib.Repr expected, but got {type(self.repr).__name__}")

        if self.style not in {
            "angle",
            "arrow",
            "braces",
            "colon",
            "equal",
            "paren",
            "repr",
            "unicode-angle",
        }:
            raise ValueError(f"unknown style: {self.style}")

    def merge(
        self,
        *,
        deduplicate_types: bool = UNSET,
        fully_qualified: bool = UNSET,
        include_traceback: bool = UNSET,
        label_primitives: bool = UNSET,
        max_depth: int = UNSET,
        max_items: int = UNSET,
        max_str: int = UNSET,
        repr: reprlib.Repr = UNSET,
        style: Style = UNSET,
    ) -> Self:
        """
        Create a new FmtOptions instance with selectively updated fields.

        If a parameter value is UNSET, no update is applied to the field.

        Args:
            deduplicate_types: Deduplicate type labels for identical types.
            fully_qualified: Whether to include FQN type names.
            include_traceback: Include exception traceback info.
            label_primitives: Whether to show type labels for int, float, str, bytes, etc.
            max_depth: Maximum depth for nested structures; overrides repr.maxlevel.
            max_items: Maximum number of elements in collections; overrides repr config.
            max_str (int): The maximum length for strings and other representations; overrides repr config.
            repr: reprlib.Repr instance controlling collection formatting and limits.
            style: Display style for type-value pairs: "angle" | "colon" | "equal" | "paren" | "repr" | "unicode-angle"

        Returns:
            New FmtOptions instance with merged configuration
        """
        deduplicate_types = ifnotunset(deduplicate_types, default=self.deduplicate_types)
        fully_qualified = ifnotunset(fully_qualified, default=self.fully_qualified)
        include_traceback = ifnotunset(include_traceback, default=self.include_traceback)
        label_primitives = ifnotunset(label_primitives, default=self.label_primitives)

        if not isinstance(repr, (reprlib.Repr, type(None), type(UNSET))):
            raise ValueError(f"reprlib.Repr or None expected, but got {type(repr).__name__}")

        r = ifnotunset(repr, default=self.repr)
        r = _fmt_repr(max_depth=max_depth, max_items=max_items, max_str=max_str, default=r)

        style = ifnotunset(style, default=self.style)

        return FmtOptions(
            deduplicate_types=deduplicate_types,
            fully_qualified=fully_qualified,
            include_traceback=include_traceback,
            label_primitives=label_primitives,
            repr=r,
            style=style,
        )

    @classmethod
    def compact(cls, max_items: int = 8, max_depth: int = 2) -> Self:
        """Minimal output for tight spaces."""
        r = _fmt_repr(max_items=max_items, max_depth=max_depth, max_str=64)
        return cls(repr=r)

    @classmethod
    def debug(cls, max_items: int = 256, max_depth: int = 5) -> Self:
        """Verbose output for debugging."""
        r = _fmt_repr(max_items=max_items, max_depth=max_depth)
        r.maxstring = r.maxother = 1024
        return cls(repr=r, label_primitives=True, include_traceback=True)

    @classmethod
    def logging(cls, max_items: int = 64, max_depth: int = 3) -> Self:
        """Balanced output for production logging."""
        r = _fmt_repr(max_items=max_items, max_depth=max_depth)
        r.maxstring = r.maxother = 128
        return cls(repr=r)

    @property
    def ellipsis(self) -> int:
        return self.repr.fillvalue

    @property
    def max_depth(self) -> int:
        """Maximum nesting depth."""
        return self.repr.maxlevel

    @property
    def max_items(self) -> int:
        """Maximum items number in repr (uses maxlist as canonical value)."""
        return self.repr.maxlist


# Methods --------------------------------------------------------------------------------------------------------------


@validate_types
def fmt_any(
    obj: Any,
    *,
    opts: FmtOptions | None = None,
    style: Style = "equal",
    max_items: int = 8,
    max_repr: int = 120,
    depth: int = 0,
    ellipsis: str | None = None,
    include_traceback: bool = False,
    label_primitives: bool = False,
) -> str:
    """Format any object for debugging, logging, and exception messages.

    Main entry point for formatting arbitrary Python objects with robust handling
    of edge cases like broken __repr__, recursive objects, and chained exceptions.
    Intelligently routes to specialized formatters based on object type while
    maintaining consistent styling and graceful error handling.

    Args:
        obj: Any Python object to format.
        style: Display style - "angle", "equal", "colon", etc.
        max_items: For collections, max items to show before truncating.
        max_repr: Maximum length of individual reprs before truncation.
        depth: Maximum recursion depth for nested structures.
        include_traceback: For exceptions, whether to include location info.
        label_primitives: Whether to show type labels for int, float, str, bytes, etc.

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

    opts = opts or FmtOptions()

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
            label_primitives=label_primitives,
        )

    # Priority 3: Sets
    if isinstance(obj, abc.Set):
        return fmt_set(
            obj,
            style=style,
            max_items=max_items,
            max_repr=max_repr,
            depth=depth,
            ellipsis=ellipsis,
            label_primitives=label_primitives,
        )

    # Priority 4: Sequences (but not text-like ones)
    if isinstance(obj, abc.Sequence) and not _is_textual(obj):
        return fmt_sequence(
            obj,
            style=style,
            max_items=max_items,
            max_repr=max_repr,
            depth=depth,
            ellipsis=ellipsis,
            label_primitives=label_primitives,
        )

    # Priority 5: Everything else (atomic values, text, custom objects)
    return fmt_value(
        obj,
        style=style,
        max_repr=max_repr,
        ellipsis=ellipsis,
        label_primitives=label_primitives,
    )


@validate_types
def fmt_exception(
    exc: Any,
    *,
    opts: FmtOptions | None = None,
    style: Style = "equal",
    max_repr: int = 120,
    include_traceback: bool = False,
    ellipsis: str | None = None,
) -> str:
    """Format exceptions representation with automatic fallback for non-exception types.

    Provides robust formatting of exception objects with type-message pairs,
    optional traceback location info, and consistent styling. Non-exception
    inputs are gracefully handled by delegating to `fmt_value`, making this
    function safe to use in error contexts where object types may be uncertain.

    Args:
        exc: The object to format. Exceptions are formatted as type-message pairs,
            while all other types delegate to `fmt_value`.
        style: Display style - "angle", "equal", "colon", etc.
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

    opts = opts or FmtOptions()

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


@validate_types
def fmt_mapping(
    mp: Any,
    *,
    opts: FmtOptions | None = None,
    style: Style = "equal",
    max_items: int = 8,
    max_repr: int = 120,
    depth: int = 0,
    ellipsis: str | None = None,
    label_primitives: bool = False,
) -> str:
    """Format mapping for display with automatic fallback for non-mapping types.

    Formats mapping objects (dicts, OrderedDict, etc.) for debugging or logging.
    Non-mapping inputs are gracefully handled by delegating to `fmt_value`,
    making this function safe to use in error contexts where object types
    may be uncertain.

    Args:
        mp: The object to format. Mappings are formatted as `{key: value}`
            pairs, while all other types delegate to `fmt_value`.
        style: Display style - "angle", "equal", "colon", etc.
        max_items: For mappings, the max key-value pairs to show before truncating.
        max_repr: Maximum length of individual key/value reprs before truncation.
        depth: Maximum recursion depth for nested structures within a mapping.
        ellipsis: Custom truncation token. Auto-selected per style if None.
        label_primitives: Whether to show type labels for int, float, str, bytes, etc.

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
        return fmt_value(
            mp, style=style, max_repr=max_repr, ellipsis=ellipsis, label_primitives=label_primitives
        )

    opts = opts or FmtOptions()

    # Support mappings without reliable len by sampling
    items_iter: Iterator[Tuple[Any, Any]] = iter(mp.items())
    sampled = list(islice(items_iter, max_items + 1))
    had_more = len(sampled) > max_items
    if had_more:
        sampled = sampled[:max_items]

    parts: list[str] = []
    for k, v in sampled:
        k_str = fmt_value(
            k, style=style, max_repr=max_repr, ellipsis=ellipsis, label_primitives=label_primitives
        )
        if depth > 0 and not _is_textual(v):
            if isinstance(v, abc.Mapping):
                v_str = fmt_mapping(
                    v,
                    style=style,
                    max_items=max_items,
                    max_repr=max_repr,
                    depth=depth - 1,
                    ellipsis=ellipsis,
                    label_primitives=label_primitives,
                )
            elif isinstance(v, abc.Set):
                v_str = fmt_set(
                    v,
                    style=style,
                    max_items=max_items,
                    max_repr=max_repr,
                    depth=depth - 1,
                    ellipsis=ellipsis,
                    label_primitives=label_primitives,
                )
            elif isinstance(v, abc.Sequence):
                v_str = fmt_sequence(
                    v,
                    style=style,
                    max_items=max_items,
                    max_repr=max_repr,
                    depth=depth - 1,
                    ellipsis=ellipsis,
                    label_primitives=label_primitives,
                )
            else:
                v_str = fmt_value(
                    v,
                    style=style,
                    max_repr=max_repr,
                    ellipsis=ellipsis,
                    label_primitives=label_primitives,
                )
        else:
            v_str = fmt_value(
                v,
                style=style,
                max_repr=max_repr,
                ellipsis=ellipsis,
                label_primitives=label_primitives,
            )
        parts.append(f"{k_str}: {v_str}")

    more = _fmt_ellipsis(style, ellipsis) if had_more else ""
    return "{" + ", ".join(parts) + more + "}"


@validate_types
def fmt_sequence(
    seq: Iterable[Any],
    *,
    opts: FmtOptions | None = None,
    style: Style = "equal",
    max_items: int = 8,
    max_repr: int = 120,
    depth: int = 0,
    ellipsis: str | None = None,
    label_primitives: bool = False,
) -> str:
    """Format sequence for display with automatic fallback for non-iterable types.

    Formats iterables (lists, tuples, sets, generators) for debugging or logging.
    Non-iterable inputs and text-like sequences (str, bytes, bytearray) are
    gracefully handled by delegating to `fmt_value`, making this function safe
    to use in error contexts where object types may be uncertain.

    Args:
        seq: The object to format. Iterables are formatted elementwise,
            while non-iterables and text types delegate to `fmt_value`.
        style: Display style - "angle", "equal", "colon", etc.
        max_items: Maximum elements to show before truncating. Default of 8.
        max_repr: Maximum length of individual element reprs before truncation.
        depth: Maximum recursion depth for nested structures. 0 treats nested objects as atomic.
        ellipsis: Custom truncation token. Auto-selected per style if None.
        label_primitives: Whether to show type labels for int, float, str, bytes, etc.

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
        return fmt_value(
            seq,
            style=style,
            max_repr=max_repr,
            ellipsis=ellipsis,
            label_primitives=label_primitives,
        )

    opts = opts or FmtOptions()

    if _is_textual(seq):
        # Treat text-like as a scalar value, not a sequence of characters
        return fmt_value(
            seq,
            style=style,
            max_repr=max_repr,
            ellipsis=ellipsis,
            label_primitives=label_primitives,
        )

    opts = FmtOptions() if opts is None else opts

    # Choose delimiters by common concrete types; fallback to []
    open_ch, close_ch = "[", "]"
    is_tuple = isinstance(seq, tuple)
    if is_tuple:
        open_ch, close_ch = "(", ")"

    items, had_more = _fmt_head(seq, max_items)

    parts = list()
    for x in items:
        # Recurse into nested structures one level at a time
        if depth > 0 and not _is_textual(x):
            if isinstance(x, abc.Mapping):
                parts.append(
                    fmt_mapping(
                        x,
                        style=style,
                        max_items=max_items,
                        max_repr=max_repr,
                        depth=depth - 1,
                        ellipsis=ellipsis,
                        label_primitives=label_primitives,
                    )
                )
                continue
            if isinstance(x, abc.Set):
                parts.append(
                    fmt_set(
                        x,
                        style=style,
                        max_items=max_items,
                        max_repr=max_repr,
                        depth=depth - 1,
                        ellipsis=ellipsis,
                        label_primitives=label_primitives,
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
                        label_primitives=label_primitives,
                    )
                )
                continue
        parts.append(
            fmt_value(
                x,
                style=style,
                max_repr=max_repr,
                ellipsis=ellipsis,
                label_primitives=label_primitives,
            )
        )

    more = _fmt_ellipsis(style, ellipsis) if had_more else ""
    # Singleton tuple needs a trailing comma for Python literal accuracy
    tail = "," if is_tuple and len(parts) == 1 and not more else ""
    return f"{open_ch}" + ", ".join(parts) + more + f"{tail}{close_ch}"


@validate_types
def fmt_set(
    st: AbstractSet[Any],
    *,
    opts: FmtOptions | None = None,
    style: Style = "equal",
    max_items: int = 8,
    max_repr: int = 120,
    depth: int = 0,
    ellipsis: str | None = None,
    label_primitives: bool = False,
) -> str:
    """Format set for display with automatic fallback for non-set types.

    Formats set objects (Set, FozenSet, etc.) for debugging or logging.
    Non-set inputs are gracefully handled by delegating to `fmt_value`,
    making this function safe to use in error contexts where object types
    may be uncertain.

    Args:
        st: The object to format. Sets are formatted as `{element, ...}`,
            while all other types delegate to `fmt_value`.
        style: Display style - "angle", "equal", "colon", etc.
        max_items: For sets, the max elements to show before truncating.
        max_repr: Maximum length of individual element repr before truncation.
        depth: Maximum recursion depth for nested structures within a set.
        ellipsis: Custom truncation token. Auto-selected per style if None.
        label_primitives: Whether to show type labels for int, float, str, bytes, etc.

    Returns:
        Formatted string. For sets: `'{type=value, ...}'`
        For non-mappings: delegated to `fmt_value`.

    Notes:
        - Non-set types automatically fall back to `fmt_value` (no exceptions)
        - Preserves insertion order for modern set-s
        - Elements are formatted using `fmt_value`
        - Broken `__repr__` methods are handled gracefully

    Examples:
        >>> # Standard mapping formatting
        >>> fmt_set({"name", "age"}, style="angle", label_primitives=True)
        "{<str: 'name'>, <str: 'age'>}"

        >>> # Truncation of large mappings
        >>> fmt_set({i for i in range(10)}, max_items=3, style="angle", label_primitives=True)
        '{<int: 0>, <int: 1>, <int: 2>...}'

        >>> # Automatic fallback for non-mappings (no error)
        >>> fmt_set("a simple string", style="angle", label_primitives=True)
        "<str: 'a simple string'>"
        >>> fmt_set(42)
        '42'

    See Also:
        fmt_value: The underlying formatter for individual values and non-mappings.
        fmt_mapping: Formats mappings with similar robustness.
    """
    # Check if it's actually a set - if not, delegate to fmt_value
    if not isinstance(st, abc.Set):
        return fmt_value(
            st, style=style, max_repr=max_repr, ellipsis=ellipsis, label_primitives=label_primitives
        )

    opts = opts or FmtOptions()

    # Support sets without reliable len by sampling
    items_iter: Iterator[Tuple[Any, Any]] = iter(st)
    sampled = list(islice(items_iter, max_items + 1))
    had_more = len(sampled) > max_items
    if had_more:
        sampled = sampled[:max_items]

    parts: list[str] = []
    for el in sampled:
        if depth > 0 and not _is_textual(el):
            if isinstance(el, abc.Mapping):
                el_str = fmt_mapping(
                    el,
                    style=style,
                    max_items=max_items,
                    max_repr=max_repr,
                    depth=depth - 1,
                    ellipsis=ellipsis,
                    label_primitives=label_primitives,
                )
            elif isinstance(el, abc.Set):
                el_str = fmt_set(
                    el,
                    style=style,
                    max_items=max_items,
                    max_repr=max_repr,
                    depth=depth - 1,
                    ellipsis=ellipsis,
                    label_primitives=label_primitives,
                )
            elif isinstance(el, abc.Sequence):
                el_str = fmt_sequence(
                    el,
                    style=style,
                    max_items=max_items,
                    max_repr=max_repr,
                    depth=depth - 1,
                    ellipsis=ellipsis,
                    label_primitives=label_primitives,
                )
            else:
                el_str = fmt_value(
                    el,
                    style=style,
                    max_repr=max_repr,
                    ellipsis=ellipsis,
                    label_primitives=label_primitives,
                )
        else:
            el_str = fmt_value(
                el,
                style=style,
                max_repr=max_repr,
                ellipsis=ellipsis,
                label_primitives=label_primitives,
            )
        parts.append(el_str)

    more = _fmt_ellipsis(style, ellipsis) if had_more else ""
    return "{" + ", ".join(parts) + more + "}"


@validate_types
def fmt_type(obj: Any, *, opts: FmtOptions | None = None) -> str:
    """Format type information for debugging, logging, and exception messages.

    Provides consistent formatting of type information for both type objects and
    instances. Complements the other fmt_* functions by focusing specifically on
    type display with optional module qualification and the same robust error handling.

    Args:
        obj: Any Python object or type to extract type information from.
        style: Display style - "angle", "equal", "colon", etc.
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
    opts = opts or FmtOptions()

    # get type name with robust edge cases
    type_name = class_name(
        obj, fully_qualified=opts.fully_qualified, fully_qualified_builtins=False
    )

    # Format as a type-no-value string
    style = opts.style or "equal"
    if style == "angle":
        return f"<{type_name}>"
    if style == "arrow":
        return type_name
    if style == "braces":
        return "{" + type_name + "}"
    if style == "colon":
        return type_name
    if style == "equal":
        return type_name
    if style == "paren":
        return type_name
    if style == "unicode-angle":
        return f"⟨{type_name}⟩"
    else:
        return type_name


@validate_types
def fmt_value(obj: Any, *, opts: FmtOptions | None = None) -> str:
    """
    Format a single value as a type–value pair for debugging, logging, and exception messages.

    Intended for robust display of arbitrary values in error contexts where safety and
    readability matter more than perfect fidelity. Handles edge cases like broken __repr__,
    recursive objects, and extremely long representations gracefully.

    Args:
        obj: Any Python object to format.
        opts: Formatting options controlling style, truncation, and type labeling.
            If None, uses default FmtOptions(). See FmtOptions for available settings:
            - style: Display style ("angle", "equal", "colon", "unicode-angle", etc.)
            - label_primitives: Whether to show type labels for int, float, str, bytes, etc.
            - repr: reprlib.Repr instance controlling truncation and recursion handling
            - ellipsis: Custom truncation token (note: reprlib uses '...' internally)

    Returns:
        Formatted string like "int=42" (equal style) or "⟨str: 'hello'⟩" (unicode-angle style).

    Notes:
        - Truncation is handled by reprlib.Repr (respects maxstring, maxother, maxlevel).
        - ASCII style escapes inner ">" to avoid conflicts with wrapper brackets.
        - Broken __repr__ methods are handled gracefully with fallback formatting.
        - Designed for exception messages and logs where robustness trumps perfect formatting.

    Examples:
        >>> fmt_value(42)
        'int=42'

        >>> opts = FmtOptions.compact()
        >>> fmt_value("hello world", opts=opts)
        "str='hello wo...'"

        >>> opts = FmtOptions(style="unicode-angle", label_primitives=True)
        >>> fmt_value([1, 2, 3], opts=opts)
        '⟨list: [1, 2, 3]⟩'

    See Also:
        fmt_sequence: Format sequences/iterables elementwise with nesting support.
        fmt_mapping: Format mappings with key-value pairs and nesting support.
    """
    opts = opts or FmtOptions()

    # Generate repr using reprlib for consistent truncation and recursion handling
    repr_ = _safe_repr(obj, opts)

    # Unlabeled primitives case
    if _is_primitive(obj) and not opts.label_primitives:
        return repr_

    # Formatted type-value pair case
    t = type(obj).__name__
    return _fmt_type_value(t, repr_, opts=opts)


# Private Methods ------------------------------------------------------------------------------------------------------


def _fmt_ellipsis(style: Style, more_token: str | None = None) -> str:
    """Decide which 'more' token to use (ellipsis vs custom)."""
    if more_token is not None:
        return more_token
    return "..." if style == "angle" else "…"


def _fmt_head(iterable: Iterable[Any], n: int) -> Tuple[list[Any], bool]:
    """Take up to n items and indicate whether there were more items."""
    it = iter(iterable)
    buf = list(islice(it, n + 1))
    if len(buf) <= n:
        return buf, False
    return buf[:n], True


def _fmt_repr(
    max_items: int | Any = None,
    max_depth: int | Any = None,
    max_str: int | Any = None,
    default: reprlib.Repr | Any = None,
) -> reprlib.Repr:
    """
    Creates and configures a customized instance of `reprlib.Repr` based on the
    given constraints for maximum items, depth, and length. This allows for fine-tuned
    control over the string representation of complex objects.

    Handles gracefully any type in params.

    Args:
        max_items (int): The maximum number of items to include in string
            representations for collections such as lists, tuples, dictionaries,
            etc.
        max_depth (int): The maximum depth allowed for nested structures in the
            string representation.
        max_str (int): The maximum length for strings and other representations.
        default (reprlib.Repr, optional): An optional existing `reprlib.Repr` instance
            whose properties are used to initialize the new instance. If provided,
            its attributes will serve as a baseline for configuration.

    Returns:
        reprlib.Repr: A configured `reprlib.Repr` instance initialized with the
        specified or default parameters for maximum items, depth, and other limits.
    """
    if isinstance(default, reprlib.Repr):
        _ = default
        r = reprlib.Repr(
            maxlevel=_.maxlevel,
            maxtuple=_.maxtuple,
            maxlist=_.maxlist,
            maxarray=_.maxarray,
            maxdict=_.maxdict,
            maxset=_.maxset,
            maxfrozenset=_.maxfrozenset,
            maxdeque=_.maxdeque,
            maxstring=_.maxstring,
            maxlong=_.maxlong,
            maxother=_.maxother,
            fillvalue=_.fillvalue,
            indent=_.indent,
        )
    else:
        r = reprlib.Repr()

    if isinstance(max_items, int):
        r.maxdict = r.maxlist = r.maxtuple = r.maxset = max_items
        r.maxfrozenset = r.maxdeque = r.maxarray = max_items

    if isinstance(max_depth, int):
        r.maxlevel = max_depth

    if isinstance(max_str, int):
        r.maxstring = r.maxother = max_str

    return r


def _fmt_truncate(repr_: str, max_len: int, ellipsis: str = "…") -> str:
    """
    Truncate repr_ to at most max_len visible characters before appending the ellipsis.

    For str/bytes/bytearray, max_len refers to the actual data content length,
    not the full repr length. The ellipsis is appended in full.
    For other types, max_len refers to the full repr string length.
    """
    s = repr_
    if max_len <= 0:
        return ""
    if len(s) <= max_len:
        return s

    # Check for bytearray repr: bytearray(b'...')
    if s.startswith("bytearray(b'") or s.startswith('bytearray(b"'):
        prefix = "bytearray(b"
        quote = s[len(prefix)]
        suffix = "')" if quote == "'" else '")'

        # max_len applies to inner content only
        inner_budget = max(1, max_len)
        inner_start = len(prefix) + 1
        inner = s[inner_start : inner_start + inner_budget]

        return f"{prefix}{quote}{inner}{ellipsis}{quote}{suffix[-1]}"

    # Check for bytes repr: b'...'
    if (s.startswith("b'") or s.startswith('b"')) and len(s) >= 3:
        prefix = "b"
        quote = s[1]

        # max_len applies to inner content only
        inner_budget = max(1, max_len)
        inner = s[2 : 2 + inner_budget]

        return f"{prefix}{quote}{inner}{ellipsis}{quote}"

    # Check for str repr: '...'
    if len(s) >= 2 and s[0] in ("'", '"') and s[-1] == s[0]:
        quote = s[0]

        # max_len applies to inner content only
        inner_budget = max(1, max_len)
        inner = s[1 : 1 + inner_budget]

        return f"{quote}{inner}{ellipsis}{quote}"

    # General case: max_len applies to full repr
    keep = max(1, max_len)
    return s[:keep] + ellipsis


def _fmt_type_value(type_name: str, value_repr: str, *, opts: FmtOptions = None) -> str:
    """
    Combine a type name and a repr into a single display token according to style.

    This method does not handle `repr` style and falls back to
    """
    opts = opts or FmtOptions()
    style = opts.style or "equal"
    if style == "angle":
        return f"<{type_name}: {value_repr}>"
    if style == "arrow":
        return f"{type_name} -> {value_repr}"
    if style == "braces":
        return "{" + f"{type_name}: {value_repr}" + "}"
    if style == "colon":
        return f"{type_name}: {value_repr}"
    if style == "equal":
        return f"{type_name}={value_repr}"
    if style == "paren":
        return f"{type_name}({value_repr})"
    if style == "unicode-angle":
        return f"⟨{type_name}: {value_repr}⟩"
    else:
        # Gracefully fallback to 'equal' if user provided invalid style
        return f"{type_name}={value_repr}"


def _safe_repr(obj, opts: FmtOptions) -> str:
    """
    Defensive repr() call using reprlib for truncation and recursion handling.

    Args:
        obj: Object to represent
        opts: Formatting options containing reprlib.Repr configuration

    Returns:
        String representation, with fallback for broken __repr__ methods
    """
    try:
        repr_ = opts.repr.repr(obj)
    except Exception as e:
        # Fallback for broken __repr__: show type and exception info
        exc_type = type(e).__name__
        repr_ = f"<{type(obj).__name__} object (repr failed: {exc_type})>"
    return repr_


def _is_primitive(obj) -> bool:
    """Check if object should be displayed without type label."""
    # We should use type(), not isinstance() here
    return type(obj) in PRIMITIVE_TYPES


def _is_textual(x: Any) -> bool:
    return isinstance(x, (str, bytes, bytearray))
