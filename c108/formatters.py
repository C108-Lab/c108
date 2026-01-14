"""
Robust formatting utilities for development and debugging.

Type-aware formatters for debugging, logging, and exception messages.
All formatters handle broken __repr__, recursive objects,
and edge cases gracefully with consistent styling across ASCII/Unicode output.
The fmt_any() function intelligently dispatches to specialized formatters.
"""

# TODO Check zero depth of recursion

# TODO * reprlib-like recursive items showcase
#      * style-aware broken __repr__ behavior (replace <>)
#      *

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

import array
import collections
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

# Local imports --------------------------------------------------------------------------------------------------------

from c108.sentinels import UNSET, ifnotunset
from c108.utils import Self, class_name

# Constants ------------------------------------------------------------------------------------------------------------

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
    repr: reprlib.Repr = field(default_factory=lambda: _repr_factory())
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
            object.__setattr__(self, "style", "repr")

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
        r = _repr_factory(max_depth=max_depth, max_items=max_items, max_str=max_str, default=r)

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
    def compact(cls, max_depth: int = 2, max_items: int = 6, max_str: int = 64) -> Self:
        """Minimal output for tight spaces."""
        r = _repr_factory(max_depth=max_depth, max_items=max_items, max_str=max_str)
        return cls(
            deduplicate_types=True,
            fully_qualified=False,
            include_traceback=False,
            label_primitives=False,
            repr=r,
        )

    @classmethod
    def debug(cls, max_depth: int = 6, max_items: int = 256, max_str=1024) -> Self:
        """Verbose output for debugging."""
        r = _repr_factory(max_depth=max_depth, max_items=max_items, max_str=max_str)
        return cls(
            deduplicate_types=True,
            fully_qualified=False,
            include_traceback=True,
            label_primitives=True,
            repr=r,
        )

    @classmethod
    def logging(cls, max_depth: int = 3, max_items: int = 64, max_str: int = 128) -> Self:
        """Balanced output for production logging."""
        r = _repr_factory(max_depth=max_depth, max_items=max_items, max_str=max_str)
        return cls(
            deduplicate_types=False,
            fully_qualified=False,
            include_traceback=False,
            label_primitives=False,
            repr=r,
        )

    @property
    def ellipsis(self) -> str:
        return self.repr.fillvalue

    @property
    def max_depth(self) -> int:
        """Maximum nesting depth."""
        return self.repr.maxlevel

    @property
    def max_items(self) -> int:
        """Maximum items number in repr (uses maxlist as canonical value)."""
        return self.repr.maxlist

    @property
    def max_str(self) -> int:
        """Maximum string length in repr (uses maxstring as canonical value)."""
        return self.repr.maxstring


# Methods --------------------------------------------------------------------------------------------------------------


def fmt_any(obj: Any, *, opts: FmtOptions | None = None) -> str:
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
    # Provide valid FmtOptions instance
    opts = _fmt_opts(opts)

    # Priority 1: Exceptions get special handling
    if isinstance(obj, BaseException):
        return fmt_exception(obj, opts=opts)

    # Priority 2: Mappings (dict, OrderedDict, etc.)
    if isinstance(obj, abc.Mapping):
        return fmt_mapping(obj, opts=opts)

    # Priority 3: Sets
    if isinstance(obj, abc.Set):
        return fmt_set(obj, opts=opts)

    # Priority 4: Sequences (but not text-like ones)
    if isinstance(obj, abc.Sequence) and not _is_textual(obj):
        return fmt_sequence(obj, opts=opts)

    # Priority 5: Everything else (atomic values, text, custom objects)
    return fmt_value(obj, opts=opts)


def fmt_exception(
    exc: Any,
    *,
    opts: FmtOptions | None = None,
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
    # Process BaseException - if not, delegate to fmt_value
    if not isinstance(exc, BaseException):
        return fmt_value(exc, opts=opts)

    # Provide valid FmtOptions instance
    opts = _fmt_opts(opts)

    # Propagate params
    style = opts.style
    max_repr = opts.repr.maxother
    ell = opts.ellipsis

    # Get exception type name
    exc_type = type(exc).__name__

    # Get exception message safely
    try:
        exc_msg = str(exc)
    except Exception:
        exc_msg = "<repr failed>"

    # Build base format based on style - TYPE NAME IS NEVER TRUNCATED
    if exc_msg:
        if style == "equal":
            # Calculate space available for message
            base_length = len(exc_type) + 1  # "ValueError="
            if max_repr > 0 and base_length + len(exc_msg) > max_repr:
                available_for_msg = max_repr - base_length - len(ell)
                if available_for_msg > 0:
                    truncated_msg = exc_msg[:available_for_msg]
                    base_format = f"{exc_type}={truncated_msg}{ell}"
                else:
                    base_format = f"{exc_type}={ell}"
            else:
                base_format = f"{exc_type}={exc_msg}"
        elif style == "unicode-angle":
            # Calculate space available for message
            base_length = len(exc_type) + 4  # "⟨ValueError: ⟩"
            if max_repr > 0 and base_length + len(exc_msg) > max_repr:
                available_for_msg = max_repr - base_length - len(ell)
                if available_for_msg > 0:
                    truncated_msg = exc_msg[:available_for_msg]
                    base_format = f"⟨{exc_type}: {truncated_msg}{ell}⟩"
                else:
                    base_format = f"⟨{exc_type}: {ell}⟩"
            else:
                base_format = f"⟨{exc_type}: {exc_msg}⟩"
        else:  # ascii
            # Calculate space available for message
            base_length = len(exc_type) + 4  # "<ValueError: >"
            if max_repr > 0 and base_length + len(exc_msg) > max_repr:
                available_for_msg = max_repr - base_length - len(ell)
                if available_for_msg > 0:
                    truncated_msg = exc_msg[:available_for_msg]
                    base_format = f"<{exc_type}: {truncated_msg}{ell}>"
                else:
                    base_format = f"<{exc_type}: {ell}>"
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
    if opts.include_traceback:
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
    opts: FmtOptions | None = None,
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
    # Process Mapping, delegate to fmt_value all the rest
    if not isinstance(mp, abc.Mapping):
        return fmt_value(mp, opts=opts)

    # Provide valid FmtOptions instance
    opts = _fmt_opts(opts)

    # Propagate Params
    max_items = opts.repr.maxdict

    # Support mappings without reliable len by sampling
    items_iter: Iterator[Tuple[Any, Any]] = iter(mp.items())
    sampled = list(islice(items_iter, max_items + 1))
    had_more = len(sampled) > max_items
    if had_more:
        sampled = sampled[:max_items]

    parts: list[str] = list()
    opts = opts.merge(max_depth=(opts.max_depth - 1))
    for k, v in sampled:
        k_str = fmt_value(k, opts=opts)
        if opts.max_depth > 0 and not _is_textual(v):
            v_str = fmt_mapping(v, opts=opts)
        else:
            v_str = fmt_value(v, opts=opts)
        parts.append(f"{k_str}: {v_str}")

    more = _fmt_ellipsis(opts.style, opts.ellipsis) if had_more else ""
    return "{" + ", ".join(parts) + more + "}"


def fmt_repr(obj: Any, *, opts: FmtOptions | None = None) -> str:
    """
    Format the object's representation safely and robustly.

    A wrapper around repr() that guarantees:
        1. No crashes (catches exceptions in broken __repr__ methods).
        2. Syntactic correctness (restores quotes/brackets even when truncated).
        3. Configurable limits (via opts).

    Args:
        obj: The object to represent.
        opts: Formatting options. If None, uses defaults.

    Returns:
        A string resembling the standard repr(), but safe and truncated.
    """
    # Provide valid FmtOptions instance
    opts = _fmt_opts(opts)

    # Get formatted representation (fail-safe)
    return _fmt_repr(obj, opts)


# ---------------------------------------------------------------


def fmt_sequence(
    seq: Iterable[Any],
    *,
    opts: FmtOptions | None = None,
) -> str:
    """Format sequence for display with automatic fallback for non-iterable types.

    Formats iterables (lists, tuples, sets, generators) for debugging or logging.
    Non-iterable inputs and text-like sequences (str, bytes, bytearray) are
    gracefully handled by delegating to `fmt_value`, making this function safe
    to use in error contexts where object types may be uncertain.

    Args:
        seq: The object to format. Iterables are formatted elementwise,
            while non-iterables and text types delegate to `fmt_value`.
        opts: Formatting options controlling style, truncation, and type labeling.

    Returns:
        Formatted string like "[<int: 1>, <str: 'hello'>...]" (list) or
        "(<int: 1>,)" (singleton tuple). Custom types show as "CustomList([...])".

    Notes:
        - Non-iterable types automatically fall back to `fmt_value` (no exceptions)
        - str/bytes/bytearray are treated as atomic (not decomposed into characters)
        - Built-in types use standard brackets: [] for lists, () for tuples
        - Custom types show with type name wrapper: CustomList([...])
        - For sized sequences over max_items, shows head...tail pattern (reprlib-style)
        - For generators/iterators, shows head...only pattern (can't peek tail)
        - Nested sequences/mappings are recursively formatted up to max_depth levels
        - Singleton tuples show trailing comma for Python literal accuracy
        - Broken __repr__ methods in elements are handled gracefully

    Examples:
        >>> fmt_sequence([1, "hello", [2, 3]])
        "[<int: 1>, <str: 'hello'>, [<int: 2>, <int: 3>]]"

        >>> fmt_sequence(range(100), max_items=6)
        '[<int: 0>, <int: 1>, <int: 2>, ..., <int: 97>, <int: 98>, <int: 99>]'

        >>> class CustomList(list): pass
        >>> fmt_sequence(CustomList([1, 2, 3]))
        'CustomList([<int: 1>, <int: 2>, <int: 3>])'

        >>> fmt_sequence("text")
        "<str: 'text'>"

        >>> fmt_sequence(42)
        '<int: 42>'

    See Also:
        fmt_value: Format individual elements with the same robustness guarantees.
        fmt_mapping: Format mappings with similar nesting support.
    """
    # Process Iterable, delegate to fmt_value all the rest
    if not isinstance(seq, abc.Iterable):
        return fmt_value(seq, opts=opts)

    if _is_textual(seq):
        # Treat text-like as a scalar value, not a sequence of characters
        return fmt_value(seq, opts=opts)

    # Provide valid FmtOptions instance
    opts = _fmt_opts(opts)

    # Determine if this is a built-in or custom type
    seq_type = type(seq)
    if seq_type in (list, tuple, set, frozenset, range, collections.deque):
        return _fmt_sequence_builtin(seq, opts)
    else:
        return _fmt_sequence_custom(seq, seq_type, opts)


def _is_builtin_sequence(seq_type: type) -> bool:
    """Check if sequence type is a built-in (list, tuple, set, etc.)."""
    builtin_types = (list, tuple, set, frozenset, range, collections.deque)
    return seq_type in builtin_types


def _fmt_sequence_builtin(seq: Iterable[Any], opts: FmtOptions) -> str:
    """Format built-in sequence types (list, tuple, set, range, etc.)."""
    is_tuple = isinstance(seq, tuple)
    if is_tuple:
        open_ch, close_ch = ("(", ")")
    else:
        open_ch, close_ch = ("[", "]")

    # Check if we can use reprlib-style head+tail truncation
    if isinstance(seq, abc.Sized) and _is_indexable_efficiently(seq):
        parts, had_more = _fmt_items_head_tail(seq, opts)
    else:
        # Generator/iterator or expensive-to-index: head-only truncation
        parts, had_more = _fmt_items_head_only(seq, opts)

    more = opts.ellipsis if had_more else ""
    # Singleton tuple needs a trailing comma for Python literal accuracy
    tail = "," if is_tuple and len(parts) == 1 and not more else ""

    return f"{open_ch}" + ", ".join(parts) + more + f"{tail}{close_ch}"


def _fmt_sequence_custom(seq: Iterable[Any], seq_type: type, opts: FmtOptions) -> str:
    """Format custom sequence types with type name wrapper."""
    type_name = class_name(
        seq, fully_qualified=opts.fully_qualified, fully_qualified_builtins=False, as_instance=False
    )

    # Check if we can use reprlib-style head+tail truncation
    if isinstance(seq, abc.Sized) and _is_indexable_efficiently(seq):
        parts, had_more = _fmt_items_head_tail(seq, opts)
    else:
        # Generator/iterator or expensive-to-index: head-only truncation
        parts, had_more = _fmt_items_head_only(seq, opts)

    more = opts.ellipsis if had_more else ""

    # Skip inner "()" for tuple-like objs, use inner "[]" for list-like
    # Default to brackets for most custom iterables
    if isinstance(seq, tuple):
        open_ch, close_ch = ("", "")
    else:
        open_ch, close_ch = ("[", "]")

    inner = f"{open_ch}" + ", ".join(parts) + more + f"{close_ch}"
    return f"{type_name}({inner})"


def _is_indexable_efficiently(seq: Any) -> bool:
    """
    Check if sequence supports efficient indexing for head+tail pattern.

    Returns True for list/tuple subclasses, False for custom Sized iterables
    that might have expensive __getitem__ operations.
    """
    seq_type = type(seq)
    return issubclass(seq_type, (list, tuple))


def _fmt_items_head_tail(seq: abc.Sized, opts: FmtOptions) -> Tuple[list[str], bool]:
    """
    Format items using reprlib-style head+tail truncation pattern.

    Returns (formatted_parts, had_more) where had_more indicates truncation.
    Shows first n//2 and last n//2 items when length exceeds max_items.
    """
    seq_len = len(seq)
    max_items = opts.max_items

    if seq_len <= max_items:
        # No truncation needed
        parts = [_fmt_item(x, opts) for x in seq]
        return parts, False

    # Truncate with head+tail pattern
    head_count = max_items // 2
    tail_count = max_items - head_count

    # Format head items
    head_parts = [_fmt_item(seq[i], opts) for i in range(head_count)]

    # Format tail items
    tail_parts = [_fmt_item(seq[-(tail_count - i)], opts) for i in range(tail_count)]

    return head_parts + tail_parts, True


def _fmt_items_head_only(seq: Iterable[Any], opts: FmtOptions) -> Tuple[list[str], bool]:
    """
    Format items using head-only truncation for generators/iterators.

    Returns (formatted_parts, had_more) where had_more indicates truncation.
    Consumes up to max_items+1 items to detect if there are more.
    """
    max_items = opts.max_items
    items = []
    had_more = False

    for i, x in enumerate(seq):
        if i >= max_items:
            had_more = True
            break
        items.append(x)

    parts = [_fmt_item(x, opts) for x in items]
    return parts, had_more


def _fmt_item(obj: Any, opts: FmtOptions) -> str:
    """Format a single item with depth-aware recursion."""
    opts_nested = opts.merge(max_depth=(opts.max_depth - 1))

    # Recurse into nested structures one level at a time
    if opts.max_depth > 0 and not _is_textual(obj):
        return fmt_any(obj, opts=opts_nested)
    else:
        return fmt_value(obj, opts=opts_nested)


# ---------------------------------------------------------------


def fmt_set(st: AbstractSet[Any], *, opts: FmtOptions | None = None) -> str:
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
    # Process Set, delegate to fmt_value everything else
    if not isinstance(st, abc.Set):
        return fmt_value(st, opts=opts)

    # Provide valid FmtOptions instance
    opts = _fmt_opts(opts)

    # Params propagate
    max_items = opts.repr.maxfrozenset if isinstance(st, abc.FrozenSet) else opts.repr.maxset

    # Support sets without reliable len by sampling
    items_iter: Iterator[Tuple[Any, Any]] = iter(st)
    sampled = list(islice(items_iter, max_items + 1))
    had_more = len(sampled) > max_items
    if had_more:
        sampled = sampled[:max_items]

    parts: list[str] = list()
    opts = opts.merge(max_depth=(opts.max_depth - 1))
    for el in sampled:
        if opts.max_depth > 0 and not _is_textual(el):
            el_str = fmt_any(el, opts=opts)
        else:
            el_str = fmt_value(el, opts=opts)
        parts.append(el_str)

    more = _fmt_ellipsis(opts.style, opts.ellipsis) if had_more else ""
    return "{" + ", ".join(parts) + more + "}"


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
    # Provide valid FmtOptions instance
    opts = _fmt_opts(opts)

    # Get type name with robust edge cases; if obj is `int` we should get `type` as its type_name
    type_name = class_name(
        obj, fully_qualified=opts.fully_qualified, fully_qualified_builtins=False, as_instance=True
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
    if style == "repr":
        return type_name
    if style == "unicode-angle":
        return f"⟨{type_name}⟩"
    else:
        return type_name


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
        fmt_any: Format object dispatcher with custom options.
        fmt_mapping: Format mappings with key-value pairs and nesting support.
        fmt_repr: Format object using reprlib with custom options.
        fmt_sequence: Format sequences/iterables elementwise with nesting support.
        fmt_set: Format sets with elementwise formatting and nesting support.
    """
    # Provide valid FmtOptions instance
    opts = _fmt_opts(opts)

    # Generate repr using reprlib for consistent truncation and recursion handling
    repr_ = _fmt_repr(obj, opts)

    # Unlabeled primitives case: shoud show repr as is
    if _is_primitive(obj) and not opts.label_primitives:
        return repr_

    # Formatted type-value pair case; if obj is `int` we should get `type` as its type_name
    type_name = class_name(
        obj, fully_qualified=opts.fully_qualified, fully_qualified_builtins=False, as_instance=True
    )

    style = opts.style or "equal"
    if style == "angle":
        return f"<{type_name}: {repr_}>"
    if style == "arrow":
        return f"{type_name} -> {repr_}"
    if style == "braces":
        return "{" + f"{type_name}: {repr_}" + "}"
    if style == "colon":
        return f"{type_name}: {repr_}"
    if style == "equal":
        return f"{type_name}={repr_}"
    if style == "paren":
        return f"{type_name}({repr_})"
    if style == "repr":
        return repr_
    if style == "unicode-angle":
        return f"⟨{type_name}: {repr_}⟩"
    else:
        # Gracefull fallback if provided invalid style
        return repr_


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


def _fmt_opts(opts: FmtOptions):
    """
    Return the given FmtOptions object or create a new one if the opts is None or invalid.

    Args:
        opts (FmtOptions | None): The options object to format. Can be None or an invalid
            type. If None or invalid, a default FmtOptions instance will be returned.

    Returns:
        FmtOptions: A valid options object derived from the one provided or a new
            FmtOptions instance if the input is None or invalid.
    """
    if opts is None or not isinstance(opts, FmtOptions):
        return FmtOptions()
    return opts


def _fmt_repr(obj, opts: FmtOptions) -> str:
    """
    Formatted defensive repr() call by using FmtOptions.repr instance.
    """
    try:
        repr_ = opts.repr.repr(obj)
        ellipsis = opts.repr.fillvalue

        # If the representation was truncated (equals the fillvalue),
        # try to expand it with type-specific formatting
        if repr_ == ellipsis:
            return _fmt_repr_expand_ellipsis(obj, ellipsis, opts)

        return repr_

    except Exception as e:
        fq = getattr(opts, "fully_qualified", False)
        # Builtins usually don't need FQ in error messages, keeping consistent with original
        fq_builtins = False

        exc_type = type(e).__name__
        obj_name = class_name(
            obj, fully_qualified=fq, fully_qualified_builtins=fq_builtins, as_instance=True
        )

        return f"<{obj_name} instance at {id(obj)} (repr failed: {exc_type})>"


def _fmt_repr_expand_ellipsis(obj, ellipsis: str, opts: FmtOptions) -> str:
    """
    Determines the formatted string for objects that hit the repr recursion limit
    or length limit (returned as ellipsis).
    """

    # 1. String-like types
    if isinstance(obj, str):
        return f"'{ellipsis}'"
    elif isinstance(obj, bytes):
        return f"b'{ellipsis}'"
    elif isinstance(obj, bytearray):
        return f"bytearray(b'{ellipsis}')"

    # 2. Collections module types (Must come BEFORE dict/list/tuple)
    elif isinstance(obj, collections.deque):
        return f"deque([{ellipsis}])"
    elif isinstance(obj, collections.OrderedDict):
        return f"OrderedDict({{{ellipsis}}})"
    elif isinstance(obj, collections.defaultdict):
        return _fmt_repr_defaultdict(obj, ellipsis, opts)
    elif isinstance(obj, collections.Counter):
        return f"Counter({{{ellipsis}}})"
    elif isinstance(obj, collections.ChainMap):
        return f"ChainMap({{{ellipsis}}})"

    # 3. Basic collections
    elif isinstance(obj, list):
        return f"[{ellipsis}]"
    elif isinstance(obj, tuple):
        return f"({ellipsis},)" if len(obj) == 1 else f"({ellipsis})"
    elif isinstance(obj, dict):
        return f"{{{ellipsis}}}"
    elif isinstance(obj, set):
        return f"{{{ellipsis}}}"
    elif isinstance(obj, frozenset):
        return f"frozenset({{{ellipsis}}})"

    # 4. Range type
    elif isinstance(obj, range):
        return f"range({ellipsis})"

    # 5. Array/Buffer types
    elif isinstance(obj, array.array):
        return f"array('{obj.typecode}', [{ellipsis}])"
    elif isinstance(obj, memoryview):
        return _fmt_repr_memoryview(obj, ellipsis)

    # Fallback: return the ellipsis itself if no specific type matched
    return ellipsis


def _fmt_repr_defaultdict(obj: collections.defaultdict, ellipsis: str, opts: FmtOptions) -> str:
    """Helper to format defaultdict with its factory."""
    fq = getattr(opts, "fully_qualified", False)
    factory = obj.default_factory
    if factory is None:
        name = "None"
    else:
        # Use as_instance=False to get 'int' from int, not 'type'
        name = class_name(
            factory, fully_qualified=fq, fully_qualified_builtins=False, as_instance=False
        )
    return f"defaultdict({name}, {{{ellipsis}}})"


def _fmt_repr_memoryview(obj: memoryview, ellipsis: str) -> str:
    """Helper to format memoryview, attempting to identify underlying buffer."""
    try:
        obj_obj = obj.obj
        if isinstance(obj_obj, bytes):
            return f"memoryview(b'{ellipsis}')"
        elif isinstance(obj_obj, bytearray):
            return f"memoryview(bytearray(b'{ellipsis}'))"
    except (AttributeError, ValueError):
        pass

    return f"memoryview({ellipsis})"


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


def _fmt_type_value(type_name: str, obj_repr: str, *, opts: FmtOptions = None) -> str:
    """
    Combine a type name and a repr into a single display token according to style.

    This method does not handle `repr` style and falls back to
    """


def _repr_factory(
    max_items: int | Any = 6,
    max_depth: int | Any = 6,
    max_str: int | Any = 120,
    default: reprlib.Repr | Any = None,
) -> reprlib.Repr:
    """
    Robustly create and configure an instance of `reprlib.Repr`.

    Handles gracefully any type in params. If default Repr has invalid
    attributes (non-int, non-str), they are reset to sensible defaults.

    Args:
        max_items: Maximum number of items for collections
        max_depth: Maximum depth for nested structures
        max_str: Maximum length for strings and other representations
        default: Optional existing `reprlib.Repr` to copy settings from

    Returns:
        Configured `reprlib.Repr` instance with validated attributes
    """
    # Create base Repr
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

    # Validate and fix any corrupted attributes
    # Only override if invalid, preserve valid values from default
    if not isinstance(r.maxlevel, int):
        r.maxlevel = 6
    if not isinstance(r.maxtuple, int):
        r.maxtuple = 6
    if not isinstance(r.maxlist, int):
        r.maxlist = 6
    if not isinstance(r.maxarray, int):
        r.maxarray = 6
    if not isinstance(r.maxdict, int):
        r.maxdict = 6
    if not isinstance(r.maxset, int):
        r.maxset = 6
    if not isinstance(r.maxfrozenset, int):
        r.maxfrozenset = 6
    if not isinstance(r.maxdeque, int):
        r.maxdeque = 6
    if not isinstance(r.maxstring, int):
        r.maxstring = 120
    if not isinstance(r.maxlong, int):
        r.maxlong = 120
    if not isinstance(r.maxother, int):
        r.maxother = 120
    if not isinstance(r.fillvalue, str):
        r.fillvalue = "..."
    if not isinstance(r.indent, (str, int, type(None))):
        r.indent = None

    # Apply explicit overrides
    if isinstance(max_depth, int):
        r.maxlevel = max_depth

    if isinstance(max_items, int):
        r.maxdict = r.maxlist = r.maxtuple = r.maxset = max_items
        r.maxfrozenset = r.maxdeque = r.maxarray = max_items

    if isinstance(max_str, int):
        r.maxstring = r.maxlong = r.maxother = max_str

    return r


def _is_primitive(obj) -> bool:
    """Check if object should be displayed without type label."""
    # We should use type(), not isinstance() here
    return type(obj) in PRIMITIVE_TYPES


def _is_textual(x: Any) -> bool:
    return isinstance(x, (str, bytes, bytearray))
