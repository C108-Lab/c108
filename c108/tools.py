#
# C108 Tools & Utilites
#

# Standard library -----------------------------------------------------------------------------------------------------
from enum import Enum, unique
from collections.abc import Mapping, Sequence
from inspect import stack
from itertools import islice
from typing import Any, Iterable, Iterator, Sequence, Tuple

# Local ----------------------------------------------------------------------------------------------------------------
from c108.abc import class_name


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


# Methods --------------------------------------------------------------------------------------------------------------

def ascii_string(s: str) -> str:
    return ''.join([i if ord(i) < 128 else '_' for i in s])


def method_name():
    return stack()[1][3]


def fmt_mapping(
        mp: Mapping[Any, Any], *,
        style: str = "ascii",
        max_items: int = 8,
        max_repr: int = 120,
        depth: int = 2,
        ellipsis: str | None = None,
) -> str:
    """
    Format a mapping as {key: value} pairs for debugging, logging, and exception messages.

    Intended for robust display of dictionaries and dict-like objects in error contexts.
    Handles problematic keys/values, deep nesting, and large mappings gracefully with
    configurable limits to prevent overwhelming log output.

    Args:
        mp: Any mapping (dict, OrderedDict, etc.) to format.
        style: Display style - "ascii" (safest, default), "unicode-angle", "equal", "paren", "colon".
        max_items: Maximum key-value pairs to show before truncating. Conservative default of 8.
        max_repr: Maximum length of individual key/value reprs before truncation.
        depth: Maximum recursion depth for nested structures. 0 treats nested objects as atomic.
        ellipsis: Custom truncation token. Auto-selected per style if None.

    Returns:
        Formatted string like "{<str: 'key'>: <int: 42>, <str: 'key2'>: <list: [1, 2]>...}".

    Notes:
        - Preserves insertion order for modern dicts.
        - Keys and values are formatted using fmt_value with the same parameters.
        - Nested sequences/mappings are recursively formatted up to 'depth' levels.
        - Broken __repr__ methods in keys or values are handled gracefully.
        - Conservative defaults prevent overwhelming exception messages.

    Examples:
        >>> fmt_mapping({"name": "Alice", "age": 30})
        "{<str: 'name'>: <str: 'Alice'>, <str: 'age'>: <int: 30>}"
        >>> fmt_mapping({i: i**2 for i in range(10)}, max_items=3)
        "{<int: 0>: <int: 0>, <int: 1>: <int: 1>, <int: 2>: <int: 4>...}"

    See Also:
        fmt_value: Format individual values with the same robustness guarantees.
        fmt_sequence: Format sequences/iterables with similar nesting support.
    """
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
            if isinstance(v, Mapping):
                v_str = fmt_mapping(
                    v,
                    style=style,
                    max_items=max_items,
                    max_repr=max_repr,
                    depth=depth - 1,
                    ellipsis=ellipsis,
                )
            elif isinstance(v, Sequence):
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
        max_items: int = 6,
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
    if _fmt_is_textual(seq):
        # Treat text-like as a scalar value, not a sequence of characters
        return fmt_value(seq, style=style, max_repr=max_repr, ellipsis=ellipsis)

    # Check if the input is actually iterable - if not, treat as atomic
    try:
        iter(seq)
    except TypeError:
        # Not iterable - fall back to fmt_value for atomic formatting
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
            if isinstance(x, Mapping):
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
            if isinstance(x, Sequence):
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


def print_method(prefix: str = "------- ",
                 suffix: str = " -------",
                 start: str = "\n\n",
                 end: str = "\n"):
    method_name = stack()[1][3]
    print_title(title=method_name, prefix=prefix, suffix=suffix, start=start, end=end)


def print_title(title,
                prefix: str = "------- ",
                suffix: str = " -------",
                start: str = "\n",
                end: str = "\n"):
    print(f"{start}{prefix}{title}{suffix}{end}")


def dict_get(source: dict, dot_key: str = None, keys: list[str] = None,
             default: Any = "") -> Any:
    """
    Get value from a dict using dot-separated Key for nested values

    Args:
        source   : Source dict
        dot_key  : The key to use as the dot-separated Key for nested values ``root.sub.sub``
        keys     : Keys as list. Overrides dot_key if non-empty
        default  : Default value to return if key not found
        keep_none: Return None for empty keys without values. Returns default if keep_none=False
    """
    if not isinstance(source, (dict, Mapping)):
        raise ValueError(f"Source <dict> | <Mapping> required but found: {type(source)}")
    if not (dot_key or keys):
        raise ValueError("One of <dot_key> or <keys> must be provided")
    if dot_key and keys:
        raise ValueError(f"Only one of <dot_key> or <keys> allowed but found dot_key='{dot_key}' and keys={keys}")
    keys = keys or dot_key.split('.')
    if len(keys) == 1:
        value = source.get(keys[0], default)
    else:
        inner_dict = source.get(keys[0], {})
        value = dict_get(inner_dict, keys=keys[1:], default=default)
    return value if (value is not None) else default


def dict_set(source: dict, dot_key: str = None, keys: list[str] = None, value: Any = None):
    """
    Set value for a dict using dot-separated Key for nested values

    Args:
        source   : Source dict
        dot_key  : The key to use as the dot-separated Key for nested values ``root.sub.sub``
        keys     : Keys as list. Overrides dot_key if non-empty
        value    : New value for key
    """
    if not isinstance(source, (dict, Mapping)):
        raise ValueError(f"Source <dict> | <Mapping> required but found: {type(source)}")
    if not (dot_key or keys):
        raise ValueError("At least one of `key` or `keys` must be provided")
    keys = keys or dot_key.split('.')
    if len(keys) == 1:
        source[keys[0]] = value
    else:
        if keys[0] not in source:
            source[keys[0]] = {}
        dict_set(source[keys[0]], keys=keys[1:], value=value)


def list_get(lst: list | None, index: int | None, default: Any = None) -> Any:
    if not isinstance(lst, (list, type(None))):
        raise TypeError(f"list_get() expected list | None but found: {type(lst)}")
    if not isinstance(index, (int, type(None))):
        raise TypeError(f"list_get() expected int | None but found: {type(index)}")
    return sequence_get(lst, index, default=default)


def listify(x: any, as_type: type = None) -> list[Any]:
    """
    Make list containing single <str> or single non-iterable <any>,
    convert iterable into <list> otherwise
    """
    if isinstance(x, str):
        # First should check for <str> as special type of Iterable
        return [as_type(x)] if as_type else [x]
    elif isinstance(x, Iterable):
        return [as_type(e) for e in x] if as_type else list(x)
    else:
        return [as_type(x)] if as_type else [x]


def obj_as_str(obj: Any, fully_qualified: bool = True, fully_qualified_builtins: bool = False) -> str:
    """
    Returns custom <str> value of object.

    If custom __str__ method not found, overrides the stdlib __str__ with optionally Fully Qualified Class Name
    """
    has_default_str = obj.__str__ == object.__str__
    if not has_default_str:
        as_str = str(obj)
    else:
        cls_name = class_name(obj, fully_qualified=fully_qualified, fully_qualified_builtins=fully_qualified_builtins)
        as_str = f'<class {cls_name}>'
    return as_str


def sequence_get(seq: Sequence | None, index: int | None, default: Any = None) -> Any:
    """
    Get item from sequence or return default if index is None or out of range

    Negative index is supported as in list
    """
    if not isinstance(seq, (Sequence, type(None))):
        raise TypeError(f"sequence_get() expected Sequence | None but found: {type(seq)}")
    if not isinstance(index, (int, type(None))):
        raise TypeError(f"index must be an int | None, got {type(index)}")
    if seq is None or index is None:
        return default
    n = len(seq)
    if -n <= index < n:
        return seq[index]
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


def _fmt_more_token(style: str, more_token: str | None) -> str:
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
