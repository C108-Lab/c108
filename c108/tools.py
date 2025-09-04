#
# C108 Tools & Utilites
#
# Standard library -----------------------------------------------------------------------------------------------------
import collections
import inspect

from collections.abc import Mapping, Sequence
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
    return inspect.stack()[1][3]


def fmt_value(
        x: Any, *,
        style: str = "unicode-angle",
        max_repr: int = 80,
        ellipsis: str | None = None,
) -> str:
    """
    Format a single value as '⟨Type: value⟩' (or alternative styles), truncating repr for readability.

    The truncation ellipsis uses ellipsis if provided; otherwise it defaults
    to "..." for ASCII style and "…" for other styles.
    """
    t = type(x).__name__
    ellipsis = _fmt_more_token(style, ellipsis)
    r = _fmt_truncate(repr(x), max_repr, ellipsis=ellipsis)
    return _fmt_format_pair(t, r, style)


def fmt_sequence(
        seq: Iterable[Any],
        *,
        style: str = "unicode-angle",
        max_items: int = 10,
        max_repr: int = 60,
        depth: int = 1,
        ellipsis: str | None = None,
) -> str:
    """
    Format a sequence elementwise using fmt_value, preserving literal shape for list/tuple.

    - Treats str/bytes/bytearray as atomic scalars (uses fmt_value).
    - Recurses into nested Sequence/Mapping up to 'depth'.
    - Truncates element reprs via 'max_repr'.
    - Limits elements via 'max_items' and appends a 'more' token when truncated.
    """
    if _fmt_is_textual(seq):
        # Treat text-like as a scalar value
        return fmt_value(seq, style=style, max_repr=max_repr, ellipsis=ellipsis)

    # Choose delimiters by common concrete types; fallback to []
    open_ch, close_ch = "[", "]"
    is_tuple = isinstance(seq, tuple)
    if is_tuple:
        open_ch, close_ch = "(", ")"
    elif isinstance(seq, list):
        open_ch, close_ch = "[", "]"

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
    # Singleton tuple needs a trailing comma
    tail = "," if is_tuple and len(parts) == 1 and not more else ""
    return f"{open_ch}" + ", ".join(parts) + more + f"{tail}{close_ch}"


def fmt_mapping(
        mp: Mapping[Any, Any],
        *,
        style: str = "unicode-angle",
        max_items: int = 10,
        max_repr: int = 60,
        depth: int = 1,
        ellipsis: str | None = None,
) -> str:
    """
    Format a mapping as {key: value}, with key/value rendered via fmt_value or recursive formatters.

    - Treats str/bytes/bytearray as atomic scalars (uses fmt_value).
    - Recurses into nested Sequence/Mapping up to 'depth'.
    - Truncates reprs via 'max_repr'.
    - Limits pairs via 'max_items' and appends a 'more' token when truncated.
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


def print_method(prefix: str = "------- ",
                 suffix: str = " -------",
                 start: str = "\n\n",
                 end: str = "\n"):
    method_name = inspect.stack()[1][3]
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
    if not isinstance(source, (dict, collections.abc.Mapping)):
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
    if not isinstance(source, (dict, collections.abc.Mapping)):
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
    """Truncate s to at most max_len characters, appending ellipsis if truncated."""
    if max_len <= 0:
        return ""
    if len(s) <= max_len:
        return s
    if max_len <= len(ellipsis):
        return ellipsis[:max_len]
    return s[: max_len - len(ellipsis)] + ellipsis


def _fmt_format_pair(type_name: str, value_repr: str, style: str) -> str:
    """Combine a type name and a repr into a single display token according to style."""
    if style == FmtStyle.ASCII:
        return f"<{type_name}: {value_repr.replace('>', '\\>')}>"
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
