#
# C108 Tools & Utilites
#
import collections
# Standard library -----------------------------------------------------------------------------------------------------
import inspect
from typing import Any, Iterable, Sequence

from c108.abc import class_name


# Methods --------------------------------------------------------------------------------------------------------------

def ascii_string(s: str) -> str:
    return ''.join([i if ord(i) < 128 else '_' for i in s])


def method_name():
    return inspect.stack()[1][3]


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
