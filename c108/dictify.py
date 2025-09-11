"""
C108 Dictify Tools
"""

# Standard library -----------------------------------------------------------------------------------------------------
import collections.abc as abc
import copy
import inspect

from enum import Enum, unique
from typing import Any, Iterable, Callable, Mapping

# Local ----------------------------------------------------------------------------------------------------------------
from .abc import is_builtin, attrs_search, attr_is_property, ObjectInfo
from .utils import class_name


# Classes --------------------------------------------------------------------------------------------------------------

@unique
class HookMode(str, Enum):
    FLEXIBLE = "flexible"
    TO_DICT = "to_dict"
    NONE = "none"


# Methods --------------------------------------------------------------------------------------------------------------

def as_dict(obj: Any,
            inc_class_name: bool = False,
            inc_none_attrs: bool = True,
            inc_none_items: bool = False,
            inc_private: bool = False,
            inc_property: bool = False,
            max_items: int = 10 ** 21,
            fq_names: bool = True,
            recursion_depth=0,
            hook_mode: str = "flexible") -> dict[str, Any]:
    """
    Convert object to dict.

    This method generates a dictionary with attributes and their corresponding values for a given instance or a class.

    Recursion of level N converts all objects from top level 0 deep to level N as dict, but their inner
    attrs keep as-is-values. Builtins are kept as-is on all recursion depth levels.

    Hook processing mode `flexible` calls obj.to_dict() if available or falls back to attribute traversal,
    `to_dict` mode requires obj.to_dict() implemented, `none` mode skips object hooks, uses attribute traversal

    The ``as_dict()`` is a sibling method to ``filter_attrs()``, both of them derive from ``core_dict()`` utility.

    Parameters:
        obj: Any - the object for which attributes need to be extracted
        inc_class_name : Include class name into dict-s created from user objects
        inc_none_attrs : Include attributes with None value from user objects
        inc_none_items : Include items with None value in dictionaries
        inc_private: Include private attributes of user classes
        inc_property: Include instance properties with assigned values, has no effect if obj is a class
        max_items: Length limit for sequence, set and mapping types including list, tuple, dict, set, frozenset
        fq_names: Use Fully Qualified class names
        recursion_depth: int - maximum recursion depth (0 is top-level processing, depth < 0 no processing)
        hook_mode: str - Hook processing mode `flexible|to_dict|none`

    Returns:
        dict[str, Any] - dictionary containing attributes and their values

    See also: ``filter_attrs()``

    Note:
        - recursion_depth < 0  returns obj as is
        - recursion_depth = 0 converts the topmost object to dict with attr names as keys, data values unchanged
        - recursion_depth = N iterates by N levels of recursion on iterables and objects expandable
          with ``as_dict()``
        - inc_property = True: the method skips properties which raise exception
    """

    def __process_obj(obj: Any) -> Any:

        if is_builtin(obj):
            return obj

        # Should convert to dict the topmost obj level but keep its inner objects as-is
        dict_ = _core_to_dict_toplevel(
            obj, inc_class_name=inc_class_name,
            inc_none_attrs=inc_none_attrs,
            inc_private=inc_private, inc_property=inc_property,
            fq_names=fq_names)
        return dict_

    # Should return builtins as is
    if is_builtin(obj):
        return obj

    return core_to_dict(obj,
                        # fn_plain specifies what to do if recursion impossible
                        fn_plain=lambda x: x,
                        # fn_process is applied on final recursion step and on always_filter types
                        fn_process=__process_obj,
                        inc_class_name=inc_class_name,
                        inc_none_attrs=inc_none_attrs,
                        inc_none_items=inc_none_items,
                        inc_private=inc_private,
                        inc_property=inc_property,
                        max_items=max_items,
                        fq_names=fq_names,
                        recursion_depth=recursion_depth,
                        hook_mode=hook_mode)


def core_to_dict(obj: Any,
                 fn_plain: Callable = lambda x: x,
                 fn_process: Callable = lambda x: x,
                 inc_class_name: bool = False,
                 inc_none_attrs: bool = True,
                 inc_none_items: bool = False,
                 inc_private: bool = False,
                 inc_property: bool = False,
                 max_items: int = 14,
                 always_filter: Iterable[type] = tuple(),
                 never_filter: Iterable[type] = tuple(),
                 to_str: Iterable[type] = (),
                 fq_names: bool = True,
                 recursion_depth=0,
                 hook_mode: str = "flexible") -> dict[str, Any]:
    """
    Return Object with simplified representation of data and public attributes
    including builtins and user classes in data format accessible for printing and debugging.

    This is the core method behind the ``as_dict()`` and ``filter_attrs()`` utilities

    Primitive data returned as is, iterables are processed recursively with empty collections handled,
    user classes and instances are returned as data-only dict.

    Parameters:
        obj       : The object for which attributes need to be processed
        fn_plain  : Plain response function to be called when recursion_depth < 0 or never_filter applied
        fn_process: Obj processing function for the case when recursion or size limit is reached
                    or when always_filter applied
        inc_class_name : Include class name into dict-s created from user objects
        inc_none_attrs : Include attributes with None value in dictionaries from user objects
        inc_none_items : Include items with None value in dictionaries
        inc_private    : Include private attributes of user classes
        inc_property   : Include instance properties with assigned values, has no effect if obj is a class
        max_items      : Length limit for sequence, set and mapping types including list, tuple, dict, set, frozenset
        always_filter  : Collection of types to be always filtered. Note that always_filter=[int] >> =[int, boolean]
        never_filter   : Collection of types to skip filtering and preserve original values
        to_str         : User types of attributes to be converted to <str>
        fq_names       : Use Fully Qualified class names
        recursion_depth: Recursion depth for item containers (sequence, set, mapping) and as_dict expandable objects

    Returns:
        dict[str, Any] - dictionary containing attributes and their values

    See also: ``as_dict()``, ``filter_attrs()``, ``print_as_yaml()``

    Note:
        - inc_property = True: the method will anyway skip properties which raise exception
        - recursion_depth < 0  returns obj without filtering
        - recursion_depth = 0 returns unfiltered obj for primitives, object info for iterables and custom classes
        - recursion_depth = N iterates by N levels of recursion on iterables and objects expandable with ``as_dict()``

    """

    # Should copy all args and kwargs immediately in the beginning, keep kwargs only
    # We handle all args of core_to_dict as read-only, so shallow copy is fine
    core_kwargs = copy.copy(dict(locals()))
    del core_kwargs['obj']

    def __core_to_dict(obj, recursion_depth: int):
        kwargs = {**core_kwargs, 'recursion_depth': recursion_depth}
        return core_to_dict(obj, **kwargs)

    def __process_items(obj, recursion_depth: int, from_object: bool = False):

        if recursion_depth < 0:
            raise OverflowError(f"Items object recursion depth out of range: {recursion_depth}")

        if recursion_depth == 0 or len(obj) > max_items:
            return fn_process(obj)

        if isinstance(obj, (list, tuple, set, frozenset, abc.Set)):
            # Other sequence types should be handled individually,
            # see str, bytes, bytearray, memoryview in filter_attrs
            return type(obj)(__core_to_dict(item, recursion_depth=recursion_depth - 1) for item in obj)

        elif isinstance(obj, (dict, abc.Mapping)):
            inc_nones = inc_none_attrs if from_object else inc_none_items
            return {k: __core_to_dict(v, recursion_depth=(recursion_depth - 1)) for k, v in obj.items()
                    if (v is not None) or inc_nones}
        else:
            raise ValueError(f"Items object must be list, tuple, set, frozenset or derived from "
                             f"collections.abc.Set or collections.abc.Mapping but found: {type(obj)}")

    # Process Hook ------------------------------------------

    if hook_mode not in HookMode:
        valid = ", ".join([f"'{v.value}'" for v in HookMode])
        raise ValueError(f"Unknown hook_mode value: {hook_mode!r}. Expected: {valid}")

    dict_ = None
    if hook_mode == HookMode.FLEXIBLE:
        fn = getattr(obj, "to_dict", None)
        if callable(fn):
            dict_ = fn()
    elif hook_mode == HookMode.TO_DICT:
        fn = getattr(obj, "to_dict", None)
        if not callable(fn):
            raise TypeError(f"{type(obj).__name__} must implement to_dict() when hook_mode='to_dict'")
        dict_ = fn()

    # If hook_mode produced a dict, finalize and return
    if dict_ is not None:
        if not isinstance(dict_, Mapping):
            raise TypeError(f"to_dict() must return a Mapping, got {type(dict_).__name__}")

        # Ensure it's mutable for class name injection
        if not isinstance(dict_, dict):
            dict_ = dict(dict_)
        if inc_class_name:
            dict_["_class_name"] = class_name(obj, fully_qualified=fq_names)
        return dict_

    # End of Hook processing -----------------------------------

    if not isinstance(recursion_depth, int):
        raise ValueError(f"Recursion depth must be int but found: {type(recursion_depth)}")

    # depth < 0 should return fn_plain(obj)
    if recursion_depth < 0:
        return fn_plain(obj)

    # Should check and replace always-filtered types
    # Always-filter include large-size builtins and known third-party large types
    builtins_always_filter = [str, bytes, bytearray, memoryview]
    always_filter = [*builtins_always_filter,
                     *list(always_filter)]

    if isinstance(obj, tuple(always_filter)):
        return fn_process(obj)

    if isinstance(obj, tuple(to_str)):
        return _core_to_str(obj, fully_qualified=fq_names)

    # Should check unfiltered types and return original obj as-is
    # Non-filtered include standard types plus never_filter types
    if isinstance(obj, (int, float, bool, complex, type(None), range)):
        return obj
    if isinstance(obj, tuple(never_filter)):
        return obj

    # Should check item-based Instances for 0-level and deeper recursion: list, tuple, set, dict
    if isinstance(obj, (list, tuple, set, frozenset, abc.Set,
                        dict, abc.Mapping)):
        return __process_items(obj, recursion_depth=recursion_depth)

    # Should check user objects for top level processing
    if recursion_depth == 0:
        return fn_process(obj)

    # Should expand any other object 1 level into deep. We arrive here only when recursion_depth > 0
    # Handle inner object: convert obj topmost container to dict but keep inner values as-is
    # then process obtained dict recursively like a std dict
    if recursion_depth < 0:
        raise OverflowError(f"Recursion depth out of range: {recursion_depth}, it must be processed earlier")
    dict_ = _core_to_dict_toplevel(
        obj, inc_class_name=inc_class_name, inc_none_attrs=inc_none_attrs,
        inc_private=inc_private, inc_property=inc_property,
        fq_names=fq_names)
    return __process_items(dict_, recursion_depth=recursion_depth, from_object=True)


def _core_to_dict_toplevel(obj: Any,
                           inc_class_name: bool = False,
                           inc_none_attrs: bool = True,
                           inc_private: bool = False,
                           inc_property: bool = False,
                           fq_names: bool = False) -> dict[str, Any]:
    """
    This method generates a dictionary with attributes and their corresponding values for a given object or class.

    No recursion supported. Method is NOT intended for primitives or inner builtin-items scanning (list, tuple, dict, etc)

    Parameters:
        obj: Any - the object for which attributes need to be extracted
        inc_none_attrs: bool - include attributes with None values
        inc_private: bool - include private attributes
        inc_property: bool - include instance properties with assigned values, has no effect if obj is a class

    Returns:
        dict[str, Any] - dictionary containing attributes and their values

    Note:
        inc_none_attrs: Include attributes with None values
        inc_private: Include private attributes of user classes
        inc_property: Include instance properties with assigned values, has no effect if obj is a class
    """
    dict_ = {}

    attributes = attrs_search(obj, inc_private=inc_private, inc_property=inc_property, inc_none_attrs=inc_none_attrs)
    is_class_or_dataclass = inspect.isclass(obj)

    for attr_name in attributes:
        if attr_name.startswith('__') and attr_name.endswith('__'):
            continue  # Skip dunder methods

        is_obj_property = attr_is_property(attr_name, obj)

        if not is_class_or_dataclass and is_obj_property:
            if not inc_property:
                continue  # Skip properties if inc_property is False

            try:
                attr_value = getattr(obj, attr_name)
            except Exception:
                continue  # Skip if instance property getter raises exception

        else:
            attr_value = getattr(obj, attr_name)

        if callable(attr_value):
            continue  # Skip methods

        dict_[attr_name] = attr_value

    if inc_class_name:
        dict_["_class_name"] = class_name(obj, fully_qualified=fq_names)
    # Sort by Key
    dict_ = dict(sorted(dict_.items()))
    return dict_


def _core_to_str(obj: Any, fully_qualified: bool = True, fully_qualified_builtins: bool = False) -> str:
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


def filter_attrs(obj: Any,
                 inc_class_name: bool = False,
                 inc_none_attrs: bool = True,
                 inc_none_items: bool = False,
                 inc_private: bool = False,
                 inc_property: bool = False,
                 max_bytes: int = 108,
                 max_items: int = 14,
                 max_str_len: int = 108,
                 max_str_prefix: int = 28,
                 always_filter: Iterable[type] = (),
                 never_filter: Iterable[type] = (),
                 to_str: Iterable[type] = (),
                 fq_names: bool = True,
                 recursion_depth=0,
                 hook_mode: str = "flexible") -> dict[str, Any]:
    """
    Return Object with simplified representation of data and public attributes
    including builtins and user classes in data format accessible for printing and debugging.

    This method provides backend to ``print_as_yaml()`` utility

    Primitive data returned as is, iterables are processed recursively with empty collections handled,
    user classes and instances are returned as data-only dict.

    Recursion of level N filters all objects from top level 0 deep to level N. At its deepest level
    attrs are shown as a single primitive value or a stats string.

    The ``filter_attrs()`` is a sibling method to ``as_dict()``, both of them derive from ``core_dict()`` utility

    Parameters:
        obj: The object for which attributes need to be processed
        inc_class_name : Include class name into dict-s created from user objects
        inc_none_attrs : Include attributes with None value in dictionaries from user objects
        inc_none_items : Include items with None value in dictionaries
        inc_private    : Include private attributes of user classes
        inc_property   : Include instance properties with assigned values, has no effect if obj is a class
        max_bytes      : Size limit for byte types, i.e. bytes, bytearray, memoryview
        max_items      : Length limit for sequence, set and mapping types including list, tuple, dict, set, frozenset
        max_str_len    : Length limit for str type
        max_str_prefix : Length limit for long str prefix showed after filtering
        always_filter  : Collection of types to be always filtered. Note that always_filter=[int] >> =[int, boolean]
        never_filter   : Collection of types to skip filtering and preserve original values
        to_str         : User types of attributes to be converted to <str>
        fq_names       : Use Fully Qualified class names
        recursion_depth: Recursion depth for item containers (sequence, set, mapping) and as_dict expandable objects

    Returns:
        dict[str, Any] - dictionary containing attributes and their values

    See also: ``print_as_yaml()``

    Note:
        - recursion_depth < 0  returns obj without filtering
        - recursion_depth = 0 returns unfiltered obj for primitives, object info for iterables and custom classes
        - recursion_depth = N iterates by N levels of recursion on iterables and objects expandable with ``as_dict()``
        - inc_property = True: the method always skips properties which raise exception
    """

    def __object_info(obj,
                      max_bytes: int = max_bytes,
                      max_str_len: int = max_str_len,
                      max_str_prefix: int = max_str_prefix,
                      fq_names: bool = fq_names) -> str:

        _info = ObjectInfo.from_object(obj, fq_name=fq_names).as_str
        # Should check non-recursive large-size builtins
        if isinstance(obj, str):
            stp = obj.strip()
            str_prefix = stp[:max_str_prefix].strip()
            return _info + f" | {str_prefix}..." if len(obj) > max_str_len else obj.strip()
        elif isinstance(obj, (bytes, bytearray, memoryview)):
            # memoryview is similar to bytes but does NOT support .strip() which is present in str and bytes
            #            it can be a view to any object, including bytes
            str_prefix = bytes(obj[:max_str_prefix])
            return _info + f" | {str_prefix}..." if len(obj) > max_bytes else obj
        else:
            return _info

    return core_to_dict(obj,
                        fn_plain=lambda x: x,
                        fn_process=__object_info,
                        inc_class_name=inc_class_name,
                        inc_none_attrs=inc_none_attrs,
                        inc_none_items=inc_none_items,
                        inc_private=inc_private,
                        inc_property=inc_property,
                        max_items=max_items,
                        always_filter=always_filter,
                        never_filter=never_filter,
                        to_str=to_str,
                        fq_names=fq_names,
                        recursion_depth=recursion_depth,
                        hook_mode=hook_mode)
