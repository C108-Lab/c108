"""
C108 Core Classes and Class related Tools
"""

# Standard library -----------------------------------------------------------------------------------------------------
import collections
import copy
import inspect
import sys

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Iterable, Callable, Set


# Classes --------------------------------------------------------------------------------------------------------------

class BiDirectionalMap:
    """
    A map that provides bidirectional lookup, ensuring both keys and values are unique.
    """

    def __init__(self, initial_map: dict = None):
        self._forward_map = {}  # key -> value
        self._backward_map = {}  # value -> key

        if initial_map:
            for key, value in initial_map.items():
                self.add(key, value)  # Use the add method to enforce uniqueness

    def add(self, key, value):
        """
        Adds a key-value pair to the map. Raises ValueError if key or value already exists.
        """
        if key in self._forward_map:
            raise ValueError(f"Key '{key}' already exists in the map, mapping to '{self._forward_map[key]}'.")
        if value in self._backward_map:
            raise ValueError(
                f"Value '{value}' already exists in the map, mapped from '{self._backward_map[value]}'. Values must be unique.")

        self._forward_map[key] = value
        self._backward_map[value] = key

    def get_value(self, key):
        """
        Looks up a value by its key.
        """
        return self._forward_map[key]

    def get_key(self, value):
        """
        Looks up a key by its value.
        """
        return self._backward_map[value]

    def __getitem__(self, key):
        """Allows dictionary-like access for key to value."""
        return self.get_value(key)

    def __contains__(self, item):
        """Checks if a key or value exists in the map."""
        return item in self._forward_map or item in self._backward_map

    def __len__(self):
        return len(self._forward_map)

    def __repr__(self):
        return f"BiDirectionalMap({self._forward_map})"

    def keys(self):
        return self._forward_map.keys()

    def values(self):
        return self._forward_map.values()

    def items(self):
        return self._forward_map.items()

    # You can add a deletion method if needed, being careful to update both maps
    def delete(self, key):
        if key not in self._forward_map:
            raise KeyError(f"Key '{key}' not found in map.")
        value_to_delete = self._forward_map.pop(key)
        del self._backward_map[value_to_delete]


@dataclass
class ObjectInfo:
    type: type
    class_name: str = ""
    size: int | list | tuple = ()
    unit: str | list | tuple = ()

    def __post_init__(self):
        """
        Checks if the 'size' and 'unit' attributes are mutually compatible
        """
        if isinstance(self.size, (list | tuple)) and isinstance(self.unit, (list | tuple)) \
                and len(self.size) != len(self.unit):
            raise ValueError("unit and size must be same length if they both are list|tuple")

        self.class_name = self.class_name or class_name(
            self.type, fully_qualified=True, fully_qualified_builtins=False)

    @classmethod
    def from_object(cls, obj: Any, fully_qualified: bool = True):
        """
        Create ObjectInfo instance from an object with size calculated as number of elements
        for iterables and approximate count of bytes for non-iterables.

        For basic Python data types size we use len() and sys.getsizeof() to determine object size
        For total byte count estimates of user defined classes the ``deep_sizeof`` is used.

        Returns:
            - int, float, bool, complex: N of bytes
            - str: N of chars
            - list, tuple, dict, set, frozenset, range: N of items
            - bytes, bytearray, memoryview: N of bytes
            - PIL.Image.Image: N of bytes
            - Class or Instance: N attrs with M bytes
        """

        if isinstance(obj, (
                int, float, bool, complex)):
            return cls(size=sys.getsizeof(obj), unit="bytes",
                       type=type(obj), class_name=class_name(obj, fully_qualified=fully_qualified))
        elif isinstance(obj, str):
            return cls(size=len(obj), unit="chars",
                       type=type(obj), class_name=class_name(obj, fully_qualified=fully_qualified))
        elif isinstance(obj, (
                bytes, bytearray, memoryview)):
            return cls(size=len(obj), unit="bytes",
                       type=type(obj), class_name=class_name(obj, fully_qualified=fully_qualified))
        elif isinstance(obj, (
                # Include list, tuple, dict, set, frozenset and derived classes
                collections.abc.Sequence, collections.abc.Set, collections.abc.Mapping)):
            return cls(size=len(obj), unit="items",
                       type=type(obj), class_name=class_name(obj, fully_qualified=fully_qualified))
        elif acts_like_image(obj):
            width, height = obj.size
            mega_px = width * height / 1e6
            return cls(size=(width, height, mega_px), unit=("width", "height", "Mpx"),
                       type=type(obj), class_name=class_name(obj, fully_qualified=fully_qualified))
        elif type(obj) is type:
            attrs = attrs_search(obj, inc_private=False, inc_property=False)
            # NOTE: self.type assignment for classes is diff then for instances
            return cls(size=len(attrs), unit="attrs",
                       type=obj, class_name=class_name(obj, fully_qualified=fully_qualified))
        elif attrs := attrs_search(obj, inc_private=False, inc_property=False):
            return cls(size=(len(attrs), deep_sizeof(obj)),
                       unit=("attrs", "bytes"),
                       type=type(obj), class_name=class_name(obj, fully_qualified=fully_qualified))
        else:
            # Other objects for which no attributes were found
            # Example: if obj is a class without class attributes it has no data
            return cls(size=deep_sizeof(obj), unit="byte",
                       type=type(obj), class_name=class_name(obj, fully_qualified=fully_qualified))

    @property
    def as_str(self) -> str:
        if acts_like_image(self.type):
            width, height, mega_px = self.size
            _, _, mega_px_unit = self.unit
            return f"<{self.class_name}> {width}⨯{height} W⨯H, {round(mega_px, ndigits=3)} {mega_px_unit}"

        elif isinstance(self.size, (list | tuple)):
            size_unit = [f"{s} {u}" for s, u in zip(self.size, self.unit)]
            return f"<{self.class_name}> {', '.join(size_unit)}"

        return f"<{self.class_name}> {self.size} {self.unit}"

    def to_str(self: Any, title: str = "") -> str:
        print("WANING: ObjectInfo.to_str() is deprecated, use ObjectInfo.as_str() instead", file=sys.stderr)
        return self.as_str


# Methods --------------------------------------------------------------------------------------------------------------

def acts_like_image(obj: Any) -> bool:
    """
    Detects if an object or its type behaves like a PIL.Image.Image.

    This function uses duck typing to check for attributes and methods
    common to PIL.Image.Image, allowing it to work on both instances
    and class types without importing PIL.

    - When given a **type**, it checks for the presence of required
      attributes and methods (e.g., does the class have a 'size' property
      and a 'save' method?).
    - When given an **instance**, it performs the same structural checks
      and also validates the *values* of the attributes (e.g., is '.size'
      a tuple of two positive integers?).

    Args:
        obj: The object instance or the class type to check.

    Returns:
        True if the object or type appears to be image-like, False otherwise.
    """
    is_class = isinstance(obj, type)
    target_cls = obj if is_class else type(obj)

    # 1. Check the class name (a quick, efficient filter).
    if 'Image' not in target_cls.__name__:
        return False

    # 2. Perform structural checks on the class.
    # These checks work for both instances (via their type) and classes directly.
    required_attrs = ['size', 'mode', 'format']
    if not all(hasattr(target_cls, attr) for attr in required_attrs):
        return False

    expected_methods = ['save', 'show', 'resize', 'crop']
    if sum(1 for method in expected_methods if
           hasattr(target_cls, method) and callable(getattr(target_cls, method))) < 3:
        return False

    # 3. If it's an instance, perform deeper, value-based checks.
    # This block is skipped if we were only given a class type.
    if not is_class:
        instance = obj
        # Validate the 'size' attribute's value.
        try:
            size = getattr(instance, 'size')
            if not (isinstance(size, tuple) and len(size) == 2 and
                    isinstance(size[0], int) and isinstance(size[1], int) and
                    size[0] > 0 and size[1] > 0):
                return False
        except (AttributeError, ValueError, TypeError):
            return False  # Fails if .size isn't accessible or has the wrong format.

        # Validate the 'mode' attribute's value.
        try:
            mode = getattr(instance, 'mode')
            if not isinstance(mode, str) or not mode:
                return False
        except (AttributeError, TypeError):
            return False

    # If all relevant checks passed, it acts like an image.
    return True


def as_dict(obj: Any,
            inc_class_name: bool = False,
            inc_none_attrs: bool = True,
            inc_none_items: bool = False,
            inc_private: bool = False,
            inc_property: bool = False,
            max_items: int = 10 ** 21,
            fq_names: bool = True,
            recursion_depth=0) -> dict[str, Any]:
    """
    Convert object to dict.

    This method generates a dictionary with attributes and their corresponding values for a given object or a class.

    Recursion of level N converts all objects from top level 0 deep to level N as dict, but their inner
    attrs keep as-is-values. Builtins are kept as-is on all recursion depth levels.

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

    Returns:
        dict[str, Any] - dictionary containing attributes and their values

    See also: ``as_dict()`` and ``filter_attrs()``

    Note:
        - Items with None values are deleted from dict representation by default
        - recursion_depth < 0  returns obj as is
        - recursion_depth = 0 converts topmost object to dict with attr names as keys, data values unchanged
        - recursion_depth = N iterates by N levels of recursion on iterables and objects expandable
          with ``as_dict()``
        - inc_property = True: the method will anyway skip properties which raise exception
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

    # Should convert builtin classes and their instances
    # to an empty dict
    if is_builtin(obj):
        return obj

    # obj._as_dict method (if found) should override further processing
    if hasattr(obj, '_as_dict'):
        dict_ = obj._as_dict() if callable(obj._as_dict) else obj._as_dict
        if inc_class_name:
            dict_["_class_name"] = class_name(obj, fully_qualified=fq_names)
        return dict_

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
                        recursion_depth=recursion_depth)


def attrs_eq_names(obj, raise_exception: bool = False) -> bool:
    """
    Check if attribute value equals attr name for each non-callable member of an Object, object attributes, and fields
    """
    attr = as_dict(obj, inc_private=False, inc_property=False, recursion_depth=0)
    for key in attr:
        if attr[key] != key:
            if raise_exception:
                raise ValueError(
                    f"Attribute '{key}' must have str value "
                    f"equal to its name '{key}' but got {attr[key]}")
            return False
    # We should reach here if no condition violations found,
    # or if attr = {}
    return True


def attrs_search(obj: Any,
                 inc_private: bool = False,
                 inc_property: bool = False,
                 inc_none_attrs: bool = True) -> list[
    str]:
    """
    Search for data attributes in object.
    Optionally includes privates and properties that do not raise exception.
    """

    def safe_getattr(obj, attr, default=None):
        try:
            return getattr(obj, attr)
        except Exception:
            return default

    ignored_types = (int, float, bool, str, list, tuple,
                     dict, set, frozenset, bytes, bytearray, complex, memoryview, range)

    # Should return empty search list for builtin-s
    if (inspect.isclass(obj) and obj in ignored_types) or isinstance(obj, tuple(ignored_types)):
        return []

    at_names = set()
    members = ((attr, safe_getattr(obj, attr, None)) for attr in dir(obj))

    for attr_name, attr_value in members:
        if attr_value is None and not inc_none_attrs:
            continue

        if not inc_property and isinstance(getattr(type(obj), attr_name, None), property):
            continue  # Include Instance properties if inc_property=True

        if isinstance(attr_value, property):
            continue  # Skip Class properties always

        if callable(attr_value) or (attr_name.startswith('__') and attr_name.endswith('__')):
            continue  # Skip callables and dunder attrs always
        at_names |= {attr_name}
    cls_name = class_name(obj, fully_qualified=False)
    at_names = remove_extra_attrs(at_names, inc_private=inc_private, inc_dunder=False, cls_name=cls_name)
    return sorted(at_names)


def class_name(obj, fully_qualified=True, fully_qualified_builtins=False,
               start: str = "", end: str = ""):
    """Get the class name from the object. Optionally get the fully qualified class name

    Parameters:
        obj: An object or a class
        fully_qualified (bool): If true, returns the fully qualified name for non builtin objects or classes
        fully_qualified_builtins (bool): If true, returns the fully qualified name for builtin objects or classes

    Returns:
        str: The class name
    """

    # Check if the obj is an instance or a class
    obj_is_class = isinstance(obj, type)

    # If obj is an instance
    if not obj_is_class:
        # Get the class of the instance
        cls = obj.__class__
    else:
        cls = obj

    # If class is builtin
    if cls.__module__ == 'builtins':
        if fully_qualified_builtins:
            # Return the fully qualified name
            return start + cls.__module__ + "." + cls.__name__ + end
        else:
            # Return only the class name
            return start + cls.__name__ + end
    else:
        if fully_qualified:
            # Return the fully qualified name
            return start + cls.__module__ + "." + cls.__name__ + end
        else:
            # Return only the class name
            return start + cls.__name__ + end


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
                 recursion_depth=0
                 ):
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

    # Should get all args and kwargs immediately in the beginning, keep kwargs only
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

        if isinstance(obj, (list, tuple, set, frozenset, collections.abc.Set)):
            # Other sequence types should be handled individually,
            # see str, bytes, bytearray, memoryview in filter_attrs
            return type(obj)(__core_to_dict(item, recursion_depth=recursion_depth - 1) for item in obj)

        elif isinstance(obj, (dict, collections.abc.Mapping)):
            inc_nones = inc_none_attrs if from_object else inc_none_items
            return {k: __core_to_dict(v, recursion_depth=(recursion_depth - 1)) for k, v in obj.items()
                    if (v is not None) or inc_nones}
        else:
            raise ValueError(f"Items object must be list, tuple, set, frozenset or derived from "
                             f"abc.Set or abc.Mapping but found: {type(obj)}")

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
        return obj_as_str(obj, fully_qualified=fq_names)

    # Should check unfiltered types and return original obj as-is
    # Non-filtered include standard types plus never_filter types
    if isinstance(obj, (int, float, bool, complex, type(None), range)):
        return obj
    if isinstance(obj, tuple(never_filter)):
        return obj

    # Should check item-based Instances for 0-level and deeper recursion: list, tuple, set, dict
    if isinstance(obj, (list, tuple, set, frozenset, collections.abc.Set,
                        dict, collections.abc.Mapping)):
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

        is_obj_property = is_property(attr_name, obj)

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


def is_builtin(obj):
    if isinstance(obj, type):  # This will be true for class objects
        return obj.__module__ == "builtins"
    else:
        return obj.__class__.__module__ == "builtins"


def is_property(attr_name: str, obj, try_callable: bool = False):
    """
    Check if a given attribute is a property of a class or an object.

    Parameters:
        attr_name (str): The name of the attribute to check.
        obj: The class or object to check the attribute in.
        try_callable (bool, optional): Whether to try calling the property's getter function. Defaults to False.

    Returns:
    bool: True if the attribute is a property, False otherwise.

    Note:
        - Flag try_callable=True on a class/dataclass will always return False from this function.
        - Flag try_callable=False on an instance returns True if attribute calculation returns
          a value and does not raise an exception.
    """
    if inspect.isclass(obj):
        if try_callable:
            return False
        attr = obj.__dict__.get(attr_name, None)
        is_property = isinstance(attr, property)

    else:
        attr = getattr(type(obj), attr_name, None)
        is_property = isinstance(attr, property)
        if is_property and try_callable:
            try:
                attr.fget(obj)  # on successful call, returns True
            except Exception:  # if an error occurs when trying to call
                return False

    return is_property


def deep_sizeof(obj: Any) -> int:
    """
    Calculate deep size of an object including all referenced objects.
    Similar to pympler.deep_sizeof but based on Python stdlib only.

    Args:
        obj: Any Python object to measure

    Returns:
        Total size in bytes including all referenced objects
    """
    return _deep_sizeof_recursive(obj, set())


def _deep_sizeof_recursive(obj: Any, seen: Set[int]) -> int:
    """
    Recursive implementation for deep_sizeof calculation with cycle detection.

    Args:
        obj: Object to measure
        seen: Set of already-seen object IDs to prevent cycles

    Returns:
        Size in bytes
    """
    obj_id = id(obj)
    if obj_id in seen:
        return 0  # Already counted

    seen.add(obj_id)
    size = sys.getsizeof(obj)

    # Handle different object types
    if isinstance(obj, dict):
        size += sum(_deep_sizeof_recursive(k, seen) + _deep_sizeof_recursive(v, seen)
                    for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(_deep_sizeof_recursive(item, seen) for item in obj)
    elif hasattr(obj, '__dict__'):
        # User-defined objects with instance attributes
        size += _deep_sizeof_recursive(obj.__dict__, seen)
    elif hasattr(obj, '__slots__'):
        # Objects with __slots__
        for slot in obj.__slots__:
            if hasattr(obj, slot):
                size += _deep_sizeof_recursive(getattr(obj, slot), seen)

    return size


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
                 recursion_depth=0):
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

        _info = ObjectInfo.from_object(obj, fully_qualified=fq_names).as_str
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
                        recursion_depth=recursion_depth)


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


def remove_extra_attrs(attrs: dict | set | list | tuple,
                       inc_private: bool = False, inc_dunder: bool = False,
                       cls_name: str = ""):
    """
    Removes mangled, dunder (optionally) and private (optionally) attributes from a collection of attrs or names.

    For dictionaries, it removes key-value pairs where the key is a mangled, dunder or private attribute.
    For sets, lists and tuples it removes elements that are mangled, dunder, or private attributes.

    Arguments:
        attrs (dict | set | list | tuple): The collection from which to remove attributes.
        inc_private (bool): Keep private attributes, no removal.
        inc_dunder (bool): Keep dunder attributes non-removed.
        cls_name (str): The class name to identify mangled attributes containing _ClassName.

    Returns:
        (dict | set | list | tuple): The collection with mangled, dunder (optional), and private (optional) attributes removed.
    """

    if inc_private and inc_dunder and not cls_name:
        return attrs
    mangled_name = f"_{cls_name}"
    rm_private = not inc_private
    rm_dunder = not inc_dunder

    if isinstance(attrs, dict):
        return {k: v for k, v in attrs.items() if
                (not (k.startswith('_') and rm_private) or k.startswith('__')) and mangled_name not in k and not (
                        k.startswith('__') and rm_dunder)}
    elif isinstance(attrs, (set, list, tuple)):
        return type(attrs)(e for e in attrs if (
                not (e.startswith('_') and rm_private) or e.startswith('__')) and mangled_name not in e and not (
                e.startswith('__') and rm_dunder))
    else:
        raise TypeError('collection must be a dict, set, list, or tuple')


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
