"""
C108 Core Classes and Class related Tools
"""

# Standard library -----------------------------------------------------------------------------------------------------
import collections
import inspect
import sys

from dataclasses import dataclass
from typing import Any, Set


# Classes --------------------------------------------------------------------------------------------------------------

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

    # 2. Perform structural checks on the class or instance.
    required_attrs = ['size', 'mode', 'format']
    if not all(hasattr(target_cls, attr) for attr in required_attrs):
        return False

    expected_methods = ['save', 'show', 'resize', 'crop']
    if sum(1 for method in expected_methods if
           hasattr(target_cls, method) and callable(getattr(target_cls, method))) < 3:
        return False

    # 3. If it's an instance, perform deeper, value-based checks.
    if not is_class:
        instance = obj
        try:
            size = getattr(instance, 'size')
            if not (isinstance(size, tuple) and len(size) == 2 and
                    isinstance(size[0], int) and isinstance(size[1], int) and
                    size[0] > 0 and size[1] > 0):
                return False
        except (AttributeError, ValueError, TypeError):
            return False

        # Validate the 'mode' attribute's value.
        try:
            mode = getattr(instance, 'mode')
            if not isinstance(mode, str) or not mode:
                return False
        except (AttributeError, TypeError):
            return False

    # If all checks passed, it acts like an image.
    return True


def attrs_eq_names(obj, raise_exception: bool = False, case_sensitive: bool = False) -> bool:
    """
    Check if attribute value equals attr name for each non-callable member of an Object.

    This function iterates through an object's attributes, skipping methods and special
    "dunder" attributes (like __init__). It compares the attribute's name to its value.

    Args:
        obj: The object to inspect.
        raise_exception: If True, raises an AssertionError on the first mismatch.
                         If False, returns False. Defaults to False.
        case_sensitive: If True, the comparison is case-sensitive. If False, it's
                        case-insensitive. Defaults to False.

    Returns:
        True if all checked attributes have values equal to their names,
        otherwise False (unless raise_exception is True).

    Raises:
        AssertionError: If `raise_exception` is True and a mismatch is found.
    """
    # inspect.getmembers() returns all the members of an object in a list of (name, value) pairs.
    for attr_name, attr_value in inspect.getmembers(obj):
        # Skip members that are callable (e.g., methods) or are internal "dunder" attributes.
        if callable(attr_value) or attr_name.startswith('__'):
            continue

        # Prepare the name and value strings for comparison.
        name_to_compare = attr_name
        value_to_compare = str(attr_value)  # Convert value to string for a consistent comparison.

        # Perform the actual comparison based on the case_sensitive flag.
        if case_sensitive:
            are_equal = (name_to_compare == value_to_compare)
        else:
            are_equal = (name_to_compare.lower() == value_to_compare.lower())

        # If they don't match, either raise an exception or return False.
        if not are_equal:
            if raise_exception:
                raise ValueError(
                    f"Attribute '{attr_name}' with value '{attr_value}' does not match its name."
                )
            return False

    # If the loop completes without any mismatches, it means all attributes passed the check.
    return True


def attr_is_property(attr_name: str, obj, try_callable: bool = False):
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


def attrs_search(obj: Any,
                 inc_private: bool = False,
                 inc_property: bool = False,
                 inc_none_attrs: bool = True) -> list[str]:
    """
    Search for data attributes in an object.

    Finds all non-callable attributes in the object that are not special methods (dunder methods).
    Can optionally include private attributes and properties.

    Args:
        obj: The object to inspect for attributes
        inc_private: If True, includes private attributes (starting with '_')
        inc_property: If True, includes properties (both instance and class properties)
        inc_none_attrs: If True, includes attributes with None values

    Returns:
        list[str]: A sorted list of attribute names that match the search criteria.

    Notes:
        - Ignores all callable attributes (methods, functions etc)
        - Ignores special/dunder methods (e.g. __str__)
        - Properties are included only if inc_property=True
        - Returns empty list for built-in types
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

        # Check if this attribute is a property
        if inspect.isclass(obj):
            # When inspecting a class object, look for descriptors on the class itself
            attr_descriptor = getattr(obj, attr_name, None)
        else:
            # When inspecting an instance, look for descriptors on its type
            attr_descriptor = getattr(type(obj), attr_name, None)

        if isinstance(attr_descriptor, property):
            if not inc_property:
                continue  # Skip properties if inc_property=False
            # If inc_property=True, include it (don't skip)

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


def is_builtin(obj: Any) -> bool:
    """
    Return True if the object is either a built-in class or an instance of a built-in class.

    Returns False for:
      - user-defined classes and their instances,
      - functions, methods, and built-in callables (inspect.isfunction/ismethod/isbuiltin),
      - modules,
      - descriptor helpers such as property, staticmethod, and classmethod.

    This definition focuses on core value types provided by Python (e.g., int, str, list, range)
    and excludes meta/descriptor utilities and non-type objects.
    """
    if isinstance(obj, type):  # class objects
        return obj.__module__ == "builtins"

        # Exclude functions/methods/builtins and modules
    if (
            inspect.isfunction(obj)
            or inspect.ismethod(obj)
            or inspect.isbuiltin(obj)
            or inspect.ismodule(obj)
    ):
        return False

    # Exclude descriptor helpers
    if isinstance(obj, (property, staticmethod, classmethod)):
        return False

    return obj.__class__.__module__ == "builtins"


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
