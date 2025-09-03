"""
C108 Core Classes and Class related Tools
"""

# Standard library -----------------------------------------------------------------------------------------------------
import collections
import inspect
import re
import sys

from dataclasses import dataclass, InitVar
from typing import Any, Set


# Classes --------------------------------------------------------------------------------------------------------------

@dataclass
class ObjectInfo:
    type: type
    size: int | list | tuple = ()
    unit: str | list | tuple = ()
    total_bytes: int | None = None

    fq_name: InitVar[bool] = True

    def __post_init__(self, fq_name: bool):
        """
        Validate that 'size' and 'unit' are mutually compatible.
        'size' is a human-facing measure that depends on object kind:
          - numbers: N bytes (sys.getsizeof)
          - str: N chars
          - containers (Sequence/Set/Mapping): N items
          - image-like: (width, height, Mpx)
          - class: N attrs
          - user-defined instance: (N attrs, deep bytes)
        'unit' must match the shape of 'size' (scalar vs tuple/list).
        """
        self._fq_name = fq_name
        if isinstance(self.size, (list | tuple)) and isinstance(self.unit, (list | tuple)) \
                and len(self.size) != len(self.unit):
            raise ValueError("unit and size must be same length if they both are list|tuple")

    @property
    def class_name(self) -> str:
        """Class name derived from 'type'."""
        return class_name(self.type, fully_qualified=self._fq_name, fully_qualified_builtins=False)

    @classmethod
    def from_object(cls, obj: Any, fq_name: bool = True) -> "ObjectInfo":
        """
        Create ObjectInfo from an object.

        'size' is chosen for human readability:
          - int, float, bool, complex: N bytes (sys.getsizeof), unit="bytes"
          - str: N of chars, unit="chars"
          - bytes/bytearray/memoryview: N bytes, unit="bytes"
          - Sequence/Set/Mapping: N items, unit="items"
          - image-like: (width, height, Mpx), unit=("width", "height", "Mpx")
          - Class (type): N attrs, unit="attrs"
          - Instance with attributes: (N attrs, deep bytes), unit=("attrs", "bytes")
          - Other: deep bytes, unit="byte"

        'total_bytes' is the deep size in bytes when meaningful (e.g., containers, user-defined objects).
        For scalar numeric types and bytes-like, total_bytes equals the scalar size (sys.getsizeof or len),
        for strings it's deep bytes via deep_sizeof (platform-dependent), and for classes may be None.
        """
        # Scalars
        if isinstance(obj, (int, float, bool, complex)):
            b = sys.getsizeof(obj)
            return cls(size=b, unit="bytes", total_bytes=b, type=type(obj))
        elif isinstance(obj, str):
            # Human-facing size is chars; deep bytes can be useful to compare memory footprint
            return cls(size=len(obj), unit="chars", total_bytes=deep_sizeof(obj), type=type(obj))
        elif isinstance(obj, (bytes, bytearray, memoryview)):
            n = len(obj)
            return cls(size=n, unit="bytes", total_bytes=n, type=type(obj))

        # Containers
        elif isinstance(obj, (collections.abc.Sequence, collections.abc.Set, collections.abc.Mapping)):
            return cls(size=len(obj), unit="items", total_bytes=deep_sizeof(obj), type=type(obj))

        # Images
        elif acts_like_image(obj):
            width, height = obj.size
            mega_px = width * height / 1e6
            # total_bytes for images: deep_sizeof can be expensive; still useful for consistency
            return cls(size=(width, height, mega_px), unit=("width", "height", "Mpx"),
                       total_bytes=deep_sizeof(obj), type=type(obj))

        # Class objects
        elif type(obj) is type:
            attrs = attrs_search(obj, inc_private=False, inc_property=False)
            return cls(size=len(attrs), unit="attrs", total_bytes=None, type=obj)

        # Instances with attributes
        elif attrs := attrs_search(obj, inc_private=False, inc_property=False):
            bytes_total = deep_sizeof(obj)
            return cls(size=(len(attrs), bytes_total), unit=("attrs", "bytes"),
                       total_bytes=bytes_total, type=type(obj))

        # Other instances with no attrs found
        else:
            bytes_total = deep_sizeof(obj)
            return cls(size=bytes_total, unit="bytes", total_bytes=bytes_total, type=type(obj))

    @property
    def as_str(self) -> str:
        # Heuristic: custom formatting for image-like triplet (width, height, Mpx)
        if isinstance(self.size, (list | tuple)) and isinstance(self.unit, (list | tuple)):
            # Normalize units for comparison
            try:
                unit_lower = tuple(str(u).lower() for u in self.unit)
            except Exception:
                unit_lower = ()
            if len(self.size) == 3 and len(unit_lower) == 3 and unit_lower == ("width", "height", "mpx"):
                width, height, mega_px = self.size
                return f"<{self.class_name}> {width}⨯{height} W⨯H, {round(mega_px, ndigits=3)} Mpx"
            # Generic tuple/list formatting
            size_unit = [f"{s} {u}" for s, u in zip(self.size, self.unit)]
            return f"<{self.class_name}> {', '.join(size_unit)}"

        return f"<{self.class_name}> {self.size} {self.unit}"


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
    Check if attribute value equals attr name for each non-callable member of an Object or Class.

    This function iterates through an object's attributes, skipping methods, private
    attributes, dunder attributes, and name-mangled attributes. It compares the
    attribute's name to its value.

    Args:
        obj: The object to inspect.
        raise_exception: If True, raises an AssertionError on the first mismatch.
                         If False, returns False.
        case_sensitive: If True, the comparison is case-sensitive. If False, it's
                        case-insensitive.

    Returns:
        True if all checked attributes have values equal to their names,
        otherwise False.

    Raises:
        ValueError: If `raise_exception` is True and a mismatch is found.
    """
    # Check all the members of an object in a list of (name, value) pairs.
    for attr_name, attr_value in inspect.getmembers(obj):
        # Skip members that are callable (e.g., methods) or are private/dunder/mangled attributes.
        if callable(attr_value) or attr_name.startswith('_'):
            continue

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

    # Here loop should complete without mismatches
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
    at_names = remove_extra_attrs(at_names,
                                  inc_private=inc_private,
                                  inc_dunder=False,
                                  inc_mangled=False,
                                  mangled_cls_name=cls_name)
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


def deep_sizeof(obj: Any) -> int:
    """
    Calculate deep size of an object including all referenced objects.
    Similar to pympler.deep_sizeof but based on Python stdlib only.

    Returns sys.getsizeof(obj) if obj is a builtin type.

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


def remove_extra_attrs_OLD(attrs: dict | set | list | tuple,
                           cls_name: str = "",
                           inc_dunder: bool = False,
                           inc_private: bool = False,
                           ) -> dict | set | list | tuple:
    """
    Returns a copy of the input collection with mangled, dunder and private attributes removed. 
    
    For dictionaries, removes key-value pairs where the key is a mangled/dunder/private attribute.
    For sets/lists/tuples, removes elements that are mangled/dunder/private attributes.
    
    Arguments:
        attrs (dict | set | list | tuple): The collection from which to remove attributes.
        cls_name (str): The class name to identify mangled attributes containing _ClassName.
        inc_dunder (bool): Keep dunder attributes non-removed.
        inc_private (bool): Keep private attributes non-removed.

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


def remove_extra_attrs(attrs: dict | set | list | tuple,
                       inc_private: bool = False,
                       inc_dunder: bool = False,
                       inc_mangled: bool = False,
                       mangled_cls_name: str | None = None,
                       ) -> dict | set | list | tuple:
    """
    Filter attributes by removing private, dunder, and/or mangled attributes.

    Always returns a copy of the input collection, even if no filtering occurs.

    Arguments:
        attrs: The collection to filter
        inc_private: If True, keep private attributes (starting with single _)
        inc_dunder: If True, keep dunder attributes (__attr__)
        inc_mangled: If True, keep mangled attributes; this flag is ignored if inc_private=False
        mangled_cls_name: Class name to identify mangled attrs. If None, removes all 
                          attributes matching likely mangled pattern _ClassName__attr

    Returns:
        New filtered collection with unwanted attributes removed
    """

    def should_keep_attribute(attr_name: str) -> bool:
        # Check if it's mangled
        if not inc_mangled:
            if mangled_cls_name:
                # Remove attributes containing the specific mangled pattern
                if f"_{mangled_cls_name}__" in attr_name:
                    return False
            else:
                # Remove likely mangled: _SomeClass__attr pattern
                if re.match(r'_[A-Za-z_]\w*__\w+', attr_name):
                    return False

        # Check dunder (must start and end with __)
        if not inc_dunder and attr_name.startswith('__') and attr_name.endswith('__'):
            return False

        # Check private (starts with _ but not dunder)
        if not inc_private and attr_name.startswith('_') and not (
                attr_name.startswith('__') and attr_name.endswith('__')):
            return False

        return True

    # Always create and return a new collection
    if isinstance(attrs, dict):
        return {k: v for k, v in attrs.items() if should_keep_attribute(k)}
    elif isinstance(attrs, (set, list, tuple)):
        return type(attrs)(e for e in attrs if should_keep_attribute(str(e)))
    else:
        raise TypeError('collection must be a dict, set, list, or tuple')
