"""
C108 Classes and Class related Tools
"""

# Standard library -----------------------------------------------------------------------------------------------------
import collections.abc as abc
import inspect
import re
import sys

from dataclasses import dataclass, InitVar
from typing import Any, Set, Sequence

# Local ----------------------------------------------------------------------------------------------------------------
from .tools import fmt_any
from .utils import class_name


# Classes --------------------------------------------------------------------------------------------------------------

# TODO make deep_size calculation optional, False by default for ObjectInfo
@dataclass
class ObjectInfo:
    """
    Summarize an object with its type, size, unit, and human-friendly presentation.

    Provides a lightweight summary of an object, including its type, a human-oriented
    size measure, unit labels, and optionally a deep byte size.

    Attributes:
        type (type): The object's type (class for instances, or the type object itself).
        size (int | float | Sequence[int|float]): Human-oriented measure:
            - numbers, bytes-like: int (bytes)
            - str: int (characters)
            - containers (Sequence/Set/Mapping): int (items_count)
            - image-like: tuple[int, int, float] (width, height, megapixels)
            - class objects: int (attrs_count)
            - user-defined instances with attrs: tuple[int, int] (attrs_count, deep)
        unit (str | Sequence[str]): Unit label(s) matching the structure of size.
            Note: a plain str is treated as a scalar unit, not a sequence.
        deep_size (int | None): Deep size in bytes (like pympler.deep_sizeof) computed
            via c108.abc.deep_sizeof() function for most objects; None for classes.

    Init vars:
        fq_name (bool): If true, class_name is fully qualified; builtins are never fully qualified.

    Raises:
        ValueError: If size and unit are sequences of different lengths.
    """
    type: type
    size: int | float | Sequence[int | float] = ()
    unit: str | Sequence[str] = ()
    deep_size: int | None = None

    fq_name: InitVar[bool] = True

    def __post_init__(self, fq_name: bool):
        """
        Post-initialization validation and options.
        """
        self._fq_name = fq_name

        # Only validate runtime logic constraints
        if isinstance(self.size, abc.Sequence) and not isinstance(self.size, (str, bytes, bytearray)):
            if isinstance(self.unit, abc.Sequence) and not isinstance(self.unit, (str, bytes, bytearray)):
                if len(self.size) != len(self.unit):
                    raise ValueError(
                        f"size and unit must be same length: "
                        f"len(size)={len(self.size)}, len(unit)={len(self.unit)}"
                    )

    @classmethod
    def from_object(cls, obj: Any, fq_name: bool = True) -> "ObjectInfo":
        """
        Build an ObjectInfo summary of 'obj'.

        Heuristics according to 'obj' type:
          - Numbers: size=N bytes (shallow), unit="bytes".
          - str: size=N chars, unit="chars".
          - bytes/bytearray/memoryview: size=N bytes, unit="bytes".
          - Sequence/Set/Mapping: size=N items, unit="items".
          - Image-like: size=(width, height, Mpx), unit=("width","height","Mpx").
          - Class (type): size=N attrs, unit="attrs"; deep_size=None.
          - Instance with attrs: size=(N attrs, deep bytes), unit=("attrs","bytes").
          - Other/no-attrs: size=deep bytes, unit="bytes"
          - Any obj: deep_size via c108.abc.deep_sizeof(); None if obj is a class.

        Parameters:
          - obj: object to summarize.
          - fq_name: whether class_name should be fully qualified for non-builtin types.

        Returns:
          - ObjectInfo with populated size, unit, deep_size, and type.
        """
        # Scalars
        if isinstance(obj, (int, float, bool, complex)):
            b = sys.getsizeof(obj)  # shallow bytes, used for human-facing size
            return cls(size=b, unit="bytes", deep_size=deep_sizeof(obj), type=type(obj), fq_name=fq_name)
        elif isinstance(obj, str):
            # Human-facing size is chars; deep bytes can be useful to compare memory footprint
            return cls(size=len(obj), unit="chars", deep_size=deep_sizeof(obj), type=type(obj), fq_name=fq_name)
        elif isinstance(obj, (bytes, bytearray, memoryview)):
            n = len(obj)
            return cls(size=n, unit="bytes", deep_size=deep_sizeof(obj), type=type(obj), fq_name=fq_name)

        # Containers
        elif isinstance(obj, (abc.Sequence, abc.Set, abc.Mapping)):
            return cls(size=len(obj), unit="items", deep_size=deep_sizeof(obj), type=type(obj), fq_name=fq_name)

        # Images
        elif acts_like_image(obj):
            width, height = obj.size
            mega_px = width * height / 1e6
            return cls(
                size=(width, height, mega_px),
                unit=("width", "height", "Mpx"),
                deep_size=deep_sizeof(obj),
                type=type(obj),
                fq_name=fq_name,
            )

        # Class objects
        elif type(obj) is type:
            attrs = attrs_search(obj, include_private=False, include_property=False)
            return cls(size=len(attrs), unit="attrs", deep_size=None, type=obj, fq_name=fq_name)

        # Instances with attributes
        elif attrs := attrs_search(obj, include_private=False, include_property=False):
            bytes_total = deep_sizeof(obj)
            return cls(size=(len(attrs), bytes_total), unit=("attrs", "bytes"),
                       deep_size=bytes_total, type=type(obj), fq_name=fq_name)

        # Other instances with no attrs found
        else:
            bytes_total = deep_sizeof(obj)
            return cls(size=bytes_total, unit="bytes", deep_size=bytes_total, type=type(obj), fq_name=fq_name)

    @property
    def as_str(self) -> str:
        """
        Human-readable one-line summary.

        Examples:
          - "<int> 28 bytes"
          - "<str> 11 chars"
          - "<list> 3 items"
          - "<PIL.Image.Image> 640⨯480 W⨯H, 0.307 Mpx"
          - "<MyClass> 4 attrs, 1024 bytes"
          - "<Other/no-attrs> 1024 bytes"
        """
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

    @property
    def class_name(self) -> str:
        """Return a display name for 'type' (fully qualified for non-builtin types if enabled)."""
        return class_name(self.type, fully_qualified=self._fq_name, fully_qualified_builtins=False)


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

    This function iterates through an object's attributes, skipping methods, properties, private
    attributes, dunder attributes, and name-mangled attributes. It compares the
    attribute's name to its value.

    Intended use:
    - Quick validation of configuration-like objects, simple enums, small data holders,
      or classes where attributes are expected to mirror their own names (e.g., constants).

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

        # Skip properties by checking if the attribute is a property on the class
        if hasattr(obj.__class__, attr_name) and isinstance(getattr(obj.__class__, attr_name), property):
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
                    f"attribute name '{attr_name}' does not match its value '{attr_value!r}'."
                )
            return False

    # Here loop should complete without mismatches
    return True


def attr_is_property(attr_name: str, obj, try_callable: bool = False) -> bool:
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
                 include_private: bool = False,
                 include_property: bool = False,
                 include_none_attrs: bool = True,
                 include_none_properties: bool = True,
                 sort: bool = False) -> list[str]:
    """
    Search for data attributes in an object.

    Finds all non-callable attributes in the object that are not special methods (dunder methods).
    Can optionally include private attributes and properties.

    Args:
        obj: The object to inspect for attributes
        include_private: If True, includes private attributes (starting with '_')
        include_property: If True, includes properties (both instance and class properties)
        include_none_attrs: If True, includes regular attributes with None values
        include_none_properties: If True, includes properties that return None.
                                Only applies when include_property=True
        sort: If True, sorts the attribute names alphabetically. Default False preserves
              the order from dir() which follows a specific pattern (special, private, public)

    Returns:
        list[str]: A list of attribute names that match the search criteria.

    Notes:
        - Ignores all callable attributes (methods, functions, etc.)
        - Ignores special/dunder methods (e.g. __str__, __init__)
        - Properties are included only if include_property=True
        - Returns empty list if obj is of built-in primitive type
        - By default, preserves the natural order from dir() unless sort=True
        - Handles property access errors gracefully by skipping problematic properties

    Examples:
        >>> class MyClass:
        ...     public = 1
        ...     _private = 2
        ...     @property
        ...     def prop(self):
        ...         return 3
        >>> obj = MyClass()
        >>> attrs_search(obj)
        ['public']
        >>> attrs_search(obj, include_private=True)
        ['_private', 'public']
        >>> attrs_search(obj, include_property=True)
        ['prop', 'public']
        >>> attrs_search(obj, sort=True)
        ['public']
    """

    def _safe_getattr(obj, attr, default=None):
        """Safely get attribute value, returning default on any error."""
        try:
            return getattr(obj, attr)
        except Exception:
            return default

    # Built-in types that should return empty results
    ignored_types = (int, float, bool, str, list, tuple,
                     dict, set, frozenset, bytes, bytearray, complex, memoryview, range)

    # Return empty list for built-in primitive types
    if (inspect.isclass(obj) and obj in ignored_types) or isinstance(obj, ignored_types):
        return []

    # Return empty list for objects that don't support dir() or for None
    try:
        attr_list = dir(obj)
    except (TypeError, AttributeError):
        return []

    attr_names = []
    seen = set()  # Track what we've added to avoid duplicates

    # Iterate over attributes returned by dir()
    for attr_name in attr_list:
        # Skip if already processed (shouldn't happen with dir(), but defensive)
        if attr_name in seen:
            continue

        # Always skip dunder attributes
        if attr_name.startswith('__') and attr_name.endswith('__'):
            continue

        # Check if this attribute is a property before getting its value
        is_property = False
        if inspect.isclass(obj):
            # When inspecting a class object, look for descriptors on the class itself
            attr_descriptor = getattr(obj, attr_name, None)
        else:
            # When inspecting an instance, look for descriptors on its type
            attr_descriptor = getattr(type(obj), attr_name, None)

        if isinstance(attr_descriptor, property):
            is_property = True
            if not include_property:
                continue  # Skip properties if include_property=False

        # Get the attribute value
        attr_value = _safe_getattr(obj, attr_name, None)

        # Handle None values differently for properties vs regular attributes
        if attr_value is None:
            if is_property and not include_none_properties:
                continue
            elif not is_property and not include_none_attrs:
                continue

        # Skip callables (methods, functions, etc.) but not properties
        # Properties are technically callable but we've already handled them above
        if not is_property and callable(attr_value):
            continue

        attr_names.append(attr_name)
        seen.add(attr_name)

    # Filter out private, dunder, and mangled attributes
    cls_name = class_name(obj, fully_qualified=False)
    attr_names = _attrs_remove_extra(attr_names,
                                     include_private=include_private,
                                     include_dunder=False,
                                     include_mangled=False,
                                     mangled_cls_name=cls_name)

    # Only sort if explicitly requested
    if sort:
        attr_names = sorted(attr_names)

    return attr_names


def _attrs_remove_extra(attrs: list[str],
                        include_private: bool = False,
                        include_dunder: bool = False,
                        include_mangled: bool = False,
                        mangled_cls_name: str | None = None,
                        ) -> list[str]:
    """
    Filter attribute names by removing private, dunder, and/or mangled attributes.

    Always returns a new list, even if no filtering occurs.

    Args:
        attrs: The list of attribute names to filter
        include_private: If True, keep private attributes (starting with single _)
        include_dunder: If True, keep dunder attributes (__attr__)
        include_mangled: If True, keep mangled attributes; this flag is ignored if include_private=False
        mangled_cls_name: Class name to identify mangled attrs. If None, removes all
                          attributes matching likely mangled pattern _ClassName__attr

    Returns:
        list[str]: New filtered list with unwanted attributes removed, preserving order.

    Notes:
        - Mangled attributes follow the pattern _ClassName__attrname
        - Dunder attributes start and end with double underscores
        - Private attributes start with single underscore but are not dunder or mangled
    """

    def _should_keep_attribute(attr_name: str) -> bool:
        """Determine if an attribute should be kept based on filtering rules."""
        # Check if it's mangled (only if we should filter them out)
        if not include_mangled:
            if mangled_cls_name:
                # Remove attributes containing the specific mangled pattern
                if f"_{mangled_cls_name}__" in attr_name:
                    return False
            else:
                # Remove likely mangled: _SomeClass__attr pattern
                if re.match(r'_[A-Z][A-Za-z0-9_]*__\w+', attr_name):
                    return False

        # Check dunder (must start and end with __)
        if not include_dunder and attr_name.startswith('__') and attr_name.endswith('__'):
            return False

        # Check private (starts with _ but not dunder)
        if not include_private and attr_name.startswith('_') and not (
                attr_name.startswith('__') and attr_name.endswith('__')):
            return False

        return True

    # Return new filtered list preserving order
    return [attr_name for attr_name in attrs if _should_keep_attribute(attr_name)]


# TODO check that deep_sizeof() usage is optional where applicable, should NOT be used by default
def deep_sizeof(obj: Any, *, exclude_types: tuple[type, ...] = ()) -> int:
    """
    Calculate the deep memory size of an object including all referenced objects.

    This function recursively traverses object references to compute total memory
    usage, similar to pympler.asizeof but using only Python stdlib. It handles
    circular references and avoids double-counting shared objects.

    Args:
        obj: Any Python object to measure
        exclude_types: Tuple of types to exclude from size calculation.
                      Useful for excluding large shared objects like modules.

    Returns:
        Total size in bytes including all referenced objects

    Example:
        >>> data = {'items': [1, 2, 3], 'nested': {'key': 'value'}}
        >>> size = deep_sizeof(data)
        >>> size > sys.getsizeof(data)
        True

        >>> # Exclude string types from calculation
        >>> size_no_strings = deep_sizeof(data, exclude_types=(str,))
    """

    return _deep_sizeof_recursive(obj, set(), exclude_types)


def _deep_sizeof_recursive(obj: Any, seen: Set[int], exclude_types: tuple[type, ...]) -> int:
    """
    Recursive implementation for deep_sizeof calculation with cycle detection.

    Args:
        obj: Object to measure
        seen: Set of already-seen object IDs to prevent cycles
        exclude_types: Types to exclude from calculation

    Returns:
        Size in bytes
    """
    # Skip excluded types
    if exclude_types and isinstance(obj, exclude_types):
        return 0

    obj_id = id(obj)
    if obj_id in seen:
        return 0  # Already counted or circular reference

    seen.add(obj_id)

    try:
        size = sys.getsizeof(obj)
    except (TypeError, AttributeError):
        # Some objects don't support getsizeof
        return 0

    # Handle container types
    try:
        if isinstance(obj, dict):
            for key, value in obj.items():
                size += _deep_sizeof_recursive(key, seen, exclude_types)
                size += _deep_sizeof_recursive(value, seen, exclude_types)
        elif isinstance(obj, (list, tuple, set, frozenset)):
            for item in obj:
                size += _deep_sizeof_recursive(item, seen, exclude_types)
        elif isinstance(obj, (str, bytes, bytearray, int, float, complex, bool, type(None))):
            # Immutable primitives - no additional references to traverse
            pass
        elif hasattr(obj, '__dict__'):
            # User-defined objects with instance attributes
            size += _deep_sizeof_recursive(obj.__dict__, seen, exclude_types)
        elif hasattr(obj, '__slots__'):
            # Objects with __slots__
            for slot in obj.__slots__:
                if hasattr(obj, slot):
                    attr_value = getattr(obj, slot)
                    size += _deep_sizeof_recursive(attr_value, seen, exclude_types)
    except (AttributeError, TypeError, RecursionError):
        # Handle edge cases gracefully - some objects may not be introspectable
        pass

    return size


def is_builtin(obj: Any) -> bool:
    """
    Check if an object is a built-in type or instance of a built-in type.

    This function identifies core Python value types (int, str, list, dict, etc.)
    and their instances, excluding meta-programming utilities, functions, and modules.

    Args:
        obj: Any Python object to check.

    Returns:
        bool: True if obj is a built-in type or instance of a built-in type.

    Examples:
        >>> is_builtin(int)          # Built-in type
        True
        >>> is_builtin(42)           # Instance of built-in type
        True
        >>> is_builtin([1, 2, 3])    # Instance of built-in type
        True
        >>> is_builtin(len)          # Built-in function
        False
        >>> is_builtin(property)     # Descriptor helper
        False
        >>> is_builtin(object())     # Instance of built-in type
        True

    Note:
        - Returns False for functions, methods, modules, and descriptor helpers
        - Returns False for user-defined classes and their instances
        - Focuses on core value types rather than meta-programming utilities
    """
    try:
        # Handle class objects (types)
        if isinstance(obj, type):
            return getattr(obj, "__module__", None) == "builtins"

        # Exclude functions, methods, built-in callables, and modules
        if (inspect.isfunction(obj) or
                inspect.ismethod(obj) or
                inspect.isbuiltin(obj) or
                inspect.ismodule(obj)):
            return False

        # Exclude descriptor helpers
        if isinstance(obj, (property, staticmethod, classmethod)):
            return False

        # Check if instance's class is from builtins
        obj_class = getattr(obj, "__class__", None)
        if obj_class is None:
            return False

        return getattr(obj_class, "__module__", None) == "builtins"

    except (AttributeError, TypeError):
        # Handle edge cases where attribute access might fail
        return False
