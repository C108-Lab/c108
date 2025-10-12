"""
A collection of basic utilities for object introspection and attribute manipulation.
"""

# Standard library -----------------------------------------------------------------------------------------------------
import collections.abc as abc
import inspect
import re
import sys
import warnings

from collections import defaultdict
from dataclasses import dataclass, InitVar
from typing import Any, Literal, Set

# Local ----------------------------------------------------------------------------------------------------------------
from .utils import class_name


# Classes --------------------------------------------------------------------------------------------------------------

@dataclass
class ObjectInfo:
    """
    Summarize an object with its type, size, unit, and human-friendly presentation.

    Lightweight, heuristic-based object inspection for quick diagnostics,
    logging, and REPL exploration. This is designed for simplistic stats and one-line
    string conversion, NOT a replacement for profiling tools or exact memory analysis.

    Prioritizes simplicity and readability over precision. Deep size calculation is opt-in
    due to performance cost on large/nested objects.

    Provides a lightweight summary of an object, including its type, a human-oriented
    size measure, unit labels, and optionally a deep byte size.

    Attributes:
        type (type): The object's type (class for instances, or the type object itself).
        size (int | float | list[int|float]): Human-oriented measure:
            - numbers, bytes-like: int (bytes)
            - str: int (characters)
            - containers (Sequence/Set/Mapping): int (items_count)
            - image-like: list[int, int, float] (width, height, megapixels)
            - class objects: int (attrs_count)
            - user-defined instances with attrs: list[int, int] (attrs_count, deep)
        unit (str | list[str]): Unit label(s) matching the structure of size.
            Note: a plain str is treated as a scalar unit, not a sequence.
        deep_size (int | None): Deep size in bytes (like pympler.deep_sizeof) computed
            via c108.abc.deep_sizeof() function for most objects; None for classes or
            when not computed.

    Init vars:
        fully_qualified (bool): If true, class_name is fully qualified; builtins are never fully qualified.

    Raises:
        ValueError: If size and unit are sequences of different lengths.

    See Also:
        :mod:`~.dictify`: Comprehensive object-to-dictionary conversion toolkit.
    """
    type: type
    size: int | float | list[int | float] = None
    unit: str | list[str] = None
    deep_size: int | None = None

    fully_qualified: InitVar[bool] = False

    def __post_init__(self, fully_qualified: bool):
        """
        Post-initialization validation and options.
        """
        self._fully_qualified = fully_qualified

        # Initialize defaults
        if self.size is None:
            self.size = []
        if self.unit is None:
            self.unit = []

        # Only validate runtime logic constraints
        if isinstance(self.size, abc.Sequence) and not isinstance(self.size, (str, bytes, bytearray)):
            if isinstance(self.unit, abc.Sequence) and not isinstance(self.unit, (str, bytes, bytearray)):
                if len(self.size) != len(self.unit):
                    raise ValueError(
                        f"size and unit must be same length: "
                        f"len(size)={len(self.size)}, len(unit)={len(self.unit)}"
                    )

    @classmethod
    def from_object(cls, obj: Any,
                    fully_qualified: bool = False,
                    deep_size: bool = False) -> "ObjectInfo":
        """
        Build an ObjectInfo summary of 'obj'.

        Heuristics according to 'obj' type:
          - Numbers: size=N bytes (shallow), unit="bytes".
          - str: size=N chars, unit="chars".
          - bytes/bytearray/memoryview: size=N bytes, unit="bytes".
          - Sequence/Set/Mapping: size=N items, unit="items".
          - Image-like: size=[width, height, Mpx], unit=["width","height","Mpx"].
          - Class (type): size=N attrs, unit="attrs"; deep_size=None.
          - Instance with attrs: size=[N attrs, deep bytes], unit=["attrs","bytes"].
          - Other/no-attrs: size = shallow bytes, unit="bytes"
          - Any obj: get deep size via c108.abc.deep_sizeof() if deep_size=True;
                     None for classes or when deep_size=False.

        Parameters:
          - obj: object to summarize.
          - fully_qualified: whether class_name should be fully qualified for non-builtin types.
          - deep_size: whether to compute deep size (can be expensive for large objects).

        Returns:
          - ObjectInfo with populated size, unit, deep_size, and type.
        """

        def __get_deep_size(o):
            try:
                deep_size_ = deep_sizeof(o) if deep_size else None
            except:
                deep_size_ = None
            return deep_size_

        def __get_shallow_size(o):
            try:
                size_ = sys.getsizeof(o)
            except:
                size_ = None
            return size_

        # Scalars
        if isinstance(obj, (int, float, bool, complex)):
            b = __get_shallow_size(obj)  # shallow bytes, used for human-facing size
            return cls(size=b, unit="bytes", deep_size=__get_deep_size(obj),
                       type=type(obj), fully_qualified=fully_qualified)
        elif isinstance(obj, str):
            # Human-facing size is chars; deep bytes can be useful to compare memory footprint
            return cls(size=len(obj), unit="chars", deep_size=__get_deep_size(obj),
                       type=type(obj), fully_qualified=fully_qualified)
        elif isinstance(obj, (bytes, bytearray, memoryview)):
            n = len(obj)
            return cls(size=n, unit="bytes", deep_size=__get_deep_size(obj),
                       type=type(obj), fully_qualified=fully_qualified)

        # Containers
        elif isinstance(obj, (abc.Sequence, abc.Set, abc.Mapping)):
            return cls(size=len(obj), unit="items", deep_size=__get_deep_size(obj),
                       type=type(obj), fully_qualified=fully_qualified)

        # Images
        elif _acts_like_image(obj):
            width, height = obj.size
            mega_px = width * height / 1e6
            return cls(
                size=[width, height, mega_px],
                unit=["width", "height", "Mpx"],
                deep_size=__get_deep_size(obj),
                type=type(obj),
                fully_qualified=fully_qualified,
            )

        # Class objects
        elif type(obj) is type:
            attrs = attrs_search(obj, include_private=False, include_property=False)
            return cls(type=obj,
                       size=len(attrs),
                       unit="attrs",
                       deep_size=None,
                       fully_qualified=fully_qualified)

        # Instances with attributes
        elif attrs := attrs_search(obj, include_private=False, include_property=False):
            return cls(type=type(obj),
                       size=len(attrs),
                       unit="attrs",
                       deep_size=__get_deep_size(obj),
                       fully_qualified=fully_qualified)

        # Other instances with no attrs found
        else:
            return cls(type=type(obj),
                       size=__get_shallow_size(obj),
                       unit="bytes",
                       deep_size=__get_deep_size(obj),
                       fully_qualified=fully_qualified)

    def to_str(self, deep_size: bool = False) -> str:
        """
        Human-readable one-line summary.

        Parameters:
            deep_size: If True and deep_size is available, append deep bytes info.

        Examples:
            >>>
            "<int> 28 bytes"
            "<str> 11 chars"
            "<list> 3 items"
            "<list> 3 items, 256 deep bytes"
            "<PIL.Image.Image> 640⨯480 W⨯H, 0.307 Mpx"
            "<PIL.Image.Image> 640⨯480 W⨯H, 0.307 Mpx, 1228800 deep bytes"
            "<MyClass> 4 attrs"
            "<MyClass> 4 attrs, 1024 deep bytes"

        Raises:
              ValueError: If size and unit lengths mismatch.
        """
        # Handle list-based size/unit pairs
        if isinstance(self.size, list) and isinstance(self.unit, list):
            if len(self.size) != len(self.unit):
                raise ValueError("Size and unit lists must have the same length")

            if _acts_like_image(self.type):
                # Special image formatting: width⨯height W⨯H, Mpx
                width, height, mega_px = self.size
                base_str = f"<{self._class_name}> {width}⨯{height} W⨯H, {round(mega_px, ndigits=3)} Mpx"
            else:
                # Generic list formatting: join size-unit pairs
                size_unit_pairs = [f"{s} {u}" for s, u in zip(self.size, self.unit)]
                base_str = f"<{self._class_name}> {', '.join(size_unit_pairs)}"
        else:
            # Single size/unit pair
            base_str = f"<{self._class_name}> {self.size} {self.unit}"

        # Consistently append deep_size info if requested and available
        if deep_size and self.deep_size is not None:
            base_str += f", {self.deep_size} deep bytes"

        return base_str

    def to_dict(self, include_none_attrs: bool = False) -> dict[str, Any]:
        """
        Export as dictionary.

        Args:
            include_none_attrs: If True, include fields with None values (like deep_size when not computed).

        Returns:
            Dictionary with keys: type, size, unit, and optionally deep_size.

        Examples:
            >>> info = ObjectInfo.from_object("hello")
            >>> info.to_dict()
            {'type': <class 'str'>, 'size': 5, 'unit': 'chars'}
        """
        result = {
            "type": self.type,
            "size": self.size,
            "unit": self.unit,
        }

        if include_none_attrs or self.deep_size is not None:
            result["deep_size"] = self.deep_size

        return result

    def __str__(self) -> str:
        """Default string representation using to_str() with default formatting."""
        return self.to_str()

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (f"ObjectInfo(type={self.type.__name__}, size={self.size}, "
                f"unit={self.unit}, deep_size={self.deep_size})")

    @property
    def _class_name(self) -> str:
        """Return a display name for 'type' (fully qualified for non-builtin types if enabled)."""
        return class_name(self.type, fully_qualified=self._fully_qualified,
                          fully_qualified_builtins=False)


# Methods --------------------------------------------------------------------------------------------------------------


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


def deep_sizeof(
        obj: Any,
        *,
        format: Literal["int", "dict"] = "int",
        exclude_types: tuple[type, ...] = (),
        exclude_ids: set[int] | None = None,
        max_depth: int | None = None,
        seen: set[int] | None = None,
        on_error: Literal["skip", "raise", "warn"] = "skip",
) -> int | dict[str, Any]:
    """
    Calculate the deep memory size of an object including all referenced objects.

    This function recursively traverses object references to compute total memory
    usage, similar to pympler.asizeof but using only Python stdlib. It handles
    circular references and avoids double-counting shared objects.

    Args:
        obj: Any Python object to measure.
        format: Output format. Default "int" returns total bytes as integer.
            Use "dict" for detailed breakdown including per-type analysis,
            object count, and maximum depth reached.
        exclude_types: Tuple of types to exclude from size calculation.
            Useful for excluding large shared objects like modules.
            Objects of these types contribute 0 bytes to the total.
        exclude_ids: Set of specific object IDs (from id()) to exclude.
            Useful for excluding particular instances rather than entire types.
            More fine-grained than exclude_types.
        max_depth: Maximum recursion depth. None (default) means unlimited.
            Useful for preventing deep recursion on heavily nested structures.
            When limit is reached, objects at that depth are counted shallowly.
        seen: Set of object IDs already counted. Pass the same set across
            multiple deep_sizeof() calls to measure exclusive sizes and avoid
            double-counting shared references between objects.
        on_error: How to handle objects that raise exceptions during size calculation:
            - "skip" (default): Skip problematic objects, continue traversal. In dict
              format, tracks errors in 'errors' field.
            - "raise": Re-raise the first exception encountered.
            - "warn": Issue warnings for problematic objects but continue.

    Returns:
        int: Total size in bytes (when format="int")
        dict: Detailed breakdown (when format="dict") containing:
            - total_bytes (int): Total size in bytes
            - by_type (dict[type, int]): Bytes per type object (not string names)
            - object_count (int): Number of objects successfully traversed
            - max_depth_reached (int): Deepest nesting level encountered
            - errors (dict[type, int]): Count of errors by exception type object
              (e.g., {TypeError: 3, AttributeError: 1})
            - problematic_types (set[type]): Type objects that raised exceptions
              during __sizeof__ or attribute access

    Raises:
        RecursionError: If Python's recursion limit is exceeded during traversal.
            Consider using max_depth parameter to prevent this.
        TypeError: Only when on_error="raise" and an object doesn't implement
            __sizeof__ properly.
        AttributeError: Only when on_error="raise" and attribute access fails
            on an object with unusual attribute handling.

    Examples:
        Basic usage:
            >>> data = {'items': [1, 2, 3], 'nested': {'key': 'value'}}
            >>> size = deep_sizeof(data)
            >>> size > sys.getsizeof(data)
            True

        Detailed breakdown with error tracking:
            >>> info = deep_sizeof(data, format="dict")
            >>> info['total_bytes']
            248
            >>> info['by_type']
            {<class 'dict'>: 128, <class 'list'>: 56, <class 'int'>: 84, <class 'str'>: 44}
            >>> info['errors']
            {}  # No errors encountered
            >>> info['problematic_types']
            set()

        Handling buggy objects:
            >>> class BuggyClass:
            ...     def __sizeof__(self):
            ...         raise RuntimeError("Broken!")
            >>> obj = {'good': [1, 2], 'bad': BuggyClass()}
            >>>
            >>> # Default: skip errors and continue
            >>> size = deep_sizeof(obj)  # Returns size of 'good' parts only
            >>>
            >>> # Get details about what failed
            >>> info = deep_sizeof(obj, format="dict")
            >>> info['errors']
            {<class 'RuntimeError'>: 1}
            >>> info['problematic_types']
            {<class '__main__.BuggyClass'>}
            >>>
            >>> # Stop on first error
            >>> deep_sizeof(obj, on_error="raise")
            RuntimeError: Broken!

        Exclude specific types:
            >>> size_no_strings = deep_sizeof(data, exclude_types=(str,))

        Limit recursion depth:
            >>> size = deep_sizeof(deeply_nested_obj, max_depth=10)

        Exclude specific objects:
            >>> global_cache = {...}
            >>> size = deep_sizeof(my_obj, exclude_ids={id(global_cache)})

        Measure exclusive sizes across multiple objects:
            >>> seen = set()
            >>> size1 = deep_sizeof(obj1, seen=seen)
            >>> size2 = deep_sizeof(obj2, seen=seen)
            >>> # size2 won't double-count objects shared with obj1

        Warning mode for debugging:
            >>> import warnings
            >>> with warnings.catch_warnings(record=True) as w:
            ...     size = deep_sizeof(obj, on_error="warn")
            ...     if w:
            ...         print(f"Encountered {len(w)} problematic objects")

    Note:
        - Circular references are handled automatically via internal tracking
        - Module objects are typically excluded by default in implementations
        - When on_error="skip", problematic objects contribute 0 bytes but
          traversal continues to their children when possible
        - The 'errors' and 'problematic_types' fields are only included in
          dict format output
        - The function is designed for diagnostic purposes, not for precise
          memory profiling. Use dedicated profiling tools for production analysis.
        - Error tracking uses actual type objects, not string names, ensuring
          robustness when same type names exist in different modules.
          Use type.__module__ and type.__name__ if string representation is needed.
    """
    # Initialize tracking structures
    if seen is None:
        seen = set()

    if exclude_ids is None:
        exclude_ids = set()

    # Detailed format tracking
    by_type = defaultdict(int) if format == "dict" else None
    error_counts = defaultdict(int) if format == "dict" else None
    problematic_types = set() if format == "dict" else None
    object_count = [0] if format == "dict" else None
    max_depth_tracker = [0] if format == "dict" else None

    # Perform recursive calculation
    total_bytes = _deep_sizeof_recursive(
        obj=obj,
        seen=seen,
        exclude_types=exclude_types,
        exclude_ids=exclude_ids,
        max_depth=max_depth,
        current_depth=0,
        on_error=on_error,
        by_type=by_type,
        error_counts=error_counts,
        problematic_types=problematic_types,
        object_count=object_count,
        max_depth_tracker=max_depth_tracker,
    )

    # Return appropriate format
    if format == "int":
        return total_bytes
    else:
        return {
            'total_bytes': total_bytes,
            'by_type': dict(by_type),
            'object_count': object_count[0],
            'max_depth_reached': max_depth_tracker[0],
            'errors': dict(error_counts),
            'problematic_types': problematic_types,
        }


def _deep_sizeof_recursive(
        obj: Any,
        seen: Set[int],
        exclude_types: tuple[type, ...],
        exclude_ids: set[int],
        max_depth: int | None,
        current_depth: int,
        on_error: str,
        by_type: dict | None,
        error_counts: dict | None,
        problematic_types: set | None,
        object_count: list | None,
        max_depth_tracker: list | None,
) -> int:
    """
    Recursive implementation for deep_sizeof calculation with cycle detection.

    Args:
        obj: Object to measure
        seen: Set of already-seen object IDs to prevent cycles
        exclude_types: Types to exclude from calculation
        exclude_ids: Specific object IDs to exclude
        max_depth: Maximum recursion depth (None = unlimited)
        current_depth: Current depth in recursion
        on_error: Error handling strategy
        by_type: Type-to-bytes mapping (for detailed format)
        error_counts: Error type counts (for detailed format)
        problematic_types: Set of problematic type names (for detailed format)
        object_count: Counter for number of objects (for detailed format)
        max_depth_tracker: Tracks maximum depth reached (for detailed format)

    Returns:
        Size in bytes
    """
    # Update depth tracking
    if max_depth_tracker is not None:
        max_depth_tracker[0] = max(max_depth_tracker[0], current_depth)

    # Check depth limit
    if max_depth is not None and current_depth >= max_depth:
        # At max depth, count shallowly only
        try:
            return sys.getsizeof(obj)
        except Exception:
            return 0

    # Skip excluded types
    if exclude_types and isinstance(obj, exclude_types):
        return 0

    # Skip excluded IDs
    obj_id = id(obj)
    if obj_id in exclude_ids:
        return 0

    # Check if already seen (circular reference or shared object)
    if obj_id in seen:
        return 0

    seen.add(obj_id)

    # Get shallow size with error handling
    size = 0
    obj_type = type(obj)

    try:
        size = sys.getsizeof(obj)
        if by_type is not None:
            by_type[obj_type] += size
        if object_count is not None:
            object_count[0] += 1
    except Exception as e:
        # Handle __sizeof__ errors
        if on_error == "raise":
            raise
        elif on_error == "warn":
            warnings.warn(
                f"Failed to get size of {obj_type.__module__}.{obj_type.__name__}: {type(e).__name__}: {e}",
                RuntimeWarning,
                stacklevel=2
            )

        if error_counts is not None:
            error_counts[type(e)] += 1
        if problematic_types is not None:
            problematic_types.add(obj_type)

        return 0  # Can't measure this object

    # Traverse child objects based on type
    next_depth = current_depth + 1

    try:
        # Dictionaries: traverse keys and values
        if isinstance(obj, dict):
            for key, value in obj.items():
                size += _deep_sizeof_recursive(
                    key, seen, exclude_types, exclude_ids, max_depth,
                    next_depth, on_error, by_type, error_counts,
                    problematic_types, object_count, max_depth_tracker
                )
                size += _deep_sizeof_recursive(
                    value, seen, exclude_types, exclude_ids, max_depth,
                    next_depth, on_error, by_type, error_counts,
                    problematic_types, object_count, max_depth_tracker
                )

        # Sequences and sets: traverse items
        elif isinstance(obj, (list, tuple, set, frozenset)):
            for item in obj:
                size += _deep_sizeof_recursive(
                    item, seen, exclude_types, exclude_ids, max_depth,
                    next_depth, on_error, by_type, error_counts,
                    problematic_types, object_count, max_depth_tracker
                )

        # Primitives: no child objects to traverse
        elif isinstance(obj, (str, bytes, bytearray, int, float, complex, bool, type(None))):
            pass  # Already counted in shallow size

        # Objects with __dict__: traverse instance attributes
        elif hasattr(obj, '__dict__'):
            try:
                obj_dict = obj.__dict__
                size += _deep_sizeof_recursive(
                    obj_dict, seen, exclude_types, exclude_ids, max_depth,
                    next_depth, on_error, by_type, error_counts,
                    problematic_types, object_count, max_depth_tracker
                )
            except Exception as e:
                if on_error == "raise":
                    raise
                elif on_error == "warn":
                    warnings.warn(
                        f"Failed to access __dict__ of {obj_type.__module__}.{obj_type.__name__}: {type(e).__name__}: {e}",
                        RuntimeWarning,
                        stacklevel=2
                    )
                if error_counts is not None:
                    error_counts[type(e)] += 1
                if problematic_types is not None:
                    problematic_types.add(obj_type)

        # Objects with __slots__: traverse slot attributes
        elif hasattr(obj, '__slots__'):
            try:
                for slot in obj.__slots__:
                    if hasattr(obj, slot):
                        attr_value = getattr(obj, slot)
                        size += _deep_sizeof_recursive(
                            attr_value, seen, exclude_types, exclude_ids, max_depth,
                            next_depth, on_error, by_type, error_counts,
                            problematic_types, object_count, max_depth_tracker
                        )
            except Exception as e:
                if on_error == "raise":
                    raise
                elif on_error == "warn":
                    warnings.warn(
                        f"Failed to access __slots__ of {obj_type.__module__}.{obj_type.__name__}: {type(e).__name__}: {e}",
                        RuntimeWarning,
                        stacklevel=2
                    )
                if error_counts is not None:
                    error_counts[type(e)] += 1
                if problematic_types is not None:
                    problematic_types.add(obj_type)

    except RecursionError:
        # Let RecursionError propagate regardless of on_error setting
        raise
    except Exception as e:
        # Catch-all for unexpected errors during traversal
        if on_error == "raise":
            raise
        elif on_error == "warn":
            warnings.warn(
                f"Error traversing {obj_type.__module__}.{obj_type.__name__}: {type(e).__name__}: {e}",
                RuntimeWarning,
                stacklevel=2
            )
        if error_counts is not None:
            error_counts[type(e)] += 1
        if problematic_types is not None:
            problematic_types.add(obj_type)

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


# Private Methods ------------------------------------------------------------------------------------------------------

def _acts_like_image(obj: Any) -> bool:
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
