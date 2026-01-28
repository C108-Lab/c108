"""
Runtime introspection and type-validation utilities for Python objects.

This module provides lightweight object summaries, deep memory sizing, and flexible attribute search
to aid debugging and diagnostics. Includes decorators and inline helpers to validate
function parameters and object attributes against type hints.
"""

# Standard library -----------------------------------------------------------------------------------------------------
import collections.abc as abc
import inspect
import functools
import re
import sys
import warnings
from collections import defaultdict
from dataclasses import InitVar, dataclass, field
from dataclasses import is_dataclass, fields as dc_fields
from types import UnionType
from typing import Any, Callable, Generic, Literal, Set, TypeVar, Union
from typing import get_type_hints, get_origin, get_args, overload

# Local ----------------------------------------------------------------------------------------------------------------
from .formatters import fmt_type, fmt_value
from .utils import Self, class_name

# Public API -----------------------------------------------------------------------------------------------------------
__all__ = [
    # Classes
    "ObjectInfo",
    # Functions
    "deep_sizeof",
    "isbuiltin",
    "search_attrs",
    "validate_param_types",
    "validate_types",
    # Decorators
    "classgetter",
]

# Classes --------------------------------------------------------------------------------------------------------------

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")
ClsT = TypeVar("ClsT", bound=type)

from .dataclasses import mergeable


@dataclass(frozen=True)
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
    size: int | float | list[int | float] = field(default_factory=list)
    unit: str | list[str] = field(default_factory=list)
    deep_size: int | None = None

    fully_qualified: InitVar[bool] = False

    def __post_init__(self, fully_qualified: bool):
        """
        Post-initialization validation and options.

        For frozen dataclasses, we must use object.__setattr__() to set attributes.
        """
        # Store fully_qualified using the frozen workaround
        object.__setattr__(self, "_fully_qualified", fully_qualified)

        # Validate runtime logic constraints
        # Both size and unit must be sequences (and not str/bytes) to validate length
        if isinstance(self.size, abc.Sequence) and not isinstance(
            self.size, (str, bytes, bytearray)
        ):
            if isinstance(self.unit, abc.Sequence) and not isinstance(
                self.unit, (str, bytes, bytearray)
            ):
                if len(self.size) != len(self.unit):
                    raise ValueError(
                        f"size and unit must be same length, but got "
                        f"len(size)={len(self.size)}, len(unit)={len(self.unit)}"
                    )

    @classmethod
    def from_object(
        cls, obj: Any, fully_qualified: bool = False, deep_size: bool = False
    ) -> "ObjectInfo":
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
            return cls(
                size=b,
                unit="bytes",
                deep_size=__get_deep_size(obj),
                type=type(obj),
                fully_qualified=fully_qualified,
            )
        elif isinstance(obj, str):
            # Human-facing size is chars; deep bytes can be useful to compare memory footprint
            return cls(
                size=len(obj),
                unit="chars",
                deep_size=__get_deep_size(obj),
                type=type(obj),
                fully_qualified=fully_qualified,
            )
        elif isinstance(obj, (bytes, bytearray, memoryview)):
            n = len(obj)
            return cls(
                size=n,
                unit="bytes",
                deep_size=__get_deep_size(obj),
                type=type(obj),
                fully_qualified=fully_qualified,
            )

        # Containers
        elif isinstance(obj, (abc.Sequence, abc.Set, abc.Mapping)):
            return cls(
                size=len(obj),
                unit="items",
                deep_size=__get_deep_size(obj),
                type=type(obj),
                fully_qualified=fully_qualified,
            )

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
            attrs = search_attrs(
                obj,
                format="list",
                include_methods=False,
                include_private=False,
                include_properties=False,
                skip_errors=True,
            )
            return cls(
                type=obj,
                size=len(attrs),
                unit="attrs",
                deep_size=None,
                fully_qualified=fully_qualified,
            )

        # Instances with attributes
        elif attrs := search_attrs(
            obj,
            format="list",
            include_methods=False,
            include_private=False,
            include_properties=False,
            skip_errors=True,
        ):
            return cls(
                type=type(obj),
                size=len(attrs),
                unit="attrs",
                deep_size=__get_deep_size(obj),
                fully_qualified=fully_qualified,
            )

        # Other instances with no attrs found
        else:
            return cls(
                type=type(obj),
                size=__get_shallow_size(obj),
                unit="bytes",
                deep_size=__get_deep_size(obj),
                fully_qualified=fully_qualified,
            )

    def to_str(self, deep_size: bool = False) -> str:
        """
        Human-readable one-line summary.

        Parameters:
            deep_size: If True and deep_size is available, append deep bytes info.

        Examples:
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
            # List lengths should be checked in __post_init__()

            if _acts_like_image(self.type):
                # Special image formatting: width⨯height W⨯H, Mpx
                width, height, mega_px = self.size
                base_str = (
                    f"<{self._class_name}> {width}⨯{height} W⨯H, {round(mega_px, ndigits=3)} Mpx"
                )
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
        return (
            f"ObjectInfo(type={self.type.__name__}, size={self.size}, "
            f"unit={self.unit}, deep_size={self.deep_size})"
        )

    @property
    def _class_name(self) -> str:
        """Return a display name for 'type' (fully qualified for non-builtin types if enabled)."""
        return class_name(
            self.type,
            fully_qualified=self._fully_qualified,
            fully_qualified_builtins=False,
        )


# Methods --------------------------------------------------------------------------------------------------------------


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
            723
            >>> info['by_type']
            {<class 'dict'>: 368, <class 'str'>: 183, <class 'list'>: 88, <class 'int'>: 84}
            >>> info['errors']
            {}
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
            {<class 'c108.abc.BuggyClass'>}
            >>>
            >>> # Stop on first error
            >>> deep_sizeof(obj, on_error="raise")
            Traceback (most recent call last):
            ...
            RuntimeError: Broken!

        Exclude specific types:
            >>> size_no_strings = deep_sizeof(data, exclude_types=(str,))

        Limit recursion depth:
            >>> deeply_nested_obj = [[[0]]]
            >>> size = deep_sizeof(deeply_nested_obj, max_depth=10)

        Exclude specific objects:
            >>> global_cache = {...}
            >>> size = deep_sizeof(obj, exclude_ids={id(global_cache)})

        Warning mode for debugging:
            >>> import warnings
            >>> with warnings.catch_warnings(record=True) as w:
            ...     size = deep_sizeof(obj, on_error="warn")
            ...     if w:
            ...         print(f"Encountered {len(w)} problematic objects")
            Encountered 1 problematic objects

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
            "total_bytes": total_bytes,
            "by_type": dict(by_type),
            "object_count": object_count[0],
            "max_depth_reached": max_depth_tracker[0],
            "errors": dict(error_counts),
            "problematic_types": problematic_types,
        }


def _deep_handle_error(
    error: Exception,
    obj_type: type,
    context: str,
    on_error: str,
    error_counts: dict | None,
    problematic_types: set | None,
) -> None:
    """
    Handle errors during sizeof traversal.

    Args:
        error: The exception that was raised
        obj_type: Type of the object that caused the error
        context: Description of what operation failed (e.g., "access __dict__")
        on_error: Error handling strategy ("raise", "warn", or "skip")
        error_counts: Dictionary tracking error counts by type
        problematic_types: Set tracking types that caused errors
    """
    if on_error == "raise":
        raise
    elif on_error == "warn":
        warnings.warn(
            f"Failed to {context} of {obj_type.__module__}.{obj_type.__name__}: {type(error).__name__}: {error}",
            RuntimeWarning,
            stacklevel=3,
        )

    if error_counts is not None:
        error_counts[type(error)] += 1
    if problematic_types is not None:
        problematic_types.add(obj_type)


def _deep_traverse_object_attributes(
    obj: Any,
    obj_type: type,
    seen: Set[int],
    exclude_types: tuple[type, ...],
    exclude_ids: set[int],
    max_depth: int | None,
    next_depth: int,
    on_error: str,
    by_type: dict | None,
    error_counts: dict | None,
    problematic_types: set | None,
    object_count: list | None,
    max_depth_tracker: list | None,
) -> int:
    """
    Traverse an object's attributes via __dict__ and __slots__.

    Returns the total size of all traversed attributes.
    """
    size = 0

    # Try to access __dict__ - AttributeError means "no dict" (normal)
    try:
        obj_dict = obj.__dict__
    except AttributeError:
        # No __dict__ or __dict__ access raised AttributeError
        # This is normal Python behavior - treat as "no dict", not an error
        pass
    except Exception as e:
        # Non-AttributeError when accessing __dict__ - this IS unusual
        _deep_handle_error(
            e, obj_type, "access __dict__", on_error, error_counts, problematic_types
        )
    else:
        # Successfully got __dict__, recurse into it
        size += _deep_sizeof_recursive(
            obj_dict,
            seen,
            exclude_types,
            exclude_ids,
            max_depth,
            next_depth,
            on_error,
            by_type,
            error_counts,
            problematic_types,
            object_count,
            max_depth_tracker,
        )

    # Try __slots__ if object might have them
    # AttributeError during __slots__ access is also normal (no slots)
    try:
        slots = obj.__slots__
    except AttributeError:
        # No __slots__ - this is normal
        pass
    except Exception as e:
        # Non-AttributeError when accessing __slots__ - unusual
        _deep_handle_error(
            e, obj_type, "access __slots__", on_error, error_counts, problematic_types
        )
    else:
        # Successfully got __slots__, traverse each slot
        # Robustness branch for dynamically defined and pathological classes
        try:
            for slot in slots:
                if hasattr(obj, slot):
                    attr_value = getattr(obj, slot)
                    size += _deep_sizeof_recursive(
                        attr_value,
                        seen,
                        exclude_types,
                        exclude_ids,
                        max_depth,
                        next_depth,
                        on_error,
                        by_type,
                        error_counts,
                        problematic_types,
                        object_count,
                        max_depth_tracker,
                    )
        except Exception as e:
            # Error iterating slots or accessing slot values
            _deep_handle_error(
                e, obj_type, "traverse __slots__", on_error, error_counts, problematic_types
            )

    return size


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
        _deep_handle_error(e, obj_type, "get size", on_error, error_counts, problematic_types)
        return 0  # Can't measure this object

    # Traverse child objects based on type
    next_depth = current_depth + 1

    try:
        # Dictionaries: traverse keys and values
        if isinstance(obj, dict):
            for key, value in obj.items():
                size += _deep_sizeof_recursive(
                    key,
                    seen,
                    exclude_types,
                    exclude_ids,
                    max_depth,
                    next_depth,
                    on_error,
                    by_type,
                    error_counts,
                    problematic_types,
                    object_count,
                    max_depth_tracker,
                )
                size += _deep_sizeof_recursive(
                    value,
                    seen,
                    exclude_types,
                    exclude_ids,
                    max_depth,
                    next_depth,
                    on_error,
                    by_type,
                    error_counts,
                    problematic_types,
                    object_count,
                    max_depth_tracker,
                )

        # Sequences and sets: traverse items
        elif isinstance(obj, (list, tuple, set, frozenset)):
            for item in obj:
                size += _deep_sizeof_recursive(
                    item,
                    seen,
                    exclude_types,
                    exclude_ids,
                    max_depth,
                    next_depth,
                    on_error,
                    by_type,
                    error_counts,
                    problematic_types,
                    object_count,
                    max_depth_tracker,
                )

        # Primitives: no child objects to traverse
        elif isinstance(obj, (str, bytes, bytearray, int, float, complex, bool, type(None))):
            pass  # Already counted in shallow size

        # Objects with __dict__ and/or __slots__: traverse instance attributes
        else:
            size += _deep_traverse_object_attributes(
                obj,
                obj_type,
                seen,
                exclude_types,
                exclude_ids,
                max_depth,
                next_depth,
                on_error,
                by_type,
                error_counts,
                problematic_types,
                object_count,
                max_depth_tracker,
            )

    except RecursionError:
        # Let RecursionError propagate regardless of on_error setting
        raise
    except Exception as e:
        # Catch-all for unexpected errors during traversal
        _deep_handle_error(e, obj_type, "traverse", on_error, error_counts, problematic_types)

    return size


def isbuiltin(obj: Any) -> bool:
    """
    Check if an object is a built-in type or instance of a built-in type.

    This function identifies core Python value types (int, str, list, dict, etc.)
    and their instances, excluding meta-programming utilities, functions, and modules.

    Args:
        obj: Any Python object to check.

    Returns:
        bool: True if obj is a built-in type or instance of a built-in type.

    Examples:
        >>> isbuiltin(int)          # Built-in type
        True
        >>> isbuiltin(42)           # Instance of built-in type
        True
        >>> isbuiltin([1, 2, 3])    # Instance of built-in type
        True
        >>> isbuiltin(len)          # Built-in function
        False
        >>> isbuiltin(property)     # Descriptor helper
        True
        >>> isbuiltin(object())     # Instance of built-in type
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

        # Check if instance's class is from builtins
        obj_class = getattr(obj, "__class__", None)
        if obj_class is None:
            return False

        return getattr(obj_class, "__module__", None) == "builtins"

    except (AttributeError, TypeError, RuntimeError):
        return False


@overload
def search_attrs(
    obj: Any,
    *,
    format: Literal["list"] = "list",
    exclude_none: bool = False,
    include_inherited: bool = True,
    include_methods: bool = False,
    include_private: bool = False,
    include_properties: bool = False,
    attr_type: type | tuple[type, ...] | None = None,
    pattern: str | None = None,
    skip_errors: bool = True,
    sort: bool = False,
) -> list[str]: ...


@overload
def search_attrs(
    obj: Any,
    *,
    format: Literal["dict"],
    exclude_none: bool = False,
    include_inherited: bool = True,
    include_methods: bool = False,
    include_private: bool = False,
    include_properties: bool = False,
    attr_type: type | tuple[type, ...] | None = None,
    pattern: str | None = None,
    skip_errors: bool = True,
    sort: bool = False,
) -> dict[str, Any]: ...


@overload
def search_attrs(
    obj: Any,
    *,
    format: Literal["items"],
    exclude_none: bool = False,
    include_inherited: bool = True,
    include_methods: bool = False,
    include_private: bool = False,
    include_properties: bool = False,
    attr_type: type | tuple[type, ...] | None = None,
    pattern: str | None = None,
    skip_errors: bool = True,
    sort: bool = False,
) -> list[tuple[str, Any]]: ...


def search_attrs(
    obj: Any,
    *,
    format: Literal["list", "dict", "items"] = "list",
    exclude_none: bool = False,
    include_inherited: bool = True,
    include_methods: bool = False,
    include_private: bool = False,
    include_properties: bool = False,
    attr_type: type | tuple[type, ...] | None = None,
    pattern: str | None = None,
    skip_errors: bool = True,
    sort: bool = False,
) -> list[str] | dict[str, Any] | list[tuple[str, Any]]:
    """
    Search for attributes in an object with flexible filtering and output formats.

    By default, returns only public, non-callable data attribute names. Use parameters
    to expand or narrow the search, and choose output format.

    Args:
        obj: The object to inspect for attributes
        format: Output format:
            - "list": list of unique attribute names (default)
            - "dict": dictionary mapping names to values (keys are unique)
            - "items": list of (name, value) tuples with unique names,
               compatible with dict() constructor
        exclude_none: If True, excludes attributes with None values
        include_inherited: If True, includes attributes from parent classes.
                          If False, only returns attributes in obj.__dict__ (instance attrs)
        include_methods: If True, includes callable attributes (methods, functions)
        include_private: If True, includes private attributes (starting with '_').
                        Does not include dunder or mangled attributes.
        include_properties: If True, includes property descriptors
        attr_type: Optional type or tuple of types to filter by attribute value type.
                  Only attributes whose values are instances of these types are included.
        pattern: Optional regex pattern to filter attribute names.
                 Must match the entire name (use '.*pattern.*' for substring matching)
        skip_errors: If True, silently skips attributes that raise errors on access.
                    If False, raises AttributeError on access failures.
        sort: If True, sorts attribute names alphabetically.
             Default False preserves dir() order.

    Returns:
        - If format="list": list[str] of attribute names
        - If format="dict": dict[str, Any] mapping names to values
        - If format="items": list[tuple[str, Any]] of (name, value) pairs

    Raises:
        AttributeError: If skip_errors=False and attribute access fails
        ValueError: If pattern is an invalid regex or format is invalid

    Notes:
        - Always excludes dunder attributes (__name__)
        - Always excludes mangled attributes (_ClassName__attr) unless include_private=True
        - Built-in primitive types return empty list/dict
        - Properties are checked by descriptor type, not by accessing values
        - When exclude_none=True or attr_type is set, properties are evaluated

    Examples:
        >>> class MyClass:
        ...     public = 1
        ...     _private = 2
        ...     none_val = None
        ...     @property
        ...     def prop(self):
        ...         return 3
        ...     def method(self):
        ...         pass
        >>> obj = MyClass()
        >>> search_attrs(obj)
        ['public', 'none_val']
        >>> search_attrs(obj, format="dict")
        {'public': 1, 'none_val': None}
        >>> search_attrs(obj, format="items")
        [('public', 1), ('none_val', None)]
        >>> search_attrs(obj, include_private=True)
        ['public', '_private', 'none_val']
        >>> search_attrs(obj, include_properties=True, format="dict")
        {'public': 1, 'none_val': None, 'prop': 3}
        >>> search_attrs(obj, exclude_none=True)
        ['public']
        >>> search_attrs(obj, pattern=r'pub.*')
        ['public']
        >>> search_attrs(obj, attr_type=int, format="dict")
        {'public': 1}
        >>> search_attrs(obj, include_methods=True, pattern=r'.*method.*')
        ['method']
    """

    def _search_attrs_empty_result(format: str) -> list | dict:
        """Return an appropriate empty result based on format."""
        if format == "dict":
            return {}
        else:
            return []

    def _search_attrs_is_property(obj: Any, attr_name: str) -> bool:
        """Check if an attribute is a property descriptor."""
        try:
            if inspect.isclass(obj):
                # Inspecting a class - look at the class itself
                descriptor = getattr(obj, attr_name, None)
            else:
                # Inspecting an instance - look at its type
                descriptor = getattr(type(obj), attr_name, None)
            return isinstance(descriptor, property)
        except (AttributeError, TypeError):
            return False

    # Validate format
    if format not in ("list", "dict", "items"):
        raise ValueError(
            f"format must be 'list', 'dict', or 'items' literal, got {fmt_value(format)}"
        )

    # Compile pattern if provided
    compiled_pattern = None
    if pattern is not None:
        try:
            compiled_pattern = re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {pattern!r}") from e

    # Built-in types that should return empty results
    ignored_types = (
        int,
        float,
        bool,
        str,
        list,
        tuple,
        dict,
        set,
        frozenset,
        bytes,
        bytearray,
        complex,
        memoryview,
        range,
        type(None),
    )

    # Return empty for primitives
    if isinstance(obj, ignored_types) or (inspect.isclass(obj) and obj in ignored_types):
        return _search_attrs_empty_result(format)

    # Get attribute source based on include_inherited
    if include_inherited:
        try:
            # dir() returns sorted list, but we want definition order
            # Build attribute list manually from __dict__ and MRO
            attr_list = []
            seen_attrs = set()

            # Get the MRO (Method Resolution Order)
            if inspect.isclass(obj):
                mro = obj.__mro__
            else:
                mro = type(obj).__mro__

            # First, add instance attributes (if it's an instance)
            if not inspect.isclass(obj) and hasattr(obj, "__dict__"):
                for attr in obj.__dict__.keys():
                    if attr not in seen_attrs:
                        attr_list.append(attr)
                        seen_attrs.add(attr)

            # Then traverse MRO to get class attributes in definition order
            for klass in mro:
                if klass is object:
                    continue
                if hasattr(klass, "__dict__"):
                    for attr in klass.__dict__.keys():
                        if attr not in seen_attrs:
                            attr_list.append(attr)
                            seen_attrs.add(attr)
        except (TypeError, AttributeError):
            return _search_attrs_empty_result(format)
    else:
        # Only instance attributes
        if hasattr(obj, "__dict__"):
            attr_list = list(obj.__dict__.keys())
        elif hasattr(obj, "__slots__"):
            # Handle __slots__ without __dict__
            attr_list = list(obj.__slots__)
        else:
            return _search_attrs_empty_result(format)

    result_names = []
    result_values = []
    seen = set()

    for attr_name in attr_list:
        # Skip if already processed
        if attr_name in seen:
            continue

        # Always skip dunder
        if attr_name.startswith("__") and attr_name.endswith("__"):
            continue

        # Handle private/mangled filtering
        if not include_private:
            # Skip all private (starts with _)
            if attr_name.startswith("_"):
                continue

        # Pattern matching
        if compiled_pattern and not compiled_pattern.fullmatch(attr_name):
            continue

        # Check if it's a property
        is_property = _search_attrs_is_property(obj, attr_name)

        if is_property and not include_properties:
            continue

        # Get attribute value (needed for type checking, None checking, callable checking)
        # Also needed for dict/tuples format
        # For properties, only access value if we have value-based filters or need the value for output
        need_value = (
            format != "list"
            or exclude_none
            or attr_type is not None
            or (not include_methods and not is_property)
        )

        if need_value:
            try:
                attr_value = getattr(obj, attr_name)
            except Exception as e:
                if skip_errors:
                    continue
                # Re-raise the original exception to preserve the message
                raise
        else:
            attr_value = None  # Won't be used

        # Check if callable (method/function)
        if not include_methods:
            # For properties, we already know they're not methods, skip the check
            if not is_property:
                is_callable = callable(attr_value)
                if is_callable:
                    continue

        # Check None exclusion
        if exclude_none and attr_value is None:
            continue

        # Check type filtering
        if attr_type is not None:
            if not isinstance(attr_value, attr_type):
                continue

        result_names.append(attr_name)
        if format != "list":
            result_values.append(attr_value)
        seen.add(attr_name)

    if sort:
        if format == "list":
            result_names.sort()
        elif format == "dict":
            # Sort by keys
            result_names, result_values = (
                zip(*sorted(zip(result_names, result_values))) if result_names else ([], [])
            )
            result_names = list(result_names)
            result_values = list(result_values)
        else:  # items
            pairs = sorted(zip(result_names, result_values))
            result_names = [name for name, _ in pairs]
            result_values = [value for _, value in pairs]

    # Return in requested format
    if format == "list":
        return result_names
    elif format == "dict":
        return dict(zip(result_names, result_values))
    else:  # items
        return list(zip(result_names, result_values))


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
    if "Image" not in target_cls.__name__:
        return False

    # 2. Perform structural checks on the class or instance.
    required_attrs = ["size", "mode", "format"]
    if not all(hasattr(target_cls, attr) for attr in required_attrs):
        return False

    expected_methods = ["save", "show", "resize", "crop"]
    if (
        sum(
            1
            for method in expected_methods
            if hasattr(target_cls, method) and callable(getattr(target_cls, method))
        )
        < 3
    ):
        return False

    # 3. If it's an instance, perform deeper, value-based checks.
    if not is_class:
        instance = obj
        try:
            size = getattr(instance, "size")
            if not (
                isinstance(size, tuple)
                and len(size) == 2
                and isinstance(size[0], int)
                and isinstance(size[1], int)
                and size[0] > 0
                and size[1] > 0
            ):
                return False
        except (AttributeError, ValueError, TypeError):
            return False

        # Validate the 'mode' attribute's value.
        try:
            mode = getattr(instance, "mode")
            if not isinstance(mode, str) or not mode:
                return False
        except (AttributeError, TypeError):
            return False

    # If all checks passed, it acts like an image.
    return True


# @classgettr decorator ------------------------------------------------------------------------------------------------


class ClassGetter(Generic[T]):
    """
    Descriptor for read-only class-level properties.

    Provides property-like access to class methods, enabling clean APIs
    where class attributes can be accessed without parentheses.

    Similar to @property but operates at the class level rather than
    instance level. Unlike @property, this is explicitly read-only and
    does not support setter/deleter methods.

    Args:
        fget: The getter function that takes the class as its argument
        cache: If True, cache the result per class to avoid recomputation
               on repeated access. Default: False.

    Attributes:
        fget: The wrapped getter function
        cache: Whether results are cached per class
        name: Attribute name (set automatically via __set_name__)
        owner: The class that owns this descriptor

    Examples:
        Basic usage:
            >>> class AWS:
            ...     s3 = "s3"
            ...     s3a = "s3a"
            ...
            ...     @classgetter
            ...     def all(cls):
            ...         return tuple(v for k, v in vars(cls).items()
            ...                     if isinstance(v, str) and not k.startswith('_'))
            ...
            >>> AWS.all  # No parentheses!
            ('s3', 's3a')

        With caching for expensive computations:
            >>> class DatabaseSchemes:
            ...     postgres = "postgresql"
            ...     mysql = "mysql"
            ...     sqlite = "sqlite"
            ...
            ...     @classgetter(cache=True)
            ...     def all(cls):
            ...         return tuple(v for k, v in vars(cls).items()
            ...                     if isinstance(v, str) and not k.startswith('_'))
            ...
            >>> DatabaseSchemes.all  # Computed once
            ('postgresql', 'mysql', 'sqlite')
            >>> DatabaseSchemes.all  # Returned from cache
            ('postgresql', 'mysql', 'sqlite')

        Instance access is prevented:
            >>> aws = AWS()
            >>> aws.all = "new_value"
            Traceback (most recent call last):
            ...
            AttributeError: 'all' is a read-only class attribute

        Class-level replacement is allowed (standard Python behavior):
            >>> AWS.all = ("s3", "s3a", "s3n")  # Replaces the descriptor
            >>> AWS.all
            ('s3', 's3a', 's3n')

    Note:
        - Cache is per-class, not per-instance
        - The descriptor pattern ensures lazy evaluation
        - Works naturally with inheritance and subclassing
        - **Instance-level assignment raises AttributeError** (read-only protection)
        - **Class-level assignment replaces the descriptor** (intentional override)
        - The cached values persist for the lifetime of the class
        - Type checkers understand the return type through Generic[T] and overloads

    See Also:
        classgetter: Decorator function for creating ClassGetter instances
    """

    def __init__(
        self,
        fget: Callable[[type], T],
        cache: bool = False,
    ) -> None:
        """
        Initialize the ClassGetter descriptor.

        Args:
            fget: Function that takes a class and returns a value
            cache: If True, cache the computed value per class
        """
        self.fget = fget
        self.cache = cache
        self._cache: dict[type, T] = {}
        functools.update_wrapper(self, fget)
        self.name: str | None = None
        self.owner: type | None = None

    def __set_name__(self, owner: type, name: str) -> None:
        """Store the attribute name when descriptor is bound to a class."""
        self.name = name
        self.owner = owner

    @overload
    def __get__(self, obj: None, objtype: type[ClsT]) -> T: ...

    @overload
    def __get__(self, obj: object, objtype: type[ClsT] | None = None) -> T: ...

    def __get__(self, obj: object | None, objtype: type | None = None) -> T:
        """
        Get the computed value for the class.

        Args:
            obj: Instance (always None for class-level access)
            objtype: The class being accessed

        Returns:
            The computed value from fget(cls)
        """
        if objtype is None:
            if obj is None:
                raise TypeError(f"__get__(None, None) is invalid for {self.__class__.__name__}")
            objtype = type(obj)

        if self.cache and objtype in self._cache:
            return self._cache[objtype]

        value = self.fget(objtype)

        if self.cache:
            self._cache[objtype] = value

        return value

    def __set__(self, obj: object, value: Any) -> None:
        """
        Prevent instance-level assignment.

        Raises:
            AttributeError: Always, as classgetter is read-only
        """
        raise AttributeError(f"{self.name!r} is a read-only class attribute")


# Overloads for classgetter decorator for proper type checking of return type
@overload
def classgetter(
    func: Callable[[type[ClsT]], T],
) -> ClassGetter[T]: ...


@overload
def classgetter(
    func: None = None,
    *,
    cache: bool = False,
) -> Callable[[Callable[[type[ClsT]], T]], ClassGetter[T]]: ...


def classgetter(
    func: Callable[[type], T] | None = None,
    *,
    cache: bool = False,
) -> ClassGetter[T] | Callable[[Callable[[type], T]], ClassGetter[T]]:
    """
    Decorator for read-only class-level properties.

    Creates a ClassGetter descriptor that allows accessing class-level
    computed values without parentheses, similar to @property but for
    class attributes instead of instance attributes.

    The decorated method is read-only: attempting to assign to it on an
    instance will raise AttributeError. However, class-level assignment
    will replace the descriptor entirely (standard Python behavior).

    Can be used with or without arguments:
        @classgetter
        def all(cls): ...

        @classgetter(cache=True)
        def all(cls): ...

    Args:
        func: Function to wrap (when used without arguments)
        cache: If True, cache the computed value per class. Useful for
               expensive computations that don't change at runtime.
               Default: False.

    Returns:
        ClassGetter descriptor instance, or a decorator function if
        called with keyword arguments.

    Examples:
        Basic usage:
            >>> class AWS:
            ...     s3 = "s3"
            ...     s3a = "s3a"
            ...
            ...     @classgetter
            ...     def all(cls):
            ...         return tuple(v for k, v in vars(cls).items()
            ...                     if isinstance(v, str) and not k.startswith('_'))
            ...
            >>> AWS.all  # No parentheses!
            ('s3', 's3a')

        With caching for expensive computations:
            >>> class DatabaseSchemes:
            ...     postgres = "postgresql"
            ...     mysql = "mysql"
            ...     sqlite = "sqlite"
            ...
            ...     @classgetter(cache=True)
            ...     def all(cls):
            ...         return tuple(v for k, v in vars(cls).items()
            ...                     if isinstance(v, str) and not k.startswith('_'))
            ...
            >>> DatabaseSchemes.all  # Computed once
            ('postgresql', 'mysql', 'sqlite')
            >>> DatabaseSchemes.all  # Returned from cache
            ('postgresql', 'mysql', 'sqlite')

        Instance access is prevented:
            >>> aws = AWS()
            >>> aws.all = "new_value"
            Traceback (most recent call last):
            ...
            AttributeError: 'all' is a read-only class attribute

        Class-level replacement is allowed (standard Python behavior):
            >>> AWS.all = ("s3", "s3a", "s3n")  # Replaces the descriptor
            >>> AWS.all
            ('s3', 's3a', 's3n')

    Note:
        - The wrapped function receives the class (not instance) as first argument
        - **Instance assignment is blocked**: obj.attr = value raises AttributeError
        - **Class assignment replaces descriptor**: Class.attr = value is allowed
        - Caching is per-class, so subclasses maintain separate caches
        - The descriptor is created at class definition time (decoration time)
        - Type checkers will understand the return type through proper annotations
        - PyCharm and other Type checkers with weak descriptors inspection may comlain for cls not callable
    """

    def decorator(f: Callable[[type], T]) -> ClassGetter[T]:
        sig = inspect.signature(f)
        params = list(sig.parameters.values())

        if len(params) != 1:
            raise TypeError(
                f"@classgetter expects a function with exactly one parameter (cls), "
                f"but {f.__name__!r} has {len(params)} parameters"
            )

        return ClassGetter(f, cache=cache)

    if func is None:
        return decorator
    else:
        return decorator(func)
