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
from typing_extensions import Self


# Local ----------------------------------------------------------------------------------------------------------------
from .formatters import fmt_type, fmt_value
from .utils import class_name

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
            if len(self.size) != len(self.unit):
                raise ValueError(
                    f"size and unit lists must be same length, but found "
                    f"len(size)={len(self.size)}, len(unit)={len(self.unit)}"
                )

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
                stacklevel=2,
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

        # Objects with __dict__: traverse instance attributes
        elif hasattr(obj, "__dict__"):
            try:
                obj_dict = obj.__dict__
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
            except Exception as e:
                if on_error == "raise":
                    raise
                elif on_error == "warn":
                    warnings.warn(
                        f"Failed to access __dict__ of {obj_type.__module__}.{obj_type.__name__}: {type(e).__name__}: {e}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                if error_counts is not None:
                    error_counts[type(e)] += 1
                if problematic_types is not None:
                    problematic_types.add(obj_type)

        # Objects with __slots__: traverse slot attributes
        elif hasattr(obj, "__slots__"):
            try:
                for slot in obj.__slots__:
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
                if on_error == "raise":
                    raise
                elif on_error == "warn":
                    warnings.warn(
                        f"Failed to access __slots__ of {obj_type.__module__}.{obj_type.__name__}: {type(e).__name__}: {e}",
                        RuntimeWarning,
                        stacklevel=2,
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
                stacklevel=2,
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
        True
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

    except (AttributeError, TypeError):
        # Handle edge cases where attribute access might fail
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
        if format == "list":
            return []
        elif format == "dict":
            return {}
        else:  # tuples
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


def valid_param_types(
    func: F = None,
    *,
    params: list[str] | None = None,
    exclude_self: bool = True,
    exclude_none: bool = False,
    strict: bool = True,
    allow_none: bool = True,
) -> F | Callable[[F], F]:
    """
    Decorator that validates function parameters match their type hints on every call.

    Pre-computes signature and type hints at decoration time for optimal performance.
    Validation happens automatically before the function executes.

    This is the decorator approach for parameter validation. For inline validation
    with more flexibility (conditional, mid-function), use validate_param_types().
    For validating object attributes, use validate_types().

    Can be used with or without arguments:
        @valid_param_types
        def func(x: int): ...

        @valid_param_types(strict=False)
        def func(x: int): ...

    Args:
        func: Function to decorate (when used without arguments)
        params: Optional list of specific parameter names to validate.
                If None, validates all annotated parameters.
        exclude_self: If True, skip validation of 'self' and 'cls' parameters
                      (useful for instance methods and classmethods)
        exclude_none: If True, skip validation for parameters with None values
        strict: If True (default), raise TypeError when encountering Union types that
                cannot be validated with isinstance() (e.g., list[int] | dict[str, int],
                Callable[[int], str] | Callable[[str], int]). If False, silently skip
                such unions. Simple unions like int | str | None are always validated
                regardless of this flag.
        allow_none: If True, None values pass validation for Optional types (T | None).
                    If False, None values must explicitly match the type hint.

    Raises:
        TypeError: If parameter type doesn't match annotation, or if strict=True
                   and a truly unvalidatable Union type is encountered
        RuntimeError: If Python version < 3.11

    Union Type Support:
        **Supported (always validated):**
            - Simple unions: int | str | float
            - Optional types: str | None, int | None
            - Union of basic types: list | dict | tuple

        **Unsupported (skipped or error based on strict flag):**
            - Parameterized generic unions: list[int] | dict[str, int]
            - Callable unions with different signatures: Callable[[int], str] | Callable[[str], int]
            - Protocol unions: SupportsInt | SupportsFloat
            - TypedDict unions: UserDict | AdminDict
            - Literal unions with complex types: Literal[SomeClass.A] | Literal[SomeClass.B]

        When strict=True: Raises TypeError for unsupported unions
        When strict=False: Silently skips validation for unsupported unions

    🚀 Performance:
        ~5-15µs per call (much faster than inline validate_param_types() approach)
        Most work happens at decoration time, minimal runtime overhead
        Recommended for production use when validation logic is fixed

    See Also:
        validate_param_types(): Inline parameter validation (more flexible)
        validate_types(): Validate object attribute types

    Examples:
        >>> # Simple usage (no arguments)
        >>> @valid_param_types
        ... def process(x: int, y: str):
        ...     return f"{x}: {y}"
        ...
        >>> process(42, "hello")  # ✅ Passes
        '42: hello'
        >>> process("invalid", "hello")  # ❌ Raises TypeError
        Traceback (most recent call last):
        ...
        TypeError: type validation failed in process():
          Parameter 'x' must be <int>, got <str>
        >>> # With configuration
        >>> @valid_param_types(strict=False, allow_none=False)
        ... def api_call(user_id: int, token: str | None):
        ...     pass
        ...
        >>> # Selective validation
        >>> @valid_param_types(params=["user_id", "token"])
        ... def endpoint(user_id: int, token: str, debug: bool = False):
        ...     pass  # Only validates user_id and token, skips debug
        >>>
        >>> # Works with instance methods
        >>> class API:
        ...     @valid_param_types
        ...     def process(self, data: int | str):
        ...         pass  # 'self' is automatically excluded
        ...
        >>> # Works with classmethods
        >>> class Factory:
        ...     @classmethod
        ...     @valid_param_types
        ...     def create(cls, config: dict):
        ...         pass  # 'cls' is automatically excluded
        >>>
        >>> # Union types work naturally
        >>> @valid_param_types
        ... def handle(data: int | str | float):
        ...     pass
        ...
        >>> handle(42)      # ✅ int
        >>> handle("text")  # ✅ str
        >>> handle(3.14)    # ✅ float
        >>> handle([1, 2])  # ❌ Raises TypeError
        Traceback (most recent call last):
        ...
        TypeError: type validation failed in handle():
          Parameter 'data' must be <int> | <str> | <float>, got <list>

        >>>
        >>> # For conditional validation, use inline approach instead:
        >>> from c108.abc import validate_param_types
        >>>
        >>> def flexible_validation(x: int, mode: str):
        ...     if mode == "strict":
        ...         validate_param_types()  # More flexible
        ...     # ... rest of function
    """

    def decorator(f: F) -> F:
        # Pre-compute everything at decoration time
        try:
            sig = inspect.signature(f)
        except (ValueError, TypeError) as e:
            raise RuntimeError(
                f"Cannot get signature for {f.__name__}: {e}. "
                f"valid_param_types may not work with this function type."
            )

        # Get type hints
        try:
            type_hints = get_type_hints(f)
        except Exception:
            # Fallback to __annotations__ if get_type_hints fails
            type_hints = getattr(f, "__annotations__", {}).copy()

        # If no type hints, just return the original function
        if not type_hints:
            return f

        # Determine which parameters to validate
        if params is not None:
            params_to_validate = set(params)
        else:
            params_to_validate = set(sig.parameters.keys())

        # Filter out parameters we should skip
        if exclude_self:
            params_to_validate.discard("self")
            params_to_validate.discard("cls")

        # Only validate parameters that have type hints
        params_to_validate = {p for p in params_to_validate if p in type_hints}

        # If nothing to validate, return original function
        if not params_to_validate:
            return f

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Bind arguments to parameter names
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
            except TypeError as e:
                # Let the original TypeError from wrong arguments pass through
                raise

            validation_errors = []

            # Validate each parameter
            for param_name in params_to_validate:
                if param_name not in bound.arguments:
                    continue

                value = bound.arguments[param_name]

                # Skip None values if requested
                if exclude_none and value is None:
                    continue

                expected_type = type_hints[param_name]

                # Reuse the existing validation logic
                error = _validate_obj_type(
                    name=param_name,
                    name_prefix="parameter",
                    value=value,
                    expected_type=expected_type,
                    allow_none=allow_none,
                    strict=strict,
                )

                if error:
                    validation_errors.append(error)

            if validation_errors:
                raise TypeError(
                    f"type validation failed in {f.__name__}():\n  "
                    + "\n  ".join(validation_errors)
                )

            # Call the original function
            return f(*args, **kwargs)

        return wrapper

    # Handle both @valid_param_types and @valid_param_types(...)
    if func is None:
        # Called with arguments: @valid_param_types(strict=True)
        return decorator
    else:
        # Called without arguments: @valid_param_types
        return decorator(func)


def validate_param_types(
    *,
    params: list[str] | None = None,
    exclude_self: bool = True,
    exclude_none: bool = False,
    strict: bool = True,
    allow_none: bool = True,
) -> None:
    """
    Validate that function parameters match their type hints (inline validation).

    Must be called from within a function to inspect its parameters and annotations.
    Uses the calling frame to automatically detect the function and its arguments.

    This is the inline validation approach. For automatic validation via decorator,
    use @valid_param_types instead. For validating object attributes, use validate_types().

    Args:
        params: Optional list of specific parameter names to validate.
                If None, validates all annotated parameters.
        exclude_self: If True, skip validation of 'self' and 'cls' parameters
                      (useful for instance methods and classmethods)
        exclude_none: If True, skip validation for parameters with None values
        strict: If True (default), raise TypeError when encountering Union types that
                cannot be validated with isinstance() (e.g., list[int] | dict[str, int],
                Callable[[int], str] | Callable[[str], int]). If False, silently skip
                such unions. Simple unions like int | str | None are always validated
                regardless of this flag.
        allow_none: If True, None values pass validation for Optional types (T | None).
                    If False, None values must explicitly match the type hint.

    Raises:
        TypeError: If parameter type doesn't match annotation, or if strict=True
                   and a truly unvalidatable Union type is encountered
        RuntimeError: If called outside a function context or Python < 3.11

    Union Type Support:
        **Supported (always validated):**
            - Simple unions: int | str | float
            - Optional types: str | None, int | None
            - Union of basic types: list | dict | tuple

        **Unsupported (skipped or error based on strict flag):**
            - Parameterized generic unions: list[int] | dict[str, int]
            - Callable unions with different signatures: Callable[[int], str] | Callable[[str], int]
            - Protocol unions: SupportsInt | SupportsFloat
            - TypedDict unions: UserDict | AdminDict
            - Literal unions with complex types: Literal[SomeClass.A] | Literal[SomeClass.B]

        When strict=True: Raises TypeError for unsupported unions
        When strict=False: Silently skips validation for unsupported unions

    🚀 Performance:
        ~50-100µs first call, ~10-20µs subsequent calls (with caching in future versions)
        For hot paths, consider using @valid_param_types decorator instead (~5-15µs)

    See Also:
        valid_param_types: Decorator for automatic parameter type validation (faster)
        validate_types(): Validate object attribute types

    💡Examples:
        >>> # Basic usage
        >>> def process_data(user_id: int, name: str | None, score: float = 0.0):
        ...     validate_param_types()
        ...     return f"{user_id}: {name} ({score})"
        ...
        >>> process_data(101, "Alice", 98.5)
        '101: Alice (98.5)'

        >>> process_data("invalid", "Alice", 98.5)  # ❌ Raises TypeError
        Traceback (most recent call last):
        ...
        TypeError: type validation failed in process_data():
          Parameter 'user_id' must be <int>, got <str>

        >>> # Validate only specific parameters
        >>> def api_endpoint(user_id: int, token: str, debug: bool = False):
        ...     validate_param_types(params=["user_id", "token"])  # Skip 'debug'
        ...     # ... rest of function
        >>>
        >>> # Works with instance methods
        >>> class DataProcessor:
        ...     def process(self, data: int | str, strict_mode: bool = False):
        ...         validate_param_types()  # Skips 'self' automatically
        ...         # ... rest of method
        ...
        >>> processor = DataProcessor()
        >>> processor.process(42)  # ✅ Passes
        >>> processor.process(3.14)  # ❌ Raises TypeError
        Traceback (most recent call last):
        ...
        TypeError: type validation failed in process():
          Parameter 'data' must be <int> | <str>, got <float>

        >>> # Conditional validation (advantage over decorator)
        >>> def handle_request(data: dict, mode: str):
        ...     if mode == "strict":
        ...         validate_param_types(strict=True)
        ...     # ... rest of function
        >>>
        >>> # For standardcases, decorator is cleaner:
        >>> from c108.abc import valid_param_types
        >>>
        >>> @valid_param_types
        ... def process(data: int | str):
        ...     return f"Processed {data}"
        >>>
        >>> process(42)
        'Processed 42'

    """
    # Get the calling frame (the function that called validate_param_types)
    frame = inspect.currentframe()
    if frame is None:
        raise RuntimeError("Cannot get current frame")

    caller_frame = frame.f_back
    if caller_frame is None:
        raise RuntimeError("validate_param_types() must be called from within a function")

    try:
        # Get the function object from the caller's frame
        func_name = caller_frame.f_code.co_name

        # Get local variables (includes all parameter values)
        local_vars = caller_frame.f_locals.copy()

        # Try to get the actual function object to access its signature
        # This is tricky because we need to find the function in the right scope
        func = None

        # Try to find function in caller's globals
        if func_name in caller_frame.f_globals:
            func = caller_frame.f_globals[func_name]

        # For methods, try to get from self/cls
        if func is None and "self" in local_vars:
            func = getattr(type(local_vars["self"]), func_name, None)
        if func is None and "cls" in local_vars:
            func = getattr(local_vars["cls"], func_name, None)

        # Locals search
        if func is None:
            # Search locals for a callable with matching code object
            for obj in caller_frame.f_locals.values():
                if (
                    callable(obj)
                    and hasattr(obj, "__code__")
                    and obj.__code__ is caller_frame.f_code
                ):
                    func = obj
                    break

        # Enclosing frames locals search
        if func is None:
            # Search enclosing frames for the function
            search_frame = caller_frame.f_back
            while search_frame is not None:
                for obj in search_frame.f_locals.values():
                    if (
                        callable(obj)
                        and hasattr(obj, "__code__")
                        and obj.__code__ is caller_frame.f_code
                    ):
                        func = obj
                        break
                if func is not None:
                    break
                search_frame = search_frame.f_back

        if func is None or not callable(func):
            raise RuntimeError(
                f"Cannot find function '{func_name}' to inspect its signature. "
                f"validate_param_types() may not work with:\n"
                f"  - Lambdas (use @valid_param_types decorator instead)\n"
                f"  - Functions created via exec() or eval()\n"
                f"  - Dynamically generated functions (e.g., via type() or metaclasses)\n"
                f"  - Certain heavily decorated functions where the wrapper obscures the original\n"
                f"  - Functions in unusual execution contexts (e.g., some REPL environments)\n"
                f"For these cases, use the @valid_param_types decorator for automatic validation."
            )

        # Get type hints
        try:
            type_hints = get_type_hints(func)
        except Exception:
            # Fallback to annotations if get_type_hints fails
            type_hints = getattr(func, "__annotations__", {}).copy()

        if not type_hints:
            # No type hints - nothing to validate
            return

        # Get function signature to match parameters
        sig = inspect.signature(func)

        # Determine which parameters to validate
        if params is not None:
            params_to_validate = params
        else:
            params_to_validate = list(sig.parameters.keys())

        validation_errors = []

        for param_name in params_to_validate:
            # Skip if not in signature
            if param_name not in sig.parameters:
                continue

            # Skip if not in type hints
            if param_name not in type_hints:
                continue

            # Skip 'self' and 'cls' if requested
            if exclude_self and param_name in ("self", "cls"):
                continue

            # Skip if parameter wasn't passed (and has no default that was used)
            if param_name not in local_vars:
                continue

            value = local_vars[param_name]

            # Skip None values if requested
            if exclude_none and value is None:
                continue

            expected_type = type_hints[param_name]

            # Reuse the existing validation logic
            error = _validate_obj_type(
                name=param_name,
                name_prefix="parameter",
                value=value,
                expected_type=expected_type,
                allow_none=allow_none,
                strict=strict,
            )

            if error:
                validation_errors.append(error)

        if validation_errors:
            raise TypeError(
                f"type validation failed in {func_name}():\n  " + "\n  ".join(validation_errors)
            )

    finally:
        # Clean up frame references to avoid reference cycles
        del frame
        del caller_frame


def validate_types(
    obj: Any,
    *,
    attrs: list[str] | None = None,
    exclude_none: bool = False,
    include_inherited: bool = True,
    include_private: bool = False,
    pattern: str | None = None,
    strict: bool = True,
    allow_none: bool = True,
    fast: bool | Literal["auto"] = "auto",
) -> None:
    """
    Validate that object attributes match their type annotations.

    Supports dataclasses, attrs classes, and regular Python classes with
    type annotations. Performance-optimized with a fast path for dataclasses.

    This function validates the types of object attributes. For validating
    function parameters, see validate_param_types() (inline) or @valid_param_types
    (decorator).

    Args:
        obj: Object instance to validate
        attrs: Optional list of specific attribute names to validate.
               If None, validates all annotated attributes.
        exclude_none: If True, skip validation for attributes with None values
        include_inherited: If True, validates inherited attributes with type hints
        include_private: If True, validates private attributes (starting with '_')
        pattern: Optional regex pattern to filter which attributes to validate
        strict: If True (default), raise TypeError when encountering Union types that
                cannot be validated with isinstance() (e.g., list[int] | dict[str, int],
                Callable[[int], str] | Callable[[str], int]). If False, silently skip
                such unions. Simple unions like int | str | None are always validated
                regardless of this flag.
        allow_none: If True, None values pass validation for Optional types (T | None).
                    If False, None values must explicitly match the type hint.
        fast: Performance mode:
              - "auto" (default): Automatically use fast path when possible
              - True: Force fast path, raise ValueError if incompatible options provided
              - False: Force slow path using search_attrs()

    Raises:
        TypeError: If attribute type doesn't match annotation, or if strict=True
                   and a truly unvalidatable Union type is encountered
        ValueError: If obj has no type annotations, or if fast=True with incompatible options
        RuntimeError: If Python version < 3.11

    🚀 Performance:
        Fast path (dataclasses only, ~5-10µs):
            - Requires: is_dataclass(obj)=True, attrs=None, pattern=None, include_private=False
            - 5-10x faster than slow path
            - Recommended for high-throughput production scenarios

        Slow path (all classes, ~30-70µs):
            - Uses search_attrs() for flexible filtering
            - Supports pattern matching, private attrs, custom attr lists
            - Suitable for validation at API boundaries, config loading

    See Also:
        validate_param_types(): Validate function parameter types (inline call)
        valid_param_types: Decorator for automatic parameter type validation

    Examples:
        >>> from dataclasses import dataclass
        >>>
        >>> @dataclass
        ... class ImageData:
        ...     id: str = "qwkfjkqfjhkgwdjhg349893874"
        ...     width: int = 1080
        ...     height: int = 1080
        ...
        ...     def __post_init__(self):
        ...         validate_types(self)  # Auto-uses fast path
        >>>
        >>> obj = ImageData()
        >>>
        >>> # Validate with default settings
        >>> validate_types(obj)
        >>>
        >>> # Force fast path (raises if incompatible)
        >>> validate_types(obj, fast=True)
        >>>
        >>> # Use slow path with pattern matching
        >>> validate_types(obj, pattern=r"^api_.*", fast=False)
        >>>
        >>> # Validate after mutations
        >>> obj.width = "invalid"
        >>> validate_types(obj)  # Raises TypeError
        Traceback (most recent call last):
        ...
        TypeError: type validation failed in <ImageData>:
          Attribute 'width' must be <int>, got <str>

        >>> # For function parameters, use validate_param_types() or @valid_param_types
        >>> def process(x: int, y: str):
        ...     validate_param_types()  # Inline validation
        ...     # ... or use @valid_param_types decorator
    """
    # Determine if we can use fast path for a dataclass
    is_dc = is_dataclass(obj)
    can_use_fast = is_dc and attrs is None and pattern is None and not include_private

    # Validate fast mode compatibility
    if fast is True and not can_use_fast:
        incompatible = []
        if not is_dc:
            incompatible.append("obj is not a dataclass")
        if attrs is not None:
            incompatible.append("attrs parameter is set")
        if pattern is not None:
            incompatible.append("pattern parameter is set")
        if include_private:
            incompatible.append("include_private=True")

        raise ValueError(
            f"cannot use fast=True with current options. "
            f"Fast path is only available for dataclasses without filtering. "
            f"Incompatible settings: {', '.join(incompatible)}. "
            f"Either remove these options or use fast=False or fast='auto'."
        )

    # Choose path
    use_fast = (fast is True) or (fast == "auto" and can_use_fast)

    if use_fast:
        _validate_dataclass_fast(
            obj,
            exclude_none=exclude_none,
            strict=strict,
            allow_none=allow_none,
        )
    else:
        _validate_with_search_attrs(
            obj,
            attrs=attrs,
            exclude_none=exclude_none,
            include_inherited=include_inherited,
            include_private=include_private,
            pattern=pattern,
            strict=strict,
            allow_none=allow_none,
        )


def _validate_dataclass_fast(
    obj: Any,
    *,
    exclude_none: bool,
    strict: bool,
    allow_none: bool,
) -> None:
    """
    Fast path validation for dataclasses without filtering.

    Performance: ~5-10µs per validation

    Optimizations:
    - Uses cached dataclass fields() metadata (no dir() calls)
    - No regex compilation or pattern matching
    - No property detection or callable checks
    - Direct field access only
    - Minimal function calls
    """
    validation_errors = []

    # fields() returns cached metadata - very fast
    for field in dc_fields(obj):
        attr_name = field.name
        expected_type = field.type

        # Direct attribute access
        value = getattr(obj, attr_name)

        # Skip None if requested
        if exclude_none and value is None:
            continue

        # Validate type
        error = _validate_obj_type(
            name=attr_name,
            name_prefix="attribute",
            value=value,
            expected_type=expected_type,
            allow_none=allow_none,
            strict=strict,
        )

        if error:
            validation_errors.append(error)

    if validation_errors:
        raise TypeError(
            f"type validation failed in {fmt_type(obj)}:\n  " + "\n  ".join(validation_errors)
        )


def _validate_obj_type(
    name: str,
    name_prefix: Literal["attribute", "parameter"],
    value: Any,
    expected_type: Any,
    allow_none: bool,
    strict: bool,
) -> str | None:
    """
    Validate a single attribute type. Returns error message or None.

    Extracted to avoid code duplication between fast/slow paths.
    Optimized for hot path performance.

    Args:
        name: Name of the attribute or parameter being validated
        name_prefix: Name prefix ('attribute' or 'parameter')
        value: The actual value to validate
        expected_type: The type annotation to validate against
        allow_none: Whether None is acceptable for Optional types
        strict: Whether to raise errors for truly unvalidatable unions

    Returns:
        Error message string if validation fails, None if validation passes
    """
    # Handle string annotations (should be rare in modern Python)
    if isinstance(expected_type, str):
        if strict:
            return f"{name_prefix.capitalize()} '{name}' has string annotation which cannot be validated"
        return None

    # Get origin for generic/union types
    origin = get_origin(expected_type)

    # Handle Union types (both T | None and Optional[T])
    # UnionType: modern syntax (int | None)
    # Union: old syntax from typing module (Optional[int] or Union[int, None])
    if origin is UnionType or origin is Union:
        args = get_args(expected_type)
        is_optional = type(None) in args
        non_none_types = tuple(t for t in args if t is not type(None))

        if is_optional:
            # Union includes None (e.g., int | None, int | str | None)
            if allow_none and value is None:
                return None  # None is explicitly allowed

            # Validate against non-None types (whether single or multiple)
            try:
                if not isinstance(value, non_none_types):
                    # Format union members nicely
                    type_names = " | ".join(fmt_type(t) for t in non_none_types)
                    if allow_none:
                        type_names += " | None"
                    return (
                        f"{name_prefix.capitalize()} '{name}' must be {type_names}, "
                        f"got {fmt_type(value)}"
                    )
                return None  # Validation passed
            except TypeError:
                # isinstance failed - truly complex union (e.g., list[int] | dict[str, int])
                if strict:
                    return (
                        f"{name_prefix.capitalize()} '{name}' has complex Union type "
                        f"which cannot be validated with isinstance()"
                    )
                return None  # Skip in non-strict mode
        else:
            # Non-Optional Union (e.g., int | str | float)
            try:
                if not isinstance(value, non_none_types):
                    # Format union members nicely
                    type_names = " | ".join(fmt_type(t) for t in non_none_types)
                    return (
                        f"{name_prefix.capitalize()} '{name}' must be {type_names}, "
                        f"got {fmt_type(value)}"
                    )
                return None  # Validation passed
            except TypeError:
                # isinstance failed - truly complex union
                if strict:
                    return (
                        f"{name_prefix.capitalize()} '{name}' has complex Union type "
                        f"which cannot be validated with isinstance()"
                    )
                return None  # Skip in non-strict mode

    # Handle other generic types (list[T], dict[K,V])
    if origin is not None:
        expected_type = origin

    # Final isinstance check for simple types
    try:
        if not isinstance(value, expected_type):
            return (
                f"{name_prefix.capitalize()} '{name}' must be {fmt_type(expected_type)}, "
                f"got {fmt_type(value)}"
            )
    except TypeError:
        # isinstance failed for non-union type
        if strict:
            return f"Cannot validate {name_prefix.lower()} '{name}' with complex type"
        return None  # Skip

    return None  # Valid


def _validate_with_search_attrs(
    obj: Any,
    *,
    attrs: list[str] | None,
    exclude_none: bool,
    include_inherited: bool,
    include_private: bool,
    pattern: str | None,
    strict: bool,
    allow_none: bool,
) -> None:
    """
    Slower path using search_attrs for complex filtering.

    Performance: ~30-70µs per validation

    Used when:
    - Not a dataclass
    - Custom attrs list provided
    - Pattern filtering needed
    - Private attribute inclusion needed
    - Non-inherited attributes only
    """

    # Get type hints
    try:
        if is_dataclass(obj):
            type_hints = {f.name: f.type for f in dc_fields(obj)}
        else:
            # Try get_type_hints first (resolves forward refs)
            try:
                type_hints = get_type_hints(obj.__class__)
            except Exception:
                # Fallback to __annotations__ (doesn't resolve forward refs)
                type_hints = getattr(obj.__class__, "__annotations__", {}).copy()
    except Exception:
        type_hints = {}

    if not type_hints:
        raise ValueError(
            f"Cannot validate {fmt_type(obj)}: no type annotations found. "
            f"Add type hints to class attributes."
        )

    # Determine which attributes to validate
    if attrs is not None:
        attrs_to_validate = attrs
    else:
        attrs_to_validate = search_attrs(
            obj,
            format="list",
            exclude_none=exclude_none,
            include_inherited=include_inherited,
            include_methods=False,
            include_private=include_private,
            include_properties=False,
            pattern=pattern,
            skip_errors=True,
        )

    validation_errors = []

    for attr_name in attrs_to_validate:
        if attr_name not in type_hints:
            continue

        try:
            value = getattr(obj, attr_name)
        except AttributeError:
            continue

        if exclude_none and value is None:
            continue

        expected_type = type_hints[attr_name]

        error = _validate_obj_type(
            name=attr_name,
            name_prefix="attribute",
            value=value,
            expected_type=expected_type,
            allow_none=allow_none,
            strict=strict,
        )

        if error:
            validation_errors.append(error)

    if validation_errors:
        raise TypeError(
            f"type validation failed in {fmt_type(obj)}:\n  " + "\n  ".join(validation_errors)
        )


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
