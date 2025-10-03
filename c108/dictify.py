"""
C108 Dictify Tools
"""

# Standard library -----------------------------------------------------------------------------------------------------
import collections.abc as abc
import inspect
import itertools
import sys

from copy import copy
from dataclasses import asdict, is_dataclass
from enum import Enum, unique
from dataclasses import dataclass, asdict, field, replace as dataclasses_replace
from typing import Any, Dict, Callable, Iterable, List, Type, ClassVar

# Local ----------------------------------------------------------------------------------------------------------------
from .abc import is_builtin, attrs_search, attr_is_property, ObjectInfo, deep_sizeof
from .tools import fmt_any, fmt_type, fmt_value
from .utils import class_name


# Classes --------------------------------------------------------------------------------------------------------------

@unique
class HookMode(str, Enum):
    """
    Object conversion strategy:
        - "dict": try object's to_dict() method, then fallback
        - "dict_strict": require to_dict() method
        - "none": skip object hooks
    """
    DICT = "dict"
    DICT_STRICT = "dict_strict"
    NONE = "none"


class MetaMixin:
    """
    A mixin for Meta-data dataclasses to provide `to_dict` method.
    """

    def to_dict(self,
                include_none_attrs: bool = False,
                include_properties: bool = True,
                sort_keys: bool = False,
                ) -> dict[str, Any]:
        """Convert instance to a dictionary representation.

        The resulting dictionary includes all dataclass fields and the values
        of any public properties.

        Args:
            include_none_attrs: If True, keys with None values are included.
            include_properties: If True, public properties are included.
            sort_keys: If True, the dictionary keys are sorted alphabetically.

        Returns:
            A dictionary representation of the instance.

        Raises:
            TypeError: If instance class is not a dataclass.
        """
        if not is_dataclass(self):
            raise TypeError(f"{self.__class__.__name__} must be a dataclass to use MetaMixin.")

        if include_properties:
            dict_ = asdict(self) | MetaMixin._get_public_properties(self)
        else:
            dict_ = asdict(self)

        if sort_keys:
            dict_ = dict(sorted(dict_.items()))

        if not include_none_attrs:
            return {k: v for k, v in dict_.items() if v is not None}

        return dict_

    @staticmethod
    def _get_public_properties(obj: Any) -> dict[str, Any]:
        """Inspect an object and return a dict of its public property values."""
        properties = {}
        for name in dir(obj.__class__):
            if name.startswith("_"):
                continue

            # Check if the attribute is a property on the class
            if isinstance(getattr(obj.__class__, name), property):
                properties[name] = getattr(obj, name)
        return properties


@dataclass(frozen=True)
class SizeMeta(MetaMixin):
    """Metadata about object size information.

    Attributes:
        len: Object's __len__ if defined.
        deep: Deep size in bytes.
        shallow: Shallow size in bytes of the source object (e.g., sys.getsizeof(obj)).
    """

    len: int | None = None
    deep: int | None = None
    shallow: int | None = None

    def __post_init__(self) -> None:
        """Validate field types, sign, and size relationships."""
        for name in ("len", "deep", "shallow"):
            val = getattr(self, name)
            if val is None:
                continue
            if isinstance(val, bool) or not isinstance(val, int):
                raise TypeError(f"SizeMeta.{name} must be an int, got {fmt_type(val)}")
            if val < 0:
                raise ValueError(f"SizeMeta.{name} must be >=0, but got {fmt_value(val)}")
        if self.deep is not None and self.shallow is not None and self.deep < self.shallow:
            raise ValueError("SizeMeta.deep >= SizeMeta.shallow expected")


@dataclass(frozen=True)
class TrimMeta(MetaMixin):
    """Metadata about collection trimming operations.

    Stores only the source values (len and shown), all other values are computed.

    Attributes:
        len: Total number of elements in the original collection.
        shown: Number of elements kept/shown after trimming.
        is_trimmed: (property) Whether original collection was trimmed.
        trimmed: (property) Number of elements removed due to trimming.
    """
    len: int | None = None
    shown: int | None = None

    def __post_init__(self) -> None:
        """Validate field types, non-negativity, and shown vs len."""
        for name in ("len", "shown"):
            val = getattr(self, name)
            if val is None:
                continue
            if isinstance(val, bool) or not isinstance(val, int):
                raise TypeError(f"TrimMeta.{name} must be an int, got {fmt_type(val)}")
            if val < 0:
                raise ValueError(f"TrimMeta.{name} must be >=0, but got {fmt_value(val)}")
        if self.len is not None and self.shown is not None and self.shown > self.len:
            raise ValueError("TrimMeta.shown <= TrimMeta.len expected")

    @classmethod
    def from_trimmed(cls, total_len: int, trimmed: int) -> "TrimMeta":
        """Create TrimMeta from total length and trimmed items count.

        Args:
            total_len: Total number of elements in the original collection.
            trimmed: Number of elements that were trimmed.

        Returns:
            TrimMeta instance with computed shown value.
        """
        shown = max(total_len - trimmed, 0)
        return cls(len=total_len, shown=shown)

    @property
    def is_trimmed(self) -> bool | None:
        """Whether the collection was trimmed."""
        if self.trimmed is not None:
            return self.trimmed > 0
        return None

    @property
    def trimmed(self) -> int | None:
        """Number of elements removed due to trimming."""
        if self.len is not None and self.shown is not None:
            return self.len - self.shown
        return None


@dataclass(frozen=True)
class TypeMeta(MetaMixin):
    """
    Metadata about type information and conversion.

    Attributes:
        from_type: Type of the original object.
        to_type: Type of the converted object.
    """

    from_type: type | None = None
    to_type: type | None = None

    def __post_init__(self):
        """Set the to_type field if not provided and validate inputs."""
        if self.to_type is None:
            # Use object.__setattr__ to bypass frozen restriction
            object.__setattr__(self, 'to_type', self.from_type)

    @property
    def is_converted(self) -> bool:
        """Check if type conversion occurred."""
        if self.from_type is None and self.to_type is None:
            return False
        return self.from_type != self.to_type

    def to_dict(self,
                include_none_attrs: bool = False,
                include_properties: bool = True,
                sort_keys: bool = False,
                ) -> dict[str, Any]:
        """Convert to dictionary representation.

        The resulting dictionary includes all dataclass fields and the values
        of any public properties.

        Args:
            include_none_attrs: If True, keys with None values are included.
            include_properties: If True, public properties are included.
            sort_keys: If True, the dictionary keys are sorted alphabetically.

        Returns:
            A dictionary representation of the instance.

        Raises:
            TypeError: If the instance class is not a dataclass.
        """
        dict_ = MetaMixin.to_dict(self, include_none_attrs, include_properties, sort_keys)

        if not self.is_converted:
            # When is not converted, to_type is redundant
            dict_.pop("to_type", None)

        return dict_


@dataclass(frozen=True)
class DictifyMeta:
    """
    Comprehensive metadata for dictify conversion operations.

    Contains information about trimming, sizing, and type conversion that
    occurred during object-to-dictionary conversion. Used internally by
    core_dictify() to inject metadata into processed collections and objects.

    Attributes:
        size: Size metadata (shallow bytes, deep bytes, length)
        trim: Collection trimming stats
        type: Type conversion metadata
    """
    VERSION: ClassVar[int] = 1  # Metadata schema version

    size: SizeMeta | None = None
    trim: TrimMeta | None = None
    type: TypeMeta | None = None

    @property
    def has_any_meta(self) -> bool:
        """Check if any metadata is present."""
        return any([self.size, self.trim, self.type])

    @property
    def is_trimmed(self) -> bool | None:
        """Check if the metadata represents a trimmed collection."""
        if self.trim is None:
            return None  # No trim metadata available
        return self.trim.is_trimmed

    def to_dict(self,
                include_none_attrs: bool = False,
                include_properties: bool = True,
                sort_keys: bool = False,
                ) -> dict[str, Any]:
        """
        Convert meta info to dictionary representation.

        If no meta attrs assigned, returns a dict containing meta schema version only.
        """

        dict_ = {}

        if isinstance(self.size, SizeMeta):
            dict_["size"] = self.size.to_dict(include_none_attrs, include_properties, sort_keys)

        if isinstance(self.trim, TrimMeta):
            dict_["trim"] = self.trim.to_dict(include_none_attrs, include_properties, sort_keys)

        if isinstance(self.type, TypeMeta):
            dict_["type"] = self.type.to_dict(include_none_attrs, include_properties, sort_keys)

        dict_["version"] = self.VERSION

        dict_ = dict(sorted(dict_.items())) if sort_keys else dict_

        if include_none_attrs:
            return dict_

        return {k: v for k, v in dict_.items() if v is not None}


@dataclass
class MetaInjectionOptions:
    """
    Metadata injection configuration for dictify operations.

    Controls what metadata gets injected into converted objects, including size information,
    trimming statistics, and type conversion details. Metadata is injected either as a
    dictionary key (for mappings) or appended as the final element (for sequences/sets).

    Attributes:
        key: Dictionary key used for metadata injection in mappings (default: "__dictify__")
        len: Include collection length in size metadata
        size: Include shallow object size in bytes (via sys.getsizeof)
        deep_size: Include deep object size calculation (expensive operation)
        trim: Inject trimming statistics when collections exceed max_items limit
        type: Include type conversion metadata when object types change during processing

    Examples:
        >>> # Enable all metadata
        >>> meta = MetaInjectionOptions(len=True, size=True, deep_size=True, type=True)

        >>> # Only trimming metadata (default)
        >>> meta = MetaInjectionOptions()  # trim=True by default

        >>> # Custom metadata key
        >>> meta = MetaInjectionOptions(key="__meta", trim=True, type=True)
    """
    key: str = "__dictify__"

    # Size-related metadata
    len: bool = False
    size: bool = False  # Shallow size
    deep_size: bool = False  # Deep size (expensive)

    # Operation metadata
    trim: bool = True  # Trimming statistics
    type: bool = False  # Type conversion info

    @property
    def any_enabled(self) -> bool:
        """Check if any metadata injection is enabled."""
        return any([self.sizes_enabled, self.trim, self.type])

    @property
    def sizes_enabled(self) -> bool:
        """Check if any size-related metadata injection is enabled."""
        return any([self.len, self.size, self.deep_size])


@dataclass
class TypeConversionOptions:
    """
    Controls type conversion for collections within the object tree during dictify processing.

    These options determine how various collection types are represented in the final
    dictionary output. The main object is always converted to a dict representation -
    these settings only affect collections found within the object's attributes.

    Attributes:
        keep_tuples: If True, preserve tuple types as tuples in output.
                    If False, convert tuples to lists for JSON compatibility.
                    Default: True (preserves original structure)

        keep_named_tuples: If True, preserve namedtuple instances as namedtuples.
                          If False, convert namedtuples to regular dict representations.
                          Default: True (preserves type information)

        items_view_to_dict: If True, convert dict.items() views to dict format {key: value}.
                           If False, convert to list of tuples [(key, value), ...].
                           Default: True (more readable for debugging)
                           Note: KeysView and ValuesView are always converted to lists

        sets_to_list: If True, convert set and frozenset to lists.
                     If False, attempt to preserve set types (may cause JSON serialization issues).
                     Default: True (ensures JSON compatibility)

        mappings_to_dict: If True, convert custom mapping types (OrderedDict, ChainMap, etc.)
                         to standard dict representations.
                         If False, attempt to preserve original mapping types.
                         Default: True (standardizes output format)

    Examples:
        >>> # Preserve original types where possible
        >>> opts = TypeConversionOptions(
        ...     keep_tuples=True,
        ...     keep_named_tuples=True,
        ...     sets_to_list=False
        ... )

        >>> # Convert everything to basic JSON-safe types
        >>> opts = TypeConversionOptions(
        ...     keep_tuples=False,
        ...     keep_named_tuples=False,
        ...     sets_to_list=True,
        ...     mappings_to_dict=True
        ... )

    Note:
        These conversions are applied recursively to all collections found within
        the object tree. When collections are oversized and require truncation,
        type conversion may be forced regardless of these settings to enable
        the truncation operation.
    """

    # Collection type preservation
    keep_tuples: bool = True  # Keep tuples as tuples vs convert to list
    keep_named_tuples: bool = True  # Keep namedtuples vs convert to dict

    # View conversions
    items_view_to_dict: bool = True  # ItemsView → dict vs list of tuples
    # KeysView, ValuesView always → list (natural representation)

    # Collection conversions
    sets_to_list: bool = True  # set/frozenset → list (JSON-safe)
    mappings_to_dict: bool = True  # OrderedDict, etc. → dict


@dataclass
class DictifyOptions:
    """
    Advanced configuration options for object-to-dictionary conversion with extensive customization.

    Provides comprehensive control over object serialization including recursion depth management,
    attribute filtering, size constraints, custom type handling, and collection processing behavior.
    Supports both debugging and production serialization scenarios with flexible hook systems.

    Core Configuration:
        max_depth: Maximum recursion depth for nested objects (default: 3)
                  - max_depth < 0: Raw mode, uses fn_raw handler
                  - max_depth = 0: Terminal mode, uses fn_terminal handler
                  - max_depth > 0: Normal recursive processing

    Attribute Control:
        include_class_name: Include '__class__' key in object representations
        inject_class_name: Inject '__class__' into to_dict() method results
        include_none_attrs: Include object attributes with None values
        include_none_items: Include dictionary items with None values
        include_private: Include private attributes (starting with _)
        include_properties: Include instance properties with assigned values

    Edge Case Handlers:
        fn_raw: Custom handler for raw mode (max_depth < 0)
               Fallback chain: fn_raw() → obj.to_dict() → identity
        fn_terminal: Custom handler for terminal mode (max_depth = 0)
                    Fallback chain: fn_terminal() → type_handlers → obj.to_dict() → identity

    Size and Performance Limits:
        max_items: Maximum items in collections before trimming (default: 1024)
                  Oversized collections get trimmed with metadata injection
        max_str_length: String truncation limit (default: 256)
        max_bytes: Bytes object truncation limit (default: 1024)

    Mapping keys handling:
        sort_keys: Enable key sorting for mappings

    Meta Data Injection:
        meta: MetaInjectionOptions controlling what metadata gets injected:
              - meta.trim: Trimming statistics for oversized collections
              - meta.type: Type conversion information
              - meta.len/size/deep_size: Object size metadata
              - meta.key: Dictionary key for metadata in mappings

    Advanced Processing:
        hook_mode: Object conversion strategy:
                  - "dict": Try to_dict() with fallback to recursive expansion
                  - "dict_strict": Require to_dict() method (raises if missing)
                  - "none": Skip object hooks, use expansion only
        fully_qualified_names: Use module.Class format vs Class only
        skip_types: Types bypassing all filtering (default: int, float, bool, complex, None)
        type_handlers: Custom type processing functions with inheritance support

    Type Handler System:
        - Supports exact type matching and inheritance-based resolution via MRO
        - Default handlers for: str, bytes, bytearray, memoryview
        - Precedence: type_handlers → obj.to_dict() → recursive expansion
        - Handlers receive (obj, options) and return processed result

    Collection Processing Features:
        - Comprehensive support for Sequences, Mappings, Sets, and MappingViews
        - Automatic trimming with metadata injection for oversized collections
        - Semantic tagging for dict keys(), values(), items() views
        - Dict-like object detection and processing via items() method

    Class Methods:
        debug_options(): Optimized for debugging (shallow depth, private attrs)
        serial_options(): Optimized for serialization (class names, no None values)

    Instance Methods:
        add_type_handler(typ, handler): Register custom type processor (chainable)
        get_type_handler(obj): Retrieve handler via inheritance resolution
        remove_type_handler(typ): Unregister type processor (chainable)

    Properties:
        type_handlers: Dict[Type, Callable] - getter/setter with validation

    Examples:
        >>> # Basic usage with defaults
        >>> options = DictifyOptions()
        >>> result = dictify(my_object, options=options)

        >>> # Debugging configuration - shallow inspection
        >>> debug_opts = DictifyOptions.debug_options()
        >>> debug_result = dictify(complex_object, options=debug_opts)

        >>> # Production serialization
        >>> serial_opts = DictifyOptions.serial_options()
        >>> json_ready = dictify(api_response, options=serial_opts)

        >>> # Custom type handlers with method chaining
        >>> import socket, threading
        >>> options = (
        ...     DictifyOptions()
        ...     .add_type_handler(socket.socket,
        ...                      lambda s, opts: {"type": "socket", "closed": s._closed})
        ...     .add_type_handler(threading.Thread,
        ...                      lambda t, opts: {"name": t.name, "alive": t.is_alive()})
        ... )

        >>> # Custom handlers without defaults
        >>> minimal_opts = (
        ...     DictifyOptions(type_handlers={})  # Empty dict = no default handlers
        ...     .add_type_handler(str, lambda s, opts: s.upper())
        ...     .add_type_handler(dict, lambda d, opts: f"<dict:{len(d)} items>")
        ... )

        >>> # Size-constrained processing
        >>> constrained = DictifyOptions(
        ...     max_items=50,           # Trim large collections
        ...     max_str_length=100,     # Truncate long strings
        ...     max_bytes=512          # Limit byte arrays
        ... )

        >>> # Deep inspection with custom terminal handler
        >>> def custom_terminal(obj, opts):
        ...     return f"<{type(obj).__name__} at depth limit>"
        >>>
        >>> deep_opts = DictifyOptions(
        ...     max_depth=10,
        ...     fn_terminal=custom_terminal
        ... )

    Processing Order:
        1. Skip types (int, float, bool, complex, None) → return as-is
        2. Edge cases (max_depth < 0 or == 0) → use fn_raw/fn_terminal chains
        3. Type handlers → custom processing
        4. Object hooks (to_dict()) → if available and hook_mode allows
        5. Collection processing → sequences, mappings, sets, views
        6. Object expansion → convert to dict with attribute filtering

    Notes:
        - All size limits apply during processing with automatic truncation
        - MRO-based inheritance resolution for type handlers
        - Properties raising exceptions are automatically skipped
        - Class name injection only affects main processing, not edge case handlers
        - Collection trimming injects metadata mapped from DictifyOptions.meta.key or as the last sequence element
    """
    max_depth: int = 3

    include_class_name: bool = False
    inject_class_name: bool = False
    include_none_attrs: bool = True
    include_none_items: bool = True
    include_private: bool = False
    include_properties: bool = False

    # Handlers for recursion edge cases
    fn_raw: Callable[[Any, 'DictifyOptions'], Any] = None
    fn_terminal: Callable[[Any, 'DictifyOptions'], Any] = None

    # Size limits
    max_items: int = 1024
    max_str_length: int = 256
    max_bytes: int = 1024

    # Mapping Keys handling
    sort_keys: bool = False

    # Meta Data Injection
    meta: MetaInjectionOptions = field(default_factory=MetaInjectionOptions)

    # Types handling for collections
    type_opt: TypeConversionOptions = field(default_factory=TypeConversionOptions)

    # Advanced
    hook_mode: str = HookMode.DICT
    fully_qualified_names: bool = False
    skip_types: tuple[type, ...] = (int, float, bool, complex, type(None))

    _type_handlers: Dict[Type, Callable[[Any, 'DictifyOptions'], Any]] = field(
        default_factory=lambda: DictifyOptions.default_type_handlers()
    )

    # Static Methods -----------------------------------

    @staticmethod
    def default_type_handlers() -> Dict[Type, Callable]:
        """
        Get default type handlers for commonly filtered types.

        Returns:
            Dictionary mapping types to their default handler functions
        """
        return {
            str: _handle_str,
            bytes: _handle_bytes,
            bytearray: _handle_bytearray,
            memoryview: _handle_memoryview,
            # TODO range: _handle_range,
        }

    # Class Methods ------------------------------------

    @classmethod
    def debug_options(cls) -> "DictifyOptions":
        """
        Create a DictifyOptions instance configured for debugging.

        Returns:
            DictifyOptions: Configuration optimized for debugging with shallow depth,
                           private attributes included, and higher item limits.
        """
        return cls(
            max_depth=1,
            include_private=True,
            max_items=50
        )

    @classmethod
    def serial_options(cls) -> "DictifyOptions":
        """
        Create a DictifyOptions instance configured for serialization.

        Returns:
            DictifyOptions: Configuration optimized for serialization with class names
                           included and None values excluded.
        """
        return cls(
            include_class_name=True,
            include_none_attrs=False,
            max_str_length=100
        )

    # Methods and Properties ---------------------

    def add_type_handler(self,
                         typ: type,
                         handler: Callable[[Any, "DictifyOptions"], Any],
                         ) -> "DictifyOptions":
        """
        Register or override a handler for a specific type.

        Args:
            typ: The concrete type to process.
            handler: A callable receiving (obj, options) and returning processed value.

        Returns:
            Self, to allow chaining.

        Raises:
            TypeError: If typ, handler, or type_handlers type is invalid.
        """
        if not isinstance(typ, type):
            raise TypeError(f"typ must be a type, got {fmt_type(typ)}")
        if not callable(handler):
            raise TypeError(f"handler must be callable, got {fmt_type(handler)}")

        # Register or override handler for the given type
        self.type_handlers[typ] = handler
        return self

    def get_type_handler(self, obj: Any) -> abc.Callable[[Any, "DictifyOptions"], Any] | None:
        """
        Get the handler function for the object's type (exact or via inheritance).

        Searches for the nearest ancestor via MRO; if ancestors not found, returns
        exact type match or None.

        Args:
            obj: Object to potentially handle.
            options: DictifyOptions instance.

        Returns:
            The handler function if found; otherwise None.
        """
        obj_type = type(obj)
        type_handlers = self.type_handlers

        # Fast path: exact type match
        if obj_type in type_handlers:
            return type_handlers[obj_type]

        # Build candidates that are supertypes of obj_type (robust to non-type keys)
        handler_type_keys = [k for k in type_handlers.keys() if isinstance(k, type)]
        candidates: list[type] = []
        for handler_type in handler_type_keys:
            try:
                if handler_type is not obj_type and issubclass(obj_type, handler_type):
                    candidates.append(handler_type)
            except TypeError:
                # Skip keys that aren't valid types
                continue

        # Prefer the nearest ancestor using the MRO
        if candidates:
            for base in obj_type.__mro__[1:]:
                if base in candidates:
                    return type_handlers[base]

            # Search exact type match
            for k in type_handlers.keys():
                if k in candidates:
                    return type_handlers[k]

        return None

    def remove_type_handler(self, typ: type) -> "DictifyOptions":
        """
        Remove a handler for a specific type.

        Args:
            typ: The concrete type to remove handler for.

        Returns:
            Self, to allow chaining.

        Raises:
            TypeError: If typ or type_handlers type is invalid.
        """
        if not isinstance(typ, type):
            raise TypeError(f"typ must be a type, got {fmt_type(typ)}")

        # Remove handler for the given type if it exists
        self.type_handlers.pop(typ, None)
        return self

    @property
    def type_handlers(self) -> Dict[Type, Callable[[Any, 'DictifyOptions'], Any]]:
        """
        Get the type handlers dictionary.

        Returns:
            Dict mapping types to their handler functions.
        """
        return self._type_handlers

    @type_handlers.setter
    def type_handlers(self, value: abc.Mapping[Type, Callable[[Any, 'DictifyOptions'], Any]] | None) -> None:
        """
        Set the type handlers dictionary.

        Args:
            value: A mapping of types to handler functions, or None to reset to defaults.

        Raises:
            TypeError: If value is not a mapping or None.
        """
        if value is None:
            self._type_handlers = DictifyOptions.default_type_handlers()
        elif isinstance(value, abc.Mapping):
            self._type_handlers = dict(value)
        else:
            raise TypeError(f"type_handlers must be a mapping or None, but got {fmt_type(value)}")


# Private Methods ------------------------------------------------------------------------------------------------------

def create_meta(obj: Any,
                processed_obj: Any,
                opt: DictifyOptions) -> DictifyMeta | None:
    """
    Create metadata object for dictify processing operations.

    Analyzes the original and processed objects to generate comprehensive metadata
    including size information, trimming statistics, and type conversion details.
    Metadata creation is controlled by the flags in opt.meta configuration.

    Args:
        obj: The original object before any processing or trimming operations.
        trimmed_obj: The object after trimming, type conversion, or other processing.
        opt: DictifyOptions instance containing metadata generation flags and limits.

    Returns:
        DictifyMeta object containing requested metadata, or None if no metadata
        was requested or could be generated.

    Example:
        >>> # Create metadata for a trimmed list
        >>> original = list(range(100))
        >>> trimmed = original[:10]
        >>> options = DictifyOptions()
        >>> meta = create_meta(original, trimmed, options)
        >>> print(meta.trim.is_trimmed)
        True
    """

    # Size metadata
    size_meta = None
    if opt.meta.sizes_enabled:

        if opt.meta.len:
            try:
                src_len = len(obj)
            except Exception:
                src_len = None
        if opt.meta.deep_size:
            # This would be expensive - implement deep size calculation
            try:
                src_deep_size = deep_sizeof(obj)
            except Exception:
                src_deep_size = None
        if opt.meta.size:
            try:
                src_shallow_size = sys.getsizeof(obj)
            except Exception:
                src_shallow_size = None

        size_meta = SizeMeta(len=src_len, deep=src_deep_size, shallow=src_shallow_size)

    # Trim metadata
    trim_meta = None
    if opt.meta.trim and _is_sized_iterable(obj) and _is_sized_iterable(processed_obj):
        # Calculate trim metadata
        src_len, dest_len = len(obj), len(processed_obj)
        trim_meta = TrimMeta(len=src_len, shown=dest_len) if src_len > dest_len else None

    # Type conversion metadata
    type_meta = None
    if opt.meta.type:
        type_meta = TypeMeta(from_type=type(obj), to_type=type(processed_obj))

    if any([size_meta, trim_meta, type_meta]):
        return DictifyMeta(size=size_meta, trim=trim_meta, type=type_meta)

    return None


def inject_meta(obj: Any, meta: DictifyMeta, opt: DictifyOptions) -> Any:
    """
    Inject metadata into object based on its type.
    """

    if meta is None:
        return obj

    meta_dict = meta.to_dict(include_none_attrs=opt.include_none_attrs,
                             include_properties=opt.include_properties,
                             sort_keys=opt.sort_keys)

    # For mappings, inject under meta key
    if isinstance(obj, dict):
        obj[opt.meta.key] = meta_dict
        return obj

    # For sequences/sets converted to lists, append metadata
    elif isinstance(obj, list):
        obj.append({opt.meta.key: meta_dict})
        return obj

    # For other types, wrap in a dict
    else:
        return {
            "value": obj,
            opt.meta.key: meta_dict
        }


# Methods --------------------------------------------------------------------------------------------------------------

def core_dictify(obj: Any,
                 *,
                 fn_raw: Callable[[Any, 'DictifyOptions'], Any] = None,
                 fn_terminal: Callable[[Any, 'DictifyOptions'], Any] | None = None,
                 options: DictifyOptions | None = None, ) -> Any:
    """
    Advanced object-to-dictionary conversion engine with comprehensive configurability.

    Core engine powering dictify() and serial_dictify() with full control over conversion
    behavior. Converts arbitrary Python objects to human-readable dictionary representations
    while preserving primitive types, handling collections intelligently, and providing
    extensive customization through DictifyOptions and processing hooks.

    Processing Pipeline:
        1. Skip Type Bypass: Objects in skip_types return unchanged
        2. Edge Case Handling:
           - max_depth < 0: Raw mode via fn_raw chain
           - max_depth = 0: Terminal mode via fn_terminal chain
        3. Type Handler Resolution: Custom processors via inheritance hierarchy
        4. Object Hook Processing: to_dict() method calls based on hook_mode
        5. Collection Processing: Sequences, mappings, sets, and views
        6. Object Expansion: Attribute extraction with filtering rules

    Args:
        obj: Any Python object to convert to dictionary representation
        fn_raw: Custom handler for raw processing mode (max_depth < 0).
               Fallback chain: fn_raw() → obj.to_dict() → identity function
        fn_terminal: Custom handler for terminal processing (max_depth = 0).
                    Fallback chain: fn_terminal() → type_handlers → obj.to_dict() → identity
        options: DictifyOptions instance controlling all conversion behaviors.
                Default DictifyOptions() used if None.

    Returns:
        Human-readable dictionary representation of the object, or processed result
        from custom handlers.

    Raises:
        TypeError: Invalid types for options, fn_raw, or fn_terminal parameters
        TypeError: hook_mode='dict_strict' if object lacks to_dict() method
        ValueError: Invalid hook_mode value specified

    Handler Precedence (Normal Processing, max_depth > 0):
        1. Type handlers (exact type or inheritance-based via MRO)
        2. Object to_dict() method (controlled by hook_mode)
        3. Collection/mapping/sequence recursive processing
        4. Object attribute expansion with filtering

    Edge Case Processing:
        Raw Mode (max_depth < 0):
            fn_raw() → obj.to_dict() → return obj unchanged

        Terminal Mode (max_depth = 0):
            fn_terminal() → type_handlers → obj.to_dict() → return obj unchanged

    Collection Processing Features:
        - Automatic size limiting with optional metadata injection
        - Comprehensive support for all Collection/MappingView types:
          * Sequences (list, tuple, str, bytes, etc.)
          * Mappings (dict, OrderedDict, etc.)
          * Sets (set, frozenset, etc.)
          * MappingViews (dict.keys(), dict.values(), dict.items())
          * Dict-like objects (custom classes with items() method)
        - Mapping keys skip recursive expansion

    Metadata Injection Features for recursive processing:
        - Injection based on detailed options.meta flags
        - Sequences/Sets: Meta appended as the final element
        - Mappings: Meta added under options.meta.key
        - Views: Converted to dict structure with optional metadata
        - Trimming meta for oversized collections (len > max_items) when options.meta.trim enabled
        - TODO Meta injection rules for User Objects converted to dicts?
        - TODO we can inject Meta after obj.to_dict() calls?
        - TODO remove Semantic tagging from MappingViews presentation? We can already have it in Metadata?
        - No Metadata injection in default fn_raw, fn_terminal, and type_handlers

    Object Expansion Rules:
        - Private attributes included only if include_private=True
        - Properties included only if include_properties=True and accessible
        - None values filtered based on include_none_attrs setting
        - Class name injection controlled by include/inject_class_name options
        - Meta injection controlled by options.meta flags
        - Attribute access exceptions automatically handled and skipped

    Examples:
        >>> # Basic conversion with defaults
        >>> result = core_dictify(my_object)

        >>> # Custom depth and terminal handling
        >>> def terminal_handler(obj, opts):
        ...     return f"<{type(obj).__name__}:truncated>"
        >>>
        >>> opts = DictifyOptions(max_depth=5)
        >>> result = core_dictify(deep_object,
        ...                      fn_terminal=terminal_handler,
        ...                      options=opts)

        >>> # Raw mode processing
        >>> raw_opts = DictifyOptions(max_depth=-1)
        >>> raw_result = core_dictify(obj, options=raw_opts)  # Minimal processing

        >>> # Collection size management
        >>> size_opts = DictifyOptions(max_items=100, max_str_length=50)
        >>> trimmed_result = core_dictify(large_collection, options=size_opts)

        >>> # Custom type handling with inheritance
        >>> class DatabaseConnection: pass
        >>> class PostgresConnection(DatabaseConnection): pass
        >>>
        >>> opts = (
        ...     DictifyOptions()
        ...     .add_type_handler(DatabaseConnection,
        ...                      lambda conn, opts: {"type": "db", "active": True})
        ... )
        >>> # PostgresConnection inherits DatabaseConnection handler
        >>> result = core_dictify(PostgresConnection(), options=opts)

        >>> # Strict object hook mode
        >>> strict_opts = DictifyOptions(hook_mode="dict_strict")
        >>> # Raises TypeError if obj lacks to_dict() method
        >>> result = core_dictify(obj_with_to_dict, options=strict_opts)

    Special Behaviors:
        - max_depth parameter controls recursion: N levels deep for collections,
          with object attributes processed at depth N-1
        - Skip types (int, float, bool, complex, None, range) bypass all processing
        - Default type handlers process str, bytes, bytearray, memoryview with size limits
        - Class name inclusion affects main processing only, not edge case handlers
        - Key sorting (if enabled) applies to main processing and to_dict() injection
        - Sets are converted to lists
        - Exception-raising properties automatically skipped during object expansion
        - MRO-based type handler resolution supports inheritance hierarchies

    Performance Notes:
        - Collection trimming prevents memory issues with large datasets
        - Type handler caching optimizes repeated conversions
        - Shallow copying for depth management minimizes overhead
        - Early returns for skip types and edge cases improve efficiency
    """
    if not isinstance(options, (DictifyOptions, type(None))):
        raise TypeError(f"options must be a DictifyOptions instance, but found {fmt_type(options)}")
    if not isinstance(fn_raw, (Callable, type(None))):
        raise TypeError(f"fn_raw must be a Callable, but found {fmt_type(fn_raw)}")
    if not isinstance(fn_terminal, (Callable, type(None))):
        raise TypeError(f"fn_terminal must be a Callable or None, but found {fmt_type(fn_terminal)}")

    # Use defaults if no options provided
    opt = options or DictifyOptions()
    opt.fn_raw = fn_raw or opt.fn_raw
    opt.fn_terminal = fn_terminal or opt.fn_terminal

    # core_dictify() body Start ---------------------------------------------------------------------------

    # Return skip_type objects as is -----------------
    if _is_skip_type(obj, options=opt):
        return obj

    # Edge Cases processing --------------------------
    if not isinstance(opt.max_depth, int):
        raise TypeError(f"Recursion depth must be int but found: {fmt_any(opt.max_depth)}")
    if opt.max_depth < 0:
        return _fn_raw_chain(obj, opt=opt)
    if opt.max_depth == 0:
        return _fn_terminal_chain(obj, opt=opt)

    # Type handling and obj.to_dict() processors -----
    if type_handler := opt.get_type_handler(obj):
        return type_handler(obj, opt)

    if dict_ := _get_from_to_dict(obj, opt=opt):
        return dict_

    # Should check sized iterables for recursion: list, tuple, set, dict, etc
    if isinstance(obj, (abc.Sized)):
        if not _is_sized_iterable(obj):
            # We should handle gracefully special cases when __iter__ or __len__ not implemented
            return _fn_terminal_chain(obj, opt=opt)
        else:
            return _process_sized_iterable(obj, max_depth=opt.max_depth, opt=opt, source_object=None)

    # We should make a dict from obj if it is NOT a collection or view
    # and go by 1 level into deep.
    dict_ = _shallow_to_dict(obj, opt=opt)
    return _process_sized_iterable(dict_, max_depth=opt.max_depth, opt=opt, source_object=obj)

    # core_dictify() body End ---------------------------------------------------------------------------


def _class_name(obj: Any, options: DictifyOptions) -> str:
    """Return instance or type class name"""
    return class_name(obj,
                      fully_qualified=options.fully_qualified_names,
                      fully_qualified_builtins=False,
                      start="", end="")


def _core_dictify(obj, max_depth: int, opt: DictifyOptions):
    """Return core_dictify() overriding opt.max_depth"""
    opt = copy(opt) or DictifyOptions()
    opt.max_depth = max_depth
    return core_dictify(obj, options=opt)


def _process_sized_iterable(obj: abc.Collection[Any] | abc.MappingView,
                            max_depth: int,
                            opt: DictifyOptions,
                            source_object: Any = None) -> Any:
    """
    Route collection processing to dedicated handlers based on type and trim status.
    """
    _validate_sized_iterable(obj)

    if max_depth < 0:
        return _fn_raw_chain(obj, opt=opt)
    if max_depth == 0:
        return _fn_terminal_chain(obj, opt=opt)

    # Route obj to appropriate processor
    if isinstance(obj, abc.KeysView):
        return _proc_keys_view(obj, max_depth, opt)
    elif isinstance(obj, abc.ValuesView):
        return _proc_values_view(obj, max_depth, opt)
    elif isinstance(obj, abc.ItemsView):
        return _proc_items_view(obj, max_depth, opt)
    elif isinstance(obj, abc.Sequence):
        return _proc_sequence(obj, max_depth, opt)
    elif isinstance(obj, abc.Set):
        return _proc_set(obj, max_depth, opt)
    elif isinstance(obj, abc.Mapping):
        return _proc_dict(obj, max_depth, opt, source_object)
    elif _is_dict_like(obj):
        return _proc_dict_like(obj, max_depth, opt, source_object)
    else:
        raise TypeError(f"Unsupported collection type: {fmt_type(obj)} "
                        f"Consider converting to stdlib Collection/View or provide a dedicated type_handler")


def _process_trim_oversized(obj: abc.Sized, opt: DictifyOptions) -> Any:
    """
    Preprocessor that trims oversized collections/views and injects stats.

    # TODO if required a type conversion, it should be done by a dedicated method before this trimming?
    #      trimming should NOT change type?

    Returns a trimmed version of the collection (no metadata injected):
    - Sequences/Sets: Items as list, Meta as last element
    - Mappings: Meta mapped from opt.meta.key
    - MappingViews: Convert to dict first, then optionally add Meta mapped from opt.meta.key

    Args:
        obj: Collection or View
        opt: DictifyOptions instance

    Returns:
        Tuple of (trimmed collection with stats injected, from_type)

    Raises:
        TypeError: If obj doesn't implement required methods
        ValueError: If max_items is invalid
    """
    _validate_sized_iterable(obj)

    if opt.max_items <= 0:
        raise ValueError(f"max_items must be positive, but found: {opt.max_items}")

    total_len = len(obj)
    if total_len <= opt.max_items:
        return obj  # No trimming required

    # Reserve one slot for stats
    items_to_show = max(1, opt.max_items - 1)

    # Handle MappingViews - convert to dict structure with stats
    if isinstance(obj, abc.KeysView):
        keys = list(obj)[:items_to_show]
        return {"keys": keys}

    elif isinstance(obj, abc.ValuesView):
        values = list(obj)[:items_to_show]
        return {"values": values}

    elif isinstance(obj, abc.ItemsView):
        items = list(obj)[:items_to_show]
        return {"items": items}

    # Handle dict-like types with .items() - convert to dict with stats
    elif hasattr(obj, "items") and callable(getattr(obj, "items")) and not isinstance(obj, (dict, abc.Mapping)):
        try:
            # Take up to items_to_show (key, value) pairs from the object's items iterator.
            trimmed_items = dict(itertools.islice(obj.items(), items_to_show))
            return trimmed_items
        except (AttributeError, TypeError):
            # Fallback - treat below as sequence
            pass

    # Handle standard Mappings - add stats under special key
    if isinstance(obj, (dict, abc.Mapping)):
        # Take up to items_to_show items from the mapping in iteration order, then append stats.
        trimmed_items = dict(itertools.islice(obj.items(), items_to_show))
        return trimmed_items

    # Handle Sequences - add stats as last element
    elif isinstance(obj, abc.Sequence):
        trimmed_items = list(obj)[:items_to_show]
        return (type(obj)(trimmed_items) if hasattr(type(obj), '__call__') \
                    else trimmed_items)

    # Handle Sets - add stats as element (convert to list to maintain order)
    elif isinstance(obj, abc.Set):
        # Take up to items_to_show elements from the set, then append stats.
        trimmed_items = list(itertools.islice(obj, items_to_show))
        # Return as list since we can't guarantee stats dict is hashable for set
        return trimmed_items

    else:
        # Fallback - treat as generic iterable, return as list
        trimmed_items = list(itertools.islice(obj, items_to_show))
        return trimmed_items


def _proc_sequence(obj: abc.Sequence, max_depth: int, opt: DictifyOptions) -> Any:
    """Process standard sequences with type conversion options"""
    as_list = [_core_dictify(item, max_depth - 1, opt) for item in obj]

    # Check conversion options
    if isinstance(obj, tuple):
        if opt.type_opt.keep_tuples:
            return tuple(as_list)  # Simple constructor works for tuple
        else:
            return as_list  # Return as list

    # Handle named tuples
    if hasattr(obj, '_fields'):  # Duck typing for namedtuple
        if opt.type_opt.keep_named_tuples:
            try:
                return type(obj)(*as_list)  # Use *args for namedtuple
            except (TypeError, ValueError):
                # Fallback to dict representation for namedtuples
                return dict(zip(obj._fields, as_list))
        else:
            return dict(zip(obj._fields, as_list))

    # For other sequences, try to preserve type or fall back to list
    try:
        return type(obj)(as_list)
    except (TypeError, ValueError):
        return as_list


def _proc_set(obj: abc.Set, max_depth: int, opt: DictifyOptions) -> List:
    """
    Process standard sets (set, frozenset, etc.)

    Processed elements returned as list to avoid hash collision
    """
    as_list = [_core_dictify(item, max_depth - 1, opt) for item in obj]
    return as_list


def _proc_dict(obj: abc.Mapping, max_depth: int, opt: DictifyOptions, source_object: Any) -> dict:
    """
    Process mappings (dict, etc.)

    Always converts to stdlib dict.
    """
    include_nones = opt.include_none_attrs if source_object else opt.include_none_items
    dict_ = {
        k: _core_dictify(v, max_depth - 1, opt)
        for k, v in obj.items()
        if (v is not None) or include_nones
    }
    if opt.include_class_name:
        dict_["__class__"] = _class_name(source_object or type(obj), opt)
    if opt.sort_keys:
        dict_ = dict(sorted(dict_.items()))
    return dict_


def _proc_dict_like(obj: Any, max_depth: int, opt: DictifyOptions, source_object: Any) -> dict:
    """
    Process dict-like objects with .items() (e.g., OrderedDict, frozendict, special implementations of mappings)

    Always converts to stdlib dict. Returns object info if processing fails.
    """
    if hasattr(obj, 'items') and callable(getattr(obj, 'items')):
        try:
            return _proc_dict(obj, max_depth, opt, source_object)
        except (TypeError, ValueError, AttributeError):
            # items() exists but doesn't work as expected
            pass

    # Fallback: return object info instead of raising Exceptions
    result = {
        '__type__': type(obj).__name__,
        '__module__': getattr(type(obj), '__module__', 'unknown'),
        '__repr__': repr(obj)[:200],  # Truncate long reprs
    }

    # Try to get some basic info about the object
    try:
        result['__len__'] = len(obj)
    except:
        pass

    try:
        result['__str__'] = str(obj)[:200]  # Truncate long strings
    except:
        pass

    if opt.include_class_name:
        result["__class__"] = _class_name(source_object or type(obj), opt)

    return result


def _proc_keys_view(obj: abc.KeysView, max_depth: int, opt: DictifyOptions) -> dict:
    """Process dict_keys view"""
    from_type = type(obj)
    keys = [k for k in obj]
    dict_ = {"keys": keys}
    if opt.include_class_name:
        dict_["__class__"] = _class_name(from_type, opt)
    if opt.sort_keys:
        dict_ = dict(sorted(dict_.items()))
    return dict_


def _proc_values_view(obj: abc.ValuesView, max_depth: int, opt: DictifyOptions) -> dict:
    """Process dict_values view"""
    from_type = type(obj)
    values = [_core_dictify(val, max_depth - 1, opt) for val in obj]
    dict_ = {"values": values}
    if opt.include_class_name:
        dict_["__class__"] = _class_name(from_type, opt)
    if opt.sort_keys:
        dict_ = dict(sorted(dict_.items()))
    return dict_


def _proc_items_view(obj: abc.ItemsView, max_depth: int, opt: DictifyOptions) -> dict:
    """Process dict_items view"""
    from_type = type(obj)
    items = [(k, _core_dictify(v, max_depth - 1, opt)) for k, v in obj]
    dict_ = {"items": items}
    if opt.include_class_name:
        dict_["__class__"] = _class_name(from_type, opt)
    if opt.sort_keys:
        dict_ = dict(sorted(dict_.items()))
    return dict_


def _fn_raw_chain(obj: Any, opt: DictifyOptions) -> Any:
    """
    fn_raw chain of object processors with priority order
    fn_raw() > obj.to_dict() > identity function
    """
    opt = opt or DictifyOptions()
    if opt.fn_raw is not None:
        return opt.fn_raw(obj, opt)
    if dict_ := _get_from_to_dict(obj, opt=opt) is not None:
        return dict_
    return object  # Final fallback


def _fn_terminal_chain(obj: Any, opt: DictifyOptions) -> Any:
    """
    fn_terminal chain of object processors with priority order
    fn_terminal() > type_handlers > obj.to_dict() > identity function
    """
    opt = opt or DictifyOptions()
    if opt.fn_terminal is not None:
        return opt.fn_terminal(obj, opt)
    if type_handler := opt.get_type_handler(obj):
        return type_handler(obj, opt)
    if dict_ := _get_from_to_dict(obj, opt=opt) is not None:
        return dict_
    return obj  # Final fallback


def _get_from_to_dict(obj, opt: DictifyOptions | None = None) -> dict[Any, Any] | None:
    """Returns obj.to_dict() value if the method is available and hook mode allows"""

    opt = opt or DictifyOptions()

    # Process HookMode ------------------------------------------

    if opt.hook_mode == HookMode.DICT:
        fn = getattr(obj, "to_dict", None)
        dict_ = fn() if callable(fn) else None

    elif opt.hook_mode == HookMode.DICT_STRICT:
        fn = getattr(obj, "to_dict", None)
        if not callable(fn):
            raise TypeError(f"Class {fmt_type(obj)} must implement to_dict() when hook_mode='{HookMode.DICT_STRICT}'")
        dict_ = fn()

    elif opt.hook_mode == HookMode.NONE:
        dict_ = None

    else:
        valid = ", ".join([f"'{v.value}'" for v in HookMode])
        raise ValueError(f"Unknown hook_mode value: {fmt_any(opt.hook_mode)}. Expected: {valid}")

    # Check the returned type -----------------------------------
    if dict_ is not None:
        if not isinstance(dict_, abc.Mapping):
            raise TypeError(f"Object's to_dict() must return a Mapping, but got {fmt_type(dict_)}")

        # returned mapping should be of dict type
        dict_ = {**dict_}
        if opt.inject_class_name:
            dict_["__class__"] = _class_name(obj, options=opt)
        if opt.sort_keys:
            dict_ = dict(sorted(dict_.items()))

    return dict_


def _is_iterable(obj: abc.Sized) -> bool:
    """Return True if `obj` can be iterated with iter()."""
    try:
        iter(obj)
    except TypeError:
        return False
    else:
        return True


def _is_sized(obj: abc.Sized) -> bool:
    """Returns True if obj implements __len__"""
    try:
        len(obj)
        return True
    except TypeError:
        return False


def _is_sized_iterable(obj: Any) -> bool:
    """
    Returns True if obj is a sized iterable (e.g. Collection or MappingView with __iter__ and __len__)

    Tries to iterate and to get length, does not rely on plain type checks.
    """
    if not isinstance(obj, (abc.Sized)):
        return False
    if not _is_iterable(obj):
        return False
    if not _is_sized(obj):
        return False
    return True


def _is_dict_like(original_obj_type: type | Any) -> bool:
    return hasattr(original_obj_type, 'items') and bool(callable(getattr(original_obj_type, 'items')))


def _is_skip_type(obj: Any, options: DictifyOptions) -> bool:
    """
    Check obj against types skipped in processing
    """
    if isinstance(obj, tuple(options.skip_types)):
        return True
    else:
        return False


def _handle_str(obj: str, options: DictifyOptions) -> str:
    """Default handler for str objects - applies max_str_length limit."""
    if len(obj) > options.max_str_length:
        return obj[:options.max_str_length] + "..."
    return obj


def _handle_bytes(obj: bytes, options: DictifyOptions) -> bytes:
    """Default handler for bytes objects - applies max_bytes limit."""
    if len(obj) > options.max_bytes:
        return obj[:options.max_bytes] + b"..."
    return obj


def _handle_bytearray(obj: bytearray, options: DictifyOptions) -> bytearray:
    """Default handler for bytearray objects - applies max_bytes limit."""
    if len(obj) > options.max_bytes:
        truncated = obj[:options.max_bytes]
        return bytearray(truncated + b"...")
    return obj


def _handle_memoryview(obj: memoryview, options: DictifyOptions) -> dict[str, Any]:
    """Default handler for memoryview objects - converts to descriptive dictionary."""
    result = {
        'type': 'memoryview',
        'nbytes': len(obj),
        'format': obj.format,
        'readonly': obj.readonly,
        'itemsize': obj.itemsize,
    }

    # Add shape info if available (for multi-dimensional views)
    if hasattr(obj, 'shape') and obj.shape is not None:
        result['shape'] = obj.shape
        result['ndim'] = obj.ndim
        result['strides'] = obj.strides if hasattr(obj, 'strides') else None

    # Add data preview if reasonably sized
    if len(obj) <= options.max_bytes:
        result['data'] = obj.tobytes()
    elif options.max_bytes > 0:
        preview_size = min(options.max_bytes // 2, 64)
        try:
            preview_data = obj[:preview_size].tobytes()
            result['data_preview'] = preview_data
            result['data_truncated'] = True
        except (ValueError, TypeError):
            result['data_preview'] = None
            result['data_truncated'] = True

    return result


def _merge_options(options: DictifyOptions | None, **kwargs) -> DictifyOptions:
    opt = DictifyOptions(**kwargs) if options is None else dataclasses_replace(options, **kwargs)
    return opt


def _shallow_to_dict(obj: Any, *, opt: DictifyOptions = None) -> dict[str, Any]:
    """
    Shallow object to dict converter for user objects.

    This method generates a dictionary with attributes and their corresponding values for a given object or class.
    No recursion supported.

    Method is NOT intended for primitives or builtins processing (list, tuple, dict, etc)

    Returns:
        dict[str, Any] - dictionary containing attributes and their values

    Note:
        include_none_attrs: Include attributes with None values
        include_private: Include private attributes of user classes
        include_property: Include instance properties with assigned values, has no effect if obj is a class
    """
    dict_ = {}

    attributes = attrs_search(obj, include_private=opt.include_private, include_property=opt.include_properties,
                              include_none_attrs=opt.include_none_attrs)
    is_class_or_dataclass = inspect.isclass(obj)

    for attr_name in attributes:
        if attr_name.startswith('__') and attr_name.endswith('__'):
            continue  # Skip dunder methods

        is_obj_property = attr_is_property(attr_name, obj)

        if not is_class_or_dataclass and is_obj_property:
            if not opt.include_properties:
                continue  # Skip properties

            try:
                attr_value = getattr(obj, attr_name)
            except Exception:
                continue  # Skip if instance property getter raises exception

        else:
            attr_value = getattr(obj, attr_name)

        if callable(attr_value):
            continue  # Should skip methods (properties see above)

        dict_[attr_name] = attr_value

    if opt.include_class_name:
        dict_["__class__"] = _class_name(obj, options=opt)

    if opt.sort_keys:
        dict_ = dict(sorted(dict_.items()))

    return dict_


def _to_str(obj: Any, opt: DictifyOptions) -> str:
    """
    Returns <str> value of object, overrides default stdlib __str__.

    If custom __str__ method not found, replaces the stdlib __str__ with optionally Fully Qualified Class Name.
    """
    has_default_str = obj.__str__ == object.__str__
    if not has_default_str:
        as_str = str(obj)
    else:
        cls_name = class_name(obj, fully_qualified=opt.fully_qualified_names,
                              fully_qualified_builtins=False,
                              start="", end="")
        as_str = f"<class {cls_name}>"
    return as_str


def _validate_sized_iterable(obj: Any):
    if not _is_sized_iterable(obj):
        raise TypeError(
            f"obj must be a Collection, MappingView or derived from Sized and implement __iter__, __len__ methods, "
            f"but found {fmt_type(obj)}")


def dictify(obj: Any, *,
            max_depth: int = 3,
            include_private: bool = False,
            include_class_name: bool = False,
            max_items: int = 50,
            options: DictifyOptions | None = None) -> Any:
    """
    Simple object-to-dict conversion with common customizations.

    Converts Python objects to human-readable dictionary representations, preserving
    built-in types while converting custom objects to dictionaries. This is a simplified
    interface to core_dictify() with sensible defaults for common use cases like debugging,
    logging, and data inspection.

    Args:
        obj: Object to convert to dictionary representation
        max_depth: Maximum recursion depth for nested objects
        include_private: Include private attributes starting with underscore
        include_class_name: Include class name in object representations
        max_items: Maximum number of items in collections
        options: DictifyOptions instance for advanced configuration; individual parameters
                passed to dictify() override corresponding options fields

    Returns:
        Human-readable data representation preserving built-in types and converting
        objects to dictionaries

    Raises:
        TypeError: If max_depth or max_items are not integers
        TypeError: If options is not a DictifyOptions instance or None

    Examples:
        # Basic object conversion
        >>> class Person:
        ...     def __init__(self, name, age):
        ...         self.name = name
        ...         self.age = age
        >>> p = Person("Alice", 7)
        >>> dictify(p)
        {'name': 'Alice', 'age': 7}

        # Include class information for debugging
        >>> dictify(p, include_class_name=True)
        {'__class__': 'Person', 'name': 'Alice', 'age': 30}

        # Control recursion depth
        >>> nested = {'users': [p], 'count': 1}
        >>> dictify(nested, max_depth=1)
        {'users': [{'name': 'Alice', 'age': 30}], 'count': 1}

        # Advanced configuration with options
        >>> from c108.dictify import DictifyOptions
        >>> opts = DictifyOptions(include_none_attrs=False, max_str_length=20)
        >>> dictify(p, max_depth=2, options=opts)

    Note:
        - Built-in types (int, str, list, dict, etc.) are preserved as-is
        - Custom objects are converted to shallow dictionaries at processing boundaries
        - Collections are recursively processed up to max_depth levels
        - Effective inspection depth may be max_depth + 1 for object attributes due to
          shallow dictionary conversion
        - Properties that raise exceptions are automatically skipped
        - For advanced control, use core_dictify() directly

    See Also:
        core_dictify: Advanced conversion with full configurability
        serial_dictify: Serialization-focused conversion with string handling
    """

    if not isinstance(max_depth, int):
        raise TypeError(f"max_depth must be int but found: {fmt_any(max_depth)}")
    if not isinstance(max_items, int):
        raise TypeError(f"max_items must be int but found: {fmt_any(max_items)}")
    if not isinstance(options, (DictifyOptions, type(None))):
        raise TypeError(f"options must be a DictifyOptions instance, but found {fmt_type(options)}")

    include_private = bool(include_private)
    include_class_name = bool(include_class_name)
    opt = _merge_options(options,
                         max_depth=max_depth,
                         include_private=include_private,
                         include_class_name=include_class_name,
                         inject_class_name=include_class_name,
                         max_items=max_items,
                         )

    def __dictify_process(obj: Any) -> Any:
        # Return never-filtered objects
        if _is_skip_type(obj, options=opt):
            return obj
        # Should convert to dict the topmost obj level but keep its inner objects as-is
        dict_ = _shallow_to_dict(obj, opt=opt)
        return dict_

    return core_dictify(obj,
                        fn_raw=lambda x: x,
                        fn_terminal=__dictify_process,
                        options=opt)


# ---------------------------------------------------------------


# TODO any sense to keep it public at all? Whats the essential diff from core_dictify which proves existence
#  of this method in public API? -- serialization safe limits or what?? The sense is that we always filter terminal
#  attrs when depth is reached or what? If we use it in YAML package only, maybe keep it there as private method?
def serial_dictify(obj: Any,
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
    Prepare objects for serialization to YAML, JSON, or other formats.

    Converts problematic types to string representations and applies
    size limits suitable for human-readable output and debugging.

    Primitive data returned as is, iterables are processed recursively with empty collections handled,
    user classes and instances are returned as data-only dict.

    Recursion of level N filters all objects from top level 0 deep to level N. At its deepest level
    attrs are shown as a single primitive value or a stats string.

    The ``serialize_object()`` is a sibling method to ``as_dict()``, both of them derive from ``core_dict()`` utility

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
        - max_depth < 0  returns obj without filtering
        - max_depth = 0 returns unfiltered obj for primitives, object info for iterables, and custom classes
        - max_depth = N iterates by N levels of recursion on iterables and objects expandable with ``as_dict()``
        - include_property = True: the method always skips properties which raise exception

    Examples:
        # For YAML serialization
        yaml_ready = serialize_object(obj, to_str=(datetime, UUID))

        # With size limits for debugging
        debug_data = serialize_object(obj, max_items=20, max_depth=2)

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

    return core_dictify(obj,
                        fn_plain=lambda x: x,
                        fn_process=__object_info,
                        hook_mode=hook_mode)
