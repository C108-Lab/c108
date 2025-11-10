"""
Comprehensive object-to-dictionary conversion toolkit.

Provides configurable, recursive transformation of arbitrary Python objects into
human-readable or JSON-serializable dictionaries. Features include dynamic depth
control, type-specific handlers, customizable attribute inclusion, collection
trimming with metadata injection, and optional class name annotation for logging,
serialization or debugging.
"""

# Standard library -----------------------------------------------------------------------------------------------------
import collections.abc as abc
import inspect
import itertools
import re
import sys

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, date, time, timedelta
from decimal import Decimal
from enum import Enum, unique
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Callable, ClassVar, Iterable, Type
from typing_extensions import Self
from uuid import UUID

# Local ----------------------------------------------------------------------------------------------------------------
from .abc import deep_sizeof, search_attrs
from .sentinels import UNSET, ifnotunset
from .formatters import fmt_any, fmt_type, fmt_value
from .utils import class_name


# Classes --------------------------------------------------------------------------------------------------------------


@dataclass
class ClassNameOptions:
    """
    Configuration for class name injection in dictified objects.

    Controls how and where class names appear in the output, useful for
    object reconstruction, debugging, and type tracking.

    Attributes:
        in_expand: Include class name in object expansion (during attribute extraction)
        in_to_dict: Inject class name into to_dict() method results
        key: Dictionary key to use for class name (default: '__class_name__')
        fully_qualified: Use 'module.ClassName' vs 'ClassName' format

    Examples:
        >>> # Minimal class names for debugging
        >>> opts = ClassNameOptions(in_expand=True, fully_qualified=False)

        >>> # Full qualification for serialization
        >>> opts = ClassNameOptions(
        ...     in_expand=True,
        ...     in_to_dict=True,
        ...     fully_qualified=True
        ... )

        >>> # Custom key to avoid collisions
        >>> opts = ClassNameOptions(in_expand=True, key='@type')
    """

    in_expand: bool = False
    in_to_dict: bool = False
    key: str = "__class_name__"
    fully_qualified: bool = False

    def merge(
        self,
        *,
        # Convenience parameter
        inject_class_name: bool = UNSET,
        # Direct attributes
        in_expand: bool = UNSET,
        in_to_dict: bool = UNSET,
        key: str = UNSET,
        fully_qualified: bool = UNSET,
    ) -> Self:
        """
        Create a new instance with merged configuration options.

        Use either the convenience parameter or the associated explicit attributes, but not both in the same call.
        Unspecified parameters retain their original values.

        Convenience Parameter:
            inject_class_name:
                - True: set both in_expand and in_to_dict to True (unless explicitly overridden)
                - False: set both in_expand and in_to_dict to False (unless explicitly overridden)
                - Cannot be used with: in_expand or in_to_dict

        Explicit Attributes:
            in_expand: Include class name in object expansion
            in_to_dict: Inject class name into to_dict() results
            key: Dictionary key for class name
            fully_qualified: Use fully qualified class names

        Returns:
            New ClassNameOptions instance with merged configuration

        Raises:
            ValueError: if both convenience parameter and explicit attributes are used in the same call.

        Examples:
            >>> # Enable class name injection everywhere
            >>> opts = ClassNameOptions()
            >>> opts = opts.merge(inject_class_name=True)

            >>> # Disable only in_to_dict while keeping in_expand
            >>> opts = opts.merge(in_to_dict=False)

            >>> # Change key and format
            >>> opts = opts.merge(key='@type', fully_qualified=True)

            >>> # Chaining
            >>> opts = (ClassNameOptions()
            ...     .merge(inject_class_name=True)
            ...     .merge(fully_qualified=True))
        """

        # Start from current values
        merged_in_expand = self.in_expand
        merged_in_to_dict = self.in_to_dict

        if inject_class_name is not UNSET:
            # Apply convenience parameter
            if in_expand is not UNSET:
                raise ValueError(
                    "cannot specify both inject_class_name and in_expand, use only one of them."
                )
            if in_to_dict is not UNSET:
                raise ValueError(
                    "cannot specify both inject_class_name and in_to_dict, use only one of them."
                )

            merged_in_expand = bool(inject_class_name)
            merged_in_to_dict = bool(inject_class_name)
        else:
            # Apply Explicit attributes
            merged_in_expand = ifnotunset(in_expand, default=merged_in_expand)
            merged_in_to_dict = ifnotunset(in_to_dict, default=merged_in_to_dict)

        return self.__class__(
            in_expand=merged_in_expand,
            in_to_dict=merged_in_to_dict,
            key=ifnotunset(key, default=self.key),
            fully_qualified=self.fully_qualified if fully_qualified is UNSET else fully_qualified,
        )


@dataclass
class Handlers:
    """
    Processing handlers for different conversion stages.

    Handlers:
        expand: Recursive convertor for expanding the topmost level of the object's tree.
                Processing chain: skip_types → raw()/terminal() → type_handlers → obj.to_dict() → expand()
        inject_meta: Inject serialization metadata into a processed object;
                     to be called within expand() and after to_dict() processing
        raw: Custom handler for raw processing mode (max_depth < 0).
             Fallback chain: raw() → obj.to_dict() → identity function
        terminal: Custom handler for terminal processing (max_depth = 0).
                  Fallback chain: terminal() → type_handlers → obj.to_dict() → identity

    """

    expand: Callable[[Any, "DictifyOptions"], Any] = None
    inject_meta: Callable[[Any, "Meta", "DictifyOptions"], Any] = None
    raw: Callable[[Any, "DictifyOptions"], Any] = None
    terminal: Callable[[Any, "DictifyOptions"], Any] = None

    def __post_init__(self):
        self.expand = self.expand or expand
        self.inject_meta = self.inject_meta or inject_meta


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

    def to_dict(
        self,
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
            TypeError: If an instance class is not a dataclass.
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
        """Validate constraints."""
        if all(val is None for val in (self.len, self.deep, self.shallow)):
            raise ValueError("SizeMeta requires at least one non-None value")

        if self.deep is not None and self.shallow is not None and self.deep < self.shallow:
            raise ValueError("SizeMeta.deep >= SizeMeta.shallow expected")

    @classmethod
    def from_object(
        cls,
        obj: Any,
        *,
        include_len: bool = False,
        include_deep: bool = False,
        include_shallow: bool = False,
    ) -> Self | None:
        """
        Create SizeMeta instance from an object with specified size measurements.

        Args:
            obj: The object to measure.
            include_len: If True, include the object's length (if it has __len__).
            include_deep: If True, include deep size in bytes using deep_sizeof().
            include_shallow: If True, include shallow size in bytes using sys.getsizeof().

        Returns:
            SizeMeta instance with requested measurements, or None if no measurements
            were requested or if all requested measurements failed to produce values.

        Note:
            At least one include_* parameter must be True to get a non-None result.
            Length is only included for objects that support __len__.
        """
        if not any([include_len, include_deep, include_shallow]):
            return None

        len_ = _length_or_none(obj) if include_len else None

        try:
            deep_ = deep_sizeof(obj) if include_deep else None
        except Exception:
            deep_ = None

        try:
            shallow_ = sys.getsizeof(obj) if include_shallow else None
        except Exception:
            shallow_ = None

        if all(val is None for val in (len_, deep_, shallow_)):
            return None

        return cls(len=len_, deep=deep_, shallow=shallow_)


@dataclass(frozen=True)
class TrimMeta(MetaMixin):
    """Metadata about collection trimming operations.

    Supports both sized collections and iterables of unknown length (generators, etc).

    Attributes:
        len: Total number of elements in original iterable, or None if unknown.
        shown: Number of elements kept/shown after trimming.
        is_trimmed: (property) Whether trimming occurred. None if len is unknown.
        trimmed: (property) Number of elements removed. None if len is unknown.
    """

    len: int | None = None
    shown: int | None = None

    def __post_init__(self) -> None:
        """Validate business logic constraints."""
        if self.shown is None:
            raise ValueError("TrimMeta requires 'shown' attribute.")

        # Only business logic validation
        if self.len is not None and self.shown is not None and self.shown > self.len:
            raise ValueError("TrimMeta.shown <= TrimMeta.len expected")

    @classmethod
    def from_objects(cls, obj: Any, processed_object: Any) -> Self | None:
        """
        Create TrimMeta instance by comparing original and processed objects.

        This returns None in cases where a size comparison would be misleading:
        - If processed_object is mapping/items-iterable but obj is not (non-mapping -> mapping).
        - If processed_object is iterable but obj is not (non-iterable -> iterable).
        - If processed_object has unknown length.

        Otherwise, len reflects the size of obj (or None if unknown), and shown reflects
        the size of processed_object.

        Args:
            obj: The original object.
            processed_object: The processed/trimmed object.

        Returns:
            TrimMeta instance with length comparison data, or None
            if comparison is not meaningfull or processed_object has unknown length.

        Examples:
            Normal comparison with both sizes known:
                >>> TrimMeta.from_objects([1, 2, 3, 4, 5], [1, 2, 3])
                TrimMeta(len=5, shown=3)

            Original size unknown, processed size known:
                >>> orig = (i for i in range(10))  # generator, unknown length
                >>> TrimMeta.from_objects(orig, [0, 1, 2])
                TrimMeta(len=None, shown=3)

            Processed size unknown (generator) -> returns None:
                >>> TrimMeta.from_objects([1, 2, 3, 4], (i for i in range(2)))


            Converted non-iterable to iterable -> returns None:
                >>> TrimMeta.from_objects(42, [42])

            Converted non-mapping to mapping/items-iterable -> returns None:
                >>> TrimMeta.from_objects({1, 2, 3}, {"a": 1, "b": 2})
        """

        if not _is_items_iterable(obj) and _is_items_iterable(processed_object):
            # If we've converted non-mapping to mapping, return None
            return None

        if not _is_iterable(obj) and _is_iterable(processed_object):
            # If we've converted non-iterable to iterable, return None
            return None

        original_len = _length_or_none(obj)
        processed_len = _length_or_none(processed_object)

        if processed_len is None:
            # Can't create metadata without knowing what we're showing
            return None

        return cls(len=original_len, shown=processed_len)

    @classmethod
    def from_trimmed(cls, total_len: int, trimmed_len: int) -> Self:
        """Create TrimMeta from total length and trimmed items count.

        Args:
            total_len: Total number of elements in the original collection.
            trimmed_len: Number of elements that were trimmed.

        Returns:
            TrimMeta instance with computed shown value.
        """
        shown = max(total_len - trimmed_len, 0)
        return cls(len=total_len, shown=shown)

    @property
    def is_trimmed(self) -> bool | None:
        """Whether the collection was trimmed.

        Returns None if source length unknown (unsized iterable).
        """
        if self.len is None:
            return None
        return self.shown < self.len

    @property
    def trimmed(self) -> int | None:
        """Number of elements removed due to trimming.

        Returns None if source length unknown (unsized iterable).
        """
        if self.len is None:
            return None
        return self.len - self.shown


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

    @classmethod
    def from_object(cls, obj: Any) -> Self:
        """
        Create TypeMeta instance from an object.

        Args:
            obj: The source object.

        Returns:
            TypeMeta instance with the runtime type of obj.

        Note:
            Captures actual type including NoneType for None value.
        """
        return cls(from_type=type(obj))

    @classmethod
    def from_objects(cls, obj: Any, processed_object: Any) -> Self:
        """
        Create TypeMeta instance by comparing original and processed objects.

        Args:
            obj: The original object.
            processed_object: The processed/converted object.

        Returns:
            TypeMeta instance with the runtime types of both objects.

        Note:
            Captures actual types including NoneType for None values.
        """
        from_type = type(obj)
        to_type = type(processed_object)

        return cls(from_type=from_type, to_type=to_type)

    @property
    def is_converted(self) -> bool:
        """Check if type conversion occurred."""
        if self.from_type is None or self.to_type is None:
            # Can't determine conversion without both types
            return False

        return self.from_type != self.to_type

    def to_dict(
        self,
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

        if not self.is_converted and not include_none_attrs:
            # When is not converted, to_type is redundant - but only remove if not including None attrs
            dict_.pop("to_type", None)

        return dict_


@dataclass(frozen=True)
class Meta:
    """
    Comprehensive metadata for dictify conversion operations.

    Contains information about trimming, sizing, and type conversion that
    occurred during object-to-dictionary conversion. Used internally by
    dictify_core() to inject metadata into processed collections and objects.

    Attributes:
        size: Size metadata (shallow bytes, deep bytes, length)
        trim: Collection trimming stats
        type: Type conversion metadata
    """

    VERSION: ClassVar[int] = 1  # Metadata schema version

    size: SizeMeta | None = None
    trim: TrimMeta | None = None
    type: TypeMeta | None = None

    @classmethod
    def from_object(cls, obj: Any, *, opts: "DictifyOptions" = None) -> Self | None:
        """
        Create metadata object for dictify processing operations.

        Analyzes the source object to create metadata with size information
        and type. The Metadata creation is controlled by the flags in opts.meta configuration.

        Args:
            obj: The original object before any processing or trimming operations.
            opts: DictifyOptions instance containing metadata generation flags.

        Returns:
            Meta object containing requested metadata, or None if no metadata requested.
        """
        opts = opts or DictifyOptions()
        if not isinstance(opts, DictifyOptions):
            raise TypeError(f"opts must be a DictifyOptions instance, got {fmt_type(opts)}")

        size_meta = SizeMeta.from_object(
            obj,
            include_len=opts.meta.len,
            include_deep=opts.meta.deep_size,
            include_shallow=opts.meta.size,
        )
        type_meta = TypeMeta.from_object(obj) if opts.meta.type else None

        if any([size_meta, type_meta]):
            return cls(size=size_meta, type=type_meta)

        return None

    @classmethod
    def from_objects(
        cls, obj: Any, processed_obj: Any, *, opts: "DictifyOptions" = None
    ) -> Self | None:
        """
        Create metadata object for dictify processing operations.

        Analyzes the original and processed objects to generate comprehensive metadata
        including size information, trimming statistics, and type conversion details.
        Metadata creation is controlled by the flags in opts.meta configuration.

        Args:
            obj: The original object before any processing or trimming operations.
            processed_obj: The object after trimming, type conversion, or other processing.
            opts: DictifyOptions instance containing metadata generation flags and limits.

        Returns:
            Meta object containing requested metadata, or None if no metadata
            was requested or could be generated.
        """
        opts = opts or DictifyOptions()
        if not isinstance(opts, DictifyOptions):
            raise TypeError(f"opts must be a DictifyOptions instance, got {fmt_type(opts)}")
        size_meta = SizeMeta.from_object(
            obj,
            include_len=opts.meta.len,
            include_deep=opts.meta.deep_size,
            include_shallow=opts.meta.size,
        )
        trim_meta = TrimMeta.from_objects(obj, processed_obj) if opts.meta.trim else None
        type_meta = TypeMeta.from_objects(obj, processed_obj) if opts.meta.type else None

        if any([size_meta, trim_meta, type_meta]):
            return cls(size=size_meta, trim=trim_meta, type=type_meta)

        return None

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

    def to_dict(
        self,
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
class MetaOptions:
    """
    Metadata generation and injection options for dictify operations.

    Controls what metadata gets injected into converted objects, including size information,
    trimming statistics, and type conversion details. Metadata is injected either as a
    dictionary key (for mappings) or appended as the final element (for sequences/sets).

    Attributes:
        in_expand: Include metadata in object expansion (during attribute extraction)
        in_to_dict: Inject metadata into to_dict() method results
        key: Dictionary key used for metadata injection in mappings (default: "__dictify__")
        len: Include collection length in size metadata
        size: Include shallow object size in bytes (via sys.getsizeof)
        deep_size: Include deep object size calculation (expensive operation)
        trim: Inject trimming statistics when collections exceed max_items limit
        type: Include type conversion metadata when object types change during processing

    Examples:
        >>> # Enable all metadata
        >>> meta = MetaOptions(len=True, size=True, deep_size=True, type=True)

        >>> # Only trimming metadata (default)
        >>> meta = MetaOptions()  # trim=True by default

        >>> # Custom metadata key
        >>> meta = MetaOptions(key="__meta", trim=True, type=True)
    """

    # Injection into object processor's output
    in_expand: bool = False
    in_to_dict: bool = False

    # Injection Key
    key: str = "__dictify__"

    # Size-related metadata
    len: bool = False
    size: bool = False  # Shallow size
    deep_size: bool = False  # Deep size (expensive)

    # Operation metadata
    trim: bool = False  # Trimming statistics
    type: bool = False  # Type conversion info

    @property
    def any_enabled(self) -> bool:
        """Check if any metadata injection is enabled."""
        return any([self.sizes_enabled, self.trim, self.type])

    @property
    def sizes_enabled(self) -> bool:
        """Check if any size-related metadata injection is enabled."""
        return any([self.len, self.size, self.deep_size])

    def merge(
        self,
        *,
        in_expand: bool = UNSET,
        in_to_dict: bool = UNSET,
        key: str = UNSET,
        len: bool = UNSET,
        size: bool = UNSET,
        deep_size: bool = UNSET,
        trim: bool = UNSET,
        type: bool = UNSET,
        inject_trim_meta: bool = UNSET,
        inject_type_meta: bool = UNSET,
    ) -> Self:
        """Create a new instance with merged configuration options.

        Use either the convenience parameter or the associated explicit attributes, but not both in the same call.
        Unspecified parameters retain their original values.

        Convenience parameters:
          - inject_trim_meta:
              - True  -> sets trim=True and forces in_expand=True and in_to_dict=True
              - False -> sets trim=False
              - Cannot be used together with: trim, in_expand, or in_to_dict
          - inject_type_meta:
              - True  -> sets type=True and forces in_expand=True and in_to_dict=True
              - False -> sets type=False
              - Cannot be used together with: type, in_expand, or in_to_dict

        Args:
            in_expand: Include metadata in object expansion (attribute extraction).
            in_to_dict: Inject metadata into to_dict() method results.
            key: Dictionary key for metadata injection in mappings.
            len: Include collection length in size metadata.
            size: Include shallow object size in bytes.
            deep_size: Include deep object size calculation.
            trim: Inject trimming statistics when collections exceed limits.
            type: Include type conversion metadata when object types change.
            inject_trim_meta: Convenience flag to toggle trimming metadata and ensure injection points.
                             Mutually exclusive with trim, in_expand, and in_to_dict.
            inject_type_meta: Convenience flag to toggle type metadata and ensure injection points.
                             Mutually exclusive with type, in_expand, and in_to_dict.

        Returns:
            MetaOptions: New MetaOptions instance with merged configuration.

        Raises:
            ValueError: If inject_trim_meta is used with trim, in_expand, or in_to_dict.
            ValueError: If inject_type_meta is used with type, in_expand, or in_to_dict.
            TypeError: If inject_trim_meta or inject_type_meta is not a bool.
        """
        # Check mutual exclusivity for inject_trim_meta
        if inject_trim_meta is not UNSET:
            if trim is not UNSET:
                raise ValueError(
                    "inject_trim_meta cannot be used together with trim. "
                    "Use either the convenience flag or the explicit parameter, not both."
                )
            if in_expand is not UNSET:
                raise ValueError(
                    "inject_trim_meta cannot be used together with in_expand. "
                    "Use either the convenience flag or the explicit parameter, not both."
                )
            if in_to_dict is not UNSET:
                raise ValueError(
                    "inject_trim_meta cannot be used together with in_to_dict. "
                    "Use either the convenience flag or the explicit parameter, not both."
                )

        # Check mutual exclusivity for inject_type_meta
        if inject_type_meta is not UNSET:
            if type is not UNSET:
                raise ValueError(
                    "inject_type_meta cannot be used together with type. "
                    "Use either the convenience flag or the explicit parameter, not both."
                )
            if in_expand is not UNSET:
                raise ValueError(
                    "inject_type_meta cannot be used together with in_expand. "
                    "Use either the convenience flag or the explicit parameter, not both."
                )
            if in_to_dict is not UNSET:
                raise ValueError(
                    "inject_type_meta cannot be used together with in_to_dict. "
                    "Use either the convenience flag or the explicit parameter, not both."
                )

        # Start from current values
        new_in_expand = self.in_expand
        new_in_to_dict = self.in_to_dict
        new_trim = self.trim
        new_type = self.type

        # Apply convenience flags
        if inject_trim_meta is not UNSET:
            inject_trim_meta = bool(inject_trim_meta)
            if inject_trim_meta:
                new_trim = True
                new_in_expand = True
                new_in_to_dict = True
            else:
                new_trim = False

        if inject_type_meta is not UNSET:
            inject_type_meta = bool(inject_type_meta)
            if inject_type_meta:
                new_type = True
                new_in_expand = True
                new_in_to_dict = True
            else:
                new_type = False

        # Apply explicit args (only if convenience flags weren't used - already validated)
        new_trim = ifnotunset(trim, default=new_trim)
        new_type = ifnotunset(type, default=new_type)
        new_in_expand = ifnotunset(in_expand, default=new_in_expand)
        new_in_to_dict = ifnotunset(in_to_dict, default=new_in_to_dict)

        # Merge remaining, non-convenience fields as usual
        new_key = _merge_new_default(key, self.key)
        new_len = _merge_new_default(len, self.len)
        new_size = _merge_new_default(size, self.size)
        new_deep_size = _merge_new_default(deep_size, self.deep_size)

        return self.__class__(
            in_expand=new_in_expand,
            in_to_dict=new_in_to_dict,
            key=new_key,
            len=new_len,
            size=new_size,
            deep_size=new_deep_size,
            trim=new_trim,
            type=new_type,
        )


@dataclass
class DictifyOptions:
    """
    Advanced configuration options for object-to-dictionary conversion with extensive customization.

    Provides comprehensive control over object serialization including recursion depth management,
    attribute filtering, size constraints, custom type handling, and collection processing behavior.
    Supports both debugging and production serialization scenarios with flexible hook systems.

    Core Configuration:
        max_depth: Maximum recursion depth for nested objects (default: 3)
                  - max_depth < 0: Raw mode, uses handlers.raw handler
                  - max_depth = 0: Terminal mode, uses handlers.terminal handler
                  - max_depth > 0: Normal recursive processing

    Attribute Control:
        - include_none_attrs: Include object attributes with None values
        - include_none_items: Include mapping items with None values
        - include_private: Include private attributes (starting with _)
        - include_properties: Include instance properties with assigned values

    Processing Handlers:
        handlers: Handlers provide processors for edge cases, mutability, and metadata injection
            - handlers.inject_meta: Handler for injecting metadata into objects
            - handlers.raw: Handler for raw mode (max_depth < 0)
                            Fallback chain: handlers.raw() → obj.to_dict() → identity
            - handlers.terminal: Handler for terminal mode (max_depth = 0)
                                 Fallback chain: handlers.terminal() → type_handlers → obj.to_dict() → identity
            - handlers.expand: Handler for recursive mode expansion (max_depth >= 1)
                               from object to a mutable collection

    🚀 Size and Performance Limits:
        - max_items: Maximum items in collections before trimming (default: 100).
                     None = no limit (process entire collection).
        - max_str_len: String truncation limit (default: 200), None = no truncation
        - max_bytes: Bytes object truncation limit (default: 512), None = no truncation

    Mapping keys and Iterable values handling:
        sort_iterables: Enable items sorting for iterables
        sort_keys: Enable key sorting for mappings

    Class Name Injection:
        class_name: ClassNameOptions controlling class name appearance:
            - class_name.fully_qualified: Use 'module.Class' vs 'Class' format
            - class_name.in_expand: Add class name during object expansion
            - class_name.in_to_dict: Add class name to to_dict() results
            - class_name.key: Dictionary key for class name (default: '__class_name__')

    Meta Data Injection:
        meta: MetaOptions controlling what metadata gets injected:
              - meta.trim: Trimming statistics for oversized collections
              - meta.type: Type conversion information
              - meta.len/size/deep_size: Object size metadata
              - meta.key: Dictionary key for metadata in mappings

    Advanced Processing:
        - hook_mode: Object conversion strategy:
                  - "dict": Try to_dict() with fallback to recursive expansion
                  - "dict_strict": Require to_dict() method (raises if missing)
                  - "none": Skip object hooks, use expansion only
        - skip_types: Types bypassing all filtering (default: int, float, bool, complex, None)
        - type_handlers: Custom type processing functions with inheritance support

    Type Handler System:
        - Used for non-recursive processing of primitives; can be expanded with user data types
        - Supports exact type matching and inheritance-based resolution via MRO
        - Default handlers for: str, bytes, bytearray, memoryview
        - Precedence: type_handlers → obj.to_dict() → recursive expansion
        - Handlers receive (obj, options) and return processed result

    Collection Processing Features:
        - Comprehensive support for Sequences, Mappings, Sets, and MappingViews
        - Automatic trimming with metadata injection for oversized collections
        - Dict-like object detection and processing via items() method

    Class Methods:
        - debug(): Comprehensive debugging with shallow depth and all attributes
        - logging(): Controlled verbosity with size limits and metadata injection
        - serial(): Clean JSON-ready output with class names for reconstruction

    Instance Methods:
        - add_type_handler(typ, handler): Register custom type processor (chainable)
        - get_type_handler(obj): Retrieve handler via inheritance resolution
        - remove_type_handler(typ): Unregister type processor (chainable)

    Properties:
        - type_handlers: Dict[Type, Callable] - getter/setter with validation

    Examples:
        >>> # Basic usage with defaults
        >>> class Obj:
        ...     a: int = 1
        ...     b: str = "2"
        >>> o = Obj()
        >>> dictify(o)
        {'a': 1, 'b': '2'}

        >>> # Debugging
        >>> dictify(o, opts=DictifyOptions.debug())
        {'a': 1, 'b': '2', '__dictify__': {'type': {'from_type': <class 'c108.dictify.Obj'>, 'to_type': <class 'dict'>}, 'version': 1}}

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
        ...     max_str_len=100,     # Truncate long strings
        ...     max_bytes=512          # Limit byte arrays
        ... )

        >>> # Deep inspection with custom terminal handler
        >>> def custom_terminal(obj, opts):
        ...     return f"<{type(obj).__name__} at depth limit>"
        >>>
        >>> deep_opts = DictifyOptions(
        ...     max_depth=10,
        ...     handlers=Handlers(terminal=custom_terminal)  # Updated syntax
        ... )

    Processing Order:
        1. Skip types (int, float, bool, complex, None) → return as-is
        2. Edge cases (max_depth < 0 or == 0) → use handlers.raw/handlers.terminal chains
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

    include_none_attrs: bool = False
    include_none_items: bool = False
    include_private: bool = False
    include_properties: bool = False

    # Processing handlers
    handlers: Handlers = field(default_factory=Handlers)

    # Size limits
    max_items: int | None = 100
    max_str_len: int | None = 200
    max_bytes: int | None = 512

    # Mapping Keys handling
    sort_keys: bool = False
    sort_iterables: bool = False

    # Class Name Injection
    class_name: ClassNameOptions = field(default_factory=ClassNameOptions)

    # Meta Data Injection
    meta: MetaOptions = field(default_factory=MetaOptions)

    # Advanced
    hook_mode: str = HookMode.DICT
    skip_types: tuple[type, ...] = (int, float, bool, complex, type(None))

    type_handlers: Dict[Type, Callable[[Any, "DictifyOptions"], Any]] = field(
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
            BaseException: _handle_exception,
            Decimal: _handle_decimal,
            Enum: _handle_enum,
            Fraction: _handle_fraction,
            Path: _handle_path,
            UUID: _handle_uuid,
            bytearray: _handle_bytearray,
            bytes: _handle_bytes,
            date: _handle_date,
            datetime: _handle_datetime,
            memoryview: _handle_memoryview,
            range: _handle_range,
            re.Pattern: _handle_regex_pattern,
            str: _handle_str,
            time: _handle_time,
            timedelta: _handle_timedelta,
        }

    # Class Methods ------------------------------------

    @classmethod
    def debug(cls) -> Self:
        """
        Create a DictifyOptions instance configured for debugging.

        Shallow inspection showing everything including internals.
        No size limits to avoid data loss during debugging.

        Returns:
            DictifyOptions: Configuration optimized for debugging with shallow depth,
                           all attributes included, and minimal filtering.
        """
        return cls(
            max_depth=2,
            include_none_attrs=True,
            include_none_items=True,
            include_private=True,
            include_properties=True,
            max_items=200,
            max_str_len=512,
            max_bytes=1024,
            sort_keys=True,
            class_name=ClassNameOptions(in_expand=True, in_to_dict=True, fully_qualified=True),
            meta=MetaOptions(
                in_expand=True,
                in_to_dict=True,
                trim=True,
                type=True,
                len=True,
                size=False,
                deep_size=False,
            ),
        )

    @classmethod
    def logging(cls) -> Self:
        """
        Create a DictifyOptions instance configured for logging.

        Controlled verbosity with size limits and metadata injection.
        Balanced between information and performance.

        Returns:
            DictifyOptions: Configuration for logging with controlled depth,
                           size limits, and helpful metadata injection.
        """
        return cls(
            max_depth=4,
            include_none_attrs=False,
            include_none_items=False,
            include_private=False,
            include_properties=True,
            max_items=50,
            max_str_len=128,
            max_bytes=512,
            sort_keys=True,
            class_name=ClassNameOptions(in_expand=True, in_to_dict=True, fully_qualified=True),
            meta=MetaOptions(
                in_expand=True,
                in_to_dict=True,
                trim=True,
                type=True,
                len=True,
                size=True,
                deep_size=False,
            ),
        )

    @classmethod
    def serial(cls) -> Self:
        """
        Create a DictifyOptions instance configured for serialization.

        Clean output optimized for JSON serialization and reconstruction.
        No None values, includes class names for type reconstruction.

        Returns:
            DictifyOptions: Configuration for serialization with class names
                           included, clean output, and JSON-friendly formatting.
        """
        return cls(
            max_depth=6,
            include_none_attrs=False,
            include_none_items=False,
            include_private=False,
            include_properties=False,
            max_items=1000,
            max_str_len=200,
            max_bytes=2048,
            sort_keys=True,
            hook_mode=HookMode.DICT_STRICT,
            class_name=ClassNameOptions(in_expand=True, in_to_dict=True, fully_qualified=True),
            meta=MetaOptions(
                in_expand=False,
                in_to_dict=False,
                trim=False,
                type=False,
                len=False,
                size=False,
                deep_size=False,
            ),
        )

    # Methods and Properties ---------------------

    def add_type_handler(
        self,
        typ: type,
        handler: Callable[[Any, "DictifyOptions"], Any],
    ) -> Self:
        """
        Register or override a handler for a specific type.

        Args:
            typ: The concrete type to process.
            handler: A callable receiving (obj, options) and returning processed value.
        """
        self.type_handlers[typ] = handler
        return self

    def get_type_handler(self, obj: Any) -> abc.Callable[[Any, Self], Any] | None:
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

    def merge(
        self,
        *,
        # Common explicit attributes
        max_depth: int = UNSET,
        max_items: int | None = UNSET,
        max_str_len: int | None = UNSET,
        max_bytes: int | None = UNSET,
        include_none_attrs: bool = UNSET,
        include_none_items: bool = UNSET,
        include_private: bool = UNSET,
        include_properties: bool = UNSET,
        sort_keys: bool = UNSET,
        sort_iterables: bool = UNSET,
        # Convenience parameters (affect multiple attributes)
        inject_class_name: bool = UNSET,
        inject_trim_meta: bool = UNSET,
        inject_type_meta: bool = UNSET,
        # Advanced nested objects
        class_name: ClassNameOptions = UNSET,
        meta: MetaOptions = UNSET,
        handlers: Handlers = UNSET,
    ) -> Self:
        """
        Create a new instance with merged configuration options.

        Supports both high-level convenience parameters that affect multiple attributes
        and explicit attribute assignment. Unspecified parameters retain their current values.
        All parameters are optional, skip to keep current values

        Convenience Parameters:
            inject_class_name: When True, sets both class_name.in_expand and
                               class_name.in_to_dict to True. When False, sets both to False.
            inject_trim_meta: When True, enables meta.trim and injection. When False, disables meta.trim.
            inject_type_meta: When True, enables meta.type and injection. When False, disables meta.type.

        Precedence:
            If both convenience flags and explicit args are provided,
            explicit args override the convenience parameters.

        Common Direct Attributes:
            max_depth: Maximum recursion depth for nested objects
            max_items: Maximum items in collections before trimming or None for untrimmed
            max_str_len: String truncation limit or None for unlimited
            max_bytes: Bytes object truncation limit or None for unlimited
            include_none_attrs: Include object attributes with None values
            include_none_items: Include dictionary items with None values
            include_private: Include private attributes (starting with _)
            include_properties: Include instance properties with assigned values
            sort_keys: Enable key sorting for mappings
            sort_iterables: Enable items sorting for iterables

        Advanced Nested Objects:
            class_name: Complete ClassNameOptions instance for full control
            meta: Complete MetaOptions instance for full control
            handlers: Complete Handlers instance for custom processing

        Returns:
            New DictifyOptions instance with merged configuration
        """
        # Handle convenience parameters by creating modified nested objects
        merged_class_name = self.class_name
        merged_meta = self.meta

        # If explicit nested objects are provided, they should override convenience flags entirely.
        if class_name is not UNSET:
            merged_class_name = class_name
        else:
            if inject_class_name is not UNSET:
                merged_class_name = merged_class_name.merge(inject_class_name=inject_class_name)

        if meta is not UNSET:
            merged_meta = meta
        else:
            if inject_trim_meta is not UNSET:
                merged_meta = merged_meta.merge(trim=inject_trim_meta)
            if inject_type_meta is not UNSET:
                merged_meta = merged_meta.merge(type=inject_type_meta)

        # Build new instance with merged values
        return self.__class__(
            max_depth=_merge_new_default(max_depth, self.max_depth),
            include_none_attrs=_merge_new_default(include_none_attrs, self.include_none_attrs),
            include_none_items=_merge_new_default(include_none_items, self.include_none_items),
            include_private=_merge_new_default(include_private, self.include_private),
            include_properties=_merge_new_default(include_properties, self.include_properties),
            handlers=_merge_new_default(handlers, self.handlers),
            max_items=_merge_new_default(max_items, self.max_items),
            max_str_len=_merge_new_default(max_str_len, self.max_str_len),
            max_bytes=_merge_new_default(max_bytes, self.max_bytes),
            sort_keys=_merge_new_default(sort_keys, self.sort_keys),
            sort_iterables=_merge_new_default(sort_iterables, self.sort_iterables),
            class_name=merged_class_name,
            meta=merged_meta,
            hook_mode=self.hook_mode,
            skip_types=self.skip_types,
            type_handlers=self.type_handlers,
        )

    def remove_type_handler(self, typ: type) -> Self:
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


# Methods --------------------------------------------------------------------------------------------------------------


def dictify_core(
    obj: Any,
    *,
    opts: DictifyOptions | None = None,
) -> Any:
    """
    Convert any Python object to a human-readable dictionary representation with full control.

    This is the main conversion engine of the dictify module, providing tunable object
    representation for CLI printing, logging, debugging, and serialization. Handles arbitrary
    Python objects by preserving primitives, intelligently processing collections, expanding
    object attributes, and offering extensive customization through DictifyOptions.

    Use DictifyOptions() for basic conversion or its specialized preset factories for
    logging/debugging/serialization scenarios.

    Processing Pipeline:
        1. Skip Types: Primitives returned unchanged
        2. Edge Case Handling:
           - max_depth < 0: Raw mode via raw() chain
           - max_depth = 0: Terminal mode via terminal() chain
        3. Type Handler Resolution: Custom processors via inheritance hierarchy
        4. Object Hook Processing: to_dict() method calls based on hook_mode
        5. Collection Processing: Sequences, mappings, sets, views, other iterables
        6. Object Expansion: Attribute extraction with filtering rules

    Args:
        obj: Any Python object to convert to dictionary representation
        opts: DictifyOptions instance controlling all conversion behaviors.
                Default DictifyOptions() used if None.

    Returns:
        Human-readable dictionary representation of the object, or processed result
        from custom handlers.

    Handler Precedence (Normal Processing, max_depth > 0):
        1. Skip types bypassed (default: int, float, bool, complex, None)
        2. Type handlers (exact type or inheritance-based via MRO)
        3. Object to_dict() method (controlled by hook_mode)
        4. Collection/mapping/sequence recursive processing
        5. Object attribute expansion with filtering

    Edge Case Processing:
        - Raw Mode (max_depth < 0): raw() → obj.to_dict() → obj identity
        - Terminal Mode (max_depth = 0): terminal() → type_handlers → obj.to_dict() → obj dentity

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
        - Affects processing with expand() and to_dict()
        - Sequences/Sets/Iterables: Meta appended as the final element
        - Mappings, ItemViews, Mapping-like objects: Meta added under options.meta.key
        - Trimming meta for oversized collections (len > max_items) when options.meta.trim enabled
        - No Metadata injection in default raw(), terminal(), and type_handlers

    Object Expansion Rules:
        - Private attributes included only if include_private=True
        - Properties included only if include_properties=True and accessible
        - None values filtered based on include_none_attrs setting
        - Class name injection controlled by class_name options
        - Meta injection controlled by options.meta flags
        - Attribute access exceptions automatically handled and skipped

    Examples:
        >>> # Basic conversion with defaults
        >>> class Person:
        ...     def __init__(self, name, age):
        ...         self.name = name
        ...         self.age = age
        >>> obj = Person("Alice", 7)
        >>> result = dictify_core(obj)

        >>> # Custom depth and terminal handling
        >>> def terminal_handler(obj, opts):
        ...     return f"<{type(obj).__name__}:truncated>"
        >>>
        >>> opts = DictifyOptions(max_depth=5, handlers=Handlers(terminal=terminal_handler))
        >>> result = dictify_core(obj, opts=opts)

        >>> # Raw mode processing
        >>> raw_opt = DictifyOptions(max_depth=-1)
        >>> raw_result = dictify_core(obj, opts=raw_opt)  # Minimal processing

        >>> # Collection size management
        >>> size_opt = DictifyOptions(max_items=100, max_str_len=50)
        >>> trimmed_result = dictify_core(obj, opts=size_opt)

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
        >>> result = dictify_core(PostgresConnection(), opts=opts)

        # Strict object hook mode
        >>> strict_opt = DictifyOptions(hook_mode="dict_strict")
        >>> dictify_core(obj, opts=strict_opt)
        Traceback (most recent call last):
        ...
        TypeError: Class <Person> must implement to_dict() when hook_mode='HookMode.DICT_STRICT'

    Special Behaviors:
        - max_depth parameter controls recursion: N levels deep for collections,
          with object attributes processed at depth N-1
        - Skip types (int, float, bool, complex, None, range) bypass all processing
        - Default type handlers process str, bytes, bytearray, memoryview with size limits
        - Class name inclusion affects main processing with expand() and to_dict() only, not edge case handlers
        - Key sorting (if enabled) applies to main processing and to_dict() injection
        - Sets are converted to lists
        - Exception-raising properties automatically skipped during object expansion
        - MRO-based type handler resolution supports inheritance hierarchies

    🚀 Performance Notes:
        - Collection trimming prevents memory issues with large datasets
        - Type handler caching optimizes repeated conversions
        - Shallow copying for depth management minimizes overhead
        - Early returns for skip types and edge cases improve efficiency
    """

    opts = opts or DictifyOptions()

    # dictify_core() body Start ---------------------------------------------------------------------------

    # Return skip_type objects as is -----------------
    if isinstance(obj, tuple(opts.skip_types)):
        return obj

    # Edge Cases processing --------------------------
    if opts.max_depth < 0:
        return _handlers_raw_chain(obj, opts=opts)
    if opts.max_depth == 0:
        return _handlers_terminal_chain(obj, opts=opts)

    # Type handling and obj.to_dict() processors -----
    if type_handler := opts.get_type_handler(obj):
        return type_handler(obj, opts)

    if dict_ := _get_from_to_dict(obj, opts=opts):
        return dict_

    # Expand the topmost level of obj tree,
    # call recursive expansion in deep
    return opts.handlers.expand(obj, opts)

    # dictify_core() body End ---------------------------------------------------------------------------


def expand(obj: Any, opts: DictifyOptions | None = None) -> list | dict:
    """
    Recursively convert an object to a list or dict representation.

    This is the main expansion function called after all type-specific processors
    (skip_types, raw()/terminal(), type_handlers, obj.to_dict()) have been applied.

    Args:
        obj: The object to convert. Can be any type.
        opts: Dictify options controlling behavior (depth, metadata, class names, etc.)

    Returns:
        - list: if obj is a non-mapping iterable (Sequence, set, etc.)
        - dict: if obj is a mapping or a regular object with attributes

    Behavior:
        - Recursively processes nested values with max_depth - 1
        - Optionally trims collections based on opts.max_items/max_str_len
        - Filters None values based on opts.include_none_items/include_none_attrs
        - Injects __class_name__ and metadata if configured in opts

    Raises:
        ValueError: If max_depth < 1

    Processing chain:
        skip_types → raw()/terminal() → type_handlers → obj.to_dict() → expand()
    """

    if opts.max_depth < 1:
        raise ValueError(
            f"max_depth >= 1 expected but {fmt_value(opts.max_depth)} found. "
            f"Edge cases and max_depth = 0 are processed by wrapper of this method"
        )

    # Handle mapping-like objects exposing items(), even if not iterable themselves.
    if _is_items_iterable(obj) and not isinstance(obj, abc.Mapping):
        items_iter = obj.items()
        # Collect up to max_items from unknown-length items iterable
        if opts.max_items is not None:
            items = list(itertools.islice(items_iter, opts.max_items))
        else:
            items = list(items_iter)
        # Prefer dict when possible; fall back to list if dict would drop entries
        try:
            result_dict = dict(items)
            if len(result_dict) == len(items):
                obj_ = result_dict
            else:
                obj_ = items
        except Exception:
            obj_ = items

        if isinstance(obj_, dict):
            obj_ = {
                k: _dictify_core(v, opts.max_depth - 1, opts)
                for k, v in obj_.items()
                if (v is not None) or opts.include_none_items
            }
        else:
            obj_ = [
                _dictify_core(item, opts.max_depth - 1, opts)
                for item in obj_
                if (item is not None) or opts.include_none_items
            ]

    elif _is_iterable(obj):
        # Keep original iterable for metadata (length/trim/type) decisions
        original_iterable = obj
        # Expand with sorting/trim to mutable shell
        obj_ = _iterable_to_mutable(original_iterable, opts=opts)

        if isinstance(obj_, list):
            # Compute trim meta BEFORE slicing result length with include_none_items filtering
            trim_meta = None
            if opts.meta.trim:
                # Determine original and shown lengths for trimming meta
                orig_len = _length_or_none(original_iterable)
                shown_len = len(obj_)
                try:
                    # If original is sized, use its true length; otherwise keep None
                    orig_len = len(original_iterable) if _is_sized(original_iterable) else orig_len
                except Exception:
                    pass
                try:
                    trim_meta = TrimMeta(len=orig_len, shown=shown_len)
                except Exception:
                    trim_meta = None

            # Recursively process the list items
            processed_list = [
                _dictify_core(item, opts.max_depth - 1, opts)
                for item in obj_
                if (item is not None) or opts.include_none_items
            ]
            obj_ = processed_list

        elif isinstance(obj_, dict):
            obj_ = {
                k: _dictify_core(v, opts.max_depth - 1, opts)
                for k, v in obj_.items()
                if (v is not None) or opts.include_none_items
            }
            trim_meta = None  # not used for dict branch here; Meta.from_objects will compute it
        else:
            raise TypeError(f"An expanded iterable must be a dict or list, but got {fmt_type(obj)}")
    else:
        obj_ = _shallow_to_mutable(obj, opts=opts)
        obj_ = {
            k: _dictify_core(v, opts.max_depth - 1, opts)
            for k, v in obj_.items()
            if (v is not None) or opts.include_none_attrs
        }
        trim_meta = None  # not applicable

    if isinstance(obj_, dict):
        if opts.class_name.in_expand:
            obj_[opts.class_name.key] = _class_name(obj, opts)
        if opts.meta.in_expand:
            meta = Meta.from_objects(obj, obj_, opts=opts)
            obj_ = opts.handlers.inject_meta(obj_, meta, opts)
    elif isinstance(obj_, list) and opts.meta.in_expand:
        # Build meta for list with correct size and type context
        # Determine source for type meta (last original element when sequence, else first element best-effort)
        source_for_type = None
        if isinstance(obj, (list, tuple)):
            source_for_type = obj[-1] if obj else None
        else:
            try:
                it = iter(obj)
                source_for_type = next(it, None)
            except Exception:
                source_for_type = None

        # Size meta should reflect the original iterable (for len/size/deep)
        size_meta = (
            SizeMeta.from_object(
                obj,
                include_len=opts.meta.len,
                include_deep=opts.meta.deep_size,
                include_shallow=opts.meta.size,
            )
            if opts.meta.sizes_enabled
            else None
        )

        # Trim meta: if we computed a specific one for list branch, prefer it; otherwise derive
        if opts.meta.trim:
            if "trim_meta" in locals() and trim_meta is not None:
                trim_part = trim_meta
            else:
                trim_part = TrimMeta.from_objects(obj, obj_)
        else:
            trim_part = None

        # Type meta: use element source if available, else the list itself
        type_part = (
            TypeMeta.from_objects(source_for_type if source_for_type is not None else obj, obj_)
            if opts.meta.type
            else None
        )

        # Assemble Meta manually to avoid recomputing inconsistent parts
        meta_obj = None
        if any([size_meta, trim_part, type_part]):
            meta_obj = Meta(size=size_meta, trim=trim_part, type=type_part)

        obj_ = opts.handlers.inject_meta(obj_, meta_obj, opts)

    return obj_


def inject_meta(
    obj: Any,
    meta: Meta | None,
    opts: DictifyOptions | None,
) -> Any:
    """
    Inject serialization metadata into a collection object.

    Embeds metadata into dict or list objects according to the configured META_KEY.
    The metadata structure and formatting are controlled by the provided options.

    Args:
        obj: The target object to inject metadata into (dict, list, or other).
        meta: Metadata to inject. If None, the object is returned unchanged.
        opts: Configuration options controlling metadata key, inclusion flags, and sorting.
             Uses default DictifyOptions if not provided.

    Returns:
        Modified object with metadata injected:
        - dict: Metadata added as {META_KEY: {...}} entry in the dictionary
        - list: Metadata appended as final element [{META_KEY: {...}}]
        - other: Original object returned unchanged

    Raises:
        TypeError: If meta is not a Meta instance or None.
        TypeError: If opts is not a DictifyOptions instance or None.
    """
    if not isinstance(opts, (DictifyOptions, type(None))):
        raise TypeError(f"opts must be a DictifyOptions, but got {fmt_type(opts)}")

    if not isinstance(meta, (Meta, type(None))):
        raise TypeError(f"meta must be a Meta, but got {fmt_type(meta)}")

    opts = opts or DictifyOptions()

    if not isinstance(obj, (list, dict)):
        return obj

    if meta is None:
        return obj

    meta_dict = meta.to_dict(
        include_none_attrs=opts.include_none_attrs,
        include_properties=opts.include_properties,
        sort_keys=opts.sort_keys,
    )

    if isinstance(obj, dict):
        obj[opts.meta.key] = meta_dict
        return obj

    elif isinstance(obj, list):
        obj.append({opts.meta.key: meta_dict})
        return obj


# Private Methods ------------------------------------------------------------------------------------------------------


def _attr_is_property(attr_name: str, obj, try_callable: bool = False) -> bool:
    """
    Check if a given attribute is a property of a class or an object.

    Performs consistent MRO (Method Resolution Order) lookup for both classes and instances,
    checking the attribute in the class hierarchy including inherited properties.

    Parameters:
        attr_name (str): The name of the attribute to check.
        obj: The class or object to check the attribute in.
        try_callable (bool, optional): Whether to test if the property getter can be called
            successfully on an instance. Defaults to False.

    Returns:
        bool: True if the attribute is a property, False otherwise.

    Behavior:
        - **Basic check** (try_callable=False): Returns True if attr_name is a property
          in the class or any parent class, regardless of whether it's accessible.

        - **Callable test** (try_callable=True):
          - On **classes**: Always returns False (properties can't be called on classes).
          - On **instances**: Returns True only if the property exists AND its getter
            executes successfully without raising an exception.

    Examples:
        >>> class Parent:
        ...     @property
        ...     def inherited(self): return 1
        >>> class Child(Parent):
        ...     @property
        ...     def own(self): return 2

        >>> # Basic property detection (works for both class and instance)
        >>> _attr_is_property('own', Child)
        True
        >>> _attr_is_property('inherited', Child) # (checks MRO)
        True
        >>> _attr_is_property('own', Child())
        True

        >>> # Callable testing
        >>> _attr_is_property('own', Child, try_callable=False)     # (callable found)
        True
        >>> _attr_is_property('own', Child(), try_callable=True)    # (getter succeeds)
        True

        >>> # Property with failing getter
        >>> class Broken:
        ...     @property
        ...     def bad(self): raise ValueError("oops")
        >>> _attr_is_property('bad', Broken())                      # (property exists)
        True
        >>> _attr_is_property('bad', Broken(), try_callable=True)   # (getter fails)
        False

    Note:
        Uses getattr() for MRO lookup, so inherited properties are detected consistently
        for both classes and instances. Catches all exceptions when testing property getters
        since property code can raise any exception type.
    """
    # Get the class to inspect
    cls = obj if inspect.isclass(obj) else type(obj)

    # Look up attribute in class MRO
    attr = getattr(cls, attr_name, None)
    is_property = isinstance(attr, property)

    # Early return for classes or when not trying callable
    if inspect.isclass(obj) or not try_callable or not is_property:
        return is_property

    # For instances with try_callable, test the getter
    try:
        attr.fget(obj)
        return True
    except Exception:
        return False


def _class_name(obj: Any, opts: DictifyOptions) -> str:
    """Return object class name."""
    return class_name(
        obj,
        fully_qualified=opts.class_name.fully_qualified,
        fully_qualified_builtins=False,
    )


def _dictify_core(obj, max_depth: int, opts: DictifyOptions):
    """Return dictify_core() overriding opts.max_depth"""
    opt_ = opts or DictifyOptions()
    opt_ = opt_.merge(max_depth=max_depth)
    return dictify_core(obj, opts=opt_)


def _iterable_to_mutable(obj: Iterable, opts: DictifyOptions) -> list | dict:
    """
    Convert iterable to a list or dict optionally applying keys/values sorting and trimming.
    """

    if not _is_iterable(obj):
        raise TypeError(f"Iterable expected but found {fmt_type(obj)}")

    # Check for named tuple
    if isinstance(obj, tuple) and hasattr(obj, "_fields") and hasattr(obj, "_asdict"):
        # It's a named tuple - convert to dict
        result_dict = dict(obj._asdict())

        # Sort first, then trim (for known length dict-like objects)
        if opts.sort_keys:
            items = sorted(result_dict.items())
        else:
            items = list(result_dict.items())

        # Apply max_items after sorting
        if opts.max_items is not None:
            items = items[: opts.max_items]

        return dict(items)

    # Fast path: Check if it's a Mapping type
    if isinstance(obj, abc.Mapping):
        # Sort first, then trim (for known length dict-like objects)
        if opts.sort_keys:
            items = sorted(obj.items())
        else:
            items = list(obj.items())

        # Apply max_items after sorting
        if opts.max_items is not None:
            items = items[: opts.max_items]

        return dict(items)

    # Handle ItemsView directly (e.g., dict.items())
    if isinstance(obj, abc.ItemsView):
        items = list(obj)

        # Sort first, then trim
        if opts.sort_keys:
            items.sort()

        # Apply max_items after sorting
        if opts.max_items is not None:
            items = items[: opts.max_items]

        # Check for hash collisions
        result_dict = dict(items)
        if len(result_dict) < len(items):
            return items

        return result_dict

    # Check if we have known length (not a generator)
    has_len = hasattr(obj, "__len__")

    if has_len and _is_items_iterable(obj):
        # Try to create dict from items() with known length
        try:
            items_iter = obj.items()
            items = list(items_iter)

            # Sort first, then trim
            if opts.sort_keys:
                items = sorted(items)

            # Apply max_items after sorting
            if opts.max_items is not None:
                items = items[: opts.max_items]

            # Try to create dict
            result_dict = dict(items)

            # Check if dict swallowed data (hash collision)
            if len(result_dict) < len(items):
                # Keep as list to preserve all items
                return items

            return result_dict

        except Exception:
            # Fall through to list creation
            pass

    # Unknown length but has .items() support (no sorting applied)
    if not has_len and _is_items_iterable(obj):
        try:
            items_iter = obj.items()

            # Apply max_items limit efficiently with islice
            if opts.max_items is not None:
                items = list(itertools.islice(items_iter, opts.max_items))
            else:
                items = list(items_iter)

            # Try to create dict from collected items
            result_dict = dict(items)

            # Check if dict swallowed data
            if len(result_dict) < len(items):
                return items

            return result_dict
        except Exception:
            # Fall through to list creation
            pass

    # Fallback: Create a list from the iterable
    # For sequences with known length, sort first then trim
    if has_len:
        result = list(obj)

        # Sort values if requested (for list-like objects)
        if opts.sort_iterables:
            result.sort()

        # Apply max_items after sorting
        if opts.max_items is not None:
            result = result[: opts.max_items]
    else:
        # Unknown length (generators) - no sorting, just trim with islice
        if opts.max_items is not None:
            result = list(itertools.islice(obj, opts.max_items))
        else:
            result = list(obj)

    return result


def _mapping_to_dict(mapping: abc.Mapping) -> dict:
    """
    Return a plain, writable dict with the same items as `mapping`.
    If `mapping` is already a dict but rejects assignment (e.g., proxy/RO),
    fall back to copying into a new dict.
    """
    if isinstance(mapping, dict):
        try:
            # Probe writability without mutating final state
            sentinel_key = object()
            mapping[sentinel_key] = None  # may raise if read-only
            del mapping[sentinel_key]
            return mapping
        except Exception:
            # Not writable: return a new plain dict copy
            return dict(mapping)
    # Not a real dict: normalize to plain dict
    return dict(mapping)


def _handlers_raw_chain(obj: Any, opts: DictifyOptions) -> Any:
    """
    handlers.raw chain of object processors with priority order
    handlers.raw() > obj.to_dict() > identity function
    """
    opts = opts or DictifyOptions()
    if opts.handlers.raw is not None:
        return opts.handlers.raw(obj, opts)
    if (dict_ := _get_from_to_dict(obj, opts=opts)) is not None:
        return dict_
    return obj  # Final fallback


def _handlers_terminal_chain(obj: Any, opts: DictifyOptions) -> Any:
    """
    handlers.terminal chain of object processors with priority order
    handlers.terminal() > type_handlers > obj.to_dict() > identity function
    """
    opts = opts or DictifyOptions()
    if opts.handlers.terminal is not None:
        return opts.handlers.terminal(obj, opts)
    if type_handler := opts.get_type_handler(obj):
        return type_handler(obj, opts)
    if (dict_ := _get_from_to_dict(obj, opts=opts)) is not None:
        return dict_
    return obj  # Final fallback


def _get_from_to_dict(obj, opts: DictifyOptions | None = None) -> dict[Any, Any] | None:
    """
    Returns obj.to_dict() value if the method is available and hook mode allows.

    Optionally injects class name and metadata
    """

    opts = opts or DictifyOptions()

    # Process HookMode ------------------------------------------

    if opts.hook_mode == HookMode.DICT:
        fn = getattr(obj, "to_dict", None)
        dict_ = fn() if callable(fn) else None

    elif opts.hook_mode == HookMode.DICT_STRICT:
        fn = getattr(obj, "to_dict", None)
        if not callable(fn):
            raise TypeError(
                f"Class {fmt_type(obj)} must implement to_dict() when hook_mode='{HookMode.DICT_STRICT}'"
            )
        dict_ = fn()

    elif opts.hook_mode == HookMode.NONE:
        dict_ = None

    else:
        valid = ", ".join([f"'{v.value}'" for v in HookMode])
        raise ValueError(f"Unknown hook_mode value: {fmt_any(opts.hook_mode)}. Expected: {valid}")

    # Check the returned type -----------------------------------
    if dict_ is not None:
        if not isinstance(dict_, abc.Mapping):
            raise TypeError(f"Object's to_dict() must return a Mapping, but got {fmt_type(dict_)}")

        # returned mapping should be of dict type
        dict_ = {**dict_}

        if opts.class_name.in_to_dict:
            dict_[opts.class_name.key] = _class_name(obj, opts)

        if opts.meta.in_to_dict:
            meta = Meta.from_objects(obj, dict_, opts=opts)
            dict_ = opts.handlers.inject_meta(dict_, meta, opts)

    return dict_


def _is_sized(obj: Any) -> bool:
    """Check whether an object is sized.

    Args:
        obj: Object to check.

    Returns:
        True if len(obj) succeeds, otherwise False.
    """
    try:
        value = len(obj)  # noqa: B008 - we're intentionally probing the protocol
        return isinstance(value, int) and value >= 0
    except TypeError:
        return False


def _is_sized_iterable(obj: Any) -> bool:
    """Check whether an object is both sized and iterable.

    Args:
        obj: Object to check.

    Returns:
        True if both iter(obj) and len(obj) succeed, otherwise False.
    """
    return _is_iterable(obj) and _is_sized(obj)


def _is_iterable(obj: Any) -> bool:
    """Check whether an object is iterable.

    Args:
        obj: Object to check.

    Returns:
        True if iter(obj) succeeds, otherwise False.
    """
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def _is_items_iterable(obj: Any) -> bool:
    """
    Return True if the object has an items() method that returns iterable.

    This is a lightweight check that assumes items() follows the mapping protocol
    (yields key-value pairs) but does NOT verify the actual item's structure.

    Does NOT consume any elements from the iterable.
    """
    if isinstance(obj, abc.Mapping):
        return True

    items_method = getattr(obj, "items", None)
    if not callable(items_method):
        return False

    try:
        items_obj = items_method()
        iter(items_obj)  # Verify it's iterable (doesn't consume items)
        return True
    except Exception:
        return False


def _handle_bytes(obj: bytes, opts: DictifyOptions) -> bytes:
    """Default handler for bytes objects - applies max_bytes limit."""
    if opts.max_bytes is None:
        return obj
    elif len(obj) > opts.max_bytes:
        return obj[: opts.max_bytes] + b"..."
    else:
        return obj


def _handle_bytearray(obj: bytearray, opts: DictifyOptions) -> bytearray:
    """Default handler for bytearray objects - applies max_bytes limit."""
    if opts.max_bytes is None:
        return obj
    elif len(obj) > opts.max_bytes:
        truncated = obj[: opts.max_bytes]
        return bytearray(truncated + b"...")
    else:
        return obj


def _handle_date(obj: date, opts: DictifyOptions) -> str:
    """Default handler for date objects - converts to ISO format string."""
    return obj.isoformat()


def _handle_datetime(obj: datetime, opts: DictifyOptions) -> str:
    """Default handler for datetime objects - converts to ISO format string."""
    return obj.isoformat()


def _handle_decimal(obj: Decimal, opts: DictifyOptions) -> str:
    """Default handler for Decimal objects - converts to string to preserve precision."""
    return str(obj)


def _handle_enum(obj: Enum, opts: DictifyOptions) -> dict[str, Any]:
    """Default handler for Enum objects - converts to dictionary with name and value."""
    return {
        "name": obj.name,
        "value": obj.value,
    }


def _handle_exception(obj: BaseException, opts: DictifyOptions) -> dict[str, Any]:
    """Default handler for Exception objects - converts to descriptive dictionary."""
    return {
        "type": "Exception",
        "class": obj.__class__.__name__,
        "message": str(obj),
        "args": obj.args,
    }


def _handle_fraction(obj: Fraction, opts: DictifyOptions) -> dict[str, Any]:
    """Default handler for Fraction objects - converts to descriptive dictionary."""
    return {
        "type": "Fraction",
        "numerator": obj.numerator,
        "denominator": obj.denominator,
        "value": float(obj),
    }


def _handle_memoryview(obj: memoryview, opts: DictifyOptions) -> dict[str, Any]:
    """Default handler for memoryview objects - converts to descriptive dictionary."""
    result = {
        "type": "memoryview",
        "nbytes": len(obj),
        "format": obj.format,
        "readonly": obj.readonly,
        "itemsize": obj.itemsize,
    }

    # Add shape info if available (for multi-dimensional views)
    if hasattr(obj, "shape") and obj.shape is not None:
        result["shape"] = obj.shape
        result["ndim"] = obj.ndim
        result["strides"] = obj.strides if hasattr(obj, "strides") else None

    # Always return a dict for memoryview:
    # - If max_bytes is None: include full untruncated data
    # - If len <= max_bytes: include full data
    # - Else: include preview and mark as truncated
    if opts.max_bytes is None:
        result["data"] = obj.tobytes()
        result["data_truncated"] = False
    # ... existing code ...
    elif len(obj) <= opts.max_bytes:
        result["data"] = obj.tobytes()
        result["data_truncated"] = False
    elif opts.max_bytes > 0:
        try:
            preview_data = obj[: opts.max_bytes].tobytes()
            result["data_preview"] = preview_data + b"..."
            result["data_truncated"] = True
        except (ValueError, TypeError):
            result["data_preview"] = None
            result["data_truncated"] = True

    return result


def _handle_path(obj: Path, options: DictifyOptions) -> dict[str, Any]:
    """Default handler for pathlib.Path objects - converts to descriptive dictionary."""
    return {
        "type": "Path",
        "path": str(obj),
        "name": obj.name,
        "suffix": obj.suffix,
        "exists": obj.exists(),
        "is_file": obj.is_file() if obj.exists() else None,
        "is_dir": obj.is_dir() if obj.exists() else None,
    }


def _handle_range(obj: range, options: DictifyOptions) -> str:
    """Default handler for range objects - converts to string representation."""
    return str(obj)


def _handle_regex_pattern(obj: re.Pattern, options: DictifyOptions) -> dict[str, Any]:
    """Default handler for compiled regex Pattern objects."""
    return {
        "type": "Pattern",
        "pattern": obj.pattern,
        "flags": obj.flags,
    }


def _handle_str(obj: str, opts: DictifyOptions) -> str:
    """Default handler for str objects - applies max_str_len limit."""
    if opts.max_str_len is None:
        return obj
    elif len(obj) > opts.max_str_len:
        return obj[: opts.max_str_len] + "..."
    else:
        return obj


def _handle_time(obj: time, opts: DictifyOptions) -> str:
    """Default handler for time objects - converts to ISO format string."""
    return obj.isoformat()


def _handle_timedelta(obj: timedelta, opts: DictifyOptions) -> dict[str, Any]:
    """Default handler for timedelta objects - converts to descriptive dictionary."""
    return {
        "type": "timedelta",
        "days": obj.days,
        "seconds": obj.seconds,
        "microseconds": obj.microseconds,
        "total_seconds": obj.total_seconds(),
    }


def _handle_uuid(obj: UUID, opts: DictifyOptions) -> str:
    """Default handler for UUID objects - converts to string representation."""
    return str(obj)


def _length_or_none(obj: Any) -> int | None:
    try:
        return len(obj)
    except Exception:
        return None


def _merge_new_default(new_, default_):
    """
    Return default_ if new_ is unset, otherwise return new_.

    This helper method allows to merging with special treatment of unset parameter as opposed to None as default.
    """
    if new_ is UNSET:
        return default_
    else:
        return new_


def _shallow_to_mutable(obj: Any, *, opts: DictifyOptions = None) -> dict[str, Any]:
    """
    Shallow object to dict converter for user objects with optional keys sorting and trimming.

    This method generates a dictionary with attributes and their corresponding values for a given object or class.
    No recursion applied.

    Method is NOT intended for primitives or iterables processing (list, tuple, dict, etc)

    Returns:
        dict[str, Any] - dictionary containing obj attributes and their values

    Note:
        include_none_attrs: Include attributes with None values
        include_private: Include private attributes of user classes
        include_property: Include instance properties with assigned values, has no effect if obj is a class
    """
    if _is_iterable(obj):
        raise ValueError(
            f"Cannot mutate an iterable type: {fmt_type(obj)}. Use _iterable_to_mutable() instead."
        )

    dict_ = {}

    attributes = search_attrs(
        obj,
        format="list",
        exclude_none=not opts.include_none_attrs,
        include_inherited=True,
        include_methods=False,
        include_private=opts.include_private,
        include_properties=opts.include_properties,
        sort=False,
        skip_errors=True,
    )
    is_class_or_dataclass = inspect.isclass(obj)

    for attr_name in attributes:
        if attr_name.startswith("__") and attr_name.endswith("__"):
            continue  # Skip dunder methods

        is_obj_property = _attr_is_property(attr_name, obj)

        if not is_class_or_dataclass and is_obj_property:
            if not opts.include_properties:
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

    # Sort first, then trim (for known length dict-like objects)
    if opts.sort_keys:
        items = sorted(dict_.items())
    else:
        items = list(dict_.items())

    # Apply max_items after sorting
    if opts.max_items is not None:
        items = items[: opts.max_items]

    return dict(items)


def _to_str(obj: Any, opts: DictifyOptions) -> str:
    """
    Returns <str> value of object, overrides default stdlib __str__.

    If custom __str__ method not found, replaces the stdlib __str__ with optionally Fully Qualified Class Name.
    """
    has_default_str = obj.__str__ == object.__str__
    if not has_default_str:
        as_str = str(obj)
    else:
        as_str = f"<class {_class_name(obj, opts)}>"
    return as_str


def _validate_sized_iterable(obj: Any):
    if not _is_sized_iterable(obj):
        raise TypeError(
            f"obj must be a Collection, MappingView or derived from Sized and implement __iter__, __len__ methods, "
            f"but found {fmt_type(obj)}"
        )


def dictify(
    obj: Any,
    *,
    max_depth: int = 3,
    max_items: int | None = 100,
    max_str_len: int | None = 200,
    max_bytes: int | None = 512,
    include_none: bool = False,
    include_private: bool = False,
    include_properties: bool = False,
    include_class_name: bool = False,
    sort_keys: bool = False,
    sort_iterables: bool = False,
    opts: DictifyOptions | None = None,
) -> Any:
    """
    Convert Python objects to human-readable dictionaries with common customizations.

    Simplified wrapper around dictify_core() for everyday use cases: CLI printing,
    basic logging, and object inspection. Provides direct parameter access to the
    most common conversion options with sensible defaults (DictifyOptions()).

    For specialized presets (logging, debugging, serialization) or advanced features
    (custom type handlers, edge-case processing, metadata injection), use dictify_core()
    with DictifyOptions.

    Args:
        obj: Object to convert to dictionary representation

        Depth & Size Controls:
            max_depth: Maximum recursion depth for nested objects (default: 3)
            max_items: Maximum items in collections before trimming (default: 100).
                       None = no limit (process entire collection).
            max_str_len: String truncation limit (default: 200), None = no truncation
            max_bytes: Bytes object truncation limit (default: 512), None = no truncation

        Attribute Filtering:
            include_none: Include attributes and dictionary items with None values
            include_private: Include private attributes starting with underscore
            include_properties: Include instance properties with assigned values

        Output Formatting:
            include_class_name: Include class name in object representations
            sort_keys: Sort dictionary keys alphabetically
            sort_iterables: Sort items in sequences, sets

        Advanced Controls:
            opts: DictifyOptions instance for advanced configuration.
                     Individual parameters override corresponding options fields.

    Returns:
        Human-readable dictionary representation preserving built-in types and
        converting objects to dictionaries

    Examples:
        >>> # Basic object conversion
        >>> class Person:
        ...     def __init__(self, name, age):
        ...         self.name = name
        ...         self.age = age
        >>> obj = Person("Alice", 7)
        >>> dictify(obj)
        {'name': 'Alice', 'age': 7}

        >>> # Include class information for debugging
        >>> dictify(obj, include_class_name=True)
        {'name': 'Alice', 'age': 7, '__class_name__': 'Person'}

        >>> # Control recursion depth and size limits
        >>> d = dictify(obj, max_depth=5, max_items=100)

        >>> # Include everything for debugging
        >>> d = dictify(obj, include_private=True, include_none=True,
        ...             include_properties=True, include_class_name=True)

        >>> # Sorted output for consistent display
        >>> d = dictify(obj, sort_keys=True, sort_iterables=True)

        >>> # Size limits for large strings/bytes
        >>> d = dictify(obj, max_str_len=200, max_bytes=512)

        >>> # Override options configuration
        >>> opts = DictifyOptions.debug()
        >>> d = dictify(obj, max_depth=10, opts=opts)  # max_depth overrides opts


    Note:
        - Built-in types (int, str, list, dict, etc.) are preserved as-is
        - Custom objects are converted to dictionaries with their attributes
        - Collections are recursively processed up to max_depth levels
        - Properties that raise exceptions are automatically skipped
        - For specialized presets, use: DictifyOptions.debug(), .logging(), .serial()
        - For custom type handlers, use dictify_core() with add_type_handler()
        - For metadata injection and edge-case handlers, use dictify_core()

    See Also:
        dictify_core: Core engine with full DictifyOptions configurability
        DictifyOptions: Configuration class with specialized preset factories
    """

    # Build options from parameters
    opts = opts or DictifyOptions()

    include_class_name = bool(include_class_name)
    include_none = bool(include_none)

    # Apply parameter overrides
    opts = opts.merge(
        max_depth=max_depth,
        max_items=max_items,
        max_str_len=max_str_len,
        max_bytes=max_bytes,
        include_none_attrs=include_none,
        include_none_items=include_none,
        include_private=bool(include_private),
        include_properties=bool(include_properties),
        sort_keys=bool(sort_keys),
        sort_iterables=bool(sort_iterables),
        class_name=ClassNameOptions(in_expand=include_class_name, in_to_dict=include_class_name),
    )

    return dictify_core(obj, opts=opts)
