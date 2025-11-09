"""
Dataclasses tools.

Provides a decorator that adds a merge() method to dataclasses, enabling
functional-style updates with sentinel values to distinguish "not provided"
from None or other values.
"""

# Standard library -----------------------------------------------------------------------------------------------------
import inspect
from dataclasses import dataclass, fields, Field
from dataclasses import _FIELD_INITVAR as dataclasses_FIELD_INITVAR
from typing import TypeVar, Callable, Any, get_type_hints, Protocol, runtime_checkable, cast
from typing_extensions import Self

# Local ----------------------------------------------------------------------------------------------------------------

from .sentinels import UNSET

# Classes --------------------------------------------------------------------------------------------------------------

T = TypeVar("T")

# Protocol for mergeable classes ---------------------------------------------------------------------------------------


@runtime_checkable
class Mergeable(Protocol[T]):
    """Protocol for classes decorated with @mergeable."""

    def merge(self: T, **kwargs: Any) -> T:
        """Create a new instance with selectively updated fields."""
        ...


# mergable -------------------------------------------------------------------------------------------------------------


def _is_init_var(field: Field) -> bool:
    """Check if a dataclass field is an InitVar."""
    return hasattr(field, "_field_type") and field._field_type == dataclasses_FIELD_INITVAR


def mergeable(
    cls: type[T] | None = None,
    *,
    sentinel=UNSET,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    include_private: bool = True,
) -> type[Mergeable[T]] | Callable[[type[T]], type[Mergeable[T]]]:
    """
    Decorator that adds a merge() method to a dataclass for creating modified copies.

    The merge() method creates a new instance with selectively updated fields,
    using a sentinel value to distinguish "not provided" from None or other values.

    Similar to dataclasses.replace() but with sentinel support and chainable syntax.

    Can be used with or without parentheses:
        @mergeable              # Uses all defaults
        @mergeable()            # Same as above
        @mergeable(sentinel=None)  # With parameters

    Args:
        cls: The dataclass to decorate (automatically provided when used without parentheses)
        sentinel: Sentinel value indicating "use existing value" (default: UNSET).
                  Common values: UNSET, MISSING (for distinguishing from None) or None.
        include: If provided, ONLY these fields can be merged (whitelist mode).
                 Overrides default field discovery. Can explicitly include private fields.
                 Cannot be used together with exclude.
        exclude: If provided, these fields are excluded from merging (blacklist mode).
                 Applied after default field discovery.
                 Cannot be used together with include.
        include_private: If True (default), private fields (starting with '_')
                        can be merged, matching dataclasses.replace() behavior.
                        If False, private fields are excluded by default.

    Returns:
        Decorated class with merge() method added, or decorator function if called with arguments.

    Raises:
        TypeError: If cls is not a dataclass
        ValueError: If both include and exclude are specified
        ValueError: If include/exclude reference non-existent fields
        ValueError: If include references fields with init=False
        ValueError: If include references InitVar fields

    Examples:
        >>> @mergeable
        ... @dataclass
        ... class Config:
        ...     timeout: int = 30
        ...     retries: int = 3
        ...     def merge(self, **kwargs) -> Self:
        ...         '''New Config instance with selectively updated fields'''
        ...         # This is a stub for Docs and type hinting
        ...         raise NotImplementedError("Implementation handled by @mergeable")

        >>> c1 = Config()
        >>> c2 = c1.merge(timeout=60)
        >>> c2.timeout
        60

        >>> @mergeable(sentinel=None)
        ... @dataclass
        ... class Options:
        ...     value: int | None = 5
        ...     def merge(self, **kwargs) -> Self:
        ...         '''New Options instance with selectively updated fields'''

        >>> o1 = Options()
        >>> o2 = o1.merge(value=None)  # None means "keep existing"
        >>> o2.value
        5

        >>> @mergeable(include=['timeout'])
        ... @dataclass
        ... class Limited:
        ...     timeout: int = 30
        ...     internal: int = 99
        ...     def merge(self, **kwargs) -> Self:
        ...         '''New Limited instance with selectively updated fields'''

        >>> lim = Limited()
        >>> lim.merge(timeout=60)
        Limited(timeout=60, internal=99)

    Notes:
        - Only fields with init=True are mergeable
        - InitVar fields are never mergeable
        - Fields with init=False are reset to defaults in new instance
        - Uses shallow copy semantics for field values
    """

    def decorator(target_cls: type[T]) -> type[Mergeable[T]]:
        # Validate it's a dataclass
        if not hasattr(target_cls, "__dataclass_fields__"):
            raise TypeError(f"{target_cls.__name__} must be a dataclass")

        # Validate include/exclude not both specified
        if include is not None and exclude is not None:
            raise ValueError("Cannot specify both 'include' and 'exclude' parameters")

        # Get all fields
        all_fields = fields(target_cls)
        field_map = {f.name: f for f in all_fields}

        # Get fields with init=True (candidates for merging)
        init_fields = [f for f in all_fields if f.init and not _is_init_var(f)]
        init_field_names = {f.name for f in init_fields}

        # Determine mergeable fields
        if include is not None:
            # Whitelist mode
            mergeable_names = set(include)

            # Validate all included fields exist
            for name in include:
                if name not in field_map:
                    raise ValueError(f"Field '{name}' does not exist on {target_cls.__name__}")
                field = field_map[name]
                if not field.init:
                    raise ValueError(f"Field '{name}' has init=False and cannot be merged")
                if _is_init_var(field):
                    raise ValueError(f"Field '{name}' is an InitVar and cannot be merged")

        elif exclude is not None:
            # Blacklist mode - start with init fields
            if include_private:
                base_names = {f.name for f in init_fields}
            else:
                base_names = {f.name for f in init_fields if not f.name.startswith("_")}

            # Filter out excluded fields (only exclude fields that actually exist)
            mergeable_names = base_names - {name for name in exclude if name in field_map}

        else:
            # Default mode
            if include_private:
                mergeable_names = {f.name for f in init_fields}
            else:
                mergeable_names = {f.name for f in init_fields if not f.name.startswith("_")}

        # Determine sentinel check function
        def __check_fn(value, default):
            return default if value is sentinel else value

        # Create merge method
        def merge(self, **kwargs):
            # Validate all kwargs are valid mergeable fields
            invalid = set(kwargs.keys()) - mergeable_names
            if invalid:
                raise TypeError(
                    f"merge() got unexpected keyword argument(s): {', '.join(sorted(invalid))}"
                )

            # Build merged dict for all init=True fields
            merged = {}
            for field in init_fields:
                fname = field.name
                if fname in mergeable_names and fname in kwargs:
                    # Mergeable field with new value provided
                    merged[fname] = __check_fn(kwargs[fname], getattr(self, fname))
                else:
                    # Not mergeable or not provided - use existing value
                    merged[fname] = getattr(self, fname)

            return target_cls(**merged)

        # Build signature for IDE support
        hints = get_type_hints(target_cls)
        params = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]

        for fname in sorted(mergeable_names):
            annotation = hints.get(fname, Any)
            params.append(
                inspect.Parameter(
                    fname, inspect.Parameter.KEYWORD_ONLY, default=sentinel, annotation=annotation
                )
            )

        merge.__signature__ = inspect.Signature(params, return_annotation=target_cls)
        merge.__annotations__ = {fname: hints.get(fname, Any) for fname in mergeable_names}
        merge.__annotations__["return"] = target_cls

        # Attach to class
        target_cls.merge = merge

        return cast(type[Mergeable[T]], target_cls)

    # Support both @mergeable and @mergeable()
    if cls is not None:
        # Called as @mergeable without parentheses
        return decorator(cls)
    else:
        # Called as @mergeable() with parentheses (or with arguments)
        return decorator
