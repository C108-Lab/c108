"""
Sentinel objects for distinguishing between unset values, None, and other states.

This module provides a set of singleton sentinel objects that can be used to
represent special states in function arguments, data structures, and control flow.
All sentinels use identity checks (using 'is') rather than equality checks.

Sentinels:
    UNSET: Represents an unprovided optional argument (distinguishes from None)
    MISSING: Marks uninitialized or missing values in data structures
    DEFAULT: Signals use of internal/calculated default value
    NOT_FOUND: Indicates failed lookup operations (alternative to None)
    STOP: Signals termination in iterators, queues, or producers/consumers

Helper Functions:
    ifunset: Return default if value is UNSET, otherwise return value
    ifmissing: Return default if value is MISSING, otherwise return value
    ifdefault: Return default if value is DEFAULT, otherwise return value
    ifnotfound: Return default if value is NOT_FOUND, otherwise return value
    ifstop: Return default if value is STOP, otherwise return value

Example:
    >>> def fetch_data(timeout: int | UnsetType = UNSET) -> dict:
    ...     timeout = ifunset(timeout, default=get_default_timeout())
    ...     # Use timeout...

    >>> result = cache.get(key, default=NOT_FOUND)
    >>> if result is NOT_FOUND:
    ...     result = expensive_computation(key)
"""

from typing import Any, Callable, Final

__all__ = [
    'UNSET',
    'MISSING',
    'DEFAULT',
    'NOT_FOUND',
    'STOP',
    'UnsetType',
    'MissingType',
    'DefaultType',
    'NotFoundType',
    'StopType',
    'ifunset',
    'ifmissing',
    'ifdefault',
    'ifnotfound',
    'ifstop',
]


# Base Sentinel --------------------------------------------------------------------------------------------------------

class _SentinelBase:
    """
    Base class for all sentinel objects.

    Sentinels are singleton objects optimized for identity checks.
    They provide clean representations and consistent behavior.
    """
    __slots__ = ('_name',)

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        """Returns a clean string representation for debugging."""
        return f'<{self._name}>'

    def __eq__(self, other: Any) -> bool:
        """Ensures identity-based comparison."""
        return self is other

    def __hash__(self) -> int:
        """Returns a hash based on object identity."""
        return id(self)

    def __bool__(self) -> bool:
        """Returns False by default (sentinels are typically falsy)."""
        return False

    def __reduce__(self) -> tuple:
        """Ensures proper behavior during pickling."""
        return (self.__class__, (self._name,))


# Sentinel Types -----------------------------------------------------------------------------------------------------

class DefaultType(_SentinelBase):
    """
    Sentinel type for DEFAULT.

    Signals that a function should use its internal or calculated default value.
    """
    _instance: 'DefaultType | None' = None

    def __new__(cls) -> 'DefaultType':
        """Ensures singleton behavior."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, '_name'):
            super().__init__("DEFAULT")

    def __bool__(self) -> bool:
        """Returns True as DEFAULT typically indicates a present value."""
        return True

    def __reduce__(self) -> tuple:
        """Ensure pickling returns the singleton instance."""
        return (self.__class__, ())


class MissingType(_SentinelBase):
    """
    Sentinel type for MISSING.

    Used to mark a value as not yet defined in a data structure,
    such as an uninitialized dataclass field or missing dictionary key.
    """
    _instance: 'MissingType | None' = None

    def __new__(cls) -> 'MissingType':
        """Ensures singleton behavior."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, '_name'):
            super().__init__("MISSING")

    def __reduce__(self) -> tuple:
        """Ensure pickling returns the singleton instance."""
        return (self.__class__, ())


class NotFoundType(_SentinelBase):
    """
    Sentinel type for NOT_FOUND.

    Return value for lookup operations that fail, where None might be ambiguous.
    Provides an alternative to raising exceptions in performance-critical code.
    """
    _instance: 'NotFoundType | None' = None

    def __new__(cls) -> 'NotFoundType':
        """Ensures singleton behavior."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, '_name'):
            super().__init__("NOT_FOUND")

    def __reduce__(self) -> tuple:
        """Ensure pickling returns the singleton instance."""
        return (self.__class__, ())


class StopType(_SentinelBase):
    """
    Sentinel type for STOP.

    Used in iterators, queues, and producers/consumers to signal termination.
    """
    _instance: 'StopType | None' = None

    def __new__(cls) -> 'StopType':
        """Ensures singleton behavior."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, '_name'):
            super().__init__("STOP")

    def __reduce__(self) -> tuple:
        """Ensure pickling returns the singleton instance."""
        return (self.__class__, ())


class UnsetType(_SentinelBase):
    """
    Sentinel type for UNSET.

    Used to distinguish between 'not provided' and 'explicitly set to None'.
    This is the most common sentinel for optional function arguments.
    """
    _instance: 'UnsetType | None' = None

    def __new__(cls) -> 'UnsetType':
        """Ensures singleton behavior."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, '_name'):
            super().__init__("UNSET")

    def __reduce__(self) -> tuple:
        """Ensure pickling returns the singleton instance."""
        return (self.__class__, ())


# Sentinel Objects -----------------------------------------------------------------------------------------------------

DEFAULT: Final[DefaultType] = DefaultType()
"""
Sentinel signaling use of internal default value.

Useful when you need to distinguish between "use this specific default"
and "calculate/use the standard default".
"""

MISSING: Final[MissingType] = MissingType()
"""
Sentinel representing an uninitialized or missing value.

Commonly used in data validation libraries and ORMs to distinguish
between "field not provided" and "field set to None".
"""

NOT_FOUND: Final[NotFoundType] = NotFoundType()
"""
Sentinel representing a failed lookup operation.

Useful for cache lookups, dictionary searches, or any operation where
None is a valid stored value but you need to signal absence.
"""

STOP: Final[StopType] = StopType()
"""
Sentinel signaling termination in iterators or queues.

Particularly useful in multi-threaded contexts where None might be
a legitimate queue item.
"""

UNSET: Final[UnsetType] = UnsetType()
"""
Sentinel representing an unprovided optional argument.

Use with identity check: `if arg is UNSET:`

This is particularly useful when None is a valid input value.
"""

# Helper Functions -----------------------------------------------------------------------------------------------------

from typing import Callable


# Helper Functions -----------------------------------------------------------------------------------------------------

def ifunset(value: Any, *, default: Any = None, default_factory: Callable[[], Any] | None = None) -> Any:
    """
    Return default if value is UNSET, otherwise return value.

    Args:
        value: The value to check against UNSET sentinel.
        default: The default value to return if value is UNSET.
        default_factory: Callable that returns the default. Takes precedence over default.

    Returns:
        The default (or result of default_factory) if value is UNSET, otherwise the original value.

    Raises:
        ValueError: If both default and default_factory are provided.

    Example:
        >>> timeout = ifunset(timeout, default=30)
        >>> items = ifunset(items, default_factory=list)
        >>> config = ifunset(config, default_factory=load_defaults)
    """
    return _if_sentinel(value, UNSET, default=default, default_factory=default_factory)


def ifmissing(value: Any, *, default: Any = None, default_factory: Callable[[], Any] | None = None) -> Any:
    """
    Return default if value is MISSING, otherwise return value.

    Args:
        value: The value to check against MISSING sentinel.
        default: The default value to return if value is MISSING.
        default_factory: Callable that returns the default. Takes precedence over default.

    Returns:
        The default (or result of default_factory) if value is MISSING, otherwise the original value.

    Raises:
        ValueError: If both default and default_factory are provided.

    Example:
        >>> field_value = ifmissing(data.get('field', MISSING), default=0)
        >>> tags = ifmissing(record.tags, default_factory=list)
    """
    return _if_sentinel(value, MISSING, default=default, default_factory=default_factory)


def ifdefault(value: Any, *, default: Any = None, default_factory: Callable[[], Any] | None = None) -> Any:
    """
    Return default if value is DEFAULT, otherwise return value.

    Args:
        value: The value to check against DEFAULT sentinel.
        default: The default value to return if value is DEFAULT.
        default_factory: Callable that returns the default. Takes precedence over default.

    Returns:
        The default (or result of default_factory) if value is DEFAULT, otherwise the original value.

    Raises:
        ValueError: If both default and default_factory are provided.

    Example:
        >>> setting = ifdefault(user_setting, default=100)
        >>> config = ifdefault(override, default_factory=compute_default)
    """
    return _if_sentinel(value, DEFAULT, default=default, default_factory=default_factory)


def ifnotfound(value: Any, *, default: Any = None, default_factory: Callable[[], Any] | None = None) -> Any:
    """
    Return default if value is NOT_FOUND, otherwise return value.

    Args:
        value: The value to check against NOT_FOUND sentinel.
        default: The default value to return if value is NOT_FOUND.
        default_factory: Callable that returns the default. Takes precedence over default.

    Returns:
        The default (or result of default_factory) if value is NOT_FOUND, otherwise the original value.

    Raises:
        ValueError: If both default and default_factory are provided.

    Example:
        >>> result = cache.get(key, NOT_FOUND)
        >>> result = ifnotfound(result, default=0)
        >>> result = ifnotfound(result, default_factory=lambda: compute_value(key))
    """
    return _if_sentinel(value, NOT_FOUND, default=default, default_factory=default_factory)


def ifstop(value: Any, *, default: Any = None, default_factory: Callable[[], Any] | None = None) -> Any:
    """
    Return default if value is STOP, otherwise return value.

    Args:
        value: The value to check against STOP sentinel.
        default: The default value to return if value is STOP.
        default_factory: Callable that returns the default. Takes precedence over default.

    Returns:
        The default (or result of default_factory) if value is STOP, otherwise the original value.

    Raises:
        ValueError: If both default and default_factory are provided.

    Example:
        >>> item = queue.get()
        >>> item = ifstop(item, default=None)
        >>> item = ifstop(item, default_factory=get_placeholder)
    """
    return _if_sentinel(value, STOP, default=default, default_factory=default_factory)


def _if_sentinel(
        value: Any,
        sentinel: Any,
        *,
        default: Any = None,
        default_factory: Callable[[], Any] | None = None
) -> Any:
    """
    Internal helper: return default if value matches sentinel, otherwise return value.

    Args:
        value: The value to check against the sentinel.
        sentinel: The sentinel object to check against.
        default: The default value to return if value matches sentinel.
        default_factory: Callable that returns the default. Takes precedence over default.

    Returns:
        The default (or result of default_factory) if value matches sentinel, otherwise the original value.

    Raises:
        ValueError: If both default and default_factory are provided.
    """
    if value is not sentinel:
        return value

    if default_factory is not None and default is not None:
        raise ValueError("cannot specify both default and default_factory, use only one of them.")

    if default_factory is not None:
        return default_factory()

    return default
