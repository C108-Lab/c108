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
    ifnotmissing: Return default if value is MISSING, otherwise return value
    ifnotdefault: Return default if value is DEFAULT, otherwise return value
    iffound: Return default if value is NOT_FOUND, otherwise return value
    ifnotstop: Return default if value is STOP, otherwise return value

Example:
    >>> def fetch_data(timeout: int | UnsetType = UNSET) -> dict:
    ...     timeout = ifnotunset(timeout, default=get_default_timeout())
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
    'ifnotunset',
    'ifnotmissing',
    'ifnotdefault',
    'iffound',
    'ifnotstop',
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

# Helper Functions -----------------------------------------------------------------------------------------------------

def _if_sentinel(
        value: Any,
        sentinel: Any,
        *,
        default: Any = None,
        default_factory: Callable[[], Any] | None = None
) -> Any:
    """
    Internal helper: return value if it doesn't match sentinel, otherwise return default.

    Args:
        value: The value to check against the sentinel.
        sentinel: The sentinel object to check against.
        default: The default value to return if value matches sentinel.
        default_factory: Callable that returns the default. Takes precedence over default.

    Returns:
        The value if it doesn't match sentinel, otherwise the default (or result of default_factory).

    Raises:
        ValueError: If both default and default_factory are provided.
    """
    if value is not sentinel:
        return value

    if default_factory is not None and default is not None:
        raise ValueError("Cannot specify both default and default_factory")

    if default_factory is not None:
        return default_factory()

    return default


def ifnotunset(value: Any, *, default: Any = None, default_factory: Callable[[], Any] | None = None) -> Any:
    """
    Return value if it's not UNSET, otherwise return default.

    This helper is useful for handling optional parameters where None is a valid value.
    When the value is UNSET (not provided), it falls back to the default.

    Args:
        value: The value to check. If not UNSET, this value is returned.
        default: The fallback value when value is UNSET.
        default_factory: Callable returning the fallback value. Takes precedence over default.

    Returns:
        The value itself if not UNSET, otherwise the default (or result of default_factory).

    Raises:
        ValueError: If both default and default_factory are provided.

    Example:
        >>> def configure(timeout: int | UnsetType = UNSET):
        ...     timeout = ifnotunset(timeout, default=30)
        ...     return timeout
        >>> configure()  # Returns 30
        >>> configure(60)  # Returns 60
        >>> configure(None)  # Returns None (None is valid, not UNSET)
    """
    return _if_sentinel(value, UNSET, default=default, default_factory=default_factory)


def ifnotmissing(value: Any, *, default: Any = None, default_factory: Callable[[], Any] | None = None) -> Any:
    """
    Return value if it's not MISSING, otherwise return default.

    This helper is useful for data validation and handling uninitialized fields
    in data structures where you need to distinguish "not provided" from None.

    Args:
        value: The value to check. If not MISSING, this value is returned.
        default: The fallback value when value is MISSING.
        default_factory: Callable returning the fallback value. Takes precedence over default.

    Returns:
        The value itself if not MISSING, otherwise the default (or result of default_factory).

    Raises:
        ValueError: If both default and default_factory are provided.

    Example:
        >>> field = data.get('optional_field', MISSING)
        >>> field = ifnotmissing(field, default=0)
        >>> tags = ifnotmissing(record.tags, default_factory=list)
    """
    return _if_sentinel(value, MISSING, default=default, default_factory=default_factory)


def ifnotdefault(value: Any, *, default: Any = None, default_factory: Callable[[], Any] | None = None) -> Any:
    """
    Return value if it's not DEFAULT, otherwise return default.

    This helper is useful when you want to allow users to explicitly request
    the default behavior by passing DEFAULT, while still accepting custom values.

    Args:
        value: The value to check. If not DEFAULT, this value is returned.
        default: The fallback value when value is DEFAULT.
        default_factory: Callable returning the fallback value. Takes precedence over default.

    Returns:
        The value itself if not DEFAULT, otherwise the default (or result of default_factory).

    Raises:
        ValueError: If both default and default_factory are provided.

    Example:
        >>> def process(mode: str | DefaultType = DEFAULT):
        ...     mode = ifnotdefault(mode, default='auto')
        ...     return mode
        >>> process('manual')  # Returns 'manual'
        >>> process(DEFAULT)   # Returns 'auto'
    """
    return _if_sentinel(value, DEFAULT, default=default, default_factory=default_factory)


def iffound(value: Any, *, default: Any = None, default_factory: Callable[[], Any] | None = None) -> Any:
    """
    Return value if it's not NOT_FOUND, otherwise return default.

    This helper is useful for lookup operations where None might be a valid
    stored value, so you need a separate sentinel to indicate "not found".

    Args:
        value: The value to check. If not NOT_FOUND, this value is returned.
        default: The fallback value when value is NOT_FOUND.
        default_factory: Callable returning the fallback value. Takes precedence over default.

    Returns:
        The value itself if not NOT_FOUND, otherwise the default (or result of default_factory).

    Raises:
        ValueError: If both default and default_factory are provided.

    Example:
        >>> result = cache.get(key, NOT_FOUND)
        >>> result = iffound(result, default=0)
        >>> result = iffound(result, default_factory=lambda: compute_value(key))
    """
    return _if_sentinel(value, NOT_FOUND, default=default, default_factory=default_factory)


def ifnotstop(value: Any, *, default: Any = None, default_factory: Callable[[], Any] | None = None) -> Any:
    """
    Return value if it's not STOP, otherwise return default.

    This helper is useful in iterators, queues, and producer-consumer patterns
    where you need to signal termination without using None or exceptions.

    Args:
        value: The value to check. If not STOP, this value is returned.
        default: The fallback value when value is STOP.
        default_factory: Callable returning the fallback value. Takes precedence over default.

    Returns:
        The value itself if not STOP, otherwise the default (or result of default_factory).

    Raises:
        ValueError: If both default and default_factory are provided.

    Example:
        >>> item = queue.get()
        >>> item = ifnotstop(item, default=None)
        >>> while (item := ifnotstop(queue.get(), default=None)) is not None:
        ...     process(item)
    """
    return _if_sentinel(value, STOP, default=default, default_factory=default_factory)
