"""
C108 Sentinels

Sentinel objects for distinguishing between unset values, None, and other states.

This module provides a set of singleton sentinel objects that can be used to
represent special states in function arguments, data structures, and control flow.
All sentinels use identity checks (using 'is') rather than equality checks.

Requires Python 3.10+ for modern union syntax support.

Example:
    >>> def fetch_data(timeout: int | UnsetType = UNSET) -> dict:
    ...     if timeout is UNSET:
    ...         timeout = get_default_timeout()
    ...     # Use timeout...

    >>> result = cache.get(key, default=NOT_FOUND)
    >>> if result is NOT_FOUND:
    ...     result = expensive_computation(key)
"""

from typing import Final, Any

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


# Sentinel Objects -----------------------------------------------------------------------------------------------------

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


UNSET: Final[UnsetType] = UnsetType()
"""
Sentinel representing an unprovided optional argument.

Use with identity check: `if arg is UNSET:`

This is particularly useful when None is a valid input value.
"""


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


MISSING: Final[MissingType] = MissingType()
"""
Sentinel representing an uninitialized or missing value.

Commonly used in data validation libraries and ORMs to distinguish
between "field not provided" and "field set to None".
"""


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


DEFAULT: Final[DefaultType] = DefaultType()
"""
Sentinel signaling use of internal default value.

Useful when you need to distinguish between "use this specific default"
and "calculate/use the standard default".
"""


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


NOT_FOUND: Final[NotFoundType] = NotFoundType()
"""
Sentinel representing a failed lookup operation.

Useful for cache lookups, dictionary searches, or any operation where
None is a valid stored value but you need to signal absence.
"""


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


STOP: Final[StopType] = StopType()
"""
Sentinel signaling termination in iterators or queues.

Particularly useful in multi-threaded contexts where None might be
a legitimate queue item.
"""
