"""
Sentinel objects for distinguishing between unset values, None, and other states.

This module provides a set of singleton sentinel objects that can be used to
represent special states in function arguments, data structures, and control flow.
All sentinels use identity checks (using 'is') rather than equality checks.

Sentinels:
    - UNSET: Represents an unprovided optional argument (distinguishes from None)
    - MISSING: Marks uninitialized or missing values in data structures
    - DEFAULT: Signals use of internal/calculated default value
    - NOT_FOUND: Indicates failed lookup operations (alternative to None)
    - STOP: Signals termination in iterators, queues, or producers/consumers

Helper Functions:
    - ifunset: Return default if value is UNSET, otherwise return value
    - ifnotmissing: Return default if value is MISSING, otherwise return value
    - ifnotdefault: Return default if value is DEFAULT, otherwise return value
    - iffound: Return default if value is NOT_FOUND, otherwise return value
    - ifnotstop: Return default if value is STOP, otherwise return value

Examples:
    >>> from c108.sentinels import UNSET, NOT_FOUND
    >>> def get_default_timeout() -> int:
    ...     return 30
    ...
    >>> def ifnotunset(value, default):
    ...     return default if value is UNSET else value
    ...
    >>> def fetch_data(timeout=UNSET):
    ...     timeout = ifnotunset(timeout, default=get_default_timeout())
    ...     return {"timeout": timeout}
    ...
    >>> fetch_data()
    {'timeout': 30}

    >>> cache = {"a": 1}
    >>> key = "b"
    >>> result = cache.get(key, NOT_FOUND)
    >>> if result is NOT_FOUND:
    ...     result = "computed"
    ...
    >>> result
    'computed'
"""

from typing import Any, Callable, Final, Type


# Base Sentinel --------------------------------------------------------------------------------------------------------


class SentinelBase:
    """
    Base class for sentinel objects.

    Sentinels are singleton objects optimized for identity checks.
    They provide clean representations and consistent behavior.
    """

    __slots__ = ("_name",)

    def __init__(self, name: str = "") -> None:
        self._name = name

    def __repr__(self) -> str:
        """Returns a clean string representation for debugging."""
        return f"<{self._name}>"

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
        """Prevent pickling of sentinels."""
        raise TypeError(
            f"Sentinels like {self!r} cannot be pickled. "
            "Import them directly where needed instead of serializing."
        )


# Sentinel Type Factory ------------------------------------------------------------------------------------------------


def create_sentinel_type(name: str, is_truthy: bool = False) -> Type[SentinelBase]:
    """
    Factory function to create singleton sentinel types.

    This method allows creating new sentinels with unified
    behavior like UNSET, MISSING, etc. provided by c108.sentinels.

    Args:
        name: The name of the sentinel (e.g., "UNSET", "MISSING", "PENDING")
        is_truthy: Whether the sentinel should evaluate to True in boolean context.
                Default is False (falsy). Set to True for sentinels that represent
                a "present" or "active" state.

    Returns:
        A new sentinel type class with singleton behavior.

    Example:
        Create a custom sentinel for async operations::

            PendingType = create_sentinel_type("PENDING")
            PENDING: Final[PendingType] = PendingType()

            async def fetch_data():
                result = PENDING
                # ... async operation ...
                return result if result is not PENDING else None

        Create a sentinel that's truthy::

            ActiveType = create_sentinel_type("ACTIVE", is_truthy=True)
            ACTIVE: Final[ActiveType] = ActiveType()

            if ACTIVE:  # Evaluates to True
                print("This will print")

        Use in function signatures::

            TimeoutType = create_sentinel_type("TIMEOUT")
            TIMEOUT: Final[TimeoutType] = TimeoutType()

            def wait_for(timeout=TIMEOUT):
                if timeout is TIMEOUT:
                    # Use default timeout logic
                    pass
    """

    class SentinelType(SentinelBase):
        __doc__ = f"Sentinel type for {name}."
        _instance: "SentinelType | None" = None

        def __new__(cls) -> "SentinelType":
            """Ensures singleton behavior."""
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

        def __init__(self) -> None:
            if not hasattr(self, "_name"):
                super().__init__(name)

        if is_truthy:

            def __bool__(self) -> bool:
                """Returns True as this sentinel indicates a present value."""
                return True

    # Set a meaningful class name for better debugging
    class_name = "".join(word.capitalize() for word in name.split("_")) + "Type"
    SentinelType.__name__ = class_name
    SentinelType.__qualname__ = class_name

    return SentinelType


# Sentinel Types -----------------------------------------------------------------------------------------------------

DefaultType = create_sentinel_type("DEFAULT", is_truthy=True)
"""
Sentinel type for DEFAULT.

Signals that a function should use its internal or calculated default value.
"""

MissingType = create_sentinel_type("MISSING")
"""
Sentinel type for MISSING.

Used to mark a value as not yet defined in a data structure,
such as an uninitialized dataclass field or missing dictionary key.
"""

NotFoundType = create_sentinel_type("NOT_FOUND")
"""
Sentinel type for NOT_FOUND.

Return value for lookup operations that fail, where None might be ambiguous.
Provides an alternative to raising exceptions in performance-critical code.
"""

StopType = create_sentinel_type("STOP")
"""
Sentinel type for STOP.

Used in iterators, queues, and producers/consumers to signal termination.
"""

UnsetType = create_sentinel_type("UNSET")
"""
Sentinel type for UNSET.

Used to distinguish between 'not provided' and 'explicitly set to None'.
This is the most common sentinel for optional function arguments.
"""

# Sentinel Objects -----------------------------------------------------------------------------------------------------

DEFAULT: Final[DefaultType] = DefaultType()
"""
Sentinel signaling use of internal default value.

Useful when you need to distinguish between "use this specific default"
and "calculate/use the standarddefault".
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


def _if_sentinel(
    value: Any,
    sentinel: Any,
    *,
    default: Any = None,
    default_factory: Callable[[], Any] | None = None,
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


def ifnotdefault(
    value: Any, *, default: Any = None, default_factory: Callable[[], Any] | None = None
) -> Any:
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
        >>> process('manual')
        'manual'
        >>> process(DEFAULT)
        'auto'
    """
    return _if_sentinel(value, DEFAULT, default=default, default_factory=default_factory)


def ifnotmissing(
    value: Any, *, default: Any = None, default_factory: Callable[[], Any] | None = None
) -> Any:
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
        >>> data = {"x": 1}
        >>> field = data.get('optional_field', MISSING)
        >>> ifnotmissing(field, default=0)
        0
    """
    return _if_sentinel(value, MISSING, default=default, default_factory=default_factory)


def ifnotnone(
    value: Any, *, default: Any = None, default_factory: Callable[[], Any] | None = None
) -> Any:
    """
    Return value if it's not None, otherwise return default.

    This helper is useful for providing fallback values when a variable might be None.
    When the value is None, it falls back to the default.

    Args:
        value: The value to check. If not None, this value is returned.
        default: The fallback value when value is None.
        default_factory: Callable returning the fallback value. Takes precedence over default.

    Returns:
        The value itself if not None, otherwise the default (or result of default_factory).

    Raises:
        ValueError: If both default and default_factory are provided.

    Example:
        >>> def get_config(timeout: int | None = None):
        ...     timeout = ifnotnone(timeout, default=30)
        ...     return timeout
        >>> get_config()
        30
        >>> get_config(60)
        60
        >>> get_config(None)
        30
    """
    return _if_sentinel(value, None, default=default, default_factory=default_factory)


def iffound(
    value: Any, *, default: Any = None, default_factory: Callable[[], Any] | None = None
) -> Any:
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
        >>> cache = {"a": 1}
        >>> result = cache.get("x", NOT_FOUND)
        >>> result = iffound(result, default=0)
        >>> result
        0
    """
    return _if_sentinel(value, NOT_FOUND, default=default, default_factory=default_factory)


def ifnotstop(
    value: Any, *, default: Any = None, default_factory: Callable[[], Any] | None = None
) -> Any:
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
        >>> from queue import Queue
        >>> queue = Queue()
        >>> queue.put(0)
        >>> queue.put(STOP)
        >>> while (item := ifnotstop(queue.get(), default=None)) is not None:
        ...     print(item)
        0
    """
    return _if_sentinel(value, STOP, default=default, default_factory=default_factory)


def ifnotunset(
    value: Any, *, default: Any = None, default_factory: Callable[[], Any] | None = None
) -> Any:
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
        >>> def configure(timeout: int | None = UNSET):
        ...     timeout = ifnotunset(timeout, default=30)
        ...     return timeout
        >>> configure()
        30
        >>> configure(60)
        60
        >>> configure(None)  # Returns None
    """
    return _if_sentinel(value, UNSET, default=default, default_factory=default_factory)
