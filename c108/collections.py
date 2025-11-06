"""
Bidirectional mapping utilities providing a generic BiDirectionalMap with forward and reverse lookups.
"""

# Standard library -----------------------------------------------------------------------------------------------------
import collections.abc as abc
from collections.abc import Iterator, KeysView, ValuesView, ItemsView
from typing import Any, Iterable, Mapping, TypeVar, Generic, overload

# Local ----------------------------------------------------------------------------------------------------------------
from .formatters import fmt_any
from .sentinels import MISSING

# Classes --------------------------------------------------------------------------------------------------------------

K = TypeVar("K")
V = TypeVar("V")


class BiDirectionalMap(Mapping[K, V], Generic[K, V]):
    """
    A bidirectional mapping that maintains one-to-one correspondence between keys and values.

    Implements the standardMapping protocol for forward lookups (key -> value) while providing
    efficient reverse lookups (value -> key). Both keys and values must be hashable and unique.

    Examples:
        >>> bimap = BiDirectionalMap({'a': 1, 'b': 2})
        >>> bimap['a']  # Forward lookup
        1
        >>> bimap.get_key(1)  # Reverse lookup
        'a'
        >>> 'a' in bimap  # Key membership
        True
        >>> bimap.has_value(1)  # Value membership
        True
        >>> bimap.to_dict()  # Extract as standarddict
        {'a': 1, 'b': 2}
        >>> bimap.reversed()  # Get reversed mapping
        BiDirectionalMap({1: 'a', 2: 'b'})

    Forward direction (key -> value):
        - Implements full Mapping protocol: __getitem__, __iter__, __len__, keys(), values(), items(), get()
        - Membership testing (x in bimap) applies to keys only, like dict

    Reverse direction (value -> key):
        - get_key(value): lookup key by value (raises KeyError if not found)
        - has_value(value): test if value exists

    Conversion and views:
        - to_dict(): extract as standarddict (key -> value)
        - reversed(): create new BiDirectionalMap with swapped keys/values

    Mutation operations maintain bidirectional consistency:
        - add(key, value): add new mapping, both must be unique
        - set(key, value): set/update mapping for key, value must be unique
        - pop(key, default=None): remove and return value for key
        - delete(key): remove mapping for key
        - update(mapping): bulk update with uniqueness validation
        - clear(): remove all mappings

    Raises:
        ValueError: When attempting to add duplicate keys or values
        KeyError: When accessing non-existent keys or values
        TypeError: When keys or values are not hashable
    """

    def __init__(self, initial: Mapping[K, V] | Iterable[tuple[K, V]] | None = None) -> None:
        """
        Initialize bidirectional map.

        Args:
            initial: Optional initial data as mapping or iterable of (key, value) pairs

        Raises:
            ValueError: If initial data contains duplicate keys or values
            TypeError: If keys or values are not hashable
        """
        self._forward_map: dict[K, V] = {}
        self._backward_map: dict[V, K] = {}
        if initial:
            self._init_from_data(initial)

    def _init_from_data(self, data: Mapping[K, V] | Iterable[tuple[K, V]]) -> None:
        """
        Initialize from data with strict duplicate validation.

        If the input `data` is another `BiDirectionalMap` instance, its internal maps are
        directly copied for efficiency, bypassing re-validation. Otherwise, it iterates
        through the data, validating uniqueness of keys and values.

        Args:
            data: Mapping or iterable of (key, value) pairs

        Raises:
            ValueError: If duplicate keys or values are found
        """
        if isinstance(data, BiDirectionalMap):
            # If initializing from another BiDirectionalMap, we can directly copy
            # its internal maps as uniqueness is already guaranteed.
            self._forward_map.update(data._forward_map)
            self._backward_map.update(data._backward_map)
        else:
            iterable = data.items() if isinstance(data, abc.Mapping) else data
            seen_keys: set[K] = set()
            seen_values: set[V] = set()

            for k, v in iterable:
                if k in seen_keys:
                    raise ValueError(f"Key already exists: {k!r}")
                if v in seen_values:
                    raise ValueError(f"Value already exists: {v!r}")
                seen_keys.add(k)
                seen_values.add(v)
                self._forward_map[k] = v
                self._backward_map[v] = k

    # ----- Mapping required methods -----

    def __getitem__(self, key: K) -> V:
        """Get value for key. Raises KeyError if key not found."""
        return self._forward_map[key]

    def __iter__(self) -> Iterator[K]:
        """Iterate over keys."""
        return iter(self._forward_map)

    def __len__(self) -> int:
        """Return number of key-value pairs."""
        return len(self._forward_map)

    def __contains__(self, key: object) -> bool:
        """Test if key exists (standardMapping behavior)."""
        return key in self._forward_map

    # ----- Mapping helpers (typed views) -----

    def keys(self) -> KeysView[K]:
        """Return view of keys."""
        return self._forward_map.keys()

    def values(self) -> ValuesView[V]:
        """Return view of values."""
        return self._forward_map.values()

    def items(self) -> ItemsView[K, V]:
        """Return view of (key, value) pairs."""
        return self._forward_map.items()

    def get(self, key: K, default: V | None = None) -> V | None:
        """Get value for key, returning default if key not found."""
        return self._forward_map.get(key, default)

    # ----- Bidirectional operations -----

    def get_value(self, key: K) -> V:
        """
        Get value for key (alias for __getitem__).

        Args:
            key: The key to look up

        Returns:
            The value associated with the key

        Raises:
            KeyError: If key not found
        """
        return self._forward_map[key]

    def get_key(self, value: V) -> K:
        """
        Get key for value (reverse lookup).

        Args:
            value: The value to look up

        Returns:
            The key associated with the value

        Raises:
            KeyError: If value not found
        """
        return self._backward_map[value]

    def has_value(self, value: V) -> bool:
        """
        Test if value exists in the mapping.

        Args:
            value: The value to test for

        Returns:
            True if value exists, False otherwise
        """
        return value in self._backward_map

    # ----- Conversion and views -----

    def to_dict(self) -> dict[K, V]:
        """
        Extract as standarddictionary (key -> value mapping).

        Returns a shallow copy of the forward mapping as a standarddict.
        Modifications to the returned dict do not affect this BiDirectionalMap.

        Returns:
            A new dict containing all key-value pairs

        Examples:
            >>> bimap = BiDirectionalMap({'a': 1, 'b': 2})
            >>> d = bimap.to_dict()
            >>> d
            {'a': 1, 'b': 2}
            >>> d['c'] = 3  # Does not affect bimap
        """
        return dict(self._forward_map)

    def reversed(self) -> "BiDirectionalMap[V, K]":
        """
        Create new BiDirectionalMap with keys and values swapped.

        Returns a new BiDirectionalMap where the original values become keys
        and the original keys become values. The original mapping is unchanged.

        Returns:
            A new BiDirectionalMap with swapped key-value pairs

        Examples:
            >>> bimap = BiDirectionalMap({'a': 1, 'b': 2})
            >>> rev = bimap.reversed()
            >>> rev.to_dict()
            {1: 'a', 2: 'b'}
            >>> rev[1]
            'a'
        """
        return BiDirectionalMap(self._backward_map)

    # ----- Mutations (keep both maps consistent) -----

    def add(self, key: K, value: V) -> None:
        """
        Add a new key-value pair. Both key and value must be unique.

        Args:
            key: The key to add
            value: The value to add

        Raises:
            ValueError: If key already exists or value already exists
            TypeError: If key or value is not hashable
        """
        if key in self._forward_map:
            raise ValueError(
                f"Key already exists: {fmt_any(key)} maps to {fmt_any(self._forward_map[key])})"
            )
        if value in self._backward_map:
            raise ValueError(
                f"Value already exists: {fmt_any(value)} mapped from {fmt_any(self._backward_map[value])})"
            )
        self._forward_map[key] = value
        self._backward_map[value] = key

    def set(self, key: K, value: V) -> None:
        """
        Set or update mapping for key. Value must be unique across all mappings.

        If key exists, its old value is released. The new value must not be
        used by any other key.

        Args:
            key: The key to set
            value: The value to associate with key

        Raises:
            ValueError: If value already exists (mapped from a different key)
            TypeError: If key or value is not hashable
        """
        if key in self._forward_map:
            old_value = self._forward_map[key]
            if old_value == value:
                return  # no-op
            if value in self._backward_map and self._backward_map[value] != key:
                raise ValueError(
                    f"Value already exists: {fmt_any(value)} mapped from {fmt_any(self._backward_map[value])}"
                )
            del self._backward_map[old_value]
        else:
            if value in self._backward_map:
                raise ValueError(
                    f"Value already exists: {fmt_any(value)} mapped from {fmt_any(self._backward_map[value])}"
                )

        self._forward_map[key] = value
        self._backward_map[value] = key

    @overload
    def pop(self, key: K) -> V: ...

    @overload
    def pop(self, key: K, default: V) -> V: ...

    def pop(self, key: K, default: V = MISSING) -> V:
        """
        Remove mapping for key and return its value.

        Args:
            key: The key to remove
            default: Value to return if key not found

        Returns:
            The value that was associated with key, or default if key not found

        Raises:
            KeyError: If key not found and no default provided
        """
        if key not in self._forward_map:
            if default is not MISSING:
                return default  # type: ignore[return-value]
            raise KeyError(key)
        value = self._forward_map.pop(key)
        del self._backward_map[value]
        return value

    def delete(self, key: K) -> None:
        """
        Remove mapping for key.

        Args:
            key: The key to remove

        Raises:
            KeyError: If key not found
        """
        value = self._forward_map.pop(key)
        del self._backward_map[value]

    def clear(self) -> None:
        """Remove all mappings."""
        self._forward_map.clear()
        self._backward_map.clear()

    def update(self, other: Mapping[K, V] | Iterable[tuple[K, V]]) -> None:
        """
        Update with key-value pairs from another mapping or iterable.

        Uses set() semantics: existing keys are updated, new keys are added.
        All values must satisfy uniqueness constraints.

        Args:
            other: Mapping or iterable of (key, value) pairs

        Raises:
            ValueError: If any value would violate uniqueness constraints
            TypeError: If any key or value is not hashable
        """
        iterable = other.items() if isinstance(other, abc.Mapping) else other
        for k, v in iterable:
            self.set(k, v)

    # ----- Equality and representation -----

    def __repr__(self) -> str:
        """Return string representation."""
        return f"BiDirectionalMap({self._forward_map!r})"

    def __eq__(self, other: Any) -> bool:
        """Test equality with another mapping based on key-value pairs."""
        if isinstance(other, abc.Mapping):
            return dict(self._forward_map.items()) == dict(other.items())
        return NotImplemented
