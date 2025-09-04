"""
C108 Collections
"""

# Standard library -----------------------------------------------------------------------------------------------------
from collections.abc import Mapping, Iterable, Iterator, KeysView, ValuesView, ItemsView
from typing import Any, Iterable, Callable, Set, Mapping, TypeVar, Generic, overload

# Local ----------------------------------------------------------------------------------------------------------------
from .tools import fmt_value

# Classes --------------------------------------------------------------------------------------------------------------

K = TypeVar("K")
V = TypeVar("V")


class BiDirectionalMap(Mapping[K, V], Generic[K, V]):
    """
    A bidirectional map with Mapping-compatible API on the forward direction.

    - Forward direction (key -> value) implements the stdlib Mapping protocol:
      __getitem__, __iter__, __len__, keys(), values(), items(), get().
      Membership (x in bimap) applies to KEYS only, like dict.
    - Reverse direction (value -> key) available via get_key(value) and has_value(value).
    - Enforces uniqueness of both keys and values (both must be hashable).
    """

    def __init__(self, initial: Mapping[K, V] | Iterable[tuple[K, V]] | None = None) -> None:
        self._forward_map: dict[K, V] = {}
        self._backward_map: dict[V, K] = {}
        if initial:
            self.update(initial)

    # ----- Mapping required methods -----

    def __getitem__(self, key: K) -> V:
        return self._forward_map[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._forward_map)

    def __len__(self) -> int:
        return len(self._forward_map)

    # ----- Mapping helpers (typed views) -----

    def keys(self) -> KeysView[K]:
        return self._forward_map.keys()

    def values(self) -> ValuesView[V]:
        return self._forward_map.values()

    def items(self) -> ItemsView[K, V]:
        return self._forward_map.items()

    def get(self, key: K, default: V | None = None) -> V | None:
        return self._forward_map.get(key, default)

    # ----- Bidirectional operations -----

    def get_value(self, key: K) -> V:
        """Lookup value by key (same as __getitem__)."""
        return self._forward_map[key]

    def get_key(self, value: V) -> K:
        """Lookup key by value."""
        return self._backward_map[value]

    def has_value(self, value: V) -> bool:
        """True if value exists in reverse map."""
        return value in self._backward_map

    # ----- Mutations (keep both maps consistent) -----

    def add(self, key: K, value: V) -> None:
        """
        Add a key-value pair; both key and value must be unique.

        Raises:
            ValueError if key already exists or value already exists (mapped from a different key).
        """
        if key in self._forward_map:
            raise ValueError(f"Key {fmt_value(key)} already exists (maps to {self._forward_map[key]!r})")
        if value in self._backward_map:
            raise ValueError(f"Value {fmt_value(value)} already exists (mapped from {self._backward_map[value]!r})")
        self._forward_map[key] = value
        self._backward_map[value] = key

    def set(self, key: K, value: V) -> None:
        """
        Set or replace a mapping for key, enforcing value uniqueness.
        If key existed, its old value is released; the new value must not be used by another key.
        """
        if key in self._forward_map:
            old_value = self._forward_map[key]
            if old_value == value:
                return  # no-op
            if value in self._backward_map and self._backward_map[value] is not key:
                raise ValueError(f"Value {fmt_value(value)} already exists (mapped from {self._backward_map[value]!r})")
            del self._backward_map[old_value]
        else:
            if value in self._backward_map:
                raise ValueError(f"Value {fmt_value(value)} already exists (mapped from {self._backward_map[value]!r})")

        self._forward_map[key] = value
        self._backward_map[value] = key

    class _Missing:
        pass

    _MISSING = _Missing()

    @overload
    def pop(self, key: K) -> V:
        ...

    @overload
    def pop(self, key: K, default: V) -> V:
        ...

    def pop(self, key: K, default: V | _Missing = _MISSING) -> V:
        """
        Remove mapping for key and return its value. If key is absent:
          - return default if provided,
          - otherwise raise KeyError.
        """
        if key not in self._forward_map:
            if not isinstance(default, BiDirectionalMap._Missing):
                return default  # type: ignore[return-value]
            raise KeyError(key)
        value = self._forward_map.pop(key)
        del self._backward_map[value]
        return value

    def delete(self, key: K) -> None:
        """Remove mapping for key. Raises KeyError if missing."""
        value = self._forward_map.pop(key)
        del self._backward_map[value]

    def clear(self) -> None:
        """Remove all entries."""
        self._forward_map.clear()
        self._backward_map.clear()

    def update(self, other: Mapping[K, V] | Iterable[tuple[K, V]]) -> None:
        """
        Bulk add/update. Enforces uniqueness constraints across all pairs.
        Uses set() semantics per key while ensuring reverse uniqueness.
        """
        iterable = other.items() if isinstance(other, Mapping) else other
        for k, v in iterable:
            self.set(k, v)

    # ----- Equality and representation -----

    def __repr__(self) -> str:
        return f"BiDirectionalMap({self._forward_map!r})"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Mapping):
            return dict(self._forward_map.items()) == dict(other.items())
        return NotImplemented
