"""
C108 Collections
"""

# Standard library -----------------------------------------------------------------------------------------------------
import collections

from typing import Any, Iterable, Callable, Set, Mapping


# Classes --------------------------------------------------------------------------------------------------------------

# TODO implement Mapping interface
class BiDirectionalMap:
    """
    A map that provides bidirectional lookup, ensuring both keys and values are unique.
    """

    def __init__(self, initial_map: dict = None):
        self._forward_map = {}  # key -> value
        self._backward_map = {}  # value -> key

        if initial_map:
            for key, value in initial_map.items():
                self.add(key, value)  # Use the add method to enforce uniqueness

    def add(self, key, value):
        """
        Adds a key-value pair to the map. Raises ValueError if key or value already exists.
        """
        if key in self._forward_map:
            raise ValueError(f"Key '{key}' already exists in the map, mapping to '{self._forward_map[key]}'.")
        if value in self._backward_map:
            raise ValueError(
                f"Value '{value}' already exists in the map, mapped from '{self._backward_map[value]}'. Values must be unique.")

        self._forward_map[key] = value
        self._backward_map[value] = key

    def get_value(self, key):
        """
        Looks up a value by its key.
        """
        return self._forward_map[key]

    def get_key(self, value):
        """
        Looks up a key by its value.
        """
        return self._backward_map[value]

    def __getitem__(self, key):
        """Allows dictionary-like access for key to value."""
        return self.get_value(key)

    def __contains__(self, item):
        """Checks if a key or value exists in the map."""
        return item in self._forward_map or item in self._backward_map

    def __len__(self):
        return len(self._forward_map)

    def __repr__(self):
        return f"BiDirectionalMap({self._forward_map})"

    def keys(self):
        return self._forward_map.keys()

    def values(self):
        return self._forward_map.values()

    def items(self):
        return self._forward_map.items()

    # You can add a deletion method if needed, being careful to update both maps
    def delete(self, key):
        if key not in self._forward_map:
            raise KeyError(f"Key '{key}' not found in map.")
        value_to_delete = self._forward_map.pop(key)
        del self._backward_map[value_to_delete]
