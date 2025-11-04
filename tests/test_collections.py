#
# C108 - Collection Tests
#

# Third-party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.collections import BiDirectionalMap


# Tests ----------------------------------------------------------------------------------------------------------------


class TestBiDirectionalMap:
    """Tests for the BiDirectionalMap class."""

    def test_init_and_lookups(self):
        """Verify initialization and basic forward/reverse lookups."""
        bimap = BiDirectionalMap({"a": 1, "b": 2})
        assert bimap["a"] == 1
        assert bimap.get_key(2) == "b"
        assert len(bimap) == 2

    def test_mapping_protocol_conformance(self):
        """Ensure it behaves like a standardmapping."""
        data = {"a": 1, "b": 2}
        bimap = BiDirectionalMap(data)
        assert list(bimap) == ["a", "b"]
        assert "a" in bimap
        assert "c" not in bimap
        assert bimap.has_value(1)
        assert not bimap.has_value(3)
        assert set(bimap.keys()) == {"a", "b"}
        assert set(bimap.values()) == {1, 2}
        assert set(bimap.items()) == {("a", 1), ("b", 2)}

    @pytest.mark.parametrize(
        "initial_data, expected_error_msg",
        [
            ([("a", 1), ("a", 2)], r"(?i)key already exists"),
            ([("a", 1), ("b", 1)], r"(?i)value already exists"),
        ],
        ids=["duplicate_key_on_init", "duplicate_value_on_init"],
    )
    def test_initialization_failures(self, initial_data, expected_error_msg):
        """Check for uniqueness validation during initialization."""
        with pytest.raises(ValueError, match=expected_error_msg):
            BiDirectionalMap(initial_data)

    @pytest.mark.parametrize(
        "key, value, expected_error_msg",
        [
            ("a", 3, r"(?i)key already exists"),
            ("c", 1, r"(?i)value already exists"),
        ],
        ids=["duplicate_key", "duplicate_value"],
    )
    def test_add_uniqueness_violations(self, key, value, expected_error_msg):
        """Check that add() raises ValueError for duplicate keys or values."""
        bimap = BiDirectionalMap({"a": 1, "b": 2})
        with pytest.raises(ValueError, match=expected_error_msg):
            bimap.add(key, value)

    def test_set_operations(self):
        """Verify set() correctly adds a new item and updates an existing one."""
        bimap = BiDirectionalMap({"a": 1})
        # Set a new item
        bimap.set("b", 2)
        assert bimap["b"] == 2
        assert bimap.get_key(2) == "b"
        # Update an existing item, releasing the old value
        bimap.set("a", 3)
        assert bimap["a"] == 3
        assert bimap.get_key(3) == "a"
        assert not bimap.has_value(1)
        with pytest.raises(ValueError, match=r"(?i)value already exists"):
            bimap.set("c", 3)

    def test_pop_and_delete(self):
        """Verify pop() and delete() remove items from both internal maps."""
        bimap = BiDirectionalMap({"a": 1, "b": 2, "c": 3})
        # Test pop returns value and removes pair
        popped_value = bimap.pop("b")
        assert popped_value == 2
        assert "b" not in bimap
        assert not bimap.has_value(2)
        assert len(bimap) == 2
        # Test delete removes pair
        bimap.delete("a")
        assert "a" not in bimap
        assert not bimap.has_value(1)
        assert len(bimap) == 1
        # Test pop with default for a missing key
        assert bimap.pop("x", 99) == 99
        assert len(bimap) == 1

    @pytest.mark.parametrize(
        "action, key_or_value, expected_exception",
        [
            (lambda m, k: m[k], "z", KeyError),
            (lambda m, v: m.get_key(v), 99, KeyError),
            (lambda m, k: m.pop(k), "z", KeyError),
            (lambda m, k: m.delete(k), "z", KeyError),
        ],
        ids=["getitem_missing", "get_key_missing", "pop_missing", "delete_missing"],
    )
    def test_lookup_and_remove_failures(self, action, key_or_value, expected_exception):
        """Confirm that lookups and removals for non-existent items raise errors."""
        bimap = BiDirectionalMap({"a": 1})
        with pytest.raises(expected_exception):
            action(bimap, key_or_value)

    def test_update_method(self):
        """Ensure update() correctly merges data and respects uniqueness."""
        bimap = BiDirectionalMap({"a": 1, "b": 2})
        bimap.update({"b": 3, "c": 4})
        expected = {"a": 1, "b": 3, "c": 4}
        assert bimap == expected
        assert bimap.get_key(3) == "b"
        assert not bimap.has_value(2)
        with pytest.raises(ValueError, match=r"(?i)value already exists"):
            bimap.update({"d": 1})

    def test_conversion_methods(self):
        """Verify to_dict() and reversed() work as expected."""
        data = {"a": 1, "b": 2}
        bimap = BiDirectionalMap(data)
        # Test to_dict()
        as_dict = bimap.to_dict()
        assert as_dict == data
        assert isinstance(as_dict, dict)
        # Test reversed()
        rev_bimap = bimap.reversed()
        assert isinstance(rev_bimap, BiDirectionalMap)
        assert rev_bimap.to_dict() == {1: "a", 2: "b"}
        assert rev_bimap.get_key("a") == 1

    def test_equality(self):
        """Verify equality checks against other mappings."""
        data = {"a": 1, "b": 2}
        bimap1 = BiDirectionalMap(data)
        bimap2 = BiDirectionalMap(data)
        assert bimap1 == bimap2
        assert bimap1 == data
        assert bimap2 == data
        assert bimap1 != {"a": 1}
        assert bimap1 != "not a map"
        assert bimap1 != None

    def test_clear(self):
        """Ensure clear() removes all items."""
        bimap = BiDirectionalMap({"a": 1, "b": 2})
        bimap.clear()
        assert len(bimap) == 0
        assert not bimap.keys()
        assert not bimap.values()
        assert not bimap.has_value(1)
        assert bimap.to_dict() == {}


# ------------------------------------------
