#
# C108 - Collection Tests
#

# Third-party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.collections import BiDirectionalMap


# Tests ----------------------------------------------------------------------------------------------------------------
class TestBiDirectionalMap:

    @pytest.fixture
    def populated_map(self):
        """Fixture providing a pre-populated BiDirectionalMap instance."""
        return BiDirectionalMap({
            1: "apple",
            2: "banana",
            3: "cherry",
        })

    def test_lookup(self, populated_map):
        assert populated_map[3] == "cherry"
        assert populated_map.get_value(1) == "apple"
        assert populated_map.get_key("banana") == 2

    def test_value_uniqueness(self, populated_map):
        """Tests that adding a duplicate value raises ValueError."""
        with pytest.raises(ValueError, match="Value 'apple' already exists"):
            populated_map.add(4, "apple")  # Try to add an existing value

    def test_key_uniqueness(self, populated_map):
        """Tests that adding a duplicate key raises ValueError."""
        with pytest.raises(ValueError, match="Key '1' already exists"):
            populated_map.add(1, "grape")  # Try to add an existing key

    def test_contains(self, populated_map):
        assert 1 in populated_map  # Checks key
        assert "apple" in populated_map  # Checks value
        assert 99 not in populated_map
        assert "zebra" not in populated_map

    def test_keys_values_items(self, populated_map):
        assert sorted(list(populated_map.keys())) == [1, 2, 3]
        assert sorted(list(populated_map.values())) == ["apple", "banana", "cherry"]
        assert set(populated_map.items()) == {(1, "apple"), (2, "banana"), (3, "cherry")}
