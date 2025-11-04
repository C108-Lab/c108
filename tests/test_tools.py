#
# C108 - Tools Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
from collections import abc, ChainMap, UserDict
from dataclasses import dataclass, field
from typing import Any, Sequence

# Third-party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.tools import dict_get, dict_set, listify, sequence_get
from c108.tools import get_caller_name, as_ascii


# Classes --------------------------------------------------------------------------------------------------------------


class Obj:
    a = 0
    to_dict = {"a": "zero"}


@dataclass
class DataClass:
    a = 0  # !!! <-- without type this is a class attr but NOT a dataclass field
    b: int = 1
    c: int = field(default=2)
    d: Obj = field(default_factory=Obj)


# Tests ----------------------------------------------------------------------------------------------------------------


class TestAsAscii:
    """
    Test suite for the as_ascii function.
    """

    @pytest.mark.parametrize(
        "s, replacement, expected",
        [
            # --- String tests ---
            # Test 1: ASCII-only string should remain unchanged
            ("Hello, world!", None, "Hello, world!"),
            # Test 2: Unicode string with default replacement
            ("你好, world!", None, "__, world!"),
            # Test 3: Unicode string with custom replacement
            ("你好, world!", "?", "??, world!"),
            # Test 4: Empty string should return an empty string
            ("", None, ""),
            # --- Bytes tests ---
            # Test 5: ASCII-only bytes should remain unchanged
            (b"Hello", None, b"Hello"),
            # Test 6: UTF-8 bytes with default replacement
            (b"caf\xc3\xa9", None, b"caf__"),
            # --- Bytearray tests ---
            # Test 7: Bytearray with a custom replacement
            (bytearray(b"data\x80\x81"), b"?", bytearray(b"data??")),
        ],
        ids=[
            "str_ascii_only",
            "str_unicode_default_replace",
            "str_unicode_custom_replace",
            "str_empty",
            "bytes_ascii_only",
            "bytes_unicode_default_replace",
            "bytearray_unicode_custom_replace",
        ],
    )
    def test_successful_conversion(self, s, replacement, expected):
        """
        Tests successful conversion of str, bytes, and bytearray to ASCII.
        """
        if replacement is None:
            result = as_ascii(s)
        else:
            result = as_ascii(s, replacement=replacement)

        assert result == expected
        assert isinstance(result, type(expected))

    @pytest.mark.parametrize(
        "s, replacement, error, match",
        [
            # --- Invalid argument type tests ---
            # Test 8: Invalid input type (not str, bytes, or bytearray)
            (12345, None, TypeError, "Input must be str, bytes, or bytearray"),
            # Test 9: Mismatched replacement type for string input
            ("abc", b"_", TypeError, "Replacement for str input must be str"),
            # Test 10: Mismatched replacement type for bytes input
            (b"abc", "_", TypeError, "Replacement for bytes input must be bytes"),
            # --- Invalid replacement value tests ---
            # Test 11: Multi-character replacement for a string
            ("abc", "xy", ValueError, "Replacement must be a single character"),
            # Test 12: Multi-byte replacement for bytes
            (b"abc", b"xy", ValueError, "Replacement must be a single byte"),
            # Test 13: Non-ASCII replacement for a string
            ("abc", "é", ValueError, "Replacement character must be ASCII"),
            # Test 14: Non-ASCII replacement for bytes
            (b"abc", b"\x80", ValueError, "Replacement byte must be ASCII"),
        ],
        ids=[
            "invalid_input_type",
            "invalid_replace_type_for_str",
            "invalid_replace_type_for_bytes",
            "invalid_replace_length_for_str",
            "invalid_replace_length_for_bytes",
            "non_ascii_replace_for_str",
            "non_ascii_replace_for_bytes",
        ],
    )
    def test_invalid_inputs_and_replacements(self, s, replacement, error, match):
        """
        Tests that as_ascii raises appropriate errors for invalid inputs.
        """
        with pytest.raises(error, match=r"(?i)" + match):
            if replacement is None:
                as_ascii(s)
            else:
                as_ascii(s, replacement=replacement)


class TestDictGet:
    @pytest.mark.parametrize(
        "source,key,expected",
        [
            ({"user": {"profile": {"name": "John"}}}, "user.profile.name", "John"),
            ({"a": {"b": {"c": 1}}}, "a.b.c", 1),
            ({"a": {"b": {"c": None}}}, "a.b.c", None),
        ],
        ids=["nested-string", "int-leaf", "none-leaf"],
    )
    def test_dot_path(self, source, key, expected):
        """Get value using dot-separated path."""
        assert dict_get(source, key) == expected

    @pytest.mark.parametrize(
        "source,key,expected",
        [
            (
                {"user": {"profile": {"name": "John"}}},
                ["user", "profile", "name"],
                "John",
            ),
            ({"a": {"b": {"c": 2}}}, ("a", "b", "c"), 2),
        ],
        ids=["list-keys", "tuple-keys"],
    )
    def test_sequence_path(self, source, key, expected):
        """Get value using sequence of keys."""
        assert dict_get(source, key) == expected

    @pytest.mark.parametrize(
        "source,key,default,expected",
        [
            ({"a": {"b": 1}}, "a.c", "missing", "missing"),
            ({"a": {"b": 1}}, ["a", "c"], None, None),
        ],
        ids=["missing-string-path", "missing-seq-path"],
    )
    def test_returns_default(self, source, key, default, expected):
        """Return the provided default for missing paths."""
        assert dict_get(source, key, default) == expected

    @pytest.mark.parametrize(
        "source",
        [123, 3.14, "not-a-mapping", ["list"], object()],
        ids=["int", "float", "str", "list", "object"],
    )
    def test_invalid_source_type(self, source):
        """Raise TypeError for non-mapping source."""
        with pytest.raises(TypeError, match=r"(?i)source.*dict.*mapping"):
            dict_get(source, "a")

    @pytest.mark.parametrize(
        "key, expected",
        [("", 2), (" ", 3)],
        ids=["empty", "whitespace"],
    )
    def test_supports_empty_or_whitespace_key(self, key, expected):
        """Return correct value for empty or whitespace-only key."""
        data = {"a": 1, "": 2, " ": 3}
        result = dict_get(data, key)
        assert result == expected

    @pytest.mark.parametrize(
        "key",
        [[], ()],
        ids=["empty-list", "empty-tuple"],
    )
    def test_empty_sequence_key(self, key):
        """Raise ValueError for empty key sequence."""
        with pytest.raises(ValueError, match=r"(?i)key.*sequence.*cannot be empty"):
            dict_get({"a": 1}, key)

    @pytest.mark.parametrize(
        "key",
        [123, 3.14, b"bytes"],
        ids=["int", "float", "bytes"],
    )
    def test_invalid_key_type(self, key):
        """Raise TypeError for unsupported key types."""
        with pytest.raises(TypeError, match=r"(?i)key.*str.*sequence"):
            dict_get({"a": 1}, key)

    @pytest.mark.parametrize(
        "source,key,separator,expected",
        [
            ({"a": {"b": {"c": 1}}}, "a/b/c", "/", 1),
            ({"a": {"b": {"c": 2}}}, "a:b:c", ":", 2),
        ],
        ids=["slash-separator", "colon-separator"],
    )
    def test_custom_sep(self, source, key, separator, expected):
        """Use a custom separator for path strings."""
        assert dict_get(source, key, separator=separator) == expected

    def test_intermediate_non_mapping(self):
        """Return default when intermediate node is not a mapping."""
        source = {"a": 1}
        assert dict_get(source, "a.b", default="x") == "x"

    def test_mapping_subclass(self):
        """Support mapping subclasses as source."""

        class MyMapping(abc.Mapping):
            def __init__(self, backing):
                self._b = dict(backing)

            def __getitem__(self, k):
                return self._b[k]

            def __iter__(self):
                return iter(self._b)

            def __len__(self):
                return len(self._b)

        m = MyMapping({"a": MyMapping({"b": 42})})
        assert dict_get(m, "a.b") == 42

    def test_chainmap(self):
        """Work with ChainMap sources."""
        cm = ChainMap({"a": {"b": 5}}, {"a": {"b": 99}})
        # ChainMap exposes keys at top level; nested dict is regular dict
        assert dict_get(cm, "a.b") == 5

    def test_no_mutation(self):
        """Do not mutate the source mapping."""
        source = {"a": {"b": {"c": 1}}}
        before = repr(source)
        _ = dict_get(source, "a.b.c")
        assert repr(source) == before

    def test_missing_without_default(self):
        """Return None when missing and no default is provided."""
        assert dict_get({"x": {}}, "x.y") is None


class TestDictSet:
    @pytest.mark.parametrize(
        "initial,key,value,expected",
        [
            ({}, "user.profile.name", "John", {"user": {"profile": {"name": "John"}}}),
            ({}, ["user", "profile", "age"], 30, {"user": {"profile": {"age": 30}}}),
        ],
        ids=["dot-str", "seq"],
    )
    def test_set_creates(self, initial, key, value, expected):
        """Set value via dot-string or sequence, creating nested dicts."""
        dict_set(initial, key, value)
        assert initial == expected

    def test_overwrite(self):
        """Overwrite existing leaf value."""
        data = {"a": {"b": {"c": 1}}}
        dict_set(data, "a.b.c", 2)
        assert data["a"]["b"]["c"] == 2

    def test_separator(self):
        """Support custom key separator."""
        data = {}
        dict_set(data, "a:b:c", 42, separator=":")
        assert data == {"a": {"b": {"c": 42}}}

    def test_missing_raises(self):
        """Raise KeyError when path is missing and create_missing is false."""
        data = {"a": {}}
        with pytest.raises(
            KeyError, match=r"(?i)intermediate key.*create_missing=False"
        ):
            dict_set(data, "a.b.c", 1, create_missing=False)

    def test_non_mapping_raises(self):
        """Raise TypeError when traversing through non-mapping."""
        data = {"a": {"b": 123}}
        with pytest.raises(
            TypeError, match=r"(?i)cannot traverse through non-dict value"
        ):
            dict_set(data, "a.b.c", 1)

    @pytest.mark.parametrize(
        "bad_dest",
        [None, 123, 3.14, "not-a-dict", [1, 2, 3]],
        ids=["none", "int", "float", "str", "list"],
    )
    def test_bad_dest(self, bad_dest):
        """Reject non-mapping destination."""
        with pytest.raises(TypeError, match=r"(?i)dest must be dict or MutableMapping"):
            dict_set(bad_dest, "a.b", 1)

    @pytest.mark.parametrize(
        "bad_key",
        ["", "   "],
        ids=["empty", "blank"],
    )
    def test_empty_key_str(self, bad_key):
        """Reject empty or blank string key."""
        with pytest.raises(ValueError, match=r"(?i)key string cannot be empty"):
            dict_set({}, bad_key, 1)

    def test_empty_key_seq(self):
        """Reject empty key sequence."""
        with pytest.raises(ValueError, match=r"(?i)key sequence cannot be empty"):
            dict_set({}, [], 1)

    @pytest.mark.parametrize(
        "bad_key",
        [b"bytes", 123, 3.14, {"k": "v"}, {1, 2}],
        ids=["bytes", "int", "float", "dict", "set"],
    )
    def test_bad_key_type(self, bad_key):
        """Reject invalid key types."""
        with pytest.raises(TypeError, match=r"(?i)key must be str or sequence"):
            dict_set({}, bad_key, 1)

    def test_mutablemapping_dest(self):
        """Accept MutableMapping destination."""
        data = UserDict()
        dict_set(data, "x.y", "ok")
        assert isinstance(data, UserDict)
        assert data["x"]["y"] == "ok"

    def test_partial_path(self):
        """Handle mixed existing and new path segments."""
        data = {"root": {"leaf": 1}}
        dict_set(data, "root.branch.leaf", 2)
        assert data == {"root": {"leaf": 1, "branch": {"leaf": 2}}}


# TestGetCallerName Helper functions:
#   MODULE-level definitions required to ensure a predictable stack frame.
def _caller_for_depth_1_test():
    """Calls get_caller_name with the default depth of 1."""
    # The immediate caller is this helper function itself.
    return get_caller_name()


def _caller_for_depth_2_test():
    """A nested function to test a stack depth of 2."""

    def _inner_caller():
        # The caller at depth 1 is `_caller_for_depth_2_test`.
        # The caller at depth 2 is the function that called `_caller_for_depth_2_test`.
        return get_caller_name(depth=2)

    return _inner_caller()


class TestGetCallerName:
    """Test suite for the get_caller_name utility function."""

    def test_get_caller_name_at_default_depth_1(self):
        """Verify it correctly identifies the immediate caller (depth=1)."""
        # The name of the helper function that directly calls get_caller_name should be returned.
        assert _caller_for_depth_1_test() == "_caller_for_depth_1_test"

    def test_get_caller_name_at_depth_2(self):
        """Verify it correctly identifies the caller's caller (depth=2)."""
        # The caller at depth=2 is the intermediate helper function, not the test method.
        # Stack: get_caller_name <- _inner_caller <- _caller_for_depth_2_test <- [test_method]
        assert _caller_for_depth_2_test() == "_caller_for_depth_2_test"

    def test_get_caller_name_within_class_method(self):
        """Verify it returns the correct name when called directly inside a test method."""
        # The caller at depth=1 is this test method.
        assert get_caller_name() == "test_get_caller_name_within_class_method"

    def test_get_caller_name_with_excessive_depth(self):
        """Verify it raises IndexError for a depth exceeding the stack size."""
        with pytest.raises(IndexError, match=r"(?i)" + "call stack is not deep enough"):
            # Use a sufficiently large number that is guaranteed to be out of bounds.
            get_caller_name(depth=100)

    @pytest.mark.parametrize(
        ("invalid_depth", "expected_message"),
        [
            (0, "must be 1 or greater"),
            (-1, "must be 1 or greater"),
            (-100, "must be 1 or greater"),
        ],
        ids=["depth_zero", "depth_negative_one", "depth_large_negative"],
    )
    def test_get_caller_name_with_invalid_value(self, invalid_depth, expected_message):
        """Verify it raises ValueError for depths less than 1."""
        with pytest.raises(ValueError, match=r"(?i)" + expected_message):
            get_caller_name(depth=invalid_depth)

    @pytest.mark.parametrize(
        ("invalid_type", "expected_message"),
        [
            ("2", "must be an integer"),
            (1.5, "must be an integer"),
            (None, "must be an integer"),
        ],
        ids=["string_type", "float_type", "none_type"],
    )
    def test_get_caller_name_with_invalid_type(self, invalid_type, expected_message):
        """Verify it raises TypeError for non-integer depth arguments."""
        with pytest.raises(TypeError, match=r"(?i)" + expected_message):
            get_caller_name(depth=invalid_type)


class TestListify:
    @pytest.mark.parametrize(
        "value",
        ["abc", b"bytes", bytearray(b"buf")],
        ids=["str", "bytes", "bytearray"],
    )
    def test_atomic_text_bytes(self, value):
        """Wrap text and bytes as a single element."""
        out = listify(value)
        assert out == [value], "Text/bytes must be wrapped as a single element"

    @pytest.mark.parametrize(
        "value, expected",
        [
            ([1, 2, "3"], [1, 2, "3"]),
            ((1, "2"), [1, "2"]),
        ],
        ids=["list", "tuple"],
    )
    def test_iterables_expand(self, value, expected):
        """Expand iterables into a list."""
        assert listify(value) == expected

    def test_generator_memoryview_expand(self):
        """Expand generators and memoryviews into lists."""
        gen = (i for i in (1, 2, 3))
        assert listify(gen) == [1, 2, 3]

        mv = memoryview(b"ab")
        assert listify(mv) == [97, 98]  # ord('a') == 97, ord('b') == 98

    @pytest.mark.parametrize(
        "value",
        [42, 3.14, object()],
        ids=["int", "float", "object"],
    )
    def test_non_iterables_wrapped(self, value):
        """Wrap non-iterables as a single element."""
        out = listify(value)
        assert len(out) == 1 and out[0] is value

    @pytest.mark.parametrize(
        "value, as_type, expected",
        [
            ((1, "2"), str, ["1", "2"]),
            (["1", "2", "3"], int, [1, 2, 3]),
            (
                "123",
                int,
                [123],
            ),  # str is atomic, conversion applies to the single wrapped value
            ((1.2, 3.4), lambda x: round(float(x)), [1, 3]),
        ],
        ids=["tuple->str", "list->int", "str->int", "float-tuple->round"],
    )
    def test_as_type_conversion(self, value, as_type, expected):
        """Convert each item to the given type."""
        assert listify(value, as_type=as_type) == expected

    def test_conversion_failure(self):
        """Raise ValueError with chained cause on conversion failure."""
        with pytest.raises(ValueError, match=r"(?i)conversion.*failed") as excinfo:
            listify(["1", "x", "3"], as_type=int)
        assert excinfo.value.__cause__ is not None

    def test_mapping_default_items(self):
        """Return items tuples by default for mappings."""
        d = {"a": 1, "b": 2}
        assert listify(d) == [("a", 1), ("b", 2)]

    @pytest.mark.parametrize(
        "mode, expected",
        [
            ("keys", ["a", "b"]),
            ("values", [1, 2]),
            ("items", [("a", 1), ("b", 2)]),
            ("atomic", [{"a": 1, "b": 2}]),
        ],
        ids=["keys", "values", "items", "atomic"],
    )
    def test_mapping_modes(self, mode, expected):
        """Honor mapping_mode for mappings."""
        d = {"a": 1, "b": 2}
        assert listify(d, mapping_mode=mode) == expected

    def test_items_conversion_callable(self):
        """Convert items tuples using callable."""
        d = {"a": 1, "b": 2}
        out = listify(d, as_type=lambda kv: (kv[0].upper(), kv[1] * 10))
        assert out == [("A", 10), ("B", 20)]

    def test_invalid_mapping_mode(self):
        """Raise ValueError on invalid mapping_mode."""
        with pytest.raises(ValueError, match=r"(?i)invalid.*mapping_mode"):
            listify({"a": 1}, mapping_mode="nope")

    def test_empty_and_edge_cases(self):
        """Handle empty iterables and bytearray edge cases."""
        assert listify([]) == []
        assert listify(()) == []
        ba = bytearray(b"xy")
        assert listify(ba) == [ba]


class TestSequenceGet:
    """Test suite for the sequence_get function."""

    TEST_SEQUENCE = [10, 20, 30, 40]

    @pytest.mark.parametrize(
        ("seq", "index", "default", "expected"),
        [
            (TEST_SEQUENCE, 0, None, 10),
            (TEST_SEQUENCE, 2, None, 30),
            (TEST_SEQUENCE, -1, None, 40),
            (("a", "b"), 1, None, "b"),
            ("hello", 4, None, "o"),
        ],
        ids=[
            "get_first_element",
            "get_middle_element",
            "get_last_element_negative_index",
            "get_from_tuple",
            "get_from_string",
        ],
    )
    def test_successful_retrieval(
        self, seq: Sequence, index: int, default: Any, expected: Any
    ):
        """Verify that items are correctly retrieved with valid inputs."""
        assert sequence_get(seq, index, default) == expected

    @pytest.mark.parametrize(
        ("seq", "index", "default", "expected"),
        [
            (TEST_SEQUENCE, 99, "not_found", "not_found"),
            (TEST_SEQUENCE, -99, "not_found", "not_found"),
            ([], 0, "empty", "empty"),
            (None, 1, "none_seq", "none_seq"),
            (TEST_SEQUENCE, None, "none_idx", "none_idx"),
            (TEST_SEQUENCE, 5, None, None),
        ],
        ids=[
            "index_out_of_bounds_positive",
            "index_out_of_bounds_negative",
            "empty_sequence",
            "none_sequence",
            "none_index",
            "default_is_none",
        ],
    )
    def test_default_value_scenarios(
        self, seq: Sequence | None, index: int | None, default: Any, expected: Any
    ):
        """Verify that the default value is returned when retrieval is not possible."""
        assert sequence_get(seq, index, default) == expected

    @pytest.mark.parametrize(
        "invalid_seq",
        [
            {1: "a", 2: "b"},
            {1, 2, 3},
            12345,
        ],
        ids=["dictionary_input", "set_input", "integer_input"],
    )
    def test_raises_on_invalid_sequence_type(self, invalid_seq: Any):
        """Verify it raises TypeError for inputs that are not Sequences."""
        with pytest.raises(TypeError, match=r"(?i)" + "expected Sequence or None"):
            sequence_get(invalid_seq, 0)

    @pytest.mark.parametrize(
        "invalid_index",
        [
            "1",
            1.0,
            [0],
        ],
        ids=["string_index", "float_index", "list_index"],
    )
    def test_raises_on_invalid_index_type(self, invalid_index: Any):
        """Verify it raises TypeError for index inputs that are not integers."""
        with pytest.raises(TypeError, match=r"(?i)" + "expected int or None for index"):
            sequence_get(self.TEST_SEQUENCE, invalid_index)
