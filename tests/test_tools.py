#
# C108 - Tools Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import re
from collections import abc, ChainMap, UserDict
from dataclasses import dataclass, field
from typing import Any, Sequence

# Third-party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.cli import cli_multiline, clify
from c108.pack import is_numbered_version, is_pep440_version, is_semantic_version

from c108.tools import fmt_any, fmt_exception, fmt_mapping, fmt_sequence, fmt_type, fmt_value
from c108.tools import dict_get, dict_set, list_get, listify, sequence_get
from c108.tools import get_caller_name, print_title, to_ascii


# Classes --------------------------------------------------------------------------------------------------------------

class AnyUserClass:
    """A simple class for testing user-defined types"""
    pass


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
    def test_get_with_dot_path(self, source, key, expected):
        assert dict_get(source, key) == expected

    @pytest.mark.parametrize(
        "source,key,expected",
        [
            ({"user": {"profile": {"name": "John"}}}, ["user", "profile", "name"], "John"),
            ({"a": {"b": {"c": 2}}}, ("a", "b", "c"), 2),
        ],
        ids=["list-keys", "tuple-keys"],
    )
    def test_get_with_sequence_path(self, source, key, expected):
        assert dict_get(source, key) == expected

    @pytest.mark.parametrize(
        "source,key,default,expected",
        [
            ({"a": {"b": 1}}, "a.c", "missing", "missing"),
            ({"a": {"b": 1}}, ["a", "c"], None, None),
        ],
        ids=["missing-string-path", "missing-seq-path"],
    )
    def test_missing_returns_default(self, source, key, default, expected):
        assert dict_get(source, key, default) == expected

    @pytest.mark.parametrize(
        "source",
        [123, 3.14, "not-a-mapping", ["list"], object()],
        ids=["int", "float", "str", "list", "object"],
    )
    def test_non_mapping_source_raises_typeerror(self, source):
        with pytest.raises(TypeError, match=r"(?i)source.*dict.*mapping"):
            dict_get(source, "a")

    @pytest.mark.parametrize(
        "key", ["", "   "],
        ids=["empty", "whitespace"],
    )
    def test_empty_key_string_raises_valueerror(self, key):
        with pytest.raises(ValueError, match=r"(?i)key.*cannot be empty"):
            dict_get({"a": 1}, key)

    @pytest.mark.parametrize(
        "key", [[], ()],
        ids=["empty-list", "empty-tuple"],
    )
    def test_empty_key_sequence_raises_valueerror(self, key):
        with pytest.raises(ValueError, match=r"(?i)key.*sequence.*cannot be empty"):
            dict_get({"a": 1}, key)

    @pytest.mark.parametrize(
        "key", [123, 3.14, b"bytes"],
        ids=["int", "float", "bytes"],
    )
    def test_key_wrong_type_raises_typeerror(self, key):
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
    def test_custom_separator(self, source, key, separator, expected):
        assert dict_get(source, key, separator=separator) == expected

    def test_intermediate_non_mapping_returns_default(self):
        source = {"a": 1}
        assert dict_get(source, "a.b", default="x") == "x"

    def test_accepts_mapping_subclass(self):
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

    def test_works_with_chainmap(self):
        cm = ChainMap({"a": {"b": 5}}, {"a": {"b": 99}})
        # ChainMap exposes keys at top level; nested dict is regular dict
        assert dict_get(cm, "a.b") == 5

    def test_source_not_mutated(self):
        source = {"a": {"b": {"c": 1}}}
        before = repr(source)
        _ = dict_get(source, "a.b.c")
        assert repr(source) == before

    def test_missing_key_without_default_returns_none(self):
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
        with pytest.raises(KeyError, match=r"(?i)intermediate key.*create_missing=False"):
            dict_set(data, "a.b.c", 1, create_missing=False)

    def test_non_mapping_raises(self):
        """Raise TypeError when traversing through non-mapping."""
        data = {"a": {"b": 123}}
        with pytest.raises(TypeError, match=r"(?i)cannot traverse through non-dict value"):
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


class TestFmtAny:
    @pytest.mark.parametrize(
        "obj,expected_substring",
        [
            # Exception dispatch
            (ValueError("test error"), "ValueError"),
            (ValueError("test error"), "test error"),
            (RuntimeError(), "RuntimeError"),

            # Mapping dispatch
            ({"key": "value"}, "key"),
            ({"key": "value"}, "value"),
            ({}, "{}"),

            # Sequence dispatch (non-textual)
            ([1, 2, 3], "int: 1"),
            ([1, 2, 3], "int: 2"),
            ([], "[]"),
            ((1, 2), "int: 1"),

            # Value dispatch (including textual sequences)
            ("hello", "str: 'hello'"),
            (42, "int: 42"),
            (3.14, "float: 3.14"),
            (True, "bool: True"),
        ],
    )
    def test_dispatch_to_correct_formatter(self, obj, expected_substring):
        """Test that fmt_any dispatches to the correct formatter based on object type."""
        result = fmt_any(obj)
        assert expected_substring in result

    @pytest.mark.parametrize("style", ["ascii", "unicode-angle"])
    def test_style_parameter_forwarding(self, style):
        """Test that style parameter is properly forwarded to underlying formatters."""
        # Test with different object types
        exc_result = fmt_any(ValueError("test"), style=style)
        dict_result = fmt_any({"key": "val"}, style=style)
        list_result = fmt_any([1, 2], style=style)
        value_result = fmt_any("text", style=style)

        # All should contain the expected content
        assert "ValueError" in exc_result and "test" in exc_result
        assert "key" in dict_result and "val" in dict_result
        assert "int: 1" in list_result
        assert "str" in value_result and "text" in value_result

    def test_exception_with_traceback_parameter(self):
        """Test that include_traceback parameter works for exceptions."""
        try:
            raise ValueError("traceback test")
        except ValueError as e:
            result_without = fmt_any(e, include_traceback=False)
            result_with = fmt_any(e, include_traceback=True)

            # Both should contain basic exception info
            assert "ValueError" in result_without
            assert "traceback test" in result_without
            assert "ValueError" in result_with
            assert "traceback test" in result_with

            # Only the traceback version should have location info
            # (exact format may vary, so check for common traceback indicators)
            has_location_info = any(indicator in result_with.lower()
                                    for indicator in ["test_fmt_any", "line", "at "])
            assert has_location_info

    @pytest.mark.parametrize("max_items", [1, 3, 5])
    def test_max_items_parameter_forwarding(self, max_items):
        """Test that max_items parameter is forwarded to collection formatters."""
        large_dict = {f"key{i}": f"val{i}" for i in range(10)}
        large_list = list(range(10))

        dict_result = fmt_any(large_dict, max_items=max_items)
        list_result = fmt_any(large_list, max_items=max_items)

        # Should contain truncation indicator if max_items < 10
        if max_items < 10:
            assert "..." in dict_result or "…" in dict_result
            assert "..." in list_result or "…" in list_result

    @pytest.mark.parametrize("max_repr", [10, 20, 50])
    def test_max_repr_parameter_forwarding(self, max_repr):
        """Test that max_repr parameter is forwarded to all formatters."""
        long_message = "x" * 100

        exc_result = fmt_any(ValueError(long_message), max_repr=max_repr)
        dict_result = fmt_any({"key": long_message}, max_repr=max_repr)
        list_result = fmt_any([long_message], max_repr=max_repr)
        value_result = fmt_any(long_message, max_repr=max_repr)

        # All results should be reasonably bounded
        assert len(exc_result) <= max_repr + 50  # Allow some overhead for formatting
        assert len(dict_result) <= max_repr + 100  # More overhead for structure
        assert len(list_result) <= max_repr + 100
        assert len(value_result) <= max_repr + 50

    def test_nested_structures_with_depth(self):
        """Test that nested structures are handled correctly with depth parameter."""
        nested = {"outer": {"inner": [1, 2, {"deep": "value"}]}}

        shallow_result = fmt_any(nested, depth=1)
        deep_result = fmt_any(nested, depth=3)

        # Both should contain outer structure
        assert "outer" in shallow_result
        assert "outer" in deep_result

        # Deep result should show more detail
        assert "inner" in deep_result
        assert "deep" in deep_result

    def test_textual_sequences_treated_as_values(self):
        """Test that textual sequences (str, bytes) are treated as atomic values."""
        text_str = "hello world"
        text_bytes = b"hello world"
        text_bytearray = bytearray(b"hello world")

        str_result = fmt_any(text_str)
        bytes_result = fmt_any(text_bytes)
        bytearray_result = fmt_any(text_bytearray)

        # Should be formatted as single values, not character sequences
        assert "str: 'hello world'" in str_result
        assert "bytes:" in bytes_result
        assert "bytearray:" in bytearray_result

        # Should NOT contain individual character formatting
        assert "str: 'h'" not in str_result

    def test_custom_ellipsis_parameter(self):
        """Test that custom ellipsis parameter is forwarded correctly."""
        large_dict = {f"k{i}": f"v{i}" for i in range(10)}
        long_string = "x" * 100

        dict_result = fmt_any(large_dict, max_items=2, ellipsis="[MORE]")
        str_result = fmt_any(long_string, max_repr=10, ellipsis="[MORE]")

        # Should use custom ellipsis
        assert "[MORE]" in dict_result
        assert "[MORE]" in str_result

    def test_edge_cases_and_special_objects(self):
        """Test edge cases and special object types."""
        # None
        none_result = fmt_any(None)
        assert "NoneType" in none_result

        # Empty collections
        empty_dict_result = fmt_any({})
        empty_list_result = fmt_any([])
        empty_tuple_result = fmt_any(())

        assert "{}" in empty_dict_result
        assert "[]" in empty_list_result
        assert "()" in empty_tuple_result

        # Complex nested empty structure
        complex_empty = {"empty_list": [], "empty_dict": {}}
        complex_result = fmt_any(complex_empty)
        assert "empty_list" in complex_result
        assert "empty_dict" in complex_result


class TestFmtException:
    @pytest.mark.parametrize(
        "exc,expected",
        [
            (ValueError("invalid input"), "<ValueError: invalid input>"),
            (RuntimeError(), "<RuntimeError>"),
        ],
    )
    def test_basic_and_empty_messages(self, exc, expected):
        result = fmt_exception(exc)
        assert result == expected

    @pytest.mark.parametrize(
        "exc,expected",
        [
            (ValueError("bad value"), "<ValueError: bad value>"),
            (TypeError("wrong type"), "<TypeError: wrong type>"),
            (RuntimeError("boom"), "<RuntimeError: boom>"),
        ],
    )
    def test_different_exception_types(self, exc, expected):
        assert fmt_exception(exc) == expected

    def test_unicode_message(self):
        exc = ValueError("Error with unicode: 🚨 αβγ")
        assert fmt_exception(exc) == "<ValueError: Error with unicode: 🚨 αβγ>"

    @pytest.mark.parametrize(
        "msg,max_repr,ellipsis,starts_with,ends_with",
        [
            # Default ellipsis "..." with truncation
            ("very " * 50 + "long message", 30, None, "<ValueError: very very", "...>"),
            # Custom ellipsis token
            ("x" * 100, 20, "[...]", "<ValueError: ", "[...]>"),
        ],
    )
    def test_truncation_and_custom_ellipsis(
            self, msg, max_repr, ellipsis, starts_with, ends_with
    ):
        try:
            raise ValueError(msg)
        except ValueError as e:
            out = fmt_exception(e, max_repr=max_repr, ellipsis=ellipsis)
            assert out.startswith(starts_with)
            assert out.endswith(ends_with)
            # Ensure truncation actually happened (shorter than message + overhead)
            assert len(out) < len(f"<ValueError: {msg}>")

    @pytest.mark.parametrize("style", ["ascii", "unicode-angle", "equal"])
    def test_style_parameter_behavior(self, style):
        # Style actually affects formatting, so test each style individually
        exc = ValueError("test message")
        out = fmt_exception(exc, style=style)

        # All styles should contain the type and message
        assert "ValueError" in out
        assert "test message" in out

        # Check style-specific formatting
        if style == "ascii":
            assert out == "<ValueError: test message>"
        elif style == "unicode-angle":
            assert out == "⟨ValueError: test message⟩"
        elif style == "equal":
            assert out == "ValueError=test message"

    @pytest.mark.parametrize("max_repr", [0, 1, 5])
    def test_max_repr_edge_cases(self, max_repr):
        exc = ValueError("short")
        out = fmt_exception(exc, max_repr=max_repr)
        # Always returns something sane with a proper wrapper and type name
        assert "ValueError" in out
        # Should not be excessively long for tiny limits
        assert len(out) <= 40

    def test_include_traceback_location_true(self):
        def _raise_here():
            raise ValueError("with tb")

        try:
            _raise_here()
        except ValueError as e:
            out = fmt_exception(e, include_traceback=True)
            # Fixed: expect the actual format with location info embedded
            assert out.startswith("<ValueError: with tb")
            assert " at " in out
            assert "_raise_here" in out
            # Should end with line number
            assert re.search(r":\d+>$", out)

    def test_include_traceback_location_false(self):
        def _raise_here():
            raise ValueError("no tb")

        try:
            _raise_here()
        except ValueError as e:
            out = fmt_exception(e, include_traceback=False)
            assert out == "<ValueError: no tb>"

    def test_broken_str_on_exception(self):
        class BrokenStrError(Exception):
            def __str__(self):
                raise RuntimeError("boom")

        exc = BrokenStrError("test message")
        # Should not raise; should fall back gracefully to the exception type
        out = fmt_exception(exc)
        assert out.startswith("<BrokenStrError")
        assert out.endswith(">")


class TestFmtMapping:
    # ---------- Basic functionality ----------

    def test_fmt_mapping_basic(self):
        mp = {"a": 1, 2: "b"}
        out = fmt_mapping(mp, style="ascii")
        # Insertion order preserved by dicts
        assert out == "{<str: 'a'>: <int: 1>, <int: 2>: <str: 'b'>}"

    def test_fmt_mapping_with_nested_sequence(self):
        mp = {"k": [1, 2]}
        out = fmt_mapping(mp, style="unicode-angle")
        assert out == "{⟨str: 'k'⟩: [⟨int: 1⟩, ⟨int: 2⟩]}"

    # ---------- Edge cases critical for exceptions/logging ----------

    def test_fmt_mapping_empty_dict(self):
        """Empty dicts are common in validation errors"""
        assert fmt_mapping({}) == "{}"

    def test_fmt_mapping_none_keys_and_values(self):
        """None keys/values are common edge cases"""
        mp = {None: "value", "key": None, None: None}
        out = fmt_mapping(mp, style="ascii")
        assert "<NoneType: None>" in out
        assert "value" in out or "key" in out

    def test_fmt_mapping_complex_key_types(self):
        """Non-string keys are common and can be problematic"""
        mp = {
            42: "int key",
            (1, 2): "tuple key",
            frozenset([3, 4]): "frozenset key",
            True: "bool key"
        }
        out = fmt_mapping(mp, style="ascii")
        assert "<int: 42>" in out
        assert "<tuple:" in out
        assert "<frozenset:" in out
        assert "<bool: True>" in out

    def test_fmt_mapping_broken_key_repr(self):
        """Keys with broken __repr__ should not crash formatting"""

        class BrokenKeyRepr:
            def __repr__(self):
                raise ValueError("Key repr is broken!")

            def __hash__(self):
                return hash("broken")

            def __eq__(self, other):
                return isinstance(other, BrokenKeyRepr)

        mp = {BrokenKeyRepr(): "value"}
        out = fmt_mapping(mp, style="ascii")
        # Should handle gracefully
        assert "BrokenKeyRepr" in out
        assert "repr failed" in out
        assert "value" in out

    def test_fmt_mapping_broken_value_repr(self):
        """Values with broken __repr__ should not crash formatting"""

        class BrokenValueRepr:
            def __repr__(self):
                raise RuntimeError("Value repr is broken!")

        mp = {"key": BrokenValueRepr()}
        out = fmt_mapping(mp, style="ascii")
        assert "key" in out
        assert "BrokenValueRepr" in out
        assert "repr failed" in out

    def test_fmt_mapping_very_large_dict(self):
        """Large dicts should be truncated appropriately"""
        big_dict = {f"key_{i}": f"value_{i}" for i in range(20)}
        out = fmt_mapping(big_dict, style="ascii", max_items=3)
        # Should only show 3 items plus ellipsis
        key_count = out.count("<str: 'key_")
        assert key_count == 3
        assert "..." in out

    def test_fmt_mapping_deeply_nested_structures(self):
        """Nested mappings and sequences should be handled with depth control"""
        nested = {
            "level1": {
                "level2": {
                    "level3": [1, 2, {"level4": "deep"}]
                }
            }
        }

        # With depth=2, should recurse into level2 but treat level3+ as atomic
        out = fmt_mapping(nested, style="ascii", depth=2)
        assert "level1" in out
        assert "level2" in out
        # level3 list should be formatted as atomic
        assert "<list:" in out

    def test_fmt_mapping_circular_references(self):
        """Circular references should not cause infinite recursion"""
        d = {"a": 1}
        d["self"] = d  # Create circular reference

        out = fmt_mapping(d, style="ascii")
        # Should handle gracefully without infinite recursion
        assert "a" in out
        assert "self" in out
        assert "..." in out or "{" in out  # Circular part shown somehow

    # ---------- Truncation robustness ----------

    @pytest.mark.parametrize(
        "style, expected_more",
        [
            ("ascii", "..."),
            ("unicode-angle", "…"),
        ],
    )
    def test_fmt_mapping_max_items_appends_ellipsis(self, style, expected_more):
        mp = {i: i for i in range(5)}
        out = fmt_mapping(mp, style=style, max_items=3)
        assert out.endswith(expected_more + "}")

    def test_fmt_mapping_custom_ellipsis(self):
        mp = {i: i for i in range(4)}
        out = fmt_mapping(mp, style="ascii", max_items=2, ellipsis="~more~")
        assert out.endswith("~more~}")

    def test_fmt_mapping_extreme_max_items_limits(self):
        """Edge cases for max_items limits"""
        mp = {"a": 1, "b": 2}

        # Zero items - should show ellipsis only
        out = fmt_mapping(mp, max_items=0)
        assert out == "{...}" or out == "{…}"

        # One item
        out = fmt_mapping(mp, max_items=1)
        item_count = out.count("<")
        assert item_count >= 2  # At least one key and one value

    # ---------- Special mapping types ----------

    def test_fmt_mapping_ordered_dict(self):
        """OrderedDict should preserve order"""
        from collections import OrderedDict
        od = OrderedDict([("first", 1), ("second", 2)])
        out = fmt_mapping(od, style="ascii")
        # Should show first before second
        first_pos = out.find("first")
        second_pos = out.find("second")
        assert first_pos < second_pos

    def test_fmt_mapping_defaultdict(self):
        """defaultdict should format like regular dict"""
        from collections import defaultdict
        dd = defaultdict(list)
        dd["key"] = [1, 2, 3]
        out = fmt_mapping(dd, style="ascii")
        assert "key" in out
        assert "[<int: 1>" in out or "<list:" in out

    def test_fmt_mapping_textual_values_are_atomic(self):
        """Text-like values should not be decomposed into characters"""
        mp = {"s": "xyz", "b": b"ab", "ba": bytearray(b"test")}
        out = fmt_mapping(mp, style="paren")
        assert "str('xyz')" in out
        assert "bytes(b'ab')" in out
        assert "bytearray(" in out

    # ---------- Parameter validation (defensive) ----------

    def test_fmt_mapping_invalid_mapping_type(self):
        """Should handle non-mapping gracefully or raise clear error"""
        try:
            out = fmt_mapping("not a mapping", style="ascii")  # type: ignore
            # If it doesn't raise, should produce some reasonable output
            assert "str" in out or "not a mapping" in out
        except (TypeError, AttributeError) as e:
            # Acceptable to raise clear error for invalid input
            assert "mapping" in str(e).lower() or "items" in str(e).lower()

    def test_fmt_mapping_negative_max_items(self):
        """Negative max_items should not crash"""
        mp = {"a": 1}
        out = fmt_mapping(mp, max_items=-1)
        # Should handle gracefully
        assert "{" in out and "}" in out

    def test_fmt_mapping_huge_individual_values(self):
        """Individual values that are very long should be truncated"""
        huge_value = "x" * 1000
        mp = {"key": huge_value}
        out = fmt_mapping(mp, style="ascii", max_repr=20)
        # Value should be truncated
        assert len(out) < 200  # Much shorter than the huge value
        assert "..." in out or "…" in out


class TestFmtSequence:
    # ---------- Basic functionality ----------

    @pytest.mark.parametrize(
        "seq, style, expected",
        [
            ([1, "a"], "ascii", "<int: 1>, <str: 'a'>"),
            ((1, "a"), "ascii", "<int: 1>, <str: 'a'>"),
        ],
    )
    def test_fmt_sequence_delimiters_list_vs_tuple(self, seq, style, expected):
        out = fmt_sequence(seq, style=style)
        if isinstance(seq, list):
            assert out == f"[{expected}]"
        else:
            assert out == f"({expected})"

    def test_fmt_sequence_singleton_tuple_trailing_comma(self):
        """Singleton tuples must show trailing comma for Python accuracy"""
        out = fmt_sequence((1,), style="ascii")
        assert out == "(<int: 1>,)"

    # ---------- Edge cases critical for exceptions/logging ----------

    def test_fmt_sequence_empty_containers(self):
        """Empty containers are common in validation errors"""
        assert fmt_sequence([]) == "[]"
        assert fmt_sequence(()) == "()"
        assert fmt_sequence(set()) == "{}"

    def test_fmt_sequence_none_elements(self):
        """None elements are common edge cases"""
        seq = [1, None, "hello", None]
        out = fmt_sequence(seq, style="ascii")
        assert "<int: 1>" in out
        assert "<NoneType: None>" in out
        assert "<str: 'hello'>" in out

    def test_fmt_sequence_non_iterable_fallback(self):
        """Non-iterables should be handled gracefully via fmt_value fallback"""
        out = fmt_sequence(42, style="ascii")  # type: ignore
        # Should fall back to fmt_value behavior for non-iterables
        assert out == "<int: 42>"

    def test_fmt_sequence_mixed_types_realistic(self):
        """Real-world sequences often contain mixed types"""
        mixed = [42, "status", None, {"error": True}, [1, 2]]
        out = fmt_sequence(mixed, style="ascii")
        assert "<int: 42>" in out
        assert "<str: 'status'>" in out
        assert "<NoneType: None>" in out
        assert "{<str: 'error'>:" in out  # nested dict
        assert "[<int: 1>" in out  # nested list

    def test_fmt_sequence_broken_element_repr(self):
        """Elements with broken __repr__ should not crash formatting"""

        class BrokenRepr:
            def __repr__(self):
                raise RuntimeError("Element repr is broken!")

        seq = [1, BrokenRepr(), "after"]
        out = fmt_sequence(seq, style="ascii")
        assert "<int: 1>" in out
        assert "BrokenRepr" in out
        assert "repr failed" in out
        assert "<str: 'after'>" in out

    def test_fmt_sequence_very_large_list(self):
        """Large sequences should be truncated appropriately"""
        big_list = list(range(50))
        out = fmt_sequence(big_list, style="ascii", max_items=3)
        # Should only show 3 items plus ellipsis
        item_count = out.count("<int:")
        assert item_count == 3
        assert "..." in out

    def test_fmt_sequence_deeply_nested_structures(self):
        """Nested sequences should be handled with depth control"""
        nested = [1, [2, [3, [4, [5]]]]]

        # With depth=2, should recurse 2 levels but treat deeper as atomic
        out = fmt_sequence(nested, style="ascii", depth=2)
        assert "<int: 1>" in out
        assert "[<int: 2>" in out  # First level of nesting
        assert "[<int: 3>" in out  # Second level of nesting
        # Deeper nesting should be atomic
        assert "<list:" in out

    def test_fmt_sequence_circular_references(self):
        """Circular references should not cause infinite recursion"""
        lst = [1, 2]
        lst.append(lst)  # Create circular reference: [1, 2, [...]]

        out = fmt_sequence(lst, style="ascii")
        # Should handle gracefully without infinite recursion
        assert "<int: 1>" in out
        assert "<int: 2>" in out
        assert "..." in out or "[" in out  # Circular part shown somehow

    def test_fmt_sequence_generators_and_iterators(self):
        """Generators and iterators should be consumable once"""

        def gen():
            yield 1
            yield 2
            yield 3

        out = fmt_sequence(gen(), style="ascii", max_items=2)
        # Should consume generator and show first 2 items
        assert "<int: 1>" in out
        assert "<int: 2>" in out
        assert "..." in out

    def test_fmt_sequence_sets_unordered(self):
        """Sets should format reasonably despite being unordered"""
        s = {3, 1, 2}
        out = fmt_sequence(s, style="ascii")
        assert out.startswith("{")
        assert out.endswith("}")
        # Should contain all elements (order may vary)
        assert "<int: 1>" in out
        assert "<int: 2>" in out
        assert "<int: 3>" in out

    # ---------- String/textual handling ----------

    def test_fmt_sequence_string_is_atomic(self):
        """Strings should be treated as atomic, not character sequences"""
        out = fmt_sequence("abc", style="colon")
        assert out == "str: 'abc'"
        # Should NOT be ['a', 'b', 'c']

    def test_fmt_sequence_bytes_is_atomic(self):
        """bytes should be treated as atomic"""
        out = fmt_sequence(b"hello", style="ascii")
        assert out == "<bytes: b'hello'>"

    def test_fmt_sequence_bytearray_is_atomic(self):
        """bytearray should be treated as atomic"""
        ba = bytearray(b"test")
        out = fmt_sequence(ba, style="ascii")
        assert out.startswith("<bytearray:")

    def test_fmt_sequence_unicode_strings(self):
        """Unicode strings should be handled safely"""
        unicode_seq = ["Hello", "世界", "🌍"]
        out = fmt_sequence(unicode_seq, style="ascii")
        assert "Hello" in out
        # Unicode should be preserved or safely escaped
        assert "世界" in out or "\\u" in out
        assert "🌍" in out or "\\u" in out

    # ---------- Truncation robustness ----------

    def test_fmt_sequence_nesting_depth_1(self):
        seq = [1, [2, 3]]
        out = fmt_sequence(seq, style="unicode-angle", depth=1)
        assert out == "[⟨int: 1⟩, [⟨int: 2⟩, ⟨int: 3⟩]]"

    def test_fmt_sequence_nesting_depth_0_treats_inner_as_atomic(self):
        seq = [1, [2, 3]]
        out = fmt_sequence(seq, style="paren", depth=0)
        # Inner list is formatted as a single value by fmt_value
        assert out == "[int(1), list([2, 3])]"

    @pytest.mark.parametrize(
        "style, expected_more",
        [
            ("ascii", "..."),
            ("unicode-angle", "…"),
        ],
    )
    def test_fmt_sequence_max_items_appends_ellipsis(self, style, expected_more):
        out = fmt_sequence(list(range(5)), style=style, max_items=3)
        # Expect 3 items then the ellipsis token
        assert out.endswith(expected_more + "]") or out.endswith(expected_more + ")")
        assert "<int: 0>" in out or "⟨int: 0⟩" in out

    def test_fmt_sequence_custom_ellipsis_propagates(self):
        out = fmt_sequence(list(range(5)), style="ascii", max_items=2, ellipsis=" [more] ")
        assert out.endswith(" [more] ]")

    def test_fmt_sequence_extreme_max_items_limits(self):
        """Edge cases for max_items limits"""
        seq = [1, 2, 3]

        # Zero items - should show ellipsis only
        out = fmt_sequence(seq, max_items=0)
        assert out == "[...]" or out == "[…]"

        # Very large max_items should work
        out = fmt_sequence(seq, max_items=1000)
        assert "<int: 1>" in out and "<int: 2>" in out and "<int: 3>" in out

    # ---------- Special sequence types ----------

    def test_fmt_sequence_range_object(self):
        """range objects should be formatted properly"""
        r = range(3, 8, 2)
        out = fmt_sequence(r, style="ascii")
        assert "<int: 3>" in out
        assert "<int: 5>" in out
        assert "<int: 7>" in out

    def test_fmt_sequence_deque(self):
        """collections.deque should format like lists"""
        from collections import deque
        d = deque([1, 2, 3])
        out = fmt_sequence(d, style="ascii")
        assert "<int: 1>" in out
        assert "<int: 2>" in out
        assert "<int: 3>" in out

    # ---------- Parameter validation (defensive) ----------

    def test_fmt_sequence_non_iterable_fallback(self):
        """Non-iterables should be handled gracefully via fmt_value"""
        # This should be caught by the textual check or fall through to fmt_value behavior
        out = fmt_sequence(42, style="ascii")  # type: ignore
        # Should either handle as atomic or raise clear error
        assert "<int: 42>" in out or "42" in out

    def test_fmt_sequence_negative_max_items(self):
        """Negative max_items should not crash"""
        seq = [1, 2, 3]
        out = fmt_sequence(seq, max_items=-1)
        # Should handle gracefully
        assert "[" in out and "]" in out

    def test_fmt_sequence_huge_individual_elements(self):
        """Individual elements that are very long should be truncated"""
        huge_str = "x" * 1000
        seq = ["small", huge_str, "small2"]
        out = fmt_sequence(seq, style="ascii", max_repr=20)
        # Huge element should be truncated
        assert len(out) < 500  # Much shorter than the huge element
        assert "small" in out
        assert "..." in out or "…" in out


class TestFmtType:
    """Tests for the fmt_type() utility."""

    @pytest.mark.parametrize(
        "obj",
        [42, "a string", ValueError("test"), AnyUserClass()],
        ids=["instance-int", "instance-str", "instance-exception", "instance-custom"],
    )
    def test_fmt_type_basic_instance_input(self, obj):
        """Test that fmt_type correctly formats the type of an instance."""
        expected = f"<type: {type(obj).__name__}>"
        assert fmt_type(obj) == expected

    @pytest.mark.parametrize(
        "obj_type",
        [int, str, ValueError, AnyUserClass],
        ids=["type-int", "type-str", "type-exception", "type-custom"],
    )
    def test_fmt_type_basic_type_input(self, obj_type):
        """Test that fmt_type correctly formats a type object directly."""
        expected = f"<type: {obj_type.__name__}>"
        assert fmt_type(obj_type) == expected

    @pytest.mark.parametrize(
        "style, expected_format",
        [
            ("ascii", "<type: {name}>"),
            ("unicode-angle", "⟨type: {name}⟩"),
            ("equal", "type={name}"),
        ],
        ids=["ascii", "unicode-angle", "equal"],
    )
    def test_fmt_type_different_styles(self, style, expected_format):
        """Test various formatting styles."""
        name = AnyUserClass.__name__
        expected = expected_format.format(name=name)
        assert fmt_type(AnyUserClass, style=style) == expected

    def test_fmt_type_show_module_flag(self):
        """Test the 'show_module' flag for built-in and custom types."""
        # For a custom class, it should show the module name.
        expected_name = f"{AnyUserClass.__module__}.{AnyUserClass.__name__}"
        assert fmt_type(AnyUserClass, show_module=True) == f"<type: {expected_name}>"

        # For a built-in type, 'builtins' should be omitted.
        assert fmt_type(list, show_module=True) == "<type: list>"

    def test_fmt_type_truncation(self):
        """Test that long type names are truncated correctly."""

        class ThisIsAVeryLongClassNameForTestingPurposes:
            pass

        out = fmt_type(ThisIsAVeryLongClassNameForTestingPurposes, max_repr=20, style="ascii")
        assert out.startswith("<type: ThisIsAVeryLongClass")
        assert out.endswith("...>")

    def test_fmt_type_truncation_with_custom_ellipsis(self):
        """Test truncation with a custom ellipsis token."""

        class AnotherLongName:
            pass

        out = fmt_type(AnotherLongName, max_repr=10, ellipsis="...[more]", style="ascii")
        assert out == "<type: AnotherLon...[more]>"

    def test_fmt_type_with_broken_name_attribute(self):
        """Test graceful fallback for types with a broken __name__."""

        class MetaWithBrokenName(type):
            @property
            def __name__(cls):
                raise AttributeError("Name is deliberately broken")

        class MyBrokenType(metaclass=MetaWithBrokenName):
            pass

        out = fmt_type(MyBrokenType, style="ascii")
        assert out.startswith("<type: <class '")
        assert "MyBrokenType" in out
        assert out.endswith(">>")


class TestFmtValue:
    # ---------- Basic functionality ----------

    @pytest.mark.parametrize(
        "style, value, expected",
        [
            ("equal", 5, "int=5"),
            ("paren", 5, "int(5)"),
            ("colon", 5, "int: 5"),
            ("unicode-angle", 5, "⟨int: 5⟩"),
            ("ascii", 5, "<int: 5>"),
        ],
    )
    def test_fmt_value_basic_styles(self, style, value, expected):
        assert fmt_value(value, style=style) == expected

    # ---------- Edge cases critical for exceptions/logging ----------

    def test_fmt_value_none_value(self):
        """None values are common in exception contexts"""
        out = fmt_value(None, style="ascii")
        assert out == "<NoneType: None>"

    def test_fmt_value_empty_string(self):
        """Empty strings are common edge cases"""
        out = fmt_value("", style="ascii")
        assert out == "<str: ''>"

    def test_fmt_value_empty_containers(self):
        """Empty containers often appear in validation errors"""
        assert fmt_value([], style="ascii") == "<list: []>"
        assert fmt_value({}, style="ascii") == "<dict: {}>"
        assert fmt_value(set(), style="ascii") == "<set: set()>"

    def test_fmt_value_very_long_string_realistic(self):
        """Test with realistic long content like file paths or SQL"""
        long_path = "/very/long/path/to/some/deeply/nested/directory/structure/file.txt"
        out = fmt_value(long_path, style="ascii", max_repr=20)
        assert "..." in out
        assert out.startswith("<str: '")

    def test_fmt_value_object_with_broken_repr(self):
        """Objects with broken __repr__ are common in exception scenarios"""

        class BrokenRepr:
            def __repr__(self):
                raise RuntimeError("Broken repr!")

        obj = BrokenRepr()
        # Should not crash - fmt_value should handle this gracefully
        try:
            out = fmt_value(obj, style="ascii")
            # If repr() fails, Python's default behavior varies
            assert "BrokenRepr" in out or "RuntimeError" in out or "repr" in out.lower()
        except Exception:
            # If it does crash, that's a bug - fmt_value should be defensive
            pytest.fail("fmt_value should handle broken __repr__ gracefully")

    def test_fmt_value_recursive_object(self):
        """Recursive objects can cause infinite recursion in repr"""
        lst = [1, 2]
        lst.append(lst)  # Create recursion: [1, 2, [...]]
        out = fmt_value(lst, style="ascii")
        assert "list" in out
        assert "..." in out or "[" in out  # Should handle recursion gracefully

    def test_fmt_value_unicode_in_strings(self):
        """Unicode content is common in modern applications"""
        unicode_str = "Hello 世界 🌍 café"
        out = fmt_value(unicode_str, style="unicode-angle")
        assert "⟨str:" in out
        assert "世界" in out or "\\u" in out  # Either preserved or escaped

    def test_fmt_value_ascii_escapes_inner_gt(self):
        """Critical for ASCII style - angle brackets in content"""
        s = "X>Y"
        out = fmt_value(s, style="ascii")
        assert out == "<str: 'X\\>Y'>"

    def test_fmt_value_large_numbers(self):
        """Large numbers common in scientific/financial contexts"""
        big_int = 123456789012345678901234567890
        out = fmt_value(big_int, style="ascii")
        assert "int" in out
        assert str(big_int) in out or "..." in out

    # ---------- Truncation robustness ----------

    @pytest.mark.parametrize(
        "style, ellipsis_expected",
        [
            ("ascii", "..."),
            ("unicode-angle", "…"),
        ],
    )
    def test_fmt_value_truncation_default_ellipsis_per_style(self, style, ellipsis_expected):
        long = "x" * 50
        out = fmt_value(long, style=style, max_repr=10)
        assert ellipsis_expected in out
        # Ensure the result ends with the chosen ellipsis inside the wrapper
        out_wo_closer = out.rstrip(">\u27e9")
        assert out_wo_closer.endswith(ellipsis_expected)

    def test_fmt_value_truncation_custom_ellipsis(self):
        out = fmt_value("abcdefghij", style="ascii", max_repr=5, ellipsis="<<more>>")
        assert out == "<str: 'a'<<more>>>"

    def test_fmt_value_truncation_extreme_limits(self):
        """Edge cases for truncation limits"""
        # Very short limit
        out = fmt_value("hello", style="ascii", max_repr=1)
        assert out.startswith("<str:")
        assert "..." in out or "…" in out

        # Zero limit - should not crash
        out = fmt_value("hello", style="ascii", max_repr=0)
        assert out.startswith("<str:")

    # ---------- Type handling for exceptions ----------

    def test_fmt_value_exception_objects(self):
        """Exception objects themselves often appear in logging"""
        exc = ValueError("Something went wrong")
        out = fmt_value(exc, style="ascii")
        assert "ValueError" in out
        assert "Something went wrong" in out

    def test_fmt_value_type_name_for_user_class(self):
        """User-defined types common in business logic errors"""

        class Foo:
            def __repr__(self):
                return "Foo()"

        f = Foo()
        out = fmt_value(f, style="equal")
        assert out.startswith("Foo=")

    def test_fmt_value_builtin_types_comprehensive(self):
        """Comprehensive test of common built-in types"""
        test_cases = [
            (42, "int"),
            (3.14, "float"),
            (True, "bool"),
            (b"bytes", "bytes"),
            (bytearray(b"ba"), "bytearray"),
            (complex(1, 2), "complex"),
            (frozenset([1, 2]), "frozenset"),
        ]

        for value, expected_type in test_cases:
            out = fmt_value(value, style="colon")
            assert f"{expected_type}:" in out

    def test_fmt_value_bytes_is_textual(self):
        """Bytes often contain binary data that needs careful handling"""
        b = b"abc\x00\xff"  # Include null and high bytes
        out = fmt_value(b, style="unicode-angle")
        assert out.startswith("⟨bytes:")
        assert "\\x" in out or "abc" in out  # Should handle binary safely

    # ---------- Parameter validation (defensive) ----------

    def test_fmt_value_invalid_style_fallback(self):
        """Should gracefully handle invalid styles"""
        out = fmt_value(123, style="nonexistent-style")
        # Should fall back to default formatting, not crash
        assert "123" in out
        assert "int" in out

    def test_fmt_value_negative_max_repr(self):
        """Edge case: negative max_repr should not crash"""
        out = fmt_value("hello", style="ascii", max_repr=-1)
        # Should handle gracefully, not crash
        assert out.startswith("<str:")


# TestGetCallerName Helper functions: defined at the module level to ensure a predictable stack frame.
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


class TestListGet:
    """Test suite for the list_get method."""

    @pytest.mark.parametrize(
        "lst, index, default, expected",
        [
            ([10, 20, 30], 0, None, 10),
            ([10, 20, 30], 2, None, 30),
            ([10, 20, 30], -1, None, 30),
            ([10, 20, 30], -3, None, 10),
        ],
        ids=[
            "get-first-item",
            "get-last-item-positive-index",
            "get-last-item-negative-index",
            "get-first-item-negative-index",
        ]
    )
    def test_get_item_successfully(self, lst, index, default, expected):
        """Test that items are retrieved successfully with valid indices."""
        assert list_get(lst, index, default=default) == expected

    @pytest.mark.parametrize(
        "lst, index, default, expected",
        [
            ([10, 20, 30], 5, "missing", "missing"),
            ([10, 20, 30], -5, "missing", "missing"),
            ([], 0, "empty", "empty"),
            (None, 0, "list_is_none", "list_is_none"),
            ([10, 20, 30], None, "index_is_none", "index_is_none"),
            ([10, 20], 2, None, None),
        ],
        ids=[
            "index-out-of-bounds-positive",
            "index-out-of-bounds-negative",
            "empty-list",
            "list-is-none",
            "index-is-none",
            "default-value-is-none",
        ]
    )
    def test_return_default_value(self, lst, index, default, expected):
        """Test that the default value is returned for various edge cases."""
        assert list_get(lst, index, default=default) == expected

    @pytest.mark.parametrize(
        "invalid_list",
        [("a", "b", "c"), {"key": "value"}, "a string", 42, ],
        ids=["tuple-instead-of-list", "dict-instead-of-list",
             "string-instead-of-list", "int-instead-of-list",
             ]
    )
    def test_raise_error_on_invalid_list_type(self, invalid_list):
        """Test that a TypeError is raised for invalid list types."""
        with pytest.raises(TypeError, match="expected list or None, got"):
            list_get(invalid_list, 0)

    @pytest.mark.parametrize(
        "invalid_index", ["1", 2.5, [1]],
        ids=["string-index", "float-index", "list-as-index", ]
    )
    def test_raise_error_on_invalid_index_type(self, invalid_index):
        """Test that a TypeError is raised for invalid index types."""
        with pytest.raises(TypeError, match="expected int or None for index, got"):
            list_get([1, 2, 3], invalid_index)


class TestListify:
    @pytest.mark.parametrize(
        "value", ["abc", b"bytes", bytearray(b"buf")],
    )
    def test_atomic_text_and_bytes(self, value):
        out = listify(value)
        assert out == [value], "Text/bytes must be wrapped as a single element"

    @pytest.mark.parametrize(
        "value, expected",
        [
            ([1, 2, "3"], [1, 2, "3"]),
            ((1, "2"), [1, "2"]),
            ({"a": 1, "b": 2}, ["a", "b"]),  # dict iterates over keys in insertion order
        ],
    )
    def test_iterables_are_expanded(self, value, expected):
        assert listify(value) == expected

    def test_generator_and_memoryview_are_expanded(self):
        gen = (i for i in (1, 2, 3))
        assert listify(gen) == [1, 2, 3]

        mv = memoryview(b"ab")
        # memoryview is iterable over ints (bytes)
        assert listify(mv) == [97, 98]  # ord('a') == 97, ord('b') == 98

    @pytest.mark.parametrize(
        "value", [42, 3.14, object()],
    )
    def test_non_iterables_are_wrapped(self, value):
        out = listify(value)
        assert len(out) == 1 and out[0] is value

    @pytest.mark.parametrize(
        "value, as_type, expected",
        [
            ((1, "2"), str, ["1", "2"]),
            (["1", "2", "3"], int, [1, 2, 3]),
            ("123", int, [123]),  # str is atomic, conversion applies to the single wrapped value
            ((1.2, 3.4), lambda x: round(float(x)), [1, 3]),
        ],
    )
    def test_as_type_conversion_for_each_item(self, value, as_type, expected):
        assert listify(value, as_type=as_type) == expected

    def test_conversion_failure_raises_valueerror_with_context(self):
        with pytest.raises(ValueError) as excinfo:
            listify(["1", "x", "3"], as_type=int)

        msg = str(excinfo.value)
        # Use substring search instead of exact match
        assert "failed to convert value" in msg
        # Ensure original exception is chained as the cause
        assert excinfo.value.__cause__ is not None

    def test_dict_keys_only_converted_when_as_type_given(self):
        d = {"a": 1, "b": 2}
        assert listify(d) == ["a", "b"]
        assert listify(d, as_type=str) == ["a", "b"]  # already strings, but still processed
        assert listify(d, as_type=lambda k: k.upper()) == ["A", "B"]

    def test_empty_iterables_and_edge_cases(self):
        assert listify([]) == []
        assert listify(()) == []
        # bytearray is atomic (unlike memoryview), should not expand
        ba = bytearray(b"xy")
        assert listify(ba) == [ba]


class TestPrintTitle:
    """
    Test suite for the print_title function.
    """

    def test_default_parameters(self, capsys):
        """
        Test print_title with default prefix, suffix, start, and end.
        """
        print_title("My Title")
        captured = capsys.readouterr()
        assert captured.out == "\n------- My Title -------\n"

    def test_custom_prefix_and_suffix(self, capsys):
        """
        Test print_title with custom prefix and suffix.

        Args:

        """
        print_title("Another Title", prefix="<<< ", suffix=" >>>")
        captured = capsys.readouterr()
        assert captured.out == "\n<<< Another Title >>>\n"

    def test_empty_title(self, capsys):
        """
        Test print_title with an empty title string.
        """
        print_title("")
        captured = capsys.readouterr()
        assert captured.out == "\n-------  -------\n"

    def test_no_newlines(self, capsys):
        """
        Test print_title with no start or end newlines.
        """
        print_title("No Newlines", start="", end="")
        captured = capsys.readouterr()
        assert captured.out == "------- No Newlines -------"

    def test_different_newlines(self, capsys):
        """
        Test print_title with different start and end characters.
        """
        print_title("Custom Newlines", start="START\n", end="\nEND")
        captured = capsys.readouterr()
        assert captured.out == "START\n------- Custom Newlines -------\nEND"

    def test_numeric_title(self, capsys):
        """
        Test print_title with a numeric title (should be converted to string).
        """
        print_title(12345)
        captured = capsys.readouterr()
        assert captured.out == "\n------- 12345 -------\n"


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
    def test_successful_retrieval(self, seq: Sequence, index: int, default: Any, expected: Any):
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
    def test_default_value_scenarios(self, seq: Sequence | None, index: int | None, default: Any, expected: Any):
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


class TestToAscii:
    """
    Test suite for the to_ascii function.
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
        ]
    )
    def test_successful_conversion(self, s, replacement, expected):
        """
        Tests successful conversion of str, bytes, and bytearray to ASCII.
        """
        if replacement is None:
            result = to_ascii(s)
        else:
            result = to_ascii(s, replacement=replacement)

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
        ]
    )
    def test_invalid_inputs_and_replacements(self, s, replacement, error, match):
        """
        Tests that to_ascii raises appropriate errors for invalid inputs.
        """
        with pytest.raises(error, match=r"(?i)" + match):
            if replacement is None:
                to_ascii(s)
            else:
                to_ascii(s, replacement=replacement)


class TestTools:

    def test_cli_multiline(self):
        cmd = "cmd sub-cmd"
        args = ["SRC", "DEST", "-h", 1, "-q", "-xyz", "--opt=2", "--is-flag"]
        args_str = cli_multiline(args, multiline_indent=4)
        print(f"ARGS:\n'{args_str}'")
        print()
        print(f"CMD+ARGS:\n'{cmd} {args_str}'")

    def test_clify(self):
        assert clify("") == []
        assert clify(1) == ["1"]

        assert clify("abc") == ["abc"]
        assert clify("abc --help", shlex_split=True) == ["abc", "--help"]
        assert clify((1, 2, 3)) == ["1", "2", "3"]

    def test_listify(self):
        assert listify(1) == [1]
        assert listify(1, as_type=str) == ["1"]

        assert listify("abc") == ["abc"]

        assert listify([1, 2, 3]) == [1, 2, 3]
        assert listify((1, 2, 3), as_type=str) == ["1", "2", "3"]
        assert listify({1, 2, 3}) == [1, 2, 3]

    def test_is_numbered_version(self):
        assert is_numbered_version(0, min_depth=0)
        assert is_numbered_version("1", min_depth=0)
        assert is_numbered_version("1x.2y", min_depth=1)
        assert is_numbered_version("1.2rc1", min_depth=1)
        assert is_numbered_version("1.2.5-alpha", max_depth=2)
        assert not is_numbered_version("1.abc23", min_depth=1)
        assert not is_numbered_version("v1", max_depth=0)

    def test_is_pep440_version(self):
        assert is_pep440_version(0, min_depth=0)
        assert is_pep440_version("25a1")
        assert is_pep440_version("1b2")
        assert is_pep440_version("1.2.3")
        assert is_pep440_version("1.2.3.4.5")
        assert is_pep440_version("1.2.3a4")
        assert is_pep440_version("1.2.3rc4")
        assert is_pep440_version("1.2.3.post4")
        assert not is_pep440_version("1.xyz23", min_depth=1)
        assert not is_pep440_version("v1", max_depth=0)

    def test_is_semantic_version(self):
        assert is_semantic_version("1", min_depth=0)
        assert is_semantic_version("1.2", min_depth=1)
        assert is_semantic_version("1.2.3", min_depth=2)
        assert is_semantic_version("1.2.5-alpha", max_depth=7, allow_meta=True)
        assert not is_semantic_version("1.2", min_depth=2)
        assert not is_semantic_version("1.2.3.4", max_depth=2)
        assert not is_semantic_version("1.2.5-alpha", max_depth=7)

# class TestDictGetSet:
#     def test_dict_get(self):
#         d = {"a": 1,
#              "b": {"c": 2},
#              "e": {"empty": None}
#              }
#         assert dict_get(d, key="a") == 1, "Should return d['a']"
#         assert dict_get(d, key=["a"]) == 1, "Should return d['a']"
#         assert dict_get(d, key="b.c") == 2, "Should return d['b']['c']"
#         assert dict_get(d, key=["b", "c"]) == 2, "Should return d['b']['c']"
#         assert dict_get(d, key="e.empty") == "", "Should return ''"
#         assert dict_get(d, key="e.empty", default=None) is None, "Should return None"
#
#     def test_dict_set(self):
#         d = {"a": 0,
#              "b": {"c": 0}
#              }
#
#         dict_set(d, dot_key="a", value=1)
#         dict_set(d, dot_key="b.c", value=2)
#         dict_set(d, dot_key="i.j.q", value=3)
#         assert d["a"] == 1
#         assert d["b"]["c"] == 2
#         assert d["i"]["j"]["q"] == 3
