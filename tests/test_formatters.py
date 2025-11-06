#
# C108 - Formatters Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import re


# Third Party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.formatters import (
    fmt_any,
    fmt_exception,
    fmt_mapping,
    fmt_sequence,
    fmt_type,
    fmt_value,
)


# Tests ----------------------------------------------------------------------------------------------------------------
class AnyClass:
    """
    A simple class for testing user-defined types
    """


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
    def test_dispatch(self, obj, expected_substring):
        """Dispatches to the correct formatter."""
        result = fmt_any(obj)
        assert expected_substring in result

    @pytest.mark.parametrize("style", ["ascii", "unicode-angle"])
    def test_style_forwarding(self, style):
        """Style is forwarded to formatters."""
        exc_result = fmt_any(ValueError("test"), style=style)
        dict_result = fmt_any({"key": "val"}, style=style)
        list_result = fmt_any([1, 2], style=style)
        value_result = fmt_any("text", style=style)

        assert "ValueError" in exc_result and "test" in exc_result
        assert "key" in dict_result and "val" in dict_result
        assert "int: 1" in list_result
        assert "str" in value_result and "text" in value_result

    def test_exception_traceback(self):
        """include_traceback toggles traceback details."""
        try:
            raise ValueError("traceback test")
        except ValueError as e:
            result_without = fmt_any(e, include_traceback=False)
            result_with = fmt_any(e, include_traceback=True)

            assert "ValueError" in result_without
            assert "traceback test" in result_without
            assert "ValueError" in result_with
            assert "traceback test" in result_with

            has_location_info = any(
                indicator in result_with.lower() for indicator in ["test_fmt_any", "line", "at "]
            )
            assert has_location_info

    @pytest.mark.parametrize("max_items", [1, 3, 5])
    def test_max_items_forwarding(self, max_items):
        """max_items is forwarded to collection formatters."""
        large_dict = {f"key{i}": f"val{i}" for i in range(10)}
        large_list = list(range(10))

        dict_result = fmt_any(large_dict, max_items=max_items)
        list_result = fmt_any(large_list, max_items=max_items)

        if max_items < 10:
            assert "..." in dict_result or "…" in dict_result
            assert "..." in list_result or "…" in list_result

    @pytest.mark.parametrize("max_repr", [10, 20, 50])
    def test_max_repr_forwarding(self, max_repr):
        """max_repr bounds formatter output."""
        long_message = "x" * 100

        exc_result = fmt_any(ValueError(long_message), max_repr=max_repr)
        dict_result = fmt_any({"key": long_message}, max_repr=max_repr)
        list_result = fmt_any([long_message], max_repr=max_repr)
        value_result = fmt_any(long_message, max_repr=max_repr)

        assert len(exc_result) <= max_repr + 50
        assert len(dict_result) <= max_repr + 100
        assert len(list_result) <= max_repr + 100
        assert len(value_result) <= max_repr + 50

    def test_depth_handling(self):
        """Depth limits nested formatting detail."""
        nested = {"outer": {"inner": [1, 2, {"deep": "value"}]}}

        shallow_result = fmt_any(nested, depth=1)
        deep_result = fmt_any(nested, depth=3)

        assert "outer" in shallow_result
        assert "outer" in deep_result
        assert "inner" in deep_result
        assert "deep" in deep_result

    def test_textual_sequences_atomic(self):
        """Textual sequences are treated as atomic values."""
        text_str = "hello world"
        text_bytes = b"hello world"
        text_bytearray = bytearray(b"hello world")

        str_result = fmt_any(text_str)
        bytes_result = fmt_any(text_bytes)
        bytearray_result = fmt_any(text_bytearray)

        assert "str: 'hello world'" in str_result
        assert "bytes:" in bytes_result
        assert "bytearray:" in bytearray_result
        assert "str: 'h'" not in str_result

    def test_custom_ellipsis(self):
        """Custom ellipsis is used when provided."""
        large_dict = {f"k{i}": f"v{i}" for i in range(10)}
        long_string = "x" * 100

        dict_result = fmt_any(large_dict, max_items=2, ellipsis="[MORE]")
        str_result = fmt_any(long_string, max_repr=10, ellipsis="[MORE]")

        assert "[MORE]" in dict_result
        assert "[MORE]" in str_result

    def test_edge_cases(self):
        """Edge cases and special object types."""
        none_result = fmt_any(None)
        assert "NoneType" in none_result

        empty_dict_result = fmt_any({})
        empty_list_result = fmt_any([])
        empty_tuple_result = fmt_any(())

        assert "{}" in empty_dict_result
        assert "[]" in empty_list_result
        assert "()" in empty_tuple_result

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
    def test_basic_and_empty(self, exc, expected):
        """Format exceptions with and without message."""
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
    def test_types(self, exc, expected):
        """Format different exception types."""
        assert fmt_exception(exc) == expected

    def test_unicode(self):
        """Handle unicode in exception message."""
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
    def test_truncate_and_ellipsis(self, msg, max_repr, ellipsis, starts_with, ends_with):
        """Truncate message and honor custom ellipsis."""
        try:
            raise ValueError(msg)
        except ValueError as e:
            out = fmt_exception(e, max_repr=max_repr, ellipsis=ellipsis)
            assert out.startswith(starts_with)
            assert out.endswith(ends_with)
            # Ensure truncation actually happened (shorter than message + overhead)
            assert len(out) < len(f"<ValueError: {msg}>")

    @pytest.mark.parametrize("style", ["ascii", "unicode-angle", "equal"])
    def test_style(self, style):
        """Apply selected style to exception formatting."""
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
    def test_max_repr_edges(self, max_repr):
        """Constrain output length for small max_repr."""
        exc = ValueError("short")
        out = fmt_exception(exc, max_repr=max_repr)
        # Always returns something sane with a proper wrapper and type name
        assert "ValueError" in out
        # Should not be excessively long for tiny limits
        assert len(out) <= 40

    def test_traceback_location_on(self):
        """Include traceback location when enabled."""

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

    def test_traceback_location_off(self):
        """Omit traceback location when disabled."""

        def _raise_here():
            raise ValueError("no tb")

        try:
            _raise_here()
        except ValueError as e:
            out = fmt_exception(e, include_traceback=False)
            assert out == "<ValueError: no tb>"

    def test_broken_str(self):
        """Fallback when __str__ raises inside exception."""

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

    def test_basic(self):
        """Format a simple mapping."""
        mp = {"a": 1, 2: "b"}
        out = fmt_mapping(mp, style="ascii")
        # Insertion order preserved by dicts
        assert out == "{<str: 'a'>: <int: 1>, <int: 2>: <str: 'b'>}"

    def test_nested_sequence(self):
        """Format mapping containing a nested sequence."""
        mp = {"k": [1, 2]}
        out = fmt_mapping(mp, style="unicode-angle")
        assert out == "{⟨str: 'k'⟩: [⟨int: 1⟩, ⟨int: 2⟩]}"

    # ---------- Edge cases critical for exceptions/logging ----------

    def test_empty(self):
        """Handle empty dicts."""
        assert fmt_mapping({}) == "{}"

    def test_none_keys_and_values(self):
        """Format mappings with None keys and values."""
        mp = {None: "value", "key": None, None: None}
        out = fmt_mapping(mp, style="ascii")
        assert "<NoneType: None>" in out
        assert "value" in out or "key" in out

    def test_complex_key_types(self):
        """Format mappings with various key types."""
        mp = {
            42: "int key",
            (1, 2): "tuple key",
            frozenset([3, 4]): "frozenset key",
            True: "bool key",
        }
        out = fmt_mapping(mp, style="ascii")
        assert "<int: 42>" in out
        assert "<tuple:" in out
        assert "<frozenset:" in out
        assert "<bool: True>" in out

    def test_broken_key_repr(self):
        """Handle keys whose __repr__ raises."""

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

    def test_broken_value_repr(self):
        """Handle values whose __repr__ raises."""

        class BrokenValueRepr:
            def __repr__(self):
                raise RuntimeError("Value repr is broken!")

        mp = {"key": BrokenValueRepr()}
        out = fmt_mapping(mp, style="ascii")
        assert "key" in out
        assert "BrokenValueRepr" in out
        assert "repr failed" in out

    def test_large_mapping_truncate(self):
        """Truncate very large mappings."""
        big_dict = {f"key_{i}": f"value_{i}" for i in range(20)}
        out = fmt_mapping(big_dict, style="ascii", max_items=3)
        # Should only show 3 items plus ellipsis
        key_count = out.count("<str: 'key_")
        assert key_count == 3
        assert "..." in out

    def test_deeply_nested(self):
        """Respect depth limits for nested structures."""
        nested = {"level1": {"level2": {"level3": [1, 2, {"level4": "deep"}]}}}

        # With depth=2, should recurse into level2 but treat level3+ as atomic
        out = fmt_mapping(nested, style="ascii", depth=2)
        assert "level1" in out
        assert "level2" in out
        # level3 list should be formatted as atomic
        assert "<list:" in out

    def test_circular_references(self):
        """Handle circular references without infinite recursion."""
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
        ids=["ascii", "unicode-angle"],
    )
    def test_max_items_appends_ellipsis(self, style, expected_more):
        """Append an ellipsis when max_items is exceeded."""
        mp = {i: i for i in range(5)}
        out = fmt_mapping(mp, style=style, max_items=3)
        assert out.endswith(expected_more + "}")

    def test_custom_ellipsis(self):
        """Use custom ellipsis token when provided."""
        mp = {i: i for i in range(4)}
        out = fmt_mapping(mp, style="ascii", max_items=2, ellipsis="~more~")
        assert out.endswith("~more~}")

    def test_extreme_max_items(self):
        """Handle edge cases for max_items limits."""
        mp = {"a": 1, "b": 2}

        # Zero items - should show ellipsis only
        out = fmt_mapping(mp, max_items=0)
        assert out == "{...}" or out == "{…}"

        # One item
        out = fmt_mapping(mp, max_items=1)
        item_count = out.count("<")
        assert item_count >= 2  # At least one key and one value

    # ---------- Special mapping types ----------

    def test_ordered_dict(self):
        """Preserve order for OrderedDict."""
        from collections import OrderedDict

        od = OrderedDict([("first", 1), ("second", 2)])
        out = fmt_mapping(od, style="ascii")
        # Should show first before second
        first_pos = out.find("first")
        second_pos = out.find("second")
        assert first_pos < second_pos

    def test_defaultdict(self):
        """Format defaultdict like a regular dict."""
        from collections import defaultdict

        dd = defaultdict(list)
        dd["key"] = [1, 2, 3]
        out = fmt_mapping(dd, style="ascii")
        assert "key" in out
        assert "[<int: 1>" in out or "<list:" in out

    def test_textual_values_atomic(self):
        """Treat text-like values as atomic."""
        mp = {"s": "xyz", "b": b"ab", "ba": bytearray(b"test")}
        out = fmt_mapping(mp, style="paren")
        assert "str('xyz')" in out
        assert "bytes(b'ab')" in out
        assert "bytearray(" in out

    # ---------- Parameter validation (defensive) ----------

    def test_invalid_mapping_type(self):
        """Handle non-mapping inputs gracefully or raise clear error."""
        try:
            out = fmt_mapping("not a mapping", style="ascii")  # type: ignore
            # If it doesn't raise, should produce some reasonable output
            assert "str" in out or "not a mapping" in out
        except (TypeError, AttributeError) as e:
            # Acceptable to raise clear error for invalid input
            assert "mapping" in str(e).lower() or "items" in str(e).lower()

    def test_negative_max_items(self):
        """Accept negative max_items without crashing."""
        mp = {"a": 1}
        out = fmt_mapping(mp, max_items=-1)
        # Should handle gracefully
        assert "{" in out and "}" in out

    def test_huge_individual_values(self):
        """Truncate very large individual values."""
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
        ids=["list", "tuple"],
    )
    def test_delimiters_list_vs_tuple(self, seq, style, expected):
        """Format delimiters for list vs tuple."""
        out = fmt_sequence(seq, style=style)
        if isinstance(seq, list):
            assert out == f"[{expected}]"
        else:
            assert out == f"({expected})"

    def test_singleton_tuple_trailing_comma(self):
        """Show trailing comma for singleton tuple."""
        out = fmt_sequence((1,), style="ascii")
        assert out == "(<int: 1>,)"

    # ---------- Edge cases critical for exceptions/logging ----------

    def test_empty_containers(self):
        """Format empty containers."""
        assert fmt_sequence([]) == "[]"
        assert fmt_sequence(()) == "()"
        assert fmt_sequence(set()) == "{}"

    def test_none_elements(self):
        """Format None elements in sequence."""
        seq = [1, None, "hello", None]
        out = fmt_sequence(seq, style="ascii")
        assert "<int: 1>" in out
        assert "<NoneType: None>" in out
        assert "<str: 'hello'>" in out

    def test_non_iterable_fallback(self):
        """Fallback to fmt_value for non-iterables."""
        out = fmt_sequence(42, style="ascii")  # type: ignore
        # Should fall back to fmt_value behavior for non-iterables
        assert out == "<int: 42>"

    def test_mixed_types(self):
        """Format realistic mix of element types."""
        mixed = [42, "status", None, {"error": True}, [1, 2]]
        out = fmt_sequence(mixed, style="ascii")
        assert "<int: 42>" in out
        assert "<str: 'status'>" in out
        assert "<NoneType: None>" in out
        assert "{<str: 'error'>:" in out  # nested dict
        assert "[<int: 1>" in out  # nested list

    def test_broken_element_repr(self):
        """Handle elements with broken __repr__."""

        class BrokenRepr:
            def __repr__(self):
                raise RuntimeError("Element repr is broken!")

        seq = [1, BrokenRepr(), "after"]
        out = fmt_sequence(seq, style="ascii")
        assert "<int: 1>" in out
        assert "BrokenRepr" in out
        assert "repr failed" in out
        assert "<str: 'after'>" in out

    def test_large_list_truncation(self):
        """Truncate large sequences."""
        big_list = list(range(50))
        out = fmt_sequence(big_list, style="ascii", max_items=3)
        # Should only show 3 items plus ellipsis
        item_count = out.count("<int:")
        assert item_count == 3
        assert "..." in out

    def test_deep_nesting(self):
        """Limit recursion depth in nested structures."""
        nested = [1, [2, [3, [4, [5]]]]]

        # With depth=2, should recurse 2 levels but treat deeper as atomic
        out = fmt_sequence(nested, style="ascii", depth=2)
        assert "<int: 1>" in out
        assert "[<int: 2>" in out  # First level of nesting
        assert "[<int: 3>" in out  # Second level of nesting
        # Deeper nesting should be atomic
        assert "<list:" in out

    def test_circular_references(self):
        """Handle circular references safely."""
        lst = [1, 2]
        lst.append(lst)  # Create circular reference: [1, 2, [...]]

        out = fmt_sequence(lst, style="ascii")
        # Should handle gracefully without infinite recursion
        assert "<int: 1>" in out
        assert "<int: 2>" in out
        assert "..." in out or "[" in out  # Circular part shown somehow

    def test_generators_and_iterators(self):
        """Consume generators and iterators once."""

        def gen():
            yield 1
            yield 2
            yield 3

        out = fmt_sequence(gen(), style="ascii", max_items=2)
        # Should consume generator and show first 2 items
        assert "<int: 1>" in out
        assert "<int: 2>" in out
        assert "..." in out

    def test_sets_unordered(self):
        """Format sets without relying on order."""
        s = {3, 1, 2}
        out = fmt_sequence(s, style="ascii")
        assert out.startswith("{")
        assert out.endswith("}")
        # Should contain all elements (order may vary)
        assert "<int: 1>" in out
        assert "<int: 2>" in out
        assert "<int: 3>" in out

    # ---------- String/textual handling ----------

    def test_string_is_atomic(self):
        """Treat strings as atomic values."""
        out = fmt_sequence("abc", style="colon")
        assert out == "str: 'abc'"
        # Should NOT be ['a', 'b', 'c']

    def test_bytes_is_atomic(self):
        """Treat bytes as atomic values."""
        out = fmt_sequence(b"hello", style="ascii")
        assert out == "<bytes: b'hello'>"

    def test_bytearray_is_atomic(self):
        """Treat bytearray as atomic value."""
        ba = bytearray(b"test")
        out = fmt_sequence(ba, style="ascii")
        assert out.startswith("<bytearray:")

    def test_unicode_strings(self):
        """Preserve or safely escape Unicode."""
        unicode_seq = ["Hello", "世界", "🌍"]
        out = fmt_sequence(unicode_seq, style="ascii")
        assert "Hello" in out
        # Unicode should be preserved or safely escaped
        assert "世界" in out or "\\u" in out
        assert "🌍" in out or "\\u" in out

    # ---------- Truncation robustness ----------

    def test_nesting_depth_1(self):
        """Format with nesting depth of 1."""
        seq = [1, [2, 3]]
        out = fmt_sequence(seq, style="unicode-angle", depth=1)
        assert out == "[⟨int: 1⟩, [⟨int: 2⟩, ⟨int: 3⟩]]"

    def test_nesting_depth_0_atomic_inner(self):
        """Treat inner containers as atomic at depth 0."""
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
        ids=["ascii", "unicode"],
    )
    def test_max_items_appends_ellipsis(self, style, expected_more):
        """Append ellipsis when exceeding max items."""
        out = fmt_sequence(list(range(5)), style=style, max_items=3)
        # Expect 3 items then the ellipsis token
        assert out.endswith(expected_more + "]") or out.endswith(expected_more + ")")
        assert "<int: 0>" in out or "⟨int: 0⟩" in out

    def test_custom_ellipsis_propagates(self):
        """Propagate custom ellipsis token."""
        out = fmt_sequence(list(range(5)), style="ascii", max_items=2, ellipsis=" [more] ")
        assert out.endswith(" [more] ]")

    def test_extreme_max_items_limits(self):
        """Handle extreme max_items values."""
        seq = [1, 2, 3]

        # Zero items - should show ellipsis only
        out = fmt_sequence(seq, max_items=0)
        assert out == "[...]" or out == "[…]"

        # Very large max_items should work
        out = fmt_sequence(seq, max_items=1000)
        assert "<int: 1>" in out and "<int: 2>" in out and "<int: 3>" in out

    # ---------- Special sequence types ----------

    def test_range_object(self):
        """Format range objects."""
        r = range(3, 8, 2)
        out = fmt_sequence(r, style="ascii")
        assert "<int: 3>" in out
        assert "<int: 5>" in out
        assert "<int: 7>" in out

    def test_deque(self):
        """Format deque like a list."""
        from collections import deque

        d = deque([1, 2, 3])
        out = fmt_sequence(d, style="ascii")
        assert "<int: 1>" in out
        assert "<int: 2>" in out
        assert "<int: 3>" in out

    # ---------- Parameter validation (defensive) ----------

    def test_negative_max_items(self):
        """Handle negative max_items gracefully."""
        seq = [1, 2, 3]
        out = fmt_sequence(seq, max_items=-1)
        # Should handle gracefully
        assert "[" in out and "]" in out

    def test_huge_elements_truncated(self):
        """Truncate very large element representations."""
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
        [42, "a string", ValueError("test"), AnyClass()],
        ids=["instance-int", "instance-str", "instance-exception", "instance-custom"],
    )
    def test_fmt_type_basic_instance_input(self, obj):
        """Test that fmt_type correctly formats the type of an instance."""
        expected = f"<{type(obj).__name__}>"
        assert fmt_type(obj) == expected

    @pytest.mark.parametrize(
        "obj_type",
        [int, str, ValueError, AnyClass],
        ids=["type-int", "type-str", "type-exception", "type-custom"],
    )
    def test_fmt_type_basic_type_input(self, obj_type):
        """Test that fmt_type correctly formats a type object directly."""
        expected = f"<{obj_type.__name__}>"
        assert fmt_type(obj_type) == expected

    @pytest.mark.parametrize(
        "style, expected_format",
        [
            ("ascii", "<{name}>"),
            ("unicode-angle", "⟨{name}⟩"),
            ("equal", "{name}"),
        ],
        ids=["ascii", "unicode-angle", "equal"],
    )
    def test_fmt_type_different_styles(self, style, expected_format):
        """Test various formatting styles."""
        name = AnyClass.__name__
        expected = expected_format.format(name=name)
        assert fmt_type(AnyClass, style=style) == expected

    def test_fmt_type_fully_qualified_flag(self):
        """Test the 'fully_qualified' flag for built-in and custom types."""
        # For a custom class, it should show the module name.
        expected_name = f"{AnyClass.__module__}.{AnyClass.__name__}"
        assert fmt_type(AnyClass, fully_qualified=True) == f"<{expected_name}>"

        # For a built-in type, 'builtins' should be omitted.
        assert fmt_type(list, fully_qualified=True) == "<list>"

    def test_fmt_type_truncation(self):
        """Test that long type names are truncated correctly."""

        class ThisIsAVeryLongClassNameForTestingPurposes:
            pass

        out = fmt_type(ThisIsAVeryLongClassNameForTestingPurposes, max_repr=20, style="ascii")
        assert out.startswith("<ThisIsAVeryLongClass")
        assert out.endswith("...>")

    def test_fmt_type_truncation_with_custom_ellipsis(self):
        """Test truncation with a custom ellipsis token."""

        class AnotherLongName:
            pass

        out = fmt_type(AnotherLongName, max_repr=10, ellipsis="...[more]", style="ascii")
        assert out == "<AnotherLon...[more]>"

    def test_fmt_type_with_broken_name_attribute(self):
        """Test graceful fallback for types with a broken __name__."""

        class MetaWithBrokenName(type):
            @property
            def __name__(cls):
                raise AttributeError("Name is deliberately broken")

        class MyBrokenType(metaclass=MetaWithBrokenName):
            pass

        out = fmt_type(MyBrokenType, style="ascii")
        assert out.startswith("<<class '")
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
        ids=["equal", "paren", "colon", "unicode-angle", "ascii"],
    )
    def test_fmt_value_styles(self, style, value, expected):
        """Format value using basic styles."""
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
        ids=["ascii", "unicode-angle"],
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
