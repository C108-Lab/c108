#
# C108 - Formatters Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import array
import collections
import re
import reprlib

from dataclasses import dataclass
from unittest.mock import Mock


# Third Party ----------------------------------------------------------------------------------------------------------
import pytest
from frozendict import frozendict

# Local ----------------------------------------------------------------------------------------------------------------
from c108.formatters import (
    FmtOptions,
    fmt_any,
    fmt_exception,
    fmt_mapping,
    fmt_repr,
    fmt_set,
    fmt_sequence,
    fmt_type,
    fmt_value,
    _fmt_repr,
)


# Tests ----------------------------------------------------------------------------------------------------------------
class AnyClass:
    """
    A simple class for testing user-defined types
    """


@dataclass(frozen=True)
class Frozen:
    a: int = 0
    b: float = 1


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
            # Set Dispatch
            ({123, 456, 789}, "}"),
            ({123, 456, 789}, "<int: 789>"),
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
        result = fmt_any(obj, style="angle", depth=1, label_primitives=True)
        assert expected_substring in result

    @pytest.mark.parametrize("style", ["angle", "unicode-angle"])
    def test_style_forwarding(self, style):
        """Style is forwarded to formatters."""
        exc_result = fmt_any(ValueError("test"), style=style)
        dict_result = fmt_any({"key": "val"}, style=style)
        list_result = fmt_any([1, 2], style=style, label_primitives=True)
        value_result = fmt_any("text", style=style, label_primitives=True)

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

        str_result = fmt_any(text_str, style="angle", label_primitives=True)
        bytes_result = fmt_any(text_bytes, style="angle", label_primitives=True)
        bytearray_result = fmt_any(text_bytearray, style="angle", label_primitives=True)

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
        none_result = fmt_any(None, label_primitives=True)
        assert "NoneType" in none_result

        empty_dict_result = fmt_any({})
        empty_list_result = fmt_any([])
        empty_tuple_result = fmt_any(())

        assert "{}" in empty_dict_result
        assert "[]" in empty_list_result
        assert "()" in empty_tuple_result

        complex_empty = {"empty_list": [], "empty_dict": {}}
        complex_result = fmt_any(complex_empty, depth=1)
        assert "empty_list" in complex_result
        assert "empty_dict" in complex_result


class TestFmtException:
    @pytest.mark.parametrize(
        "exc,expected",
        [
            (ValueError("invalid input"), "<ValueError: invalid input>"),
            (123, "123"),
            (RuntimeError(), "<RuntimeError>"),
        ],
    )
    def test_basic_and_empty(self, exc, expected):
        """Format exceptions with and without message."""
        result = fmt_exception(exc, style="angle")
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
        assert fmt_exception(exc, style="angle") == expected

    def test_unicode(self):
        """Handle unicode in exception message."""
        exc = ValueError("Error with unicode: 🚨 αβγ " + "0123456789" * 10)
        assert (
            fmt_exception(exc, style="unicode-angle", max_repr=52)
            == "⟨ValueError: Error with unicode: 🚨 αβγ 01234567890…⟩"
        )

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
            out = fmt_exception(e, max_repr=max_repr, ellipsis=ellipsis, style="angle")
            assert out.startswith(starts_with)
            assert out.endswith(ends_with)
            # Ensure truncation actually happened (shorter than message + overhead)
            assert len(out) < len(f"<ValueError: {msg}>")

    @pytest.mark.parametrize("style", ["angle", "unicode-angle", "equal"])
    def test_style(self, style):
        """Apply selected style to exception formatting."""
        # Style actually affects formatting, so test each style individually
        exc = ValueError("test message")
        out = fmt_exception(exc, style=style)

        # All styles should contain the type and message
        assert "ValueError" in out
        assert "test message" in out

        # Check style-specific formatting
        if style == "angle":
            assert out == "<ValueError: test message>"
        elif style == "unicode-angle":
            assert out == "⟨ValueError: test message⟩"
        elif style == "equal":
            assert out == "ValueError=test message"

    @pytest.mark.parametrize("max_repr", [0, 1, 5])
    def test_max_repr_edges(self, max_repr):
        """Constrain output length for small max_repr."""
        exc = ValueError("short")
        out = fmt_exception(exc, max_repr=max_repr, style="angle")
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
            out = fmt_exception(e, include_traceback=True, style="angle")
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
            out = fmt_exception(e, include_traceback=False, style="angle")
            assert out == "<ValueError: no tb>"

    def test_broken_str(self):
        """Fallback when __str__ raises inside exception."""

        class BrokenStrError(Exception):
            def __str__(self):
                raise RuntimeError("boom")

        exc = BrokenStrError("test message")
        # Should not raise; should fall back gracefully to the exception type
        out = fmt_exception(exc, opts=FmtOptions(style="angle"))
        assert out.startswith("<BrokenStrError")
        assert out.endswith(">")
        assert out == "<BrokenStrError: <repr failed>>"

    @pytest.mark.parametrize(
        "exc, style, max_repr, expected_sub",
        [
            pytest.param(ValueError("x" * 50), "equal", 10, "ValueError=...", id="equal_trunc"),
            pytest.param(
                ValueError("y" * 50), "unicode-angle", 10, "⟨ValueError: …⟩", id="unicode_trunc"
            ),
        ],
    )
    def test_truncation_branches(self, exc, style, max_repr, expected_sub):
        """Cover truncation branches for equal and unicode styles."""
        result = fmt_exception(exc, style=style, max_repr=max_repr)
        assert expected_sub in result

    @pytest.mark.parametrize(
        "exc, style, expected_sub",
        [
            pytest.param(RuntimeError(), "equal", "RuntimeError", id="equal_nomsg"),
            pytest.param(RuntimeError(), "unicode-angle", "⟨RuntimeError⟩", id="unicode_nomsg"),
        ],
    )
    def test_no_message_styles(self, exc, style, expected_sub):
        """Cover no-message branches for equal and unicode styles."""
        result = fmt_exception(exc, style=style)
        assert expected_sub in result

    def test_include_traceback_equal_and_fail_safe(self):
        """Cover traceback inclusion for equal style and fallback on failure."""

        def inner_func():
            raise ValueError("traceback test")

        try:
            inner_func()
        except ValueError as e:
            result = fmt_exception(e, style="equal", include_traceback=True)
            assert "ValueError" in result
            assert "traceback test" in result
            assert " at " in result

        # Simulate broken traceback attribute to trigger exception handling
        class BrokenExc(Exception):
            @property
            def __traceback__(self):
                raise RuntimeError("broken tb")

        broken = BrokenExc("fail tb")
        result = fmt_exception(broken, include_traceback=True)
        assert "BrokenExc" in result
        assert "fail tb" in result


class TestFmtMapping:
    # ---------- Basic functionality ----------

    def test_basic(self):
        """Format a simple mapping."""
        mp = {"a": 1, 2: "b"}
        opts = FmtOptions(style="angle", label_primitives=True)
        out = fmt_mapping(mp, opts=opts)
        # Insertion order preserved by dicts
        assert out == "{<str: 'a'>: <int: 1>, <int: 2>: <str: 'b'>}"

    def test_nested_sequence(self):
        """Format mapping containing a nested sequence."""
        mp = {"k": [1, 2]}
        opts = FmtOptions(style="unicode-angle", label_primitives=True).merge(max_depth=2)
        out = fmt_mapping(mp, opts=opts)
        assert out == "{⟨str: 'k'⟩: [⟨int: 1⟩, ⟨int: 2⟩]}"

    def test_nested_set(self):
        """Format mapping containing a nested sequence."""
        mp = {"k": {1, 2}}
        opts = FmtOptions(style="unicode-angle", label_primitives=True).merge(max_depth=2)
        out = fmt_mapping(mp, opts=opts)
        assert out == "{⟨str: 'k'⟩: {⟨int: 1⟩, ⟨int: 2⟩}}"

    # ---------- Edge cases critical for exceptions/logging ----------

    def test_empty(self):
        """Handle empty dicts."""
        assert fmt_mapping({}) == "{}"

    def test_none_keys_and_values(self):
        """Format mappings with None keys and values."""
        mp = {None: "value", "key": None, None: None}
        opts = FmtOptions(style="angle", label_primitives=True)
        out = fmt_mapping(mp, opts=opts)
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
        opts = FmtOptions(style="angle", label_primitives=True)
        out = fmt_mapping(mp, opts=opts)
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
        out = fmt_mapping(mp)
        # Should handle gracefully
        assert "BrokenKeyRepr instance at" in out
        assert "value" in out

    def test_broken_value_repr(self):
        """Handle values whose __repr__ raises."""

        class BrokenValueRepr:
            def __repr__(self):
                raise RuntimeError("Value repr is broken!")

        mp = {"key": BrokenValueRepr()}
        out = fmt_mapping(mp, opts=FmtOptions(style="angle"))
        assert "key" in out
        assert "<BrokenValueRepr:" in out
        assert "BrokenValueRepr instance at" in out

    def test_large_mapping_truncate(self):
        """Truncate very large mappings."""
        big_dict = {f"key_{i}": f"value_{i}" for i in range(20)}
        out = fmt_mapping(big_dict, opts=FmtOptions(style="angle").merge(max_items=3))
        # Should only show 3 items plus ellipsis
        key_count = out.count("key_")
        assert key_count == 3
        assert "..." in out

    def test_deeply_nested(self):
        """Respect depth limits for nested structures."""
        mp = {1: "a"}
        mp[2] = mp  # Add circular reference

        # With depth=2, should recurse into level2 but treat level3+ as atomic
        out = fmt_mapping(mp, opts=FmtOptions(style="angle").merge(max_depth=2))
        assert out == "{1: 'a', 2: {1: 'a', 2: <dict: {...}>}}"

    def test_circular_references(self):
        """Handle circular references without infinite recursion."""
        d = {"a": 1}
        d["self"] = d  # Create circular reference

        out = fmt_mapping(d)
        # Should handle gracefully without infinite recursion
        assert "a" in out
        assert "self" in out
        assert "..." in out or "{" in out  # Circular part shown somehow

    # ---------- Truncation robustness ----------

    @pytest.mark.parametrize(
        "style",
        [
            ("angle"),
            ("arrow"),
        ],
        ids=["angle", "unicode-angle"],
    )
    def test_max_items_appends_ellipsis(self, style):
        """Append an ellipsis when max_items is exceeded."""
        mp = {i: i for i in range(5)}
        out = fmt_mapping(mp, opts=FmtOptions(style=style).merge(max_items=3))
        assert "..." in out

    def test_extreme_max_items(self):
        """Handle edge cases for max_items limits."""
        mp = {"a": 1, "b": 2}

        # Zero items - should show ellipsis only
        out = fmt_mapping(mp, opts=FmtOptions().merge(max_items=0))
        assert out == "{...}" or out == "{…}"

        # One item
        out = fmt_mapping(
            mp,
            opts=FmtOptions(style="angle", label_primitives=True).merge(max_items=1, max_depth=1),
        )
        item_count = out.count("<")
        assert item_count >= 2  # At least one key and one value

    # ---------- Special mapping types ----------

    def test_defaultdict(self):
        """Format defaultdict like a regular dict."""
        from collections import defaultdict

        dd = defaultdict(str)
        for i in range(100):
            dd[str(i)] = i
        out = fmt_mapping(
            dd, opts=FmtOptions(style="angle", label_primitives=True).merge(max_items=2)
        )
        assert out == "defaultdict({<str: '0'>: <int: 0>, <str: '1'>: <int: 1>, ...}})"

    def test_frozendict(self):
        """Format defaultdict like a regular dict."""
        from collections import defaultdict

        dd = defaultdict(str)
        for i in range(100):
            dd[str(i)] = i
        fd = frozendict(dd)
        out = fmt_mapping(
            fd, opts=FmtOptions(style="angle", label_primitives=True).merge(max_items=2)
        )
        assert out == "frozendict({<str: '0'>: <int: 0>, <str: '1'>: <int: 1>, ...}})"

    def test_ordered_dict(self):
        """Preserve order for OrderedDict."""
        from collections import OrderedDict

        od = OrderedDict([("first", 1), ("second", 2)])
        out = fmt_mapping(od)
        # Should show first before second
        first_pos = out.find("first")
        second_pos = out.find("second")
        assert first_pos < second_pos

    def test_textual_values_atomic(self):
        """Treat text-like values as atomic."""
        mp = {"s": "xyz", "b": b"ab", "ba": bytearray(b"test")}
        out = fmt_mapping(mp, opts=FmtOptions(style="paren", label_primitives=True))
        assert "str('xyz')" in out
        assert "bytes(b'ab')" in out
        assert "bytearray(" in out

    # ---------- Parameter validation (defensive) ----------

    def test_invalid_mapping_type(self):
        """Handle non-mapping inputs gracefully or raise clear error."""
        try:
            out = fmt_mapping("not a mapping", style="angle")  # type: ignore
            # If it doesn't raise, should produce some reasonable output
            assert "str" in out or "not a mapping" in out
        except (TypeError, AttributeError) as e:
            # Acceptable to raise clear error for invalid input
            assert "mapping" in str(e).lower() or "items" in str(e).lower()

    def test_negative_max_items(self):
        """Accept negative max_items without crashing."""
        mp = {"a": 1}
        out = fmt_mapping(mp, opts=FmtOptions().merge(max_items=-1))
        # Should handle gracefully
        assert "{" in out and "}" in out

    def test_huge_individual_values(self):
        """Truncate very large individual values."""
        huge_value = "x" * 1000
        mp = {"key": huge_value}
        out = fmt_mapping(mp, opts=FmtOptions(style="angle").merge(max_str=20))
        # Value should be truncated
        assert len(out) < 200  # Much shorter than the huge value
        assert "..." in out or "…" in out


class TestFmtOptions:
    """Core tests for FmtOptions."""

    def test_defaults_are_sensible(self):
        """FmtOptions defaults to reasonable values."""
        opts = FmtOptions()
        assert opts.deduplicate_types is False
        assert opts.fully_qualified is False
        assert opts.include_traceback is False
        assert opts.label_primitives is False
        assert opts.style == "repr"
        assert 2 <= opts.repr.maxlevel <= 10
        min_items, max_items = (2, 10)
        assert min_items <= opts.repr.maxtuple <= max_items
        assert min_items <= opts.repr.maxlist <= max_items
        assert min_items <= opts.repr.maxarray <= max_items
        assert min_items <= opts.repr.maxdict <= max_items
        assert min_items <= opts.repr.maxset <= max_items
        assert min_items <= opts.repr.maxfrozenset <= max_items
        assert min_items <= opts.repr.maxdeque <= max_items
        assert 80 <= opts.repr.maxstring <= 210
        assert 80 <= opts.repr.maxother <= 210

    def test_deduplicate_types(self):
        """Cast deduplicate_types to bool."""
        opts = FmtOptions(deduplicate_types=1)
        assert opts.deduplicate_types is True
        opts = FmtOptions(deduplicate_types=None)
        assert opts.deduplicate_types is False

    def test_fully_qualified(self):
        """Cast fully_qualified to bool."""
        opts = FmtOptions(fully_qualified=1)
        assert opts.fully_qualified is True
        opts = FmtOptions(fully_qualified=None)
        assert opts.fully_qualified is False

    def test_include_traceback(self):
        """Cast include_traceback to bool."""
        opts = FmtOptions(include_traceback=1)
        assert opts.include_traceback is True
        opts = FmtOptions(include_traceback=None)
        assert opts.include_traceback is False

    def test_label_primitives(self):
        """Cast label_primitives to bool."""
        opts = FmtOptions(label_primitives=1)
        assert opts.label_primitives is True
        opts = FmtOptions(label_primitives=None)
        assert opts.label_primitives is False

    def test_repr_type_validation(self):
        """Reject non-Repr repr argument."""
        with pytest.raises(TypeError, match=r"(?i).*reprlib\.Repr.*"):
            FmtOptions(repr="not-a-repr")

    @pytest.mark.parametrize(
        "style",
        [
            pytest.param("angle", id="angle"),
            pytest.param("arrow", id="arrow"),
            pytest.param("braces", id="braces"),
            pytest.param("colon", id="colon"),
            pytest.param("equal", id="equal"),
            pytest.param("paren", id="paren"),
            pytest.param("repr", id="repr"),
            pytest.param("unicode-angle", id="unicode-angle"),
        ],
    )
    def test_style_accept_valid(self, style):
        """Accept valid styles."""
        opts = FmtOptions(style=style)
        assert opts.style == style

    def test_style_fallback(self):
        """Fallback from invalid style."""
        assert FmtOptions(style="not-a-style").style == FmtOptions().style

    def test_merge_selective(self):
        """Update multiple fields in merge."""
        r = reprlib.Repr(fillvalue="###")
        old = FmtOptions(
            deduplicate_types=True,
            fully_qualified=True,
            include_traceback=True,
            label_primitives=True,
            style="angle",
            repr=r,
        )
        new = old.merge(style="colon")
        assert new.deduplicate_types is True
        assert new.fully_qualified is True
        assert new.include_traceback is True
        assert new.label_primitives is True
        assert new.style == "colon"
        assert new.repr is not old.repr
        assert new.repr is not r
        assert new.repr.fillvalue == "###"

    def test_merge_max_depth_items_str(self):
        """Update multiple fields in merge."""
        max_depth = 763547223
        max_items = 387468734
        max_str = 376236453
        opts = FmtOptions().merge(max_depth=max_depth, max_items=max_items, max_str=max_str)
        assert opts.repr.maxlist == max_items
        assert opts.repr.maxtuple == max_items
        assert opts.repr.maxdict == max_items
        assert opts.repr.maxset == max_items
        assert opts.repr.maxfrozenset == max_items
        assert opts.repr.maxdeque == max_items
        assert opts.repr.maxlevel == max_depth
        assert opts.repr.maxstring == max_str
        assert opts.repr.maxother == max_str

    def test_merge_reject_non_repr(self):
        """Reject invalid repr in merge."""
        base = FmtOptions(style="angle")
        with pytest.raises(ValueError, match=r"(?i).*reprlib\.Repr.*"):
            base.merge(repr="not-a-repr")

    def test_compact_minimal(self):
        """Produce minimal preset for compact()."""
        c = FmtOptions.compact()
        assert isinstance(c, FmtOptions)
        assert isinstance(c.repr, reprlib.Repr)
        assert c.repr.maxlevel == 2

    def test_debug_verbose(self):
        """Produce verbose preset for debug()."""
        d = FmtOptions.debug()
        assert isinstance(d, FmtOptions)
        # Debug likely enables label_primitives=True, adjust as needed.
        # Ensure type correctness:
        assert isinstance(d.repr, reprlib.Repr)
        assert d.repr.maxlevel >= 5
        assert d.label_primitives == True
        assert d.include_traceback == True

    def test_logging_balanced(self):
        """Produce balanced preset for logging()."""
        lg = FmtOptions.logging()
        assert isinstance(lg, FmtOptions)
        assert isinstance(lg.repr, reprlib.Repr)
        assert lg.repr.maxlevel == 3

    def test_property_max_depth_items_str(self):
        max_depth = 276457625123
        max_items = 98437345224
        max_str = 98798798336
        r = reprlib.Repr(maxlevel=max_depth, maxlist=max_items, maxstring=max_str)
        opts = FmtOptions(repr=r)
        assert opts.max_depth == r.maxlevel
        assert opts.max_items == r.maxlist
        assert opts.max_str == r.maxstring

    def test_compact_debug_logging(self):
        max_items = 98437345
        max_depth = 123

        opts = FmtOptions.compact(max_items=max_items, max_depth=max_depth)
        assert opts.repr.maxlist == max_items
        assert opts.repr.maxlevel == max_depth
        assert opts.repr.maxstring == 64
        assert opts.repr.maxother == 64

        opts = FmtOptions.debug(max_items=max_items, max_depth=max_depth)
        assert opts.repr.maxlist == max_items
        assert opts.repr.maxlevel == max_depth
        assert opts.repr.maxstring == 1024
        assert opts.repr.maxother == 1024

        opts = FmtOptions.logging(max_items=max_items, max_depth=max_depth)
        assert opts.repr.maxlist == max_items
        assert opts.repr.maxlevel == max_depth
        assert opts.repr.maxstring == 128
        assert opts.repr.maxother == 128


class TestFmtRepr:
    @pytest.mark.parametrize(
        "obj, expected_fmt",
        [
            pytest.param("a_string", "'a_string'", id="str"),
            pytest.param(b"bytes", "b'bytes'", id="bytes"),
            pytest.param(bytearray(b"long"), "bytearray(b'long')", id="bytearray"),
            pytest.param([1, 2, 3], "[1, 2, 3]", id="list"),
            pytest.param({1: 2}, "{1: 2}", id="dict"),
            pytest.param({1, 2}, "{1, 2}", id="set"),
            pytest.param(frozenset({1}), "frozenset({1})", id="frozenset"),
            pytest.param((1, 2), "(1, 2)", id="tuple_multi"),
            pytest.param((1,), "(1,)", id="tuple_singleton"),
            pytest.param(range(100), "range(0, 100)", id="range"),
        ],
    )
    def test_fmt_repr_basic(self, obj, expected_fmt):
        """Verify ellipsis wrapping logic for builtin Python types."""
        opts = FmtOptions()
        assert fmt_repr(obj, opts=opts) == expected_fmt

    @pytest.mark.parametrize(
        "obj, expected_fmt",
        [
            pytest.param("long_string", "'...'", id="str"),
            pytest.param(b"long_bytes", "b'...'", id="bytes"),
            pytest.param(bytearray(b"long"), "bytearray(b'...')", id="bytearray"),
            pytest.param([1, 2, 3], "[...]", id="list"),
            pytest.param({1: 2}, "{...}", id="dict"),
            pytest.param({1, 2}, "{...}", id="set"),
            pytest.param(frozenset({1}), "frozenset({...})", id="frozenset"),
            pytest.param((1, 2), "(...)", id="tuple_multi"),
            pytest.param((1,), "(...,)", id="tuple_singleton"),
            pytest.param(range(100), "range(...)", id="range"),
        ],
    )
    def test_fmt_repr_builtins_ellipsis(self, obj, expected_fmt):
        """Verify ellipsis wrapping logic for builtin Python types."""
        opts = Mock()
        opts.repr.fillvalue = "..."
        opts.repr.repr.return_value = "..."

        # Explicitly disable fully qualified names to prevent Mock from returning True
        opts.fully_qualified = False
        opts.fully_qualified_builtins = False

        assert _fmt_repr(obj, opts) == expected_fmt

    @pytest.mark.parametrize(
        "obj, expected_fmt",
        [
            pytest.param(collections.deque([1]), "deque([...])", id="deque"),
            pytest.param(collections.OrderedDict(a=1), "OrderedDict({...})", id="ordered_dict"),
            pytest.param(collections.Counter(a=1), "Counter({...})", id="counter"),
            pytest.param(collections.ChainMap({}, {}), "ChainMap({...})", id="chain_map"),
            pytest.param(
                collections.defaultdict(int, {1: 1}),
                "defaultdict(type, {...})",
                id="defaultdict_int",
            ),
            pytest.param(
                collections.defaultdict(None, {1: 1}),
                "defaultdict(None, {...})",
                id="defaultdict_none",
            ),
            pytest.param(array.array("i", [1, 2]), "array('i', [...])", id="array"),
            pytest.param(memoryview(b"abc"), "memoryview(b'...')", id="memoryview_bytes"),
            pytest.param(
                memoryview(bytearray(b"abc")),
                "memoryview(bytearray(b'...'))",
                id="memoryview_bytearray",
            ),
            pytest.param(
                memoryview(array.array("i", [1])), "memoryview(...)", id="memoryview_generic"
            ),
        ],
    )
    def test_fmt_repr_stdlib_types_ellipsis(self, obj, expected_fmt):
        """Verify ellipsis wrapping logic for standard library collection types."""
        opts = Mock()
        opts.repr.fillvalue = "..."
        opts.repr.repr.return_value = "..."

        # Explicitly disable fully qualified names
        opts.fully_qualified = False
        opts.fully_qualified_builtins = False

        assert _fmt_repr(obj, opts) == expected_fmt

    def test_fmt_repr_exception_handling(self):
        """Ensure objects with broken __repr__ methods are handled defensively."""

        class BrokenRepr:
            def __repr__(self):
                raise ValueError("Simulated internal failure")

        obj = BrokenRepr()
        opts = Mock()
        opts.repr.repr.side_effect = ValueError("Simulated internal failure")

        # Explicitly disable fully qualified names
        opts.fully_qualified = False
        opts.fully_qualified_builtins = False

        result = _fmt_repr(obj, opts)

        # Verify fallback format: <Type instance at ID (repr failed: Error)>
        # With fully_qualified=False, this will match "BrokenRepr" instead of "tests....BrokenRepr"
        assert result.startswith("<BrokenRepr instance at")
        assert "(repr failed: ValueError)>" in result


class TestFmtSet:
    @pytest.mark.parametrize(
        "obj,expected_substring",
        [
            # Set ProcessDispatch
            pytest.param({123, 456}, "}", id="curly_braces"),
            pytest.param(42, "int: 42", id="int"),
            pytest.param({123, Frozen(a=1, b=2)}, "Frozen(a=1, b=2)", id="dataclass"),
            pytest.param(
                frozenset(range(100)),
                "frozenset({<int: 0>, <int: 1>, <int: 2>, ...})",
                id="oversized",
            ),
            # Value Dispatch
            pytest.param("hello", "str: 'hello'", id="str"),
            pytest.param(3.14, "float: 3.14", id="float"),
            pytest.param(True, "bool: True", id="bool"),
        ],
    )
    def test_fmt_set(self, obj, expected_substring):
        """Format sets."""
        opts = FmtOptions(style="angle", label_primitives=True).merge(max_depth=1, max_items=3)
        result = fmt_set(obj, opts=opts)
        assert expected_substring in result

    def test_fmt_frozen_set(self):
        """Format frozenset."""

        st = frozenset({1, 2, 3})
        opts = FmtOptions(style="angle", label_primitives=True).merge(max_depth=1, max_items=3)
        out = fmt_set(st, opts=opts)
        assert out == "frozenset({<int: 1>, <int: 2>, <int: 3>})"

    def test_fmt_custom_set(self):
        """Format custom set."""

        class CustomSet(set):
            pass

        st = CustomSet({1, 2, 3})
        opts = FmtOptions(style="angle", label_primitives=True).merge(max_depth=1, max_items=3)
        out = fmt_set(st, opts=opts)
        assert out == "CustomSet({<int: 1>, <int: 2>, <int: 3>})"


class TestFmtSequence:
    # ---------- Basic functionality ----------
    @pytest.mark.parametrize(
        "seq, style, expected",
        [
            pytest.param([1, "a"], "angle", "[<int: 1>, <str: 'a'>]", id="list"),
            pytest.param((1, "a"), "angle", "(<int: 1>, <str: 'a'>)", id="tuple"),
            pytest.param(range(10), "angle", "range(0, 10)", id="range"),
            pytest.param(
                collections.deque([1, "a"]), "angle", "deque([<int: 1>, <str: 'a'>])", id="deque"
            ),
        ],
    )
    def test_builtins(self, seq, style, expected):
        """Format buuiltins."""
        out = fmt_sequence(seq, opts=FmtOptions(style=style, label_primitives=True))
        assert out == expected

    def test_sets(self):
        """Format buuiltins."""
        opts = FmtOptions(style="angle", label_primitives=True)
        set1 = fmt_sequence({1, 2}, opts=opts)
        set2 = fmt_sequence(frozenset({1, 2}), opts=opts)
        assert set1 in ["{<int: 1>, <int: 2>}", "{<int: 2>, <int: 1>}"]
        assert set2 in ["frozenset({<int: 1>, <int: 2>})", "frozenset({<int: 2>, <int: 1>})"]

    def test_sequence_custom(self):
        """Format custom sequence."""

        class CustomList(list):
            pass

        opts = FmtOptions(style="angle", label_primitives=True).merge(max_items=4)

        custom_list = fmt_sequence(CustomList(range(10)), opts=opts)
        assert custom_list == "CustomList([<int: 0>, <int: 1>, ..., <int: 8>, <int: 9>])"

        custom_list = fmt_sequence(CustomList(range(4)), opts=opts)
        assert custom_list == "CustomList([<int: 0>, <int: 1>, <int: 2>, <int: 3>])"

    def test_singleton_tuple_trailing_comma(self):
        """Show trailing comma for singleton tuple."""
        out = fmt_sequence((1,), opts=FmtOptions(style="angle", label_primitives=True))
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
        out = fmt_sequence(seq, opts=FmtOptions(style="angle", label_primitives=True))
        assert "<int: 1>" in out
        assert "<NoneType: None>" in out
        assert "<str: 'hello'>" in out

    def test_non_iterable_fallback(self):
        """Fallback to fmt_value for non-iterables."""
        out = fmt_sequence(42, opts=FmtOptions(style="angle", label_primitives=True))
        # Should fall back to fmt_value behavior for non-iterables
        assert out == "<int: 42>"

    def test_mixed_types(self):
        """Format realistic mix of element types."""
        mixed = [42, "status", None, {"error": True}, [1, 2]]
        opts = FmtOptions(style="angle", label_primitives=True).merge(max_depth=2)
        out = fmt_sequence(mixed, opts=opts)
        assert (
            out
            == "[<int: 42>, <str: 'status'>, <NoneType: None>, {<str: 'error'>: <bool: True>}, [<int: 1>, <int: 2>]]"
        )

    def test_broken_element_repr(self):
        """Handle elements with broken __repr__."""

        class BrokenRepr:
            def __repr__(self):
                raise RuntimeError("Element repr is broken!")

        seq = [1, BrokenRepr(), "after"]
        out = fmt_sequence(
            seq, opts=FmtOptions(style="angle", label_primitives=True, deduplicate_types=True)
        )
        assert "<int: 1>" in out
        assert "BrokenRepr:" in out
        assert "BrokenRepr instance at" in out
        assert "<str: 'after'>" in out

    def test_large_list_truncation(self):
        """Truncate large sequences."""
        big_list = list(range(50))
        opts = FmtOptions(style="angle", label_primitives=True).merge(max_items=3)
        out = fmt_sequence(big_list, opts=opts)
        # Should only show 3 items plus ellipsis
        item_count = out.count("<int:")
        assert item_count == 3
        assert "..." in out

    def test_deep_nesting(self):
        """Limit recursion depth in nested structures."""
        nested = [1, [2, [3, [4, [5]]]]]

        # With depth=2, should recurse 2 levels but treat deeper as atomic
        opts = FmtOptions(style="angle", label_primitives=True).merge(max_depth=3)
        out = fmt_sequence(nested, opts=opts)
        assert "[<int: 1>" in out  # First level of nesting
        assert "[<int: 2>" in out
        assert "[<int: 3>" in out  # Max level of nesting
        # Deeper nesting should be atomic
        assert "<list: [...]>" in out

    def test_circular_references(self):
        """Handle circular references safely."""
        lst = [1, 2]
        lst.append(lst)  # Create circular reference: [1, 2, [...]]

        out = fmt_sequence(lst, opts=FmtOptions(style="angle", label_primitives=True))
        # Should handle gracefully without infinite recursion
        assert "<int: 1>" in out
        assert "<int: 2>" in out
        assert "..." in out or "[" in out  # Circular part shown somehow

    def test_generators_and_iterators(self):
        """Consume generators and iterators."""

        def gen():
            yield 1
            yield 2
            yield 3

        opts = FmtOptions(style="angle", label_primitives=True).merge(max_items=2)
        out = fmt_sequence(gen(), opts=opts)
        # Should consume generator and show first 2 items
        assert out == "generator([<int: 1>, <int: 2>, ...])"

    def test_iterator_truncation(self):
        class Counter:
            def __iter__(self):
                current = 1
                while True:
                    yield current
                    current += 1

        opts = FmtOptions(style="angle", label_primitives=True).merge(max_items=3)
        out = fmt_sequence(iter(Counter()), opts=opts)

        assert out == "generator([<int: 1>, <int: 2>, <int: 3>, ...])"

    def test_sets_unordered(self):
        """Format sets without relying on order."""
        s = {3, 1, 2}
        out = fmt_sequence(s, opts=FmtOptions(style="angle", label_primitives=True))
        assert out.startswith("{")
        assert out.endswith("}")
        # Should contain all elements (order may vary)
        assert "<int: 1>" in out
        assert "<int: 2>" in out
        assert "<int: 3>" in out

    # ---------- String/textual handling ----------

    def test_string_is_atomic(self):
        """Treat strings as atomic values."""
        out = fmt_sequence("abc", opts=FmtOptions(style="colon", label_primitives=True))
        assert out == "str: 'abc'"
        # Should NOT be ['a', 'b', 'c']

    def test_bytes_is_atomic(self):
        """Treat bytes as atomic values."""
        out = fmt_sequence(b"hello", opts=FmtOptions(style="angle", label_primitives=True))
        assert out == "<bytes: b'hello'>"

    def test_bytearray_is_atomic(self):
        """Treat bytearray as atomic value."""
        ba = bytearray(b"test")
        out = fmt_sequence(ba, opts=FmtOptions(style="angle", label_primitives=True))
        assert out.startswith("<bytearray:")

    def test_unicode_strings(self):
        """Preserve or safely escape Unicode."""
        unicode_seq = ["Hello", "世界", "🌍"]
        out = fmt_sequence(unicode_seq, opts=FmtOptions(style="angle", label_primitives=True))
        assert "Hello" in out
        # Unicode should be preserved or safely escaped
        assert "世界" in out or "\\u" in out
        assert "🌍" in out or "\\u" in out

    # ---------- Truncation robustness ----------

    def test_nesting_depth_1(self):
        """Format inner containers at depth >=1"""
        seq = [1]
        seq.append(seq)  # Create recursion: [1, [...]]
        opts = FmtOptions().merge(max_depth=1)
        out = fmt_sequence(seq, opts=opts)
        assert out == "[1, [1, [...]]]"

    def test_nesting_depth_0(self):
        """Format inner containers at depth 0."""
        seq = [0]
        seq.append(seq)
        opts = FmtOptions().merge(max_depth=0)
        out = fmt_sequence(seq, opts=opts)
        assert out == "[0, [...]]"

    def test_nesting_depth_negative(self):
        """Format inner containers at depth <0."""
        seq = [-100]
        seq.append(seq)
        opts = FmtOptions().merge(max_depth=-100)
        out = fmt_sequence(seq, opts=opts)
        assert out == "[-100, [...]]"

    @pytest.mark.parametrize(
        "style, expected_more",
        [
            ("angle", "..."),
            ("unicode-angle", "…"),
        ],
        ids=["angle", "unicode"],
    )
    def test_max_items_uses_ellipsis(self, style, expected_more):
        """Insert ellipsis when exceeding max items."""
        opts = FmtOptions(style=style, label_primitives=True).merge(max_items=4)
        out = fmt_sequence(list(range(5)), opts=opts)
        assert opts.ellipsis in out
        assert "int: 0" in out
        assert "int: 1" in out

    def test_custom_ellipsis_propagates(self):
        """Propagate custom ellipsis token."""
        r = FmtOptions().repr
        r.fillvalue = " [more] "
        opts = FmtOptions(style="angle").merge(max_items=4, repr=r)
        out = fmt_sequence(list(range(5)), opts=opts)
        assert out == "[0, 1,  [more] , 3, 4]"

    def test_extreme_max_items_limits(self):
        """Handle extreme max_items values."""
        seq = [1, 2, 3, 4]

        # Zero items - should show ellipsis only
        opts = FmtOptions().merge(max_items=0)
        out = fmt_sequence(seq, opts=opts)
        assert out == "[...]"
        opts = FmtOptions().merge(max_items=3)
        out = fmt_sequence(seq, opts=opts)
        assert out == "[1, 2, ..., 4]"

        # Very extra-large max_items should work
        opts = FmtOptions(label_primitives=True, style="angle").merge(max_depth=1, max_items=1000)
        out = fmt_sequence(seq, opts=opts)
        assert out == "[<int: 1>, <int: 2>, <int: 3>, <int: 4>]"

    # ---------- Special sequence types ----------

    def test_range_object(self):
        """Format range objects uses common repr."""
        r = range(7, 3, 21)
        out = fmt_sequence(r, opts=FmtOptions(style="angle", label_primitives=True))
        assert out == "range(7, 3, 21)"

    def test_deque(self):
        """Format deque like a list."""
        from collections import deque

        d = deque([1, 2, 3])
        out = fmt_sequence(d, opts=FmtOptions(style="angle", label_primitives=True))
        assert "<int: 1>" in out
        assert "<int: 2>" in out
        assert "<int: 3>" in out

    # ---------- Parameter validation (defensive) ----------

    def test_negative_max_items(self):
        """Handle negative max_items gracefully."""
        seq = [1, 2, 3]
        opts = FmtOptions().merge(max_items=-1)
        out = fmt_sequence(seq, opts=opts)
        # Should handle gracefully
        assert "[" in out and "]" in out

    def test_huge_elements_truncated(self):
        """Truncate very large element representations."""
        huge_str = "x" * 1000
        seq = ["small", huge_str, "small2"]
        opts = FmtOptions(style="angle").merge(max_str=20)
        out = fmt_sequence(seq, opts=opts)
        # Huge element should be truncated
        assert len(out) < 50  # Much shorter than the huge element
        assert "small" in out
        assert "..." in out


class TestFmtType:
    """Tests for the fmt_type() utility."""

    @pytest.mark.parametrize(
        "style, expected_format",
        [
            ("angle", "<{name}>"),
            ("arrow", "{name}"),
            ("braces", "{{name}}"),
            ("colon", "{name}"),
            ("equal", "{name}"),
            ("paren", "{name}"),
            ("repr", "{name}"),
            ("unicode-angle", "⟨{name}⟩"),
        ],
    )
    def test_fmt_type_styles(self, style, expected_format):
        """Test various formatting styles."""
        name = AnyClass.__name__
        expected = expected_format.replace("{name}", name)
        assert fmt_type(AnyClass(), opts=FmtOptions(style=style)) == expected

    @pytest.mark.parametrize(
        "obj,expected",
        [
            pytest.param(42, "<int>"),
            pytest.param("a string", "<str>"),
            pytest.param(ValueError("test"), "<ValueError>"),
            pytest.param(AnyClass(), "<AnyClass>"),
        ],
    )
    def test_fmt_type_instance(self, obj, expected):
        """Test that fmt_type correctly formats the type of an instance."""
        assert fmt_type(obj, opts=FmtOptions(style="angle")) == expected

    @pytest.mark.parametrize(
        "obj,expected",
        [
            pytest.param(int, "<class int>"),
            pytest.param(str, "<class str>"),
            pytest.param(ValueError, "<class ValueError>"),
            pytest.param(AnyClass, "<class AnyClass>"),
        ],
    )
    def test_fmt_type_class(self, obj, expected):
        """Test that fmt_type correctly formats a type object directly."""
        assert fmt_type(obj, opts=FmtOptions(style="angle")) == expected

    def test_fmt_type_fully_qualified_flag(self):
        """Test the 'fully_qualified' flag for built-in and custom types."""

        # For a custom class, it should show the module name.
        expected_name = f"{AnyClass.__module__}.{AnyClass.__name__}"
        assert (
            fmt_type(AnyClass(), opts=FmtOptions(fully_qualified=True, style="angle"))
            == f"<{expected_name}>"
        )

        # For a built-in type, 'builtins' should be omitted.
        assert fmt_type(list(), opts=FmtOptions(fully_qualified=True, style="angle")) == "<list>"

    def test_fmt_type_with_broken_name_attribute(self):
        """Test graceful fallback for types with a broken __name__."""

        class MetaWithBrokenName(type):
            @property
            def __name__(cls):
                raise AttributeError("Name is deliberately broken")

        class MyBrokenType(metaclass=MetaWithBrokenName):
            pass

        out = fmt_type(MyBrokenType(), opts=FmtOptions(style="angle"))
        assert out.startswith("<")
        assert "MyBrokenType" in out
        assert out.endswith(">")


class TestFmtValue:
    # ---------- Basic functionality ----------

    @pytest.mark.parametrize(
        "style, value, expected",
        [
            ("angle", 5, "<int: 5>"),
            ("arrow", 5, "int -> 5"),
            ("braces", 5, "{int: 5}"),
            ("colon", 5, "int: 5"),
            ("equal", 5, "int=5"),
            ("paren", 5, "int(5)"),
            ("repr", 5, "5"),
            ("unicode-angle", 5, "⟨int: 5⟩"),
        ],
    )
    def test_fmt_value_styles_primitives_labeled(self, style, value, expected):
        """Format value using basic styles."""
        assert fmt_value(value, opts=FmtOptions(style=style, label_primitives=True)) == expected

    @pytest.mark.parametrize(
        "label_primitives, value, expected",
        [
            (False, None, "None"),
            (False, True, "True"),
            (False, 5, "5"),
            (False, 1.23, "1.23"),
            (False, 3 + 4j, "(3+4j)"),
            (False, "abc", "'abc'"),
            (False, b"abc", "b'abc'"),
            (False, "123456789_" * 10, "'123456789_1234567...3456789_123456789_'"),
            (False, b"123456789_" * 10, "b'123456789_123456...3456789_123456789_'"),
            (
                False,
                bytearray(b"123456789_" * 10),
                "bytearray=bytearray(b'123456...456789_123456789_')",
            ),
            (False, Ellipsis, "Ellipsis"),
            (False, NotImplemented, "NotImplemented"),
            (True, None, "NoneType=None"),
            (True, True, "bool=True"),
            (True, 5, "int=5"),
            (True, 1.23, "float=1.23"),
            (True, 3 + 4j, "complex=(3+4j)"),
            (True, "abc", "str='abc'"),
            (True, b"abc", "bytes=b'abc'"),
            (True, "123456789_" * 10, "str='123456789_1234567...3456789_123456789_'"),
            (True, b"123456789_" * 10, "bytes=b'123456789_123456...3456789_123456789_'"),
            (
                True,
                bytearray(b"123456789_" * 10),
                "bytearray=bytearray(b'123456...456789_123456789_')",
            ),
            (True, Ellipsis, "ellipsis=Ellipsis"),
            (True, NotImplemented, "NotImplementedType=NotImplemented"),
        ],
    )
    def test_fmt_value_primitives(self, label_primitives, value, expected):
        """Format value using basic styles."""
        opts = FmtOptions.logging().merge(max_str=40)
        opts = opts.merge(style="equal", label_primitives=label_primitives)
        assert fmt_value(value, opts=opts) == expected

    @pytest.mark.parametrize(
        "style, expected",
        [
            pytest.param("angle", "<Obj: Obj(a=0, b='abc')>", id="angle"),
            pytest.param("arrow", "Obj -> Obj(a=0, b='abc')", id="arrow"),
            pytest.param("braces", "{Obj: Obj(a=0, b='abc')}", id="braces"),
            pytest.param("colon", "Obj: Obj(a=0, b='abc')", id="colon"),
            pytest.param("equal", "Obj=Obj(a=0, b='abc')", id="equal"),
            pytest.param("paren", "Obj(a=0, b='abc')", id="paren"),
            pytest.param("repr", "Obj(a=0, b='abc')", id="repr"),
            pytest.param("unicode-angle", "⟨Obj: Obj(a=0, b='abc')⟩", id="unicode-angle"),
        ],
    )
    def test_obj_styles_full(self, style, expected):
        """Format value using basic styles."""

        @dataclass
        class Obj:
            a: int = 0
            b: str = "abc"

        obj = Obj()
        opts = FmtOptions(style=style, deduplicate_types=False)
        assert fmt_value(obj, opts=opts) == expected

    @pytest.mark.parametrize(
        "style, expected",
        [
            pytest.param("angle", "<Obj: Obj(a=0, b='abc')>", id="angle"),
            pytest.param("arrow", "Obj -> Obj(a=0, b='abc')", id="arrow"),
            pytest.param("braces", "{Obj: Obj(a=0, b='abc')}", id="braces"),
            pytest.param("colon", "Obj: Obj(a=0, b='abc')", id="colon"),
            pytest.param("equal", "Obj=Obj(a=0, b='abc')", id="equal"),
            pytest.param("paren", "Obj(a=0, b='abc')", id="paren"),
            pytest.param("repr", "Obj(a=0, b='abc')", id="repr"),
            pytest.param("unicode-angle", "⟨Obj: Obj(a=0, b='abc')⟩", id="unicode-angle"),
        ],
    )
    def test_obj_styles_deduplicate(self, style, expected):
        """Format value using basic styles."""

        # TODO looks like deduplicate_types=True has NO effect at all

        @dataclass
        class Obj:
            a: int = 0
            b: str = "abc"

        obj = Obj()
        opts = FmtOptions(style=style, deduplicate_types=True)
        assert fmt_value(obj, opts=opts) == expected

    @pytest.mark.parametrize(
        "style, expected",
        [
            pytest.param("angle", "<defaultdict: defaultdict(<class 'int'>, {})>", id="angle"),
            pytest.param("arrow", "defaultdict -> defaultdict(<class 'int'>, {})", id="arrow"),
        ],
    )
    def test_defaultdict(self, style, expected):
        """Format value using basic styles."""
        obj = collections.defaultdict(int)
        opts = FmtOptions(style=style, deduplicate_types=False)
        assert fmt_value(obj, opts=opts) == expected

    @pytest.mark.parametrize(
        "style, expected_template",
        [
            # We add the expected suffix after '...'
            pytest.param("angle", "<BrokenRepr: BrokenRepr instance at ...>", id="angle"),
            pytest.param("arrow", "BrokenRepr -> <BrokenRepr instance at ...>", id="arrow"),
            pytest.param("braces", "{BrokenRepr: BrokenRepr instance at ...}", id="braces"),
            pytest.param("colon", "BrokenRepr: <BrokenRepr instance at ...>", id="colon"),
            pytest.param("equal", "BrokenRepr=<BrokenRepr instance at ...>", id="equal"),
            pytest.param("paren", "BrokenRepr(BrokenRepr instance at ...)", id="paren"),
            pytest.param("repr", "<BrokenRepr instance at ...>", id="repr"),
            pytest.param(
                "unicode-angle", "⟨BrokenRepr: BrokenRepr instance at ...⟩", id="unicode-angle"
            ),
        ],
    )
    def test_broken_repr_styles(self, style, expected_template):
        """Format value using basic styles."""

        class BrokenRepr:
            def __repr__(self):
                raise RuntimeError("Broken repr!")

        obj = BrokenRepr()
        repr_ = fmt_value(obj, opts=FmtOptions(style=style))

        # Split the template into static parts
        prefix, suffix = expected_template.split("...")

        # Verify exact structure ignoring the address in the middle
        assert repr_.startswith(prefix)
        assert repr_.endswith(suffix)
        # Optional: Ensure the address part is not empty/missing
        assert len(repr_) > len(prefix) + len(suffix)

    # ---------- Edge cases critical for exceptions/logging ----------

    def test_fmt_value_none_value(self):
        """None values are common in exception contexts"""
        out = fmt_value(None, opts=FmtOptions(style="angle", label_primitives=True))
        assert out == "<NoneType: None>"

    def test_fmt_value_empty_string(self):
        """Empty strings are common edge cases"""
        out = fmt_value("", opts=FmtOptions(style="angle", label_primitives=True))
        assert out == "<str: ''>"

    def test_fmt_value_empty_containers(self):
        """Empty containers often appear in validation errors"""
        opts = FmtOptions(style="angle", label_primitives=True)
        assert fmt_value([], opts=opts) == "<list: []>"
        assert fmt_value({}, opts=opts) == "<dict: {}>"
        assert fmt_value(set(), opts=opts) == "<set: set()>"

    def test_fmt_value_very_long_string_realistic(self):
        """Test with realistic long content like file paths or SQL"""
        long_path = "/very/long/path" * 1000
        out = fmt_value(long_path, opts=FmtOptions(style="angle", label_primitives=True))
        assert "..." in out
        assert out.startswith("<str: '")

    def test_fmt_value_repr_recursive(self):
        """Recursive objects can cause infinite recursion in repr"""
        lst = [1, 2]
        lst.append(lst)  # Create recursion: [1, 2, [...]]
        out = fmt_value(lst, opts=FmtOptions().merge(max_depth=3, style="angle"))
        assert out == "<list: [1, 2, [1, 2, [1, 2, [...]]]]>"

    def test_fmt_value_repr_indent(self):
        """Recursive objects can cause infinite recursion in repr"""
        lst = [1, 2]
        lst.append(lst)  # Create recursion: [1, 2, [...]]
        repr_ = reprlib.Repr(indent="....")
        opts = FmtOptions().merge(max_depth=1, repr=repr_, style="angle")
        out = fmt_value(lst, opts=opts)
        assert (
            out
            == """<list: [
....1,
....2,
....[...],
]>"""
        )

    def test_fmt_value_unicode_in_strings(self):
        """Unicode content is common in modern applications"""
        unicode_str = "Hello 世界 🌍 café"
        out = fmt_value(unicode_str, opts=FmtOptions(style="unicode-angle", label_primitives=True))
        assert "⟨str: 'Hello" in out
        assert "世界" in out or "\\u" in out  # Either preserved or escaped

    def test_fmt_value_ascii_inner_gt(self):
        """ASCII style angle brackets in content"""
        s = "<X<Y>>"
        out = fmt_value(s, opts=FmtOptions(style="angle", label_primitives=True))
        assert out == "<str: '<X<Y>>'>"

    def test_fmt_value_large_numbers(self):
        """Large numbers common in scientific/financial contexts"""
        big_int = 123456789012345678901234567890
        out = fmt_value(big_int, opts=FmtOptions(style="angle", label_primitives=True))
        assert "int" in out
        assert str(big_int) in out or "..." in out

    # ---------- Type handling for exceptions ----------

    def test_fmt_value_exception_objects(self):
        """Exception objects themselves often appear in logging"""
        opts = FmtOptions(style="equal", deduplicate_types=False)
        exc = ValueError("Something went wrong")
        out = fmt_value(exc, opts=opts)
        assert out == "ValueError=ValueError('Something went wrong')"

    def test_fmt_value_type_name_for_user_class(self):
        """User-defined types common in business logic errors"""

        class Foo:
            def __repr__(self):
                return "Foo()"

        f = Foo()
        opts = FmtOptions(style="equal", deduplicate_types=False)
        out = fmt_value(f, opts=opts)
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
            out = fmt_value(value, opts=FmtOptions(style="colon", label_primitives=True))
            assert f"{expected_type}:" in out

    def test_fmt_value_bytes_is_textual(self):
        """Bytes often contain binary data that needs careful handling"""
        b = b"abc\x00\xff"  # Include null and high bytes
        out = fmt_value(b, opts=FmtOptions(style="unicode-angle", label_primitives=True))
        assert out.startswith("⟨bytes:")
        assert "\\x" in out or "abc" in out  # Should handle binary safely

    # ---------- Parameter validation (defensive) ----------

    def test_fmt_value_invalid_style_fallback(self):
        """Should gracefully handle invalid styles"""
        out = fmt_value({123: 456}, opts=FmtOptions(style="nonexistent-style"))
        # Should fall back to default formatting, not crash
        assert out == "{123: 456}"

    def test_fmt_value_negative_max_str(self):
        """Edge case: negative max_str on a string"""
        out = fmt_value(
            "hello" * 3, opts=FmtOptions(style="angle", label_primitives=True).merge(max_str=-5)
        )
        # Should handle gracefully with reprlib, not crash
        assert out == "<str: '...'>"

    def test_fmt_value_negative_max_str_bytes(self):
        """Edge case: negative max_str on bytes"""
        out = fmt_value(
            b"hello" * 3, opts=FmtOptions(style="angle", label_primitives=True).merge(max_str=-5)
        )
        # Should handle gracefully with reprlib, not crash
        assert out == "<bytes: b'...'>"

    def test_fmt_value_negative_max_str_long(self):
        """Edge case: negative max_str on long"""
        out = fmt_value(
            10**200, opts=FmtOptions(style="angle", label_primitives=True).merge(max_str=3)
        )
        # Should handle gracefully with reprlib, not crash
        assert out == "<int: ...>"
