#
# C108 - Dictify Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import collections.abc as abc
from dataclasses import dataclass, field

# Third-party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.dictify import core_dictify, DictifyOptions, HookMode
from c108.tools import print_title
from c108.utils import class_name


# Classes --------------------------------------------------------------------------------------------------------------


# Tests ----------------------------------------------------------------------------------------------------------------

class TestCoreDictify:
    @pytest.mark.parametrize(
        "value",
        [42, 3.14, True, 2 + 3j, None, range(3)],
        ids=["int", "float", "bool", "complex", "none", "range"],
    )
    def test_never_filtered_as_is(self, value):
        """Return never-filtered builtins as is."""
        assert core_dictify(value) is value

    @pytest.mark.parametrize(
        "value",
        ["abc", b"x", bytearray(b"x"), memoryview(b"x")],
        ids=["str", "bytes", "bytearray", "memoryview"],
    )
    def test_always_filtered_uses_fn_process(self, value):
        """Apply fn_process to always-filtered builtins."""
        result = core_dictify(value, fn_process=lambda x: ("processed", type(x).__name__))
        assert result == ("processed", type(value).__name__)

    def test_options_type_error(self):
        """Validate options type check."""
        with pytest.raises(TypeError, match=r"(?i)options must be a DictifyOptions"):
            core_dictify(object(), options=123)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("kw", "bad"),
        [("fn_plain", 123), ("fn_process", 123)],
        ids=["fn_plain", "fn_process"],
    )
    def test_fn_callable_type_error(self, kw, bad):
        """Validate fn_plain/fn_process type checks."""
        kwargs = {kw: bad}
        with pytest.raises(TypeError, match=r"(?i)must be a Callable"):
            core_dictify(object(), **kwargs)

    def test_hook_mode_dict_calls_to_dict_and_injects_class(self):
        """Inject class name when to_dict returns mapping."""

        class WithToDict:
            def to_dict(self):
                return {"x": 1}

        opts = DictifyOptions(hook_mode=HookMode.DICT, include_class_name=True, fully_qualified_names=False)
        res = core_dictify(WithToDict(), options=opts)
        assert res["x"] == 1
        assert res["_class_name"] == "WithToDict"

    def test_hook_mode_strict_missing_to_dict_raises(self):
        """Raise when DICT_STRICT and no to_dict."""
        opts = DictifyOptions(hook_mode=HookMode.DICT_STRICT)
        with pytest.raises(TypeError, match=r"(?i)must implement to_dict"):
            core_dictify(object(), options=opts)

    def test_to_dict_non_mapping_raises(self):
        """Raise when to_dict returns non-mapping."""

        class BadToDict:
            def to_dict(self):
                return [("k", "v")]

        opts = DictifyOptions(hook_mode=HookMode.DICT)
        with pytest.raises(TypeError, match=r"(?i)must return a Mapping"):
            core_dictify(BadToDict(), options=opts)

    def test_max_depth_negative_uses_fn_plain(self):
        """Return fn_plain when max_depth is negative."""
        marker = object()
        opts = DictifyOptions(max_depth=-1)
        res = core_dictify(object(), options=opts, fn_plain=lambda x: marker)
        assert res is marker

    def test_sequence_without_len_falls_back_to_fn_process(self):
        """Apply fn_process for Sequence lacking __len__."""

        class MySeqNoLen:
            def __iter__(self):
                yield from (1, 2, 3)

            # no __len__

        # Virtually register as Sequence while lacking __len__
        abc.Sequence.register(MySeqNoLen)

        marker = ("processed", "no-len")
        res = core_dictify(MySeqNoLen(), fn_process=lambda x: marker)
        assert res == marker

    def test_max_items_triggers_fn_process(self):
        """Apply fn_process when collection exceeds max_items."""
        opts = DictifyOptions(max_depth=3, max_items=1)
        marker = "too-big"
        res = core_dictify([1, 2, 3], options=opts, fn_process=lambda x: marker)
        assert res == marker

    @pytest.mark.parametrize(
        ("include_none_items", "expected_keys"),
        [(False, {"a"}), (True, {"a", "b"})],
        ids=["drop-none", "keep-none"],
    )
    def test_mapping_include_none_items(self, include_none_items, expected_keys):
        """Respect include_none_items for plain mappings."""
        opts = DictifyOptions(include_none_items=include_none_items)
        res = core_dictify({"a": 1, "b": None}, options=opts)
        assert set(res.keys()) == expected_keys

    def test_to_str_converts_selected_types(self):
        """Convert selected types to string via to_str."""

        class Fancy:
            def __str__(self):
                return "FANCY"

        opts = DictifyOptions(to_str=(Fancy,))
        res = core_dictify(Fancy(), options=opts)
        assert res == "FANCY"

    def test_object_expansion_toplevel_filters_attrs(self):
        """Expand object attributes and respect include_none_attrs."""

        class Obj:
            def __init__(self):
                self.a = 1
                self.b = None

        opts = DictifyOptions(max_depth=1, include_none_attrs=False, include_class_name=False)
        res = core_dictify(Obj(), options=opts)
        assert res == {"a": 1}

    def test_depth_zero_uses_fn_process_on_user_object(self):
        """Use fn_process when max_depth is zero for user object."""

        class Foo:
            pass

        marker = ("processed", "Foo")
        opts = DictifyOptions(max_depth=0)
        res = core_dictify(Foo(), options=opts, fn_process=lambda x: marker)
        assert res == marker

    def test_recursive_sequence_respects_depth(self):
        """Process nested sequences with proper depth control."""

        class Foo:
            def __init__(self):
                self.value = 42

        data = [[Foo()]]
        opts = DictifyOptions(max_depth=3, include_class_name=False)  # Need depth=3!
        res = core_dictify(data, options=opts)
        assert res == [[{"value": 42}]]

    def test_object_tree_depth_control(self):
        """Expand object to dict but keep nested objects as raw values at depth 1."""

        class Node:
            def __init__(self, name, child=None):
                self.name = name
                self.child = child

        leaf = Node("leaf")
        root = Node("root", leaf)
        opts = DictifyOptions(max_depth=1, include_class_name=False)
        res = core_dictify(root, options=opts, fn_process=lambda x: {"info": type(x).__name__})
        assert res["name"] == "root"
        assert res["child"] is leaf  # Raw object, not processed

    def test_invalid_hook_mode_raises_value_error(self):
        """Raise ValueError on invalid hook_mode."""
        bad_opts = DictifyOptions(hook_mode="unexpected")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=r"(?i)unknown hook_mode value"):
            core_dictify(object(), options=bad_opts)

    def test_property_exception_is_skipped(self):
        """Skip properties that raise exceptions when include_properties is on."""

        class WithBadProp:
            def __init__(self):
                self.ok = 1

            @property
            def bad(self):
                raise RuntimeError("boom")

        opts = DictifyOptions(max_depth=1, include_properties=True)
        res = core_dictify(WithBadProp(), options=opts)
        assert res == {"ok": 1}
