#
# C108 - Dictify Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import collections.abc as abc
import sys
from dataclasses import dataclass, field
from typing import Any

# Third-party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.dictify import (DictifyOptions, HookMode, MetaMixin, DictifyMeta, SizeMeta, TrimMeta, TypeMeta,
                          core_dictify, dictify)
from c108.tools import print_title
from c108.utils import class_name


# Classes --------------------------------------------------------------------------------------------------------------

@dataclass
class SimpleDC(MetaMixin):
    a: int
    b: str | None = None


@dataclass
class WithProps(MetaMixin):
    x: int
    y: int | None = None

    @property
    def sum(self) -> int:
        return self.x + (self.y or 0)

    @property
    def _hidden(self) -> str:  # should be ignored
        return "hidden"


class NotDataClass(MetaMixin):
    def __init__(self) -> None:
        self.z = 1


# Tests ----------------------------------------------------------------------------------------------------------------

class TestMetaMixin:
    def test_requires_dataclass(self):
        """Raise on non-dataclass instances."""
        obj = NotDataClass()
        with pytest.raises(TypeError, match=r"(?i)dataclass"):
            obj.to_dict()

    @pytest.mark.parametrize(
        "inst, include_none, expected",
        [
            pytest.param(SimpleDC(a=1, b=None), False, {"a": 1}, id="simple-exclude-none"),
            pytest.param(SimpleDC(a=1, b=None), True, {"a": 1, "b": None}, id="simple-include-none"),
        ],
    )
    def test_none_filtering(self, inst: MetaMixin, include_none: bool, expected: dict[str, Any]):
        """Control inclusion of None values."""
        assert inst.to_dict(include_none_values=include_none) == expected

    def test_include_properties(self):
        """Include public properties by default."""
        inst = WithProps(x=2, y=3)
        result = inst.to_dict(include_properties=True)
        assert result["x"] == 2
        assert result["y"] == 3
        assert result["sum"] == 5
        assert "_hidden" not in result

    def test_exclude_properties(self):
        """Exclude properties when requested."""
        inst = WithProps(x=2, y=3)
        result = inst.to_dict(include_properties=False)
        assert result == {"x": 2, "y": 3}

    @pytest.mark.parametrize(
        "sort_keys, expected_keys",
        [
            pytest.param(False, ["x", "y", "sum"], id="unsorted"),
            pytest.param(True, ["sum", "x", "y"], id="sorted"),
        ],
    )
    def test_sort_keys(self, sort_keys: bool, expected_keys: list[str]):
        """Sort result keys when requested."""
        inst = WithProps(x=1, y=2)
        result = inst.to_dict(sort_keys=sort_keys)
        assert list(result.keys()) == expected_keys

    def test_property_inclusion_with_none_filtering(self):
        """Filter None values including property results."""
        inst = WithProps(x=5, y=None)
        result = inst.to_dict(include_none_values=False, include_properties=True)
        # y should be dropped, sum computed as 5 (still included)
        assert result == {"x": 5, "sum": 5}

    def test_property_computation_errors_surface(self):
        """Surface property access errors."""

        @dataclass
        class BadProp(MetaMixin):
            v: int

            @property
            def boom(self) -> int:
                raise ValueError("boom!")

        inst = BadProp(v=1)
        with pytest.raises(ValueError, match=r"(?i)boom"):
            inst.to_dict()

    def test_property_name_filtering(self):
        """Ignore private-like properties."""

        @dataclass
        class PrivateProps(MetaMixin):
            p: int = 1

            @property
            def _private(self) -> int:
                return 7

            @property
            def public(self) -> int:
                return 3

        inst = PrivateProps()
        result = inst.to_dict()
        assert "public" in result and result["public"] == 3
        assert "_private" not in result

    def test_merged_property_and_field_keys(self):
        """Merge dataclass fields with properties."""

        @dataclass
        class Overlap(MetaMixin):
            val: int = 2

            @property
            def val_prop(self) -> int:
                return self.val * 2

        inst = Overlap()
        result = inst.to_dict()
        assert result["val"] == 2
        assert result["val_prop"] == 4


class TestSizeMeta:
    def test_defaults(self):
        """Create with defaults and succeed."""
        sm = SizeMeta()
        assert sm.len is None and sm.deep is None and sm.shallow is None

    @pytest.mark.parametrize(
        "field, value",
        [
            pytest.param("len", -1, id="len-negative"),
            pytest.param("deep", -5, id="deep-negative"),
            pytest.param("shallow", -2, id="shallow-negative"),
        ],
    )
    def test_negative_values(self, field: str, value: int):
        """Reject negative integers."""
        kwargs = {field: value}
        with pytest.raises(ValueError, match=r"(?i)>=0"):
            SizeMeta(**kwargs)

    @pytest.mark.parametrize(
        "field, value",
        [
            pytest.param("len", 3.14, id="len-float"),
            pytest.param("deep", "100", id="deep-str"),
            pytest.param("shallow", object(), id="shallow-object"),
            pytest.param("len", True, id="len-bool"),
            pytest.param("deep", False, id="deep-bool"),
        ],
    )
    def test_type_validation(self, field: str, value):
        """Reject non-int and bool values."""
        kwargs = {field: value}
        with pytest.raises(TypeError, match=r"(?i)must be an int"):
            SizeMeta(**kwargs)

    def test_deep_not_less_than_shallow(self):
        """Enforce deep >= shallow relation."""
        with pytest.raises(ValueError, match=r"(?i)deep.*>=.*shallow"):
            SizeMeta(deep=9, shallow=10)

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param(dict(len=0, deep=0, shallow=0), id="all-zero"),
            pytest.param(dict(len=5, deep=10, shallow=10), id="equal-deep-shallow"),
            pytest.param(dict(len=None, deep=20, shallow=10), id="deep-greater"),
        ],
    )
    def test_valid_configurations(self, kwargs):
        """Accept valid combinations."""
        sm = SizeMeta(**kwargs)
        for k, v in kwargs.items():
            assert getattr(sm, k) == v

    def test_to_dict_integration(self):
        """Convert to dict via mixin."""
        sm = SizeMeta(len=7, deep=100, shallow=60)
        d = sm.to_dict(sort_keys=True)
        assert list(d.keys()) == ["deep", "len", "shallow"]
        assert d == {"len": 7, "deep": 100, "shallow": 60}


class TestCoreDictify:

    def test_basic_object_conversion(self):
        """Convert simple object to dictionary."""

        class Person:
            def __init__(self, name, age):
                self.name = name
                self.age = age

        person = Person("Alice", 7)
        result = core_dictify(person)
        assert result == {"name": "Alice", "age": 7}

    @pytest.mark.parametrize(
        "value",
        [42, 3.14, True, 2 + 3j, None],
        ids=["int", "float", "bool", "complex", "none"],
    )
    def test_never_filtered_as_is(self, value):
        """Return never-filtered builtins as is."""
        assert core_dictify(value) is value

    def test_options_type_error(self):
        """Validate options type check."""
        with pytest.raises(TypeError, match=r"(?i)options must be a DictifyOptions"):
            core_dictify(object(), options=123)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("kw", "bad"),
        [("fn_raw", 123), ("fn_terminal", 123)],
        ids=["fn_raw", "fn_terminal"],
    )
    def test_fn_callable_type_error(self, kw, bad):
        """Validate fn_raw/fn_terminal type checks."""
        kwargs = {kw: bad}
        with pytest.raises(TypeError, match=r"(?i)must be a Callable"):
            core_dictify(object(), **kwargs)

    def test_hook_mode_dict_calls_to_dict_and_injects_class(self):
        """Inject class name when to_dict returns mapping."""

        class WithToDict:
            def to_dict(self):
                return {"x": 1}

        opts = DictifyOptions(hook_mode=HookMode.DICT, inject_class_name=True, fully_qualified_names=False)
        res = core_dictify(WithToDict(), options=opts)

        print("res:", res)

        assert res["x"] == 1
        assert res["__class__"] == "WithToDict"

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
        """Return fn_raw when max_depth is negative."""
        marker = object()
        opts = DictifyOptions(max_depth=-1)
        res = core_dictify(object(), options=opts, fn_raw=lambda x, opt: marker)
        assert res is marker

    def test_sequence_without_len_falls_back_to_fn_process(self):
        """Apply fn_terminal for Sequence lacking __len__."""

        class MySeqNoLen:
            def __iter__(self):
                yield from (1, 2, 3)

            # no __len__

        # Virtually register as Sequence while lacking __len__
        abc.Sequence.register(MySeqNoLen)

        marker = ("processed", "no-len")
        res = core_dictify(MySeqNoLen(), fn_terminal=lambda x, opt: marker)
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
        """Use fn_terminal when max_depth is zero for user object."""

        class Foo:
            pass

        marker = ("processed", "Foo")
        opts = DictifyOptions(max_depth=0)
        res = core_dictify(Foo(), options=opts, fn_terminal=lambda x, opt: marker)
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

        leaf = Node(name="leaf")
        root = Node(name="root", child=leaf)

        # Use max_depth=1 so only the root is expanded; nested objects remain raw.
        opts = DictifyOptions(max_depth=1)
        # Do not pass fn_terminal; identity fallback keeps terminal objects as-is.
        res = core_dictify(root, options=opts)

        print("result:", res)

        assert isinstance(res, dict)
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

    @pytest.mark.parametrize("fqn", [False, True], ids=["short-name", "fully-qualified"])
    def test_include_class_name_attrs(self, fqn):
        """Include class name during normal attribute scanning with optional FQN."""

        class Obj:
            def __init__(self):
                self.a = 1

        opts = DictifyOptions(
            max_depth=1,
            include_class_name=True,
            fully_qualified_names=fqn,
        )
        res = core_dictify(Obj(), options=opts)

        expected_class = Obj.__name__ if not fqn else f"{Obj.__module__}.{Obj.__name__}"
        assert res["a"] == 1
        assert res["__class__"] == expected_class

    def test_include_class_name_attrs_disabled(self):
        """Do not include class name when option is disabled."""

        class Obj:
            def __init__(self):
                self.a = 1

        opts = DictifyOptions(max_depth=1, include_class_name=False)
        res = core_dictify(Obj(), options=opts)
        assert res == {"a": 1}

    def test_to_dict_injects_class_name_fqn(self):
        """Inject class name into to_dict result with fully qualified name."""

        class WithToDict:
            def to_dict(self):
                return {"x": 1}

        opts = DictifyOptions(
            hook_mode=HookMode.DICT,
            inject_class_name=True,
            fully_qualified_names=True,
        )
        res = core_dictify(WithToDict(), options=opts)

        expected_class = f"{WithToDict.__module__}.{WithToDict.__name__}"
        assert res["x"] == 1
        assert res["__class__"] == expected_class

    def test_to_dict_no_injection_when_disabled(self):
        """Do not inject class name when inject_class_name is False for to_dict."""

        class WithToDict:
            def to_dict(self):
                return {"x": 1}

        opts = DictifyOptions(
            hook_mode=HookMode.DICT,
            inject_class_name=False,
            fully_qualified_names=True,
        )
        res = core_dictify(WithToDict(), options=opts)
        assert res == {"x": 1}

    def test_depth_partial_object_expansion(self):
        """Expand two levels of object tree and keep deeper nodes raw."""

        class Node:
            def __init__(self, name, child=None):
                self.name = name
                self.child = child

        leaf = Node("leaf")
        mid = Node("mid", child=leaf)
        root = Node("root", child=mid)

        # Depth=2: root expanded (depth->1), child expanded (depth->0), grandchild stays raw.
        opts = DictifyOptions(max_depth=2, include_class_name=False)
        res = core_dictify(root, options=opts)

        assert res["name"] == "root"
        assert res["child"]["name"] == "mid"
        assert res["child"]["child"] is leaf  # Raw at terminal depth

    def test_fn_terminal_output_not_modified(self):
        """Do not inject class name into fn_terminal output."""

        class Foo:
            pass

        # At depth=0, fn_terminal is used and its output must not be modified.
        opts = DictifyOptions(max_depth=0, include_class_name=True, fully_qualified_names=True)
        res = core_dictify(Foo(), options=opts, fn_terminal=lambda x, opt: {"marker": "terminal"})
        assert res == {"marker": "terminal"}


#
# class TestCoreVsDictify:
#     def test_object_tree_depth_control(self):
#         """Expand object to dict but keep nested objects as raw values at depth 1."""
#
#         class Node:
#             def __init__(self, name=None, child=None):
#                 self.a = name
#                 self.b = 'b'
#                 self.child = child
#
#         def fn_terminal(obj):
#             if isinstance(obj, (int, float, str)):
#                 return obj
#             return {"terminal": f"{obj.a} - child:{bool(obj.child)} - {sys.getsizeof(obj)} bytes"}
#
#         leaf_2 = Node(name="leaf_2")
#         leaf_1 = Node(name="leaf_1", child=leaf_2)
#         leaf_0 = Node(name="leaf_0", child=leaf_1)
#         root = Node(name="root", child=leaf_0)
#
#         for d in [-1, 0, 1, 2, 3, 4, 10]:
#             print("\ndepth       :", d)
#             # print("dictify     :", dictify(root, max_depth=d))
#             print("core_dictify:", core_dictify(root, fn_terminal=fn_terminal, options=DictifyOptions(max_depth=d)))
#
#         # assert res["name"] == "root"
#         # assert res["child"] is leaf  # Raw object, not processed
#

class TestDictify:
    """Test suite for dictify() method."""

    def test_basic_object_conversion(self):
        """Convert simple object to dictionary."""

        class Person:
            def __init__(self, name, age):
                self.name = name
                self.age = age

        person = Person("Dodo", 5)
        result = dictify(person)

        print("\nresult:", result)

        assert result == {"name": "Dodo", "age": 5}

    # @pytest.mark.parametrize("primitive", [
    #     42, 3.14, True, None, range(0, 7)
    # ], ids=["int", "float", "bool", "none", "range"])
    # def test_primitive_types_preserved(self, primitive):
    #     """Preserve built-in types as-is."""
    #     result = dictify(primitive)
    #     assert result == primitive
    #
    # def test_max_depth_control(self):
    #     """Control recursion depth with max_depth parameter."""
    #
    #     raise NotImplemented("not implemented, try to use simplified logic of test_object_tree_depth_control()")
    #
    # def test_include_private_attributes(self):
    #     """Include private attributes when include_private=True."""
    #
    #     class TestClass:
    #         def __init__(self):
    #             self.public = "visible"
    #             self._private = "hidden"
    #
    #     obj = TestClass()
    #
    #     # Default: exclude private
    #     result_default = dictify(obj)
    #     assert "_private" not in result_default
    #     assert result_default == {"public": "visible"}
    #
    #     # Include private
    #     result_with_private = dictify(obj, include_private=True)
    #     assert result_with_private == {"public": "visible", "_private": "hidden"}
    #
    # def test_include_class_name(self):
    #     """Include class name when include_class_name=True."""
    #
    #     class MyClass:
    #         def __init__(self):
    #             self.attr = "value"
    #
    #     obj = MyClass()
    #     opt = DictifyOptions(fully_qualified_names=False)
    #     result = dictify(obj, include_class_name=True, options=opt)
    #     assert result["__class__"] == "MyClass"
    #     assert result["attr"] == "value"
    #
    # def test_max_items_limitation(self):
    #     """Limit collection size with max_items parameter."""
    #     large_dict = {f"key_{i}": i for i in range(100)}
    #     result = dictify(large_dict, max_items=10)
    #     assert len(result) <= 10
    #
    # def test_options_override_parameters(self):
    #     """Use DictifyOptions to override individual parameters."""
    #
    #     class TestClass:
    #         def __init__(self):
    #             self.attr = "value"
    #             self._private = "hidden"
    #
    #     obj = TestClass()
    #     # Options should be used when provided as kwargs
    #     result = dictify(obj, include_private=True, include_class_name=True)
    #     print("\nresult", result)
    #     assert "_private" in result
    #     assert "__class__" in result
    #
    # @pytest.mark.parametrize(
    #     "invalid_depth",
    #     ["not_int", 3.5, None, [], {}],
    #     ids=["str", "float", "none", "list", "dict"],
    # )
    # def test_max_depth_type_validation(self, invalid_depth):
    #     """Raise TypeError for invalid max_depth types."""
    #     obj = {"test": "value"}
    #     with pytest.raises(TypeError, match=r"(?i)int"):
    #         dictify(obj, max_depth=invalid_depth)
    #
    # @pytest.mark.parametrize(
    #     "invalid_items",
    #     ["not_int", 50.5, None, (), {"a": 1}],
    #     ids=["str", "float", "none", "tuple", "dict"],
    # )
    # def test_max_items_type_validation(self, invalid_items):
    #     """Raise TypeError for invalid max_items types."""
    #     obj = {"test": "value"}
    #     with pytest.raises(TypeError, match=r"(?i)int"):
    #         dictify(obj, max_items=invalid_items)
    #
    # @pytest.mark.parametrize("invalid_options", [
    #     "not_options",
    #     123,
    #     {"not": "options"},
    # ], ids=["str", "int", "dict"])
    # def test_options_type_validation(self, invalid_options):
    #     """Raise TypeError for invalid options type."""
    #     obj = {"test": "value"}
    #
    #     with pytest.raises(TypeError, match=r"(?i)DictifyOptions"):
    #         dictify(obj, options=invalid_options)
    #
    # def test_nested_collections_processing(self):
    #     """Process nested collections up to max_depth levels."""
    #
    #     class Item:
    #         def __init__(self, name):
    #             self.name = name
    #
    #     nested_data = {
    #         "items": [Item("first"), Item("second")],
    #         "metadata": {"count": 2, "nested": Item("meta")}
    #     }
    #
    #     result = dictify(nested_data, max_depth=3)
    #     expected = {"items": [{"name": "first"},
    #                           {"name": "second"}],
    #                 "metadata": {"count": 2,
    #                              "nested": {"name": 'meta'}}
    #                 }
    #     assert result == expected
    #     assert result["items"][0] == {"name": "first"}
    #     assert result["items"][1] == {"name": "second"}
    #     assert result["metadata"]["nested"] == {"name": "meta"}
    #
    # def test_exception_properties_skipped(self):
    #     """Skip properties that raise exceptions during access."""
    #
    #     class ProblematicClass:
    #         def __init__(self):
    #             self.good_attr = "accessible"
    #
    #         @property
    #         def bad_property(self):
    #             raise RuntimeError("Cannot access this property")
    #
    #     obj = ProblematicClass()
    #     result = dictify(obj)
    #
    #     # Should contain good attribute but skip the problematic property
    #     assert "good_attr" in result
    #     assert "bad_property" not in result
    #     assert result["good_attr"] == "accessible"
    #
    # def test_complex_nested_structure(self):
    #     """Handle complex nested structures with mixed types."""
    #
    #     class Address:
    #         def __init__(self, street, city):
    #             self.street = street
    #             self.city = city
    #
    #     class Person:
    #         def __init__(self, name, addresses):
    #             self.name = name
    #             self.addresses = addresses
    #             self._id = 12345
    #
    #     person = Person("Bob", [
    #         Address("123 Main St", "Springfield"),
    #         Address("456 Oak Ave", "Shelbyville")
    #     ])
    #
    #     result = dictify(person, max_depth=3, include_private=True)
    #
    #     expected = {
    #         "name": "Bob",
    #         "_id": 12345,
    #         "addresses": [
    #             {"street": "123 Main St", "city": "Springfield"},
    #             {"street": "456 Oak Ave", "city": "Shelbyville"}
    #         ]
    #     }
    #
    #     assert result == expected
    #
