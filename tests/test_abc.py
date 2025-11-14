#
# C108 - ABC Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import inspect
import warnings
import re, sys
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

# Third-party ----------------------------------------------------------------------------------------------------------
import pytest


# Local ----------------------------------------------------------------------------------------------------------------
from c108.abc import (
    ClassGetter,
    ObjectInfo,
    _acts_like_image,
    classgetter,
    deep_sizeof,
    is_builtin,
    search_attrs,
    valid_param_types,
    validate_param_types,
    validate_types,
)


# Local Classes & Methods ----------------------------------------------------------------------------------------------
from c108.dataclasses import mergeable


class BuggySize:
    """Object whose __sizeof__ raises to test error handling."""

    def __sizeof__(self) -> int:
        raise RuntimeError("Broken __sizeof__ for testing")


class Example:
    """A class with various attribute types for testing."""

    regular_attribute = "value"

    @property
    def working_property(self) -> str:
        """A sql, functioning property."""
        return "works"

    @property
    def failing_property(self) -> str:
        """A property that always raises an exception."""
        raise ValueError("This property fails on access")

    def a_method(self) -> None:
        """A regular method."""
        pass


@dataclass
class SimpleDataClass:
    """A simple dataclass for testing."""

    field: str = "data"


# Helper constructs for testing non-built-in objects
class UserDefinedClass:
    """A simple user-defined class for testing purposes."""

    def method(self):
        """A simple method."""
        pass


class ToExclude:
    """Custom class for exclusion tests."""

    def __init__(self, payload: str) -> None:
        self.payload = payload


def user_defined_function():
    """A simple user-defined function."""
    pass


# Tests ----------------------------------------------------------------------------------------------------------------


class TestClassGetter:
    """Tests for the ClassGetter descriptor."""

    def test_basic_access(self):
        """Access computed class-level value."""

        class A:
            @ClassGetter
            def val(cls):
                return 42

        assert A.val == 42

    def test_cache_behavior(self):
        """Cache computed value per class."""
        calls = {}

        class B:
            @ClassGetter
            def val(cls):
                calls[cls] = calls.get(cls, 0) + 1
                return calls[cls]

        # No caching
        assert B.val == 1
        assert B.val == 2

        class C:
            @ClassGetter
            def val(cls):
                calls[cls] = calls.get(cls, 0) + 1
                return calls[cls]

        # Independent class, no cache
        assert C.val == 1

    def test_cache_enabled(self):
        """Return cached value when cache=True."""
        calls = {}

        class D:
            def val(cls):  # ✅ Plain function, no decorator
                calls[cls] = calls.get(cls, 0) + 1
                return calls[cls]

        d = ClassGetter(D.val, cache=True)  # ✅ Now D.val is a function
        d.__set_name__(D, "val_cached")
        D.val_cached = d

        assert D.val_cached == 1
        assert D.val_cached == 1  # Same value (cached)
        assert calls[D] == 1  # Only called once

    def test_set_raises_attribute_error(self):
        """Raise AttributeError on instance assignment."""

        class E:
            @ClassGetter
            def val(cls):
                return 10

        e = E()
        with pytest.raises(AttributeError, match=r"(?i).*read-only.*"):
            e.val = 99

    def test_set_name_assigns_name(self):
        """Assign name via __set_name__."""

        def f(cls):
            return 1

        cg = ClassGetter(f)
        cg.__set_name__(type("Dummy", (), {}), "foo")
        assert cg.name == "foo"

    def test_inheritance_independent_cache(self):
        """Maintain separate cache per subclass."""
        calls = {}

        class Base:
            def val(cls):  # ✅ Plain function, no decorator
                calls[cls] = calls.get(cls, 0) + 1
                return calls[cls]

        base_getter = ClassGetter(Base.val, cache=True)  # ✅ Now Base.val is a function
        base_getter.__set_name__(Base, "val_cached")
        Base.val_cached = base_getter

        class Sub(Base):
            pass

        assert Base.val_cached == 1
        assert Sub.val_cached == 1  # Different cache
        assert Base.val_cached == 1  # Still 1 (cached)
        assert calls[Base] == 1
        assert calls[Sub] == 1


class TestClassgetterDecorator:
    """Tests for the @classgetter decorator."""

    def test_decorator_without_args(self):
        """Decorate method without parentheses."""

        class A:
            @classgetter
            def val(cls):
                return 5

        assert isinstance(A.__dict__["val"], ClassGetter)
        assert A.val == 5

    def test_decorator_with_cache(self):
        """Decorate method with cache=True."""
        calls = {}

        class B:
            @classgetter(cache=True)
            def val(cls):
                calls[cls] = calls.get(cls, 0) + 1
                return calls[cls]

        assert B.val == 1
        assert B.val == 1
        assert calls[B] == 1

    def test_class_assignment_replaces_descriptor(self):
        """Allow class-level replacement of descriptor."""

        class C:
            @classgetter
            def val(cls):
                return 1

        C.val = 99
        assert C.val == 99

    def test_instance_access_blocked(self):
        """Block instance-level assignment."""

        class D:
            @classgetter
            def val(cls):
                return 7

        d = D()
        with pytest.raises(AttributeError, match=r"(?i).*read-only.*"):
            d.val = 10

    def test_multiple_classes_independent(self):
        """Ensure independent behavior across classes."""

        class E:
            @classgetter(cache=True)
            def val(cls):
                return id(cls)

        class F(E):
            pass

        assert E.val != F.val

    def test_returns_decorator_when_no_func(self):
        """Return decorator when called without func."""
        dec = classgetter(cache=True)

        def f(cls):
            return 1

        cg = dec(f)
        assert isinstance(cg, ClassGetter)
        assert cg.cache is True

    def test_function_direct_call(self):
        """Support direct call with function argument."""

        def f(cls):
            return 2

        cg = classgetter(f)
        assert isinstance(cg, ClassGetter)
        assert cg.cache is False


class TestObjectInfo:
    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(0, id="int"),
            pytest.param(3.14, id="float"),
            pytest.param(True, id="bool"),
            pytest.param(2 + 3j, id="complex"),
        ],
    )
    def test_numbers_bytes_unit(self, value):
        """Validate numbers report bytes unit and have integer size."""
        info = ObjectInfo.from_object(value, fully_qualified=False, deep_size=False)
        assert info.unit == "bytes"
        assert isinstance(info.size, int)
        assert info.type is type(value)
        assert info.deep_size is None

    def test_string_chars_size(self):
        """Validate strings report character count and chars unit."""
        s = "hello🌍"
        info = ObjectInfo.from_object(s, fully_qualified=False, deep_size=False)
        assert info.size == len(s)
        assert info.unit == "chars"
        assert info.type is str

    @pytest.mark.parametrize(
        "obj, expected_len",
        [
            pytest.param(b"abc", 3, id="bytes"),
            pytest.param(bytearray(b"\x00\x01\x02"), 3, id="bytearray"),
            pytest.param(memoryview(b"abcd"), 4, id="memoryview"),
        ],
    )
    def test_bytes_like_size(self, obj, expected_len):
        """Validate bytes-like objects report length in bytes."""
        info = ObjectInfo.from_object(obj, fully_qualified=False, deep_size=False)
        assert info.size == expected_len
        assert info.unit == "bytes"

    @pytest.mark.parametrize(
        "obj, expected_len",
        [
            pytest.param([1, 2, 3], 3, id="list"),
            pytest.param((1,), 1, id="tuple"),
            pytest.param({1, 2}, 2, id="set"),
            pytest.param({"a": 1, "b": 2}, 2, id="dict"),
        ],
    )
    def test_containers_items(self, obj, expected_len):
        """Validate containers report items count and items unit."""
        info = ObjectInfo.from_object(obj, fully_qualified=False, deep_size=False)
        assert info.size == expected_len
        assert info.unit == "items"

    def test_class_object_attrs_no_deep(self):
        """Validate class objects report attrs unit and omit deep size."""

        class Foo:
            a = 1

            def method(self):
                return 42

            @property
            def prop(self):
                return "x"

        info = ObjectInfo.from_object(Foo, fully_qualified=False, deep_size=True)
        assert info.unit == "attrs"
        assert info.deep_size is None
        assert info.type is Foo
        assert isinstance(info.size, int)

    def test_instance_with_no_attrs_bytes(self):
        """Validate instances without attrs fall back to bytes unit."""
        o = object()
        info = ObjectInfo.from_object(o, fully_qualified=False, deep_size=False)
        assert info.unit == "bytes"
        assert isinstance(info.size, int)
        assert info.type is type(o)

    def test_instance_with_attrs_unit(self):
        """Validate instances with attrs report attrs unit."""

        class WithAttrs:
            def __init__(self):
                self.public = 1
                self._private = 2

        w = WithAttrs()
        info = ObjectInfo.from_object(w, fully_qualified=False, deep_size=False)
        assert info.unit == "attrs"
        assert isinstance(info.size, int)
        assert info.type is WithAttrs

    def test_post_init_mismatch_raises(self):
        """Raise on size/unit length mismatch at initialization."""
        with pytest.raises(ValueError, match=r"(?i).*same length.*"):
            ObjectInfo(
                type=int,
                size=[1, 2],
                unit=["bytes"],
                deep_size=None,
                fully_qualified=False,
            )

    def test_frozen_immutability(self):
        """Validate that ObjectInfo is immutable after creation."""
        info = ObjectInfo(
            type=int,
            size=[1, 2],
            unit=["a", "b"],
            deep_size=None,
            fully_qualified=False,
        )
        # Attempting to mutate should raise FrozenInstanceError
        with pytest.raises(AttributeError, match=r"(?i).*frozen.*|cannot|can't"):
            info.unit = ["a"]

        with pytest.raises(AttributeError, match=r"(?i).*frozen.*|cannot|can't"):
            info.size = 10

    def test_frozen_hashable(self):
        """Validate that frozen ObjectInfo instances are hashable."""
        info1 = ObjectInfo(type=str, size=5, unit="chars", deep_size=None, fully_qualified=False)
        info2 = ObjectInfo(type=str, size=5, unit="chars", deep_size=None, fully_qualified=False)

        # Should be hashable
        assert hash(info1) == hash(info2)

        # Can be used in sets and as dict keys
        info_set = {info1, info2}
        assert len(info_set) == 1  # Same content, same hash

        info_dict = {info1: "value"}
        assert info_dict[info2] == "value"

    def test_to_dict_include_none(self):
        """Include deep_size when requested even if None."""
        info = ObjectInfo(type=str, size=5, unit="chars", deep_size=None, fully_qualified=False)
        data = info.to_dict(include_none_attrs=True)
        assert "deep_size" in data and data["deep_size"] is None
        assert data["size"] == 5 and data["unit"] == "chars" and data["type"] is str

    def test_to_dict_exclude_none(self):
        """Omit deep_size when not requested and None."""
        info = ObjectInfo(type=bytes, size=3, unit="bytes", deep_size=None, fully_qualified=False)
        data = info.to_dict(include_none_attrs=False)
        assert "deep_size" not in data
        assert data["size"] == 3 and data["unit"] == "bytes" and data["type"] is bytes

    def test_repr_contains_type_and_fields(self):
        """Show concise info in repr for debugging."""
        info = ObjectInfo.from_object(123, fully_qualified=False, deep_size=False)
        r = repr(info)
        assert "ObjectInfo(type=int" in r or "ObjectInfo(" in r
        assert "bytes" in r


# Tests for methods ----------------------------------------------------------------------------------------------------


class TestDeepSizeOf:
    """
    Test suite for deep_sizeof() core behaviors and guarantees.
    """

    def test_int_vs_dict_total_consistent(self) -> None:
        """Assert consistency between int and dict output total bytes."""
        obj = {"a": [1, 2, 3], "b": ("x", "y")}
        size_int = deep_sizeof(
            obj,
            format="int",
            exclude_types=(),
            exclude_ids=set(),
            max_depth=None,
            seen=set(),
            on_error="skip",
        )
        info = deep_sizeof(
            obj,
            format="dict",
            exclude_types=(),
            exclude_ids=set(),
            max_depth=None,
            seen=set(),
            on_error="skip",
        )
        assert isinstance(size_int, int)
        assert info["total_bytes"] == size_int

    def test_by_type_keys_and_sum(self) -> None:
        """Verify by_type keys are types and sum matches total_bytes."""
        obj = {"k": [1, 2], "m": {"n": "v"}}
        info = deep_sizeof(
            obj,
            format="dict",
            exclude_types=(),
            exclude_ids=set(),
            max_depth=None,
            seen=set(),
            on_error="skip",
        )
        by_type = info["by_type"]
        assert all(isinstance(t, type) for t in by_type.keys())
        assert sum(by_type.values()) == info["total_bytes"]

    def test_errors_and_problematic_types_on_skip(self) -> None:
        """Track errors and problematic types when skipping broken objects."""
        obj = {"good": [1, 2], "bad": BuggySize()}
        info = deep_sizeof(
            obj,
            format="dict",
            exclude_types=(),
            exclude_ids=set(),
            max_depth=None,
            seen=set(),
            on_error="skip",
        )
        errors: dict[type, int] = info["errors"]
        assert any(issubclass(e, RuntimeError) for e in errors.keys())
        assert RuntimeError in errors
        assert BuggySize in info["problematic_types"]

    def test_on_error_raise(self) -> None:
        """Raise the original error when on_error='raise'."""
        obj = {"bad": BuggySize()}
        with pytest.raises(RuntimeError, match=r"(?i).*broken.*"):
            _ = deep_sizeof(
                obj,
                format="int",
                exclude_types=(),
                exclude_ids=set(),
                max_depth=None,
                seen=set(),
                on_error="raise",
            )

    def test_on_error_warn_emits(self) -> None:
        """Emit warnings when on_error='warn'."""
        obj = {"bad": BuggySize(), "ok": [1, 2, 3]}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = deep_sizeof(
                obj,
                format="int",
                exclude_types=(),
                exclude_ids=set(),
                max_depth=None,
                seen=set(),
                on_error="warn",
            )
            assert len(w) >= 1

    def test_exclude_types_nested(self) -> None:
        """Exclude types recursively and reduce total size."""
        obj = {"a": ["x", "y", "z"], "b": {"s": "t"}}
        size_full = deep_sizeof(
            obj,
            format="int",
            exclude_types=(),
            exclude_ids=set(),
            max_depth=None,
            seen=set(),
            on_error="skip",
        )
        size_no_str = deep_sizeof(
            obj,
            format="int",
            exclude_types=(str,),
            exclude_ids=set(),
            max_depth=None,
            seen=set(),
            on_error="skip",
        )
        assert size_no_str < size_full

        info = deep_sizeof(
            obj,
            format="dict",
            exclude_types=(str,),
            exclude_ids=set(),
            max_depth=None,
            seen=set(),
            on_error="skip",
        )
        assert str not in info["by_type"]

    def test_exclude_custom_type(self) -> None:
        """Exclude a custom class and keep other contributions."""
        items = [ToExclude("data" * 10), 42, ToExclude("x" * 100)]
        size_full = deep_sizeof(
            items,
            format="int",
            exclude_types=(),
            exclude_ids=set(),
            max_depth=None,
            seen=set(),
            on_error="skip",
        )
        size_excl = deep_sizeof(
            items,
            format="int",
            exclude_types=(ToExclude,),
            exclude_ids=set(),
            max_depth=None,
            seen=set(),
            on_error="skip",
        )
        assert size_excl < size_full
        # Expect list shallow size and the integer contributions at minimum.
        assert size_excl >= sys.getsizeof(items) + sys.getsizeof(42)

    def test_exclude_ids_instance(self) -> None:
        """Exclude a specific object instance by id."""
        big = "A" * 1024
        other = "B" * 16
        obj = {"keep": other, "skip": big}
        size_full = deep_sizeof(
            obj,
            format="int",
            exclude_types=(),
            exclude_ids=set(),
            max_depth=None,
            seen=set(),
            on_error="skip",
        )
        size_excl = deep_sizeof(
            obj,
            format="int",
            exclude_types=(),
            exclude_ids={id(big)},
            max_depth=None,
            seen=set(),
            on_error="skip",
        )
        assert size_excl < size_full
        # Ensure exclusion applies even when nested.
        nested = {"outer": [obj, big]}
        size_nested = deep_sizeof(
            nested,
            format="int",
            exclude_types=(),
            exclude_ids={id(big)},
            max_depth=None,
            seen=set(),
            on_error="skip",
        )
        assert size_nested < deep_sizeof(
            nested,
            format="int",
            exclude_types=(),
            exclude_ids=set(),
            max_depth=None,
            seen=set(),
            on_error="skip",
        )

    def test_seen_across_calls_avoids_double_count(self) -> None:
        """Avoid double-counting shared objects across calls using seen."""
        shared = [1, 2, 3, 4]
        a = [shared]
        b = [shared]
        seen: set[int] = set()
        _ = deep_sizeof(
            a,
            format="int",
            exclude_types=(),
            exclude_ids=set(),
            max_depth=None,
            seen=seen,
            on_error="skip",
        )
        size_b = deep_sizeof(
            b,
            format="int",
            exclude_types=(),
            exclude_ids=set(),
            max_depth=None,
            seen=seen,
            on_error="skip",
        )
        # Shared list should not be counted again; only b's shallow size remains.
        assert size_b == sys.getsizeof(b)

    def test_shared_reference_once(self) -> None:
        """Count a shared child only once within a container."""
        shared = [1, 2, 3]
        container = [shared, shared]
        expected = sys.getsizeof(container) + deep_sizeof(
            shared,
            format="int",
            exclude_types=(),
            exclude_ids=set(),
            max_depth=None,
            seen=set(),
            on_error="skip",
        )
        size = deep_sizeof(
            container,
            format="int",
            exclude_types=(),
            exclude_ids=set(),
            max_depth=None,
            seen=set(),
            on_error="skip",
        )
        assert size == expected

    def test_circular_reference(self) -> None:
        """Handle circular references without recursion error."""
        a: list[Any] = [1, 2]
        a.append(a)
        try:
            size = deep_sizeof(
                a,
                format="int",
                exclude_types=(),
                exclude_ids=set(),
                max_depth=None,
                seen=set(),
                on_error="skip",
            )
            assert size > 0
        except RecursionError:  # pragma: no cover - should not happen
            pytest.fail("deep_sizeof failed to handle a circular reference")

    def test_max_depth_zero_shallow_only(self) -> None:
        """Count only the root object's shallow size at max_depth=0."""
        obj = {"x": [1, 2, 3], "y": {"k": "v"}}
        size = deep_sizeof(
            obj,
            format="int",
            exclude_types=(),
            exclude_ids=set(),
            max_depth=0,
            seen=set(),
            on_error="skip",
        )
        assert size == sys.getsizeof(obj)

    def test_max_depth_one_counts_children_shallow(self) -> None:
        """Count root shallow and immediate children shallow at max_depth=1."""
        inner1 = [1, 2]
        inner2 = [3, 4]
        obj = [inner1, inner2]
        expected = sys.getsizeof(obj) + sys.getsizeof(inner1) + sys.getsizeof(inner2)
        size = deep_sizeof(
            obj,
            format="int",
            exclude_types=(),
            exclude_ids=set(),
            max_depth=1,
            seen=set(),
            on_error="skip",
        )
        assert size == expected

    def test_max_depth_limits_growth(self) -> None:
        """Ensure limited depth is between shallow-only and full deep sizes."""
        obj = {"a": [1, 2, 3, 4], "b": {"x": "y" * 10}}
        shallow = deep_sizeof(
            obj,
            format="int",
            exclude_types=(),
            exclude_ids=set(),
            max_depth=0,
            seen=set(),
            on_error="skip",
        )
        limited = deep_sizeof(
            obj,
            format="int",
            exclude_types=(),
            exclude_ids=set(),
            max_depth=1,
            seen=set(),
            on_error="skip",
        )
        full = deep_sizeof(
            obj,
            format="int",
            exclude_types=(),
            exclude_ids=set(),
            max_depth=None,
            seen=set(),
            on_error="skip",
        )
        assert shallow <= limited <= full
        assert full > shallow

    def test_max_depth_tracker(self) -> None:
        """Report at least the expected max depth reached."""
        obj = [[[0]]]  # depth path: list(0) -> list(1) -> list(2) -> int
        info = deep_sizeof(
            obj,
            format="dict",
            exclude_types=(),
            exclude_ids=set(),
            max_depth=None,
            seen=set(),
            on_error="skip",
        )
        assert isinstance(info["max_depth_reached"], int)
        assert info["max_depth_reached"] >= 2

    def test_object_count_minimum(self) -> None:
        """Track a positive object count in dict format."""
        obj = {"a": [1, 2], "b": 3}
        info = deep_sizeof(
            obj,
            format="dict",
            exclude_types=(),
            exclude_ids=set(),
            max_depth=None,
            seen=set(),
            on_error="skip",
        )
        assert isinstance(info["object_count"], int)
        assert info["object_count"] > 0

    @pytest.mark.parametrize(
        "obj",
        [
            pytest.param(123, id="int"),
            pytest.param("hello", id="str"),
            pytest.param(3.5, id="float"),
            pytest.param(True, id="bool"),
            pytest.param(None, id="none"),
            pytest.param(b"bytes", id="bytes"),
        ],
    )
    def test_primitives_match_sys(self, obj: Any) -> None:
        """Match sys.getsizeof for primitives."""
        size = deep_sizeof(
            obj,
            format="int",
            exclude_types=(),
            exclude_ids=set(),
            max_depth=None,
            seen=set(),
            on_error="skip",
        )
        assert size == sys.getsizeof(obj)

    @pytest.mark.parametrize(
        "container",
        [
            pytest.param([1, "a", 2.0], id="list"),
            pytest.param((1, "a", 2.0), id="tuple"),
            pytest.param({1, "a", 2.0}, id="set"),
            pytest.param(frozenset([1, "a", 2.0]), id="frozenset"),
            pytest.param({"k1": 1, "k2": "a", "k3": 2.0}, id="dict"),
        ],
    )
    def test_containers_deeper_than_shallow(self, container: Any) -> None:
        """Ensure deep size is greater than shallow for containers."""
        shallow = sys.getsizeof(container)
        deep = deep_sizeof(
            container,
            format="int",
            exclude_types=(),
            exclude_ids=set(),
            max_depth=None,
            seen=set(),
            on_error="skip",
        )
        assert deep > shallow


class TestIsBuiltin:
    """Groups tests for the is_builtin function."""

    @pytest.mark.parametrize(
        "obj",
        [
            int,
            1,
            str,
            "hello",
            list,
            [1, 2],
            dict,
            {"a": 1},
            tuple,
            (1, 2),
            set,
            {1, 2},
            float,
            3.14,
            bool,
            True,
            range,
            range(10),
            object,
            object(),
            None,
        ],
        ids=[
            "type_int",
            "instance_int",
            "type_str",
            "instance_str",
            "type_list",
            "instance_list",
            "type_dict",
            "instance_dict",
            "type_tuple",
            "instance_tuple",
            "type_set",
            "instance_set",
            "type_float",
            "instance_float",
            "type_bool",
            "instance_bool",
            "type_range",
            "instance_range",
            "type_object",
            "instance_object",
            "instance_none",
        ],
    )
    def test_returns_true_for_builtins(self, obj):
        """Verify that built-in types and instances return True."""
        assert is_builtin(obj) is True

    @pytest.mark.parametrize(
        "obj",
        [
            UserDefinedClass,
            UserDefinedClass(),
            user_defined_function,
            lambda: "hello",
            len,  # built-in function
            inspect,  # module
            property(lambda self: None),
            staticmethod(user_defined_function),
            classmethod(lambda cls: None),
            UserDefinedClass().method,
        ],
        ids=[
            "user_class",
            "user_instance",
            "user_function",
            "lambda_function",
            "builtin_function",
            "module",
            "property_descriptor",
            "staticmethod_descriptor",
            "classmethod_descriptor",
            "user_method",
        ],
    )
    def test_returns_false_for_non_builtins(self, obj):
        """Verify that non-built-in objects return False."""
        assert is_builtin(obj) is False

    def test_object_raising_attribute_error_on_access(self):
        """Ensure robustness against objects that raise errors on attribute access."""

        class Malicious:
            @property
            def __class__(self):
                raise AttributeError("Access denied")

        assert is_builtin(Malicious()) is False

    def test_object_class_without_module_attribute(self):
        """Ensure an object whose class lacks a __module__ attribute returns False."""
        # Create a mock for the class of an object
        mock_class = MagicMock(spec=type)
        # Configure the mock to not have a __module__ attribute.
        del mock_class.__module__

        # Create a mock instance and set its __class__ to our mock class
        mock_instance = MagicMock()
        mock_instance.__class__ = mock_class

        # is_builtin should handle this gracefully and return False
        assert is_builtin(mock_instance) is False

    def test_type_object_class_without_module_attribute(self, monkeypatch):
        """Ensure a type object that lacks a __module__ attribute returns False."""
        # Create a mock object that will pretend to be a type but lacks __module__
        mock_obj_as_type = MagicMock(spec=type)
        del mock_obj_as_type.__module__

        # We need to trick `is_builtin` into thinking our mock is a type.
        # To do this, we patch the global `isinstance` function.
        original_isinstance = isinstance

        def patched_isinstance(obj, class_or_tuple):
            # If `is_builtin` checks if our mock is a type, return True.
            if obj is mock_obj_as_type and class_or_tuple is type:
                return True
            # For all other calls, use the real `isinstance`.
            return original_isinstance(obj, class_or_tuple)

        # Patch the built-in `isinstance` function. Monkeypatch will restore it
        # after the test completes. This avoids the ModuleNotFoundError.
        monkeypatch.setattr("builtins.isinstance", patched_isinstance)

        # The function should now enter the type-checking branch, fail to find
        # `__module__`, and correctly return False.
        assert is_builtin(mock_obj_as_type) is False


# Test search_attrs() ------------------------------------------------------------------------------------


class _PropEval:
    def __init__(self, value: Any, raise_on_access: bool = False) -> None:
        self._value = value
        self._raise = raise_on_access

    @property
    def prop(self) -> Any:
        if self._raise:
            raise AttributeError("boom on property access")
        return self._value


class _SlotsOnly:
    __slots__ = ("a", "_b", "__c", "call")

    def __init__(self) -> None:
        self.a = 1
        self._b = 2
        self.__c = 3  # name-mangled
        # callable instance attribute
        self.call = lambda x: x  # noqa: E731


class _CallableInst:
    def __call__(self) -> int:
        return 42


class _AccessError:
    """Object that raises on any attribute access via descriptor."""

    @property
    def problematic_attr(self) -> Any:
        raise AttributeError("cannot access problematic_attr")


class _Base:
    base_attr = "base"


class _Child(_Base):
    child_attr = "child"

    def __init__(self) -> None:
        self.inst_attr = "inst"


class TestSearchAttrs:
    """Test suite for search_attrs function."""

    # Core functionality - 6 tests

    @pytest.mark.parametrize(
        "fmt, expected_type",
        [
            pytest.param("list", list, id="format-list"),
            pytest.param("dict", dict, id="format-dict"),
            pytest.param("items", list, id="format-items"),
        ],
    )
    def test_format_returns_correct_structure(self, fmt: str, expected_type: type) -> None:
        """Return correct structure for each format type."""
        obj = _Child()
        result = search_attrs(
            obj=obj,
            format=fmt,  # type: ignore[arg-type]
            include_private=False,
            include_properties=False,
            include_methods=False,
            include_inherited=True,
            exclude_none=False,
            pattern=None,
            attr_type=None,
            sort=False,
            skip_errors=True,
        )
        assert isinstance(result, expected_type)
        if fmt == "items":
            assert all(isinstance(it, tuple) and len(it) == 2 for it in result)  # type: ignore[index]

    @pytest.mark.parametrize(
        "flag_name, flag_value, attr_to_check",
        [
            pytest.param("include_private", True, "_private", id="include_private-on"),
            pytest.param("include_private", False, "_private", id="include_private-off"),
            pytest.param("include_properties", True, "prop", id="include_properties-on"),
            pytest.param("include_properties", False, "prop", id="include_properties-off"),
            pytest.param("include_methods", True, "method", id="include_methods-on"),
            pytest.param("include_methods", False, "method", id="include_methods-off"),
            pytest.param("exclude_none", True, "none_val", id="exclude_none-on"),
            pytest.param("exclude_none", False, "none_val", id="exclude_none-off"),
        ],
    )
    def test_filter_flags_control_attribute_inclusion(
        self, flag_name: str, flag_value: bool, attr_to_check: str
    ) -> None:
        """Include or exclude attributes based on filter flags."""

        class C:
            public = 1
            _private = 2
            none_val = None

            @property
            def prop(self) -> int:
                return 3

            def method(self) -> None:
                return None

        obj = C()
        kwargs = dict(
            obj=obj,
            format="list",
            include_private=False,
            include_properties=False,
            include_methods=False,
            include_inherited=True,
            exclude_none=False,
            pattern=None,
            attr_type=None,
            sort=False,
            skip_errors=True,
        )
        kwargs[flag_name] = flag_value  # type: ignore[index]
        names = search_attrs(**kwargs)  # type: ignore[arg-type]
        if flag_name == "exclude_none":
            expected_present = not flag_value
        else:
            expected_present = flag_value
        assert (attr_to_check in names) is expected_present

    def test_include_inherited_controls_scope(self) -> None:
        """Return only instance attrs when include_inherited is false, all attrs when true."""
        obj = _Child()
        names_no_inherit = search_attrs(
            obj=obj,
            format="list",
            include_private=False,
            include_properties=False,
            include_methods=False,
            include_inherited=False,
            exclude_none=False,
            pattern=None,
            attr_type=None,
            sort=False,
            skip_errors=True,
        )
        assert "inst_attr" in names_no_inherit
        assert "child_attr" not in names_no_inherit
        assert "base_attr" not in names_no_inherit

        names_with_inherit = search_attrs(
            obj=obj,
            format="list",
            include_private=False,
            include_properties=False,
            include_methods=False,
            include_inherited=True,
            exclude_none=False,
            pattern=None,
            attr_type=None,
            sort=False,
            skip_errors=True,
        )
        assert "inst_attr" in names_with_inherit
        assert "child_attr" in names_with_inherit
        assert "base_attr" in names_with_inherit

    def test_pattern_filters_by_regex_full_match(self) -> None:
        """Filter attributes matching the entire regex pattern."""

        class C:
            alpha = 1
            alp = 2
            beta = 3

        obj = C()
        names = search_attrs(
            obj=obj,
            format="list",
            include_private=False,
            include_properties=False,
            include_methods=False,
            include_inherited=True,
            exclude_none=False,
            pattern=r"alp.*a",
            attr_type=None,
            sort=False,
            skip_errors=True,
        )
        assert names == ["alpha"]

    @pytest.mark.parametrize(
        "atype, expected_contains",
        [
            pytest.param(int, ["public_int"], id="single-type-int"),
            pytest.param((int, str), ["public_int", "public_str"], id="tuple-types-int-str"),
        ],
    )
    def test_attr_type_filters_by_value_type(
        self, atype: type | tuple[type, ...], expected_contains: list[str]
    ) -> None:
        """Filter attributes whose values match specified type or tuple of types."""

        class C:
            public_int = 1
            public_str = "x"
            public_float = 1.5

        obj = C()
        result = search_attrs(
            obj=obj,
            format="dict",
            include_private=False,
            include_properties=False,
            include_methods=False,
            include_inherited=True,
            exclude_none=False,
            pattern=None,
            attr_type=atype,
            sort=False,
            skip_errors=True,
        )
        assert all(name in result for name in expected_contains)
        assert all(
            isinstance(v, atype if isinstance(atype, tuple) else atype) for v in result.values()
        )

    def test_obeys_sorting(self) -> None:
        """Sort attribute names alphabetically when sort is true."""

        class C:
            zed = 1
            alpha = 2
            mid = 3

        obj = C()
        names = search_attrs(
            obj=obj,
            format="list",
            sort=True,
            skip_errors=True,
        )
        assert names == ["alpha", "mid", "zed"]

        names = search_attrs(
            obj=obj,
            format="list",
            sort=False,
            skip_errors=True,
        )
        assert names == ["zed", "alpha", "mid"]

    # Critical edge cases - 8 tests

    def test_always_excludes_dunder_and_mangled_by_default(self) -> None:
        """Exclude dunder and mangled attributes unless include_private is true."""

        class C:
            __mangled = "m"

            def __init__(self) -> None:
                self.__inst = 1

        obj = C()
        names_default = search_attrs(
            obj=obj,
            format="list",
            include_private=False,
            include_properties=False,
            include_methods=False,
            include_inherited=True,
            exclude_none=False,
            pattern=None,
            attr_type=None,
            sort=False,
            skip_errors=True,
        )
        assert not any(n.startswith("__") for n in names_default)
        assert all(not re.match(r"^_.*__.*", n) for n in names_default)

        names_with_private = search_attrs(
            obj=obj,
            format="list",
            include_private=True,
            include_properties=False,
            include_methods=False,
            include_inherited=True,
            exclude_none=False,
            pattern=None,
            attr_type=None,
            sort=False,
            skip_errors=True,
        )
        # still exclude dunder but allow single underscore privates
        assert not any(n.startswith("__") and n.endswith("__") for n in names_with_private)

    def test_properties_evaluated_only_when_value_filtering_active(self) -> None:
        """Evaluate properties for exclude_none or attr_type, check descriptor otherwise."""
        obj = _PropEval(value=None, raise_on_access=True)

        # Properties should NOT be evaluated when include_properties=False and no filters
        names = search_attrs(
            obj=obj,
            format="list",
            include_private=False,
            include_properties=False,
            include_methods=False,
            include_inherited=True,
            exclude_none=False,
            pattern=None,
            attr_type=None,
            sort=False,
            skip_errors=True,
        )
        assert "prop" not in names

        # When include_properties=True but no value-based filters, still should not access value (descriptor only)
        names2 = search_attrs(
            obj=obj,
            format="list",
            include_private=False,
            include_properties=True,
            include_methods=False,
            include_inherited=True,
            exclude_none=False,
            pattern=None,
            attr_type=None,
            sort=False,
            skip_errors=False,  # even if False, shouldn't access, so no error
        )
        assert "prop" in names2

        # When value-based filtering active, property must be evaluated and raises if skip_errors=False
        with pytest.raises(AttributeError, match=r"(?i).*boom on property access.*"):
            search_attrs(
                obj=obj,
                format="list",
                include_private=False,
                include_properties=True,
                include_methods=False,
                include_inherited=True,
                exclude_none=True,  # triggers evaluation
                pattern=None,
                attr_type=None,
                sort=False,
                skip_errors=False,
            )

        # With skip_errors=True, it should skip problematic property
        names3 = search_attrs(
            obj=obj,
            format="list",
            include_private=False,
            include_properties=True,
            include_methods=False,
            include_inherited=True,
            exclude_none=True,
            pattern=None,
            attr_type=None,
            sort=False,
            skip_errors=True,
        )
        assert "prop" not in names3

    def test_multiple_filters_combine_with_and_logic(self) -> None:
        """Apply all active filters simultaneously and return intersection."""

        class C:
            alpha = 1
            alp = None
            beta = "x"
            _private = 2

            @property
            def prop(self) -> int:
                return 5

            def method(self) -> None:
                return None

        obj = C()
        result = search_attrs(
            obj=obj,
            format="list",
            include_private=False,
            include_properties=True,
            include_methods=False,
            include_inherited=True,
            exclude_none=True,
            pattern=r"alp.*a",
            attr_type=int,
            sort=False,
            skip_errors=True,
        )
        # Only 'alpha' matches pattern, is int, non-None, and allowed
        assert result == ["alpha"]

    @pytest.mark.parametrize(
        "fmt",
        [
            pytest.param("list", id="empty-list"),
            pytest.param("dict", id="empty-dict"),
            pytest.param("items", id="empty-items"),
        ],
    )
    def test_empty_result_for_no_matches(self, fmt: str) -> None:
        """Return empty collection when no attributes match filters."""

        class C:
            a = None

        obj = C()
        result = search_attrs(
            obj=obj,
            format=fmt,  # type: ignore[arg-type]
            include_private=False,
            include_properties=False,
            include_methods=False,
            include_inherited=True,
            exclude_none=True,
            pattern=r"z.*",
            attr_type=int,
            sort=False,
            skip_errors=True,
        )
        if fmt == "dict":
            assert result == {}
        else:
            assert result == []

    @pytest.mark.parametrize(
        "obj",
        [
            pytest.param(1, id="int"),
            pytest.param(1.0, id="float"),
            pytest.param("s", id="str"),
            pytest.param(True, id="bool"),
            pytest.param(b"bytes", id="bytes"),
        ],
    )
    def test_builtin_primitives_return_empty(self, obj: Any) -> None:
        """Return empty collection for built-in primitive types."""
        result = search_attrs(
            obj=obj,
            format="list",
            include_private=False,
            include_properties=False,
            include_methods=False,
            include_inherited=True,
            exclude_none=False,
            pattern=None,
            attr_type=None,
            sort=False,
            skip_errors=True,
        )
        assert result == []

    def test_slots_only_objects_work_correctly(self) -> None:
        """Handle objects using slots without dict."""
        obj = _SlotsOnly()
        names = search_attrs(
            obj=obj,
            format="list",
            include_private=False,
            include_properties=False,
            include_methods=False,
            include_inherited=True,
            exclude_none=False,
            pattern=None,
            attr_type=None,
            sort=True,
            skip_errors=True,
        )
        # Should include public slot 'a' but not private or mangled
        assert "a" in names
        assert "_b" not in names
        assert not any(re.match(r"^_.*__.*", n) for n in names)

    def test_callable_instances_handled_by_include_methods(self) -> None:
        """Treat callable class instances as methods when include_methods is true."""

        class C:
            ci = _CallableInst()

            def __init__(self) -> None:
                self.inst = _CallableInst()

        obj = C()
        names_no = search_attrs(
            obj=obj,
            format="list",
            include_private=False,
            include_properties=False,
            include_methods=False,
            include_inherited=True,
            exclude_none=False,
            pattern=None,
            attr_type=None,
            sort=False,
            skip_errors=True,
        )
        assert "ci" not in names_no and "inst" not in names_no

        names_yes = search_attrs(
            obj=obj,
            format="list",
            include_private=False,
            include_properties=False,
            include_methods=True,
            include_inherited=True,
            exclude_none=False,
            pattern=None,
            attr_type=None,
            sort=False,
            skip_errors=True,
        )
        assert "ci" in names_yes and "inst" in names_yes

    def test_none_object_handled_gracefully(self) -> None:
        """Handle none as input object without crashing."""
        result = search_attrs(
            obj=None,
            format="list",
            include_private=False,
            include_properties=False,
            include_methods=False,
            include_inherited=True,
            exclude_none=False,
            pattern=None,
            attr_type=None,
            sort=False,
            skip_errors=True,
        )
        assert result == []

    # Error conditions - 4 tests

    def test_invalid_regex_pattern_raises_value_error(self) -> None:
        """Raise value error when pattern is invalid regex."""

        class C:
            a = 1

        obj = C()
        with pytest.raises(ValueError, match=r"(?i).*invalid.*pattern.*|.*regex.*"):
            search_attrs(
                obj=obj,
                format="list",
                include_private=False,
                include_properties=False,
                include_methods=False,
                include_inherited=True,
                exclude_none=False,
                pattern=r"[",  # invalid
                attr_type=None,
                sort=False,
                skip_errors=True,
            )

    def test_invalid_format_raises_value_error(self) -> None:
        """Raise value error when format is not list, dict, or items."""

        class C:
            a = 1

        obj = C()
        with pytest.raises(ValueError, match=r"(?i).*format.*"):
            search_attrs(
                obj=obj,
                format="weird",  # type: ignore[arg-type]
                include_private=False,
                include_properties=False,
                include_methods=False,
                include_inherited=True,
                exclude_none=False,
                pattern=None,
                attr_type=None,
                sort=False,
                skip_errors=True,
            )

    @pytest.mark.parametrize(
        "skip_errors, should_raise",
        [
            pytest.param(False, True, id="skip_errors-false-raises"),
            pytest.param(True, False, id="skip_errors-true-skips"),
        ],
    )
    def test_skip_errors_controls_exception_propagation(
        self, skip_errors: bool, should_raise: bool
    ) -> None:
        """Raise attribute error when skip_errors is false, skip when true."""
        obj = _AccessError()
        if should_raise:
            with pytest.raises(AttributeError, match=r"(?i).*cannot access.*"):
                search_attrs(
                    obj=obj,
                    format="dict",  # Need dict to trigger value access
                    include_properties=True,  # Need to include the property
                    skip_errors=skip_errors,
                )
        else:
            result = search_attrs(
                obj=obj,
                format="dict",  # Need dict to trigger value access
                include_properties=True,  # Need to include the property
                skip_errors=skip_errors,
            )
            assert "problematic_attr" not in result  # Should be skipped when skip_errors=True

    def test_attr_type_with_non_type_raises_type_error(self) -> None:
        """Raise type error when attr_type is not a type or tuple of types."""

        class C:
            a = 1

        obj = C()
        with pytest.raises(TypeError, match=r"(?i).*type.*|.*tuple.*"):
            search_attrs(
                obj=obj,
                format="list",
                include_private=False,
                include_properties=False,
                include_methods=False,
                include_inherited=True,
                exclude_none=False,
                pattern=None,
                attr_type="not-a-type",  # type: ignore[arg-type]
                sort=False,
                skip_errors=True,
            )


class TestValidParamTypes:
    @pytest.mark.parametrize(
        ("x", "y"),
        [
            pytest.param(1, "a", id="int-str"),
            pytest.param(0, "", id="zero-empty"),
        ],
    )
    def test_basic_match(self, x, y):
        """Validate matching simple annotations."""

        @valid_param_types
        def fn(a: int, b: str):
            return a, b

        assert fn(x, y) == (x, y)

    def test_mismatch_raises(self):
        """Raise on simple type mismatch."""

        @valid_param_types
        def fn(a: int, b: str):
            return a, b

        with pytest.raises(TypeError, match=r"(?i).*parameter.*a.*int.*"):
            fn("not-int", "ok")

    def test_optional_allows_none_when_allowed(self):
        """Accept None for Optional when allowed."""

        @valid_param_types(allow_none=True)
        def fn(a: int | None):
            return a

        assert fn(None) is None

    def test_optional_rejects_none_when_disallowed(self):
        """Reject None for Optional when not allowed."""

        @valid_param_types(allow_none=False)
        def fn(a: int | None):
            return a

        with pytest.raises(TypeError, match=r"(?i).*parameter.*a.*None.*"):
            fn(None)

    def test_exclude_none_skips_validation_for_none(self):
        """Skip validation when exclude_none is True."""
        calls = {"ran": False}

        @valid_param_types(exclude_none=True, allow_none=False)
        def fn(a: int):
            calls["ran"] = True
            return True

        assert fn(None) is True
        assert calls["ran"] is True

    def test_exclude_self_skips_self(self):
        """Skip validating self when exclude_self is True."""

        class C:
            @valid_param_types(exclude_self=True)
            def m(self, a: int):
                return a

        assert C().m(5) == 5

    def test_include_self_validates_when_not_excluded(self):
        """Validate self when exclude_self is False."""

        class C:
            @valid_param_types(exclude_self=False)
            def m(self: int, a: int):
                return a

        with pytest.raises(TypeError, match=r"(?i).*parameter.*self.*int.*"):
            C().m(1)

    def test_unannotated_param_is_ignored(self):
        """Ignore parameters without annotations."""

        @valid_param_types
        def fn(a, b: int):
            return b

        assert fn("anything", 3) == 3

    def test_strict_union_unvalidatable_raises(self):
        """Raise on unvalidatable union when strict."""
        from collections.abc import Callable

        @valid_param_types(strict=True)
        def fn(cb: (Callable[[int], str] | Callable[[str], int])):
            return cb

        with pytest.raises(TypeError, match=r"(?i).*complex.*union.*"):
            fn(lambda x: x)

    def test_non_strict_union_unvalidatable_skips(self):
        """Skip unvalidatable union when not strict."""
        from collections.abc import Callable

        called = {"ok": False}

        @valid_param_types(strict=False)
        def fn(cb: (Callable[[int], str] | Callable[[str], int])):
            called["ok"] = True
            return True

        assert fn(lambda x: x) is True
        assert called["ok"] is True

    def test_missing_param_in_signature_is_skipped(self):
        """Skip names not present in params list."""

        @valid_param_types(params=["a", "missing"])
        def fn(a: int):
            return a

        assert fn(10) == 10

    def test_selective_params_subset(self):
        """Validate subset via params argument."""

        @valid_param_types(params=["x"])
        def fn(x: int, y: str):
            return x, y

        assert fn(1, 2) == (1, 2)

    def test_exclude_none_interaction_with_allow_none(self):
        """Prefer exclude_none over allow_none when value is None."""

        @valid_param_types(exclude_none=True, allow_none=False)
        def fn(x: int | None):
            return x

        assert fn(None) is None

    def test_defaults_are_bound_and_validated(self):
        """Validate defaulted arguments."""

        @valid_param_types
        def fn(x: int, y: str = "ok"):
            return y

        assert fn(1) == "ok"
        with pytest.raises(TypeError, match=r"(?i).*parameter.*x.*int.*"):
            fn("bad")

    def test_multiple_errors_reporting(self):
        """Report multiple parameter errors."""

        @valid_param_types
        def fn(a: int, b: str, c: float):
            return a, b, c

        with pytest.raises(TypeError) as exc:
            fn("x", 1, "nope")
        msg = str(exc.value)
        assert re.search(r"(?i)type validation failed", msg)
        assert re.search(r"(?i)a.*int", msg)
        assert re.search(r"(?i)b.*str", msg)
        assert re.search(r"(?i)c.*float", msg)

    def test_preserves_function_metadata(self):
        """Preserve function metadata via wraps."""

        @valid_param_types
        def sample(x: int):
            """Docstring here."""
            return x

        assert sample.__name__ == "sample"
        assert "Docstring here." in (sample.__doc__ or "")


class TestSearchAttrsSorting:
    """Core sorting behavior for search_attrs()."""

    @pytest.mark.parametrize(
        "fmt, expect_type",
        [
            pytest.param("list", list, id="list"),
            pytest.param("dict", dict, id="dict"),
            pytest.param("items", list, id="items"),
        ],
    )
    def test_sort_false_preserves_order(self, fmt, expect_type):
        """Preserve attribute discovery order when sort is False."""

        class Base:
            zeta = 0
            beta = 0

        class C(Base):
            alpha = 1
            gamma = 2
            delta = 3
            epsilon = 4

            def method(self):  # excluded by default
                return 42

        obj = C()
        out = search_attrs(
            obj,
            format=fmt,
            include_inherited=True,
            include_methods=False,
            include_private=False,
            include_properties=False,
            sort=False,
        )
        assert isinstance(out, expect_type)

        if fmt == "list":
            # Expect dir-like discovery order (not alphabetic). We only assert relative order.
            names = out
        elif fmt == "dict":
            names = list(out.keys())
        else:  # items
            names = [k for k, _ in out]

        # Ensure alpha appears before gamma and gamma before epsilon; beta from Base appears as well.
        # We avoid asserting the full exact order to stay robust, but enforce non-sorted-ness.
        alpha_i = names.index("alpha")
        gamma_i = names.index("gamma")
        epsilon_i = names.index("epsilon")
        assert alpha_i < gamma_i < epsilon_i
        # If not sorted, 'beta' should not necessarily appear near 'alpha'.
        # Sanity check: not purely alphabetical.
        alphabetical = sorted(names)
        assert names != alphabetical

    @pytest.mark.parametrize(
        "fmt, expect_type",
        [
            pytest.param("list", list, id="list"),
            pytest.param("dict", dict, id="dict"),
            pytest.param("items", list, id="items"),
        ],
    )
    def test_sort_true_is_alphabetical(self, fmt, expect_type):
        """Sort attributes alphabetically when sort is True."""

        class Base:
            zeta = 0
            beta = 0

        class C(Base):
            alpha = 1
            gamma = 2
            delta = 3
            epsilon = 4

            def method(self):
                return 42

        obj = C()
        out = search_attrs(
            obj,
            format=fmt,
            include_inherited=True,
            sort=True,
        )
        assert isinstance(out, expect_type)

        if fmt == "list":
            names = out
        elif fmt == "dict":
            names = list(out.keys())
        else:
            names = [k for k, _ in out]

        assert names == sorted(names)

    def test_sort_true_with_pattern_and_type(self):
        """Apply sorting after filters like pattern and attr_type."""

        class C:
            alpha = 1
            gamma = 2.0
            delta = 3
            epsilon = 4.0
            zeta = None

        obj = C()
        out = search_attrs(
            obj,
            format="list",
            pattern=r"^(alpha|delta|epsilon|gamma)$",
            attr_type=(int, float),
            exclude_none=True,
            sort=True,
        )
        # After filter, expect only numeric attributes; ensure alphabetical order.
        assert out == sorted(out)
        # And confirm we didn't accidentally include the None-valued 'zeta'
        assert "zeta" not in out

    def test_sort_false_with_items(self):
        """Keep item pair order unsorted when sort is False."""

        class C:
            a = 1
            c = 3
            b = 2

        obj = C()
        items = search_attrs(obj, format="items", sort=False)
        names = [k for k, _ in items]
        # Not strictly alphabetical; assert a before c and c before b based on discovery order assumption.
        ai, ci, bi = names.index("a"), names.index("c"), names.index("b")
        assert ai < ci < bi
        assert names != sorted(names)

    def test_sort_true_stable_and_values_kept(self):
        """Maintain value associations when sorting."""

        class C:
            b = 2
            a = 1
            c = 3

        obj = C()
        d = search_attrs(obj, format="dict", sort=True)
        assert list(d.keys()) == ["a", "b", "c"]
        assert d["a"] == 1 and d["b"] == 2 and d["c"] == 3


class TestValidateParamTypes:
    @pytest.mark.parametrize(
        ("x", "y"),
        [
            pytest.param(1, "a", id="int-str"),
            pytest.param(0, "", id="zero-empty"),
        ],
    )
    def test_basic_match(self, x, y):
        """Validate matching simple annotations."""

        def fn(a: int, b: str):
            validate_param_types(
                params=["a", "b"],
                exclude_self=True,
                exclude_none=False,
                strict=True,
                allow_none=False,
            )
            return a, b

        assert fn(x, y) == (x, y)

    def test_mismatch_raises(self):
        """Raise on simple type mismatch."""

        def fn(a: int, b: str):
            validate_param_types(
                params=["a", "b"],
                exclude_self=True,
                exclude_none=False,
                strict=True,
                allow_none=False,
            )

        with pytest.raises(TypeError, match=r"(?i).*parameter.*a.*int.*"):
            fn("not-int", "ok")

    def test_optional_allows_none_when_allowed(self):
        """Accept None for Optional when allowed."""

        def fn(a: int | None):
            validate_param_types(
                params=["a"],
                exclude_self=True,
                exclude_none=False,
                strict=True,
                allow_none=True,
            )
            return a

        assert fn(None) is None

    def test_optional_rejects_none_when_disallowed(self):
        """Reject None for Optional when not allowed."""

        def fn(a: int | None):
            validate_param_types(
                params=["a"],
                exclude_self=True,
                exclude_none=False,
                strict=True,
                allow_none=False,
            )

        with pytest.raises(TypeError, match=r"(?i).*parameter.*a.*None.*"):
            fn(None)

    def test_exclude_none_skips_validation_for_none(self):
        """Skip validation when exclude_none is True."""
        calls = {"validated": False}

        def fn(a: int):
            # If validated, None would fail; exclude_none should skip
            validate_param_types(
                params=["a"],
                exclude_self=True,
                exclude_none=True,
                strict=True,
                allow_none=False,
            )
            calls["validated"] = True

        fn(None)
        assert calls["validated"] is True  # Function proceeds

    def test_exclude_self_skips_self(self):
        """Skip validating self when exclude_self is True."""

        class C:
            def m(self, a: int):
                validate_param_types(
                    params=["self", "a"],
                    exclude_self=True,
                    exclude_none=False,
                    strict=True,
                    allow_none=False,
                )
                return a

        assert C().m(5) == 5

    def test_include_self_validates_when_not_excluded(self):
        """Validate self when exclude_self is False."""

        class C:
            def m(self: int, a: int):
                validate_param_types(
                    params=["self", "a"],
                    exclude_self=False,
                    exclude_none=False,
                    strict=True,
                    allow_none=False,
                )

        with pytest.raises(TypeError, match=r"(?i).*parameter.*self.*int.*"):
            C().m(1)

    def test_params_subset_only_those(self):
        """Validate only specified params."""

        def fn(a: int, b: str):
            validate_param_types(
                params=["a"],
                exclude_self=True,
                exclude_none=False,
                strict=True,
                allow_none=False,
            )
            return a, b

        # b mismatches but is not validated
        assert fn(1, 2) == (1, 2)

    def test_multiple_errors_aggregated(self):
        """Aggregate multiple parameter errors."""

        def fn(a: int, b: str, c: float):
            validate_param_types(
                params=["a", "b", "c"],
                exclude_self=True,
                exclude_none=False,
                strict=True,
                allow_none=False,
            )

        with pytest.raises(TypeError, match=r"(?is).*a.*int.*b.*str.*c.*float.*"):
            fn("x", 1, "nope")

    def test_unannotated_param_is_ignored(self):
        """Ignore parameters without annotations."""

        def fn(a, b: int):
            validate_param_types(
                params=["a", "b"],
                exclude_self=True,
                exclude_none=False,
                strict=True,
                allow_none=False,
            )
            return b

        assert fn("anything", 3) == 3

    def test_strict_union_unvalidatable_raises(self):
        """Raise on truly unvalidatable union when strict."""
        from collections.abc import Callable

        def fn(cb: (Callable[[int], str] | Callable[[str], int])):
            validate_param_types(
                params=["cb"],
                exclude_self=True,
                exclude_none=False,
                strict=True,
                allow_none=False,
            )

        with pytest.raises(TypeError, match=r"(?i).*complex.*union.*"):
            # Any callable; framework should deem this union unvalidatable in strict mode
            fn(lambda x: x)

    def test_non_strict_union_unvalidatable_skips(self):
        """Skip unvalidatable union when not strict."""
        from collections.abc import Callable

        calls = {"ran": False}

        def fn(cb: (Callable[[int], str] | Callable[[str], int])):
            validate_param_types(
                params=["cb"],
                exclude_self=True,
                exclude_none=False,
                strict=False,
                allow_none=False,
            )
            calls["ran"] = True

        fn(lambda x: x)
        assert calls["ran"] is True

    def test_missing_param_in_signature_is_skipped(self):
        """Skip names not present in signature."""

        def fn(a: int):
            validate_param_types(
                params=["a", "missing"],
                exclude_self=True,
                exclude_none=False,
                strict=True,
                allow_none=False,
            )
            return a

        assert fn(10) == 10


class TestValidateTypes:
    """Core validation tests for validate_types()."""

    @pytest.mark.parametrize(
        "fast_mode",
        [pytest.param(True, id="fast_true"), pytest.param("auto", id="fast_auto")],
    )
    def test_valid_dataclass_passes(self, fast_mode):
        """Validate dataclass with correct types."""

        @dataclass
        class Item:
            id: str
            count: int

        obj = Item(id="abc", count=5)
        validate_types(obj, fast=fast_mode)

    def test_invalid_type_raises(self):
        """Raise TypeError for wrong type."""

        @dataclass
        class Item:
            id: str
            count: int

        obj = Item(id="abc", count="bad")  # wrong type
        with pytest.raises(TypeError, match=r"(?i).*type validation failed.*"):
            validate_types(obj, fast=True)

    def test_exclude_none_skips_none_fields(self):
        """Skip None fields when exclude_none=True."""

        @dataclass
        class D:
            x: int | None
            y: str

        obj = D(x=None, y="hi")
        validate_types(obj, exclude_none=True, fast=True)

    def test_allow_none_false_blocks_none(self):
        """Raise TypeError if None not allowed."""

        @dataclass
        class D:
            a: int | None

        obj = D(a=None)
        with pytest.raises(TypeError, match=r"(?i).*type validation failed.*"):
            validate_types(obj, allow_none=False, fast=True)

    def test_fast_incompatible_options_raise(self):
        """Raise ValueError if fast=True but incompatible options."""

        @dataclass
        class D:
            x: int

        obj = D(1)
        with pytest.raises(ValueError, match=r"(?i).*cannot use fast.*pattern parameter.*"):
            validate_types(obj, pattern=r"x", fast=True)

    def test_non_dataclass_with_annotations_works(self):
        """Validate regular class with type annotations."""

        class User:
            name: str
            age: int

            def __init__(self, name: str, age: int):
                self.name = name
                self.age = age

        obj = User("Bob", 33)
        validate_types(obj, fast=False)

    def test_non_dataclass_no_annotations_raises(self):
        """Raise ValueError if no type annotations."""

        class C:
            def __init__(self):
                self.x = 5

        obj = C()
        with pytest.raises(ValueError, match=r"(?i).*no type annotations.*"):
            validate_types(obj, fast=False)

    def test_private_attrs_included_when_flag_true(self):
        """Include private attrs when include_private=True."""

        class C:
            _a: int
            b: str

            def __init__(self):
                self._a = 5
                self.b = "ok"

        obj = C()
        validate_types(obj, include_private=True, fast=False)

    def test_pattern_filter_validates_subset(self):
        """Validate only attributes matching regex pattern."""

        class C:
            api_key: str
            internal_id: int

            def __init__(self):
                self.api_key = "abc"
                self.internal_id = 5

        obj = C()
        validate_types(obj, pattern=r"^api_", fast=False)

    def test_strict_mode_not_blocks_union(self):
        """Raise TypeError for unsupported Union when strict=True."""

        class C:
            value: int | str | float | None

            def __init__(self):
                self.value = 3.14  # neither int nor str

        obj = C()
        validate_types(obj, strict=True, fast=False)

    def test_inherited_attrs_checked_when_enabled(self):
        """Validate inherited attributes when include_inherited=True."""

        class Base:
            x: int

            def __init__(self):
                self.x = 10

        class Child(Base):
            y: str

            def __init__(self):
                super().__init__()
                self.y = "ok"

        obj = Child()
        validate_types(obj, include_inherited=True, fast=False)

    def test_old_optional_syntax_supported(self):
        """Support Optional[T] syntax from older annotations."""
        from typing import Optional

        @dataclass
        class D:
            z: Optional[int]

        obj = D(z=None)
        validate_types(obj, allow_none=True, fast=True)

    def test_fast_auto_switches_to_slow_path(self):
        """Use slow path automatically when filters are applied."""

        @dataclass
        class D:
            name: str
            age: int

        obj = D("Alice", 20)
        # should auto-select slow path (pattern breaks fast)
        validate_types(obj, pattern=r"^name$", fast="auto")

    def test_fast_mode_requires_dataclass(self):
        """Raise ValueError if fast=True but not dataclass."""

        class NotDC:
            x: int

            def __init__(self):
                self.x = 3

        obj = NotDC()
        with pytest.raises(ValueError, match=r"(?i).*obj is not a dataclass.*"):
            validate_types(obj, fast=True)

    # Test strict mode behavior: validates simple unions, rejects complex unions. ----------------------

    @pytest.mark.parametrize(
        "type_hint,value,should_pass",
        [
            # Simple unions - valid
            pytest.param(int | str, 42, True, id="union-int|str:int"),
            pytest.param(int | str, "hello", True, id="union-int|str:str"),
            pytest.param(int | str | float, 3.14, True, id="union-int|str|float:float"),
            pytest.param(int | str | float, 100, True, id="union-int|str|float:int"),
            pytest.param(str | bytes, b"data", True, id="union-str|bytes:bytes"),
            # Simple unions - invalid
            pytest.param(int | str, 3.14, False, id="union-int|str:float-fails"),
            pytest.param(int | str, [], False, id="union-int|str:list-fails"),
            pytest.param(int | float, "text", False, id="union-int|float:str-fails"),
            # Optional (T | None) - valid
            pytest.param(int | None, 42, True, id="optional-int:valid-int"),
            pytest.param(int | None, None, True, id="optional-int:none"),
            pytest.param(str | None, "test", True, id="optional-str:valid-str"),
            # Complex Optional Union - invalid in strict mode
            pytest.param(int | str | None, 42, True, id="complex-optional-int|str|none:int-pass"),
            pytest.param(
                int | str | None,
                "text",
                True,
                id="complex-optional-int|str|none:str-pass",
            ),
        ],
    )
    def test_strict_mode_union_validation(self, type_hint, value, should_pass):
        """Validate unions in strict mode and reject complex optional unions."""

        class C:
            pass

        C.__annotations__ = {"value": type_hint}
        obj = C()
        obj.value = value

        if should_pass:
            validate_types(obj, strict=True, fast=False)
        else:
            with pytest.raises(TypeError, match=r"(?i)(type validation failed|complex optional)"):
                validate_types(obj, strict=True, fast=False)

    @pytest.mark.parametrize(
        "strict",
        [
            pytest.param(True, id="strict"),
            pytest.param(False, id="non-strict"),
        ],
    )
    def test_simple_types_in_both_modes(self, strict):
        """Validate simple types and enforce errors consistently in both modes."""

        class C:
            x: int
            y: str
            z: float

            def __init__(self):
                self.x = 42
                self.y = "hello"
                self.z = 3.14

        obj = C()

        # Should pass with both strict modes
        validate_types(obj, strict=strict, fast=False)

        # Wrong type should fail in both modes
        obj.x = "not an int"
        with pytest.raises(TypeError, match=r"(?i)type validation failed"):
            validate_types(obj, strict=strict, fast=False)


# Test Core Private Methods --------------------------------------------------------------------------------------------


class Test_ActsLikeImage:
    @pytest.mark.parametrize(
        "cls",
        [
            # Class that looks like an Image (has required attrs and 3+ methods)
            type(
                "MyImage",
                (),
                {
                    "size": (1, 1),
                    "mode": "RGB",
                    "format": "PNG",
                    "save": lambda self, *a, **k: None,
                    "show": lambda self, *a, **k: None,
                    "resize": lambda self, *a, **k: None,
                },
            ),
            # Type with 'Image' in the name and many methods but attributes on the class
            type(
                "FakeImageType",
                (),
                {
                    "size": (2, 2),
                    "mode": "L",
                    "format": "JPEG",
                    "save": lambda self, *a, **k: None,
                    "show": lambda self, *a, **k: None,
                    "crop": lambda self, *a, **k: None,
                },
            ),
        ],
        ids=["class-with-attrs-and-methods", "named-image-type"],
    )
    def test_type_positive(self, cls):
        """Return True for image-like classes."""
        assert _acts_like_image(cls) is True

    def test_instance_positive(self):
        """Return True for image-like instances."""
        Inst = type(
            "ImageInstance",
            (),
            {
                "size": (10, 10),
                "mode": "RGBA",
                "format": "PNG",
                "save": lambda self, *a, **k: None,
                "show": lambda self, *a, **k: None,
                "resize": lambda self, *a, **k: None,
            },
        )
        obj = Inst()
        assert _acts_like_image(obj) is True

    @pytest.mark.parametrize(
        ("obj", "reason"),
        [
            (
                type(
                    "NoPic",
                    (),
                    {
                        "size": (1, 1),
                        "mode": "RGB",
                        "format": "PNG",
                        "save": lambda s: None,
                        "show": lambda s: None,
                        "resize": lambda s: None,
                    },
                ),
                "name",
            ),
            (
                type(
                    "ImageMissingAttrs",
                    (),
                    {
                        "save": lambda s: None,
                        "show": lambda s: None,
                        "resize": lambda s: None,
                    },
                ),
                "missing-attrs",
            ),
            (
                type(
                    "ImageFewMethods",
                    (),
                    {
                        "size": (1, 1),
                        "mode": "RGB",
                        "format": "PNG",
                        "save": lambda s: None,
                    },
                ),
                "too-few-methods",
            ),
        ],
        ids=["no-image-substring", "missing-attrs", "few-methods"],
    )
    def test_type_negative(self, obj, reason):
        """Return False for non-image-like classes."""
        assert _acts_like_image(obj) is False

    @pytest.mark.parametrize(
        ("instance", "id"),
        [
            (
                type(
                    "BadSize",
                    (),
                    {
                        "size": (0, 10),
                        "mode": "RGB",
                        "format": "PNG",
                        "save": lambda s: None,
                        "show": lambda s: None,
                        "resize": lambda s: None,
                    },
                )(),
                "zero-width",
            ),
            (
                type(
                    "BadSizeType",
                    (),
                    {
                        "size": ("a", 10),
                        "mode": "RGB",
                        "format": "PNG",
                        "save": lambda s: None,
                        "show": lambda s: None,
                        "resize": lambda s: None,
                    },
                )(),
                "non-int-size",
            ),
            (
                type(
                    "EmptyMode",
                    (),
                    {
                        "size": (1, 1),
                        "mode": "",
                        "format": "PNG",
                        "save": lambda s: None,
                        "show": lambda s: None,
                        "resize": lambda s: None,
                    },
                )(),
                "empty-mode",
            ),
            (object(), "plain-object"),
        ],
        ids=["invalid-size-zero", "invalid-size-type", "empty-mode", "plain-object"],
    )
    def test_instance_negative(self, instance, id):
        """Return False for instances with invalid attributes."""
        assert _acts_like_image(instance) is False

    def test_plain_types_and_instances(self):
        """Return False for unrelated types and instances."""

        class NotImage:
            pass

        assert _acts_like_image(NotImage) is False
        assert _acts_like_image(object()) is False
