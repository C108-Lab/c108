#
# C108 - ABC Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import inspect
import warnings
import sys
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

# Third-party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.abc import (ObjectInfo,
                      _acts_like_image,
                      attrs_search,
                      deep_sizeof,
                      is_builtin,
                      )


# Local Classes & Methods ----------------------------------------------------------------------------------------------


class AttrsObj:
    """A complex class with various attribute types for testing."""
    public_attr = "value"
    _private_attr = 123
    public_none_attr: str | None = None
    _private_none_attr: int | None = None
    __mangled_attr = "mangled"  # This will be mangled to _TestSubject__mangled_attr

    def __init__(self) -> None:
        """Initialize instance attributes."""
        self.instance_attr = "instance_value"
        self._instance_private_attr = 456
        self.instance_none_attr: str | None = None

    @property
    def public_property(self) -> str:
        """A public property."""
        return "prop_value"

    @property
    def _private_property(self) -> str:
        """A private property."""
        return "prop_value_private"

    @property
    def none_property(self) -> None:
        """A property that returns None."""
        return None

    @property
    def error_property(self) -> Any:
        """A property that raises an exception."""
        raise ValueError("This property fails on access.")

    def public_method(self) -> None:
        """A public method that should be ignored."""
        pass

    def _private_method(self) -> None:
        """A private method that should be ignored."""
        pass


class BuggySize:
    """Object whose __sizeof__ raises to test error handling."""

    def __sizeof__(self) -> int:
        raise RuntimeError("Broken __sizeof__ for testing")


class Example:
    """A class with various attribute types for testing."""
    regular_attribute = "value"

    @property
    def working_property(self) -> str:
        """A standard, functioning property."""
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


# Tests for Classes ----------------------------------------------------------------------------------------------------

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
            ObjectInfo(type=int, size=[1, 2], unit=["bytes"], deep_size=None, fully_qualified=False)

    def test_to_str_mismatch_after_mutation_raises(self):
        """Raise on size/unit mismatch detected by to_str."""
        info = ObjectInfo(type=int, size=[1, 2], unit=["a", "b"], deep_size=None, fully_qualified=False)
        info.unit = ["a"]  # induce mismatch post-init
        with pytest.raises(ValueError, match=r"(?i).*size and unit lists must.*"):
            info.to_str(deep_size=False)

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
        assert "ObjectInfo(type=int" in r
        assert "unit=bytes" in r


# Tests for methods ----------------------------------------------------------------------------------------------------

@pytest.fixture
def subject() -> AttrsObj:
    """Provide an instance of the AttrsObj class."""
    return AttrsObj()


class TestAttrsSearch:
    """Test suite for the attrs_search function."""

    def test_include_private(self, subject: AttrsObj):
        """Verify it includes private attributes when requested."""
        result = attrs_search(subject, include_private=True, sort=True)
        expected = [
            "_instance_private_attr",
            "_private_attr",
            "_private_none_attr",
            "instance_attr",
            "instance_none_attr",
            "public_attr",
            "public_none_attr",
        ]
        assert sorted(result) == expected

    def test_include_property(self, subject: AttrsObj):
        """Verify it includes properties when requested."""
        result = attrs_search(subject, include_property=True, sort=True)
        expected = [
            "error_property",
            "instance_attr",
            "instance_none_attr",
            "none_property",
            "public_attr",
            "public_none_attr",
            "public_property",
        ]
        assert sorted(result) == expected

    def test_exclude_none_attrs(self, subject: AttrsObj):
        """Verify it excludes attributes with a value of None."""
        result = attrs_search(subject, include_none_attrs=False, sort=True)
        expected = ["instance_attr", "public_attr"]
        assert sorted(result) == expected

    def test_exclude_none_properties(self, subject: AttrsObj):
        """Verify it excludes properties that return None."""
        result = attrs_search(
            subject,
            include_property=True,
            include_none_properties=False,
            sort=True
        )
        expected = [
            "instance_attr",
            "instance_none_attr",
            "public_attr",
            "public_none_attr",
            "public_property",
        ]
        assert sorted(result) == expected

    def test_sort_output(self, subject: AttrsObj):
        """Verify it sorts the output alphabetically when requested."""
        unsorted_result = attrs_search(subject, sort=False)
        sorted_result = attrs_search(subject, sort=True)
        assert sorted_result == sorted(unsorted_result)
        assert sorted_result == ["instance_attr", "instance_none_attr", "public_attr", "public_none_attr"]

    def test_all_options_enabled(self, subject: AttrsObj):
        """Verify it returns all attributes and properties when all flags are True."""
        result = attrs_search(
            subject,
            include_private=True,
            include_property=True,
            include_none_attrs=True,
            include_none_properties=True,
            sort=True,
        )
        expected = [
            "_instance_private_attr",
            "_private_attr",
            "_private_none_attr",
            "_private_property",
            "error_property",
            "instance_attr",
            "instance_none_attr",
            "none_property",
            "public_attr",
            "public_none_attr",
            "public_property",
        ]
        assert sorted(result) == expected

    def test_error_property_is_handled(self, subject: AttrsObj):
        """Verify it gracefully handles properties that raise exceptions."""
        # The function should not raise an exception.
        # 'error_property' is included because _safe_getattr returns None,
        # and include_none_properties is True by default.
        result = attrs_search(subject, include_property=True)
        assert "error_property" in result

    def test_on_class_object(self):
        """Verify it correctly inspects a class object instead of an instance."""
        result = attrs_search(AttrsObj, include_private=True, include_property=True, sort=True)
        # Instance attributes should not be present
        expected = [
            "_private_attr",
            "_private_none_attr",
            "_private_property",
            "error_property",
            "none_property",
            "public_attr",
            "public_none_attr",
            "public_property",
        ]
        assert sorted(result) == expected

    @pytest.mark.parametrize(
        "obj",
        [
            pytest.param(123, id="integer_instance"),
            pytest.param("hello", id="string_instance"),
            pytest.param([1, 2], id="list_instance"),
            pytest.param({1: "a"}, id="dict_instance"),
            pytest.param(int, id="integer_class"),
            pytest.param(str, id="string_class"),
            pytest.param(list, id="list_class"),
        ],
    )
    def test_primitive_types_return_empty(self, obj: Any):
        """Verify it returns an empty list for built-in primitive types and instances."""
        assert attrs_search(obj) == []

    def test_none_input_returns_empty(self):
        """Verify it returns an empty list when the input object is None."""
        assert attrs_search(None) == []

    def test_mangled_attributes_are_excluded(self, subject: AttrsObj):
        """Verify it excludes name-mangled attributes by default."""
        # Check with all flags to ensure it's always excluded
        result = attrs_search(
            subject,
            include_private=True,
            include_property=True,
            sort=True
        )
        assert "_TestSubject__mangled_attr" not in result
        assert "__mangled_attr" not in result


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
            exclude_types=(), exclude_ids=set(), max_depth=None, seen=set(),
            on_error="skip",
        )
        info = deep_sizeof(
            obj,
            format="dict",
            exclude_types=(), exclude_ids=set(), max_depth=None, seen=set(),
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
            exclude_types=(), exclude_ids=set(), max_depth=None, seen=set(),
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
            exclude_types=(), exclude_ids=set(), max_depth=None, seen=set(),
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
                exclude_types=(), exclude_ids=set(), max_depth=None, seen=set(),
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
                exclude_types=(), exclude_ids=set(), max_depth=None, seen=set(),
                on_error="warn",
            )
            assert len(w) >= 1

    def test_exclude_types_nested(self) -> None:
        """Exclude types recursively and reduce total size."""
        obj = {"a": ["x", "y", "z"], "b": {"s": "t"}}
        size_full = deep_sizeof(
            obj,
            format="int",
            exclude_types=(), exclude_ids=set(), max_depth=None, seen=set(),
            on_error="skip",
        )
        size_no_str = deep_sizeof(
            obj,
            format="int",
            exclude_types=(str,),
            exclude_ids=set(),
            max_depth=None, seen=set(),
            on_error="skip",
        )
        assert size_no_str < size_full

        info = deep_sizeof(
            obj,
            format="dict",
            exclude_types=(str,),
            exclude_ids=set(),
            max_depth=None, seen=set(),
            on_error="skip",
        )
        assert str not in info["by_type"]

    def test_exclude_custom_type(self) -> None:
        """Exclude a custom class and keep other contributions."""
        items = [ToExclude("data" * 10), 42, ToExclude("x" * 100)]
        size_full = deep_sizeof(
            items,
            format="int",
            exclude_types=(), exclude_ids=set(), max_depth=None, seen=set(),
            on_error="skip",
        )
        size_excl = deep_sizeof(
            items,
            format="int",
            exclude_types=(ToExclude,),
            exclude_ids=set(), max_depth=None, seen=set(),
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
            exclude_types=(), exclude_ids=set(), max_depth=None, seen=set(),
            on_error="skip",
        )
        size_excl = deep_sizeof(
            obj,
            format="int",
            exclude_types=(),
            exclude_ids={id(big)},
            max_depth=None, seen=set(),
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
            max_depth=None, seen=set(),
            on_error="skip",
        )
        assert size_nested < deep_sizeof(
            nested,
            format="int",
            exclude_types=(), exclude_ids=set(), max_depth=None, seen=set(),
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
            exclude_types=(), exclude_ids=set(), max_depth=None,
            seen=seen,
            on_error="skip",
        )
        size_b = deep_sizeof(
            b,
            format="int",
            exclude_types=(), exclude_ids=set(), max_depth=None,
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
            exclude_types=(), exclude_ids=set(), max_depth=None, seen=set(),
            on_error="skip",
        )
        size = deep_sizeof(
            container,
            format="int",
            exclude_types=(), exclude_ids=set(), max_depth=None, seen=set(),
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
                exclude_types=(), exclude_ids=set(), max_depth=None, seen=set(),
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
            exclude_types=(), exclude_ids=set(),
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
            exclude_types=(), exclude_ids=set(),
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
            exclude_types=(), exclude_ids=set(),
            max_depth=0,
            seen=set(),
            on_error="skip",
        )
        limited = deep_sizeof(
            obj,
            format="int",
            exclude_types=(), exclude_ids=set(),
            max_depth=1,
            seen=set(),
            on_error="skip",
        )
        full = deep_sizeof(
            obj,
            format="int",
            exclude_types=(), exclude_ids=set(), max_depth=None, seen=set(),
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
            exclude_types=(), exclude_ids=set(), max_depth=None, seen=set(),
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
            exclude_types=(), exclude_ids=set(), max_depth=None, seen=set(),
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
            exclude_types=(), exclude_ids=set(), max_depth=None, seen=set(),
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
            exclude_types=(), exclude_ids=set(), max_depth=None, seen=set(),
            on_error="skip",
        )
        assert deep > shallow


class TestIsBuiltin:
    """Groups tests for the is_builtin function."""

    @pytest.mark.parametrize(
        "obj",
        [
            int, 1,
            str, "hello",
            list, [1, 2],
            dict, {"a": 1},
            tuple, (1, 2),
            set, {1, 2},
            float, 3.14,
            bool, True,
            range, range(10),
            object, object(),
            None,
        ],
        ids=[
            "type_int", "instance_int",
            "type_str", "instance_str",
            "type_list", "instance_list",
            "type_dict", "instance_dict",
            "type_tuple", "instance_tuple",
            "type_set", "instance_set",
            "type_float", "instance_float",
            "type_bool", "instance_bool",
            "type_range", "instance_range",
            "type_object", "instance_object",
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


# Test Core Private Methods --------------------------------------------------------------------------------------------

class Test_ActsLikeImage:
    @pytest.mark.parametrize(
        "cls",
        [
            # Class that looks like an Image (has required attrs and 3+ methods)
            type("MyImage", (),
                 {"size": (1, 1), "mode": "RGB", "format": "PNG",
                  "save": lambda self, *a, **k: None,
                  "show": lambda self, *a, **k: None,
                  "resize": lambda self, *a, **k: None,
                  }, ),
            # Type with 'Image' in the name and many methods but attributes on the class
            type("FakeImageType", (),
                 {"size": (2, 2), "mode": "L", "format": "JPEG",
                  "save": lambda self, *a, **k: None,
                  "show": lambda self, *a, **k: None,
                  "crop": lambda self, *a, **k: None,
                  }, ),
        ],
        ids=["class-with-attrs-and-methods", "named-image-type"],
    )
    def test_type_positive(self, cls):
        """Return True for image-like classes."""
        assert _acts_like_image(cls) is True

    def test_instance_positive(self):
        """Return True for image-like instances."""
        Inst = type("ImageInstance", (),
                    {"size": (10, 10), "mode": "RGBA", "format": "PNG",
                     "save": lambda self, *a, **k: None,
                     "show": lambda self, *a, **k: None,
                     "resize": lambda self, *a, **k: None,
                     }, )
        obj = Inst()
        assert _acts_like_image(obj) is True

    @pytest.mark.parametrize(
        ("obj", "reason"),
        [
            (type("NoPic", (),
                  {"size": (1, 1), "mode": "RGB", "format": "PNG", "save": lambda s: None, "show": lambda s: None,
                   "resize": lambda s: None}), "name"),
            (type("ImageMissingAttrs", (), {"save": lambda s: None, "show": lambda s: None, "resize": lambda s: None}),
             "missing-attrs"),
            (type("ImageFewMethods", (), {"size": (1, 1), "mode": "RGB", "format": "PNG", "save": lambda s: None}),
             "too-few-methods"),
        ],
        ids=["no-image-substring", "missing-attrs", "few-methods"],
    )
    def test_type_negative(self, obj, reason):
        """Return False for non-image-like classes."""
        assert _acts_like_image(obj) is False

    @pytest.mark.parametrize(
        ("instance", "id"),
        [
            (type("BadSize", (),
                  {"size": (0, 10), "mode": "RGB", "format": "PNG", "save": lambda s: None, "show": lambda s: None,
                   "resize": lambda s: None})(), "zero-width"),
            (type("BadSizeType", (),
                  {"size": ("a", 10), "mode": "RGB", "format": "PNG", "save": lambda s: None, "show": lambda s: None,
                   "resize": lambda s: None})(), "non-int-size"),
            (type("EmptyMode", (),
                  {"size": (1, 1), "mode": "", "format": "PNG", "save": lambda s: None, "show": lambda s: None,
                   "resize": lambda s: None})(), "empty-mode"),
            (object(), "plain-object"),
        ],
        ids=["invalid-size-zero", "invalid-size-type", "empty-mode", "plain-object"],
    )
    def test_instance_negative(self, instance, id):
        """Return False for instances with invalid attributes."""
        assert _acts_like_image(instance) is False

    def test_plain_types_and_instances(self):
        """Return False for unrelated types and instances."""

        class NotImage: pass

        assert _acts_like_image(NotImage) is False
        assert _acts_like_image(object()) is False
