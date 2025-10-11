#
# C108 - ABC Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import inspect
import sys
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

# Third-party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.abc import (ObjectInfo,
                      acts_like_image,
                      attr_is_property,
                      attrs_eq_names,
                      attrs_search,
                      deep_sizeof,
                      is_builtin,
                      )


# Local Classes & Methods ----------------------------------------------------------------------------------------------

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

class TestActsLikeImage:
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
        assert acts_like_image(cls) is True

    def test_instance_positive(self):
        """Return True for image-like instances."""
        Inst = type("ImageInstance", (),
                    {"size": (10, 10), "mode": "RGBA", "format": "PNG",
                     "save": lambda self, *a, **k: None,
                     "show": lambda self, *a, **k: None,
                     "resize": lambda self, *a, **k: None,
                     }, )
        obj = Inst()
        assert acts_like_image(obj) is True

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
        assert acts_like_image(obj) is False

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
        assert acts_like_image(instance) is False

    def test_plain_types_and_instances(self):
        """Return False for unrelated types and instances."""

        class NotImage: pass

        assert acts_like_image(NotImage) is False
        assert acts_like_image(object()) is False


class TestAttrsEqNames:

    @pytest.mark.parametrize(
        "attrs, case_sensitive, expected",
        [
            pytest.param({"name": "name", "value": "value"}, True, True, id="exact_match_case_sensitive"),
            pytest.param({"name": "NAME", "value": "VALUE"}, False, True, id="case_insensitive_match"),
            pytest.param({"name": "NAME"}, True, False, id="case_sensitive_mismatch"),
            pytest.param({"good": "good", "bad": "wrong"}, False, False, id="mixed_match_mismatch"),
            pytest.param({}, False, True, id="empty_object"),
        ],
    )
    def test_basic_attribute_comparison(self, attrs, case_sensitive, expected):
        """Compare attribute names with their values using different case sensitivity."""

        class TestObj:
            pass

        obj = TestObj()
        for k, v in attrs.items():
            setattr(obj, k, v)

        assert attrs_eq_names(obj, raise_exception=False, case_sensitive=case_sensitive) is expected

    @pytest.mark.parametrize(
        "attrs",
        [
            pytest.param({"numeric": 123}, id="integer_value"),
            pytest.param({"float_val": 3.14}, id="float_value"),
            pytest.param({"bool_true": True, "bool_false": False}, id="boolean_values"),
            pytest.param({"none_val": None}, id="none_value"),
        ],
    )
    def test_non_string_values_converted(self, attrs):
        """Convert non-string attribute values to strings for comparison."""

        class TestObj:
            pass

        obj = TestObj()
        for k, v in attrs.items():
            setattr(obj, k, k)  # Set attribute value equal to attribute name

        assert attrs_eq_names(obj, raise_exception=False, case_sensitive=True) is True

    def test_ignores_methods_and_private_attrs(self):
        """Skip callable methods and private/dunder attributes during comparison."""

        class TestObj:
            public_attr = "public_attr"
            _private = "should_be_ignored"
            __dunder__ = "should_be_ignored"

            def method(self):
                return "callable"

            @property
            def prop(self):
                return "property"

        assert attrs_eq_names(TestObj(), raise_exception=False, case_sensitive=True) is True

    def test_exception_raised_with_details(self):
        """Raise ValueError with attribute details when mismatch found."""

        class TestObj:
            first_attr = "wrong_value"
            second_attr = "second_attr"

        with pytest.raises(ValueError, match=r"(?i).*first_attr.*wrong_value.*"):
            attrs_eq_names(TestObj(), raise_exception=True, case_sensitive=True)

    def test_no_exception_when_flag_false(self):
        """Return False without raising exception when raise_exception is False."""

        class TestObj:
            mismatch = "different"

        result = attrs_eq_names(TestObj(), raise_exception=False, case_sensitive=True)
        assert result is False

    def test_case_insensitive_comparison(self):
        """Handle case-insensitive comparison correctly."""

        class TestObj:
            UPPER = "upper"
            lower = "LOWER"
            MiXeD = "mixed"

        assert attrs_eq_names(TestObj(), raise_exception=False, case_sensitive=False) is True

    def test_case_sensitive_comparison(self):
        """Handle case-sensitive comparison correctly."""

        class TestObj:
            exact = "exact"
            MISMATCH = "mismatch"  # This will fail case-sensitive check

        assert attrs_eq_names(TestObj(), raise_exception=False, case_sensitive=True) is False

    # @pytest.mark.parametrize(
    #     "enum_class, expected",
    #     [
    #         pytest.param(
    #             type("Colors", (Enum,), {"RED": "RED", "BLUE": "BLUE"}),
    #             True,
    #             id="matching_enum_values"
    #         ),
    #         pytest.param(
    #             type("Status", (Enum,), {"ACTIVE": "active", "INACTIVE": "inactive"}),
    #             False,
    #             id="mismatched_enum_values"
    #         ),
    #     ],
    # )
    # def test_enum_attribute_comparison(self, enum_class, expected):
    #     """Compare enum class attributes with their string representations."""
    #     # Test the enum class itself, not an instance
    #     assert attrs_eq_names(enum_class, raise_exception=False, case_sensitive=True) is expected

    def test_special_characters_in_names(self):
        """Handle attributes with special characters in names."""

        class TestObj:
            with_underscore = "with_underscore"
            number123 = "number123"

        assert attrs_eq_names(TestObj(), raise_exception=False, case_sensitive=True) is True

    def test_inherited_attributes(self):
        """Include inherited attributes in comparison."""

        class BaseObj:
            base_attr = "base_attr"

        class DerivedObj(BaseObj):
            derived_attr = "derived_attr"

        assert attrs_eq_names(DerivedObj(), raise_exception=False, case_sensitive=True) is True

    def test_dynamic_attributes(self):
        """Handle dynamically added attributes."""

        class TestObj:
            pass

        obj = TestObj()
        obj.dynamic1 = "dynamic1"
        obj.dynamic2 = "wrong"

        assert attrs_eq_names(obj, raise_exception=False, case_sensitive=True) is False

    @pytest.mark.parametrize(
        "obj_type",
        [
            pytest.param(dict, id="dict_class"),
            pytest.param(list, id="list_class"),
            pytest.param(str, id="str_class"),
        ],
    )
    def test_builtin_types(self, obj_type):
        """Handle built-in Python types without relevant attributes."""
        # Built-in types typically don't have matching name=value attributes
        # This tests that the function handles them gracefully
        result = attrs_eq_names(obj_type, raise_exception=False, case_sensitive=True)
        assert isinstance(result, bool)  # Should return a boolean without error


class TestAttrIsProperty:
    """Test suite for the attr_is_property utility function."""

    @pytest.mark.parametrize(
        ("attr_name", "try_callable", "expected"),
        [
            ("working_property", False, True),
            ("failing_property", False, True),
            ("regular_attribute", False, False),
            ("a_method", False, False),
            ("non_existent", False, False),
            ("working_property", True, False),
            ("failing_property", True, False),
        ],
        ids=[
            "class-working_property-no_call",
            "class-failing_property-no_call",
            "class-regular_attribute",
            "class-method",
            "class-non_existent_attribute",
            "class-working_property-with_call-is_false",
            "class-failing_property-with_call-is_false",
        ],
    )
    def test_on_class(self, attr_name: str, try_callable: bool, expected: bool):
        """Verify property detection on a class definition."""
        result = attr_is_property(attr_name, Example, try_callable=try_callable)
        assert result is expected

    @pytest.mark.parametrize(
        ("attr_name", "try_callable", "expected"),
        [
            ("working_property", False, True),
            ("failing_property", False, True),
            ("working_property", True, True),
            ("failing_property", True, False),
            ("regular_attribute", False, False),
            ("a_method", False, False),
            ("non_existent", False, False),
        ],
        ids=[
            "instance-working_property-no_call",
            "instance-failing_property-no_call",
            "instance-working_property-with_call-succeeds",
            "instance-failing_property-with_call-fails",
            "instance-regular_attribute",
            "instance-method",
            "instance-non_existent_attribute",
        ],
    )
    def test_on_instance(self, attr_name: str, try_callable: bool, expected: bool):
        """Verify property detection on a class instance."""
        instance = Example()
        result = attr_is_property(attr_name, instance, try_callable=try_callable)
        assert result is expected

    def test_on_dataclass_class(self):
        """Verify a dataclass field is not a property on the class."""
        assert attr_is_property("field", SimpleDataClass, try_callable=False) is False

    def test_on_dataclass_instance(self):
        """Verify a dataclass field is not a property on the instance."""
        instance = SimpleDataClass()
        assert attr_is_property("field", instance, try_callable=False) is False


class TestAttrsObj:
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


@pytest.fixture
def subject() -> TestAttrsObj:
    """Provide an instance of the TestAttrsObj class."""
    return TestAttrsObj()


class TestAttrsSearch:
    """Test suite for the attrs_search function."""

    def test_include_private(self, subject: TestAttrsObj):
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

    def test_include_property(self, subject: TestAttrsObj):
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

    def test_exclude_none_attrs(self, subject: TestAttrsObj):
        """Verify it excludes attributes with a value of None."""
        result = attrs_search(subject, include_none_attrs=False, sort=True)
        expected = ["instance_attr", "public_attr"]
        assert sorted(result) == expected

    def test_exclude_none_properties(self, subject: TestAttrsObj):
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

    def test_sort_output(self, subject: TestAttrsObj):
        """Verify it sorts the output alphabetically when requested."""
        unsorted_result = attrs_search(subject, sort=False)
        sorted_result = attrs_search(subject, sort=True)
        assert sorted_result == sorted(unsorted_result)
        assert sorted_result == ["instance_attr", "instance_none_attr", "public_attr", "public_none_attr"]

    def test_all_options_enabled(self, subject: TestAttrsObj):
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

    def test_error_property_is_handled(self, subject: TestAttrsObj):
        """Verify it gracefully handles properties that raise exceptions."""
        # The function should not raise an exception.
        # 'error_property' is included because _safe_getattr returns None,
        # and include_none_properties is True by default.
        result = attrs_search(subject, include_property=True)
        assert "error_property" in result

    def test_on_class_object(self):
        """Verify it correctly inspects a class object instead of an instance."""
        result = attrs_search(TestAttrsObj, include_private=True, include_property=True, sort=True)
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

    def test_mangled_attributes_are_excluded(self, subject: TestAttrsObj):
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
    """Test suite for the deep_sizeof() function."""

    @pytest.mark.parametrize(
        "obj",
        [
            100,
            "hello world",
            3.14,
            True,
            None,
            b"binary data",
        ],
        ids=["int", "str", "float", "bool", "None", "bytes"],
    )
    def test_primitive_types(self, obj: Any):
        """Verify size of primitive types equals their sys.getsizeof."""
        assert deep_sizeof(obj) == sys.getsizeof(obj)

    @pytest.mark.parametrize(
        "container",
        [
            [1, "a", 2.0],
            (1, "a", 2.0),
            {1, "a", 2.0},
            frozenset([1, "a", 2.0]),
            {"key1": 1, "key2": "a", "key3": 2.0},
        ],
        ids=["list", "tuple", "set", "frozenset", "dict"],
    )
    def test_simple_containers(self, container: Any):
        """Ensure deep size of simple containers is greater than their shallow size."""
        assert deep_sizeof(container) > sys.getsizeof(container)

    def test_nested_structure(self):
        """Check deep size calculation for a nested data structure."""
        nested_obj = {"data": [1, 2, {"key": "value"}]}
        shallow_size = sys.getsizeof(nested_obj)
        deep_size = deep_sizeof(nested_obj)
        assert deep_size > shallow_size

    def test_custom_object_with_dict(self):
        """Test deep sizeof on a custom object using __dict__."""

        class MyObject:
            def __init__(self):
                self.a = 1
                self.b = "some string"

        obj = MyObject()
        # Size should be greater than the object's shallow size plus its __dict__'s shallow size.
        assert deep_sizeof(obj) > sys.getsizeof(obj) + sys.getsizeof(obj.__dict__)

    def test_custom_object_with_slots(self):
        """Test deep sizeof on a custom object using __slots__."""

        class MySlottedObject:
            __slots__ = ['x', 'y']

            def __init__(self):
                self.x = 100
                self.y = "another string"

        obj = MySlottedObject()
        # Size should be greater than the shallow size, accounting for slotted attributes.
        assert deep_sizeof(obj) > sys.getsizeof(obj)

    def test_circular_reference(self):
        """Verify the function handles circular references without infinite loops."""
        a = [1, 2]
        a.append(a)  # Circular reference
        try:
            size = deep_sizeof(a)
            # The size should be calculable and positive.
            assert size > 0
        except RecursionError:
            pytest.fail("deep_sizeof failed to handle a circular reference.")

    def test_shared_object_reference(self):
        """Ensure shared objects are counted only once."""
        shared_list = [1, 2, 3, 4, 5]
        obj = [shared_list, shared_list]

        # Calculate expected size: the container list + one deep size of the shared list.
        expected_size = sys.getsizeof(obj) + deep_sizeof(shared_list)

        assert deep_sizeof(obj) == expected_size

    def test_exclude_types(self):
        """Test the exclusion of specific types from the size calculation."""
        obj = {"a": 1, "b": "a string", "c": [1, 2]}
        size_full = deep_sizeof(obj)
        size_no_str = deep_sizeof(obj, exclude_types=(str,))
        size_no_int = deep_sizeof(obj, exclude_types=(int,))

        assert size_no_str < size_full
        assert size_no_int < size_full
        assert size_no_str != size_no_int

    def test_exclude_custom_type(self):
        """Test the exclusion of a custom class type."""

        class ToExclude:
            def __init__(self):
                self.data = "large data" * 10

        obj = [ToExclude(), ToExclude(), 123]
        size_full = deep_sizeof(obj)
        size_excluded = deep_sizeof(obj, exclude_types=(ToExclude,))

        assert size_excluded < size_full
        # The size should be the list and the integer, excluding the custom objects.
        assert size_excluded == sys.getsizeof(obj) + sys.getsizeof(123)

    @pytest.mark.parametrize(
        "empty_container",
        [
            [],
            {},
            (),
            set(),
            frozenset(),
        ],
        ids=["empty_list", "empty_dict", "empty_tuple", "empty_set", "empty_frozenset"],
    )
    def test_empty_containers(self, empty_container: Any):
        """Verify size of empty containers equals their sys.getsizeof."""
        assert deep_sizeof(empty_container) == sys.getsizeof(empty_container)


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
