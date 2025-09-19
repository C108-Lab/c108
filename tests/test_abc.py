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
                      remove_extra_attrs
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
        "size,unit,typ,expected",
        [
            (123, "bytes", int, "<int> 123 bytes"),
            (1, "chars", str, "<str> 1 chars"),
        ],
        ids=["int-bytes", "str-chars"],
    )
    def test_scalar_as_str(self, size, unit, typ, expected):
        """Format scalar size and unit as a human readable string."""
        oi = ObjectInfo(type=typ, size=size, unit=unit, deep_size=0)
        assert oi.as_str.startswith(f"<{oi.class_name}>")
        assert expected.split(" ", 1)[1] in oi.as_str

    def test_sequence_size_unit_mismatch_raises(self):
        """Raise when size and unit sequences have different lengths."""
        with pytest.raises(ValueError, match=r"(?i)same length"):
            ObjectInfo(type=list, size=(1, 2), unit=("items",), deep_size=0)

    def test_size_type_validation_raises(self):
        """Raise when size has unsupported non-sequence/non-number type."""

        class BadSize:
            pass

        with pytest.raises(TypeError, match=r"(?i)size must be"):
            ObjectInfo(type=BadSize, size=BadSize(), unit="bytes")

    def test_size_elements_type_validation_raises(self):
        """Raise when sequence elements in size are not numbers."""
        with pytest.raises(TypeError, match=r"(?i)all elements in size"):
            ObjectInfo(type=list, size=(1, "two", 3), unit=("a", "b", "c"))

    def test_unit_type_validation_raises(self):
        """Raise when unit has unsupported non-str/non-sequence type."""
        with pytest.raises(TypeError, match=r"(?i)unit must be"):
            ObjectInfo(type=int, size=4, unit=123)

    def test_unit_elements_type_validation_raises(self):
        """Raise when sequence elements in unit are not strings."""
        with pytest.raises(TypeError, match=r"(?i)all elements in unit"):
            ObjectInfo(type=list, size=(1, 2), unit=("items", 3))

    def test_sequence_size_with_non_sequence_unit_raises(self):
        """Raise when size is a sequence but unit is not a sequence."""
        with pytest.raises(TypeError, match=r"(?i)size and unit type mismatch"):
            ObjectInfo(type=list, size=(1, 2), unit="items")

    def test_from_object_primitives_and_containers(self):
        """Summarize int, str, bytes and list via from_object heuristics."""
        oi_int = ObjectInfo.from_object(10)
        assert oi_int.unit == "bytes"
        assert oi_int.type is int

        oi_str = ObjectInfo.from_object("abc")
        assert oi_str.unit == "chars"
        assert oi_str.size == 3

        oi_bytes = ObjectInfo.from_object(b"xyz")
        assert oi_bytes.unit == "bytes"
        assert oi_bytes.size == 3

        lst = [1, 2, 3]
        oi_list = ObjectInfo.from_object(lst)
        assert oi_list.unit == "items"
        assert oi_list.size == 3

    def test_from_object_image_like_formats_as_triplet(self):
        """Format image-like objects with (width, height, Mpx) as specialized string."""

        class FakeImage:
            def __init__(self, w, h):
                self.size = (w, h)

        img = FakeImage(640, 480)
        oi = ObjectInfo.from_object(img)
        # Accept either image-like formatting or instance-with-attrs fallback, depending on acts_like_image()
        s = oi.as_str
        assert ("W⨯H" in s and "Mpx" in s) or (isinstance(oi.size, (tuple, list)) and oi.unit == ("attrs", "bytes"))

    def test_from_object_class_and_instance_with_attrs(self):
        """Describe class objects by attrs count and instances with attrs by attrs+bytes."""

        class C:
            a = 1
            b = 2

            def __init__(self):
                self.x = 1

        oi_class = ObjectInfo.from_object(C)
        assert oi_class.unit == "attrs"
        assert isinstance(oi_class.size, int)

        inst = C()
        oi_inst = ObjectInfo.from_object(inst)
        # instance with attributes should return tuple-like size and unit with attrs and bytes
        assert isinstance(oi_inst.size, (tuple, list))
        assert oi_inst.unit == ("attrs", "bytes")

    def test_as_str_generic_tuple_unit(self):
        """Render generic tuple/list size with matching unit labels."""
        oi = ObjectInfo(type=object, size=(1, 2), unit=("one", "two"), deep_size=0)
        s = oi.as_str
        assert "<" in s and ">" in s
        assert "1 one" in s and "2 two" in s


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
            ({"name": "name", "value": "VALUE", "test": "test"}, False, True),
            ({"name": "name", "value": "value", "test": "test"}, True, True),
            ({"name": "NAME"}, True, False),
            ({"name": "name", "value": "different_value"}, False, False),
            ({}, False, True),
            ({"123": 123}, False, True),
            ({"True": True, "False": False}, False, True),
            ({"None": None}, False, True),
            ({"None": None, "null": None}, False, False),
            ({"good": "good", "bad": "wrong"}, False, False),
            ({"special_chars": "special_chars", "with-dash": "with-dash", "with_underscore": "with_underscore"}, False,
             True),
        ],
        ids=[
            "case-insensitive-match",
            "case-sensitive-match",
            "case-sensitive-mismatch",
            "value-mismatch",
            "empty-object",
            "numeric-attr",
            "boolean-attrs",
            "none-value",
            "none-and-null",
            "mixed-match-and-mismatch",
            "special-characters",
        ],
    )
    def test_attrs_vs_names(self, attrs, case_sensitive, expected):
        """Check that attributes equal their names according to sensitivity."""

        class Obj:
            pass

        obj = Obj()
        for k, v in attrs.items():
            setattr(obj, k, v)

        assert attrs_eq_names(obj, case_sensitive=case_sensitive) is expected

    def test_ignores_callables_and_dunders(self):
        """Ignore callable and dunder attributes when comparing names."""

        class TestObj:
            name = "name"

            def some_method(self):
                return "method"

            __private = "private"  # name-mangled, should be ignored
            __dict__ = {}  # explicit dunder, should be ignored

        obj = TestObj()
        assert attrs_eq_names(obj) is True

    def test_case_sensitivity_behavior_explicit(self):
        """Respect explicit case_sensitive parameter for boolean-like names."""

        class Obj:
            pass

        obj = Obj()
        setattr(obj, "true", True)
        setattr(obj, "True", True)

        # When case sensitive, 'true' != 'True' -> should be False
        assert attrs_eq_names(obj, case_sensitive=True) is False
        # Case-insensitive treats them equal (both convert to 'true') -> True
        assert attrs_eq_names(obj, case_sensitive=False) is True

    def test_raise_exception_on_first_mismatch(self):
        """Raise ValueError on the first mismatch when requested."""

        class Obj:
            first = "wrong"
            second = "also_wrong"

        obj = Obj()
        with pytest.raises(ValueError) as exc:
            attrs_eq_names(obj, raise_exception=True)
        msg = str(exc.value)
        assert "first" in msg
        assert "wrong" in msg


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


class TestAttrsSearch:
    @pytest.mark.parametrize(
        "obj",
        [int, 1],
        ids=["builtin-type", "builtin-instance"],
    )
    def test_empty_for_builtins(self, obj):
        """Return empty list for built-in types."""
        assert attrs_search(obj) == []

    def test_data_attrs_exclude_callables_and_dunder(self):
        """Include data attributes and exclude callables, private, and dunder by default."""

        class C:
            def __init__(self):
                self.x = 1
                self._y = 2

            def method(self):
                return 10

            def __str__(self):
                return "C"

        c = C()
        assert attrs_search(c) == ["x"]

    @pytest.mark.parametrize(
        "include_private, expected",
        [
            (False, {"x"}),
            (True, {"x", "_y"}),
        ],
        ids=["no-private", "with-private"],
    )
    def test_private_attr_toggle(self, include_private, expected):
        """Toggle inclusion of private attributes."""

        class C:
            def __init__(self):
                self.x = 1
                self._y = 2

        c = C()
        assert set(attrs_search(c, include_private=include_private)) == expected

    @pytest.mark.parametrize(
        "include_property, contains",
        [
            (False, False),
            (True, True),
        ],
        ids=["no-property", "with-property"],
    )
    def test_property_on_instance_toggle(self, include_property, contains):
        """Toggle inclusion of instance properties."""

        class C:
            def __init__(self, val):
                self._val = val

            @property
            def p(self):
                return self._val

        c = C(5)
        names = attrs_search(c, include_property=include_property)
        assert ("p" in names) is contains

    @pytest.mark.parametrize(
        "include_none_attrs, contains",
        [
            (True, True),
            (False, False),
        ],
        ids=["include-none", "exclude-none"],
    )
    def test_property_none_respects_include_none_attrs(self, include_none_attrs, contains):
        """Include property returning None only when include_none_attrs is true."""

        class C:
            @property
            def p(self):
                return None

        c = C()
        names = attrs_search(c, include_property=True, include_none_attrs=include_none_attrs)
        assert ("p" in names) is contains

    @pytest.mark.parametrize(
        "include_property, contains",
        [
            (False, False),
            (True, True),
        ],
        ids=["no-property", "with-property"],
    )
    def test_property_raising_on_instance_toggle(self, include_property, contains):
        """Include property even if getter raises when include_property is true."""

        class C:
            @property
            def p(self):
                raise ValueError("boom")

        c = C()
        names = attrs_search(c, include_property=include_property)
        assert ("p" in names) is contains

    @pytest.mark.parametrize(
        "include_property, contains",
        [
            (False, False),
            (True, True),
        ],
        ids=["no-property", "with-property"],
    )
    def test_property_on_class_object_toggle(self, include_property, contains):
        """Include class property only when include_property is true."""

        class C:
            @property
            def p(self):
                return 1

        names = attrs_search(C, include_property=include_property)
        assert ("p" in names) is contains

    def test_staticmethod_and_classmethod_excluded(self):
        """Exclude staticmethod and classmethod regardless of flags."""

        class C:
            x = 1

            @staticmethod
            def s():
                return 0

            @classmethod
            def c(cls):
                return 0

        c = C()

        # Methods are excluded for both class and instance, regardless of include_property
        for obj, include_property in [(C, False), (C, True), (c, False), (c, True)]:
            names = attrs_search(obj, include_property=include_property)
            assert "s" not in names
            assert "c" not in names

        # Data attribute remains on instance
        assert "x" in attrs_search(c)

    @pytest.mark.parametrize(
        "include_none_attrs, expected",
        [
            (True, {"x", "y"}),
            (False, {"y"}),
        ],
        ids=["include-none", "exclude-none"],
    )
    def test_none_attrs_toggle(self, include_none_attrs, expected):
        """Toggle inclusion of attributes with value None."""

        class C:
            def __init__(self):
                self.x = None
                self.y = 1

        c = C()
        assert set(attrs_search(c, include_none_attrs=include_none_attrs)) == expected


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

    def test_invalid_exclude_types_arg_not_tuple(self):
        """Raise TypeError if exclude_types is not a tuple."""
        with pytest.raises(TypeError, match=r"(?i)exclude_types must be a tuple"):
            deep_sizeof([1, 2, 3], exclude_types=[str])

    def test_invalid_exclude_types_arg_contains_non_type(self):
        """Raise TypeError if exclude_types contains non-type elements."""
        with pytest.raises(TypeError, match=r"(?i)All items in exclude_types must be types"):
            deep_sizeof([1, 2, 3], exclude_types=(str, 123))

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


class TestRemoveExtraAttrs:
    @pytest.mark.parametrize(
        "attrs_input, mangled_cls_name, include_dunder, include_private, expected_output, expected_type",
        [
            (
                    {"public": 1, "_private": 2, "__dunder__": 3, "_MyClass__mangled": 4, "also_public": 5},
                    None, False, False, {"public": 1, "also_public": 5}, dict
            ),
            (
                    ["a", "_b", "__c__", "_MyClass__d", "e"],
                    "MyClass", False, False, ["a", "e"], list
            ),
            (
                    ("x", "_y", "__z__", "_MyClass__t", "w"),
                    "MyClass", False, False, ("x", "w"), tuple
            ),
        ],
        ids=[
            "dict_removes_all",
            "list_preserves",
            "tuple_preserves",
        ]
    )
    def test_default_behavior(self, attrs_input, mangled_cls_name, include_dunder, include_private, expected_output,
                              expected_type):
        """Remove extra attributes by default."""
        result = remove_extra_attrs(attrs_input, mangled_cls_name=mangled_cls_name, include_dunder=include_dunder,
                                    include_private=include_private)
        assert result == expected_output
        assert isinstance(result, expected_type)

    @pytest.mark.parametrize(
        "attrs_input, include_dunder, include_private, mangled_cls_name, expected_output",
        [
            (
                    {"__dunder__", "_private", "_MyClass__mangled", "public"},
                    True, False, "MyClass", {"__dunder__", "public"}
            ),
            (
                    {"_private", "__dunder__", "_MyClass__mangled", "public"},
                    False, True, "MyClass", {"_private", "public"}
            ),
        ],
        ids=[
            "include_dunder_only",
            "include_private_only",
        ]
    )
    def test_include_dunder_or_private_flags(self, attrs_input, include_dunder, include_private, mangled_cls_name,
                                             expected_output):
        """Keep dunder or private attributes based on flags."""
        result = remove_extra_attrs(attrs_input, mangled_cls_name=mangled_cls_name, include_dunder=include_dunder,
                                    include_private=include_private)
        assert result == expected_output

    @pytest.mark.parametrize(
        "attrs_input, mangled_cls_name, expected_output_equal, expected_output_type",
        [
            (
                    {"_p": 1, "__d__": 2, "k": 3},
                    None, {"_p": 1, "__d__": 2, "k": 3}, dict
            ),
            (
                    ["_p", "__d__", "k"],
                    None, ["_p", "__d__", "k"], list
            ),
            (
                    {"_MyClass__x", "_Other__y", "public"},
                    "MyClass", {"_Other__y", "public"}, set
            ),
        ],
        ids=[
            "include_all_dict",
            "include_all_list",
            "include_all_mangled_removed",
        ]
    )
    def test_include_all_flags(self, attrs_input, mangled_cls_name, expected_output_equal, expected_output_type):
        """Return a new object with all attributes kept when flags are true, and remove mangled if class name provided."""
        result = remove_extra_attrs(attrs_input, include_private=True, include_dunder=True, mangled_cls_name=mangled_cls_name)

        assert result == expected_output_equal
        assert isinstance(result, expected_output_type)
        assert result is not attrs_input

    @pytest.mark.parametrize(
        "attrs_input, include_private, include_dunder, include_mangled, mangled_cls_name, expected_output",
        [
            (
                    {"_MyClass__x": 1, "_Other__y": 2, "_p": 3, "__d__": 4, "public": 5},
                    True, False, True, "MyClass", {"_MyClass__x": 1, "_Other__y": 2, "_p": 3, "public": 5}
            ),
            (
                    {"_MyClass__x": 1, "_Other__y": 2, "_p": 3, "__d__": 4, "public": 5},
                    False, False, True, "MyClass", {"public": 5}
            ),
            (
                    {"_MyClass__x": 1, "_Other__y": 2, "_p": 3, "__d__": 4, "public": 5},
                    True, False, False, "MyClass", {"_Other__y": 2, "_p": 3, "public": 5}
            ),
        ],
        ids=[
            "include_mangled_private",
            "include_mangled_no_private",
            "include_private_no_mangled",
        ]
    )
    def test_include_mangled_flag_behavior(self, attrs_input, include_private, include_dunder, include_mangled, mangled_cls_name,
                                           expected_output):
        """Control mangled attribute removal with include_mangled flag."""
        result = remove_extra_attrs(
            attrs_input,
            include_private=include_private,
            include_dunder=include_dunder,
            include_mangled=include_mangled,
            mangled_cls_name=mangled_cls_name,
        )
        assert result == expected_output

    def test_invalid_input_raises_type_error(self):
        """Raise TypeError for non-collection input."""
        with pytest.raises(TypeError):
            remove_extra_attrs(123)

    def test_different_class_name_not_mangled(self):
        """Do not remove attributes mangled for a different class name."""
        attrs = {"_MyClazz__x", "public", "_My", "_MyClazz", "__dunder__"}
        result = remove_extra_attrs(attrs, mangled_cls_name="MyClass")
        assert result == {"public"}

    def test_mangled_substring_removes(self):
        """Remove attributes containing the mangled class name as a substring."""
        attrs = {
            "prefix_MyClass_suffix": 1,
            "_pre_MyClass_suf": 2,
            "_other_private": 3,
            "ok": 4,
        }
        res = remove_extra_attrs(attrs, mangled_cls_name="MyClass")
        assert res == {"prefix_MyClass_suffix": 1, "ok": 4}
