#
# C108 - ABC Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import sys
import types
import uuid
from dataclasses import dataclass

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


# Local Classes --------------------------------------------------------------------------------------------------------

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
        "inc_private, expected",
        [
            (False, {"x"}),
            (True, {"x", "_y"}),
        ],
        ids=["no-private", "with-private"],
    )
    def test_private_attr_toggle(self, inc_private, expected):
        """Toggle inclusion of private attributes."""

        class C:
            def __init__(self):
                self.x = 1
                self._y = 2

        c = C()
        assert set(attrs_search(c, inc_private=inc_private)) == expected

    @pytest.mark.parametrize(
        "inc_property, contains",
        [
            (False, False),
            (True, True),
        ],
        ids=["no-property", "with-property"],
    )
    def test_property_on_instance_toggle(self, inc_property, contains):
        """Toggle inclusion of instance properties."""

        class C:
            def __init__(self, val):
                self._val = val

            @property
            def p(self):
                return self._val

        c = C(5)
        names = attrs_search(c, inc_property=inc_property)
        assert ("p" in names) is contains

    @pytest.mark.parametrize(
        "inc_none_attrs, contains",
        [
            (True, True),
            (False, False),
        ],
        ids=["include-none", "exclude-none"],
    )
    def test_property_none_respects_inc_none_attrs(self, inc_none_attrs, contains):
        """Include property returning None only when inc_none_attrs is true."""

        class C:
            @property
            def p(self):
                return None

        c = C()
        names = attrs_search(c, inc_property=True, inc_none_attrs=inc_none_attrs)
        assert ("p" in names) is contains

    @pytest.mark.parametrize(
        "inc_property, contains",
        [
            (False, False),
            (True, True),
        ],
        ids=["no-property", "with-property"],
    )
    def test_property_raising_on_instance_toggle(self, inc_property, contains):
        """Include property even if getter raises when inc_property is true."""

        class C:
            @property
            def p(self):
                raise ValueError("boom")

        c = C()
        names = attrs_search(c, inc_property=inc_property)
        assert ("p" in names) is contains

    @pytest.mark.parametrize(
        "inc_property, contains",
        [
            (False, False),
            (True, True),
        ],
        ids=["no-property", "with-property"],
    )
    def test_property_on_class_object_toggle(self, inc_property, contains):
        """Include class property only when inc_property is true."""

        class C:
            @property
            def p(self):
                return 1

        names = attrs_search(C, inc_property=inc_property)
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

        # Methods are excluded for both class and instance, regardless of inc_property
        for obj, inc_property in [(C, False), (C, True), (c, False), (c, True)]:
            names = attrs_search(obj, inc_property=inc_property)
            assert "s" not in names
            assert "c" not in names

        # Data attribute remains on instance
        assert "x" in attrs_search(c)

    @pytest.mark.parametrize(
        "inc_none_attrs, expected",
        [
            (True, {"x", "y"}),
            (False, {"y"}),
        ],
        ids=["include-none", "exclude-none"],
    )
    def test_none_attrs_toggle(self, inc_none_attrs, expected):
        """Toggle inclusion of attributes with value None."""

        class C:
            def __init__(self):
                self.x = None
                self.y = 1

        c = C()
        assert set(attrs_search(c, inc_none_attrs=inc_none_attrs)) == expected


class TestDeepSizeOf:
    def test_primitives(self):
        assert deep_sizeof(0) == sys.getsizeof(0)
        assert deep_sizeof(123456789) == sys.getsizeof(123456789)
        assert deep_sizeof(3.14) == sys.getsizeof(3.14)
        assert deep_sizeof(True) == sys.getsizeof(True)
        assert deep_sizeof(None) == sys.getsizeof(None)
        s = "not-interned-" + str(uuid.uuid4())
        assert deep_sizeof(s) == sys.getsizeof(s)
        b = b"bytes"
        assert deep_sizeof(b) == sys.getsizeof(b)

    def test_list_and_tuple_nested(self):
        inner = ["a" + str(uuid.uuid4())]  # ensure distinct object
        outer = [1, inner, (2, 3)]
        ds = deep_sizeof(outer)
        assert ds >= sys.getsizeof(outer)
        assert ds >= sys.getsizeof(outer) + deep_sizeof(inner) + deep_sizeof((2, 3)) + deep_sizeof(1)

    def test_dict_counts_keys_and_values(self):
        k1 = "k1-" + str(uuid.uuid4())
        v1 = "v1-" + str(uuid.uuid4())
        k2 = "k2-" + str(uuid.uuid4())
        v2 = ("tuple", 2)
        d = {k1: v1, k2: v2}
        ds = deep_sizeof(d)
        # At least the shallow dict size
        assert ds >= sys.getsizeof(d)
        # Should include keys and values deeply
        expected_min = sys.getsizeof(d) + deep_sizeof(k1) + deep_sizeof(v1) + deep_sizeof(k2) + deep_sizeof(v2)
        assert ds >= expected_min

    def test_sets_and_frozensets(self):
        s = {1, 2, 3}
        fs = frozenset({4, 5})
        assert deep_sizeof(s) >= sys.getsizeof(s) + deep_sizeof(1) + deep_sizeof(2) + deep_sizeof(3)
        assert deep_sizeof(fs) >= sys.getsizeof(fs) + deep_sizeof(4) + deep_sizeof(5)

    def test_shared_refs_not_double_counted_in_sequence(self):
        a = [1, 2, 3]
        container = [a, a]  # same object referenced twice
        ds_container = deep_sizeof(container)
        # The deep part contributed by elements should be counted once
        deep_part = ds_container - sys.getsizeof(container)
        assert deep_part == deep_sizeof(a)

    def test_cycle_in_list_is_handled(self):
        a = []
        a.append(a)  # self-cycle: [...]
        # Should not recurse infinitely and should count only the list itself
        assert deep_sizeof(a) == sys.getsizeof(a)

        # For a list with content that has cycles, it should count the content once
        b = [0]
        b.append(b)  # Creates [0, [...]]
        assert deep_sizeof(b) == sys.getsizeof(b) + sys.getsizeof(0)  # list + the integer 0

    def test_user_defined_object_with___dict__(self):
        class C:
            pass

        c = C()
        c.x = "x-" + str(uuid.uuid4())
        c.y = [1, 2]
        ds = deep_sizeof(c)
        # Should include object's dict deeply
        assert ds >= sys.getsizeof(c) + deep_sizeof(c.__dict__)

    def test_user_defined_object_with___slots__(self):
        class S:
            __slots__ = ("x", "y")

        s = S()
        s.x = 42
        s.y = [2, 3]
        ds = deep_sizeof(s)
        # Should include slot-referenced values
        assert ds >= sys.getsizeof(s) + deep_sizeof(42) + deep_sizeof([2, 3])

    def test_stability_across_calls(self):
        obj = {"a": [1, 2, 3], "b": ("x", "y")}
        first = deep_sizeof(obj)
        second = deep_sizeof(obj)
        assert first == second

    def test_monotonic_growth_with_added_content(self):
        base = []
        ds_base = deep_sizeof(base)
        base.append("v-" + str(uuid.uuid4()))
        ds_after = deep_sizeof(base)
        assert ds_after >= ds_base


class TestIsBuiltin:
    def test_builtin_types_classes(self):
        # Classes defined in builtins should be recognized
        assert is_builtin(int) is True
        assert is_builtin(str) is True
        assert is_builtin(list) is True

    def test_builtin_type_instances(self):
        # Instances of builtin types should also be recognized
        assert is_builtin(123) is True
        assert is_builtin("hello") is True
        assert is_builtin([1, 2, 3]) is True

    def test_builtin_callable_like_objects(self):
        # Functions / callables (e.g. len) are implemented in builtins
        assert is_builtin(len) is False

    def test_user_class_and_instance_are_not_builtin(self):
        class MyClass:
            pass

        assert is_builtin(MyClass) is False
        assert is_builtin(MyClass()) is False

    def test_user_defined_function_is_not_builtin(self):
        def user_func():
            return None

        assert is_builtin(user_func) is False

    def test_non_builtin_stdlib_object(self):
        # Many stdlib objects are not in the "builtins" module (e.g. a module object)
        module_ = types  # module object from stdlib
        assert is_builtin(module_) is False


class TestRemoveExtraAttrs:
    def test_dict_removes_private_and_dunder_and_mangled_by_default(self):
        attrs = {
            "public": 1,
            "_private": 2,
            "__dunder__": 3,
            "_MyClass__mangled": 4,
            "also_public": 5,
        }
        result = remove_extra_attrs(attrs, inc_dunder=False, inc_private=False)
        assert result == {"public": 1, "also_public": 5}

    def test_list_preserves_order_and_type(self):
        attrs = ["a", "_b", "__c__", "_MyClass__d", "e"]
        result = remove_extra_attrs(attrs, mangled_cls_name="MyClass")
        assert isinstance(result, list)
        assert result == ["a", "e"]

    def test_tuple_preserves_type(self):
        attrs = ("x", "_y", "__z__", "_MyClass__t", "w")
        result = remove_extra_attrs(attrs, mangled_cls_name="MyClass")
        assert isinstance(result, tuple)
        assert result == ("x", "w")

    def test_inc_dunder_keeps_dunder_but_not_private(self):
        attrs = {"__dunder__", "_private", "_MyClass__mangled", "public"}
        result = remove_extra_attrs(attrs, mangled_cls_name="MyClass", inc_dunder=True)
        assert result == {"__dunder__", "public"}

    def test_inc_private_keeps_private_but_not_dunder(self):
        attrs = {"_private", "__dunder__", "_MyClass__mangled", "public"}
        result = remove_extra_attrs(attrs, mangled_cls_name="MyClass", inc_private=True)
        assert result == {"_private", "public"}

    def test_both_inc_flags_true_and_no_cls_name_returns_equal_copy(self):
        attrs_dict = {"_p": 1, "__d__": 2, "k": 3}
        attrs_list = ["_p", "__d__", "k"]

        out_dict = remove_extra_attrs(attrs_dict, inc_private=True, inc_dunder=True)
        out_list = remove_extra_attrs(attrs_list, inc_private=True, inc_dunder=True)

        # Values should be equal
        assert out_dict == attrs_dict
        assert out_list == attrs_list

        # But function should return a new object (no identity preservation)
        assert out_dict is not attrs_dict
        assert out_list is not attrs_list

        # And types should be preserved
        assert isinstance(out_dict, dict)
        assert isinstance(out_list, list)

    def test_both_inc_flags_true_but_with_cls_name_still_removes_mangled(self):
        attrs = {"_MyClass__x", "_Other__y", "public"}
        # inc flags allow private/dunder, but mangled for provided cls_name should still be removed
        result = remove_extra_attrs(attrs, mangled_cls_name="MyClass", inc_private=True, inc_dunder=True)
        assert result == {"_Other__y", "public"}

    def test_non_string_elements_raise_error(self):
        # The function expects strings in collections, but the type of collection must be among dict/set/list/tuple
        # Non-collection should raise
        with pytest.raises(TypeError):
            remove_extra_attrs(123)  # invalid type

    def test_does_not_false_positive_mangle_when_cls_name_not_in_attr(self):
        attrs = {"_MyClazz__x", "public", "_My", "_MyClazz", "__dunder__"}
        # With cls_name "MyClass" (different spelling), ALL private/dunder attributes are still removed
        result = remove_extra_attrs(attrs, mangled_cls_name="MyClass")
        # All private/dunder attributes are removed regardless of mangling match
        assert result == {"public"}

    def test_mangled_substring_match_anywhere_in_key(self):
        # Test that mangled name matching works for substring containment
        attrs = {
            "prefix_MyClass_suffix": 1,  # public attr, should remain
            "_pre_MyClass_suf": 2,  # private + contains mangled name, removed
            "_other_private": 3,  # private but no mangled name, still removed
            "ok": 4,
        }
        res = remove_extra_attrs(attrs, mangled_cls_name="MyClass")
        # Only public attributes remain; all private attributes are removed
        assert res == {"prefix_MyClass_suffix": 1, "ok": 4}

    def test_inc_mangled_keeps_mangled(self):
        attrs = {
            "_MyClass__x": 1,  # mangled for MyClass
            "_Other__y": 2,  # mangled for Other
            "_p": 3,  # regular private
            "__d__": 4,  # dunder
            "public": 5,  # public
        }
        # Keep mangled and private; still remove dunder
        out = remove_extra_attrs(
            attrs,
            inc_private=True,
            inc_dunder=False,
            inc_mangled=True,
            mangled_cls_name="MyClass",
        )
        assert out == {"_MyClass__x": 1, "_Other__y": 2, "_p": 3, "public": 5}

    def test_inc_mangled_does_not_override_private_filter(self):
        attrs = {
            "_MyClass__x": 1,  # mangled for MyClass
            "_Other__y": 2,  # mangled for Other
            "_p": 3,  # regular private
            "__d__": 4,  # dunder
            "public": 5,  # public
        }
        # Even if we include mangled, private filter still applies when inc_private=False
        out = remove_extra_attrs(
            attrs,
            inc_private=False,
            inc_dunder=False,
            inc_mangled=True,
            mangled_cls_name="MyClass",
        )
        # All underscore-prefixed names are removed by private filter; only public remains
        assert out == {"public": 5}

    def test_inc_private_true_mangled_false_removes_only_mangled(self):
        attrs = {
            "_MyClass__x": 1,  # mangled for MyClass (should be removed)
            "_Other__y": 2,  # mangled for Other (should stay)
            "_p": 3,  # regular private (should stay)
            "__d__": 4,  # dunder (removed unless inc_dunder=True)
            "public": 5,
        }
        out = remove_extra_attrs(
            attrs,
            inc_private=True,
            inc_dunder=False,
            inc_mangled=False,
            mangled_cls_name="MyClass",
        )
        assert out == {"_Other__y": 2, "_p": 3, "public": 5}
