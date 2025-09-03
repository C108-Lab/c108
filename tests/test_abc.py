#
# C108 - ABC Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import types
from dataclasses import dataclass, field

# Third-party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.abc import (ObjectInfo,
                      acts_like_image,
                      attrs_eq_names,
                      attrs_search,
                      class_name,
                      is_builtin,
                      attr_is_property
                      )

from c108.tools import print_method, print_title


# Classes --------------------------------------------------------------------------------------------------------------

class ObjClass:
    b: int = 1
    a: int = 0
    _private: int = 0
    __dunder: int = 0

    def __init__(self):
        self.c = "c"
        self.d = "d"
        self.none = None

    def a_method(self):
        return 0


@dataclass
class DatClass:
    b: int = 1  # field is instance attr
    a: int = 0  # field is instance attr

    def a_method(self):
        return 0


# Non-frozen Dataclass
@dataclass
class DatDynamic:
    aa: int = 12  # field is instance attr
    bb: int = 24  # field is instance attr


# Frozen Dataclass
@dataclass(frozen=True)
class DatFrozen:
    p: int = 0


class ObjAsDict:
    a = 0  # class attr
    to_dict = {"a": "zero"}


class ObjAsDictCallable:
    aa = 0  # class attr

    def to_dict(self):
        return {"aa": "zero"}


@dataclass
class DataClass:
    a = 0  # !!! <-- without type this is a class attr but NOT a dataclass field
    b: int = 1
    c: int = field(default=2)
    d: DatDynamic = field(default_factory=DatDynamic)
    o: type = ObjAsDict


@dataclass
class DatClassDeep:
    a: int = 0
    d: DatDynamic = field(default_factory=DatDynamic)
    f: DatFrozen = DatFrozen(p=1)
    dict: dict = field(default_factory=dict)
    list: list = field(default_factory=list)
    _private: int = 0
    __dunder: int = 22
    ___tripler: int = 333

    @property
    def property_good(self):
        return self.a + 1

    @property
    def property_except(self):
        raise ValueError("Non-calculatable property")

    def dat_method(self):
        return 0


# python
class FakeImage:
    # emulate PIL.Image.Image's API via properties/methods on the class
    def __init__(self, size=(10, 20), mode="RGB", fmt="PNG"):
        self._size = size
        self._mode = mode
        self._format = fmt

    @property
    def size(self): return self._size

    @property
    def mode(self): return self._mode

    @property
    def format(self): return self._format

    def save(self, fp): pass

    def show(self): pass

    def resize(self, size): return FakeImage(size=size, mode=self.mode, fmt=self.format)

    def crop(self, box): return self


class Testable:
    instances: list
    classes: list

    def __init__(self):
        # Primitives
        primitives = [int(1), float(2), bool(True), complex(1 + 2j), "QWERTY"]
        self.primitives = primitives

        # Instances
        items = (
            [1], (1, 2),
            {'a': 1, 'b': 2, 'c': 2}, {1, 2, 3}, frozenset({1, 2, 3}),
            range(0, 14)
        )
        bytes_ = bytes("QWERTY", "utf-8")
        image_ = FakeImage()
        instance_obj = ObjClass()
        instanse_datdeep = DatClassDeep(dict={1: "one", 2: "two"}, list=[1, 2, 3])
        self.instances = [*primitives, *items, bytes_, image_, instance_obj, instanse_datdeep]

        # Deep Instances
        self.deep_instances = [instance_obj, instanse_datdeep]

        # Classes
        self.classes = [ObjClass, DatClass, DatClassDeep, int]

        # Deep recursion items
        dict_deep = {1: "a", 2: "b", 7: {11: "2a", 22: "2b", 14: {111: "3a", 222: "3b"}}}
        list_deep = [1, 2, [11, 22, [111, 222, 333]]]
        tuple_deep = (1, 2, (11, 22, (111, 222, 333)))
        set_deep = {1, 2, frozenset({11, 22, frozenset({111, 222, 333})})}  # We use inner frozenset to make it hashable
        self.deep_items = [list_deep, tuple_deep, set_deep, dict_deep]

        # Oversized Instances
        str_over = ' '.join(str(i) for i in range(1080))
        bytes_over = str_over.encode('utf-8')
        bytearray_over = bytearray(str_over, 'utf-8')
        memoryview_over = memoryview(bytes_over)
        list_over = [i for i in range(1080)]
        tuple_over = tuple(i for i in range(1080))
        set_over = {i for i in range(1080)}
        dict_over = {i: i ** 2 for i in range(1080)}

        self.oversized_objects = [str_over, bytes_over, bytearray_over, memoryview_over,
                                  list_over, tuple_over, set_over, dict_over]


TESTABLE = Testable()


# Tests ----------------------------------------------------------------------------------------------------------------

class TestActsLikeImage:

    def test_acts_like_image_with_fake_instance(self):
        img = FakeImage()
        assert acts_like_image(img) is True

    def test_acts_like_image_with_fake_type(self):
        assert acts_like_image(FakeImage) is True

    def test_acts_like_image_negative_cases(self):
        class NotImage: pass

        assert acts_like_image(NotImage) is False
        assert acts_like_image(object()) is False


class TestAttrIsProperty:

    def test_class_try_callable_false_true(self):
        class C:
            @property
            def x(self):
                return 1

        # On classes, properties are detectable
        assert attr_is_property("x", C, try_callable=False) is True
        # But try_callable=True on a class must return False
        assert attr_is_property("x", C, try_callable=True) is False

    def test_instance_try_callable_false_true(self):
        class C:
            @property
            def x(self):
                return 42

        obj = C()
        # On instances, property is identified
        assert attr_is_property("x", obj, try_callable=False) is True
        # With try_callable=True, getter is invoked and succeeds -> True
        assert attr_is_property("x", obj, try_callable=True) is True

    def test_property_false_when_it_raises(self):
        class C:
            def __init__(self, ok):
                self._ok = ok

            @property
            def x(self):
                if not self._ok:
                    raise ValueError("boom")
                return "ok"

        good = C(ok=True)
        bad = C(ok=False)

        # try_callable=False doesn't call the getter, still a property
        assert attr_is_property("x", bad, try_callable=False) is True
        # try_callable=True calls the getter -> good is True, bad is False
        assert attr_is_property("x", good, try_callable=True) is True
        assert attr_is_property("x", bad, try_callable=True) is False

    def test_non_property_returns_false(self):
        class C:
            x = 5

            def method(self):
                return 10

            @staticmethod
            def s():
                return 0

            @classmethod
            def c(cls):
                return 0

        obj = C()
        assert attr_is_property("x", C) is False
        assert attr_is_property("method", C) is False
        assert attr_is_property("s", C) is False
        assert attr_is_property("c", C) is False

        assert attr_is_property("x", obj) is False
        assert attr_is_property("method", obj) is False
        assert attr_is_property("s", obj) is False
        assert attr_is_property("c", obj) is False

    def test_missing_returns_false(self):
        class C:
            pass

        obj = C()
        assert attr_is_property("missing", C) is False
        assert attr_is_property("missing", obj) is False

    def test_dataclass_property(self):
        @dataclass
        class D:
            value: int

            @property
            def doubled(self) -> int:
                return self.value * 2

        # On dataclass class: detectable with try_callable=False
        assert attr_is_property("doubled", D, try_callable=False) is True
        # And forced False with try_callable=True (per contract)
        assert attr_is_property("doubled", D, try_callable=True) is False

        # On instances: behaves like a normal class
        d = D(3)
        assert attr_is_property("doubled", d, try_callable=False) is True
        assert attr_is_property("doubled", d, try_callable=True) is True


class TestAttrsSearch:

    def test_returns_empty_for_builtin_types(self):
        assert attrs_search(int) == []
        assert attrs_search(1) == []

    def test_includes_data_attrs_excludes_callables_and_dunder(self):
        class C:
            def __init__(self):
                self.x = 1
                self._y = 2

            def method(self):
                return 10

            def __str__(self):
                return "C"

        c = C()
        # Default: exclude private, callables, and dunder
        assert attrs_search(c) == ["x"]

    def test_private_attr_inclusion_flag(self):
        class C:
            def __init__(self):
                self.x = 1
                self._y = 2

        c = C()
        assert attrs_search(c, inc_private=False) == ["x"]
        assert set(attrs_search(c, inc_private=True)) == {"x", "_y"}

    def test_property_inclusion_on_instance(self):
        class C:
            def __init__(self, val):
                self._val = val

            @property
            def p(self):
                return self._val

        c = C(5)
        # By default properties are excluded
        assert "p" not in attrs_search(c)
        # When inc_property=True, include instance properties
        assert "p" in attrs_search(c, inc_property=True)

    def test_property_returning_none_respects_inc_none_attrs(self):
        class C:
            @property
            def p(self):
                return None

        c = C()
        # inc_property=True, inc_none_attrs=True -> include
        assert "p" in attrs_search(c, inc_property=True, inc_none_attrs=True)
        # inc_property=True, inc_none_attrs=False -> exclude because value is None
        assert "p" not in attrs_search(c, inc_property=True, inc_none_attrs=False)

    def test_property_raising_on_instance_is_included_when_flag_true(self):
        class C:
            @property
            def p(self):
                raise ValueError("boom")

        c = C()
        # Default (inc_property=False): exclude properties
        assert "p" not in attrs_search(c, inc_property=False)
        # When inc_property=True, include even if getter raises (treated as present)
        assert "p" in attrs_search(c, inc_property=True)

    def test_class_property_on_class_object(self):
        class C:
            @property
            def p(self):
                return 1

        # On the class object, property should be included only when inc_property=True
        assert "p" not in attrs_search(C, inc_property=False)
        assert "p" in attrs_search(C, inc_property=True)

    def test_staticmethod_and_classmethod_are_excluded(self):
        class C:
            x = 1

            @staticmethod
            def s():
                return 0

            @classmethod
            def c(cls):
                return 0

        c = C()
        # Methods are callable and must be excluded regardless of inc_property
        assert "s" not in attrs_search(C)
        assert "c" not in attrs_search(C)
        assert "s" not in attrs_search(c, inc_property=True)
        assert "c" not in attrs_search(c, inc_property=True)
        # Data attribute remains
        assert "x" in attrs_search(c)

    def test_inc_none_attrs_false_excludes_plain_none_attributes(self):
        class C:
            def __init__(self):
                self.x = None
                self.y = 1

        c = C()
        assert set(attrs_search(c, inc_none_attrs=True)) == {"x", "y"}
        assert set(attrs_search(c, inc_none_attrs=False)) == {"y"}


class TestAttrsEqNames:

    def test_all_attributes_match_names_case_insensitive(self):
        """Test when all attributes match their names (case insensitive)."""

        class TestObj:
            name = "name"
            value = "VALUE"  # Different case
            test = "test"

        obj = TestObj()
        assert attrs_eq_names(obj) is True
        assert attrs_eq_names(obj, case_sensitive=False) is True

    def test_all_attributes_match_names_case_sensitive(self):
        """Test when all attributes match their names (case sensitive)."""

        class TestObj:
            name = "name"
            value = "value"
            test = "test"

        obj = TestObj()
        assert attrs_eq_names(obj, case_sensitive=True) is True

    def test_case_sensitive_mismatch(self):
        """Test case sensitive comparison with mismatched case."""

        class TestObj:
            name = "NAME"  # Different case

        obj = TestObj()
        assert attrs_eq_names(obj, case_sensitive=True) is False
        assert attrs_eq_names(obj, case_sensitive=False) is True

    def test_attribute_value_mismatch_returns_false(self):
        """Test when attribute value doesn't match name."""

        class TestObj:
            name = "name"
            value = "different_value"

        obj = TestObj()
        assert attrs_eq_names(obj) is False

    def test_attribute_value_mismatch_raises_exception(self):
        """Test raising exception on mismatch."""

        class TestObj:
            name = "wrong_name"

        obj = TestObj()
        with pytest.raises(ValueError, match="Attribute 'name' with value 'wrong_name' does not match its name"):
            attrs_eq_names(obj, raise_exception=True)

    def test_empty_object(self):
        """Test object with no attributes."""

        class EmptyObj:
            pass

        obj = EmptyObj()
        assert attrs_eq_names(obj) is True

    def test_ignores_methods(self):
        """Test that callable methods are ignored."""

        class TestObj:
            name = "name"

            def some_method(self):
                return "method"

        obj = TestObj()
        assert attrs_eq_names(obj) is True

    def test_ignores_dunder_attributes(self):
        """Test that dunder attributes are ignored."""

        class TestObj:
            name = "name"
            __private = "private"  # This gets name-mangled to _TestObj__private
            __dict__ = {}  # This is a dunder attribute

        obj = TestObj()
        # Should only check 'name', ignoring all private/dunder/mangled attrs
        assert attrs_eq_names(obj) is True

    def test_numeric_attributes_converted_to_string(self):
        """Test that numeric values are converted to strings for comparison."""

        class TestObj:
            number = 123  # Will be compared as "123" vs "number"

        obj = TestObj()
        assert attrs_eq_names(obj) is False

    def test_numeric_attributes_matching_string_names(self):
        """Test numeric attributes that match when converted to string."""

        class TestObj:
            pass

        obj = TestObj()
        setattr(obj, '123', 123)  # Attribute name '123' with value 123
        assert attrs_eq_names(obj) is True

    def test_boolean_attributes(self):
        """Test boolean attributes converted to strings."""

        class TestObj:
            pass

        obj = TestObj()
        setattr(obj, 'True', True)  # Should match
        setattr(obj, 'False', False)  # Should match
        assert attrs_eq_names(obj) is True

        # Test mismatch
        setattr(obj, 'true', True)  # 'true' != 'True'
        assert attrs_eq_names(obj, case_sensitive=True) is False
        assert attrs_eq_names(obj, case_sensitive=False) is True

    def test_none_values(self):
        """Test attributes with None values."""

        class TestObj:
            pass

        obj = TestObj()
        setattr(obj, 'None', None)  # 'None' == str(None) == 'None'
        assert attrs_eq_names(obj) is True

        setattr(obj, 'null', None)  # 'null' != 'None'
        assert attrs_eq_names(obj) is False

    def test_mixed_matching_and_non_matching(self):
        """Test object with both matching and non-matching attributes."""

        class TestObj:
            good = "good"
            bad = "wrong"

        obj = TestObj()
        assert attrs_eq_names(obj) is False

    def test_first_mismatch_stops_execution_with_exception(self):
        """Test that exception is raised on first mismatch, not later ones."""

        class TestObj:
            first = "wrong"
            second = "also_wrong"

        obj = TestObj()
        # Should raise exception for 'first', not 'second'
        with pytest.raises(ValueError, match="Attribute 'first'"):
            attrs_eq_names(obj, raise_exception=True)

    def test_special_characters_in_values(self):
        """Test attributes with special characters."""

        class TestObj:
            pass

        obj = TestObj()
        setattr(obj, 'special_chars', 'special_chars')
        setattr(obj, 'with-dash', 'with-dash')
        setattr(obj, 'with_underscore', 'with_underscore')
        assert attrs_eq_names(obj) is True

    def test_unicode_attributes(self):
        """Test unicode character handling."""

        class TestObj:
            pass

        obj = TestObj()
        setattr(obj, 'café', 'café')
        setattr(obj, '测试', '测试')
        assert attrs_eq_names(obj) is True


class TestClassName:
    def test_builtin_default(self):
        # Builtins should return just the class name by default
        assert class_name(int) == "int"
        assert class_name(123) == "int"
        assert class_name(str) == "str"
        assert class_name("abc") == "str"

    def test_fully_qualified_builtins(self):
        # When fully_qualified_builtins is True, include the module for builtins
        assert class_name(int, fully_qualified_builtins=True) == "builtins.int"
        assert class_name(1, fully_qualified_builtins=True) == "builtins.int"
        assert class_name("x", fully_qualified_builtins=True) == "builtins.str"

    def test_user_fully_qualified(self):
        class MyClass:
            pass

        # Default fully_qualified=True for non-builtin classes includes module path
        expected = f"{MyClass.__module__}.{MyClass.__name__}"
        assert class_name(MyClass) == expected
        assert class_name(MyClass()) == expected

    def test_user_not_fully_qualified(self):
        class MyClass:
            pass

        # When fully_qualified=False, return only the class name
        assert class_name(MyClass, fully_qualified=False) == "MyClass"
        assert class_name(MyClass(), fully_qualified=False) == "MyClass"

    def test_start_and_end_wrapping(self):
        class MyClass:
            pass

        # Wrapping should apply regardless of builtin/non-builtin and flags
        assert class_name(1, start="<", end=">") == "<int>"
        assert class_name(int, fully_qualified_builtins=True, start="[", end="]") == "[builtins.int]"

        expected = f"{MyClass.__module__}.{MyClass.__name__}"
        assert class_name(MyClass(), start="{", end="}") == "{" + expected + "}"

    def test_subclass_of_builtin_is_treated_as_non_builtin(self):
        class MyList(list):
            pass

        # Subclass of a builtin is not itself builtin, so default is fully qualified
        expected = f"{MyList.__module__}.{MyList.__name__}"
        assert class_name(MyList) == expected
        assert class_name(MyList([1, 2])) == expected


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


class TestObjectInfo:

    def test_from_object(self):
        print_method()

        for x in TESTABLE.instances:
            print(f"\n{type(x)}")
            print(type(ObjectInfo.from_object(x, fully_qualified=False)))
            print(ObjectInfo.from_object(x, fully_qualified=False))

    def test_to_str_value(self):
        print_method()
        print_title("Instances", end="")
        for x in TESTABLE.instances:
            print(f"\n{class_name(x, fully_qualified=True)}")
            print(ObjectInfo.from_object(x, fully_qualified=True).as_str)
        print_title("Classes", end="")
        for x in TESTABLE.classes:
            print(f"\n{class_name(x, fully_qualified=True)}")
            print(ObjectInfo.from_object(x, fully_qualified=True).as_str)


class TestUtilMethods:

    def test_is_property(self):
        print_method()
        print_title("Class")

        print("attr_is_property('property_good', DatClassDeep)                   :",
              attr_is_property("property_good", DatClassDeep))
        print("attr_is_property('property_except', DatClassDeep)                 :",
              attr_is_property("property_except", DatClassDeep))
        print("attr_is_property('property_good', DatClassDeep, try_callable=True):",
              attr_is_property("property_good", DatClassDeep, try_callable=True))
        print("attr_is_property('invalid', DatClassDeep)                         :",
              attr_is_property("invalid", DatClassDeep))

        print_title("Instance")
        print("attr_is_property('property_good', DatClassDeep())                     :",
              attr_is_property("property_good", DatClassDeep()))
        print("attr_is_property('property_except', DatClassDeep())                   :",
              attr_is_property("property_except", DatClassDeep()))
        print("attr_is_property('property_good', DatClassDeep(), try_callable=True)  :",
              attr_is_property("property_good", DatClassDeep(), try_callable=True))
        print("attr_is_property('property_except', DatClassDeep(), try_callable=True):",
              attr_is_property("property_except", DatClassDeep(), try_callable=True))
