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
                      is_property
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


class TestAttrsTools:

    def test_attrs_eq_names(self):
        class One:
            a = "a"
            b = "b"

        class Two:
            c = 3

        print_method()
        print_title("One")
        print(One)
        print_title("Two")
        print(Two)
        print("attrs_eq_names(One):", attrs_eq_names(One))
        print("attrs_eq_names(Two):", attrs_eq_names(Two))
        with pytest.raises(ValueError):
            print("attrs_eq_names(Two): Check it raises ValueError Exception...", end="")
            attrs_eq_names(Two, raise_exception=True)

    def test_attrs_search(self):
        print_method()

        print_title("Classes")
        print("attrs_search(ObjClass)\n", attrs_search(ObjClass))
        print("attrs_search(ObjClass, inc_private=True)\n",
              attrs_search(ObjClass, inc_private=True))
        print("attrs_search(DatClassDeep, inc_private=True, inc_property=True)\n",
              attrs_search(DatClassDeep, inc_private=True, inc_property=True))

        print_title("Instances")
        print("attrs_search(DatClassDeep()))\n", attrs_search(DatClassDeep()))

        print("attrs_search(DatClassDeep(), inc_private=True))\n", attrs_search(DatClassDeep(), inc_private=True))
        print(
            "attrs_search(DatClassDeep(), inc_private=True, inc_property=True)",
            attrs_search(DatClassDeep(), inc_private=True, inc_property=True))

        print_title("bultin Classes")
        print("attrs_search(int)", attrs_search(int))
        print("attrs_search(str)", attrs_search(str))
        print("attrs_search(1)", attrs_search(1))
        print("attrs_search('_')", attrs_search("_"))


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

    def test_user_defined_class_and_instance_are_not_builtin(self):
        class MyClass:
            pass

        assert is_builtin(MyClass) is False
        assert is_builtin(MyClass()) is False

    def test_user_defined_function_is_not_builtin(self):
        def user_func():
            return None

        assert is_builtin(user_func) is False

    def test_non_builtin_standard_library_object(self):
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

    def test_acts_like_image_with_fake_instance(self):
        img = FakeImage()
        assert acts_like_image(img) is True

    def test_acts_like_image_with_fake_type(self):
        assert acts_like_image(FakeImage) is True

    def test_acts_like_image_negative_cases(self):
        class NotImage: pass

        assert acts_like_image(NotImage) is False
        assert acts_like_image(object()) is False

    def test_class_name_assert(self):
        obj = ObjClass()
        assert class_name(obj, fully_qualified=False) == "ObjClass", "Should return obj Class name"
        assert class_name(obj, fully_qualified=True) == "test_abc.ObjClass", "Should return obj Module.Class name specs"

    def test_class_name_plain(self):
        print_method()
        fq_names = True
        for x in TESTABLE.classes:
            print(f"\n{type(x)}")
            print(class_name(x, fully_qualified=True, fully_qualified_builtins=False))

    def test_class_name_types(self):
        print_method()
        print_title("Instances", end="")
        fq_names = True
        fq_buitins = False
        for x in TESTABLE.instances:
            print(f"\n{type(x)}")
            print(class_name(x, fully_qualified=fq_names, fully_qualified_builtins=fq_buitins,
                             start="< -- ", end=" -- >"))
        print_title("Classes", end="")
        for x in TESTABLE.classes:
            print(f"\n{type(x)}")
            print(class_name(x, fully_qualified=fq_names, fully_qualified_builtins=fq_buitins))

    def test_is_property(self):
        print_method()
        print_title("Class")

        print("is_property('property_good', DatClassDeep)                   :",
              is_property("property_good", DatClassDeep))
        print("is_property('property_except', DatClassDeep)                 :",
              is_property("property_except", DatClassDeep))
        print("is_property('property_good', DatClassDeep, try_callable=True):",
              is_property("property_good", DatClassDeep, try_callable=True))
        print("is_property('invalid', DatClassDeep)                         :", is_property("invalid", DatClassDeep))

        print_title("Instance")
        print("is_property('property_good', DatClassDeep())                     :",
              is_property("property_good", DatClassDeep()))
        print("is_property('property_except', DatClassDeep())                   :",
              is_property("property_except", DatClassDeep()))
        print("is_property('property_good', DatClassDeep(), try_callable=True)  :",
              is_property("property_good", DatClassDeep(), try_callable=True))
        print("is_property('property_except', DatClassDeep(), try_callable=True):",
              is_property("property_except", DatClassDeep(), try_callable=True))
