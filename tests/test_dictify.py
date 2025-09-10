#
# C108 - Dictify Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
from dataclasses import dataclass, field

# Third-party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.utils import class_name
from c108.dictify import as_dict, core_to_dict, filter_attrs
from c108.tools import print_title


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

class TestCoreToDict:

    def test_core_to_dict(self):
        z = ObjClass()
        z.none = None
        z.yes = "yes"
        objects = [0, {"x": None, "y": 1}, z]
        for o in objects:
            print_title(str(o))
            print(core_to_dict(o, inc_none_items=True, inc_none_attrs=True, recursion_depth=2))


class TestAsDict_and_AttrsTools:

    def test_as_dict(self):

        for recursion_depth in [0, 108]:
            print_title(f"Primitives | recursion_depth={recursion_depth}", end="")
            for x in [1, 3.14, {1: 1, 2: 2}, [1, 2, 3]]:
                print(f" {class_name(x, fully_qualified=True):10} {str(x):32} >> ", end=" ")
                print(as_dict(x, recursion_depth=recursion_depth))

        for recursion_depth in [0, 108]:
            print_title(f"Instances | recursion_depth={recursion_depth}", end="")
            for x in [ObjClass(), DatClassDeep(dict={1: 1, 2: 2}, list=[1, 2, 3])]:
                print(f" {class_name(x, fully_qualified=True)}")
                print(as_dict(x, recursion_depth=recursion_depth))

        print("\nNOTE: notice how as_dict() returns as-is-values for builtins.")

    def test_filter_attrs(self):

        recursion_depth = 0
        print_title(f"Instances | always_filter=[complex], recursion_depth={recursion_depth}", end="")
        for x in TESTABLE.instances:
            print(f" {str(type(x)):42}  {filter_attrs(x, always_filter=[complex], recursion_depth=recursion_depth)}")

        recursion_depth = 1
        print_title(f"Instances | always_filter=[complex], recursion_depth={recursion_depth}", end="")
        for x in TESTABLE.instances:
            print(f" {str(type(x)):42}  {filter_attrs(x, always_filter=[complex], recursion_depth=recursion_depth)}")

        recursion_depth = 2
        print_title(f"Instances | always_filter=[complex], recursion_depth={recursion_depth}", end="")
        for x in TESTABLE.instances:
            print()
            print("to_str:\n  ", x, type(x))
            print("filtered:\n  ", filter_attrs(x, always_filter=[complex], recursion_depth=recursion_depth))

        recursion_depth = 2
        print_title(f"Deep Instances - Private and Properties "
                    f"| inc_class_name=True, inc_private=True, inc_property=True, recursion_depth={recursion_depth}",
                    end="")
        for x in TESTABLE.deep_instances:
            print()
            print("to_str:\n  ", x, type(x))
            print("filtered:\n  ",
                  filter_attrs(x, inc_class_name=True, inc_private=True, inc_property=True,
                               recursion_depth=recursion_depth))

        recursion_depth = 2
        print_title(f"Classes | always_filter=[complex], recursion_depth={recursion_depth}", end="")
        for x in TESTABLE.classes:
            print()
            print("to_str:\n  ", x, type(x))
            print("filtered:\n  ", filter_attrs(x, always_filter=[complex], recursion_depth=recursion_depth))

        recursion_depth = 2
        print_title(f"Deep Items | recursion_depth={recursion_depth}", end="")
        for x in TESTABLE.deep_items:
            print()
            print("to_str:\n  ", type(x), str(x))
            print("filtered:\n  ", filter_attrs(x, recursion_depth=recursion_depth))

        recursion_depth = 1080
        print_title(f"Oversized Objects | recursion_depth={recursion_depth}", end="")
        for x in TESTABLE.oversized_objects:
            print("to_str:", type(x))
            print(filter_attrs(x, recursion_depth=recursion_depth))
