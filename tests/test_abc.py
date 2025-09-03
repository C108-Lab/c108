#
# C108 - ABC Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
from dataclasses import dataclass, field

# Third-party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.abc import (BiDirectionalMap,
                      ObjectInfo,
                      acts_like_image,
                      as_dict,
                      attrs_eq_names,
                      attrs_search,
                      class_name,
                      core_to_dict,
                      dict_get,
                      dict_set,
                      filter_attrs,
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

class TestBiDirectionalMap:

    @pytest.fixture
    def populated_map(self):
        """Fixture providing a pre-populated BiDirectionalMap instance."""
        return BiDirectionalMap({
            1: "apple",
            2: "banana",
            3: "cherry",
        })

    def test_lookup(self, populated_map):
        assert populated_map[3] == "cherry"
        assert populated_map.get_value(1) == "apple"
        assert populated_map.get_key("banana") == 2

    def test_value_uniqueness(self, populated_map):
        """Tests that adding a duplicate value raises ValueError."""
        with pytest.raises(ValueError, match="Value 'apple' already exists"):
            populated_map.add(4, "apple")  # Try to add an existing value

    def test_key_uniqueness(self, populated_map):
        """Tests that adding a duplicate key raises ValueError."""
        with pytest.raises(ValueError, match="Key '1' already exists"):
            populated_map.add(1, "grape")  # Try to add an existing key

    def test_contains(self, populated_map):
        assert 1 in populated_map  # Checks key
        assert "apple" in populated_map  # Checks value
        assert 99 not in populated_map
        assert "zebra" not in populated_map

    def test_keys_values_items(self, populated_map):
        assert sorted(list(populated_map.keys())) == [1, 2, 3]
        assert sorted(list(populated_map.values())) == ["apple", "banana", "cherry"]
        assert set(populated_map.items()) == {(1, "apple"), (2, "banana"), (3, "cherry")}


class TestCoreToDict:

    def test_core_to_dict(self):
        z = ObjClass()
        z.none = None
        z.yes = "yes"
        objects = [0, {"x": None, "y": 1}, z]
        for o in objects:
            print_title(str(o))
            print(core_to_dict(o, inc_none_items=True, inc_none_attrs=True, recursion_depth=2))


class TestAsDict_from_Attrs:

    def testto_dict_from_val(self):
        print_method()
        a = ObjAsDict()
        print(as_dict(a))

    def testto_dict_from_callable(self):
        print_method()
        aa = ObjAsDictCallable()
        print(as_dict(aa))


class TestAsDict_and_AttrsTools:

    def testto_dict(self):
        print_method()

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

    def test_attrs_eq_names(self):
        class One:
            a = "a"
            b = "b"

        class Two:
            c = 3

        print_method()
        print_title("One")
        print(as_dict(One))
        print_title("Two")
        print(as_dict(Two))
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

    def test_filter_attrs(self):
        print_method()

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
            print("as_str:\n  ", x, type(x))
            print("filtered:\n  ", filter_attrs(x, always_filter=[complex], recursion_depth=recursion_depth))

        recursion_depth = 2
        print_title(f"Deep Instances - Private and Properties "
                    f"| inc_class_name=True, inc_private=True, inc_property=True, recursion_depth={recursion_depth}",
                    end="")
        for x in TESTABLE.deep_instances:
            print()
            print("as_str:\n  ", x, type(x))
            print("filtered:\n  ",
                  filter_attrs(x, inc_class_name=True, inc_private=True, inc_property=True,
                               recursion_depth=recursion_depth))

        recursion_depth = 2
        print_title(f"Classes | always_filter=[complex], recursion_depth={recursion_depth}", end="")
        for x in TESTABLE.classes:
            print()
            print("as_str:\n  ", x, type(x))
            print("filtered:\n  ", filter_attrs(x, always_filter=[complex], recursion_depth=recursion_depth))

        recursion_depth = 2
        print_title(f"Deep Items | recursion_depth={recursion_depth}", end="")
        for x in TESTABLE.deep_items:
            print()
            print("as_str:\n  ", type(x), str(x))
            print("filtered:\n  ", filter_attrs(x, recursion_depth=recursion_depth))

        recursion_depth = 1080
        print_title(f"Oversized Objects | recursion_depth={recursion_depth}", end="")
        for x in TESTABLE.oversized_objects:
            print("as_str:", type(x))
            print(filter_attrs(x, recursion_depth=recursion_depth))


class TestDictGetSet:
    def test_dict_get(self):
        d = {"a": 1,
             "b": {"c": 2},
             "e": {"empty": None}
             }
        assert dict_get(d, dot_key="a") == 1, "Should return d['a']"
        assert dict_get(d, keys=["a"]) == 1, "Should return d['a']"
        assert dict_get(d, dot_key="b.c") == 2, "Should return d['b']['c']"
        assert dict_get(d, keys=["b", "c"]) == 2, "Should return d['b']['c']"
        assert dict_get(d, dot_key="e.empty") == "", "Should return ''"
        assert dict_get(d, dot_key="e.empty", default=None) is None, "Should return None"

    def test_dict_set(self):
        d = {"a": 0,
             "b": {"c": 0}
             }

        dict_set(d, dot_key="a", value=1)
        dict_set(d, dot_key="b.c", value=2)
        dict_set(d, dot_key="i.j.q", value=3)
        assert d["a"] == 1
        assert d["b"]["c"] == 2
        assert d["i"]["j"]["q"] == 3


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


class TestUtils:

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

    def test_is_builtin(self):
        pass

