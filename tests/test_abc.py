#
# C108 - ABC Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import sys
import types
import uuid
from dataclasses import dataclass, field

# Third-party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.abc import (ObjectInfo,
                      acts_like_image,
                      attr_is_property,
                      attrs_eq_names,
                      attrs_search,
                      class_name,
                      deep_sizeof,
                      is_builtin,
                      remove_extra_attrs
                      )

from c108.tools import print_method, print_title


# Classes --------------------------------------------------------------------------------------------------------------

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

    def test_case_insensitive(self):
        """Test when all attributes match their names (case insensitive)."""

        class TestObj:
            name = "name"
            value = "VALUE"  # Different case
            test = "test"

        obj = TestObj()
        assert attrs_eq_names(obj) is True
        assert attrs_eq_names(obj, case_sensitive=False) is True

    def test_case_sensitive(self):
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

    def test_value_mismatch(self):
        """Test when attribute value doesn't match name."""

        class TestObj:
            name = "name"
            value = "different_value"

        obj = TestObj()
        assert attrs_eq_names(obj) is False

    def test_value_mismatch_raises(self):
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

    def test_ignores_dunder_attrs(self):
        """Test that dunder attributes are ignored."""

        class TestObj:
            name = "name"
            __private = "private"  # This gets name-mangled to _TestObj__private
            __dict__ = {}  # This is a dunder attribute

        obj = TestObj()
        # Should only check 'name', ignoring all private/dunder/mangled attrs
        assert attrs_eq_names(obj) is True

    def test_numeric_attrs_match_names(self):
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

    def test_first_mismatch_exception(self):
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


class TestObjectInfo:
    def test_numbers(self):
        oi = ObjectInfo.from_object(123)
        assert oi.type is int
        assert oi.unit == "bytes"
        assert isinstance(oi.size, int)
        assert oi.size == sys.getsizeof(123)
        assert oi.total_bytes == oi.size
        # class_name is derived from type
        assert "int" in oi.as_str

    def test_str(self):
        s = "hello world"
        oi = ObjectInfo.from_object(s)
        assert oi.type is str
        assert oi.unit == "chars"
        assert oi.size == len(s)
        assert isinstance(oi.total_bytes, int)
        assert oi.total_bytes == deep_sizeof(s)
        assert "<str" in oi.as_str or "str>" in oi.as_str

    def test_bytes_like(self):
        b = b"\x00\x01\x02"
        oi = ObjectInfo.from_object(b)
        assert oi.type is bytes
        assert oi.unit == "bytes"
        assert oi.size == len(b)
        assert oi.total_bytes == len(b)
        # Presentation includes unit
        assert "bytes" in oi.as_str

    def test_container_sequence(self):
        obj = [1, 2, 3]
        oi = ObjectInfo.from_object(obj)
        assert oi.unit == "items"
        assert oi.size == len(obj)
        assert oi.total_bytes == deep_sizeof(obj)
        # class_name resolved from type
        assert "list" in oi.class_name

    def test_class_object_counts_attrs(self):
        class C:
            a = 1
            b = 2

            def m(self):  # callable should not count as an attr
                return 3

        oi = ObjectInfo.from_object(C)
        # size is number of non-callable, non-private/dunder attributes discovered
        expected_attrs = attrs_search(C, inc_private=False, inc_property=False)
        assert oi.unit == "attrs"
        assert oi.size == len(expected_attrs)
        assert oi.total_bytes is None
        assert oi.type is C

    def test_instance_returns_tuple_and_bytes(self):
        class D:
            def __init__(self):
                self.x = "x"
                self.y = [1, 2]

        d = D()
        oi = ObjectInfo.from_object(d)
        assert isinstance(oi.size, tuple) and oi.unit == ("attrs", "bytes")
        n_attrs, total = oi.size
        assert n_attrs == len(attrs_search(d, inc_private=False, inc_property=False))
        assert total == oi.total_bytes == deep_sizeof(d)

    def test_instance_no_attrs_uses_bytes_scalar(self):
        class NoAttrs:
            __slots__ = ()  # no instance dict and no slots set

        x = NoAttrs()
        oi = ObjectInfo.from_object(x)
        assert oi.unit == "bytes"
        assert isinstance(oi.size, int)
        assert oi.total_bytes == oi.size == deep_sizeof(x)

    def test_post_init_raises_on_mismatched_size_unit_lengths(self):
        with pytest.raises(ValueError):
            ObjectInfo(type=int, size=(1, 2), unit=("one",))  # lengths differ

    def test_as_str_generic_tuple_formatting(self):
        oi = ObjectInfo(type=list, size=(3, 10), unit=("attrs", "bytes"), total_bytes=10)
        s = oi.as_str
        assert "attrs" in s and "bytes" in s
        assert "<" in s and ">" in s

    def test_as_str_image_formatting(self, monkeypatch):
        # Create a simple image-like stub
        class FakeImageSmall:
            def __init__(self, w, h):
                self.size = (w, h)

        img = FakeImageSmall(640, 480)

        # Patch acts_like_image in the same module where ObjectInfo is defined
        import sys
        mod = sys.modules[ObjectInfo.__module__]
        monkeypatch.setattr(mod, "acts_like_image", lambda o: isinstance(o, FakeImageSmall), raising=True)

        oi = ObjectInfo.from_object(img)
        assert oi.unit == ("width", "height", "Mpx")
        s = oi.as_str
        assert "640" in s and "480" in s
        assert "Mpx" in s
        assert "W⨯H" in s or "W×H" in s


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
