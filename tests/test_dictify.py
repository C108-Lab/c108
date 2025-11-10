#
# C108 - Dictify Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import collections.abc as abc
import re
import sys
from dataclasses import dataclass, is_dataclass, fields
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

# Third-party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.dictify import (
    ClassNameOptions,
    DictifyOptions,
    HookMode,
    MetaMixin,
    Meta,
    SizeMeta,
    TrimMeta,
    TypeMeta,
    MetaOptions,
    UNSET,
    dictify_core,
    dictify,
    Handlers,
    _attr_is_property,
)
from tests.test_abc import Example, SimpleDataClass


# Classes --------------------------------------------------------------------------------------------------------------


@dataclass
class SimpleDC(MetaMixin):
    a: int
    b: str | None = None


@dataclass
class WithProps(MetaMixin):
    x: int
    y: int | None = None

    @property
    def sum(self) -> int:
        return self.x + (self.y or 0)

    @property
    def _hidden(self) -> str:  # should be ignored
        return "hidden"


class NotDataClass(MetaMixin):
    def __init__(self) -> None:
        self.z = 1


# Helper Classes Tests -------------------------------------------------------------------------------------------------


class TestClassNameOptions:
    """Test suite for ClassNameOptions.merge() method."""

    def test_merge_inject_class_name_true(self) -> None:
        """Enable class name injection in both expand and to_dict."""
        opts = ClassNameOptions()
        result = opts.merge(inject_class_name=True)
        assert result.in_expand is True
        assert result.in_to_dict is True
        assert result.key == "__class_name__"
        assert result.fully_qualified is False

    def test_merge_inject_class_name_false(self) -> None:
        """Disable class name injection in both expand and to_dict."""
        opts = ClassNameOptions(in_expand=True, in_to_dict=True)
        result = opts.merge(inject_class_name=False)
        assert result.in_expand is False
        assert result.in_to_dict is False
        assert result.key == "__class_name__"
        assert result.fully_qualified is False

    @pytest.mark.parametrize(
        "initial_expand, initial_to_dict, inject_value, expected_expand, expected_to_dict",
        [
            pytest.param(False, False, True, True, True, id="false_false_to_true_true"),
            pytest.param(True, True, False, False, False, id="true_true_to_false_false"),
            pytest.param(True, False, True, True, True, id="true_false_to_true_true"),
            pytest.param(False, True, False, False, False, id="false_true_to_false_false"),
        ],
    )
    def test_merge_inject_class_name_parametrized(
        self,
        initial_expand: bool,
        initial_to_dict: bool,
        inject_value: bool,
        expected_expand: bool,
        expected_to_dict: bool,
    ) -> None:
        """Test inject_class_name parameter with various initial states."""
        opts = ClassNameOptions(in_expand=initial_expand, in_to_dict=initial_to_dict)
        result = opts.merge(inject_class_name=inject_value)
        assert result.in_expand is expected_expand
        assert result.in_to_dict is expected_to_dict

    def test_merge_individual_attributes(self) -> None:
        """Update individual attributes without affecting others."""
        opts = ClassNameOptions(in_expand=True, key="old_key")
        result = opts.merge(in_to_dict=True, key="new_key", fully_qualified=True)
        assert result.in_expand is True  # Unchanged
        assert result.in_to_dict is True
        assert result.key == "new_key"
        assert result.fully_qualified is True

    def test_merge_all_attributes_at_once(self) -> None:
        """Replace all configuration options simultaneously."""
        opts = ClassNameOptions()
        result = opts.merge(in_expand=True, in_to_dict=True, key="custom_key", fully_qualified=True)
        assert result.in_expand is True
        assert result.in_to_dict is True
        assert result.key == "custom_key"
        assert result.fully_qualified is True

    def test_merge_chaining_operations(self) -> None:
        """Chain multiple merge operations sequentially."""
        opts = ClassNameOptions()
        result = opts.merge(inject_class_name=True).merge(fully_qualified=True).merge(key="@type")
        assert result.in_expand is True
        assert result.in_to_dict is True
        assert result.key == "@type"
        assert result.fully_qualified is True


class TestMeta:
    def test_has_any_meta(self):
        """Report presence of any meta."""
        trim = TrimMeta(len=10, shown=7)
        size = SizeMeta(len=10, deep=200, shallow=150)
        typ = TypeMeta(from_type=list, to_type=tuple)
        meta = Meta(trim=trim, size=size, type=typ)
        assert meta.has_any_meta is True

    def test_is_trimmed_values(self):
        """Report trimmed state via TrimMeta."""
        meta_none = Meta(trim=None)
        assert meta_none.is_trimmed is None
        meta_no_trim = Meta(trim=TrimMeta(len=5, shown=5))
        assert meta_no_trim.is_trimmed is False
        meta_trimmed = Meta(trim=TrimMeta(len=5, shown=3))
        assert meta_trimmed.is_trimmed is True

    def test_to_dict_minimal(self):
        """Return version-only when empty."""
        meta = Meta(trim=None, size=None, type=None)
        result = meta.to_dict(include_none_attrs=False, include_properties=True, sort_keys=False)
        assert result == {"version": Meta.VERSION}

    def test_to_dict_full_sorted(self):
        """Include all sections and sort keys."""
        meta = Meta(
            trim=TrimMeta(len=10, shown=8),
            size=SizeMeta(len=10, deep=1024, shallow=512),
            type=TypeMeta(from_type=list, to_type=list),
        )
        result = meta.to_dict(include_none_attrs=True, include_properties=True, sort_keys=True)
        assert list(result.keys()) == ["size", "trim", "type", "version"]
        assert result["version"] == Meta.VERSION
        assert result["trim"] == {
            "is_trimmed": True,
            "len": 10,
            "shown": 8,
            "trimmed": 2,
        }
        # SizeMeta includes all fields when include_none_attrs=True
        assert result["size"] == {"deep": 1024, "len": 10, "shallow": 512}
        # TypeMeta not converted -> to_dict omits redundant to_type
        assert result["type"] == {
            "from_type": list,
            "is_converted": False,
            "to_type": list,
        }

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            pytest.param(
                dict(trim=TrimMeta(len=3, shown=1)),
                {
                    "trim": {"is_trimmed": True, "len": 3, "shown": 1, "trimmed": 2},
                    "version": Meta.VERSION,
                },
                id="only-trim",
            ),
            pytest.param(
                dict(size=SizeMeta(len=None, deep=10, shallow=10)),
                {"size": {"deep": 10, "shallow": 10}, "version": Meta.VERSION},
                id="only-size",
            ),
            pytest.param(
                dict(type=TypeMeta(from_type=dict, to_type=None)),
                {
                    "type": {"from_type": dict, "is_converted": False},
                    "version": Meta.VERSION,
                },
                id="only-type-not-converted",
            ),
            pytest.param(
                dict(type=TypeMeta(from_type=set, to_type=frozenset)),
                {
                    "type": {
                        "from_type": set,
                        "is_converted": True,
                        "to_type": frozenset,
                    },
                    "version": Meta.VERSION,
                },
                id="only-type-converted",
            ),
        ],
    )
    def test_to_dict_partial_sections(self, kwargs, expected):
        """Include only present sections."""
        meta = Meta(**kwargs)
        result = meta.to_dict(include_none_attrs=False, include_properties=True, sort_keys=False)
        assert result == expected

    def test_typ_is_converted_property(self):
        """Compute type conversion flag."""
        t1 = TypeMeta(from_type=str, to_type=str)
        assert t1.is_converted is False
        t2 = TypeMeta(from_type=str, to_type=None)  # will default to from_type
        assert t2.is_converted is False
        t3 = TypeMeta(from_type=list, to_type=tuple)
        assert t3.is_converted is True

    def test_trimmeta_from_trimmed(self):
        """Construct TrimMeta from totals."""
        tm = TrimMeta.from_trimmed(total_len=12, trimmed_len=5)
        assert is_dataclass(tm)
        assert tm.len == 12
        assert tm.shown == 7
        assert tm.trimmed == 5
        assert tm.is_trimmed is True

    @pytest.mark.parametrize(
        "factory, kwargs, exc, msg",
        [
            pytest.param(
                SizeMeta,
                dict(deep=1, shallow=2),
                ValueError,
                r"(?i).*deep.*>=.*shallow",
                id="size-deep-lt-shallow",
            ),
            pytest.param(
                TrimMeta,
                dict(len=3, shown=5),
                ValueError,
                r"(?i).*shown.*<=.*len",
                id="trim-shown-gt-len",
            ),
        ],
    )
    def test_validation_errors(self, factory, kwargs, exc, msg):
        """Validate error conditions."""
        with pytest.raises(exc, match=msg):
            factory(**kwargs)

    def test_metamixin_to_dict_controls(self):
        """Honor MetaMixin controls."""

        class SampleMeta(MetaMixin):
            def __init__(self, a: Any = None, b: Any = 2):
                self.a = a
                self.b = b

        # Ensure TypeError when not a dataclass using MetaMixin
        with pytest.raises(TypeError, match=r"(?i) must be a dataclass"):
            SampleMeta().to_dict(include_none_attrs=False, include_properties=True, sort_keys=True)

        # Works via dataclass subclass using provided metas
        sm = SizeMeta(len=None, deep=1, shallow=None)
        d = sm.to_dict(include_none_attrs=False, include_properties=True, sort_keys=True)
        assert d == {"deep": 1}


class TestMetaFromObjects:
    def test_none_when_all_disabled(self):
        """Return None when all meta flags are disabled."""
        opts = DictifyOptions(
            meta=MetaOptions(len=False, size=False, deep_size=False, trim=False, type=False)
        )
        meta = Meta.from_objects([1, 2, 3], [1, 2, 3], opts=opts)
        assert meta is None

    def test_size_only_len(self):
        """Create size meta when only len is enabled."""
        opts = DictifyOptions(
            meta=MetaOptions(len=True, size=False, deep_size=False, trim=False, type=False)
        )
        obj = [1, 2, 3]
        meta = Meta.from_objects(obj, obj, opts=opts)
        assert isinstance(meta, Meta)
        assert isinstance(meta.size, SizeMeta)
        assert meta.trim is None
        assert meta.type is None

    def test_trim_only(self):
        """Create trim meta when trim is enabled."""
        opts = DictifyOptions(
            meta=MetaOptions(len=False, size=False, deep_size=False, trim=True, type=False)
        )
        original = list(range(10))
        processed = original[:5]
        meta = Meta.from_objects(original, processed, opts=opts)
        assert isinstance(meta, Meta)
        assert meta.size is None
        assert isinstance(meta.trim, TrimMeta)
        assert isinstance(meta.is_trimmed, (bool, type(None)))

    def test_type_only_same(self):
        """Create type meta with no conversion for same types."""
        opts = DictifyOptions(
            meta=MetaOptions(len=False, size=False, deep_size=False, trim=False, type=True)
        )
        original = {"a": 1}
        processed = {"a": 1}
        meta = Meta.from_objects(original, processed, opts=opts)
        assert isinstance(meta, Meta)
        assert meta.size is None and meta.trim is None
        assert isinstance(meta.type, TypeMeta)
        assert meta.type.from_type is dict
        assert meta.type.to_type is dict
        assert meta.type.is_converted is False

    def test_type_only_different(self):
        """Create type meta with conversion for different types."""
        opts = DictifyOptions(
            meta=MetaOptions(len=False, size=False, deep_size=False, trim=False, type=True)
        )
        original = (1, 2)
        processed = [1, 2]
        meta = Meta.from_objects(original, processed, opts=opts)
        assert isinstance(meta, Meta)
        assert isinstance(meta.type, TypeMeta)
        assert meta.type.from_type is tuple
        assert meta.type.to_type is list
        assert meta.type.is_converted is True

    def test_type_only_none_processed(self):
        """Create type meta when processed object is None."""
        opts = DictifyOptions(
            meta=MetaOptions(len=False, size=False, deep_size=False, trim=False, type=True)
        )
        original = "x"
        processed = None
        meta = Meta.from_objects(original, processed, opts=opts)
        assert isinstance(meta, Meta)
        assert meta.type.from_type is str
        assert meta.type.to_type is type(None)
        assert meta.type.is_converted is True

    def test_all_meta(self):
        """Create all meta sections when all flags are enabled."""
        opts = DictifyOptions(
            meta=MetaOptions(len=True, size=True, deep_size=False, trim=True, type=True)
        )
        original = list(range(8))
        processed = original[:5]
        meta = Meta.from_objects(original, processed, opts=opts)
        assert isinstance(meta, Meta)
        assert isinstance(meta.size, SizeMeta)
        assert isinstance(meta.trim, TrimMeta)
        assert isinstance(meta.type, TypeMeta)

    def test_to_dict_integration(self):
        """Include version and enabled meta sections in to_dict."""
        opts = DictifyOptions(meta=MetaOptions(len=True, trim=True, type=True))
        original = [1, 2, 3, 4]
        processed = original[:2]
        meta = Meta.from_objects(original, processed, opts=opts)
        d1 = meta.to_dict(include_none_attrs=False, include_properties=True, sort_keys=True)
        assert "version" in d1 and isinstance(d1["version"], int)
        assert "size" in d1 and "trim" in d1 and "type" in d1

        d2 = meta.to_dict(include_none_attrs=True, include_properties=True, sort_keys=False)
        assert "version" in d2 and "size" in d2 and "trim" in d2 and "type" in d2


class TestDictifyOptions:
    """Test suite for DictifyOptions methods."""

    # Test type_handlers functionality ------------------------------------------

    def test_add_type_handler_override(self):
        """Override existing handler and use the new one."""
        handler_id = {}

        def h1(x, o):
            handler_id["h"] = "h1"
            return "one"

        def h2(x, o):
            handler_id["h"] = "h2"
            return "two"

        opts = DictifyOptions(type_handlers={})
        opts.add_type_handler(str, h1).add_type_handler(str, h2)
        handler = opts.get_type_handler("xyz")
        assert callable(handler)
        assert handler("xyz", opts) == "two"
        assert handler_id["h"] == "h2"

    def test_get_type_handler_exact_match(self):
        """Return exact match when handler registered for the type."""
        opts = DictifyOptions(type_handlers={})
        opts.add_type_handler(bytes, lambda b, o: len(b))
        h = opts.get_type_handler(b"\x00\x01")
        assert callable(h)
        assert h(b"\x00\x01", opts) == 2

    def test_get_type_handler_inheritance_mro_nearest(self):
        """Select nearest ancestor handler via MRO."""

        class A: ...

        class B(A): ...

        class C(B): ...

        opts = DictifyOptions(type_handlers={})
        opts.add_type_handler(A, lambda x, o: "A")
        opts.add_type_handler(B, lambda x, o: "B")
        h_c = opts.get_type_handler(C())
        h_b = opts.get_type_handler(B())
        h_a = opts.get_type_handler(A())
        assert callable(h_c) and h_c(C(), opts) == "B"
        assert callable(h_b) and h_b(B(), opts) == "B"
        assert callable(h_a) and h_a(A(), opts) == "A"

    def test_get_type_handler_none_when_absent(self):
        """Return None when no handler matches."""
        opts = DictifyOptions(type_handlers={})
        assert opts.get_type_handler(3.14) is None

    def test_get_type_handler_ignores_non_type_keys(self):
        """Ignore invalid keys in type_handlers mapping."""
        opts = DictifyOptions(type_handlers={})
        # Inject an invalid key directly into the mapping
        opts.type_handlers["not-a-type"] = lambda x, o: "bad"
        opts.add_type_handler(str, lambda s, o: "ok")
        h = opts.get_type_handler("s")
        assert callable(h)
        assert h("s", opts) == "ok"

    # Tests for presence and basic behavior of default type handlers.

    @pytest.mark.parametrize(
        "instance, expected_type",
        [
            pytest.param("s", str, id="str"),
            pytest.param(b"\x00\x01", bytes, id="bytes"),
            pytest.param(bytearray(b"\x00\x01"), bytearray, id="bytearray"),
            pytest.param(memoryview(b"\x00\x01"), memoryview, id="memoryview"),
            pytest.param(date(2023, 1, 2), date, id="date"),
            pytest.param(datetime(2023, 1, 2, 3, 4, 5), datetime, id="datetime"),
            pytest.param(Decimal("1.23"), Decimal, id="decimal"),
            pytest.param(Fraction(3, 4), Fraction, id="fraction"),
            pytest.param(Path("."), Path, id="path"),
            pytest.param(range(3), range, id="range"),
            pytest.param(re.compile(r"x"), re.Pattern, id="regex-pattern"),
            pytest.param(time(1, 2, 3), time, id="time"),
            pytest.param(timedelta(seconds=5), timedelta, id="timedelta"),
            pytest.param(uuid4(), UUID, id="uuid"),
        ],
    )
    def test_registry_includes(self, instance, expected_type):
        """Assert presence of handler in registry."""
        opts = DictifyOptions()
        h = opts.get_type_handler(instance)
        assert callable(h), f"Missing default handler for {expected_type.__name__}"

    def test_registry_includes_enum(self):
        """Assert presence of handler for Enum."""

        class C(Enum):
            A = 1

        instance = C.A
        opts = DictifyOptions()
        h = opts.get_type_handler(instance)
        assert callable(h), "Missing default handler for Enum"

    def test_registry_includes_exception(self):
        """Assert presence of handler for Exception."""
        instance = ValueError("boom")
        opts = DictifyOptions(max_str_len=None, max_bytes=None)
        h = opts.get_type_handler(instance)
        assert callable(h), "Missing default handler for BaseException"

    @pytest.mark.parametrize(
        "instance",
        [
            pytest.param("hello", id="str"),
            pytest.param(b"\x00\x01\x02", id="bytes"),
            pytest.param(bytearray(b"\x00\x01\x02"), id="bytearray"),
            pytest.param(memoryview(b"\x00\x01\x02"), id="memoryview"),
            pytest.param(date(2023, 1, 2), id="date"),
            pytest.param(datetime(2023, 1, 2, 3, 4, 5), id="datetime"),
            pytest.param(Decimal("123.4500"), id="decimal"),
            pytest.param(Fraction(7, 8), id="fraction"),
            pytest.param(Path("."), id="path"),
            pytest.param(range(0, 5, 2), id="range"),
            pytest.param(re.compile(r"abc", flags=re.I), id="regex-pattern"),
            pytest.param(time(12, 34, 56), id="time"),
            pytest.param(timedelta(days=1, seconds=2), id="timedelta"),
            pytest.param(uuid4(), id="uuid"),
            pytest.param(ValueError("boom"), id="exception"),
        ],
    )
    def test_handler_smoke_returns_value(self, instance):
        """Ensure handler returns a basic value."""
        opts = DictifyOptions(max_str_len=None, max_bytes=None)
        h = opts.get_type_handler(instance)
        assert callable(h)
        out = h(instance, opts)
        basic_types = (
            str,
            int,
            float,
            bool,
            type(None),
            dict,
            list,
            bytes,
            bytearray,
            memoryview,
        )
        assert isinstance(out, basic_types)

    def test_str_truncation_respected(self):
        """Verify str truncation via options."""
        s = "abcdefgh"
        opts = DictifyOptions(max_str_len=5, max_bytes=None)
        h = opts.get_type_handler(s)
        out = h(s, opts)
        assert isinstance(out, str)
        assert out.startswith("abcde")
        assert out.endswith("...")
        assert len(out) == 8

    @pytest.mark.parametrize(
        "data, max_bytes, expected_prefix, expect_ellipsis",
        [
            pytest.param(b"abcdef", 3, b"abc", True, id="bytes-trunc"),
            pytest.param(b"ab", 5, b"ab", False, id="bytes-no-trunc"),
        ],
    )
    def test_bytes_truncation_respected(self, data, max_bytes, expected_prefix, expect_ellipsis):
        """Verify bytes truncation via options."""
        opts = DictifyOptions(max_str_len=None, max_bytes=max_bytes)
        h = opts.get_type_handler(data)
        out = h(data, opts)
        assert isinstance(out, bytes)
        assert out.startswith(expected_prefix)
        if expect_ellipsis:
            assert out.endswith(b"...")
        else:
            assert not out.endswith(b"...")

    @pytest.mark.parametrize(
        "data, max_bytes, expected_prefix, expect_ellipsis",
        [
            pytest.param(b"abcdef", 3, b"abc", True, id="bytearray-trunc"),
            pytest.param(b"ab", 5, b"ab", False, id="bytearray-no-trunc"),
        ],
    )
    def test_bytearray_truncation_respected(
        self, data, max_bytes, expected_prefix, expect_ellipsis
    ):
        """Verify bytearray truncation via options."""
        ba = bytearray(data)
        opts = DictifyOptions(max_str_len=None, max_bytes=max_bytes)
        h = opts.get_type_handler(ba)
        out = h(ba, opts)
        assert isinstance(out, bytearray)
        b = bytes(out)
        assert b.startswith(expected_prefix)
        if expect_ellipsis:
            assert b.endswith(b"...")
        else:
            assert not b.endswith(b"...")

    def test_memoryview_preview_when_truncated(self):
        """Verify memoryview preview and flags."""
        mv = memoryview(b"wxyz")
        opts = DictifyOptions(max_bytes=2)
        h = opts.get_type_handler(mv)
        out = h(mv, opts)
        assert isinstance(out, dict)
        assert out.get("type") == "memoryview"
        assert out.get("data_truncated") is True
        assert out.get("data_preview") == b"wx..."

    # Test Presets for base/debug/logging/serialization

    @pytest.mark.parametrize(
        "factory",
        [
            pytest.param(DictifyOptions.debug, id="debug"),
            pytest.param(DictifyOptions.logging, id="logging"),
            pytest.param(DictifyOptions.serial, id="serial"),
        ],
    )
    def test_presets_return_options_instances(self, factory):
        """Return DictifyOptions instances from presets."""
        opts = factory()
        assert isinstance(opts, DictifyOptions)

    # Test .merge() ------------------------------------------------------------------------------------

    def test_merge_basic_attributes(self) -> None:
        """Merge should update basic attributes when provided."""
        original = DictifyOptions(max_depth=3, include_private=False)
        merged = original.merge(max_depth=5, include_private=True)

        assert merged.max_depth == 5
        assert merged.include_private is True
        # Unchanged attributes should remain the same
        assert merged.include_none_attrs == original.include_none_attrs

    def test_merge_nested_class_name_options(self) -> None:
        """Merge should replace entire class_name nested object when provided."""

        original = DictifyOptions()
        new_class_name = ClassNameOptions(in_expand=True, in_to_dict=False, fully_qualified=True)
        merged = original.merge(class_name=new_class_name)

        assert merged.class_name is new_class_name
        assert merged.class_name.in_expand is True
        assert merged.class_name.in_to_dict is False

    def test_merge_nested_meta_options(self) -> None:
        """Merge should replace entire meta nested object when provided."""

        original = DictifyOptions()
        new_meta = MetaOptions(trim=True, type=True, len=False)
        merged = original.merge(meta=new_meta)

        assert merged.meta is new_meta
        assert merged.meta.trim is True
        assert merged.meta.type is True
        assert merged.meta.len is False

    def test_merge_convenience_flag_class_name(self) -> None:
        """Merge should set class_name flags when inject_class_name convenience flag is used."""
        original = DictifyOptions()
        # Ensure original values are different
        original.class_name.in_expand = False
        original.class_name.in_to_dict = False

        merged = original.merge(inject_class_name=True)

        assert merged.class_name.in_expand is True
        assert merged.class_name.in_to_dict is True

    def test_merge_convenience_flags_meta(self) -> None:
        """Merge should set meta flags when inject_trim_meta and inject_type_meta convenience flags are used."""
        original = DictifyOptions()
        # Ensure original values are different
        original.meta.trim = False
        original.meta.type = False

        merged = original.merge(inject_trim_meta=True, inject_type_meta=True)

        assert merged.meta.trim is True
        assert merged.meta.type is True

    def test_merge_convenience_vs_explicit_precedence(self) -> None:
        """Merge should prioritize explicit attributes over convenience flags."""
        original = DictifyOptions()
        original.class_name.in_expand = False
        original.class_name.in_to_dict = False

        # Provide both convenience flag and explicit attribute
        merged = original.merge(
            inject_class_name=True,  # Convenience flag
            class_name=original.class_name.__class__(
                in_expand=False, in_to_dict=False, fully_qualified=False
            ),
            # Explicit
        )

        # Explicit should win
        assert merged.class_name.in_expand is False
        assert merged.class_name.in_to_dict is False

    def test_merge_returns_new_instance(self) -> None:
        """Merge should return a new DictifyOptions instance."""
        original = DictifyOptions()
        merged = original.merge(max_depth=10)

        assert merged is not original
        assert isinstance(merged, DictifyOptions)

    def test_merge_with_no_changes_returns_new_instance(self) -> None:
        """Merge should return a new instance even when no changes are specified."""
        original = DictifyOptions()
        merged = original.merge()

        assert merged is not original
        assert isinstance(merged, DictifyOptions)
        # All values should be equal
        assert merged.max_depth == original.max_depth
        assert merged.include_private == original.include_private


class TestMetaMixin:
    def test_requires_dataclass(self):
        """Raise on non-dataclass instances."""
        obj = NotDataClass()
        with pytest.raises(TypeError, match=r"(?i)dataclass"):
            obj.to_dict()

    @pytest.mark.parametrize(
        "inst, include_none, expected",
        [
            pytest.param(SimpleDC(a=1, b=None), False, {"a": 1}, id="simple-exclude-none"),
            pytest.param(
                SimpleDC(a=1, b=None),
                True,
                {"a": 1, "b": None},
                id="simple-include-none",
            ),
        ],
    )
    def test_none_filtering(self, inst: MetaMixin, include_none: bool, expected: dict[str, Any]):
        """Control inclusion of None values."""
        assert inst.to_dict(include_none_attrs=include_none) == expected

    def test_include_properties(self):
        """Include public properties."""
        inst = WithProps(x=2, y=3)
        result = inst.to_dict(include_properties=True)
        assert result["x"] == 2
        assert result["y"] == 3
        assert result["sum"] == 5
        assert "_hidden" not in result

    def test_exclude_properties(self):
        """Exclude properties when requested."""
        inst = WithProps(x=2, y=3)
        result = inst.to_dict(include_properties=False)
        assert result == {"x": 2, "y": 3}

    @pytest.mark.parametrize(
        "sort_keys, expected_keys",
        [
            pytest.param(False, ["x", "y", "sum"], id="unsorted"),
            pytest.param(True, ["sum", "x", "y"], id="sorted"),
        ],
    )
    def test_sort_keys(self, sort_keys: bool, expected_keys: list[str]):
        """Sort result keys when requested."""
        inst = WithProps(x=1, y=2)
        result = inst.to_dict(sort_keys=sort_keys)
        assert list(result.keys()) == expected_keys

    def test_property_inclusion_with_none_filtering(self):
        """Filter None values including property results."""
        inst = WithProps(x=5, y=None)
        result = inst.to_dict(include_none_attrs=False, include_properties=True)
        # y should be dropped, sum computed as 5 (still included)
        assert result == {"x": 5, "sum": 5}

    def test_property_computation_errors_surface(self):
        """Surface property access errors."""

        @dataclass
        class BadProp(MetaMixin):
            v: int

            @property
            def boom(self) -> int:
                raise ValueError("boom!")

        inst = BadProp(v=1)
        with pytest.raises(ValueError, match=r"(?i)boom"):
            inst.to_dict()

    def test_property_name_filtering(self):
        """Ignore private-like properties."""

        @dataclass
        class PrivateProps(MetaMixin):
            p: int = 1

            @property
            def _private(self) -> int:
                return 7

            @property
            def public(self) -> int:
                return 3

        inst = PrivateProps()
        result = inst.to_dict()
        assert "public" in result and result["public"] == 3
        assert "_private" not in result

    def test_merged_property_and_field_keys(self):
        """Merge dataclass fields with properties."""

        @dataclass
        class Overlap(MetaMixin):
            val: int = 2

            @property
            def val_prop(self) -> int:
                return self.val * 2

        inst = Overlap()
        result = inst.to_dict()
        assert result["val"] == 2
        assert result["val_prop"] == 4


class TestMetaOptions:
    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            pytest.param({}, False, id="default_no_sizes"),
            pytest.param({"len": True}, True, id="len_enabled"),
            pytest.param({"size": True}, True, id="size_enabled"),
            pytest.param({"deep_size": True}, True, id="deep_size_enabled"),
            pytest.param(
                {"len": False, "size": False, "deep_size": False},
                False,
                id="all_sizes_disabled",
            ),
        ],
    )
    def test_sizes_enabled_property(self, kwargs, expected):
        """Verify sizes_enabled property correctly reports size metadata status."""
        meta_options = MetaOptions(**kwargs)
        assert meta_options.sizes_enabled == expected

    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            pytest.param({"trim": False}, False, id="trim_disabled"),
            pytest.param({"type": True}, True, id="type_enabled"),
            pytest.param({"trim": False, "type": False}, False, id="all_metadata_disabled"),
        ],
    )
    def test_any_enabled_property(self, kwargs, expected):
        """Verify any_enabled property correctly reports metadata injection status."""
        meta_options = MetaOptions(**kwargs)
        assert meta_options.any_enabled == expected

    def test_merge_is_safe_for_default_values(self):
        """Ensure merge method preserves default values when no arguments are provided."""
        original = MetaOptions()
        merged = original.merge()

        for field in fields(MetaOptions):
            assert getattr(merged, field.name) == getattr(original, field.name)

    @pytest.mark.parametrize(
        "merge_kwargs,expected",
        [
            pytest.param({"len": True}, True, id="merge_len_true"),
            pytest.param({"size": False}, False, id="merge_size_false"),
            pytest.param({"key": "custom_key"}, "custom_key", id="merge_custom_key"),
        ],
    )
    def test_merge_specific_attributes(self, merge_kwargs, expected):
        """Verify merge method correctly updates specific attributes."""
        original = MetaOptions()
        merged = original.merge(**merge_kwargs)

        for key, value in merge_kwargs.items():
            assert getattr(merged, key) == value

    def test_merge_preserves_original(self):
        """Ensure merge creates a new instance without modifying the original."""
        original = MetaOptions(len=False, size=True)
        merged = original.merge(len=True)

        assert merged is not original
        assert merged.len is True
        assert original.len is False

    def test_merge_multiple_attributes(self):
        """Verify merge can update multiple attributes simultaneously."""
        original = MetaOptions()
        merged = original.merge(len=True, size=True, key="custom_meta")

        assert merged.len is True
        assert merged.size is True
        assert merged.key == "custom_meta"


class TestSizeMeta:
    def test_all_none_rejected(self):
        """Reject construction with all fields None."""
        with pytest.raises(ValueError, match=r"(?i)at least one non-None"):
            SizeMeta(len=None, deep=None, shallow=None)

    def test_deep_not_less_than_shallow(self):
        """Enforce deep >= shallow relation."""
        with pytest.raises(ValueError, match=r"(?i)deep.*>=.*shallow"):
            SizeMeta(len=0, deep=9, shallow=10)

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param(dict(len=0, deep=0, shallow=0), id="all-zero"),
            pytest.param(dict(len=5, deep=10, shallow=10), id="equal-deep-shallow"),
            pytest.param(dict(len=None, deep=20, shallow=10), id="deep-greater"),
            pytest.param(dict(len=3, deep=None, shallow=None), id="only-len"),
            pytest.param(dict(len=None, deep=4, shallow=None), id="only-deep"),
            pytest.param(dict(len=None, deep=None, shallow=7), id="only-shallow"),
        ],
    )
    def test_valid_configurations(self, kwargs):
        """Accept valid combinations."""
        sm = SizeMeta(**kwargs)
        for k, v in kwargs.items():
            assert getattr(sm, k) == v

    def test_to_dict_integration(self):
        """Convert to dict via mixin."""
        sm = SizeMeta(len=7, deep=100, shallow=60)
        d = sm.to_dict(sort_keys=True, include_none_attrs=True, include_properties=False)
        assert list(d.keys()) == ["deep", "len", "shallow"]
        assert d == {"len": 7, "deep": 100, "shallow": 60}

    # -------- from_object tests --------

    def test_from_object_returns_none_when_no_flags(self):
        """Return None when no include_* flags are set."""
        obj = [1, 2, 3]
        assert (
            SizeMeta.from_object(obj, include_len=False, include_deep=False, include_shallow=False)
            is None
        )

    def test_from_object_len_only_for_sized_objects(self):
        """Include length only for sized objects."""
        obj = [1, 2, 3]
        sm = SizeMeta.from_object(obj, include_len=True, include_deep=False, include_shallow=False)
        assert sm is not None
        assert sm.len == 3
        assert sm.deep is None
        assert sm.shallow is None

    def test_from_object_len_skipped_for_unsized(self):
        """Skip len for unsized objects."""

        class Unsized:
            pass

        obj = Unsized()
        sm = SizeMeta.from_object(obj, include_len=True, include_deep=False, include_shallow=False)
        assert sm is None  # no other fields requested and len not available

    def test_from_object_shallow_only(self):
        """Include shallow size only."""
        obj = {"a": 1, "b": 2}
        sm = SizeMeta.from_object(obj, include_len=False, include_deep=False, include_shallow=True)
        assert sm is not None
        assert sm.len is None
        assert sm.deep is None
        assert isinstance(sm.shallow, int)
        assert sm.shallow == sys.getsizeof(obj)

    def test_from_object_multiple_fields(self):
        """Include requested fields and allow None for others."""
        obj = "abcdef"
        sm = SizeMeta.from_object(obj, include_len=True, include_deep=False, include_shallow=True)
        assert sm is not None
        assert sm.len == len(obj)
        assert sm.deep is None
        assert isinstance(sm.shallow, int)


class TestTrimMeta:
    def test_nones(self):
        """Require shown; allow unknown len."""
        with pytest.raises(ValueError, match=r"(?i)requires 'shown'"):
            TrimMeta(None, None)
        with pytest.raises(ValueError, match=r"(?i)requires 'shown'"):
            TrimMeta(len=5, shown=None)

        tm = TrimMeta(len=None, shown=5)
        assert tm.len is None
        assert tm.shown == 5
        assert tm.trimmed is None
        assert tm.is_trimmed is None

    def test_shown_not_exceed_len(self):
        """Enforce shown <= len."""
        with pytest.raises(ValueError, match=r"(?i)shown.*<=.*len"):
            TrimMeta(len=3, shown=4)

    @pytest.mark.parametrize(
        "total_len, trimmed_len, expected_shown",
        [
            pytest.param(10, 0, 10, id="none-trimmed"),
            pytest.param(10, 3, 7, id="some-trimmed"),
            pytest.param(5, 10, 0, id="over-trimmed-clamped"),
        ],
    )
    def test_from_trimmed(self, total_len: int, trimmed_len: int, expected_shown: int):
        """Construct from total and trimmed."""
        tm = TrimMeta.from_trimmed(total_len, trimmed_len)
        assert tm.len == total_len
        assert tm.shown == expected_shown
        assert tm.trimmed == total_len - expected_shown

    def test_trimmed_property_and_is_trimmed(self):
        """Compute trimmed and is_trimmed."""
        tm = TrimMeta(len=8, shown=5)
        assert tm.trimmed == 3
        assert tm.is_trimmed is True

    def test_to_dict_integration(self):
        """Convert to dict via mixin."""
        tm = TrimMeta(len=9, shown=4)
        d = tm.to_dict(sort_keys=True, include_properties=True)
        assert list(d.keys()) == ["is_trimmed", "len", "shown", "trimmed"]
        assert d["len"] == 9 and d["shown"] == 4 and d["trimmed"] == 5 and d["is_trimmed"] is True

    def test_from_objects_success(self):
        class C:
            def __init__(self, n):
                self._n = n

            def __len__(self):
                return self._n

        tm = TrimMeta.from_objects(C(7), C(3))
        assert tm is not None
        assert tm.len == 7
        assert tm.shown == 3
        assert tm.trimmed == 4
        assert tm.is_trimmed is True

    def test_from_objects_when_lengths_unknown(self):
        """Handle unknown lengths and generators in from_objects."""

        def gen(n: int):
            for i in range(n):
                yield i

        # Original unknown (generator), processed known (list)
        tm = TrimMeta.from_objects(gen(5), [0, 1, 2])
        assert tm is not None
        assert tm.len is None
        assert tm.shown == 3
        assert tm.trimmed is None
        assert tm.is_trimmed is None

        # Processed unknown (generator) -> cannot create metadata
        tm2 = TrimMeta.from_objects([0, 1, 2, 3], gen(2))
        assert tm2 is None

    def test_from_objects_equal_lengths_not_trimmed(self):
        tm = TrimMeta.from_objects([1, 2, 3], (1, 2, 3))
        assert tm is not None
        assert tm.len == 3
        assert tm.shown == 3
        assert tm.trimmed == 0
        assert tm.is_trimmed is False


class TestTypeMeta:
    def test_nones(self):
        """Create with Nones and succeed."""
        tm = TypeMeta(from_type=None, to_type=None)
        assert tm.from_type is None
        assert tm.to_type is None
        assert tm.is_converted is False

    @pytest.mark.parametrize(
        "from_t, to_t, expected_flag",
        [
            pytest.param(int, int, False, id="same-types"),
            pytest.param(int, float, True, id="different-types"),
            pytest.param(
                None, int, False, id="from-none-to-type"
            ),  # Changed: can't determine conversion
            pytest.param(
                int, None, False, id="to-none-no-conversion"
            ),  # Changed: can't determine conversion
            pytest.param(None, None, False, id="both-none"),
        ],
    )
    def test_is_converted_logic(self, from_t, to_t, expected_flag):
        """Compute is_converted flag correctly."""
        tm = TypeMeta(from_type=from_t, to_type=to_t)
        assert tm.is_converted is expected_flag

    def test_to_type_no_longer_defaults(self):
        """to_type no longer defaults to from_type when missing."""
        tm = TypeMeta(from_type=int, to_type=None)
        assert tm.from_type is int
        assert tm.to_type is None  # No longer defaults
        assert tm.is_converted is False  # Can't determine conversion

    def test_to_dict_excludes_redundant_to_type(self):
        """Exclude to_type when not converted."""
        tm = TypeMeta(from_type=int, to_type=int)  # Changed: explicit same type
        d = tm.to_dict(
            include_none_attrs=False, include_properties=True, sort_keys=True
        )  # Changed: False instead of True
        assert "from_type" in d
        assert "is_converted" in d and d["is_converted"] is False
        assert "to_type" not in d

    def test_to_dict_includes_to_type_when_converted(self):
        """Include to_type when converted."""
        tm = TypeMeta(from_type=int, to_type=float)
        d = tm.to_dict(include_none_attrs=False, include_properties=True, sort_keys=True)
        assert list(d.keys()) == ["from_type", "is_converted", "to_type"]
        assert d["from_type"] is int and d["to_type"] is float and d["is_converted"] is True

    @pytest.mark.parametrize(
        "include_none, expected_keys",
        [
            pytest.param(False, ["is_converted"], id="exclude-none"),
            pytest.param(True, ["from_type", "is_converted", "to_type"], id="include-none"),
            # Changed: to_type now included
        ],
    )
    def test_include_none_behavior(self, include_none, expected_keys):
        """Control inclusion of None values in dict."""
        tm = TypeMeta()  # both None -> not converted; to_type no longer removed automatically
        d = tm.to_dict(include_none_attrs=include_none, include_properties=True, sort_keys=True)
        assert list(d.keys()) == expected_keys

    def test_disable_properties_path(self):
        """Honor include_properties flag path."""
        tm = TypeMeta(from_type=bytes, to_type=str)
        d = tm.to_dict(include_none_attrs=False, include_properties=False, sort_keys=True)
        # Properties are excluded, so 'is_converted' is not present here
        assert list(d.keys()) == ["from_type", "to_type"]

    def test_repr_types_identity(self):
        """Maintain identity of type objects."""
        tm = TypeMeta(from_type=dict, to_type=dict)
        assert tm.from_type is dict
        assert tm.to_type is dict
        assert tm.is_converted is False
        d = tm.to_dict(include_none_attrs=False, include_properties=True, sort_keys=False)
        assert d["from_type"] is dict

    # Tests for from_object/objects methods

    @pytest.mark.parametrize(
        ("input_obj", "expected_type"),
        [
            pytest.param(42, int, id="integer"),
            pytest.param("hello", str, id="string"),
            pytest.param([1, 2, 3], list, id="list"),
            pytest.param({"a": 1}, dict, id="dict"),
            pytest.param((1, 2), tuple, id="tuple"),
            pytest.param({1, 2, 3}, set, id="set"),
            pytest.param(True, bool, id="boolean"),
            pytest.param(3.14, float, id="float"),
            pytest.param(None, type(None), id="None"),
        ],
    )
    def test_handles_common_types(self, input_obj, expected_type):
        """Create instance with correct from_type for various object types."""
        result = TypeMeta.from_object(input_obj)
        assert isinstance(result, TypeMeta)
        assert result.from_type is expected_type
        assert result.to_type is None

    def test_handles_custom_class_instance(self):
        """Handle custom class instances."""

        class CustomClass:
            pass

        obj = CustomClass()
        result = TypeMeta.from_object(obj)
        assert isinstance(result, TypeMeta)
        assert result.from_type is CustomClass
        assert result.to_type is None

    def test_from_objects_success(self):
        """Create TypeMeta from two objects with different types."""
        tm = TypeMeta.from_objects(42, "hello")
        assert tm.from_type is int
        assert tm.to_type is str
        assert tm.is_converted is True

    def test_from_objects_same_types(self):
        """Create TypeMeta from objects with same type."""
        tm = TypeMeta.from_objects([1, 2], [3, 4])
        assert tm.from_type is list
        assert tm.to_type is list
        assert tm.is_converted is False

    def test_from_objects_with_none_processed(self):
        """Create TypeMeta when processed_object is None."""
        tm = TypeMeta.from_objects("test", None)
        assert tm.from_type is str
        assert tm.to_type is type(None)
        assert tm.is_converted is True

    def test_from_objects_both_none_objects(self):
        """Create TypeMeta when both objects are None."""
        tm = TypeMeta.from_objects(None, None)
        assert tm.from_type is type(None)
        assert tm.to_type is type(None)
        assert tm.is_converted is False

    def test_from_objects_with_none_original(self):
        """Create TypeMeta when original object is None."""
        tm = TypeMeta.from_objects(None, "hello")
        assert tm.from_type is type(None)
        assert tm.to_type is str
        assert tm.is_converted is True


class TestInjectMeta:
    """Test inject_meta() functionality."""

    def test_inject_meta_returns_obj_when_meta_is_none(self):
        """Return original object when meta is None."""
        from c108.dictify import inject_meta

        opts = DictifyOptions()
        obj = {"key": "value"}
        result = inject_meta(obj, None, opts)
        assert result is obj

    def test_inject_meta_into_dict(self):
        """Inject metadata into dict under meta key."""
        from c108.dictify import inject_meta

        opts = DictifyOptions()
        obj = {"key": "value"}
        meta = Meta(size=SizeMeta(len=5))

        result = inject_meta(obj, meta, opts)

        assert isinstance(result, dict)
        assert "key" in result
        assert opts.meta.key in result
        assert "size" in result[opts.meta.key]

    def test_inject_meta_into_mapping(self):
        """Inject metadata into abc.Mapping by converting to dict."""
        from c108.dictify import inject_meta
        from collections import OrderedDict

        opts = DictifyOptions()
        obj = OrderedDict([("a", 1), ("b", 2)])
        meta = Meta(size=SizeMeta(len=2))

        result = inject_meta(obj, meta, opts)

        assert isinstance(result, dict)
        assert "a" in result
        assert opts.meta.key in result

    def test_inject_meta_into_list(self):
        """Inject metadata into list as last element."""
        from c108.dictify import inject_meta

        opts = DictifyOptions()
        obj = [1, 2, 3]
        meta = Meta(size=SizeMeta(len=3))

        result = inject_meta(obj, meta, opts)

        assert isinstance(result, list)
        assert len(result) == 4
        assert result[:3] == [1, 2, 3]
        assert isinstance(result[3], dict)
        assert opts.meta.key in result[3]

    def test_inject_meta_into_tuple_returns_as_is(self):
        """Inject metadata into tuple returns identity."""
        from c108.dictify import inject_meta

        opts = DictifyOptions()
        obj = (1, 2, 3)
        meta = Meta(size=SizeMeta(len=3))

        result = inject_meta(obj, meta, opts)

        assert result == obj

    def test_inject_meta_into_set_returns_as_is(self):
        """Inject metadata into set returns identity."""
        from c108.dictify import inject_meta

        opts = DictifyOptions()
        obj = {1, 2, 3}
        meta = Meta(size=SizeMeta(len=3))

        result = inject_meta(obj, meta, opts)

        assert result == obj

    def test_inject_meta_into_unsupported_type_returns_as_is(self):
        """Return object as-is for unsupported types without wrapping."""
        from c108.dictify import inject_meta

        opts = DictifyOptions()

        # Test with various unsupported types
        unsupported = [42, "string", 3.14, True, None, object()]

        for obj in unsupported:
            meta = Meta(size=SizeMeta(len=1))
            result = inject_meta(obj, meta, opts)
            assert result is obj, f"Failed for type {type(obj)}"

    def test_inject_meta_respects_custom_meta_key(self):
        """Use custom meta key for injection."""
        from c108.dictify import inject_meta

        opts = DictifyOptions()
        opts.meta.key = "__custom_meta__"

        obj = {"data": "value"}
        meta = Meta(size=SizeMeta(len=1))

        result = inject_meta(obj, meta, opts)

        assert "__custom_meta__" in result
        assert "__dictify__" not in result

    def test_inject_meta_with_different_meta_types(self):
        """Inject different metadata types correctly."""
        from c108.dictify import inject_meta

        opts = DictifyOptions()

        obj = {"key": "value"}
        meta = Meta(
            size=SizeMeta(len=10, shallow=100),
            trim=TrimMeta(len=100, shown=10),
            type=TypeMeta(from_type=list, to_type=dict),
        )

        result = inject_meta(obj, meta, opts)

        meta_content = result[opts.meta.key]
        assert "size" in meta_content
        assert "trim" in meta_content
        assert "type" in meta_content


# Main Functionality Tests ---------------------------------------------------------------------------------------------


class TestDictifyCore:
    def test_basic_object_conversion(self):
        """Convert simple object to dictionary."""

        class Person:
            def __init__(self, name, age):
                self.name = name
                self.age = age

        person = Person("Alice", 7)
        result = dictify_core(person)
        assert result == {"name": "Alice", "age": 7}
        assert str(result) == "{'name': 'Alice', 'age': 7}"

    @pytest.mark.parametrize(
        "value",
        [42, 3.14, True, 2 + 3j, None],
        ids=["int", "float", "bool", "complex", "none"],
    )
    def test_never_filtered_as_is(self, value):
        """Return never-filtered builtins as is."""
        assert dictify_core(value) is value

    def test_hook_mode_dict_calls_to_dict_and_injects_class(self):
        """Inject class name when to_dict returns mapping."""

        class WithToDict:
            def to_dict(self):
                return {"x": 1}

        opts = DictifyOptions(hook_mode=HookMode.DICT).merge(inject_class_name=True)
        res = dictify_core(WithToDict(), opts=opts)
        assert res["x"] == 1
        assert res["__class_name__"] == "WithToDict"

    def test_hook_mode_strict_missing_to_dict_raises(self):
        """Raise when DICT_STRICT and no to_dict."""
        opts = DictifyOptions(hook_mode=HookMode.DICT_STRICT)
        with pytest.raises(TypeError, match=r"(?i)must implement to_dict"):
            dictify_core(object(), opts=opts)

    def test_to_dict_non_mapping_raises(self):
        """Raise when to_dict returns non-mapping."""

        class BadToDict:
            def to_dict(self):
                return [("k", "v")]

        opts = DictifyOptions(hook_mode=HookMode.DICT)
        with pytest.raises(TypeError, match=r"(?i)must return a Mapping"):
            dictify_core(BadToDict(), opts=opts)

    def test_max_depth_negative_uses_raw(self):
        """Return handlers.raw when max_depth is negative."""
        marker = object()
        opts = DictifyOptions(max_depth=-1, handlers=Handlers(raw=lambda x, opts: marker))
        res = dictify_core(object(), opts=opts)
        assert res is marker

    def test_sequence_without_len_trimming(self):
        """Apply handlers.terminal for Sequence lacking __len__."""

        class MySeqNoLen:
            def __iter__(self):
                yield from (1, 2, 3, 4, 5)

            # no __len__

        # Virtually register as Sequence while lacking __len__
        abc.Sequence.register(MySeqNoLen)

        expected = [1, 2, 3]
        opts = DictifyOptions(max_items=3)
        res = dictify_core(MySeqNoLen(), opts=opts)
        assert res == expected

    @pytest.mark.parametrize(
        ("include_none_items", "expected_keys"),
        [(False, {"a"}), (True, {"a", "b"})],
        ids=["drop-none", "keep-none"],
    )
    def test_mapping_include_none_items(self, include_none_items, expected_keys):
        """Respect include_none_items for plain mappings."""
        opts = DictifyOptions().merge(include_none_items=include_none_items)
        res = dictify_core({"a": 1, "b": None}, opts=opts)
        assert set(res.keys()) == expected_keys

    def test_object_expansion_toplevel_filters_attrs(self):
        """Expand object attributes and respect include_none_attrs."""

        class Obj:
            def __init__(self):
                self.a = 1
                self.b = None

        opts = DictifyOptions(max_depth=1, include_none_attrs=False).merge(inject_class_name=False)
        res = dictify_core(Obj(), opts=opts)
        assert res == {"a": 1}

    def test_depth_zero_uses_handlers_terminal_on_user_object(self):
        """Use handlers.terminal when max_depth is zero for user object."""

        class Foo:
            pass

        marker = ("processed", "Foo")
        opts = DictifyOptions(max_depth=0, handlers=Handlers(terminal=lambda x, opts: marker))
        res = dictify_core(Foo(), opts=opts)
        assert res == marker

    def test_recursive_sequence_respects_depth(self):
        """Process nested sequences with proper depth control."""

        class Foo:
            def __init__(self):
                self.value = 42

        data = [[Foo()]]
        opts = DictifyOptions().merge(
            max_depth=3,  # Need depth=3!
            inject_class_name=False,
        )
        res = dictify_core(data, opts=opts)
        assert res == [[{"value": 42}]]

    def test_object_tree_depth_control(self):
        """Expand object to dict but keep nested objects as raw values at depth 1."""

        class Node:
            def __init__(self, name, child=None):
                self.name = name
                self.child = child

        leaf = Node(name="leaf")
        root = Node(name="root", child=leaf)

        # Use max_depth=1 so only the root is expanded; nested objects remain raw.
        opts = DictifyOptions(max_depth=1)
        # Do not pass terminal(); identity fallback keeps terminal objects as-is.
        res = dictify_core(root, opts=opts)

        assert isinstance(res, dict)
        assert res["name"] == "root"
        assert res["child"] is leaf  # Raw object, not processed

    def test_invalid_hook_mode_raises_value_error(self):
        """Raise ValueError on invalid hook_mode."""
        bad_opts = DictifyOptions(hook_mode="unexpected")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=r"(?i)unknown hook_mode value"):
            dictify_core(object(), opts=bad_opts)

    def test_property_exception_is_skipped(self):
        """Skip properties that raise exceptions when include_properties is on."""

        class WithBadProp:
            def __init__(self):
                self.ok = 1

            @property
            def bad(self):
                raise RuntimeError("boom")

        opts = DictifyOptions(max_depth=1, include_properties=True)
        res = dictify_core(WithBadProp(), opts=opts)
        assert res == {"ok": 1}

    @pytest.mark.parametrize("fqn", [False, True], ids=["short-name", "fully-qualified"])
    def test_include_class_name_attrs(self, fqn):
        """Include class name during normal attribute scanning with optional FQN."""

        class Obj:
            def __init__(self):
                self.a = 1

        opts = DictifyOptions(
            max_depth=1,
            class_name=ClassNameOptions(in_expand=True, fully_qualified=fqn),
        )
        res = dictify_core(Obj(), opts=opts)

        expected_class = Obj.__name__ if not fqn else f"{Obj.__module__}.{Obj.__name__}"
        assert res["a"] == 1
        assert res["__class_name__"] == expected_class

    def test_include_class_name_attrs_disabled(self):
        """Do not include class name when option is disabled."""

        class Obj:
            def __init__(self):
                self.a = 1

        opts = DictifyOptions(max_depth=1).merge(inject_class_name=False)
        res = dictify_core(Obj(), opts=opts)
        assert res == {"a": 1}

    def test_to_dict_injects_class_name_fqn(self):
        """Inject class name into to_dict result with fully qualified name."""

        class WithToDict:
            def to_dict(self):
                return {"x": 1}

        opts = DictifyOptions(
            hook_mode=HookMode.DICT,
            class_name=ClassNameOptions(in_to_dict=True, fully_qualified=True),
        )
        res = dictify_core(WithToDict(), opts=opts)

        expected_class = f"{WithToDict.__module__}.{WithToDict.__name__}"
        assert res["x"] == 1
        assert res["__class_name__"] == expected_class

    def test_to_dict_no_injection_when_disabled(self):
        """Do not inject class name when inject_class_name is False for to_dict."""

        class WithToDict:
            def to_dict(self):
                return {"x": 1}

        opts = DictifyOptions(hook_mode=HookMode.DICT)
        res = dictify_core(WithToDict(), opts=opts)
        assert res == {"x": 1}

    def test_depth_partial_object_expansion(self):
        """Expand two levels of object tree and keep deeper nodes raw."""

        class Node:
            def __init__(self, name, child=None):
                self.name = name
                self.child = child

        leaf = Node("leaf")
        mid = Node("mid", child=leaf)
        root = Node("root", child=mid)

        # Depth=2: root expanded (depth->1), child expanded (depth->0), grandchild stays raw.
        opts = DictifyOptions(
            max_depth=2,
        )
        res = dictify_core(root, opts=opts)

        assert res["name"] == "root"
        assert res["child"]["name"] == "mid"
        assert res["child"]["child"] is leaf  # Raw at terminal depth

    def test_handlers_terminal_output_not_modified(self):
        """Do not inject class name into terminal() output."""

        class Foo:
            pass

        # At depth=0, terminal() is used and its output must not be modified.
        opts = DictifyOptions(
            max_depth=0,
            handlers=Handlers(terminal=lambda x, opts: {"marker": "terminal"}),
        )
        res = dictify_core(Foo(), opts=opts)
        assert res == {"marker": "terminal"}

    def test_type_handlers_str_truncation_and_passthrough(self):
        opts = DictifyOptions(max_str_len=5)
        assert dictify_core("abcdef", opts=opts) == "abcde..."
        # unchanged when within limit
        assert dictify_core("abc", opts=opts) == "abc"

    def test_type_handlers_bytes_truncation(self):
        data = b"a" * 10
        opts = DictifyOptions(max_bytes=6)
        assert dictify_core(data, opts=opts) == (b"a" * 6) + b"..."

    def test_type_handlers_bytearray_truncation(self):
        data = bytearray(b"a" * 10)
        opts = DictifyOptions(max_bytes=6)
        out = dictify_core(data, opts=opts)
        assert isinstance(out, bytearray)
        assert out == bytearray(b"a" * 6 + b"...")

    def test_type_handlers_memoryview_default_handler(self):
        mv = memoryview(b"abcdef")
        opts = DictifyOptions(max_bytes=3)
        res = dictify_core(mv, opts=opts)
        assert isinstance(res, dict)
        assert res["type"] == "memoryview"
        assert res["nbytes"] == 6
        assert res["data_preview"] == b"abc..."
        assert res["data_truncated"] is True

    def test_itemsview_to_dict_and_sorting(self):
        d = {"b": 2, "a": 1}
        opts = DictifyOptions(max_depth=1, sort_keys=True)
        res = dictify_core(d.items(), opts=opts)
        # ItemsView becomes dict with sorted keys at top level expansion
        assert res == {"a": 1, "b": 2}

    def test_namedtuple_converts_to_dict(self):
        from collections import namedtuple

        NT = namedtuple("NT", "x y")
        obj = NT(1, 2)
        opts = DictifyOptions(max_depth=1)
        res = dictify_core(obj, opts=opts)
        assert res == {"x": 1, "y": 2}

    def test_sets_become_lists(self):
        s = {3, 1, 2}
        opts = DictifyOptions(max_depth=1, sort_iterables=True)
        res = dictify_core(s, opts=opts)
        assert isinstance(res, list)
        assert res == [1, 2, 3]

    def test_mapping_like_items_with_hash_collision_preserve_as_list(self):
        class WeirdItems:
            def __len__(self):
                return 3

            def __iter__(self):
                return iter(())

            def items(self):
                # two distinct keys considered equal when hashed to same value by dict – simulate by equal keys
                return [("k", 1), ("k", 2), ("k", 3)]

        opts = DictifyOptions(max_depth=1)
        res = dictify_core(WeirdItems(), opts=opts)
        # collisions -> kept as list of pairs
        assert isinstance(res, list)
        assert res == [("k", 1), ("k", 2), ("k", 3)]

    def test_unsized_items_iterable_trimmed_without_sort(self):
        class GenItems:
            def items(self):
                def gen():
                    for i in range(10):
                        yield (f"k{i}", i)

                return gen()

        opts = DictifyOptions(max_depth=3, max_items=3)
        res = dictify_core(GenItems(), opts=opts)

        # unsized with items() -> collected up to max_items, converted to dict
        assert isinstance(res, dict)
        assert len(res) == 3
        assert set(res.keys()) == {"k0", "k1", "k2"}

    def test_iterable_list_trimming_and_meta_injection(self):
        data = list(range(10))
        opts = DictifyOptions(max_depth=1, max_items=3, meta=MetaOptions(trim=True, in_expand=True))
        res = dictify_core(data, opts=opts)
        # list trimmed to 3 with meta as last element
        assert res[:3] == [0, 1, 2]
        assert isinstance(res[-1], dict)
        meta_dict = res[-1].get(opts.meta.key)
        assert isinstance(meta_dict, dict)
        assert meta_dict["trim"]["shown"] == 3
        # len may be known
        assert meta_dict["trim"]["len"] in (10,)

    def test_mapping_trimming_and_meta_injection(self):
        data = {f"k{i}": i for i in range(10)}
        opts = DictifyOptions(
            max_depth=1,
            max_items=4,
            sort_keys=True,
            meta=MetaOptions(trim=True, in_expand=True),
        )
        res = dictify_core(data, opts=opts)
        assert isinstance(res, dict)
        # only first 4 sorted keys remain
        assert set(res.keys()).issuperset({"k0", "k1", "k2", "k3"})
        meta_dict = res[opts.meta.key]
        assert isinstance(meta_dict, dict)
        assert meta_dict["trim"]["shown"] == 4
        assert meta_dict["trim"]["len"] == 10

    def test_size_meta_injection_len_size_deep(self, monkeypatch):
        obj = [1, 2, 3, 4]
        # ensure deep_sizeof is called safely (already handled), keep defaults
        opts = DictifyOptions(
            max_depth=1,
            meta=MetaOptions(in_expand=True, len=True, size=True, deep_size=True),
        )
        res = dictify_core(obj, opts=opts)
        meta = res[-1][opts.meta.key]
        assert "size" in meta
        size = meta["size"]
        # len should be present and correct
        assert size.get("len") == 4

    def test_type_meta_injection_on_conversion(self):
        class Foo:
            pass

        data = [Foo()]
        opts = DictifyOptions(max_depth=1, meta=MetaOptions(in_expand=True, type=True))
        res = dictify_core(data, opts=opts)
        meta = res[-1][opts.meta.key]
        assert "type" in meta
        t = meta["type"]
        assert t["from_type"] == Foo
        # to_type should be list for outer or dict for inner expansion; type meta here reflects list -> list, but still present
        assert "to_type" in t

    def test_iterable_to_mutable_on_list_respects_sort_and_trim(self):
        data = [3, 1, 2, 5, 4]
        opts = DictifyOptions(max_depth=1, sort_iterables=True, max_items=3)
        res = dictify_core(data, opts=opts)
        assert res == [1, 2, 3]

    def test_object_without_iterable_goes_to_shallow_to_mutable(self):
        class A:
            def __init__(self):
                self.x = 1
                self.y = 2

        opts = DictifyOptions(max_depth=1)
        res = dictify_core(A(), opts=opts)
        assert res == {"x": 1, "y": 2}

    def test_shallow_to_mutable_respects_include_private_and_properties(self):
        class A:
            def __init__(self):
                self.x = 1
                self._y = 2

            @property
            def p(self):
                return 3

        opts = DictifyOptions(max_depth=1, include_private=True, include_properties=True)
        res = dictify_core(A(), opts=opts)
        assert res["x"] == 1 and res["_y"] == 2 and res["p"] == 3

    def test_shallow_to_mutable_skips_dunder_and_callables(self):
        class A:
            def __init__(self):
                self.x = 1

            def method(self):
                return 2

            __hidden__ = "v"

        opts = DictifyOptions(max_depth=1)
        res = dictify_core(A(), opts=opts)
        assert "method" not in res
        assert "__hidden__" not in res

    def test_to_dict_meta_injection_enabled(self):
        class WithToDict:
            def to_dict(self):
                return {"x": 1}

        opts = DictifyOptions(
            hook_mode=HookMode.DICT,
            meta=MetaOptions(in_to_dict=True, trim=True, type=True, len=True),
        )
        res = dictify_core(WithToDict(), opts=opts)
        assert isinstance(res, dict)
        assert "__dictify__" in res

    def test_hook_mode_none_skips_to_dict(self):
        class WithToDict:
            def __init__(self):
                self.a = 1

            def to_dict(self):
                return {"x": 2}

        opts = DictifyOptions(hook_mode=HookMode.NONE, max_depth=1)
        res = dictify_core(WithToDict(), opts=opts)
        assert res == {"a": 1}

    def test_handlers_raw_chain_fallbacks(self):
        class WithToDict:
            def to_dict(self):
                return {"x": 1}

        opts = DictifyOptions(max_depth=-1)
        res = dictify_core(WithToDict(), opts=opts)
        # in raw mode, falls back to to_dict() if no handlers.raw provided
        assert res == {"x": 1}

    def test_handlers_terminal_chain_precedence(self):
        class Foo:
            pass

        calls = []

        def term(o, _):
            calls.append("terminal")
            return {"t": 1}

        opts = DictifyOptions(max_depth=0, handlers=Handlers(terminal=term))
        res = dictify_core(Foo(), opts=opts)
        assert res == {"t": 1}
        assert calls == ["terminal"]

    def test_expand_with_invalid_max_depth_raises(self):
        # expand is internal but exercised via dictify_core path where max_depth < 1 triggers ValueError
        class A:
            pass

        opts = DictifyOptions(max_depth=0, handlers=Handlers(terminal=lambda o, _: {"x": 1}))
        res = dictify_core(A(), opts=opts)
        assert res == {"x": 1}

    # Tests for dictify_core when size limits are None.

    @pytest.mark.parametrize(
        "obj, options, expected_len",
        [
            pytest.param(
                list(range(500)),
                DictifyOptions(max_items=None, max_depth=3),
                500,
                id="list_unlimited_items",
            ),
            pytest.param(
                tuple(range(300)),
                DictifyOptions(max_items=None, max_depth=3),
                300,
                id="tuple_unlimited_items",
            ),
            pytest.param(
                set(range(400)),
                DictifyOptions(max_items=None, max_depth=3),
                400,
                id="set_unlimited_items",
            ),
        ],
    )
    def test_max_items_none_iterables(self, obj, options, expected_len):
        """Ensure no trimming when max_items is None."""
        out = dictify_core(obj, opts=options)
        assert isinstance(out, list)
        assert len(out) == expected_len

    def test_max_items_none_mapping(self):
        """Ensure mappings are not trimmed when max_items is None."""
        d = {f"k{i}": i for i in range(350)}
        opts = DictifyOptions(max_items=None, max_depth=3, sort_keys=False)
        out = dictify_core(d, opts=opts)
        assert isinstance(out, dict)
        assert len(out) == 350
        # Verify representative elements present
        assert out["k0"] == 0 and out["k349"] == 349

    @pytest.mark.parametrize(
        "s, options, expected_prefix, expected_suffix_len",
        [
            pytest.param(
                "x" * 10000,
                DictifyOptions(max_str_len=None, max_depth=3),
                "x" * 10,
                10000,
                id="str_no_truncation",
            ),
            pytest.param(
                "hello world",
                DictifyOptions(max_str_len=None, max_depth=3),
                "hello",
                11,
                id="short_str_no_truncation",
            ),
        ],
    )
    def test_max_str_len_none(self, s, options, expected_prefix, expected_suffix_len):
        """Ensure strings are not truncated when max_str_len is None."""
        out = dictify_core(s, opts=options)
        assert isinstance(out, str)
        assert out.startswith(expected_prefix)
        assert len(out) == expected_suffix_len

    @pytest.mark.parametrize(
        "b, options, expected_len",
        [
            pytest.param(
                b"a" * 10000,
                DictifyOptions(max_bytes=None, max_depth=3),
                10000,
                id="bytes_no_truncation",
            ),
            pytest.param(
                bytearray(b"b" * 7777),
                DictifyOptions(max_bytes=None, max_depth=3),
                7777,
                id="bytearray_no_truncation",
            ),
            pytest.param(
                memoryview(b"c" * 4096),
                DictifyOptions(max_bytes=None, max_depth=3),
                4096,
                id="memoryview_no_truncation",
            ),
        ],
    )
    def test_max_bytes_none(self, b, options, expected_len):
        """Ensure bytes-like are not truncated when max_bytes is None."""
        out = dictify_core(b, opts=options)
        # For bytearray and memoryview handlers, result may be converted to bytes or remain original;
        # assert by length after possible conversion to a bytes-like interface.
        if isinstance(out, (bytes, bytearray, memoryview)):
            length = len(out)
        else:
            # If handler returns a dict (e.g., for memoryview), use the contained data length
            if isinstance(b, memoryview) and isinstance(out, dict):
                data = out.get("data")
                assert data is not None, (
                    "memoryview dict must include full 'data' when max_bytes is None"
                )
                length = len(data)
            else:
                # If a handler converts memoryview/bytearray to bytes
                try:
                    length = len(out)  # fall back on generic len
                except Exception as e:
                    pytest.fail(f"Unexpected result type without __len__: {type(out)}; error: {e}")
        assert length == expected_len

    def test_combined_none_limits(self):
        """Ensure all limits None disable truncation consistently."""
        big = {
            "list": list(range(2000)),
            "tuple": tuple(range(1500)),
            "set": set(range(1800)),
            "dict": {str(i): i for i in range(1600)},
            "s": "z" * 5000,
            "b": b"y" * 6000,
        }
        opts = DictifyOptions(
            max_items=None,
            max_str_len=None,
            max_bytes=None,
            max_depth=4,
            sort_keys=False,
        )
        out = dictify_core(big, opts=opts)
        assert isinstance(out, dict)
        assert isinstance(out["list"], list) and len(out["list"]) == 2000
        assert isinstance(out["tuple"], list) and len(out["tuple"]) == 1500
        assert isinstance(out["set"], list) and len(out["set"]) == 1800
        assert isinstance(out["dict"], dict) and len(out["dict"]) == 1600
        assert isinstance(out["s"], str) and len(out["s"]) == 5000
        assert len(out["b"]) == 6000

    def test_invalid_none_interaction_depth(self):
        """Ensure no errors when limits None with terminal edge case."""
        obj = "a" * 1234
        opts = DictifyOptions(max_depth=0, max_str_len=None, max_items=None, max_bytes=None)
        out = dictify_core(obj, opts=opts)
        # Terminal mode may route through terminal handler or type handlers; ensure no exception and type preserved/processed.
        assert out is not None


class TestDictify:
    """
    Verify parameter override and options behavior.
    """

    @pytest.mark.parametrize(
        "obj,max_depth,expected_keys",
        [
            pytest.param({"a": {"b": 1}}, 1, {"a"}, id="depth_1_stops"),
            pytest.param({"a": {"b": 1}}, 2, {"a"}, id="depth_2_keeps_structure"),
        ],
    )
    def test_respects_max_depth(self, obj, max_depth, expected_keys):
        """Respect max_depth overrides."""
        out = dictify(
            obj,
            max_depth=max_depth,
            max_items=100,
            max_str_len=200,
            max_bytes=512,
            include_none=False,
            include_private=False,
            include_properties=False,
            include_class_name=False,
            sort_keys=False,
            sort_iterables=False,
        )
        assert set(out.keys()) == expected_keys

    @pytest.mark.parametrize(
        "obj,max_items,expected_len",
        [
            pytest.param(list(range(10)), 5, 5, id="list_trim"),
            pytest.param(tuple(range(10)), 3, 3, id="tuple_trim"),
            pytest.param({i: i for i in range(8)}, 4, 4, id="dict_trim"),
        ],
    )
    def test_respects_max_items(self, obj, max_items, expected_len):
        """Respect max_items trimming."""
        out = dictify(
            obj,
            max_depth=3,
            max_items=max_items,
            max_str_len=200,
            max_bytes=512,
            include_none=False,
            include_private=False,
            include_properties=False,
            include_class_name=False,
            sort_keys=False,
            sort_iterables=False,
        )
        # Output type differs by input; verify size effect only
        if isinstance(out, dict):
            size = len(out)
        elif isinstance(out, (list, tuple, set)):
            size = len(out)
        else:
            size = len(out) if hasattr(out, "__len__") else None
        assert size == expected_len

    @pytest.mark.parametrize(
        "s,max_str_len,expected",
        [
            pytest.param("abcdefghij", 4, "abcd...", id="truncate_str"),
            pytest.param("hello", 5, "hello", id="equal_limit"),
        ],
    )
    def test_respects_max_str_len(self, s, max_str_len, expected):
        """Respect max_str_len truncation."""
        out = dictify(
            s,
            max_depth=3,
            max_items=100,
            max_str_len=max_str_len,
            max_bytes=512,
            include_none=False,
            include_private=False,
            include_properties=False,
            include_class_name=False,
            sort_keys=False,
            sort_iterables=False,
        )
        assert isinstance(out, str)
        assert out == expected

    @pytest.mark.parametrize(
        "b,max_bytes,expected",
        [
            pytest.param(b"x" * 20, 7, b"xxxxxxx...", id="bytes_truncate"),
            pytest.param(bytearray(b"y" * 12), 8, b"yyyyyyyy...", id="bytearray_truncate"),
        ],
    )
    def test_respects_max_bytes(self, b, max_bytes, expected):
        """Respect max_bytes truncation for bytes-like."""
        out = dictify(
            b,
            max_depth=3,
            max_items=100,
            max_str_len=200,
            max_bytes=max_bytes,
            include_none=False,
            include_private=False,
            include_properties=False,
            include_class_name=False,
            sort_keys=False,
            sort_iterables=False,
        )
        # Output may be bytes or bytearray depending on core; compare by len
        assert out == expected

    def test_include_class_name_overrides(self):
        """Include class name when requested."""

        class P:
            def __init__(self):
                self.x = 1

        out = dictify(
            P(),
            max_depth=3,
            max_items=100,
            max_str_len=200,
            max_bytes=512,
            include_none=False,
            include_private=False,
            include_properties=False,
            include_class_name=True,
            sort_keys=False,
            sort_iterables=False,
        )
        assert isinstance(out, dict)
        assert "__class_name__" in out
        assert out["__class_name__"] in {"P", f"{P.__module__}.P"}

    @pytest.mark.parametrize(
        "obj,include_none,expected_present",
        [
            pytest.param({"a": None, "b": 2}, True, {"a", "b"}, id="include_none_true"),
            pytest.param({"a": None, "b": 2}, False, {"b"}, id="include_none_false"),
        ],
    )
    def test_include_none_flags(self, obj, include_none, expected_present):
        """Respect include_none flags for mapping items."""
        out = dictify(
            obj,
            max_depth=3,
            max_items=100,
            max_str_len=200,
            max_bytes=512,
            include_none=include_none,
            include_private=False,
            include_properties=False,
            include_class_name=False,
            sort_keys=False,
            sort_iterables=False,
        )
        assert set(out.keys()) == expected_present

    def test_respects_custom_options_handlers(self):
        """Respect user-provided options without overriding handlers."""
        sentinel = object()

        def custom_terminal(o, _):
            return {"term": sentinel}

        opts = DictifyOptions()
        opts.handlers.terminal = custom_terminal

        out = dictify(
            {"k": "v"},
            max_depth=0,  # triggers terminal handler
            max_items=100,
            max_str_len=200,
            max_bytes=512,
            include_none=False,
            include_private=False,
            include_properties=False,
            include_class_name=False,
            sort_keys=False,
            sort_iterables=False,
            opts=opts,
        )
        assert out == {"term": sentinel}

    def test_sorting_flags_affect_output_shape(self):
        """Respect sort_keys and sort_iterables flags."""
        obj = {"b": 2, "a": 1, "c": [3, 1, 2]}
        out = dictify(
            obj,
            max_depth=3,
            max_items=100,
            max_str_len=200,
            max_bytes=512,
            include_none=False,
            include_private=False,
            include_properties=False,
            include_class_name=False,
            sort_keys=True,
            sort_iterables=True,
        )
        assert list(out.keys()) == ["a", "b", "c"]
        assert out["c"] == [1, 2, 3]


# Test Core Private Methods --------------------------------------------------------------------------------------------


class Test_AttrIsProperty:
    """Test suite for the _attr_is_property utility function."""

    @pytest.mark.parametrize(
        ("attr_name", "try_callable", "expected"),
        [
            pytest.param("working_property", False, True, id="cls-prop"),
            pytest.param("failing_property", False, True, id="cls-prop-failing"),
            pytest.param("regular_attribute", False, False, id="cls-attr"),
            pytest.param("a_method", False, False, id="cls-method"),
            pytest.param("non_existent", False, False, id="cls-missing"),
            pytest.param("working_property", True, True, id="cls-prop-call"),
            pytest.param("failing_property", True, True, id="cls-prop-failing-call"),
        ],
    )
    def test_on_class(self, attr_name: str, try_callable: bool, expected: bool):
        """Detect property on class."""
        result = _attr_is_property(attr_name, Example, try_callable=try_callable)
        assert result is expected

    @pytest.mark.parametrize(
        ("attr_name", "try_callable", "expected"),
        [
            pytest.param("working_property", False, True, id="inst-prop"),
            pytest.param("failing_property", False, True, id="inst-prop-failing"),
            pytest.param("working_property", True, True, id="inst-prop-call-ok"),
            pytest.param("failing_property", True, False, id="inst-prop-call-err"),
            pytest.param("regular_attribute", False, False, id="inst-attr"),
            pytest.param("a_method", False, False, id="inst-method"),
            pytest.param("non_existent", False, False, id="inst-missing"),
        ],
    )
    def test_on_instance(self, attr_name: str, try_callable: bool, expected: bool):
        """Detect property on instance."""
        instance = Example()
        result = _attr_is_property(attr_name, instance, try_callable=try_callable)
        assert result is expected

    def test_on_dataclass_class(self):
        """Treat dataclass field as non-property on class."""
        assert _attr_is_property("field", SimpleDataClass, try_callable=False) is False

    def test_on_dataclass_instance(self):
        """Treat dataclass field as non-property on instance."""
        instance = SimpleDataClass()
        assert _attr_is_property("field", instance, try_callable=False) is False
