#
# C108 - Dictify Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import collections.abc as abc
import sys
from dataclasses import dataclass, is_dataclass, fields
from typing import Any

# Third-party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.dictify import (ClassNameOptions, DictifyOptions, HookMode, MetaMixin, DictifyMeta,
                          SizeMeta, TrimMeta, TypeMeta, MetaOptions, _Unset,
                          core_dictify, dictify)
from c108.tools import print_title
from c108.utils import class_name


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

    def test_merge_inject_class_name_with_override(self) -> None:
        """Override specific attributes when using inject_class_name."""
        opts = ClassNameOptions()
        result = opts.merge(inject_class_name=True, in_to_dict=False)
        assert result.in_expand is True
        assert result.in_to_dict is False
        assert result.key == "__class_name__"
        assert result.fully_qualified is False

    @pytest.mark.parametrize("initial_expand, initial_to_dict, inject_value, expected_expand, expected_to_dict", [
        pytest.param(False, False, True, True, True, id="false_false_to_true_true"),
        pytest.param(True, True, False, False, False, id="true_true_to_false_false"),
        pytest.param(True, False, True, True, True, id="true_false_to_true_true"),
        pytest.param(False, True, False, False, False, id="false_true_to_false_false"),
    ])
    def test_merge_inject_class_name_parametrized(self, initial_expand: bool, initial_to_dict: bool, inject_value: bool,
                                                  expected_expand: bool, expected_to_dict: bool) -> None:
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
        result = opts.merge(
            inject_class_name=False,
            in_expand=True,
            in_to_dict=True,
            key="custom_key",
            fully_qualified=True
        )
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


class TestDictifyMeta:

    def test_has_any_meta(self):
        """Report presence of any meta."""
        trim = TrimMeta(len=10, shown=7)
        size = SizeMeta(len=10, deep=200, shallow=150)
        typ = TypeMeta(from_type=list, to_type=tuple)
        meta = DictifyMeta(trim=trim, size=size, type=typ)
        assert meta.has_any_meta is True

    def test_is_trimmed_values(self):
        """Report trimmed state via TrimMeta."""
        meta_none = DictifyMeta(trim=None)
        assert meta_none.is_trimmed is None
        meta_no_trim = DictifyMeta(trim=TrimMeta(len=5, shown=5))
        assert meta_no_trim.is_trimmed is False
        meta_trimmed = DictifyMeta(trim=TrimMeta(len=5, shown=3))
        assert meta_trimmed.is_trimmed is True

    def test_to_dict_minimal(self):
        """Return version-only when empty."""
        meta = DictifyMeta(trim=None, size=None, type=None)
        result = meta.to_dict(include_none_attrs=False, include_properties=True, sort_keys=False)
        assert result == {"version": DictifyMeta.VERSION}

    def test_to_dict_full_sorted(self):
        """Include all sections and sort keys."""
        meta = DictifyMeta(
            trim=TrimMeta(len=10, shown=8),
            size=SizeMeta(len=10, deep=1024, shallow=512),
            type=TypeMeta(from_type=list, to_type=list),
        )
        result = meta.to_dict(include_none_attrs=True, include_properties=True, sort_keys=True)
        assert list(result.keys()) == ["size", "trim", "type", "version"]
        assert result["version"] == DictifyMeta.VERSION
        assert result["trim"] == {"is_trimmed": True, "len": 10, "shown": 8, "trimmed": 2}
        # SizeMeta includes all fields when include_none_attrs=True
        assert result["size"] == {"deep": 1024, "len": 10, "shallow": 512}
        # TypeMeta not converted -> to_dict omits redundant to_type
        assert result["type"] == {"from_type": list, "is_converted": False, "to_type": list, }

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            pytest.param(
                dict(trim=TrimMeta(len=3, shown=1)),
                {"trim": {"is_trimmed": True, "len": 3, "shown": 1, "trimmed": 2}, "version": DictifyMeta.VERSION},
                id="only-trim",
            ),
            pytest.param(
                dict(size=SizeMeta(len=None, deep=10, shallow=10)),
                {"size": {"deep": 10, "shallow": 10}, "version": DictifyMeta.VERSION},
                id="only-size",
            ),
            pytest.param(
                dict(type=TypeMeta(from_type=dict, to_type=None)),
                {"type": {"from_type": dict, "is_converted": False}, "version": DictifyMeta.VERSION},
                id="only-type-not-converted",
            ),
            pytest.param(
                dict(type=TypeMeta(from_type=set, to_type=frozenset)),
                {"type": {"from_type": set, "is_converted": True, "to_type": frozenset},
                 "version": DictifyMeta.VERSION},
                id="only-type-converted",
            ),
        ],
    )
    def test_to_dict_partial_sections(self, kwargs, expected):
        """Include only present sections."""
        meta = DictifyMeta(**kwargs)
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
            pytest.param(SizeMeta, dict(len=-1), ValueError, r"(?i) >=0", id="size-len-negative"),
            pytest.param(SizeMeta, dict(deep=True), TypeError, r"(?i) must be an int", id="size-deep-bool"),
            pytest.param(SizeMeta, dict(deep=1, shallow=2), ValueError, r"(?i).*deep.*>=.*shallow",
                         id="size-deep-lt-shallow"),
            pytest.param(TrimMeta, dict(len=-2, shown=0), ValueError, r"(?i) >=0", id="trim-len-negative"),
            pytest.param(TrimMeta, dict(len=True, shown=0), TypeError, r"(?i) must be an int", id="trim-shown-bool"),
            pytest.param(TrimMeta, dict(len=3, shown=5), ValueError, r"(?i).*shown.*<=.*len", id="trim-shown-gt-len"),
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


class TestDictifyMetaFromObjects:
    def test_none_when_all_disabled(self):
        """Return None when all meta flags are disabled."""
        opts = DictifyOptions(meta=MetaOptions(len=False, size=False, deep_size=False, trim=False, type=False))
        meta = DictifyMeta.from_objects([1, 2, 3], [1, 2, 3], opts)
        assert meta is None

    def test_size_only_len(self):
        """Create size meta when only len is enabled."""
        opts = DictifyOptions(meta=MetaOptions(len=True, size=False, deep_size=False, trim=False, type=False))
        obj = [1, 2, 3]
        meta = DictifyMeta.from_objects(obj, obj, opts)
        assert isinstance(meta, DictifyMeta)
        assert isinstance(meta.size, SizeMeta)
        assert meta.trim is None
        assert meta.type is None

    def test_trim_only(self):
        """Create trim meta when trim is enabled."""
        opts = DictifyOptions(meta=MetaOptions(len=False, size=False, deep_size=False, trim=True, type=False))
        original = list(range(10))
        processed = original[:5]
        meta = DictifyMeta.from_objects(original, processed, opts)
        assert isinstance(meta, DictifyMeta)
        assert meta.size is None
        assert isinstance(meta.trim, TrimMeta)
        assert isinstance(meta.is_trimmed, (bool, type(None)))

    def test_type_only_same(self):
        """Create type meta with no conversion for same types."""
        opts = DictifyOptions(meta=MetaOptions(len=False, size=False, deep_size=False, trim=False, type=True))
        original = {"a": 1}
        processed = {"a": 1}
        meta = DictifyMeta.from_objects(original, processed, opts)
        assert isinstance(meta, DictifyMeta)
        assert meta.size is None and meta.trim is None
        assert isinstance(meta.type, TypeMeta)
        assert meta.type.from_type is dict
        assert meta.type.to_type is dict
        assert meta.type.is_converted is False

    def test_type_only_different(self):
        """Create type meta with conversion for different types."""
        opts = DictifyOptions(meta=MetaOptions(len=False, size=False, deep_size=False, trim=False, type=True))
        original = (1, 2)
        processed = [1, 2]
        meta = DictifyMeta.from_objects(original, processed, opts)
        assert isinstance(meta, DictifyMeta)
        assert isinstance(meta.type, TypeMeta)
        assert meta.type.from_type is tuple
        assert meta.type.to_type is list
        assert meta.type.is_converted is True

    def test_type_only_none_processed(self):
        """Create type meta when processed object is None."""
        opts = DictifyOptions(meta=MetaOptions(len=False, size=False, deep_size=False, trim=False, type=True))
        original = "x"
        processed = None
        meta = DictifyMeta.from_objects(original, processed, opts)
        assert isinstance(meta, DictifyMeta)
        assert meta.type.from_type is str
        assert meta.type.to_type is type(None)
        assert meta.type.is_converted is True

    def test_all_meta(self):
        """Create all meta sections when all flags are enabled."""
        opts = DictifyOptions(meta=MetaOptions(len=True, size=True, deep_size=False, trim=True, type=True))
        original = list(range(8))
        processed = original[:5]
        meta = DictifyMeta.from_objects(original, processed, opts)
        assert isinstance(meta, DictifyMeta)
        assert isinstance(meta.size, SizeMeta)
        assert isinstance(meta.trim, TrimMeta)
        assert isinstance(meta.type, TypeMeta)

    def test_to_dict_integration(self):
        """Include version and enabled meta sections in to_dict."""
        opts = DictifyOptions(meta=MetaOptions(len=True, trim=True, type=True))
        original = [1, 2, 3, 4]
        processed = original[:2]
        meta = DictifyMeta.from_objects(original, processed, opts)
        d1 = meta.to_dict(include_none_attrs=False, include_properties=True, sort_keys=True)
        assert "version" in d1 and isinstance(d1["version"], int)
        assert "size" in d1 and "trim" in d1 and "type" in d1

        d2 = meta.to_dict(include_none_attrs=True, include_properties=True, sort_keys=False)
        assert "version" in d2 and "size" in d2 and "trim" in d2 and "type" in d2


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
            pytest.param(SimpleDC(a=1, b=None), True, {"a": 1, "b": None}, id="simple-include-none"),
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
    @pytest.mark.parametrize("kwargs,expected", [
        pytest.param({}, False, id="default_no_sizes"),
        pytest.param({"len": True}, True, id="len_enabled"),
        pytest.param({"size": True}, True, id="size_enabled"),
        pytest.param({"deep_size": True}, True, id="deep_size_enabled"),
        pytest.param({"len": False, "size": False, "deep_size": False}, False, id="all_sizes_disabled"),
    ])
    def test_sizes_enabled_property(self, kwargs, expected):
        """Verify sizes_enabled property correctly reports size metadata status."""
        meta_options = MetaOptions(**kwargs)
        assert meta_options.sizes_enabled == expected

    @pytest.mark.parametrize("kwargs,expected", [
        pytest.param({}, True, id="default_trim_enabled"),
        pytest.param({"trim": False}, False, id="trim_disabled"),
        pytest.param({"type": True}, True, id="type_enabled"),
        pytest.param({"trim": False, "type": False}, False, id="all_metadata_disabled"),
    ])
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

    @pytest.mark.parametrize("merge_kwargs,expected", [
        pytest.param({"len": True}, True, id="merge_len_true"),
        pytest.param({"size": False}, False, id="merge_size_false"),
        pytest.param({"key": "custom_key"}, "custom_key", id="merge_custom_key"),
    ])
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
    @pytest.mark.parametrize(
        "field, value",
        [
            pytest.param("len", -1, id="len-negative"),
            pytest.param("deep", -5, id="deep-negative"),
            pytest.param("shallow", -2, id="shallow-negative"),
        ],
    )
    def test_negative_values(self, field: str, value: int):
        """Reject negative integers."""
        kwargs = {field: value, "len": 0} if field != "len" else {field: value, "deep": 0}
        with pytest.raises(ValueError, match=r"(?i)>=0"):
            SizeMeta(**kwargs)

    @pytest.mark.parametrize(
        "field, value",
        [
            pytest.param("len", 3.14, id="len-float"),
            pytest.param("deep", "100", id="deep-str"),
            pytest.param("shallow", object(), id="shallow-object"),
            pytest.param("len", True, id="len-bool"),
            pytest.param("deep", False, id="deep-bool"),
        ],
    )
    def test_type_validation(self, field: str, value):
        """Reject non-int and bool values."""
        base = {"len": 0}
        base.pop(field, None)
        kwargs = {**base, field: value}
        with pytest.raises(TypeError, match=r"(?i)must be an int"):
            SizeMeta(**kwargs)

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
        assert SizeMeta.from_object(obj, include_len=False, include_deep=False, include_shallow=False) is None

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
        with pytest.raises(ValueError, match=r"(?i)requires at least 'shown'"):
            TrimMeta(None, None)
        with pytest.raises(ValueError, match=r"(?i)requires at least 'shown'"):
            TrimMeta(len=5, shown=None)

        tm = TrimMeta(len=None, shown=5)
        assert tm.len is None
        assert tm.shown == 5
        assert tm.trimmed is None
        assert tm.is_trimmed is None

    @pytest.mark.parametrize(
        "field, value",
        [
            pytest.param("len", -1, id="len-negative"),
            pytest.param("shown", -2, id="shown-negative"),
        ],
    )
    def test_negative_values(self, field: str, value: int):
        """Reject negative integers."""
        kwargs = {"len": 1, "shown": 1}
        kwargs[field] = value
        with pytest.raises(ValueError, match=r"(?i)>=0"):
            TrimMeta(**kwargs)

    @pytest.mark.parametrize(
        "field, value",
        [
            pytest.param("len", 3.14, id="len-float"),
            pytest.param("shown", "5", id="shown-str"),
            pytest.param("len", True, id="len-bool"),
            pytest.param("shown", False, id="shown-bool"),
        ],
    )
    def test_type_validation(self, field: str, value):
        """Reject non-int and bool values."""
        kwargs = {"len": 1, "shown": 1}
        kwargs[field] = value
        with pytest.raises(TypeError, match=r"(?i)must be an int"):
            TrimMeta(**kwargs)

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
            def __init__(self, n): self._n = n

            def __len__(self): return self._n

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
            pytest.param(None, int, False, id="from-none-to-type"),  # Changed: can't determine conversion
            pytest.param(int, None, False, id="to-none-no-conversion"),  # Changed: can't determine conversion
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
        d = tm.to_dict(include_none_attrs=False, include_properties=True,
                       sort_keys=True)  # Changed: False instead of True
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

    # Updated tests for from_objects

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

    # New tests for type validation

    def test_type_validation_from_type(self):
        """Validate that from_type must be a type or None."""
        with pytest.raises(TypeError, match="'from_type' must be a type or None"):
            TypeMeta(from_type="not_a_type", to_type=int)

    def test_type_validation_to_type(self):
        """Validate that to_type must be a type or None."""
        with pytest.raises(TypeError, match="'to_type' must be a type or None"):
            TypeMeta(from_type=int, to_type="not_a_type")


class TestInjectMeta:
    """Test inject_meta() functionality."""

    def test_inject_meta_returns_obj_when_meta_is_none(self):
        """Return original object when meta is None."""
        from c108.dictify import inject_meta
        opt = DictifyOptions()
        obj = {"key": "value"}
        result = inject_meta(obj, None, opt)
        assert result is obj

    def test_inject_meta_into_dict(self):
        """Inject metadata into dict under meta key."""
        from c108.dictify import inject_meta
        opt = DictifyOptions()
        obj = {"key": "value"}
        meta = DictifyMeta(size=SizeMeta(len=5))

        result = inject_meta(obj, meta, opt)

        assert isinstance(result, dict)
        assert "key" in result
        assert opt.meta.key in result
        assert "size" in result[opt.meta.key]

    def test_inject_meta_into_mapping(self):
        """Inject metadata into abc.Mapping by converting to dict."""
        from c108.dictify import inject_meta
        from collections import OrderedDict

        opt = DictifyOptions()
        obj = OrderedDict([("a", 1), ("b", 2)])
        meta = DictifyMeta(size=SizeMeta(len=2))

        result = inject_meta(obj, meta, opt)

        assert isinstance(result, dict)
        assert "a" in result
        assert opt.meta.key in result

    def test_inject_meta_into_list(self):
        """Inject metadata into list as last element."""
        from c108.dictify import inject_meta
        opt = DictifyOptions()
        obj = [1, 2, 3]
        meta = DictifyMeta(size=SizeMeta(len=3))

        result = inject_meta(obj, meta, opt)

        assert isinstance(result, list)
        assert len(result) == 4
        assert result[:3] == [1, 2, 3]
        assert isinstance(result[3], dict)
        assert opt.meta.key in result[3]

    def test_inject_meta_into_tuple_returns_as_is(self):
        """Inject metadata into tuple returns identity."""
        from c108.dictify import inject_meta
        opt = DictifyOptions()
        obj = (1, 2, 3)
        meta = DictifyMeta(size=SizeMeta(len=3))

        result = inject_meta(obj, meta, opt)

        assert result == obj

    def test_inject_meta_into_set_returns_as_is(self):
        """Inject metadata into set returns identity."""
        from c108.dictify import inject_meta
        opt = DictifyOptions()
        obj = {1, 2, 3}
        meta = DictifyMeta(size=SizeMeta(len=3))

        result = inject_meta(obj, meta, opt)

        assert result == obj

    def test_inject_meta_into_unsupported_type_returns_as_is(self):
        """Return object as-is for unsupported types without wrapping."""
        from c108.dictify import inject_meta
        opt = DictifyOptions()

        # Test with various unsupported types
        unsupported = [42, "string", 3.14, True, None, object()]

        for obj in unsupported:
            meta = DictifyMeta(size=SizeMeta(len=1))
            result = inject_meta(obj, meta, opt)
            assert result is obj, f"Failed for type {type(obj)}"

    def test_inject_meta_respects_custom_meta_key(self):
        """Use custom meta key for injection."""
        from c108.dictify import inject_meta
        opt = DictifyOptions()
        opt.meta.key = "__custom_meta__"

        obj = {"data": "value"}
        meta = DictifyMeta(size=SizeMeta(len=1))

        result = inject_meta(obj, meta, opt)

        assert "__custom_meta__" in result
        assert "__dictify__" not in result

    def test_inject_meta_with_different_meta_types(self):
        """Inject different metadata types correctly."""
        from c108.dictify import inject_meta
        opt = DictifyOptions()

        obj = {"key": "value"}
        meta = DictifyMeta(
            size=SizeMeta(len=10, shallow=100),
            trim=TrimMeta(len=100, shown=10),
            type=TypeMeta(from_type=list, to_type=dict)
        )

        result = inject_meta(obj, meta, opt)

        meta_content = result[opt.meta.key]
        assert "size" in meta_content
        assert "trim" in meta_content
        assert "type" in meta_content


# Main Functionality Tests ---------------------------------------------------------------------------------------------

class TestCoreDictify:

    def test_basic_object_conversion(self):
        """Convert simple object to dictionary."""

        class Person:
            def __init__(self, name, age):
                self.name = name
                self.age = age

        person = Person("Alice", 7)
        result = core_dictify(person)
        assert result == {"name": "Alice", "age": 7}

    @pytest.mark.parametrize(
        "value",
        [42, 3.14, True, 2 + 3j, None],
        ids=["int", "float", "bool", "complex", "none"],
    )
    def test_never_filtered_as_is(self, value):
        """Return never-filtered builtins as is."""
        assert core_dictify(value) is value

    def test_options_type_error(self):
        """Validate options type check."""
        with pytest.raises(TypeError, match=r"(?i)options must be a DictifyOptions"):
            core_dictify(object(), options=123)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("kw", "bad"),
        [("fn_raw", 123), ("fn_terminal", 123)],
        ids=["fn_raw", "fn_terminal"],
    )
    def test_fn_callable_type_error(self, kw, bad):
        """Validate fn_raw/fn_terminal type checks."""
        kwargs = {kw: bad}
        with pytest.raises(TypeError, match=r"(?i)must be a Callable"):
            core_dictify(object(), **kwargs)

    def test_hook_mode_dict_calls_to_dict_and_injects_class(self):
        """Inject class name when to_dict returns mapping."""

        class WithToDict:
            def to_dict(self):
                return {"x": 1}

        opts = DictifyOptions(hook_mode=HookMode.DICT, inject_class_name=True, fully_qualified_names=False)
        res = core_dictify(WithToDict(), options=opts)

        print("res:", res)

        assert res["x"] == 1
        assert res["__class__"] == "WithToDict"

    def test_hook_mode_strict_missing_to_dict_raises(self):
        """Raise when DICT_STRICT and no to_dict."""
        opts = DictifyOptions(hook_mode=HookMode.DICT_STRICT)
        with pytest.raises(TypeError, match=r"(?i)must implement to_dict"):
            core_dictify(object(), options=opts)

    def test_to_dict_non_mapping_raises(self):
        """Raise when to_dict returns non-mapping."""

        class BadToDict:
            def to_dict(self):
                return [("k", "v")]

        opts = DictifyOptions(hook_mode=HookMode.DICT)
        with pytest.raises(TypeError, match=r"(?i)must return a Mapping"):
            core_dictify(BadToDict(), options=opts)

    def test_max_depth_negative_uses_fn_plain(self):
        """Return fn_raw when max_depth is negative."""
        marker = object()
        opts = DictifyOptions(max_depth=-1)
        res = core_dictify(object(), options=opts, fn_raw=lambda x, opt: marker)
        assert res is marker

    def test_sequence_without_len_falls_back_to_fn_process(self):
        """Apply fn_terminal for Sequence lacking __len__."""

        class MySeqNoLen:
            def __iter__(self):
                yield from (1, 2, 3)

            # no __len__

        # Virtually register as Sequence while lacking __len__
        abc.Sequence.register(MySeqNoLen)

        marker = ("processed", "no-len")
        res = core_dictify(MySeqNoLen(), fn_terminal=lambda x, opt: marker)
        print("res:", res)
        assert res == marker

    @pytest.mark.parametrize(
        ("include_none_items", "expected_keys"),
        [(False, {"a"}), (True, {"a", "b"})],
        ids=["drop-none", "keep-none"],
    )
    def test_mapping_include_none_items(self, include_none_items, expected_keys):
        """Respect include_none_items for plain mappings."""
        opts = DictifyOptions(include_none_items=include_none_items)
        res = core_dictify({"a": 1, "b": None}, options=opts)
        assert set(res.keys()) == expected_keys

    def test_object_expansion_toplevel_filters_attrs(self):
        """Expand object attributes and respect include_none_attrs."""

        class Obj:
            def __init__(self):
                self.a = 1
                self.b = None

        opts = DictifyOptions(max_depth=1, include_none_attrs=False, include_class_name=False)
        res = core_dictify(Obj(), options=opts)
        assert res == {"a": 1}

    def test_depth_zero_uses_fn_process_on_user_object(self):
        """Use fn_terminal when max_depth is zero for user object."""

        class Foo:
            pass

        marker = ("processed", "Foo")
        opts = DictifyOptions(max_depth=0)
        res = core_dictify(Foo(), options=opts, fn_terminal=lambda x, opt: marker)
        assert res == marker

    def test_recursive_sequence_respects_depth(self):
        """Process nested sequences with proper depth control."""

        class Foo:
            def __init__(self):
                self.value = 42

        data = [[Foo()]]
        opts = DictifyOptions(max_depth=3, include_class_name=False)  # Need depth=3!
        res = core_dictify(data, options=opts)
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
        # Do not pass fn_terminal; identity fallback keeps terminal objects as-is.
        res = core_dictify(root, options=opts)

        print("result:", res)

        assert isinstance(res, dict)
        assert res["name"] == "root"
        assert res["child"] is leaf  # Raw object, not processed

    def test_invalid_hook_mode_raises_value_error(self):
        """Raise ValueError on invalid hook_mode."""
        bad_opts = DictifyOptions(hook_mode="unexpected")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=r"(?i)unknown hook_mode value"):
            core_dictify(object(), options=bad_opts)

    def test_property_exception_is_skipped(self):
        """Skip properties that raise exceptions when include_properties is on."""

        class WithBadProp:
            def __init__(self):
                self.ok = 1

            @property
            def bad(self):
                raise RuntimeError("boom")

        opts = DictifyOptions(max_depth=1, include_properties=True)
        res = core_dictify(WithBadProp(), options=opts)
        assert res == {"ok": 1}

    @pytest.mark.parametrize("fqn", [False, True], ids=["short-name", "fully-qualified"])
    def test_include_class_name_attrs(self, fqn):
        """Include class name during normal attribute scanning with optional FQN."""

        class Obj:
            def __init__(self):
                self.a = 1

        opts = DictifyOptions(
            max_depth=1,
            include_class_name=True,
            fully_qualified_names=fqn,
        )
        res = core_dictify(Obj(), options=opts)

        expected_class = Obj.__name__ if not fqn else f"{Obj.__module__}.{Obj.__name__}"
        assert res["a"] == 1
        assert res["__class__"] == expected_class

    def test_include_class_name_attrs_disabled(self):
        """Do not include class name when option is disabled."""

        class Obj:
            def __init__(self):
                self.a = 1

        opts = DictifyOptions(max_depth=1, include_class_name=False)
        res = core_dictify(Obj(), options=opts)
        assert res == {"a": 1}

    def test_to_dict_injects_class_name_fqn(self):
        """Inject class name into to_dict result with fully qualified name."""

        class WithToDict:
            def to_dict(self):
                return {"x": 1}

        opts = DictifyOptions(
            hook_mode=HookMode.DICT,
            inject_class_name=True,
            fully_qualified_names=True,
        )
        res = core_dictify(WithToDict(), options=opts)

        expected_class = f"{WithToDict.__module__}.{WithToDict.__name__}"
        assert res["x"] == 1
        assert res["__class__"] == expected_class

    def test_to_dict_no_injection_when_disabled(self):
        """Do not inject class name when inject_class_name is False for to_dict."""

        class WithToDict:
            def to_dict(self):
                return {"x": 1}

        opts = DictifyOptions(
            hook_mode=HookMode.DICT,
            inject_class_name=False,
            fully_qualified_names=True,
        )
        res = core_dictify(WithToDict(), options=opts)
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
        opts = DictifyOptions(max_depth=2, include_class_name=False)
        res = core_dictify(root, options=opts)

        assert res["name"] == "root"
        assert res["child"]["name"] == "mid"
        assert res["child"]["child"] is leaf  # Raw at terminal depth

    def test_fn_terminal_output_not_modified(self):
        """Do not inject class name into fn_terminal output."""

        class Foo:
            pass

        # At depth=0, fn_terminal is used and its output must not be modified.
        opts = DictifyOptions(max_depth=0, include_class_name=True, fully_qualified_names=True)
        res = core_dictify(Foo(), options=opts, fn_terminal=lambda x, opt: {"marker": "terminal"})
        assert res == {"marker": "terminal"}


#
# class TestCoreVsDictify:
#     def test_object_tree_depth_control(self):
#         """Expand object to dict but keep nested objects as raw values at depth 1."""
#
#         class Node:
#             def __init__(self, name=None, child=None):
#                 self.a = name
#                 self.b = 'b'
#                 self.child = child
#
#         def fn_terminal(obj):
#             if isinstance(obj, (int, float, str)):
#                 return obj
#             return {"terminal": f"{obj.a} - child:{bool(obj.child)} - {sys.getsizeof(obj)} bytes"}
#
#         leaf_2 = Node(name="leaf_2")
#         leaf_1 = Node(name="leaf_1", child=leaf_2)
#         leaf_0 = Node(name="leaf_0", child=leaf_1)
#         root = Node(name="root", child=leaf_0)
#
#         for d in [-1, 0, 1, 2, 3, 4, 10]:
#             print("\ndepth       :", d)
#             # print("dictify     :", dictify(root, max_depth=d))
#             print("core_dictify:", core_dictify(root, fn_terminal=fn_terminal, options=DictifyOptions(max_depth=d)))
#
#         # assert res["name"] == "root"
#         # assert res["child"] is leaf  # Raw object, not processed
#

class TestDictify:
    """Test suite for dictify() method."""

    def test_basic_object_conversion(self):
        """Convert simple object to dictionary."""

        class Person:
            def __init__(self, name, age):
                self.name = name
                self.age = age

        person = Person("Dodo", 5)
        result = dictify(person)

        print("\nresult:", result)

        assert result == {"name": "Dodo", "age": 5}

    # @pytest.mark.parametrize("primitive", [
    #     42, 3.14, True, None, range(0, 7)
    # ], ids=["int", "float", "bool", "none", "range"])
    # def test_primitive_types_preserved(self, primitive):
    #     """Preserve built-in types as-is."""
    #     result = dictify(primitive)
    #     assert result == primitive
    #
    # def test_max_depth_control(self):
    #     """Control recursion depth with max_depth parameter."""
    #
    #     raise NotImplemented("not implemented, try to use simplified logic of test_object_tree_depth_control()")
    #
    # def test_include_private_attributes(self):
    #     """Include private attributes when include_private=True."""
    #
    #     class TestClass:
    #         def __init__(self):
    #             self.public = "visible"
    #             self._private = "hidden"
    #
    #     obj = TestClass()
    #
    #     # Default: exclude private
    #     result_default = dictify(obj)
    #     assert "_private" not in result_default
    #     assert result_default == {"public": "visible"}
    #
    #     # Include private
    #     result_with_private = dictify(obj, include_private=True)
    #     assert result_with_private == {"public": "visible", "_private": "hidden"}
    #
    # def test_include_class_name(self):
    #     """Include class name when include_class_name=True."""
    #
    #     class MyClass:
    #         def __init__(self):
    #             self.attr = "value"
    #
    #     obj = MyClass()
    #     opt = DictifyOptions(fully_qualified_names=False)
    #     result = dictify(obj, include_class_name=True, options=opt)
    #     assert result["__class__"] == "MyClass"
    #     assert result["attr"] == "value"
    #
    # def test_max_items_limitation(self):
    #     """Limit collection size with max_items parameter."""
    #     large_dict = {f"key_{i}": i for i in range(100)}
    #     result = dictify(large_dict, max_items=10)
    #     assert len(result) <= 10
    #
    # def test_options_override_parameters(self):
    #     """Use DictifyOptions to override individual parameters."""
    #
    #     class TestClass:
    #         def __init__(self):
    #             self.attr = "value"
    #             self._private = "hidden"
    #
    #     obj = TestClass()
    #     # Options should be used when provided as kwargs
    #     result = dictify(obj, include_private=True, include_class_name=True)
    #     print("\nresult", result)
    #     assert "_private" in result
    #     assert "__class__" in result
    #
    # @pytest.mark.parametrize(
    #     "invalid_depth",
    #     ["not_int", 3.5, None, [], {}],
    #     ids=["str", "float", "none", "list", "dict"],
    # )
    # def test_max_depth_type_validation(self, invalid_depth):
    #     """Raise TypeError for invalid max_depth types."""
    #     obj = {"test": "value"}
    #     with pytest.raises(TypeError, match=r"(?i)int"):
    #         dictify(obj, max_depth=invalid_depth)
    #
    # @pytest.mark.parametrize(
    #     "invalid_items",
    #     ["not_int", 50.5, None, (), {"a": 1}],
    #     ids=["str", "float", "none", "tuple", "dict"],
    # )
    # def test_max_items_type_validation(self, invalid_items):
    #     """Raise TypeError for invalid max_items types."""
    #     obj = {"test": "value"}
    #     with pytest.raises(TypeError, match=r"(?i)int"):
    #         dictify(obj, max_items=invalid_items)
    #
    # @pytest.mark.parametrize("invalid_options", [
    #     "not_options",
    #     123,
    #     {"not": "options"},
    # ], ids=["str", "int", "dict"])
    # def test_options_type_validation(self, invalid_options):
    #     """Raise TypeError for invalid options type."""
    #     obj = {"test": "value"}
    #
    #     with pytest.raises(TypeError, match=r"(?i)DictifyOptions"):
    #         dictify(obj, options=invalid_options)
    #
    # def test_nested_collections_processing(self):
    #     """Process nested collections up to max_depth levels."""
    #
    #     class Item:
    #         def __init__(self, name):
    #             self.name = name
    #
    #     nested_data = {
    #         "items": [Item("first"), Item("second")],
    #         "metadata": {"count": 2, "nested": Item("meta")}
    #     }
    #
    #     result = dictify(nested_data, max_depth=3)
    #     expected = {"items": [{"name": "first"},
    #                           {"name": "second"}],
    #                 "metadata": {"count": 2,
    #                              "nested": {"name": 'meta'}}
    #                 }
    #     assert result == expected
    #     assert result["items"][0] == {"name": "first"}
    #     assert result["items"][1] == {"name": "second"}
    #     assert result["metadata"]["nested"] == {"name": "meta"}
    #
    # def test_exception_properties_skipped(self):
    #     """Skip properties that raise exceptions during access."""
    #
    #     class ProblematicClass:
    #         def __init__(self):
    #             self.good_attr = "accessible"
    #
    #         @property
    #         def bad_property(self):
    #             raise RuntimeError("Cannot access this property")
    #
    #     obj = ProblematicClass()
    #     result = dictify(obj)
    #
    #     # Should contain good attribute but skip the problematic property
    #     assert "good_attr" in result
    #     assert "bad_property" not in result
    #     assert result["good_attr"] == "accessible"
    #
    # def test_complex_nested_structure(self):
    #     """Handle complex nested structures with mixed types."""
    #
    #     class Address:
    #         def __init__(self, street, city):
    #             self.street = street
    #             self.city = city
    #
    #     class Person:
    #         def __init__(self, name, addresses):
    #             self.name = name
    #             self.addresses = addresses
    #             self._id = 12345
    #
    #     person = Person("Bob", [
    #         Address("123 Main St", "Springfield"),
    #         Address("456 Oak Ave", "Shelbyville")
    #     ])
    #
    #     result = dictify(person, max_depth=3, include_private=True)
    #
    #     expected = {
    #         "name": "Bob",
    #         "_id": 12345,
    #         "addresses": [
    #             {"street": "123 Main St", "city": "Springfield"},
    #             {"street": "456 Oak Ave", "city": "Shelbyville"}
    #         ]
    #     }
    #
    #     assert result == expected
    #
