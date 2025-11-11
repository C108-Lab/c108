#
# C108 - dataclasses.py Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
from dataclasses import dataclass, field, InitVar


# Third-party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.dataclasses import mergeable
from c108.sentinels import UNSET


# Tests ----------------------------------------------------------------------------------------------------------------


class TestMergeableBasic:
    """Test basic mergeable() functionality."""

    def test_merge_single_field(self):
        """Merge a single field and keep others unchanged."""

        @mergeable
        @dataclass
        class Config:
            timeout: int = 30
            retries: int = 3

        c1 = Config()
        c2 = c1.merge(timeout=60)

        assert c2.timeout == 60
        assert c2.retries == 3

    def test_merge_multiple_fields(self):
        """Merge multiple fields at once."""

        @mergeable
        @dataclass
        class Config:
            timeout: int = 30
            retries: int = 3
            server: str = "localhost"

        c1 = Config()
        c2 = c1.merge(timeout=60, server="prod")

        assert c2.timeout == 60
        assert c2.retries == 3
        assert c2.server == "prod"

    def test_merge_with_unset_ignored(self):
        """Merge ignores fields explicitly set to UNSET."""

        @mergeable
        @dataclass
        class Config:
            timeout: int = 30
            retries: int = 3

        c1 = Config()
        c2 = c1.merge(timeout=60, retries=UNSET)

        assert c2.timeout == 60
        assert c2.retries == 3

    def test_merge_returns_new_instance(self):
        """Merge returns a new instance without modifying the original."""

        @mergeable
        @dataclass
        class Config:
            value: int = 1

        c1 = Config()
        c2 = c1.merge(value=2)

        assert c1.value == 1
        assert c2.value == 2
        assert c1 is not c2

    def test_merge_chaining(self):
        """Chain multiple merge calls fluently."""

        @mergeable
        @dataclass
        class Config:
            x: int = 1
            y: int = 2
            z: int = 3

        c1 = Config()
        c2 = c1.merge(x=10).merge(y=20).merge(z=30)

        assert c2.x == 10
        assert c2.y == 20
        assert c2.z == 30


class TestMergeableSentinels:
    """Test different sentinel configurations."""

    def test_none_sentinel_keeps_existing(self):
        """Use None as sentinel to keep existing values."""

        @mergeable(sentinel=None)
        @dataclass
        class Options:
            value: int | None = 5

        o1 = Options()
        o2 = o1.merge(value=None)

        assert o2.value == 5

    def test_none_sentinel_updates_value(self):
        """Update value when using None as sentinel."""

        @mergeable(sentinel=None)
        @dataclass
        class Options:
            value: int | None = 5

        o1 = Options()
        o2 = o1.merge(value=10)

        assert o2.value == 10

    def test_unset_sentinel_sets_none(self):
        """Set None explicitly when using UNSET sentinel."""

        @mergeable(sentinel=UNSET)
        @dataclass
        class Config:
            value: int | None = 5

        c1 = Config()
        c2 = c1.merge(value=None)

        assert c2.value is None


class TestMergeableIncludeExclude:
    """Test field filtering with include/exclude."""

    def test_include_whitelist_only(self):
        """Merge only fields in include list."""

        @mergeable(include=["timeout"])
        @dataclass
        class Limited:
            timeout: int = 30
            internal: int = 99

        lim = Limited()
        result = lim.merge(timeout=60)

        assert result.timeout == 60
        assert result.internal == 99

    def test_include_rejects_unlisted_field(self):
        """Raise TypeError when merging unlisted field with include."""

        @mergeable(include=["timeout"])
        @dataclass
        class Limited:
            timeout: int = 30
            internal: int = 99

        lim = Limited()

        with pytest.raises(TypeError, match=r"(?i).*unexpected.*internal.*"):
            lim.merge(internal=1)

    def test_exclude_blacklist_fields(self):
        """Exclude specific fields from merging."""

        @mergeable(exclude=["_internal"])
        @dataclass
        class Config:
            public_val: int = 1
            another: str = "test"
            _internal: int = 999

        cfg = Config()
        result = cfg.merge(public_val=42, another="updated")

        assert result.public_val == 42
        assert result.another == "updated"
        assert result._internal == 999

    def test_exclude_rejects_excluded_field(self):
        """Raise TypeError when merging excluded field."""

        @mergeable(exclude=["_internal"])
        @dataclass
        class Config:
            public_val: int = 1
            _internal: int = 999

        cfg = Config()

        with pytest.raises(TypeError, match=r"(?i).*unexpected.*_internal.*"):
            cfg.merge(_internal=0)


class TestMergeablePrivateFields:
    """Test private field handling."""

    def test_private_fields_allowed_by_default(self):
        """Allow merging private fields by default."""

        @mergeable(include_private=True)
        @dataclass
        class State:
            public: int = 1
            _private: int = 2

        s1 = State()
        s2 = s1.merge(_private=99)

        assert s2._private == 99
        assert s2.public == 1

    def test_private_fields_excluded_when_disabled(self):
        """Exclude private fields when include_private=False."""

        @mergeable(include_private=False)
        @dataclass
        class State:
            public: int = 1
            _private: int = 2

        s1 = State()

        with pytest.raises(TypeError, match=r"(?i).*unexpected.*_private.*"):
            s1.merge(_private=99)

    def test_private_fields_copied_when_excluded(self):
        """Copy private field values even when not mergeable."""

        @mergeable(include_private=False)
        @dataclass
        class State:
            public: int = 1
            _private: int = 2

        s1 = State(_private=999)
        s2 = s1.merge(public=42)

        assert s2.public == 42
        assert s2._private == 999


class TestMergeableFrozen:
    """Test with frozen dataclasses."""

    def test_frozen_dataclass_merge(self):
        """Merge frozen dataclass creates new instance."""

        @mergeable
        @dataclass(frozen=True)
        class Immutable:
            x: int = 1
            y: int = 2

        obj = Immutable()
        result = obj.merge(x=10)

        assert result.x == 10
        assert result.y == 2

    def test_frozen_dataclass_original_unchanged(self):
        """Original frozen instance remains unchanged after merge."""

        @mergeable
        @dataclass(frozen=True)
        class Immutable:
            value: int = 1

        obj = Immutable()
        result = obj.merge(value=99)

        assert obj.value == 1
        assert result.value == 99


class TestMergeableFieldTypes:
    """Test with different field types."""

    def test_init_false_fields_reset(self):
        """Fields with init=False reset to defaults after merge."""

        @mergeable
        @dataclass
        class WithComputed:
            x: int = 1
            y: int = field(init=False, default=999)

        obj = WithComputed(x=5)
        obj.y = 100
        result = obj.merge(x=10)

        assert result.x == 10
        assert result.y == 999

    def test_default_factory_shallow_copy(self):
        """Default factory fields use shallow copy semantics."""

        @mergeable
        @dataclass
        class WithList:
            value: int = 1
            items: list[int] = field(default_factory=list)

        obj = WithList()
        obj.items.append(42)
        result = obj.merge(value=2)

        assert result.value == 2
        assert result.items == [42]
        assert result.items is obj.items

    def test_initvar_fields_not_mergeable(self):
        """InitVar fields cannot be merged."""

        @mergeable
        @dataclass
        class WithInitVar:
            timeout: int = 30
            debug: InitVar[bool] = False

            def __post_init__(self, debug):
                pass

        obj = WithInitVar()

        with pytest.raises(TypeError, match=r"(?i).*unexpected.*debug.*"):
            obj.merge(debug=True)


class TestMergeableValidation:
    """Test decorator validation."""

    def test_reject_both_include_and_exclude(self):
        """Raise ValueError when both include and exclude specified."""
        with pytest.raises(ValueError, match=r"(?i).*both.*include.*exclude.*"):

            @mergeable(include=["x"], exclude=["y"])
            @dataclass
            class Invalid:
                x: int = 1
                y: int = 2

    def test_reject_nonexistent_field_in_include(self):
        """Raise ValueError when include references non-existent field."""
        with pytest.raises(ValueError, match=r"(?i).*'nonexistent'.*does not exist.*"):

            @mergeable(include=["nonexistent"])
            @dataclass
            class Invalid:
                x: int = 1

    def test_reject_init_false_field_in_include(self):
        """Raise ValueError when include references init=False field."""
        with pytest.raises(ValueError, match=r"(?i).*'computed'.*init=False.*"):

            @mergeable(include=["computed"])
            @dataclass
            class Invalid:
                x: int = 1
                computed: int = field(init=False, default=0)

    def test_reject_initvar_field_in_include(self):
        """Raise ValueError when include references InitVar field."""
        with pytest.raises(ValueError, match=r"(?i).*'debug'.*Invalid.*"):

            @mergeable(include=["debug"])
            @dataclass
            class Invalid:
                x: int = 1
                debug: InitVar[bool] = False

                def __post_init__(self, debug):
                    pass

    def test_reject_non_dataclass(self):
        """Raise TypeError when decorating non-dataclass."""
        with pytest.raises(TypeError, match=r"(?i).*must be a dataclass.*"):

            @mergeable
            class NotDataclass:
                x: int = 1


class TestMergeableExamples:
    """Test examples from documentation."""

    def test_display_symbols_example(self):
        """Merge DisplaySymbols with unicode updates."""

        @mergeable
        @dataclass(frozen=True)
        class DisplaySymbols:
            nan: str = "NaN"
            pos_infinity: str = "inf"
            mult: str = "*"
            separator: str = " "

        symbols = DisplaySymbols()
        updated = symbols.merge(pos_infinity="∞", mult="×")

        assert updated.pos_infinity == "∞"
        assert updated.mult == "×"
        assert updated.nan == "NaN"
        assert updated.separator == " "

    def test_transfer_options_example(self):
        """Merge TransferOptions with None sentinel."""

        @mergeable(sentinel=None)
        @dataclass
        class TransferOptions:
            base_timeout: float = 5.0
            max_retries: int = 0
            speed: float = 100.0

        opts = TransferOptions()
        opts2 = opts.merge(speed=None, max_retries=3)

        assert opts2.speed == pytest.approx(100.0)
        assert opts2.max_retries == 3
        assert opts2.base_timeout == pytest.approx(5.0)

    def test_config_with_exclude_example(self):
        """Merge Config with excluded internal field."""

        @mergeable(exclude=["_internal"])
        @dataclass
        class Config:
            public_val: int = 1
