#
# C108 - Sentinels Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import pickle

# Third-party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.sentinels import (
    UNSET, MISSING, DEFAULT, NOT_FOUND, STOP,
    UnsetType, MissingType, DefaultType, NotFoundType, StopType,
    ifdefault, ifmissing, ifnotfound, ifstop, ifunset
)


# Local Classes & Methods ----------------------------------------------------------------------------------------------

class TestSentinels:
    def test_singleton_identity(self):
        """Ensure each sentinel is a singleton object."""
        assert UNSET is UnsetType()
        assert MISSING is MissingType()
        assert DEFAULT is DefaultType()
        assert NOT_FOUND is NotFoundType()
        assert STOP is StopType()

    @pytest.mark.parametrize(
        ("sentinel", "expected"),
        [
            pytest.param(UNSET, "<UNSET>", id="unset"),
            pytest.param(MISSING, "<MISSING>", id="missing"),
            pytest.param(DEFAULT, "<DEFAULT>", id="default"),
            pytest.param(NOT_FOUND, "<NOT_FOUND>", id="not_found"),
            pytest.param(STOP, "<STOP>", id="stop"),
        ],
    )
    def test_repr_clean(self, sentinel, expected):
        """Assert repr shows clean angle-bracketed name."""
        assert repr(sentinel) == expected

    @pytest.mark.parametrize(
        ("s1", "s2", "is_equal", "eq_result"),
        [
            pytest.param(UNSET, UNSET, True, True, id="unset-self"),
            pytest.param(UNSET, MISSING, False, False, id="unset-missing"),
            pytest.param(MISSING, MISSING, True, True, id="missing-self"),
            pytest.param(DEFAULT, DEFAULT, True, True, id="default-self"),
            pytest.param(NOT_FOUND, NOT_FOUND, True, True, id="not_found-self"),
            pytest.param(STOP, STOP, True, True, id="stop-self"),
        ],
    )
    def test_identity_and_eq(self, s1, s2, is_equal, eq_result):
        """Verify identity and equality are aligned."""
        assert (s1 is s2) is is_equal
        assert (s1 == s2) is eq_result  # type: ignore[comparison-overlap]

    @pytest.mark.parametrize(
        ("sentinel", "expected_bool"),
        [
            pytest.param(UNSET, False, id="unset-false"),
            pytest.param(MISSING, False, id="missing-false"),
            pytest.param(DEFAULT, True, id="default-true"),
            pytest.param(NOT_FOUND, False, id="not_found-false"),
            pytest.param(STOP, False, id="stop-false"),
        ],
    )
    def test_truthiness(self, sentinel, expected_bool):
        """Check boolean conversion semantics."""
        assert bool(sentinel) is expected_bool

    @pytest.mark.parametrize(
        ("sentinel",),
        [
            pytest.param(UNSET, id="unset"),
            pytest.param(MISSING, id="missing"),
            pytest.param(DEFAULT, id="default"),
            pytest.param(NOT_FOUND, id="not_found"),
            pytest.param(STOP, id="stop"),
        ],
    )
    def test_hash_is_identity_based(self, sentinel):
        """Confirm hash is consistent with identity."""
        assert hash(sentinel) == id(sentinel)
        # Also ensure hash is stable across calls
        assert hash(sentinel) == hash(sentinel)

    @pytest.mark.parametrize(
        ("sentinel",),
        [
            pytest.param(UNSET, id="unset"),
            pytest.param(MISSING, id="missing"),
            pytest.param(DEFAULT, id="default"),
            pytest.param(NOT_FOUND, id="not_found"),
            pytest.param(STOP, id="stop"),
        ],
    )
    def test_pickle_roundtrip(self, sentinel):
        """Ensure pickling preserves singleton identity."""
        data = pickle.dumps(sentinel, protocol=pickle.HIGHEST_PROTOCOL)
        loaded = pickle.loads(data)
        assert loaded is sentinel

    @pytest.mark.parametrize(
        ("cls", "name", "expected_singleton"),
        [
            pytest.param(UnsetType, "UNSET", UNSET, id="UnsetType"),
            pytest.param(MissingType, "MISSING", MISSING, id="MissingType"),
            pytest.param(DefaultType, "DEFAULT", DEFAULT, id="DefaultType"),
            pytest.param(NotFoundType, "NOT_FOUND", NOT_FOUND, id="NotFoundType"),
            pytest.param(StopType, "STOP", STOP, id="StopType"),
        ],
    )
    def test_reduce_protocol(self, cls, name, expected_singleton):
        """Validate __reduce__ returns reconstructable callable and args."""
        tmp = cls()
        reduce_tuple = tmp.__reduce__()
        assert isinstance(reduce_tuple, tuple) and len(reduce_tuple) == 2
        ctor, args = reduce_tuple
        assert ctor is cls
        # Accept both legacy (name,) and current () forms but ensure reconstruction works
        assert isinstance(args, tuple)
        reconstructed = ctor(*args)
        assert reconstructed is expected_singleton

    def test_equality_with_non_sentinel(self):
        """Ensure equality with non-sentinel is false."""
        assert (UNSET == object()) is False  # type: ignore[comparison-overlap]
        assert (MISSING == None) is False  # noqa: E711  # type: ignore[comparison-overlap]


class TestIfWrappers:
    """Tests for public sentinel wrapper functions (ifunset, ifmissing, ifdefault, ifnotfound, ifstop)."""

    @pytest.mark.parametrize(
        "func,sentinel,value,default,default_factory,expected",
        [
            pytest.param(ifunset, UNSET, "x", "d", None, "x", id="ifunset_not_sentinel"),
            pytest.param(ifunset, UNSET, UNSET, "d", None, "d", id="ifunset_default"),
            pytest.param(ifunset, UNSET, UNSET, None, lambda: "f", "f",
                         id="ifunset_factory"),
            pytest.param(ifmissing, MISSING, "x", "d", None, "x", id="ifmissing_not_sentinel"),
            pytest.param(ifmissing, MISSING, MISSING, "d", None, "d",
                         id="ifmissing_default"),
            pytest.param(ifmissing, MISSING, MISSING, None, lambda: "f", "f",
                         id="ifmissing_factory"),
            pytest.param(ifdefault, DEFAULT, "x", "d", None, "x", id="ifdefault_not_sentinel"),
            pytest.param(ifdefault, DEFAULT, DEFAULT, "d", None, "d",
                         id="ifdefault_default"),
            pytest.param(ifdefault, DEFAULT, DEFAULT, None, lambda: "f", "f",
                         id="ifdefault_factory"),
            pytest.param(ifnotfound, NOT_FOUND, "x", "d", None, "x", id="ifnotfound_not_sentinel"),
            pytest.param(ifnotfound, NOT_FOUND, NOT_FOUND, "d", None, "d",
                         id="ifnotfound_default"),
            pytest.param(ifnotfound, NOT_FOUND, NOT_FOUND, None, lambda: "f", "f",
                         id="ifnotfound_factory"),
            pytest.param(ifstop, STOP, "x", "d", None, "x", id="ifstop_not_sentinel"),
            pytest.param(ifstop, STOP, STOP, "d", None, "d", id="ifstop_default"),
            pytest.param(ifstop, STOP, STOP, None, lambda: "f", "f", id="ifstop_factory"),
        ],
    )
    def test_core_behavior(self, func, sentinel, value, default, default_factory, expected):
        """Return correct value, default, or factory result depending on sentinel match."""
        # Parametrize: [func, sentinel, value, default, default_factory, expected]
        if default_factory:
            result = func(value, default_factory=default_factory)
        else:
            result = func(value, default=default)
        assert result == expected

    @pytest.mark.parametrize(
        "func,sentinel,value",
        [
            pytest.param(ifunset, UNSET, UNSET, id="ifunset"),
            pytest.param(ifmissing, MISSING, MISSING, id="ifmissing"),
            pytest.param(ifdefault, DEFAULT, DEFAULT, id="ifdefault"),
            pytest.param(ifnotfound, NOT_FOUND, NOT_FOUND, id="ifnotfound"),
            pytest.param(ifstop, STOP, STOP, id="ifstop"),
        ],
    )
    def test_raises_when_both_default_and_factory(self, func, sentinel, value):
        """Raise ValueError when both default and default_factory are provided."""
        # Parametrize: [func, sentinel, value]
        with pytest.raises(ValueError, match=r"(?i)both default and default_factory"):
            func(value, default="d", default_factory=lambda: "f")

    def test_factory_not_called_when_not_sentinel(self):
        """Ensure default_factory is not called when value does not match sentinel."""
        called = {"count": 0}

        def factory():
            called["count"] += 1
            return "f"

        result = ifunset("x", default_factory=factory)
        assert result == "x"
        assert called["count"] == 0

    @pytest.mark.parametrize(
        "func,sentinel",
        [
            pytest.param(ifunset, UNSET, id="ifunset"),
            pytest.param(ifmissing, MISSING, id="ifmissing"),
            pytest.param(ifdefault, DEFAULT, id="ifdefault"),
            pytest.param(ifnotfound, NOT_FOUND, id="ifnotfound"),
            pytest.param(ifstop, STOP, id="ifstop"),
        ],
    )
    def test_default_none_behavior(self, func, sentinel):
        """Return None when sentinel matches and no default or factory provided."""
        # Parametrize: [func, sentinel]
        result = func(sentinel)
        assert result is None


class Test_IfSentinel:
    """Tests for the internal _if_sentinel() helper."""

    @pytest.mark.parametrize(
        "value,sentinel,default,expected",
        [
            pytest.param("x", "y", "d", "x", id="value_not_sentinel_returns_value"),
            pytest.param("x", "x", "d", "d", id="value_is_sentinel_returns_default"),
        ],
    )
    def test_basic_behavior(self, value, sentinel, default, expected):
        """Return correct value or default depending on sentinel match."""
        from c108.sentinels import _if_sentinel
        result = _if_sentinel(value, sentinel, default=default)
        assert result == expected

    def test_raises_when_both_default_and_factory(self):
        """Raise ValueError when both default and default_factory are provided."""
        from c108.sentinels import _if_sentinel
        with pytest.raises(ValueError, match=r"(?i)both default and default_factory"):
            _if_sentinel("x", "x", default="d", default_factory=lambda: "f")

    def test_uses_default_factory_when_provided(self):
        """Return result of default_factory when sentinel matches."""
        from c108.sentinels import _if_sentinel
        factory_called = {"count": 0}

        def factory():
            factory_called["count"] += 1
            return "factory_value"

        result = _if_sentinel("x", "x", default_factory=factory)
        assert result == "factory_value"
        assert factory_called["count"] == 1

    def test_returns_value_when_not_matching_sentinel(self):
        """Return original value when it does not match sentinel."""
        from c108.sentinels import _if_sentinel
        factory_called = {"count": 0}

        def factory():
            factory_called["count"] += 1
            return "factory_value"

        result = _if_sentinel("a", "b", default="d", default_factory=factory)
        assert result == "a"
        assert factory_called["count"] == 0
