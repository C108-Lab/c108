#
# C108 - Sentinels Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import pickle

# Third-party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.sentinels import (
    UNSET,
    MISSING,
    DEFAULT,
    NOT_FOUND,
    STOP,
    SentinelBase,
    UnsetType,
    MissingType,
    DefaultType,
    NotFoundType,
    StopType,
    create_sentinel_type,
    ifnotdefault,
    ifnotmissing,
    ifnotnone,
    iffound,
    ifnotstop,
    ifnotunset,
)


# Local Classes & Methods ----------------------------------------------------------------------------------------------


class TestCreateSentinelType:
    def test_type_creation(self):
        """Create a sentinel type with explicit name and falsy behavior."""
        T = create_sentinel_type(name="UNSET", is_truthy=False)
        assert issubclass(T, SentinelBase)
        assert T.__name__ == "UnsetType"
        assert T.__qualname__ == "UnsetType"

    def test_singleton_behavior(self):
        """Ensure instances of the sentinel type are singletons."""
        T = create_sentinel_type(name="MISSING", is_truthy=False)
        a = T()
        b = T()
        assert a is b
        assert a == b
        assert hash(a) == hash(b)

    def test_repr_and_name(self):
        """Verify repr uses provided name."""
        T = create_sentinel_type(name="PENDING", is_truthy=False)
        s = T()
        assert repr(s) == "<PENDING>"

    def test_truthiness_false(self):
        """Assert falsy sentinel when is_truthy is False."""
        T = create_sentinel_type(name="OFF", is_truthy=False)
        s = T()
        assert bool(s) is False

    def test_truthiness_true(self):
        """Assert truthy sentinel when is_truthy is True."""
        T = create_sentinel_type(name="ACTIVE", is_truthy=True)
        s = T()
        assert bool(s) is True

    def test_pickle_raises_error(self):
        """Sentinels should not be pickleable."""
        with pytest.raises(TypeError, match="cannot be pickled"):
            T = create_sentinel_type("CUSTOM")
            pickle.dumps(T())

    def test_identity_equality_only(self):
        """Compare only by identity, not value."""
        T = create_sentinel_type(name="TOKEN", is_truthy=False)
        s = T()
        assert (s == s) is True
        assert (s == object()) is False  # type: ignore[comparison-overlap]
        assert (s != s) is False

    @pytest.mark.parametrize(
        "name, expected",
        [
            pytest.param("UNSET", "UnsetType", id="upper_single"),
            pytest.param("multi_word", "MultiWordType", id="snake_case"),
            pytest.param("alreadyCaps", "AlreadycapsType", id="mixed_caps"),
            pytest.param("x", "XType", id="single_char"),
        ],
    )
    def test_class_naming(self, name, expected):
        """Generate class name based on input name."""
        T = create_sentinel_type(name=name, is_truthy=False)
        assert T.__name__ == expected
        assert T.__qualname__ == expected

    def test_custom_docstring(self):
        """Set docstring with sentinel name."""
        T = create_sentinel_type(name="DOC_TEST", is_truthy=False)
        assert T.__doc__ == "Sentinel type for DOC_TEST."

    def test_init_not_overwriting_name_on_reinit(self):
        """Preserve name across repeated __init__ calls."""
        T = create_sentinel_type(name="REINIT", is_truthy=False)
        s = T()
        # Force another __init__ call by constructing again
        _ = T()
        assert repr(s) == "<REINIT>"

    def test_hash_is_stable(self):
        """Ensure hash is stable across calls."""
        T = create_sentinel_type(name="STABLE", is_truthy=False)
        s = T()
        h1 = hash(s)
        h2 = hash(s)
        assert h1 == h2

    def test_invalid_equality_with_other_type(self):
        """Ensure equality with non-sentinel is False."""
        T = create_sentinel_type(name="CMP", is_truthy=False)
        s = T()
        assert (s == 0) is False  # type: ignore[comparison-overlap]


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

    def test_reduce_raises_error(self):
        """__reduce__ should raise TypeError."""
        with pytest.raises(TypeError, match="cannot be pickled"):
            UNSET.__reduce__()

    def test_equality_with_non_sentinel(self):
        """Ensure equality with non-sentinel is false."""
        assert (UNSET == object()) is False  # type: ignore[comparison-overlap]
        assert (MISSING == None) is False  # noqa: E711  # type: ignore[comparison-overlap]


class TestIfWrappers:
    """Tests for public sentinel wrapper functions (ifnotunset, ifnotmissing, ifnotdefault, iffound, ifnotstop, ifnotnone)."""

    @pytest.mark.parametrize(
        "func,sentinel,value,default,default_factory,expected",
        [
            pytest.param(ifnotunset, UNSET, "x", "d", None, "x", id="ifunset_not_sentinel"),
            pytest.param(ifnotunset, UNSET, UNSET, "d", None, "d", id="ifunset_default"),
            pytest.param(ifnotunset, UNSET, UNSET, None, lambda: "f", "f", id="ifunset_factory"),
            pytest.param(ifnotmissing, MISSING, "x", "d", None, "x", id="ifmissing_not_sentinel"),
            pytest.param(ifnotmissing, MISSING, MISSING, "d", None, "d", id="ifmissing_default"),
            pytest.param(
                ifnotmissing,
                MISSING,
                MISSING,
                None,
                lambda: "f",
                "f",
                id="ifmissing_factory",
            ),
            pytest.param(ifnotdefault, DEFAULT, "x", "d", None, "x", id="ifdefault_not_sentinel"),
            pytest.param(ifnotdefault, DEFAULT, DEFAULT, "d", None, "d", id="ifdefault_default"),
            pytest.param(
                ifnotdefault,
                DEFAULT,
                DEFAULT,
                None,
                lambda: "f",
                "f",
                id="ifdefault_factory",
            ),
            pytest.param(iffound, NOT_FOUND, "x", "d", None, "x", id="ifnotfound_not_sentinel"),
            pytest.param(iffound, NOT_FOUND, NOT_FOUND, "d", None, "d", id="ifnotfound_default"),
            pytest.param(
                iffound,
                NOT_FOUND,
                NOT_FOUND,
                None,
                lambda: "f",
                "f",
                id="ifnotfound_factory",
            ),
            pytest.param(ifnotstop, STOP, "x", "d", None, "x", id="ifstop_not_sentinel"),
            pytest.param(ifnotstop, STOP, STOP, "d", None, "d", id="ifstop_default"),
            pytest.param(ifnotstop, STOP, STOP, None, lambda: "f", "f", id="ifstop_factory"),
            pytest.param(ifnotnone, None, "x", "d", None, "x", id="ifnotnone_not_sentinel"),
            pytest.param(ifnotnone, None, None, "d", None, "d", id="ifnotnone_default"),
            pytest.param(ifnotnone, None, None, None, lambda: "f", "f", id="ifnotnone_factory"),
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
            pytest.param(ifnotunset, UNSET, UNSET, id="ifnotunset"),
            pytest.param(ifnotmissing, MISSING, MISSING, id="ifnotmissing"),
            pytest.param(ifnotdefault, DEFAULT, DEFAULT, id="ifnotdefault"),
            pytest.param(iffound, NOT_FOUND, NOT_FOUND, id="iffound"),
            pytest.param(ifnotstop, STOP, STOP, id="ifnotstop"),
            pytest.param(ifnotnone, None, None, id="ifnotnone"),
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

        result = ifnotunset("x", default_factory=factory)
        assert result == "x"
        assert called["count"] == 0

    @pytest.mark.parametrize(
        "func,sentinel",
        [
            pytest.param(ifnotunset, UNSET, id="ifnotunset"),
            pytest.param(ifnotmissing, MISSING, id="ifnotmissing"),
            pytest.param(ifnotdefault, DEFAULT, id="ifnotdefault"),
            pytest.param(iffound, NOT_FOUND, id="iffound"),
            pytest.param(ifnotstop, STOP, id="ifnotstop"),
            pytest.param(ifnotnone, None, id="ifnotnone"),
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
