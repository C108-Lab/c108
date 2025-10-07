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
