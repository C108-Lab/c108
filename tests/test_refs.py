#
# C108 - Validators Tests
#

# Standard library -----------------------------------------------------------------------------------------------------

# Third party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.refs.schemes import SchemeGroup, Scheme


# Tests ----------------------------------------------------------------------------------------------------------------


class TestSchemeGroupAll:
    @pytest.mark.parametrize(
        "attrs, expected",
        [
            pytest.param(
                {"HTTP": "http", "HTTPS": "https"},
                ("http", "https"),
                id="flat-strings",
            ),
            pytest.param(
                {"_PRIVATE": "x", "VISIBLE": "v"},
                ("v",),
                id="ignore-private",
            ),
            pytest.param(
                {
                    "A": "a",
                    "Sub": type(
                        "Sub",
                        (SchemeGroup,),
                        {
                            "B": "b",
                            "C": "c",
                        },
                    ),
                },
                ("a", "b", "c"),
                id="nested-group",
            ),
            pytest.param(
                {
                    "A": "a",
                    "Sub1": type(
                        "Sub1",
                        (SchemeGroup,),
                        {
                            "B": "b",
                            "Sub2": type(
                                "Sub2",
                                (SchemeGroup,),
                                {
                                    "C": "c",
                                },
                            ),
                        },
                    ),
                },
                ("a", "b", "c"),
                id="deep-nested-groups",
            ),
        ],
    )
    def test_all_collects_expected(self, attrs, expected):
        """Collect expected schemes across attributes and nested groups."""
        Dynamic = type("Dynamic", (SchemeGroup,), attrs)
        assert Dynamic.all == expected

    @pytest.mark.parametrize(
        "attrs",
        [
            pytest.param({"X": 1, "Y": object()}, id="non-string-non-group"),
            pytest.param(
                {
                    "A": "a",
                    "Weird": type("Weird", (), {"Z": "z"}),  # not a SchemeGroup subclass
                },
                id="ignore-non-subclass-type",
            ),
        ],
    )
    def test_all_ignores_unrelated_members(self, attrs):
        """Ignore attributes that are neither strings nor SchemeGroup subclasses."""
        Dynamic = type("Dynamic", (SchemeGroup,), attrs)
        assert Dynamic.all == tuple(s for s in attrs.values() if isinstance(s, str))
