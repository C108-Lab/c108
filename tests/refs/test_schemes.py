#
# C108 - Ref Schemes Tests
#

# Standard library -----------------------------------------------------------------------------------------------------

# Third party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.refs.schemes import (
    AWSDatabase,
    AWSStorage,
    Analytical,
    AzureDatabase,
    AzureStorage,
    DataVersioning,
    Distributed,
    GCPDatabase,
    GCPStorage,
    Graph,
    Hadoop,
    Lakehouse,
    Local,
    MLDataset,
    MLFlow,
    MLHub,
    MLTracking,
    NetworkFS,
    NoSQL,
    SQL,
    SchemeBase,
    Schemes,
    Search,
    TimeSeries,
    Vector,
    Web,
)


# Tests ----------------------------------------------------------------------------------------------------------------


class TestSchemeBase:
    def test_all_collects_flat_strings(self):
        """Collect plain string attributes into .all."""
        cls = type("T", (SchemeBase,), {"HTTP": "http", "HTTPS": "https"})
        assert cls.all == ("http", "https")

    def test_all_collects_nested_group(self):
        """Recurse into nested SchemeBase subclasses."""
        Sub = type("Sub", (SchemeBase,), {"B": "b", "C": "c"})
        cls = type("T", (SchemeBase,), {"A": "a", "Sub": Sub})
        assert cls.all == ("a", "b", "c")

    def test_all_collects_deeply_nested_groups(self):
        """Recurse into multiple levels of nesting."""
        Sub2 = type("Sub2", (SchemeBase,), {"C": "c"})
        Sub1 = type("Sub1", (SchemeBase,), {"B": "b", "Sub2": Sub2})
        cls = type("T", (SchemeBase,), {"A": "a", "Sub1": Sub1})
        assert cls.all == ("a", "b", "c")

    def test_all_ignores_private_attrs(self):
        """Exclude private attributes from .all."""
        cls = type("T", (SchemeBase,), {"_PRIVATE": "x", "VISIBLE": "v"})
        assert cls.all == ("v",)

    def test_all_ignores_non_string_non_group_attrs(self):
        """Ignore attributes that are not strings or SchemeBase subclasses."""
        cls = type("T", (SchemeBase,), {"X": 1, "Y": object()})
        assert cls.all == ()

    def test_all_ignores_non_schemebase_type(self):
        """Ignore nested types that are not SchemeBase subclasses."""
        Unrelated = type("Unrelated", (), {"Z": "z"})
        cls = type("T", (SchemeBase,), {"A": "a", "Unrelated": Unrelated})
        assert cls.all == ("a",)


class TestLeafSchemeIntegrity:
    def test_all_schemes_are_non_empty_strings(self):
        """Verify every scheme constant is a non-empty string."""
        leaf_classes = [
            AWSStorage,
            AWSDatabase,
            AzureStorage,
            AzureDatabase,
            GCPStorage,
            GCPDatabase,
            SQL,
            NoSQL,
            Vector,
            Graph,
            Analytical,
            TimeSeries,
            Search,
            MLTracking,
            MLHub,
            MLDataset,
            DataVersioning,
            MLFlow,
            Distributed,
            Hadoop,
            Lakehouse,
            NetworkFS,
            Local,
            Web,
        ]
        for cls in leaf_classes:
            for scheme in cls.all:
                assert isinstance(scheme, str) and scheme.strip(), (
                    f"Invalid scheme {scheme!r} in {cls.__name__}"
                )

    def test_all_schemes_are_lowercase(self):
        """Verify every scheme constant is fully lowercase."""
        leaf_classes = [
            AWSStorage,
            AWSDatabase,
            AzureStorage,
            AzureDatabase,
            GCPStorage,
            GCPDatabase,
            SQL,
            NoSQL,
            Vector,
            Graph,
            Analytical,
            TimeSeries,
            Search,
            MLTracking,
            MLHub,
            MLDataset,
            DataVersioning,
            MLFlow,
            Distributed,
            Hadoop,
            Lakehouse,
            NetworkFS,
            Local,
            Web,
        ]
        for cls in leaf_classes:
            for scheme in cls.all:
                assert scheme == scheme.lower(), (
                    f"Non-lowercase scheme {scheme!r} in {cls.__name__}"
                )

    def test_all_schemes_contain_only_valid_uri_chars(self):
        """Verify every scheme uses only letters, digits, plus, hyphen, or dot (RFC 3986)."""
        import re

        pattern = re.compile(r"^[a-z][a-z0-9+\-.]*$")
        leaf_classes = [
            AWSStorage,
            AWSDatabase,
            AzureStorage,
            AzureDatabase,
            GCPStorage,
            GCPDatabase,
            SQL,
            NoSQL,
            Vector,
            Graph,
            Analytical,
            TimeSeries,
            Search,
            MLTracking,
            MLHub,
            MLDataset,
            DataVersioning,
            MLFlow,
            Distributed,
            Hadoop,
            Lakehouse,
            NetworkFS,
            Local,
            Web,
        ]
        for cls in leaf_classes:
            for scheme in cls.all:
                assert pattern.match(scheme), (
                    f"Invalid URI scheme chars in {scheme!r} ({cls.__name__})"
                )

    def test_no_duplicates_within_each_leaf_class(self):
        """Verify no scheme is duplicated within a single leaf class."""
        leaf_classes = [
            AWSStorage,
            AWSDatabase,
            AzureStorage,
            AzureDatabase,
            GCPStorage,
            GCPDatabase,
            SQL,
            NoSQL,
            Vector,
            Graph,
            Analytical,
            TimeSeries,
            Search,
            MLTracking,
            MLHub,
            MLDataset,
            DataVersioning,
            MLFlow,
            Distributed,
            Hadoop,
            Lakehouse,
            NetworkFS,
            Local,
            Web,
        ]
        for cls in leaf_classes:
            seen = set()
            for scheme in cls.all:
                assert scheme not in seen, f"Duplicate scheme {scheme!r} in {cls.__name__}"
                seen.add(scheme)

    def test_each_leaf_class_is_non_empty(self):
        """Verify every leaf class exposes at least one scheme."""
        leaf_classes = [
            AWSStorage,
            AWSDatabase,
            AzureStorage,
            AzureDatabase,
            GCPStorage,
            GCPDatabase,
            SQL,
            NoSQL,
            Vector,
            Graph,
            Analytical,
            TimeSeries,
            Search,
            MLTracking,
            MLHub,
            MLDataset,
            DataVersioning,
            MLFlow,
            Distributed,
            Hadoop,
            Lakehouse,
            NetworkFS,
            Local,
            Web,
        ]
        for cls in leaf_classes:
            assert len(cls.all) > 0, f"{cls.__name__}.all is empty"


class TestSchemes:
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
                        (SchemeBase,),
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
                        (SchemeBase,),
                        {
                            "B": "b",
                            "Sub2": type(
                                "Sub2",
                                (SchemeBase,),
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
    def test_scheme_base_all_collects_expected(self, attrs, expected):
        """Collect schemes from strings and nested groups."""
        dynamic_cls = type("Dynamic", (SchemeBase,), attrs)
        assert dynamic_cls.all == expected

    @pytest.mark.parametrize(
        "attrs",
        [
            pytest.param({"X": 1, "Y": object()}, id="non-string-non-group"),
            pytest.param(
                {
                    "A": "a",
                    "Weird": type("Weird", (), {"Z": "z"}),  # not a SchemeBase subclass
                },
                id="ignore-non-subclass-type",
            ),
        ],
    )
    def test_scheme_base_all_ignores_unrelated_members(self, attrs):
        """Ignore members that are not strings or SchemeBase subclasses."""
        dynamic_cls = type("Dynamic", (SchemeBase,), attrs)
        assert dynamic_cls.all == tuple(value for value in attrs.values() if isinstance(value, str))

    @pytest.mark.parametrize(
        "actual, expected",
        [
            pytest.param(Schemes.aws.all, (*AWSStorage.all, *AWSDatabase.all), id="aws-all"),
            pytest.param(
                Schemes.azure.all, (*AzureStorage.all, *AzureDatabase.all), id="azure-all"
            ),
            pytest.param(Schemes.gcp.all, (*GCPStorage.all, *GCPDatabase.all), id="gcp-all"),
            pytest.param(
                Schemes.db.all,
                (
                    *SQL.all,
                    *NoSQL.all,
                    *Vector.all,
                    *Graph.all,
                    *Analytical.all,
                    *TimeSeries.all,
                    *Search.all,
                    *AWSDatabase.all,
                    *AzureDatabase.all,
                    *GCPDatabase.all,
                ),
                id="db-all",
            ),
            pytest.param(
                Schemes.ml.all,
                (*MLTracking.all, *MLHub.all, *MLDataset.all, *DataVersioning.all, *MLFlow.all),
                id="ml-all",
            ),
            pytest.param(
                Schemes.cloud,
                (*AWSStorage.all, *AzureStorage.all, *GCPStorage.all),
                id="cloud-all",
            ),
            pytest.param(
                Schemes.distributed,
                (*Distributed.all, *Hadoop.all, *Lakehouse.all),
                id="distributed-all",
            ),
            pytest.param(
                Schemes.all,
                (
                    *AWSStorage.all,
                    *AWSDatabase.all,
                    *AzureStorage.all,
                    *AzureDatabase.all,
                    *GCPStorage.all,
                    *GCPDatabase.all,
                    *SQL.all,
                    *NoSQL.all,
                    *Vector.all,
                    *Graph.all,
                    *Analytical.all,
                    *TimeSeries.all,
                    *Search.all,
                    *MLTracking.all,
                    *MLHub.all,
                    *MLDataset.all,
                    *DataVersioning.all,
                    *MLFlow.all,
                    *Distributed.all,
                    *Hadoop.all,
                    *Lakehouse.all,
                    *NetworkFS.all,
                    *Local.all,
                    *Web.all,
                ),
                id="schemes-all",
            ),
        ],
    )
    def test_schemes_composition(self, actual, expected):
        """Match collection composition in declared category order."""
        assert actual == expected

    @pytest.mark.parametrize(
        "collection, sample",
        [
            pytest.param(Schemes.ml.tracking, "wandb", id="ml-tracking-sample"),
            pytest.param(Schemes.db.vector, "pinecone", id="db-vector-sample"),
            pytest.param(Schemes.db.cloud, "bigquery", id="db-cloud-sample"),
            pytest.param(Schemes.distributed, "hdfs", id="distributed-sample"),
        ],
    )
    def test_schemes_representative_values(self, collection, sample):
        """Contain representative constants from each major subset."""
        assert sample in collection
