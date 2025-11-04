#
# C108 - Validators Tests
#

# Standard library -----------------------------------------------------------------------------------------------------

# Third party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.validators import (
    validate_categorical,
    validate_email,
    validate_ip_address,
    validate_language_code,
    validate_not_empty,
    validate_shape,
    validate_uri,
)
from c108.validators.schemes import SchemeGroup, Scheme


# Classes --------------------------------------------------------------------------------------------------------------


class DummyArray:
    """Simple mock for sizable objects with shape and size attributes."""

    def __init__(self, shape: tuple[int, ...], size: int = 1):
        self.shape = shape
        self.size = size


class DummyTensor:
    """Simple mock for tensor-like objects with shape and numel method."""

    def __init__(self, shape: tuple[int, ...], numel_value: int):
        self.shape = shape
        self._numel_value = numel_value

    def numel(self) -> int:
        return self._numel_value


class DummyPandasLike:
    """Simple mock for pandas-like objects with empty attribute."""

    def __init__(self, empty: bool):
        self.empty = empty


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
                    "Weird": type(
                        "Weird", (), {"Z": "z"}
                    ),  # not a SchemeGroup subclass
                },
                id="ignore-non-subclass-type",
            ),
        ],
    )
    def test_all_ignores_unrelated_members(self, attrs):
        """Ignore attributes that are neither strings nor SchemeGroup subclasses."""
        Dynamic = type("Dynamic", (SchemeGroup,), attrs)
        assert Dynamic.all == tuple(s for s in attrs.values() if isinstance(s, str))


class TestValidateCategorical:
    """Test suite for validate_categorical function."""

    @pytest.mark.parametrize(
        "value,categories,expected",
        [
            pytest.param("red", ["red", "green", "blue"], "red", id="basic_match"),
            pytest.param(
                "  green  ", ["red", "green", "blue"], "green", id="strip_whitespace"
            ),
        ],
    )
    def test_valid_basic(self, value, categories, expected):
        """Validate correct values return expected normalized output."""
        result = validate_categorical(value, categories=categories)
        assert result == expected

    @pytest.mark.parametrize(
        "value,categories,expected",
        [
            pytest.param(
                "RED", ["red", "green", "blue"], "RED", id="case_insensitive_upper"
            ),
            pytest.param(
                "Blue", ("red", "green", "blue"), "Blue", id="case_insensitive_tuple"
            ),
        ],
    )
    def test_valid_case_insensitive(self, value, categories, expected):
        """Validate case-insensitive matching preserves original casing."""
        result = validate_categorical(value, categories=categories, case=False)
        assert result == expected

    def test_invalid_value_raises_valueerror(self):
        """Raise ValueError for value not in categories."""
        with pytest.raises(ValueError, match=r"(?i).*invalid value.*allowed.*"):
            validate_categorical("yellow", categories=["red", "green", "blue"])

    def test_empty_categories_raises_valueerror(self):
        """Raise ValueError when categories is empty."""
        with pytest.raises(ValueError, match=r"(?i).*categories cannot be empty.*"):
            validate_categorical("red", categories=[])

    @pytest.mark.parametrize(
        "bad_value",
        [
            pytest.param(None, id="none_value"),
            pytest.param(123, id="non_string_value"),
        ],
    )
    def test_invalid_value_type_raises_typeerror(self, bad_value):
        """Raise TypeError for non-string or None value."""
        with pytest.raises(TypeError, match=r"(?i).*value must be a string.*"):
            validate_categorical(bad_value, categories=["red", "green"])

    def test_non_string_in_categories_raises_typeerror(self):
        """Raise TypeError when categories contain non-string elements."""
        with pytest.raises(
            TypeError, match=r"(?i).*categories must contain only strings.*"
        ):
            validate_categorical("red", categories=["red", 42, "blue"])

    def test_non_iterable_categories_raises_typeerror(self):
        """Raise TypeError when categories is not iterable."""
        with pytest.raises(TypeError, match=r"(?i).*categories must be iterable.*"):
            validate_categorical("red", categories=None)

    def test_strip_false_preserves_whitespace(self):
        """Validate strip=False preserves whitespace and fails if not exact match."""
        with pytest.raises(ValueError, match=r"(?i).*invalid value.*allowed.*"):
            validate_categorical(
                "  red  ", categories=["red", "green", "blue"], strip=False
            )

    def test_tuple_and_set_categories_equivalence(self):
        """Validate tuple and set categories behave equivalently."""
        assert (
            validate_categorical("green", categories=("red", "green", "blue"))
            == "green"
        )
        assert (
            validate_categorical("green", categories={"red", "green", "blue"})
            == "green"
        )


class TestValidateEmail:
    @pytest.mark.parametrize(
        "email,strip,lowercase,expected",
        [
            pytest.param(
                "User@Example.COM",
                True,
                True,
                "user@example.com",
                id="normalize-lowercase",
            ),
            pytest.param(
                "  user@example.com  ",
                True,
                True,
                "user@example.com",
                id="strip-and-lower",
            ),
            pytest.param(
                "User@Example.COM", True, False, "User@Example.COM", id="preserve-case"
            ),
        ],
    )
    def test_ok_variants(self, email, strip, lowercase, expected):
        """Validate and normalize when options explicitly set."""
        assert validate_email(email, strip=strip, lowercase=lowercase) == expected

    def test_strip_disabled_requires_exact_whitespace(self):
        """Reject when strip disabled and whitespace present."""
        with pytest.raises(ValueError, match=r"(?i).*leading or trailing whitespace.*"):
            validate_email("  test@example.com  ", strip=False, lowercase=True)

    @pytest.mark.parametrize(
        "email",
        [
            pytest.param("", id="empty-string"),
            pytest.param("   ", id="spaces-only"),
        ],
    )
    def test_empty_after_processing(self, email):
        """Reject empty after explicit stripping."""
        with pytest.raises(ValueError, match=r"(?i).*empty.*"):
            validate_email(email, strip=True, lowercase=True)

    @pytest.mark.parametrize(
        "email",
        [
            pytest.param("invalid.email", id="no-at"),
            pytest.param("user@", id="missing-domain"),
            pytest.param("@example.com", id="missing-local"),
        ],
    )
    def test_invalid_formats(self, email):
        """Reject structurally invalid formats."""
        with pytest.raises(ValueError, match=r"(?i).*(missing|invalid).*"):
            validate_email(email, strip=True, lowercase=True)

    def test_local_part_length_limit(self):
        """Reject when local part exceeds 64 chars."""
        long_local = "a" * 65 + "@example.com"
        with pytest.raises(ValueError, match=r"(?i).*exceeds maximum length.*64.*"):
            validate_email(long_local, strip=True, lowercase=True)

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(None, id="none"),
            pytest.param(123, id="int"),
            pytest.param(b"user@example.com", id="bytes"),
            pytest.param(["user@example.com"], id="list"),
        ],
    )
    def test_type_errors(self, value):
        """Reject non-string inputs with type error."""
        with pytest.raises(TypeError, match=r"(?i).*Email must be a string.*"):
            validate_email(value, strip=True, lowercase=True)


class TestValidateIpAddress:
    """Test suite for validate_ip_address function."""

    @pytest.mark.parametrize(
        "ip, version, expected",
        [
            pytest.param("192.168.1.1", 4, "192.168.1.1", id="ipv4-basic"),
            pytest.param("10.0.0.1", 4, "10.0.0.1", id="ipv4-private"),
            pytest.param("::1", 6, "::1", id="ipv6-loopback"),
            pytest.param("2001:db8::1", 6, "2001:db8::1", id="ipv6-global"),
        ],
    )
    def test_valid_ips(self, ip: str, version: int, expected: str):
        """Validate correct IPv4 and IPv6 addresses."""
        result = validate_ip_address(ip, version=version)
        assert result == expected

    @pytest.mark.parametrize(
        "ip, version, expected",
        [
            pytest.param(" 192.168.0.1 ", 4, "192.168.0.1", id="ipv4-strip"),
            pytest.param("\tfe80::1\n", 6, "fe80::1", id="ipv6-strip"),
        ],
    )
    def test_strip_whitespace(self, ip: str, version: int, expected: str):
        """Strip whitespace before validation."""
        result = validate_ip_address(ip, version=version, strip=True)
        assert result == expected

    @pytest.mark.parametrize(
        "ip, version",
        [
            pytest.param("192.168.001.001", 4, id="ipv4-leading-zeros"),
        ],
    )
    def test_allow_leading_zeros(self, ip: str, version: int):
        """Allow IPv4 leading zeros when enabled."""
        result = validate_ip_address(ip, version=version, leading_zeros=True)
        assert result == "192.168.1.1"

    @pytest.mark.parametrize(
        "ip, version",
        [
            pytest.param("192.168.001.001", 4, id="ipv4-leading-zeros-disabled"),
        ],
    )
    def test_reject_leading_zeros(self, ip: str, version: int):
        """Reject IPv4 leading zeros when disabled."""
        with pytest.raises(ValueError, match=r"(?i).*invalid.*"):
            validate_ip_address(ip, version=version, leading_zeros=False)

    @pytest.mark.parametrize(
        "ip, version",
        [
            pytest.param("192.168.1.1", 6, id="ipv4-as-ipv6"),
            pytest.param("::1", 4, id="ipv6-as-ipv4"),
        ],
    )
    def test_version_mismatch(self, ip: str, version: int):
        """Reject IPs that do not match required version."""
        with pytest.raises(ValueError, match=r"(?i).*version.*"):
            validate_ip_address(ip, version=version)

    @pytest.mark.parametrize(
        "ip",
        [
            pytest.param("not.an.ip", id="nonsense"),
            pytest.param("192.168.1", id="incomplete-ipv4"),
            pytest.param("gggg::1", id="invalid-ipv6"),
        ],
    )
    def test_invalid_format(self, ip: str):
        """Reject invalid IP address formats."""
        with pytest.raises(ValueError, match=r"(?i).*invalid.*"):
            validate_ip_address(ip)

    @pytest.mark.parametrize(
        "ip",
        [
            pytest.param("", id="empty-string"),
            pytest.param("   ", id="whitespace-only"),
        ],
    )
    def test_empty_input(self, ip: str):
        """Reject empty or whitespace-only input."""
        with pytest.raises(ValueError, match=r"(?i).*empty.*"):
            validate_ip_address(ip)

    @pytest.mark.parametrize(
        "ip",
        [
            pytest.param(12345, id="non-string-int"),
            pytest.param(None, id="non-string-none"),
        ],
    )
    def test_non_string_input(self, ip):
        """Reject non-string input types."""
        with pytest.raises(TypeError, match=r"(?i).*string.*"):
            validate_ip_address(ip)

    def test_invalid_version_type(self):
        """Reject invalid version argument."""
        with pytest.raises(TypeError, match=r"(?i).*version.*"):
            validate_ip_address("192.168.1.1", version="ANY")


class TestValidateLanguageCode:
    """Test suite for validate_language_code function."""

    @pytest.mark.parametrize(
        "language_code,expected",
        [
            pytest.param("en", "en", id="iso639_1_lowercase"),
            pytest.param("EN", "en", id="iso639_1_uppercase"),
            pytest.param(" fr ", "fr", id="iso639_1_with_whitespace"),
        ],
    )
    def test_valid_iso639_1_codes(self, language_code: str, expected: str) -> None:
        """Validate ISO 639-1 codes."""
        result = validate_language_code(
            language_code, allow_iso639_1=True, allow_bcp47=False
        )
        assert result == expected

    @pytest.mark.parametrize(
        "language_code,bcp47_parts,expected",
        [
            pytest.param(
                "en-US", "language-region", "en-us", id="bcp47_language_region"
            ),
            pytest.param(
                "zh-Hans", "language-script", "zh-hans", id="bcp47_language_script"
            ),
            pytest.param(
                "zh-Hans-CN",
                "language-script-region",
                "zh-hans-cn",
                id="bcp47_language_script_region",
            ),
        ],
    )
    def test_valid_bcp47_codes(
        self, language_code: str, bcp47_parts: str, expected: str
    ) -> None:
        """Validate BCP 47 codes with different part structures."""
        result = validate_language_code(
            language_code,
            allow_iso639_1=False,
            allow_bcp47=True,
            bcp47_parts=bcp47_parts,
            strict=False,
        )
        assert result == expected

    def test_case_sensitive_preserves_case(self) -> None:
        """Preserve case when case_sensitive=True."""
        result = validate_language_code(
            "EN-US", allow_bcp47=True, case_sensitive=True, strict=False
        )
        assert result == "EN-US"

    def test_invalid_type_raises_typeerror(self) -> None:
        """Raise TypeError when input is not a string."""
        with pytest.raises(TypeError, match=r"(?i).*str.*"):
            validate_language_code(123)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "language_code",
        [
            pytest.param("", id="empty_string"),
            pytest.param("   ", id="whitespace_only"),
        ],
    )
    def test_empty_or_whitespace_raises_valueerror(self, language_code: str) -> None:
        """Raise ValueError for empty or whitespace-only input."""
        with pytest.raises(ValueError, match=r"(?i).*empty.*"):
            validate_language_code(language_code)

    def test_invalid_format_raises_valueerror(self) -> None:
        """Raise ValueError for invalid format."""
        with pytest.raises(ValueError, match=r"(?i).*invalid.*"):
            validate_language_code("english")

    def test_disallow_iso639_1_raises_valueerror(self) -> None:
        """Raise ValueError when ISO 639-1 code is disallowed."""
        with pytest.raises(ValueError, match=r"(?i).*not allowed.*"):
            validate_language_code("en", allow_iso639_1=False, allow_bcp47=True)

    def test_disallow_bcp47_raises_valueerror(self) -> None:
        """Raise ValueError when BCP 47 code is disallowed."""
        with pytest.raises(ValueError, match=r"(?i).*not allowed.*"):
            validate_language_code("en-US", allow_iso639_1=True, allow_bcp47=False)

    def test_strict_mode_rejects_unknown_code(self) -> None:
        """Raise ValueError for unknown code in strict mode."""
        with pytest.raises(ValueError, match=r"(?i).*(invalid|unknown).*"):
            validate_language_code("xx", strict=True)

    def test_non_strict_accepts_unknown_code(self) -> None:
        """Accept unknown code when strict=False."""
        result = validate_language_code("xx", strict=False)
        assert result == "xx"


class TestValidateNotEmpty:
    """Core tests for validate_not_empty function."""

    @pytest.mark.parametrize(
        "collection",
        [
            pytest.param([1, 2], id="list"),
            pytest.param((1,), id="tuple"),
            pytest.param({1, 2}, id="set"),
            pytest.param({"a": 1}, id="dict"),
            pytest.param(frozenset({1}), id="frozenset"),
        ],
    )
    def test_valid_collections(self, collection):
        """Return same collection when non-empty."""
        result = validate_not_empty(collection)
        assert result is collection

    @pytest.mark.parametrize(
        "collection, name",
        [
            pytest.param([], "empty_list", id="list"),
            pytest.param({}, "empty_dict", id="dict"),
            pytest.param(set(), "empty_set", id="set"),
            pytest.param((), "empty_tuple", id="tuple"),
            pytest.param(frozenset(), "empty_frozenset", id="frozenset"),
        ],
    )
    def test_empty_collections_raise_value_error(self, collection, name):
        """Raise ValueError for empty standard collections."""
        with pytest.raises(ValueError, match=rf"(?i){name}.*must not be empty"):
            validate_not_empty(collection, name=name)

    @pytest.mark.parametrize(
        "collection",
        [
            pytest.param(DummyArray((2, 2), 4), id="array_non_empty"),
            pytest.param(DummyTensor((3,), 3), id="tensor_non_empty"),
            pytest.param(DummyPandasLike(False), id="pandas_non_empty"),
        ],
    )
    def test_non_empty_custom_like_objects(self, collection):
        """Return same object for non-empty custom-like objects."""
        result = validate_not_empty(collection)
        assert result is collection

    @pytest.mark.parametrize(
        "collection",
        [
            pytest.param(DummyArray((0,), 0), id="array_empty"),
            pytest.param(DummyTensor((0,), 0), id="tensor_empty"),
            pytest.param(DummyPandasLike(True), id="pandas_empty"),
        ],
    )
    def test_empty_custom_like_objects_raise_value_error(self, collection):
        """Raise ValueError for empty custom-like objects."""
        with pytest.raises(ValueError, match=r"(?i)must not be empty"):
            validate_not_empty(collection)

    def test_none_raises_type_error(self):
        """Raise TypeError when collection is None."""
        with pytest.raises(TypeError, match=r"(?i)cannot be None"):
            validate_not_empty(None)

    @pytest.mark.parametrize(
        "collection",
        [
            pytest.param("abc", id="string"),
            pytest.param(b"bytes", id="bytes"),
        ],
    )
    def test_strings_raise_type_error(self, collection):
        """Raise TypeError for string or bytes input."""
        with pytest.raises(TypeError, match=r"(?i)strings.*not supported"):
            validate_not_empty(collection)

    def test_generator_raises_type_error(self):
        """Raise TypeError for generator input."""
        gen = (x for x in range(3))
        with pytest.raises(TypeError, match=r"(?i)generators.*not supported"):
            validate_not_empty(gen)

    def test_non_collection_type_raises_type_error(self):
        """Raise TypeError for unsupported non-collection type."""
        with pytest.raises(
            TypeError, match=r"(?i)collection.*must be a collection type"
        ):
            validate_not_empty(123)

    def test_object_with_len_but_invalid_raises_type_error(self):
        """Raise TypeError when __len__ raises internally."""

        class BadLen:
            def __len__(self):
                raise TypeError("bad len")

        with pytest.raises(
            TypeError, match=r"(?i)collection.*must be a collection type"
        ):
            validate_not_empty(BadLen())


class TestValidateURI:
    """Test suite for core logic of validate_uri()."""

    @pytest.mark.parametrize(
        "uri,schemes,expected",
        [
            pytest.param(
                "https://example.com",
                ["https"],
                "https://example.com",
                id="https_basic",
            ),
            pytest.param("s3://bucket/path", ["s3"], "s3://bucket/path", id="s3_basic"),
            pytest.param(
                "file:///tmp/data.csv",
                ["file"],
                "file:///tmp/data.csv",
                id="file_scheme",
            ),
        ],
    )
    def test_valid_basic_uris(self, uri, schemes, expected):
        """Validate that basic URIs with allowed schemes pass."""
        result = validate_uri(uri, schemes=schemes)
        assert result == expected

    @pytest.mark.parametrize(
        "uri,schemes",
        [
            pytest.param("https://example.com", ["http"], id="unsupported_scheme"),
            pytest.param("ftp://example.com", ["https"], id="ftp_not_allowed"),
        ],
    )
    def test_invalid_scheme(self, uri, schemes):
        """Raise ValueError for unsupported schemes."""
        with pytest.raises(ValueError, match=r"(?i).*unsupported uri scheme.*"):
            validate_uri(uri, schemes=schemes)

    def test_invalid_uri_format_raises(self):
        """Raise ValueError for malformed URI."""
        with pytest.raises(ValueError, match=r"(?i).*invalid uri format.*"):
            validate_uri("://invalid.uri", schemes=["https"])

    def test_missing_scheme_raises(self):
        """Raise ValueError when URI has no scheme."""
        with pytest.raises(ValueError, match=r"(?i).*missing or invalid scheme.*"):
            validate_uri("example.com/path", schemes=["https"])

    def test_non_string_uri_type(self):
        """Raise TypeError when uri is not a string."""
        with pytest.raises(TypeError, match=r"(?i).*must be a string.*"):
            validate_uri(12345, schemes=["https"])

    def test_invalid_schemes_type(self):
        """Raise TypeError when schemes is not str, list, tuple, or None."""
        with pytest.raises(TypeError, match=r"(?i).*schemes must be a list or tuple.*"):
            validate_uri("https://example.com", schemes={"https"})

    def test_invalid_max_length_type(self):
        """Raise TypeError when max_length is not an int."""
        with pytest.raises(TypeError, match=r"(?i).*max_length must be a int.*"):
            validate_uri("https://example.com", schemes=["https"], max_length="8192")

    def test_empty_uri_raises(self):
        """Raise ValueError when uri is empty after stripping."""
        with pytest.raises(ValueError, match=r"(?i).*cannot be empty.*"):
            validate_uri("   ", schemes=["https"])

    def test_uri_exceeds_max_length(self):
        """Raise ValueError when uri exceeds max_length."""
        long_uri = "https://" + "a" * 9000 + ".com"
        with pytest.raises(ValueError, match=r"(?i).*exceeds maximum length.*"):
            validate_uri(long_uri, schemes=["https"], max_length=1000)

    def test_missing_host_raises(self):
        """Raise ValueError when host is missing and require_host=True."""
        with pytest.raises(ValueError, match=r"(?i).*missing network location.*"):
            validate_uri("https://", schemes=["https"], require_host=True)

    def test_allow_query_false_raises(self):
        """Raise ValueError when query present but allow_query=False."""
        uri = "https://example.com/path?token=abc"
        with pytest.raises(
            ValueError, match=r"(?i).*query parameters are not allowed.*"
        ):
            validate_uri(uri, schemes=["https"], allow_query=False)

    def test_allow_query_true_passes(self):
        """Validate URI with query when allow_query=True."""
        uri = "https://example.com/path?token=abc"
        result = validate_uri(uri, schemes=["https"], allow_query=True)
        assert result == uri

    def test_allow_relative_path(self):
        """Return relative path when allow_relative=True."""
        uri = "relative/path/to/file"
        result = validate_uri(uri, schemes=["file"], allow_relative=True)
        assert result == uri

    def test_strip_whitespace(self):
        """Strip leading and trailing whitespace from URI."""
        uri = "   https://example.com/resource   "
        result = validate_uri(uri, schemes=["https"])
        assert result == "https://example.com/resource"

    def test_no_host_allowed_when_require_host_false(self):
        """Allow URI without host when require_host=False."""
        uri = "file:///tmp/data.csv"
        result = validate_uri(uri, schemes=["file"], require_host=False)
        assert result == uri

    def test_relative_path_disallowed(self):
        """Raise ValueError when relative path given and allow_relative=False."""
        with pytest.raises(ValueError, match=r"(?i).*missing or invalid scheme.*"):
            validate_uri("relative/path", schemes=["file"], allow_relative=False)

    def test_cloud_names_disabled_skips_bucket_validation(self):
        """Skip cloud bucket validation when cloud_names=False."""
        uri = "s3://Invalid_Bucket-Name"
        result = validate_uri(uri, schemes=["s3"], cloud_names=False)
        assert result == uri

    def test_uri_length_equal_to_max_length(self):
        """Allow URI when length equals max_length."""
        uri = "https://example.com"
        result = validate_uri(uri, schemes=["https"], max_length=len(uri))
        assert result == uri

    def test_default_schemes_allows_common_scheme(self):
        """Allow common scheme when schemes=None (default)."""
        uri = "https://example.com"
        result = validate_uri(uri, schemes=None)
        assert result == uri


class TestValidateURI_AWSDb:
    @pytest.mark.parametrize(
        "uri,schemes,expect_msg",
        [
            pytest.param(
                "redshift://cluster.region.redshift.amazonaws.com:5439/mydb",
                Scheme.db.cloud.aws.all,
                None,
                id="redshift_ok",
            ),
            pytest.param(
                "redshift://cluster:badport/mydb",
                Scheme.db.cloud.aws.all,
                r"(?i).*invalid redshift host or port.*",
                id="redshift_bad_host",
            ),
        ],
    )
    def test_redshift(self, uri, schemes, expect_msg):
        """Validate Redshift URIs."""
        if expect_msg is None:
            assert validate_uri(uri, schemes=schemes) == uri
        else:
            with pytest.raises(ValueError, match=expect_msg):
                validate_uri(uri, schemes=schemes)

    @pytest.mark.parametrize(
        "uri,schemes,expect_ok,expect_msg",
        [
            pytest.param(
                "dynamodb://us-west-2/my-table",
                Scheme.db.cloud.aws.all,
                True,
                None,
                id="dynamodb_ok",
            ),
            pytest.param(
                "dynamodb://db.example.com/my-table",
                Scheme.db.cloud.aws.all,
                False,
                r"(?i).*region identifier, not a host.*",
                id="dynamodb_host_like_netloc",
            ),
            pytest.param(
                "dynamodb://us-east-1",
                Scheme.db.cloud.aws.all,
                False,
                r"(?i).*must include table.*",
                id="dynamodb_missing_table",
            ),
        ],
    )
    def test_dynamodb(self, uri, schemes, expect_ok, expect_msg):
        """Validate DynamoDB URIs."""
        if expect_ok:
            assert validate_uri(uri, schemes=schemes) == uri
        else:
            with pytest.raises(ValueError, match=expect_msg):
                validate_uri(uri, schemes=schemes)

    @pytest.mark.parametrize(
        "uri,schemes,expect_msg",
        [
            pytest.param(
                "athena://AwsDataCatalog/mydb",
                Scheme.db.cloud.aws.all,
                None,
                id="athena_ok",
            ),
            pytest.param(
                "athena:///mydb",
                Scheme.db.cloud.aws.all,
                r"(?i).*must include catalog.*",
                id="athena_missing_catalog",
            ),
            pytest.param(
                "athena://AwsDataCatalog/",
                Scheme.db.cloud.aws.all,
                r"(?i).*must include database.*",
                id="athena_missing_db",
            ),
        ],
    )
    def test_athena(self, uri, schemes, expect_msg):
        """Validate Athena URIs."""
        if expect_msg is None:
            assert validate_uri(uri, schemes=schemes) == uri
        else:
            with pytest.raises(ValueError, match=expect_msg):
                validate_uri(uri, schemes=schemes)

    @pytest.mark.parametrize(
        "uri,schemes,expect_msg",
        [
            pytest.param(
                "timestream://us-east-1/metrics_db",
                Scheme.db.cloud.aws.all,
                None,
                id="timestream_ok",
            ),
            pytest.param(
                "timestream:///metrics_db",
                Scheme.db.cloud.aws.all,
                r"(?i).*must include region.*",
                id="timestream_missing_region",
            ),
            pytest.param(
                "timestream://us-west-2/",
                Scheme.db.cloud.aws.all,
                r"(?i).*must include database.*",
                id="timestream_missing_db",
            ),
        ],
    )
    def test_timestream(self, uri, schemes, expect_msg):
        """Validate Timestream URIs."""
        if expect_msg is None:
            assert validate_uri(uri, schemes=schemes) == uri
        else:
            with pytest.raises(ValueError, match=expect_msg):
                validate_uri(uri, schemes=schemes)

    @pytest.mark.parametrize(
        "scheme,host,expect_msg",
        [
            pytest.param("rds", "db.example.internal:5432", None, id="rds_ok"),
            pytest.param(
                "aurora",
                "cluster-1.cluster-aaaa.us-east-1.rds.amazonaws.com",
                None,
                id="aurora_ok",
            ),
            pytest.param(
                "documentdb",
                "",
                r"(?i).*must include a host.*",
                id="documentdb_missing_host",
            ),
            pytest.param(
                "neptune-db",
                "bad host",
                r"(?i).*invalid host for neptune-db.*",
                id="neptune_bad_host",
            ),
        ],
    )
    def test_rds_like_families(self, scheme, host, expect_msg):
        """Validate RDS/Aurora/DocumentDB/Neptune URIs."""
        uri = f"{scheme}://{host}"
        if expect_msg is None:
            assert validate_uri(uri, schemes=Scheme.db.cloud.aws.all) == uri
        else:
            with pytest.raises(ValueError, match=expect_msg):
                validate_uri(uri, schemes=Scheme.db.cloud.aws.all)


class TestValidateURI_AWSS3Bucket:
    @pytest.mark.parametrize(
        "uri,schemes",
        [
            pytest.param(
                "s3://my-bucket/path/file.txt", Scheme.cloud(), id="s3_simple"
            ),
            pytest.param("s3a://bucket-123/data", Scheme.cloud(), id="s3a_simple"),
            pytest.param(
                "s3n://a.bucket.with.dots/obj", Scheme.cloud(), id="s3n_with_dots"
            ),
        ],
    )
    def test_bucket_ok(self, uri, schemes):
        """Accept valid S3-like bucket names."""
        assert validate_uri(uri, schemes=schemes) == uri

    @pytest.mark.parametrize(
        "uri,schemes,expect_msg",
        [
            pytest.param(
                "s3://ab/x",
                Scheme.cloud(),
                r"(?i).*must be 3-63 characters.*",
                id="too_short",
            ),
            pytest.param(
                f"s3://{'a' * 64}/x",
                Scheme.cloud(),
                r"(?i).*must be 3-63 characters.*",
                id="too_long",
            ),
            pytest.param(
                "s3://My-Bucket/x",
                Scheme.cloud(),
                r"(?i).*must be lowercase.*",
                id="uppercase",
            ),
            pytest.param(
                "s3://-badstart/x",
                Scheme.cloud(),
                r"(?i).*must start/end with.*",
                id="bad_start_char",
            ),
            pytest.param(
                "s3://badend-/x",
                Scheme.cloud(),
                r"(?i).*must start/end with.*",
                id="bad_end_char",
            ),
        ],
    )
    def test_bucket_length_and_case_and_edges(self, uri, schemes, expect_msg):
        """Reject buckets with bad length, case, or edge chars."""
        with pytest.raises(ValueError, match=expect_msg):
            validate_uri(uri, schemes=schemes)

    @pytest.mark.parametrize(
        "uri,schemes,expect_msg",
        [
            pytest.param(
                "s3://a..b/x",
                Scheme.cloud(),
                r"(?i).*cannot contain consecutive dots.*",
                id="double_dot",
            ),
            pytest.param(
                "s3://a.-b/x",
                Scheme.cloud(),
                r"(?i).*dot-dash combinations.*",
                id="dot_dash",
            ),
            pytest.param(
                "s3://a-.b/x",
                Scheme.cloud(),
                r"(?i).*dot-dash combinations.*",
                id="dash_dot",
            ),
        ],
    )
    def test_bucket_forbidden_combos(self, uri, schemes, expect_msg):
        """Reject buckets with forbidden dot/dash combos."""
        with pytest.raises(ValueError, match=expect_msg):
            validate_uri(uri, schemes=schemes)

    @pytest.mark.parametrize(
        "uri,schemes,expect_msg",
        [
            pytest.param(
                "s3://192.168.0.1/x",
                Scheme.cloud(),
                r"(?i).*cannot be formatted as IP address.*",
                id="ip_like_bucket",
            ),
        ],
    )
    def test_bucket_ip_like(self, uri, schemes, expect_msg):
        """Reject IP-like bucket names."""
        with pytest.raises(ValueError, match=expect_msg):
            validate_uri(uri, schemes=schemes)

    @pytest.mark.parametrize(
        "uri,schemes,expect_msg",
        [
            pytest.param(
                "s3://bad_char$/x",
                Scheme.cloud(),
                r"(?i).*contain only lowercase letters, numbers, hyphens, and dots.*",
                id="invalid_char",
            ),
        ],
    )
    def test_bucket_invalid_chars(self, uri, schemes, expect_msg):
        """Reject buckets with invalid characters."""
        with pytest.raises(ValueError, match=expect_msg):
            validate_uri(uri, schemes=schemes)


class TestValidateURI_AzureDb:
    @pytest.mark.parametrize(
        "uri,schemes,expect_msg",
        [
            pytest.param(
                "cosmosdb://myaccount.documents.azure.com/mydb",
                Scheme.db.cloud.azure.all,
                None,
                id="cosmosdb_ok_fqdn",
            ),
            pytest.param(
                "cosmosdb://acct-123/mydb",
                Scheme.db.cloud.azure.all,
                None,
                id="cosmosdb_ok_account_only",
            ),
            pytest.param(
                "cosmosdb:///mydb",
                Scheme.db.cloud.azure.all,
                r"(?i).*must include account host.*",
                id="cosmosdb_missing_host",
            ),
            pytest.param(
                "cosmosdb://A$@/mydb",
                Scheme.db.cloud.azure.all,
                r"(?i).*invalid cosmos db account name.*",
                id="cosmosdb_bad_account",
            ),
        ],
    )
    def test_cosmosdb(self, uri, schemes, expect_msg):
        """Validate Cosmos DB URIs."""
        if expect_msg is None:
            assert validate_uri(uri, schemes=schemes) == uri
        else:
            with pytest.raises(ValueError, match=expect_msg):
                validate_uri(uri, schemes=schemes)

    @pytest.mark.parametrize(
        "uri,schemes,expect_msg",
        [
            pytest.param(
                "synapse://workspace-01.sql.azuresynapse.net/pool1/db1",
                Scheme.db.cloud.azure.all,
                None,
                id="synapse_ok",
            ),
            pytest.param(
                "sqldw://myworkspace.dev.azuresynapse.net",
                Scheme.db.cloud.azure.all,
                None,
                id="sqldw_ok",
            ),
            pytest.param(
                "synapse://",
                Scheme.db.cloud.azure.all,
                r"(?i).*must include workspace/host.*",
                id="synapse_missing_host",
            ),
            pytest.param(
                "synapse://bad host/name",
                Scheme.db.cloud.azure.all,
                r"(?i).*invalid synapse host.*",
                id="synapse_bad_host",
            ),
        ],
    )
    def test_synapse_and_sqldw(self, uri, schemes, expect_msg):
        """Validate Synapse and SQL DW URIs."""
        if expect_msg is None:
            assert validate_uri(uri, schemes=schemes) == uri
        else:
            with pytest.raises(ValueError, match=expect_msg):
                validate_uri(uri, schemes=schemes)

    @pytest.mark.parametrize(
        "uri,schemes,expect_msg",
        [
            pytest.param(
                "azuresql://server01.database.windows.net/mydb",
                Scheme.db.cloud.azure.all,
                None,
                id="azuresql_ok",
            ),
            pytest.param(
                "azuresql://",
                Scheme.db.cloud.azure.all,
                r"(?i).*must include server host.*",
                id="azuresql_missing_host",
            ),
            pytest.param(
                "azuresql://bad host/name",
                Scheme.db.cloud.azure.all,
                r"(?i).*invalid azure sql server host.*",
                id="azuresql_bad_host",
            ),
        ],
    )
    def test_azuresql(self, uri, schemes, expect_msg):
        """Validate Azure SQL URIs."""
        if expect_msg is None:
            assert validate_uri(uri, schemes=schemes) == uri
        else:
            with pytest.raises(ValueError, match=expect_msg):
                validate_uri(uri, schemes=schemes)


class TestValidateURI_AzureStorage:
    @pytest.mark.parametrize(
        "uri,schemes",
        [
            pytest.param(
                "abfs://container-01@accountname.dfs.core.windows.net/path/file.parquet",
                Scheme.azure.all,
                id="abfs_ok_container_at_account",
            ),
            pytest.param(
                "wasbs://container-abc@acct123.blob.core.windows.net/dir",
                Scheme.azure.all,
                id="wasbs_ok_container_at_account",
            ),
            pytest.param(
                "adl://accountname.azuredatalakestore.net/mydir/data",
                Scheme.azure.all,
                id="adl_ok_account_fqdn",
            ),
            pytest.param(
                "az://container-9/path/to/blob",
                Scheme.azure.all,
                id="az_ok_container_only",
            ),
        ],
    )
    def test_base(self, uri, schemes):
        """Accept valid Azure storage URIs."""
        assert validate_uri(uri, schemes=schemes, cloud_names=True) == uri

    @pytest.mark.parametrize(
        "uri,schemes,expect_msg",
        [
            pytest.param(
                "adl://ab.azuredatalakestore.net/path",
                Scheme.azure.all,
                r"(?i).*data lake account.*3-24.*",
                id="adl_account_too_short",
            ),
            pytest.param(
                f"adl://{'a' * 25}.azuredatalakestore.net/path",
                Scheme.azure.all,
                r"(?i).*data lake account.*3-24.*",
                id="adl_account_too_long",
            ),
            pytest.param(
                "adl://BadAcct.azuredatalakestore.net/path",
                Scheme.azure.all,
                r"(?i).*data lake account.*lowercase alphanumeric.*",
                id="adl_account_uppercase",
            ),
            pytest.param(
                "adl://acct-!@.azuredatalakestore.net/path",
                Scheme.azure.all,
                r"(?i).*data lake account.*lowercase alphanumeric.*",
                id="adl_account_invalid_chars",
            ),
        ],
    )
    def test_adl_account_rules(self, uri, schemes, expect_msg):
        """Reject invalid ADL account names."""
        with pytest.raises(ValueError, match=expect_msg):
            validate_uri(uri, schemes=schemes, cloud_names=True)

    @pytest.mark.parametrize(
        "uri,schemes,expect_msg",
        [
            pytest.param(
                "az://ab/path",
                Scheme.azure.all,
                r"(?i).*container name.*3-63.*",
                id="az_container_too_short",
            ),
            pytest.param(
                f"az://{'a' * 64}/path",
                Scheme.azure.all,
                r"(?i).*container name.*3-63.*",
                id="az_container_too_long",
            ),
            pytest.param(
                "az://-bad/path",
                Scheme.azure.all,
                r"(?i).*start/end with.*letter or number.*",
                id="az_container_bad_start",
            ),
            pytest.param(
                "az://bad-/path",
                Scheme.azure.all,
                r"(?i).*start/end with.*letter or number.*",
                id="az_container_bad_end",
            ),
            pytest.param(
                "az://bad_underscore/path",
                Scheme.azure.all,
                r"(?i).*lowercase alphanumeric and hyphens.*",
                id="az_container_invalid_char",
            ),
        ],
    )
    def test_az_container_rules(self, uri, schemes, expect_msg):
        """Reject invalid az:// container names."""
        with pytest.raises(ValueError, match=expect_msg):
            validate_uri(uri, schemes=schemes, cloud_names=True)

    @pytest.mark.parametrize(
        "uri,schemes,expect_msg",
        [
            pytest.param(
                "abfs://BadContainer@account.dfs.core.windows.net/dir",
                Scheme.azure.all,
                r"(?i).*invalid azure container name.*",
                id="abfs_container_uppercase",
            ),
            pytest.param(
                "abfss://c@short.blob.core.windows.net/dir",
                Scheme.azure.all,
                r"(?i).*invalid azure container name.*3-63.*",
                id="abfss_container_too_short",
            ),
            pytest.param(
                "wasb://container@A.blob.core.windows.net/dir",
                Scheme.azure.all,
                r"(?i).*storage account.*3-24.*",
                id="wasb_account_too_short",
            ),
            pytest.param(
                f"wasb://container@{'a' * 25}.blob.core.windows.net/dir",
                Scheme.azure.all,
                r"(?i).*storage account.*3-24.*",
                id="wasb_account_too_long",
            ),
            pytest.param(
                "wasb://container@BadAcct.blob.core.windows.net/dir",
                Scheme.azure.all,
                r"(?i).*storage account.*lowercase alphanumeric.*",
                id="wasb_account_uppercase",
            ),
            pytest.param(
                "wasbs://container@acct-!.blob.core.windows.net/dir",
                Scheme.azure.all,
                r"(?i).*storage account.*lowercase alphanumeric.*",
                id="wasbs_account_invalid_chars",
            ),
        ],
    )
    def test_container_at_account_rules(self, uri, schemes, expect_msg):
        """Reject invalid container/account combos with @ account syntax."""
        with pytest.raises(ValueError, match=expect_msg):
            validate_uri(uri, schemes=schemes, cloud_names=True)


class TestValidateURI_GCPDb:
    @pytest.mark.parametrize(
        "uri,schemes,expect_msg",
        [
            pytest.param(
                "bigquery://my-project/dataset_1/table.name$20240101",
                Scheme.db.cloud.gcp.all,
                None,
                id="bigquery_ok_dataset_and_table",
            ),
            pytest.param(
                "bigquery://my-project/dataset_1",
                Scheme.db.cloud.gcp.all,
                None,
                id="bigquery_ok_dataset_only",
            ),
            pytest.param(
                "bigquery:///dataset_1",
                Scheme.db.cloud.gcp.all,
                r"(?i).*must include project id as netloc.*",
                id="bigquery_missing_project",
            ),
            pytest.param(
                "bigquery://my-project/1bad",
                Scheme.db.cloud.gcp.all,
                r"(?i).*invalid bigquery dataset name.*",
                id="bigquery_bad_dataset_name",
            ),
            pytest.param(
                "bigquery://my-project/ds/invalid*table",
                Scheme.db.cloud.gcp.all,
                r"(?i).*invalid bigquery table name.*",
                id="bigquery_bad_table_name",
            ),
        ],
    )
    def test_bigquery(self, uri, schemes, expect_msg):
        """Validate BigQuery URIs."""
        if expect_msg is None:
            assert validate_uri(uri, schemes=schemes) == uri
        else:
            with pytest.raises(ValueError, match=expect_msg):
                validate_uri(uri, schemes=schemes)

    @pytest.mark.parametrize(
        "uri,schemes,expect_msg",
        [
            pytest.param(
                "bigtable://instance-1/metrics",
                Scheme.db.cloud.gcp.all,
                None,
                id="bigtable_ok",
            ),
            pytest.param(
                "bigtable:///metrics",
                Scheme.db.cloud.gcp.all,
                r"(?i).*must include instance as netloc.*",
                id="bigtable_missing_instance",
            ),
            pytest.param(
                "bigtable://instance-1/",
                Scheme.db.cloud.gcp.all,
                r"(?i).*must include table.*",
                id="bigtable_missing_table",
            ),
        ],
    )
    def test_bigtable(self, uri, schemes, expect_msg):
        """Validate Bigtable URIs."""
        if expect_msg is None:
            assert validate_uri(uri, schemes=schemes) == uri
        else:
            with pytest.raises(ValueError, match=expect_msg):
                validate_uri(uri, schemes=schemes)

    @pytest.mark.parametrize(
        "uri,schemes,expect_msg",
        [
            pytest.param(
                "spanner://orders-instance/orders-db",
                Scheme.db.cloud.gcp.all,
                None,
                id="spanner_ok",
            ),
            pytest.param(
                "spanner:///orders-db",
                Scheme.db.cloud.gcp.all,
                r"(?i).*must include instance as netloc.*",
                id="spanner_missing_instance",
            ),
            pytest.param(
                "spanner://orders-instance/",
                Scheme.db.cloud.gcp.all,
                r"(?i).*must include database.*",
                id="spanner_missing_database",
            ),
        ],
    )
    def test_spanner(self, uri, schemes, expect_msg):
        """Validate Spanner URIs."""
        if expect_msg is None:
            assert validate_uri(uri, schemes=schemes) == uri
        else:
            with pytest.raises(ValueError, match=expect_msg):
                validate_uri(uri, schemes=schemes)

    @pytest.mark.parametrize(
        "uri,schemes,expect_msg",
        [
            pytest.param(
                "firestore://my-project/collection_1/doc-42",
                Scheme.db.cloud.gcp.all,
                None,
                id="firestore_ok_with_path",
            ),
            pytest.param(
                "firestore://my-project",
                Scheme.db.cloud.gcp.all,
                None,
                id="firestore_ok_project_only",
            ),
            pytest.param(
                "datastore://my-project/bad*collection",
                Scheme.db.cloud.gcp.all,
                r"(?i).*invalid datastore collection.*",
                id="datastore_bad_collection",
            ),
            pytest.param(
                "firestore://",
                Scheme.db.cloud.gcp.all,
                r"(?i).*must include project as netloc.*",
                id="firestore_missing_project",
            ),
        ],
    )
    def test_firestore_and_datastore(self, uri, schemes, expect_msg):
        """Validate Firestore/Datastore URIs."""
        if expect_msg is None:
            assert validate_uri(uri, schemes=schemes) == uri
        else:
            with pytest.raises(ValueError, match=expect_msg):
                validate_uri(uri, schemes=schemes)


class TestValidateURI_GCSBucket:
    @pytest.mark.parametrize(
        "uri,schemes",
        [
            pytest.param(
                "gs://my-bucket/data/file.txt", Scheme.gcp.all, id="gs_simple"
            ),
            pytest.param(
                "gs://a.bucket_with.mixed-separators/obj",
                Scheme.gcp.all,
                id="gs_mixed_separators",
            ),
            pytest.param(
                f"gs://{'a' * 63}/x", Scheme.gcp.all, id="gs_len_63_subdomain_style"
            ),
            pytest.param(
                "gs://a" * 1 + "b.c" * 50, Scheme.gcp.all, id="gs_domain_named_long_ok"
            ),
            # ensures domain-style up to 222 is allowed
        ],
    )
    def test_bucket_ok(self, uri, schemes):
        """Accept valid GCS bucket names."""
        assert validate_uri(uri, schemes=schemes) == uri

    @pytest.mark.parametrize(
        "uri,schemes,expect_msg",
        [
            pytest.param(
                "gs://ab/x",
                Scheme.gcp.all,
                r"(?i).*must be 3-63 characters.*",
                id="too_short",
            ),
            pytest.param(
                f"gs://{'a' * 223}/x",
                Scheme.gcp.all,
                r"(?i).*up to 222.*",
                id="too_long_domain_named",
            ),
        ],
    )
    def test_bucket_length_bounds(self, uri, schemes, expect_msg):
        """Reject buckets violating length bounds."""
        with pytest.raises(ValueError, match=expect_msg):
            validate_uri(uri, schemes=schemes)

    @pytest.mark.parametrize(
        "uri,schemes,expect_msg",
        [
            pytest.param(
                "gs://My-Bucket/x",
                Scheme.gcp.all,
                r"(?i).*must be lowercase.*",
                id="uppercase",
            ),
        ],
    )
    def test_bucket_lowercase(self, uri, schemes, expect_msg):
        """Reject non-lowercase bucket names."""
        with pytest.raises(ValueError, match=expect_msg):
            validate_uri(uri, schemes=schemes)

    @pytest.mark.parametrize(
        "uri,schemes,expect_msg",
        [
            pytest.param(
                "gs://-badstart/x",
                Scheme.gcp.all,
                r"(?i).*must start/end with.*",
                id="bad_start_char",
            ),
            pytest.param(
                "gs://badend-/x",
                Scheme.gcp.all,
                r"(?i).*must start/end with.*",
                id="bad_end_char",
            ),
            pytest.param(
                "gs://bad_underscore_/x",
                Scheme.gcp.all,
                r"(?i).*contain only lowercase letters, numbers, hyphens, underscores, and dots.*",
                id="bad_underscore_end",
            ),
            pytest.param(
                "gs://bad$char/x",
                Scheme.gcp.all,
                r"(?i).*contain only lowercase letters, numbers, hyphens, underscores, and dots.*",
                id="invalid_char",
            ),
        ],
    )
    def test_bucket_charset_and_edges(self, uri, schemes, expect_msg):
        """Reject buckets with invalid charset or edge chars."""
        with pytest.raises(ValueError, match=expect_msg):
            validate_uri(uri, schemes=schemes)

    @pytest.mark.parametrize(
        "uri,schemes,expect_msg",
        [
            pytest.param(
                "gs://192.168.0.1/x",
                Scheme.gcp.all,
                r"(?i).*cannot be formatted as IP address.*",
                id="ip_like_bucket",
            ),
        ],
    )
    def test_bucket_ip_like(self, uri, schemes, expect_msg):
        """Reject IP-like bucket names."""
        with pytest.raises(ValueError, match=expect_msg):
            validate_uri(uri, schemes=schemes)


class TestValidateURI_MLflowModels:
    @pytest.mark.parametrize(
        "uri,schemes",
        [
            pytest.param(
                "models:/my-model/1", Scheme.ml.mlflow.all, id="numeric_version"
            ),
            pytest.param(
                "models:/recommender/Production",
                Scheme.ml.mlflow.all,
                id="stage_production",
            ),
            pytest.param(
                "models:/classifier/Staging", Scheme.ml.mlflow.all, id="stage_staging"
            ),
            pytest.param(
                "models:/segmenter/None", Scheme.ml.mlflow.all, id="stage_none"
            ),
            pytest.param(
                "models:/archiver/Archived", Scheme.ml.mlflow.all, id="stage_archived"
            ),
        ],
    )
    def test_ok(self, uri, schemes):
        """Accept valid models:/ URIs."""
        assert validate_uri(uri, schemes=schemes) == uri

    @pytest.mark.parametrize(
        "uri,schemes,expect_msg",
        [
            pytest.param(
                "models:my-model/1",  # missing slash after colon
                Scheme.ml.mlflow.all,
                r"(?i).*expected: models:/<name>/<version_or_stage>.*",
                id="missing_leading_slash",
            ),
            pytest.param(
                "models:/onlyname",  # missing version_or_stage
                Scheme.ml.mlflow.all,
                r"(?i).*expected: models:/<name>/<version_or_stage>.*",
                id="missing_version_or_stage_segment",
            ),
            pytest.param(
                "models://name/1",  # models:// is not supported by validator's format
                Scheme.ml.mlflow.all,
                r"(?i).*expected: models:/<name>/<version_or_stage>.*",
                id="double_slash_after_scheme",
            ),
        ],
    )
    def test_format_errors(self, uri, schemes, expect_msg):
        """Reject malformed models:/ URIs."""
        with pytest.raises(ValueError, match=expect_msg):
            validate_uri(uri, schemes=schemes)

    @pytest.mark.parametrize(
        "uri,schemes,expect_msg",
        [
            pytest.param(
                "models://",  # empty path entirely
                Scheme.ml.mlflow.all,
                r"(?i).*expected: models:/<name>/<version_or_stage>.*",
                id="empty_path",
            ),
            pytest.param(
                "models:/",  # path has only the leading slash, name empty -> triggers generic format error first
                Scheme.ml.mlflow.all,
                r"(?i).*expected: models:/<name>/<version_or_stage>.*",
                id="empty_model_name_generic",
            ),
            pytest.param(
                "models:/my-model/",  # empty version or stage
                Scheme.ml.mlflow.all,
                r"(?i).*version or stage cannot be empty.*",
                id="empty_version_or_stage",
            ),
        ],
    )
    def test_empty_parts(self, uri, schemes, expect_msg):
        """Reject empty model name or version/stage."""
        with pytest.raises(ValueError, match=expect_msg):
            validate_uri(uri, schemes=schemes)

    @pytest.mark.parametrize(
        "uri,schemes,expect_msg",
        [
            pytest.param(
                "models:/my-model/dev",  # not a valid stage and not numeric
                Scheme.ml.mlflow.all,
                r"(?i).*expected: numeric version or one of.*",
                id="invalid_stage",
            ),
            pytest.param(
                "models:/my-model/v2",  # leading letter invalid for version
                Scheme.ml.mlflow.all,
                r"(?i).*expected: numeric version or one of.*",
                id="alphanumeric_version_invalid",
            ),
        ],
    )
    def test_invalid_version_or_stage(self, uri, schemes, expect_msg):
        """Reject invalid version or stage token."""
        with pytest.raises(ValueError, match=expect_msg):
            validate_uri(uri, schemes=schemes)


class TestValidateURI_MLflowRuns:
    @pytest.mark.parametrize(
        "uri,schemes",
        [
            pytest.param(
                "runs:/0123456789abcdef0123456789abcdef/model/weights.pth",
                Scheme.ml.mlflow.all,
                id="hex32_run_id_with_path",
            ),
            pytest.param(
                "runs:/run_ABC-123/artifacts/model.ckpt",
                Scheme.ml.mlflow.all,
                id="alnum_underscore_dash_run_id",
            ),
        ],
    )
    def test_ok(self, uri, schemes):
        """Accept valid runs:/ URIs."""
        assert validate_uri(uri, schemes=schemes) == uri

    @pytest.mark.parametrize(
        "uri,schemes,expect_msg",
        [
            pytest.param(
                "runs:abc123/model.pt",  # missing slash after scheme colon
                Scheme.ml.mlflow.all,
                r"(?i).*expected: runs:/<run_id>/path.*",
                id="missing_leading_slash_after_scheme",
            ),
            # With runs://abc123/model.pt, urlparse puts 'abc123' into netloc and first
            # path segment becomes 'model.pt' which is used as run_id; validator then
            # complains about invalid run id. Expect run-id error, not generic format.
            pytest.param(
                "runs://abc123/model.pt",
                Scheme.ml.mlflow.all,
                r"(?i).*invalid mlflow run id.*",
                id="double_slash_after_scheme_netloc_form",
            ),
            pytest.param(
                "runs:/",  # no run id nor path
                Scheme.ml.mlflow.all,
                r"(?i).*expected: runs:/<run_id>/path.*",
                id="empty_path_only_slash",
            ),
        ],
    )
    def test_format_errors(self, uri, schemes, expect_msg):
        """Reject malformed runs:/ URIs."""
        with pytest.raises(ValueError, match=expect_msg):
            validate_uri(uri, schemes=schemes)

    @pytest.mark.parametrize(
        "uri,schemes,expect_msg",
        [
            pytest.param(
                "runs:/not*valid/model.pt",  # run_id contains illegal '*'
                Scheme.ml.mlflow.all,
                r"(?i).*invalid mlflow run id.*",
                id="invalid_chars_in_run_id",
            ),
            pytest.param(
                "runs:/😀/model.pt",  # emoji is invalid
                Scheme.ml.mlflow.all,
                r"(?i).*invalid mlflow run id.*",
                id="emoji_in_run_id",
            ),
        ],
    )
    def test_invalid_run_id_pattern(self, uri, schemes, expect_msg):
        """Reject runs:/ with invalid run id token."""
        with pytest.raises(ValueError, match=expect_msg):
            validate_uri(uri, schemes=schemes)

    @pytest.mark.parametrize(
        "uri,schemes",
        [
            # A valid run_id with no '/path' should be accepted by current validator logic,
            # which only checks the run_id token; it does not enforce presence of a trailing path.
            pytest.param(
                "runs:/0123456789abcdef0123456789abcdef",
                Scheme.ml.mlflow.all,
                id="only_run_id_hex32_allowed",
            ),
            pytest.param(
                "runs:/run_ABC-123",
                Scheme.ml.mlflow.all,
                id="only_run_id_alnum_allowed",
            ),
        ],
    )
    def test_only_run_id_current_behavior(self, uri, schemes):
        """Accept URIs that contain only run id (validator allows it)."""
        assert validate_uri(uri, schemes=schemes) == uri


class TestValidateURI_MongoDB:
    @pytest.mark.parametrize(
        "uri,schemes",
        [
            pytest.param(
                "mongodb://user:pass@db.example.com:27017/mydb",
                Scheme.db.nosql.all,
                id="single_host_with_auth_and_db",
            ),
            pytest.param(
                "mongo://host1:27017,host2:27018,host3:27019/replicaDb",
                Scheme.db.nosql.all,
                id="replica_set_hosts_with_db",
            ),
            pytest.param(
                "mongodb://host.local/mydb-name_123",
                Scheme.db.nosql.all,
                id="dbname_simple_charclass",
            ),
        ],
    )
    def test_ok(self, uri, schemes):
        """Accept valid MongoDB URIs."""
        assert validate_uri(uri, schemes=schemes) == uri

    @pytest.mark.parametrize(
        "uri,schemes,expect_msg",
        [
            pytest.param(
                "mongodb://",  # missing host(s)
                Scheme.db.nosql.all,
                r"(?i).*must include host\(s\).*",
                id="missing_hosts",
            ),
            pytest.param(
                "mongodb://@host/db",  # empty userinfo before @
                Scheme.db.nosql.all,
                r"(?i).*credentials marker '@' present but empty userinfo.*",
                id="empty_userinfo",
            ),
            pytest.param(
                "mongodb://:pass@host/db",  # empty username when ':' present
                Scheme.db.nosql.all,
                r"(?i).*username cannot be empty.*",
                id="empty_username_with_password",
            ),
            pytest.param(
                "mongodb://good:pwd@bad host/db",  # space in host
                Scheme.db.nosql.all,
                r"(?i).*invalid mongodb host entry.*",
                id="invalid_host_space",
            ),
            pytest.param(
                "mongodb://host1:27017,host2:notaport/db",
                Scheme.db.nosql.all,
                r"(?i).*invalid mongodb host entry.*",
                id="invalid_port_non_numeric",
            ),
            pytest.param(
                "mongodb://host/db$bad",  # illegal char in db name
                Scheme.db.nosql.all,
                r"(?i).*invalid mongodb database name.*",
                id="invalid_db_chars",
            ),
        ],
    )
    def test_errors(self, uri, schemes, expect_msg):
        """Reject invalid MongoDB URIs."""
        with pytest.raises(ValueError, match=expect_msg):
            validate_uri(uri, schemes=schemes)


class TestValidateURI_Neo4j:
    @pytest.mark.parametrize(
        "uri,schemes",
        [
            pytest.param(
                "neo4j://graph.example.com:7687/db/mydb",
                (Scheme.db.graph.neo4j, Scheme.db.graph.neo4js),
                id="neo4j_db_prefix",
            ),
            pytest.param(
                "neo4j://graph.example.com:7687/mydb",
                (Scheme.db.graph.neo4j, Scheme.db.graph.neo4js),
                id="neo4j_single_db_segment",
            ),
            pytest.param(
                "neo4js://neo4j.internal:7474",
                (Scheme.db.graph.neo4j, Scheme.db.graph.neo4js),
                id="neo4js_basic_host_port",
            ),
        ],
    )
    def test_ok(self, uri, schemes):
        """Accept valid Neo4j URIs."""
        assert validate_uri(uri, schemes=schemes) == uri

    @pytest.mark.parametrize(
        "uri,schemes,expect_msg",
        [
            pytest.param(
                "neo4j://",  # missing host
                (Scheme.db.graph.neo4j, Scheme.db.graph.neo4js),
                r"(?i).*must include host.*",
                id="missing_host",
            ),
            pytest.param(
                "neo4j://bad host:7687/db/mydb",  # space in host
                (Scheme.db.graph.neo4j, Scheme.db.graph.neo4js),
                r"(?i).*invalid neo4j host or port.*",
                id="bad_host_space",
            ),
            pytest.param(
                "neo4j://graph.example.com:abc/db/mydb",  # non-numeric port
                (Scheme.db.graph.neo4j, Scheme.db.graph.neo4js),
                r"(?i).*invalid neo4j host or port.*",
                id="bad_port_non_numeric",
            ),
            pytest.param(
                "neo4j://graph.example.com/db",  # 'db' without name
                (Scheme.db.graph.neo4j, Scheme.db.graph.neo4js),
                r"(?i).*path 'db' must be followed by database name.*",
                id="db_prefix_without_name",
            ),
            pytest.param(
                "neo4j://graph.example.com/db/invalid*name",  # invalid chars in db name
                (Scheme.db.graph.neo4j, Scheme.db.graph.neo4js),
                r"(?i).*invalid neo4j database name.*",
                id="invalid_db_name_chars",
            ),
        ],
    )
    def test_errors(self, uri, schemes, expect_msg):
        """Reject invalid Neo4j URIs."""
        with pytest.raises(ValueError, match=expect_msg):
            validate_uri(uri, schemes=schemes)


class TestValidateURI_VectorDB:
    @pytest.mark.parametrize(
        "uri,schemes",
        [
            pytest.param(
                "pinecone://index-xyz.svc.us-west1-aws.pinecone.io",
                Scheme.db.vector.all,
                id="pinecone_fqdn_ok",
            ),
            pytest.param(
                "weaviate://weaviate.local:8080/MyClass",
                Scheme.db.vector.all,
                id="weaviate_with_class_ok",
            ),
            pytest.param(
                "qdrant://qdrant.internal:6333/collections/my_vectors",
                Scheme.db.vector.all,
                id="qdrant_collections_path_ok",
            ),
            pytest.param(
                "milvus://milvus.svc.cluster.local:19530",
                Scheme.db.vector.all,
                id="milvus_basic_ok",
            ),
            pytest.param(
                "chroma://chroma.host:8000",
                Scheme.db.vector.all,
                id="chroma_basic_ok",
            ),
            pytest.param(
                "chromadb://chroma.db.local",
                Scheme.db.vector.all,
                id="chromadb_basic_ok",
            ),
            pytest.param(
                "qdrant://node1:6333,node2:6333,node3:6333",
                Scheme.db.vector.all,
                id="qdrant_multi_host_ok",
            ),
        ],
    )
    def test_ok_variants(self, uri, schemes):
        """Accept valid vector DB URIs."""
        assert validate_uri(uri, schemes=schemes) == uri

    @pytest.mark.parametrize(
        "uri,schemes,expect_msg",
        [
            pytest.param(
                "pinecone:///index",  # missing host
                Scheme.db.vector.all,
                r"(?i).*must include host.*",
                id="missing_host",
            ),
            pytest.param(
                "weaviate://bad host:8080",  # space in host
                Scheme.db.vector.all,
                r"(?i).*invalid host format.*",
                id="bad_host_space",
            ),
            pytest.param(
                "qdrant://host:notaport",  # non-numeric port
                Scheme.db.vector.all,
                r"(?i).*invalid host format.*",
                id="bad_port",
            ),
            pytest.param(
                "pinecone://index-xyz.service.domain.io",  # FQDN without 'pinecone'
                Scheme.db.vector.all,
                r"(?i).*pinecone host should contain 'pinecone'.*",
                id="pinecone_missing_domain_hint",
            ),
        ],
    )
    def test_host_and_scheme_specific_errors(self, uri, schemes, expect_msg):
        """Reject invalid host and Pinecone-specific domain cases."""
        with pytest.raises(ValueError, match=expect_msg):
            validate_uri(uri, schemes=schemes)

    @pytest.mark.parametrize(
        "uri,schemes,expect_msg",
        [
            pytest.param(
                "qdrant://qdrant.local/invalid*path",
                Scheme.db.vector.all,
                r"(?i).*invalid qdrant path.*",
                id="qdrant_bad_path_chars",
            ),
            # The validator accepts any non-empty path or one that starts with 'collections/'
            # followed by name; plain 'badprefix/...' currently passes, so expect success.
            pytest.param(
                "qdrant://qdrant.local/badprefix/my_vectors",
                Scheme.db.vector.all,
                None,
                id="qdrant_non_enforced_prefix_current_behavior",
            ),
        ],
    )
    def test_qdrant_path_rules(self, uri, schemes, expect_msg):
        """Validate Qdrant path rules or current behavior."""
        if expect_msg is None:
            assert validate_uri(uri, schemes=schemes) == uri
        else:
            with pytest.raises(ValueError, match=expect_msg):
                validate_uri(uri, schemes=schemes)

    @pytest.mark.parametrize(
        "uri,schemes,expect_msg",
        [
            pytest.param(
                "chroma://",  # empty host
                Scheme.db.vector.all,
                r"(?i).*must include host.*",
                id="chroma_missing_host",
            ),
            pytest.param(
                "chromadb://bad host",
                Scheme.db.vector.all,
                r"(?i).*invalid host format.*",
                id="chromadb_bad_host",
            ),
        ],
    )
    def test_chroma_hosts(self, uri, schemes, expect_msg):
        """Reject Chroma URIs with invalid host formats."""
        with pytest.raises(ValueError, match=expect_msg):
            validate_uri(uri, schemes=schemes)


class TestValidateShape:
    """Core stdlib-only tests for validate_shape()."""

    @pytest.mark.parametrize(
        "array,shape",
        [
            pytest.param(DummyArray((2, 2)), (2, 2), id="dummy_exact_2x2"),
            pytest.param(DummyArray((3, 1)), (3, 1), id="dummy_3x1"),
            pytest.param(DummyArray((0, 2)), (0, 2), id="dummy_empty_0x2"),
            pytest.param(DummyArray((2, 3, 4)), (2, 3, 4), id="dummy_2x3x4"),
            pytest.param(DummyArray((1,)), (1,), id="dummy_scalar_like_1d_len1"),
            pytest.param(DummyArray((3, 0)), (3, 0), id="dummy_zero_cols"),
        ],
    )
    def test_array_pass(self, array, shape):
        """Validate dummy array shapes and pass."""
        out = validate_shape(array, shape=shape)
        assert out is array

    @pytest.mark.parametrize(
        "array,shape,err_sub",
        [
            pytest.param(DummyArray((2, 2)), (2, 3), "Shape mismatch", id="wrong_cols"),
            pytest.param(DummyArray((2, 2)), (3, 2), "Shape mismatch", id="wrong_rows"),
        ],
    )
    def test_array_fail(self, array, shape, err_sub):
        """Raise on dummy array shape mismatch."""
        with pytest.raises(ValueError, match=rf"(?i).*{err_sub}.*"):
            validate_shape(array, shape=shape)

    def test_non_strict_trailing_match(self):
        """Allow leading batch dims in non-strict mode."""
        arr = DummyArray((5, 3, 2))
        out = validate_shape(arr, shape=(3, 2), strict=False)
        assert out is arr

    def test_dimension_specific_mismatch_message(self):
        """Include dimension index in mismatch message."""
        arr = DummyArray((3, 2))
        with pytest.raises(ValueError, match=r"(?i).*dimension 1.*expected 3.*"):
            validate_shape(arr, shape=(3, 3))

    @pytest.mark.parametrize(
        "shape,err_sub",
        [
            pytest.param((-1, 2), "must be non-negative", id="negative_rows"),
            pytest.param((2, -1), "must be non-negative", id="negative_cols"),
        ],
    )
    def test_negative_dimensions(self, shape, err_sub):
        """Raise on negative dimensions in shape."""
        arr = DummyArray((2, 2))
        with pytest.raises(ValueError, match=rf"(?i).*{err_sub}.*"):
            validate_shape(arr, shape=shape)

    @pytest.mark.parametrize(
        "array,shape,strict,should_pass",
        [
            pytest.param(DummyArray((3, 2)), (3, 2), True, True, id="exact_pass"),
            pytest.param(DummyArray((3, 2)), (2, 3), True, False, id="mismatch_fail"),
            pytest.param(
                DummyArray((5, 3, 2)), (3, 2), False, True, id="non_strict_pass"
            ),
            pytest.param(DummyArray((5, 3, 2)), (3, 2), True, False, id="strict_fail"),
        ],
    )
    def test_strict_and_non_strict(self, array, shape, strict, should_pass):
        """Validate arrays with strict and non-strict modes."""
        if should_pass:
            out = validate_shape(array, shape=shape, strict=strict)
            assert out is array
        else:
            with pytest.raises(ValueError, match=r"(?i).*shape mismatch.*"):
                validate_shape(array, shape=shape, strict=strict)

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((3, "foo"), id="invalid_literal"),
            pytest.param((3.5, 2), id="float_dimension"),
            pytest.param(("any", -1), id="negative_with_any"),
        ],
    )
    def test_invalid_shape_spec(self, shape):
        """Raise on invalid shape specification."""
        arr = DummyArray((3, 2))
        with pytest.raises(
            (TypeError, ValueError), match=r"(?i).*invalid|non-negative.*"
        ):
            validate_shape(arr, shape=shape)

    @pytest.mark.parametrize(
        "obj",
        [
            pytest.param("not an array", id="string_input"),
            pytest.param({"a": 1}, id="dict_input"),
            pytest.param(object(), id="plain_object"),
        ],
    )
    def test_unsupported_type_raises(self, obj):
        """Raise TypeError for unsupported input types."""
        with pytest.raises(TypeError, match=r"(?i).*array-like.*"):
            validate_shape(obj, shape=(1,))

    def test_non_strict_requires_dimensions(self):
        """Require at least ndim in non-strict mode."""
        arr = DummyArray((3, 2))
        with pytest.raises(ValueError, match=r"(?i).*dimensions.*"):
            validate_shape(arr, shape=(3, 2, 1), strict=False)

    @pytest.mark.parametrize(
        "data,shape,should_pass",
        [
            pytest.param([[1, 2], [3, 4]], (2, 2), True, id="list_2x2_pass"),
            pytest.param([[1, 2], [3, 4, 5]], (2, 2), False, id="list_irregular_fail"),
            pytest.param([], (0,), True, id="empty_list_1d_pass"),
        ],
    )
    def test_list_input_shapes(self, data, shape, should_pass):
        """Validate list inputs as sizable."""
        if should_pass:
            out = validate_shape(data, shape=shape)
            assert out == data
        else:
            with pytest.raises(
                ValueError, match=r"(?i).*(ragged|inconsistent|shape mismatch).*"
            ):
                validate_shape(data, shape=shape)

    @pytest.mark.parametrize(
        "array,shape",
        [
            pytest.param(DummyArray(()), (), id="empty_shape_match"),
            pytest.param(DummyArray((0,)), (0,), id="zero_length_dim_match"),
        ],
    )
    def test_empty_and_zero_length_shapes(self, array, shape):
        """Handle empty and zero-length shapes correctly."""
        out = validate_shape(array, shape=shape)
        assert out is array

    @pytest.mark.parametrize(
        "array,shape,should_pass",
        [
            pytest.param(DummyArray((3, 4)), (3, "any"), True, id="mixed_any_last"),
            pytest.param(
                DummyArray((3, 4, 5)),
                ("any", "any", 5),
                True,
                id="mixed_any_extra_dims",
            ),
            pytest.param(
                DummyArray((2, 4)), ("any", 4), True, id="mixed_any_first_mismatch"
            ),
            pytest.param(
                DummyArray((1, 2)), ("any",), False, id="mixed_any_first_mismatch"
            ),
        ],
    )
    def test_any_and_fixed_dimensions(self, array, shape, should_pass):
        """Validate mixed fixed and 'all' wildcard dimensions."""
        if should_pass:
            out = validate_shape(array, shape=shape)
            assert out is array
        else:
            with pytest.raises(ValueError, match=r"(?i).*shape mismatch.*"):
                validate_shape(array, shape=shape)

    def test_empty_list_with_all_shape(self):
        """Allow empty list when shape='all'."""
        data = []
        out = validate_shape(data, shape=("any",))
        assert out == data

    def test_empty_list_with_explicit_shape(self):
        """Reject empty list when shape expects nonzero dims."""
        data = []
        with pytest.raises(ValueError, match=r"(?i).*shape mismatch.*"):
            validate_shape(data, shape=(1,))

    def test_shape_as_empty_tuple_with_nonempty_array(self):
        """Reject non-empty array when shape=()."""
        arr = DummyArray((2, 2))
        with pytest.raises(ValueError, match=r"(?i).*shape mismatch.*"):
            validate_shape(arr, shape=())

    @pytest.mark.parametrize(
        "array,shape",
        [
            pytest.param(DummyArray((1,)), (1,), id="dummy_scalar_like_1d_len1"),
            pytest.param(DummyArray((3, 0)), (3, 0), id="dummy_zero_cols"),
        ],
    )
    def test_additional_pass_shapes(self, array, shape):
        out = validate_shape(array, shape=shape)
        assert out is array

    def test_non_strict_trailing_match_deeper_batch(self):
        """Non-strict allows multiple leading batch dims."""
        arr = DummyArray((7, 5, 3, 2))
        out = validate_shape(arr, shape=(3, 2), strict=False)
        assert out is arr

    @pytest.mark.parametrize(
        "array,shape,expected_dim_index,expected_expected",
        [
            pytest.param(DummyArray((4, 2, 1)), (4, 3, 1), 1, 3, id="dim1_mismatch"),
            pytest.param(DummyArray((2, 2, 2)), (3, 2, 2), 0, 3, id="dim0_mismatch"),
        ],
    )
    def test_dimension_specific_message_includes_index_and_expected(
        self, array, shape, expected_dim_index, expected_expected
    ):
        with pytest.raises(
            ValueError,
            match=rf"(?i).*dimension {expected_dim_index}.*expected {expected_expected}.*",
        ):
            validate_shape(array, shape=shape)

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((-2,), id="single_negative"),
            pytest.param((1, -3, 2), id="mixed_negative"),
        ],
    )
    def test_more_negative_dimensions(self, shape):
        arr = DummyArray((1, 2, 2))
        with pytest.raises(ValueError, match=r"(?i).*non-negative.*"):
            validate_shape(arr, shape=shape)

    @pytest.mark.parametrize(
        "array,shape,strict,should_pass",
        [
            pytest.param(
                DummyArray((2, 3, 4)), (3, 4), False, True, id="ns_tail_match"
            ),
            pytest.param(
                DummyArray((2, 3, 4)), (2, 3, 4), True, True, id="strict_full_match"
            ),
            pytest.param(
                DummyArray((2, 3, 4)), (4, 3), False, False, id="ns_tail_mismatch"
            ),
        ],
    )
    def test_strict_vs_non_strict_tail_matching(
        self, array, shape, strict, should_pass
    ):
        if should_pass:
            out = validate_shape(array, shape=shape, strict=strict)
            assert out is array
        else:
            with pytest.raises(ValueError, match=r"(?i).*shape mismatch.*"):
                validate_shape(array, shape=shape, strict=strict)

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((object(),), id="object_dimension"),
            pytest.param((None,), id="none_dimension"),
        ],
    )
    def test_invalid_shape_types(self, shape):
        arr = DummyArray((1,))
        with pytest.raises(
            (TypeError, ValueError), match=r"(?i).*invalid|unsupported.*"
        ):
            validate_shape(arr, shape=shape)

    def test_unsupported_type_message_includes_hint(self):
        with pytest.raises(TypeError, match=r"(?i).*str.*"):
            validate_shape("string", shape=(1,))

    def test_non_strict_requires_enough_trailing_dims(self):
        arr = DummyArray((2,))
        with pytest.raises(ValueError, match=r"(?i).*dimensions.*"):
            validate_shape(arr, shape=(2, 1), strict=False)

    @pytest.mark.parametrize(
        "array,shape,should_pass",
        [
            pytest.param(DummyArray((3, 5)), (3, "any"), True, id="any_last_ok"),
            pytest.param(DummyArray((3, 5)), ("any", 5), True, id="any_first_ok"),
            pytest.param(
                DummyArray((3, 5)), ("any", 6), False, id="any_fixed_mismatch"
            ),
        ],
    )
    def test_any_wildcard_positions(self, array, shape, should_pass):
        if should_pass:
            out = validate_shape(array, shape=shape)
            assert out is array
        else:
            with pytest.raises(ValueError, match=r"(?i).*shape mismatch.*"):
                validate_shape(array, shape=shape)

    def test_shape_empty_tuple_requires_scalar_like(self):
        arr = DummyArray(())
        out = validate_shape(arr, shape=())
        assert out is arr
