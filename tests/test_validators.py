#
# C108 - Validators Tests
#

# Standard library -----------------------------------------------------------------------------------------------------

# Third party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.validators import (
    Scheme,
    SchemeGroup,
    validate_email,
    validate_ip_address,
    validate_language_code,
    validate_uri,
)


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


import pytest
from c108.validators import validate_ip_address


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


import pytest
from c108.validators import validate_language_code


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
