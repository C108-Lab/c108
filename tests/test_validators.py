#
# C108 - Validators Tests
#

# Standard library -----------------------------------------------------------------------------------------------------

# Third party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.validators import validate_email, validate_ip_address, validate_language_code


# Tests ----------------------------------------------------------------------------------------------------------------


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