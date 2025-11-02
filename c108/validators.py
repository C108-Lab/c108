"""
C108 Data Validators

This module contains validation functions for data formats and values
including emails, IP addresses, language codes, and URLs.

These validators check the **content/format** of values, not their types.
For runtime type validation, see the abc module.
"""

# Standard library -----------------------------------------------------------------------------------------------------
import ipaddress
import re

from typing import Literal

from .abc import valid_param_types
# Local ----------------------------------------------------------------------------------------------------------------

from .tools import fmt_type, fmt_value


# Methods --------------------------------------------------------------------------------------------------------------

def validate_email(email: str, *, strip: bool = True, lowercase: bool = True) -> str:
    """
    Validate email address format according to simplified RFC 5322 rules.

    This function performs basic email sanity checks for:
        - Non-empty input
        - Valid format (local-part@domain)
        - Length constraints (RFC 5321)
        - Character set compliance

    Note:
        This validation is intentionally simplified for lightweight use cases like
        form prototypes, data preprocessing, unit tests, and educational examples.
        It does NOT cover all RFC 5322 edge cases (quoted strings, comments,
        IP-literals, internationalized domains) or verify deliverability.

        For production systems requiring full RFC compliance, deliverability
        verification, or high-performance validation, use dedicated libraries:
            - `email-validator`: Full RFC compliance with DNS MX lookup support
            - `pydantic[email]`: Rust-backed validation for speed
            - SMTP verification libraries for deliverability checks (slow)

        While RFC 5321 technically allows case-sensitive local parts, virtually
        all modern email providers treat addresses case-insensitively. The default
        behavior normalizes to lowercase per community standards.

    Args:
        email: Email address string to validate.
        strip: If True, strip leading/trailing whitespace before validation.
            Defaults to True.
        lowercase: If True, convert email to lowercase after validation. Defaults
            to True. Recommended for consistency since email addresses are
            case-insensitive in practice.

    Returns:
        The validated email address. Transformations applied in order:
        1. Strip whitespace (if strip=True)
        2. Validate format
        3. Convert to lowercase (if lowercase=True)

    Raises:
        TypeError: If email is not a string.
        ValueError: If email is empty (after stripping if enabled), exceeds length
            limits (254 chars total, 64 for local part), or has invalid format.

    Examples:
        >>> validate_email("User@example.COM")
        'user@example.com'
        >>> validate_email("")
        Traceback (most recent call last):
            ...
        ValueError: email cannot be empty
        >>> validate_email("invalid.email")
        Traceback (most recent call last):
            ...
        ValueError: invalid email format: missing '@' symbol: 'invalid.email'
        >>> validate_email("a" * 65 + "@example.com")
        Traceback (most recent call last):
            ...
        ValueError: email local part exceeds maximum length of 64 characters (got 65)
    """

    # Type check
    if not isinstance(email, str):
        raise TypeError(f"Email must be a string, got {fmt_type(email)}")

    # Handle whitespace
    if strip:
        email = email.strip()
    elif email != email.strip():
        raise ValueError(
            f"invalid email format: contains leading or trailing whitespace: '{email}'"
        )

    # Check for empty
    if not email:
        raise ValueError("email cannot be empty")

    # Check overall length (RFC 5321: 254 chars max)
    if len(email) > 254:
        raise ValueError(
            f"email exceeds maximum length of 254 characters (got {len(email)})"
        )

    # Check for @ symbol
    if "@" not in email:
        raise ValueError(f"invalid email format: missing '@' symbol: '{email}'")

    # Split and validate structure
    parts = email.rsplit("@", 1)
    if len(parts) != 2:
        raise ValueError(f"invalid email format: multiple '@' symbols: '{email}'")

    local_part, domain = parts

    # Validate local part
    if not local_part:
        raise ValueError(f"invalid email format: missing local part: '{email}'")
    if len(local_part) > 64:
        raise ValueError(
            f"email local part exceeds maximum length of 64 characters (got {len(local_part)})"
        )

    # Validate domain
    if not domain:
        raise ValueError(f"invalid email format: missing domain: '{email}'")
    if domain.startswith(".") or domain.endswith("."):
        raise ValueError(f"invalid email format: domain cannot start or end with '.': '{email}'")
    if ".." in domain:
        raise ValueError(f"invalid email format: domain cannot contain consecutive dots: '{email}'")
    if "." not in domain:
        raise ValueError(f"invalid email format: domain must contain at least one dot: '{email}'")

    # Regex pattern for detailed validation (case-insensitive)
    email_pattern = r"^[a-zA-Z0-9][a-zA-Z0-9._%+-]*[a-zA-Z0-9]@[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?)*\.[a-zA-Z]{2,}$"

    # Special case: single character local part
    if len(local_part) == 1:
        email_pattern = r"^[a-zA-Z0-9]@[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?)*\.[a-zA-Z]{2,}$"

    if not re.match(email_pattern, email):
        raise ValueError(f"invalid email format: '{email}'")

    # Normalize to lowercase if requested
    if lowercase:
        email = email.lower()

    return email


def validate_ip_address(
        ip: str,
        *,
        strip: bool = True,
        version: Literal[4, 6, "any"] = "any",
        leading_zeros: bool = False
) -> str:
    """
    Validate IP address format for IPv4 and/or IPv6.

    This function uses Python's `ipaddress` module for standards-compliant
    validation per RFC 791 (IPv4) and RFC 4291 (IPv6).

    Args:
        ip: IP address string to validate.
        strip: If True, strip leading/trailing whitespace before validation.
            Defaults to True.
        version: Required IP version. Use 4 for IPv4 only, 6 for IPv6 only,
            or "any" to accept either. Defaults to "any".
        leading_zeros: If True, allow leading zeros in IPv4 octets
            (e.g., '192.168.001.001'). Defaults to False as leading zeros
            can be ambiguous (octal vs decimal). This parameter only affects
            IPv4; IPv6 leading zeros are always allowed as they are standard
            in hexadecimal notation.

    Returns:
        The validated IP address string. If strip=True, returns the stripped
        version; otherwise returns the original input if valid. The returned
        format is normalized (IPv6 may be compressed per RFC 5952).

    Raises:
        TypeError: If ip is not a string or version is invalid.
        ValueError: If ip is empty (after stripping if enabled), has invalid
            format, or doesn't match the required version.

    Examples:
        IPv4 validation:
        >>> validate_ip_address("192.168.1.1")
        '192.168.1.1'
        >>> validate_ip_address("  10.0.0.1  ")
        '10.0.0.1'
        >>> validate_ip_address("255.255.255.255")
        '255.255.255.255'
        >>> validate_ip_address("0.0.0.0")
        '0.0.0.0'

        IPv6 validation (leading zeros always allowed):
        >>> validate_ip_address("2001:0db8:85a3:0000:0000:8a2e:0370:7334")
        '2001:db8:85a3::8a2e:370:7334'
        >>> validate_ip_address("::1")
        '::1'
        >>> validate_ip_address("fe80::1")
        'fe80::1'
        >>> validate_ip_address("2001:0db8::0001")
        '2001:db8::1'

        Raises errors:
        >>> validate_ip_address("192.168.1")
        Traceback (most recent call last):
            ...
        ValueError: invalid IP address format: '192.168.1'
        >>> validate_ip_address("not.an.ip.address")
        Traceback (most recent call last):
            ...
        ValueError: invalid IP address format: 'not.an.ip.address'
        >>> validate_ip_address("gggg::1")
        Traceback (most recent call last):
            ...
        ValueError: invalid IP address format: 'gggg::1'
    """
    # Type check
    if not isinstance(ip, str):
        raise TypeError(f"IP address must be a string, got {fmt_type(ip)}")

    if version not in (4, 6, "any"):
        raise TypeError(f"version must be 4, 6, or 'any', got {fmt_value(version)}")

    # Handle whitespace
    if strip:
        ip = ip.strip()
    elif ip != ip.strip():
        raise ValueError(
            f"invalid IP address format: contains leading or trailing whitespace: '{ip}'"
        )

    # Check for empty
    if not ip:
        raise ValueError("IP address cannot be empty")

    # Handle leading zeros in IPv4
    # IPv6 leading zeros are standard in hex notation, so no processing needed
    if "." in ip and ":" not in ip:
        # Likely IPv4
        parts = ip.split(".")
        if len(parts) == 4:
            # Check if any part has leading zeros
            has_leading_zeros = any(
                len(part) > 1 and part[0] == "0" for part in parts
            )

            if has_leading_zeros:
                if not leading_zeros:
                    # Reject leading zeros
                    raise ValueError(
                        f"invalid IPv4 address: leading zeros not allowed (octal ambiguity): '{ip}'"
                    )
                else:
                    # Strip leading zeros from each octet
                    # Convert each part to int and back to str to remove leading zeros
                    try:
                        normalized_parts = [str(int(part)) for part in parts]
                        ip = ".".join(normalized_parts)
                    except ValueError:
                        # If int() fails, it's not a valid number - let ipaddress handle it
                        pass

    # Validate using ipaddress module
    try:
        ip_obj = ipaddress.ip_address(ip)
    except ValueError:
        raise ValueError(f"invalid IP address format: '{ip}'")

    # Check version requirement
    if version != "any":
        if ip_obj.version != version:
            actual_version = "IPv4" if ip_obj.version == 4 else "IPv6"
            expected_version = "IPv4" if version == 4 else "IPv6"
            raise ValueError(
                f"IP address version mismatch: '{ip}' is {actual_version}, expected {expected_version}"
            )

    # Return normalized format (ipaddress module normalizes both IPv4 and IPv6)
    return str(ip_obj)


def validate_language_code(
        language_code: str, allow_ISO: bool = True, allow_BCP47: bool = True
) -> str:
    """
    Validate language code follows ISO 639-1 or BCP 47 format

    Args:
        language_code: The language code to validate
        allow_ISO: Whether to allow ISO 639-1 format (e.g., 'en', 'fr')
        allow_BCP47: Whether to allow BCP 47 format (e.g., 'en-US', 'fr-CA')

    Returns:
        str: The original language code if valid

    Raises:
        TypeError: If language code type is not str
        ValueError: If language code is invalid or format not allowed
    """
    if not language_code:
        raise ValueError("Language code cannot be empty")

    if not allow_ISO and not allow_BCP47:
        raise ValueError("At least one format (ISO or BCP47) must be allowed")

    if not isinstance(language_code, str):
        raise TypeError(f"Language code must be a string, got {fmt_type(language_code)}")

    # Convert to lowercase for validation
    lang_lower = language_code.lower()

    # ISO 639-1 two-letter codes - ALL Countries
    iso_639_1_codes = {
        "af",
        "al",
        "dz",
        "as",
        "ad",
        "ao",
        "ai",
        "aq",
        "ag",
        "ar",
        "am",
        "aw",
        "au",
        "at",
        "az",
        "bs",
        "bh",
        "bd",
        "bb",
        "by",
        "be",
        "bz",
        "bj",
        "bm",
        "bt",
        "bo",
        "bq",
        "ba",
        "bw",
        "bv",
        "br",
        "io",
        "bn",
        "bg",
        "bf",
        "bi",
        "cv",
        "kh",
        "cm",
        "ca",
        "ky",
        "cf",
        "td",
        "cl",
        "cn",
        "cx",
        "cc",
        "co",
        "km",
        "cd",
        "cg",
        "ck",
        "cr",
        "hr",
        "cu",
        "cw",
        "cy",
        "cz",
        "ci",
        "dk",
        "dj",
        "dm",
        "do",
        "ec",
        "eg",
        "sv",
        "gq",
        "er",
        "ee",
        "sz",
        "et",
        "fk",
        "fo",
        "fj",
        "fi",
        "fr",
        "gf",
        "pf",
        "tf",
        "ga",
        "gm",
        "ge",
        "de",
        "gh",
        "gi",
        "gr",
        "gl",
        "gd",
        "gp",
        "gu",
        "gt",
        "gg",
        "gn",
        "gw",
        "gy",
        "ht",
        "hm",
        "va",
        "hn",
        "hk",
        "hu",
        "is",
        "in",
        "id",
        "ir",
        "iq",
        "ie",
        "im",
        "il",
        "it",
        "jm",
        "jp",
        "je",
        "jo",
        "kz",
        "ke",
        "ki",
        "kp",
        "kr",
        "kw",
        "kg",
        "la",
        "lv",
        "lb",
        "ls",
        "lr",
        "ly",
        "li",
        "lt",
        "lu",
        "mo",
        "mg",
        "mw",
        "my",
        "mv",
        "ml",
        "mt",
        "mh",
        "mq",
        "mr",
        "mu",
        "yt",
        "mx",
        "fm",
        "md",
        "mc",
        "mn",
        "me",
        "ms",
        "ma",
        "mz",
        "mm",
        "na",
        "nr",
        "np",
        "nl",
        "nc",
        "nz",
        "ni",
        "ne",
        "ng",
        "nu",
        "nf",
        "mp",
        "no",
        "om",
        "pk",
        "pw",
        "ps",
        "pa",
        "pg",
        "py",
        "pe",
        "ph",
        "pn",
        "pl",
        "pt",
        "pr",
        "qa",
        "mk",
        "ro",
        "ru",
        "rw",
        "re",
        "bl",
        "sh",
        "kn",
        "lc",
        "mf",
        "pm",
        "vc",
        "ws",
        "sm",
        "st",
        "sa",
        "sn",
        "rs",
        "sc",
        "sl",
        "sg",
        "sx",
        "sk",
        "si",
        "sb",
        "so",
        "za",
        "gs",
        "ss",
        "es",
        "lk",
        "sd",
        "sr",
        "sj",
        "se",
        "ch",
        "sy",
        "tw",
        "tj",
        "tz",
        "th",
        "tl",
        "tg",
        "tk",
        "to",
        "tt",
        "tn",
        "tr",
        "tm",
        "tc",
        "tv",
        "ug",
        "ua",
        "ae",
        "gb",
        "um",
        "us",
        "uy",
        "uz",
        "vu",
        "ve",
        "vn",
        "vg",
        "vi",
        "wf",
        "eh",
        "ye",
        "zm",
        "zw",
        "ax",
    }
    # BCP 47 format (language-region) codes - Core Selection of Languages (Major Languages only)
    bcp_47_codes = {
        "af-na",
        "af-za",
        "ar-ae",
        "ar-bh",
        "ar-dj",
        "ar-dz",
        "ar-eg",
        "ar-er",
        "ar-iq",
        "ar-jo",
        "ar-km",
        "ar-kw",
        "ar-lb",
        "ar-ly",
        "ar-ma",
        "ar-mr",
        "ar-om",
        "ar-qa",
        "ar-sa",
        "ar-sd",
        "ar-so",
        "ar-sy",
        "ar-td",
        "ar-tn",
        "ar-ye",
        "az-az",
        "be-by",
        "bg-bg",
        "bn-bd",
        "bn-in",
        "ca-es",
        "cs-cz",
        "da-dk",
        "da-gl",
        "de-at",
        "de-ch",
        "de-de",
        "de-li",
        "de-lu",
        "el-cy",
        "el-gr",
        "en-au",
        "en-ca",
        "en-gb",
        "en-ie",
        "en-in",
        "en-nz",
        "en-us",
        "en-za",
        "es-ar",
        "es-bo",
        "es-cl",
        "es-co",
        "es-cr",
        "es-cu",
        "es-do",
        "es-ec",
        "es-es",
        "es-gt",
        "es-hn",
        "es-mx",
        "es-ni",
        "es-pa",
        "es-pe",
        "es-pr",
        "es-py",
        "es-sv",
        "es-uy",
        "es-ve",
        "et-ee",
        "eu-es",
        "fi-fi",
        "fr-be",
        "fr-ca",
        "fr-ch",
        "fr-fr",
        "fr-lu",
        "fr-mc",
        "ga-ie",
        "gl-es",
        "gu-in",
        "he-il",
        "hi-in",
        "hr-ba",
        "hr-hr",
        "hu-hu",
        "hy-am",
        "id-id",
        "is-is",
        "it-ch",
        "it-it",
        "it-sm",
        "it-va",
        "ja-jp",
        "ka-ge",
        "kk-kz",
        "kn-in",
        "ko-kp",
        "ko-kr",
        "ky-kg",
        "lt-lt",
        "lv-lv",
        "mk-mk",
        "ml-in",
        "mr-in",
        "ms-bn",
        "ms-my",
        "ms-sg",
        "mt-mt",
        "nb-no",
        "nl-be",
        "nl-nl",
        "nl-sr",
        "nn-no",
        "no-no",
        "nr-za",
        "nso-za",
        "or-in",
        "pa-in",
        "pa-pk",
        "pl-pl",
        "pt-ao",
        "pt-br",
        "pt-cv",
        "pt-gw",
        "pt-mz",
        "pt-pt",
        "pt-st",
        "pt-tl",
        "rm-ch",
        "ro-md",
        "ro-ro",
        "ru-by",
        "ru-kg",
        "ru-kz",
        "ru-md",
        "ru-ru",
        "ru-ua",
        "si-lk",
        "sk-sk",
        "sl-si",
        "sq-al",
        "sr-ba",
        "sr-me",
        "sr-rs",
        "sr-xk",
        "ss-sz",
        "ss-za",
        "st-ls",
        "st-za",
        "sv-fi",
        "sv-se",
        "sw-bi",
        "sw-cd",
        "sw-ke",
        "sw-km",
        "sw-mg",
        "sw-mw",
        "sw-mz",
        "sw-rw",
        "sw-tz",
        "sw-ug",
        "sw-zm",
        "ta-in",
        "ta-lk",
        "ta-my",
        "ta-sg",
        "te-in",
        "tg-tj",
        "th-th",
        "tk-tm",
        "tl-ph",
        "tn-bw",
        "tn-za",
        "tr-cy",
        "tr-tr",
        "ts-mz",
        "ts-za",
        "uk-ua",
        "ur-in",
        "ur-pk",
        "uz-uz",
        "ve-za",
        "vi-vn",
        "xh-za",
        "zh-cn",
        "zh-hk",
        "zh-mo",
        "zh-sg",
        "zh-tw",
        "zu-za",
    }
    # Check if it's a known ISO code
    is_iso_code = lang_lower in iso_639_1_codes
    # Check if it's a known BCP 47 code
    is_bcp47_code = lang_lower in bcp_47_codes

    # If exact match found, check if format is allowed
    if is_iso_code:
        if not allow_ISO:
            raise ValueError(
                f"ISO 639-1 format not allowed: '{language_code}'. "
                f"Use BCP 47 format (e.g., 'en-US') instead."
            )
        return language_code

    if is_bcp47_code:
        if not allow_BCP47:
            raise ValueError(
                f"BCP 47 format not allowed: '{language_code}'. "
                f"Use ISO 639-1 format (e.g., 'en') instead."
            )
        return language_code

    # If no exact match, check patterns for flexibility
    # Pattern for ISO 639-1 (2 letters)
    iso_639_1_pattern = r"^[a-z]{2}$"
    # Pattern for BCP 47 (language-region: 2-5 letters, dash, 2-3 letters/digits)
    bcp_47_pattern = r"^[a-z]{2,5}-[a-z0-9]{2,3}$"

    matches_iso_pattern = re.match(iso_639_1_pattern, lang_lower)
    matches_bcp47_pattern = re.match(bcp_47_pattern, lang_lower)

    # Check pattern matches against allowed formats
    if matches_iso_pattern:
        if not allow_ISO:
            raise ValueError(
                f"ISO 639-1 format not allowed: '{language_code}'. "
                f"Use BCP 47 format (e.g., 'en-US') instead."
            )
        return language_code

    if matches_bcp47_pattern:
        if not allow_BCP47:
            raise ValueError(
                f"BCP 47 format not allowed: '{language_code}'. "
                f"Use ISO 639-1 format (e.g., 'en') instead."
            )
        return language_code

    # If no pattern matches, provide appropriate error message
    allowed_formats = []
    if allow_ISO:
        allowed_formats.append("ISO 639-1 format (e.g., 'en', 'fr')")
    if allow_BCP47:
        allowed_formats.append("BCP 47 format (e.g., 'en-US', 'fr-CA')")

    formats_str = " or ".join(allowed_formats)
    raise ValueError(f"Invalid language code: '{language_code}'. Use {formats_str}")


def validate_url(url: str) -> str:
    """Validate URL format

    Returns:
        str: the original URL if valid

    Examples:
        >>> url = "https://www.example.com"
        >>> validate_url(url)
        'https://www.example.com'

        >>> url = "not-a-url"
        >>> validate_url(url)
        Traceback (most recent call last):
        ...
        ValueError: Invalid URL format: 'not-a-url'
    """
    if not url:
        raise ValueError("URL cannot be empty")

    url_pattern = r"^https?://[^\s/$.?#].[^\s]*$"
    if not re.match(url_pattern, url):
        raise ValueError(f"Invalid URL format: '{url}'")

    return url


def validate_range(
        value: float, min_value: float | None = None, max_value: float | None = None
) -> float:
    """
    Validate numeric value falls within specified bounds.

    Checks that a numeric value is within the specified minimum and maximum bounds (inclusive).
    Useful for validating ML features, hyperparameters, or any numeric data with known constraints.
    At least one bound (min or max) should be specified.

    Args:
        value: The numeric value to validate
        min_value: Minimum allowed value (inclusive), or None for no lower bound
        max_value: Maximum allowed value (inclusive), or None for no upper bound

    Returns:
        float: The original value if valid

    Raises:
        ValueError: If value is outside the specified range
    """


def validate_probability(value: float) -> float:
    """
    Validate value is a valid probability between 0 and 1 (inclusive).

    Ensures a numeric value falls within the [0, 1] interval, which is required for probabilities,
    confidence scores, model outputs from sigmoid/softmax layers, mixing weights, and similar
    ML/statistical values. This is a specialized case of range validation optimized for the
    common probability constraint.

    Args:
        value: The numeric value to validate as a probability

    Returns:
        float: The original value if valid

    Raises:
        ValueError: If value is not in [0, 1]
    """


def validate_positive(value: float, strict: bool = True) -> float:
    """
    Validate value is positive (optionally allowing zero).

    Checks that a numeric value is greater than zero (or greater than/equal to zero if strict=False).
    Common for counts, distances, rates, durations, and other naturally positive quantities in
    ML pipelines. Use strict=False when zero is a valid value (e.g., counts can be zero).

    Args:
        value: The numeric value to validate
        strict: If True, value must be > 0; if False, value must be >= 0

    Returns:
        float: The original value if valid

    Raises:
        ValueError: If value violates the positivity constraint
    """


# Data Structure Validators


def validate_not_empty(collection) -> any:
    """
    Validate collection is not empty.

    Ensures a collection (list, tuple, set, dict, array, DataFrame, etc.) contains at least one
    element. Prevents silent failures from empty batches, missing data, or incorrectly filtered
    datasets that would cause downstream errors in ML pipelines or data processing.

    Args:
        collection: Any collection type that supports len() or has a boolean context

    Returns:
        The original collection if not empty

    Raises:
        ValueError: If collection is empty
    """


def validate_shape(array, expected_shape: tuple[int | None, ...]) -> any:
    """
    Validate array has expected shape/dimensions.

    Checks that an array-like object (numpy array, tensor, nested list) matches the expected shape.
    Use None in expected_shape for dimensions that can be any size (e.g., (None, 3) accepts any
    number of rows with 3 columns). Essential for validating inputs/outputs in neural networks
    and matrix operations.

    Args:
        array: Array-like object with a .shape attribute or nested structure
        expected_shape: Tuple of expected dimensions, use None for flexible dimensions

    Returns:
        The original array if shape matches

    Raises:
        ValueError: If shape doesn't match expected_shape
    """


def validate_unique(collection, key=None) -> any:
    """
    Validate all elements in collection are unique.

    Ensures no duplicate values exist in a collection, which is critical for unique identifiers,
    primary keys, index values, or when building lookup structures. For collections of objects,
    use the key parameter to specify which attribute should be unique (similar to sorted/max).

    Args:
        collection: Iterable collection to check for uniqueness
        key: Optional function to extract comparison value from each element

    Returns:
        The original collection if all elements are unique

    Raises:
        ValueError: If duplicate elements are found
    """


def validate_categorical(value: str, allowed_values: set[str] | list[str]) -> str:
    """
    Validate value is in the set of allowed categorical values.

    Checks that a value belongs to a predefined set of allowed values, which is essential for
    validating categorical features, enum-like fields, or any constrained choice fields in ML
    pipelines. Catches typos, data corruption, or schema violations early before they propagate
    through the pipeline.

    Args:
        value: The value to validate
        allowed_values: Set or list of permitted values

    Returns:
        str: The original value if it's in allowed_values

    Raises:
        ValueError: If value is not in allowed_values
    """


def validate_timestamp(timestamp_str: str, format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Validate timestamp string matches expected format and is parseable.

    Checks that a timestamp string conforms to the specified format using strftime/strptime
    conventions and represents a valid datetime. Handles full date+time validation in one step,
    which is the most common temporal data format in ML pipelines (logs, events, time-series).
    Supports ISO8601, custom formats, and can validate Unix timestamps with format="unix".

    Args:
        timestamp_str: The timestamp string to validate
        format: strftime/strptime format string (default: "%Y-%m-%d %H:%M:%S")
                Use "unix" for Unix timestamp (seconds since epoch)

    Returns:
        str: The original timestamp string if valid and parseable

    Raises:
        ValueError: If timestamp doesn't match format or represents invalid datetime
    """


def validate_timestamp_range(
        timestamp_str: str,
        start: str | None = None,
        end: str | None = None,
        format: str = "%Y-%m-%d %H:%M:%S",
) -> str:
    """
    Validate timestamp falls within specified datetime range.

    Checks that a timestamp (after parsing) falls between start and end datetimes (inclusive).
    Essential for validating temporal boundaries in time-series data, filtering event logs,
    or ensuring train/test temporal splits. All timestamps must use the same format string.

    Args:
        timestamp_str: The timestamp string to validate
        start: Minimum allowed timestamp (inclusive), or None for no lower bound
        end: Maximum allowed timestamp (inclusive), or None for no upper bound
        format: strftime/strptime format string (default: "%Y-%m-%d %H:%M:%S")

    Returns:
        str: The original timestamp string if within range

    Raises:
        ValueError: If timestamp is outside range or parsing fails
    """
