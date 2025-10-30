"""
C108 Data Validators

This module contains validation functions for data formats and values
including emails, IP addresses, language codes, and URLs.

These validators check the **content/format** of values, not their types.
For runtime type validation, see the abc module.
"""

# Standard library -----------------------------------------------------------------------------------------------------
import re


# Methods --------------------------------------------------------------------------------------------------------------

def validate_email(email: str) -> str:
    """
    Validate email address format

    Returns:
        str: The original email if valid
    """
    if not email:
        raise ValueError("Email cannot be empty")

    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        raise ValueError(f"Invalid email format: '{email}'")

    return email


def validate_ip_address(ip: str) -> str:
    """
    Validate IPv4 address format

    Returns:
        str: The original email if valid
    """
    if not ip:
        raise ValueError("IP address cannot be empty")

    parts = ip.split('.')
    if len(parts) != 4:
        raise ValueError(f"Invalid IP address format: '{ip}'")

    for part in parts:
        try:
            num = int(part)
            if not (0 <= num <= 255):
                raise ValueError(f"Invalid IP address format: '{ip}'")
        except ValueError:
            raise ValueError(f"Invalid IP address format: '{ip}'")

    return ip


def validate_language_code(language_code: str, allow_ISO: bool = True, allow_BCP47: bool = True) -> str:
    """
    Validate language code follows ISO 639-1 or BCP 47 format

    Args:
        language_code: The language code to validate
        allow_ISO: Whether to allow ISO 639-1 format (e.g., 'en', 'fr')
        allow_BCP47: Whether to allow BCP 47 format (e.g., 'en-US', 'fr-CA')

    Returns:
        str: The original language code if valid

    Raises:
        ValueError: If language code is invalid or format not allowed
    """
    if not language_code:
        raise ValueError("Language code cannot be empty")

    if not allow_ISO and not allow_BCP47:
        raise ValueError("At least one format (ISO or BCP47) must be allowed")

    # Convert to lowercase for validation
    lang_lower = language_code.lower()

    # ISO 639-1 two-letter codes - ALL Countries
    iso_639_1_codes = {
        'af', 'al', 'dz', 'as', 'ad', 'ao', 'ai', 'aq', 'ag', 'ar', 'am', 'aw', 'au', 'at', 'az',
        'bs', 'bh', 'bd', 'bb', 'by', 'be', 'bz', 'bj', 'bm', 'bt', 'bo', 'bq', 'ba', 'bw', 'bv', 'br', 'io', 'bn',
        'bg', 'bf', 'bi', 'cv', 'kh', 'cm', 'ca', 'ky', 'cf', 'td', 'cl', 'cn', 'cx', 'cc', 'co', 'km', 'cd', 'cg',
        'ck', 'cr', 'hr', 'cu', 'cw', 'cy', 'cz', 'ci', 'dk', 'dj', 'dm', 'do', 'ec', 'eg', 'sv', 'gq', 'er', 'ee',
        'sz', 'et', 'fk', 'fo', 'fj', 'fi', 'fr', 'gf', 'pf', 'tf', 'ga', 'gm', 'ge', 'de', 'gh', 'gi', 'gr', 'gl',
        'gd', 'gp', 'gu', 'gt', 'gg', 'gn', 'gw', 'gy', 'ht', 'hm', 'va', 'hn', 'hk', 'hu', 'is', 'in', 'id', 'ir',
        'iq', 'ie', 'im', 'il', 'it', 'jm', 'jp', 'je', 'jo', 'kz', 'ke', 'ki', 'kp', 'kr', 'kw', 'kg', 'la', 'lv',
        'lb', 'ls', 'lr', 'ly', 'li', 'lt', 'lu', 'mo', 'mg', 'mw', 'my', 'mv', 'ml', 'mt', 'mh', 'mq', 'mr', 'mu',
        'yt', 'mx', 'fm', 'md', 'mc', 'mn', 'me', 'ms', 'ma', 'mz', 'mm', 'na', 'nr', 'np', 'nl', 'nc', 'nz', 'ni',
        'ne', 'ng', 'nu', 'nf', 'mp', 'no', 'om', 'pk', 'pw', 'ps', 'pa', 'pg', 'py', 'pe', 'ph', 'pn', 'pl', 'pt',
        'pr', 'qa', 'mk', 'ro', 'ru', 'rw', 're', 'bl', 'sh', 'kn', 'lc', 'mf', 'pm', 'vc', 'ws', 'sm', 'st', 'sa',
        'sn', 'rs', 'sc', 'sl', 'sg', 'sx', 'sk', 'si', 'sb', 'so', 'za', 'gs', 'ss', 'es', 'lk', 'sd', 'sr', 'sj',
        'se', 'ch', 'sy', 'tw', 'tj', 'tz', 'th', 'tl', 'tg', 'tk', 'to', 'tt', 'tn', 'tr', 'tm', 'tc', 'tv', 'ug',
        'ua', 'ae', 'gb', 'um', 'us', 'uy', 'uz', 'vu', 've', 'vn', 'vg', 'vi', 'wf', 'eh', 'ye', 'zm', 'zw', 'ax'
    }
    # BCP 47 format (language-region) codes - Core Selection of Languages (Major Languages only)
    bcp_47_codes = {
        'af-na', 'af-za',
        'ar-ae', 'ar-bh', 'ar-dj', 'ar-dz', 'ar-eg', 'ar-er', 'ar-iq', 'ar-jo', 'ar-km', 'ar-kw', 'ar-lb', 'ar-ly',
        'ar-ma', 'ar-mr', 'ar-om', 'ar-qa', 'ar-sa', 'ar-sd', 'ar-so', 'ar-sy', 'ar-td', 'ar-tn', 'ar-ye',
        'az-az',
        'be-by',
        'bg-bg',
        'bn-bd', 'bn-in',
        'ca-es',
        'cs-cz',
        'da-dk', 'da-gl',
        'de-at', 'de-ch', 'de-de', 'de-li', 'de-lu',
        'el-cy', 'el-gr',
        'en-au', 'en-ca', 'en-gb', 'en-ie', 'en-in', 'en-nz', 'en-us', 'en-za',
        'es-ar', 'es-bo', 'es-cl', 'es-co', 'es-cr', 'es-cu', 'es-do', 'es-ec', 'es-es', 'es-gt', 'es-hn', 'es-mx',
        'es-ni', 'es-pa', 'es-pe', 'es-pr', 'es-py', 'es-sv', 'es-uy', 'es-ve',
        'et-ee',
        'eu-es',
        'fi-fi',
        'fr-be', 'fr-ca', 'fr-ch', 'fr-fr', 'fr-lu', 'fr-mc',
        'ga-ie',
        'gl-es',
        'gu-in',
        'he-il',
        'hi-in',
        'hr-ba', 'hr-hr',
        'hu-hu',
        'hy-am',
        'id-id',
        'is-is',
        'it-ch', 'it-it', 'it-sm', 'it-va',
        'ja-jp',
        'ka-ge',
        'kk-kz',
        'kn-in',
        'ko-kp', 'ko-kr',
        'ky-kg',
        'lt-lt',
        'lv-lv',
        'mk-mk',
        'ml-in',
        'mr-in',
        'ms-bn', 'ms-my', 'ms-sg',
        'mt-mt',
        'nb-no',
        'nl-be', 'nl-nl', 'nl-sr',
        'nn-no',
        'no-no',
        'nr-za',
        'nso-za',
        'or-in',
        'pa-in', 'pa-pk',
        'pl-pl',
        'pt-ao', 'pt-br', 'pt-cv', 'pt-gw', 'pt-mz', 'pt-pt', 'pt-st', 'pt-tl',
        'rm-ch',
        'ro-md', 'ro-ro',
        'ru-by', 'ru-kg', 'ru-kz', 'ru-md', 'ru-ru', 'ru-ua',
        'si-lk',
        'sk-sk',
        'sl-si',
        'sq-al',
        'sr-ba', 'sr-me', 'sr-rs', 'sr-xk',
        'ss-sz', 'ss-za',
        'st-ls', 'st-za',
        'sv-fi', 'sv-se',
        'sw-bi', 'sw-cd', 'sw-ke', 'sw-km', 'sw-mg', 'sw-mw', 'sw-mz', 'sw-rw', 'sw-tz', 'sw-ug', 'sw-zm',
        'ta-in', 'ta-lk', 'ta-my', 'ta-sg',
        'te-in',
        'tg-tj',
        'th-th',
        'tk-tm',
        'tl-ph',
        'tn-bw', 'tn-za',
        'tr-cy', 'tr-tr',
        'ts-mz', 'ts-za',
        'uk-ua',
        'ur-in', 'ur-pk',
        'uz-uz',
        've-za',
        'vi-vn',
        'xh-za',
        'zh-cn', 'zh-hk', 'zh-mo', 'zh-sg', 'zh-tw',
        'zu-za'
    }
    # Check if it's a known ISO code
    is_iso_code = lang_lower in iso_639_1_codes
    # Check if it's a known BCP 47 code
    is_bcp47_code = lang_lower in bcp_47_codes

    # If exact match found, check if format is allowed
    if is_iso_code:
        if not allow_ISO:
            raise ValueError(f"ISO 639-1 format not allowed: '{language_code}'. "
                             f"Use BCP 47 format (e.g., 'en-US') instead.")
        return language_code

    if is_bcp47_code:
        if not allow_BCP47:
            raise ValueError(f"BCP 47 format not allowed: '{language_code}'. "
                             f"Use ISO 639-1 format (e.g., 'en') instead.")
        return language_code

    # If no exact match, check patterns for flexibility
    # Pattern for ISO 639-1 (2 letters)
    iso_639_1_pattern = r'^[a-z]{2}$'
    # Pattern for BCP 47 (language-region: 2-5 letters, dash, 2-3 letters/digits)
    bcp_47_pattern = r'^[a-z]{2,5}-[a-z0-9]{2,3}$'

    matches_iso_pattern = re.match(iso_639_1_pattern, lang_lower)
    matches_bcp47_pattern = re.match(bcp_47_pattern, lang_lower)

    # Check pattern matches against allowed formats
    if matches_iso_pattern:
        if not allow_ISO:
            raise ValueError(f"ISO 639-1 format not allowed: '{language_code}'. "
                             f"Use BCP 47 format (e.g., 'en-US') instead.")
        return language_code

    if matches_bcp47_pattern:
        if not allow_BCP47:
            raise ValueError(f"BCP 47 format not allowed: '{language_code}'. "
                             f"Use ISO 639-1 format (e.g., 'en') instead.")
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
    """
    if not url:
        raise ValueError("URL cannot be empty")

    url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    if not re.match(url_pattern, url):
        raise ValueError(f"Invalid URL format: '{url}'")

    return url
