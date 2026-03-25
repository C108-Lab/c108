#
# C108 - Ref ISO Tests
#

# Standard library -----------------------------------------------------------------------------------------------------

# Third party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.refs.iso import CountryCodes, CurrencyCodes, LanguageCodes


# Tests ----------------------------------------------------------------------------------------------------------------


class TestCountryCodes:
    def test_all_codes_are_two_lowercase_letters(self):
        """Verify each country code is exactly 2 lowercase letters."""
        for code in CountryCodes.ISO_3166_1_CODES:
            assert len(code) == 2 and code.islower(), f"Invalid country code: {code!r}"

    def test_no_blank_entries(self):
        """Verify no country code is empty or whitespace-only."""
        for code in CountryCodes.ISO_3166_1_CODES:
            assert code.strip(), f"Blank country code found: {code!r}"

    def test_count_within_iso_range(self):
        """Verify country code count is within expected ISO 3166-1 range."""
        assert 240 <= len(CountryCodes.ISO_3166_1_CODES) <= 260


class TestCurrencyCodes:
    def test_all_codes_are_three_lowercase_letters(self):
        """Verify each currency code is exactly 3 lowercase letters."""
        for code in CurrencyCodes.ISO_4217_CODES:
            assert len(code) == 3 and code.islower(), f"Invalid currency code: {code!r}"

    def test_no_blank_entries(self):
        """Verify no currency code is empty or whitespace-only."""
        for code in CurrencyCodes.ISO_4217_CODES:
            assert code.strip(), f"Blank currency code found: {code!r}"

    def test_count_within_iso_range(self):
        """Verify currency code count is within expected ISO 4217 range."""
        assert 150 <= len(CurrencyCodes.ISO_4217_CODES) <= 200


class TestLanguageCodes:
    def test_639_1_all_codes_are_two_lowercase_letters(self):
        """Verify each ISO 639-1 language code is exactly 2 lowercase letters."""
        for code in LanguageCodes.ISO_639_1_CODES:
            assert len(code) == 2 and code.islower(), f"Invalid ISO 639-1 code: {code!r}"

    def test_15924_all_codes_are_four_lowercase_letters(self):
        """Verify each ISO 15924 script code is exactly 4 lowercase letters."""
        for code in LanguageCodes.ISO_15924_CODES:
            assert len(code) == 4 and code.islower(), f"Invalid ISO 15924 code: {code!r}"

    def test_no_blank_entries_in_639_1(self):
        """Verify no ISO 639-1 language code is empty or whitespace-only."""
        for code in LanguageCodes.ISO_639_1_CODES:
            assert code.strip(), f"Blank ISO 639-1 code found: {code!r}"

    def test_no_blank_entries_in_15924(self):
        """Verify no ISO 15924 script code is empty or whitespace-only."""
        for code in LanguageCodes.ISO_15924_CODES:
            assert code.strip(), f"Blank ISO 15924 code found: {code!r}"

    def test_639_1_count_within_iso_range(self):
        """Verify ISO 639-1 code count is within the expected standard range."""
        assert 180 <= len(LanguageCodes.ISO_639_1_CODES) <= 200

    def test_15924_count_within_iso_range(self):
        """Verify ISO 15924 code count is within the expected standard range."""
        assert 200 <= len(LanguageCodes.ISO_15924_CODES) <= 230
