#
# Unicode Tools test suite
#

import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.unicode import to_sub, to_sup


class TestToSub:
    """Tests for the to_sub conversion to Unicode subscripts."""

    @pytest.mark.parametrize(
        "input_value, expected",
        [
            pytest.param(0, "₀", id="zero"),
            pytest.param(1, "₁", id="one"),
            pytest.param(2, "₂", id="two"),
            pytest.param(3, "₃", id="three"),
            pytest.param(456789, "₄₅₆₇₈₉", id="multi-digit"),
        ],
    )
    def test_digits_basic(self, input_value: int, expected: str) -> None:
        """Convert basic positive digits to subscript."""
        assert to_sub(input_value) == expected

    @pytest.mark.parametrize(
        "input_value, expected",
        [
            pytest.param(-1, "₋₁", id="minus-one"),
            pytest.param(-203, "₋₂₀₃", id="minus-multi-digit"),
        ],
    )
    def test_digits_negative(self, input_value: int, expected: str) -> None:
        """Convert negative integer to subscript with minus sign."""
        assert to_sub(input_value) == expected

    @pytest.mark.parametrize(
        "input_value, expected",
        [
            pytest.param(0, "₀", id="int-zero"),
            pytest.param("0", "₀", id="str-zero"),
        ],
    )
    def test_digits_zero(self, input_value: int | str, expected: str) -> None:
        """Convert zero to subscript zero."""
        assert to_sub(input_value) == expected

    @pytest.mark.parametrize(
        "input_value, expected",
        [
            pytest.param(12, "₁₂", id="int"),
            pytest.param(12.0, "₁₂.₀", id="float"),
            pytest.param("12", "₁₂", id="str-digits"),
            pytest.param("-7.5", "₋₇.₅", id="str-float"),
        ],
    )
    def test_type_conversion_int_float_str(
        self, input_value: int | float | str, expected: str
    ) -> None:
        """Accept int, float, and str inputs and convert consistently."""
        assert to_sub(input_value) == expected

    @pytest.mark.parametrize(
        "input_value, expected",
        [
            pytest.param("@", "@", id="at-sign"),
            pytest.param("#", "#", id="hash"),
            pytest.param("x_y", "ₓ_y", id="underscore-passthrough"),
            pytest.param("Q!", "Q!", id="unsupported-upper-and-exclam"),
        ],
    )
    def test_unsupported_chars_passthrough(
        self, input_value: str, expected: str
    ) -> None:
        """Leave unsupported characters unchanged."""
        assert to_sub(input_value) == expected

    def test_empty_string(self) -> None:
        """Return empty string for empty input string."""
        assert to_sub("") == ""

    @pytest.mark.parametrize(
        "input_value, expected",
        [
            pytest.param("x2+3", "ₓ₂₊₃", id="letters-digits-plus"),
            pytest.param("(n+1)", "₍ₙ₊₁₎", id="parens-n-plus-one"),
            pytest.param("a^2+b^2", "ₐ^₂₊b^₂", id="caret-unchanged"),
        ],
    )
    def test_mixed_letters_digits_ops(self, input_value: str, expected: str) -> None:
        """Convert mixed letters, digits, and operators correctly."""
        assert to_sub(input_value) == expected

    @pytest.mark.parametrize(
        "input_value, expected",
        [
            pytest.param("(1+2)=3", "₍₁₊₂₎₌₃", id="parens-plus-equals"),
            pytest.param("-5*(2-1)", "₋₅*₍₂₋₁₎", id="asterisk-unchanged"),
        ],
    )
    def test_parentheses_and_operators(self, input_value: str, expected: str) -> None:
        """Convert parentheses and operators to subscript variants."""
        assert to_sub(input_value) == expected

    @pytest.mark.parametrize(
        "input_value, expected",
        [
            pytest.param("abcxyz", "ₐbcₓyz", id="partial-abcxyz"),
            pytest.param("faq", "fₐq", id="partial-faq"),
        ],
    )
    def test_partial_letter_support_lower(
        self, input_value: str, expected: str
    ) -> None:
        """Convert supported lowercase letters and pass unsupported ones."""
        assert to_sub(input_value) == expected

    @pytest.mark.parametrize(
        "input_value, expected",
        [
            pytest.param("CO2", "CO₂", id="partial-CO2"),
            pytest.param("ABC", "ABC", id="pass-ABC"),
            pytest.param("QWERTY", "QWERTY", id="pass-QWERTY"),
        ],
    )
    def test_partial_letter_support_upper(
        self, input_value: str, expected: str
    ) -> None:
        """Convert supported uppercase letters and pass unsupported ones."""
        assert to_sub(input_value) == expected

    @pytest.mark.parametrize(
        "input_value, expected_contains",
        [
            pytest.param(3.1415, "₃.₁₄₁₅", id="float-pi-like"),
            pytest.param("0.001", "₀.₀₀₁", id="str-leading-zeros"),
        ],
    )
    def test_float_string_representation(
        self, input_value: float | str, expected_contains: str
    ) -> None:
        """Handle float string representation without errors."""
        result = to_sub(input_value)
        assert expected_contains in result

    @pytest.mark.parametrize(
        "input_value, expected",
        [
            pytest.param("α2β", "α₂β", id="greek-with-digit"),
            pytest.param("你好3", "你好₃", id="chinese-with-digit"),
            pytest.param("ñ + 1", "ñ ₊ ₁", id="latin-extended-with-ops"),
        ],
    )
    def test_non_ascii_input_passthrough(self, input_value: str, expected: str) -> None:
        """Preserve non-ASCII characters outside mapping."""
        assert to_sub(input_value) == expected

    @pytest.mark.parametrize(
        "input_value, expected",
        [
            pytest.param("  x2  ", "  ₓ₂  ", id="leading-trailing-spaces"),
            pytest.param("a b c 1 2", "ₐ b c ₁ ₂", id="multiple-spaces-preserved"),
        ],
    )
    def test_whitespace_preservation(self, input_value: str, expected: str) -> None:
        """Preserve spaces and surrounding whitespace."""
        assert to_sub(input_value) == expected

    @pytest.mark.parametrize(
        "repeat_count",
        [
            pytest.param(1000, id="1k"),
        ],
    )
    def test_large_number_performance_sanity(self, repeat_count: int) -> None:
        """Process long numeric string without degradation."""
        input_value: str = "1234567890" * repeat_count
        expected: str = "₁₂₃₄₅₆₇₈₉₀" * repeat_count
        assert to_sub(input_value) == expected


class TestToSup:
    @pytest.mark.parametrize(
        "input_value, expected",
        [
            pytest.param(0, "⁰", id="zero"),
            pytest.param(1, "¹", id="one"),
            pytest.param(2, "²", id="two"),
            pytest.param(3, "³", id="three"),
            pytest.param(456, "⁴⁵⁶", id="multi-digit"),
        ],
    )
    def test_digits_basic(self, input_value: int, expected: str) -> None:
        """Convert basic positive digits to superscript."""
        assert to_sup(input_value) == expected

    @pytest.mark.parametrize(
        "input_value, expected",
        [
            pytest.param(-1, "⁻¹", id="minus-one"),
            pytest.param(-203, "⁻²⁰³", id="minus-multi-digit"),
        ],
    )
    def test_digits_negative(self, input_value: int, expected: str) -> None:
        """Convert negative integer to superscript with minus sign."""
        assert to_sup(input_value) == expected

    @pytest.mark.parametrize(
        "input_value, expected",
        [
            pytest.param(0, "⁰", id="int-zero"),
            pytest.param("0", "⁰", id="str-zero"),
        ],
    )
    def test_digits_zero(self, input_value: int | str, expected: str) -> None:
        """Convert zero to superscript zero."""
        assert to_sup(input_value) == expected

    @pytest.mark.parametrize(
        "input_value, expected",
        [
            pytest.param(12, "¹²", id="int"),
            pytest.param(12.0, "¹².⁰", id="float"),
            pytest.param("12", "¹²", id="str-digits"),
            pytest.param("-7.5", "⁻⁷.⁵", id="str-float"),
        ],
    )
    def test_type_conversion_int_float_str(
        self, input_value: int | float | str, expected: str
    ) -> None:
        """Accept int, float, and str inputs and convert consistently."""
        assert to_sup(input_value) == expected

    @pytest.mark.parametrize(
        "input_value, expected",
        [
            pytest.param("@", "@", id="at-sign"),
            pytest.param("#", "#", id="hash"),
            pytest.param("x_y", "ˣ_ʸ", id="underscore-passthrough"),
            pytest.param("Q!", "Q!", id="unsupported-upper-and-exclam"),
        ],
    )
    def test_unsupported_chars_passthrough(
        self, input_value: str, expected: str
    ) -> None:
        """Leave unsupported characters unchanged."""
        assert to_sup(input_value) == expected

    def test_empty_string(self) -> None:
        """Return empty string for empty input string."""
        assert to_sup("") == ""

    @pytest.mark.parametrize(
        "input_value, expected",
        [
            pytest.param("x2+3", "ˣ²⁺³", id="letters-digits-plus"),
            pytest.param("(n+1)", "⁽ⁿ⁺¹⁾", id="parens-n-plus-one"),
            pytest.param("a^2+b^2", "ᵃ^²⁺ᵇ^²", id="caret-unchanged"),
        ],
    )
    def test_mixed_letters_digits_ops(self, input_value: str, expected: str) -> None:
        """Convert mixed letters, digits, and operators correctly."""
        assert to_sup(input_value) == expected

    @pytest.mark.parametrize(
        "input_value, expected",
        [
            pytest.param("(1+2)=3", "⁽¹⁺²⁾⁼³", id="parens-plus-equals"),
            pytest.param("-5*(2-1)", "⁻⁵*⁽²⁻¹⁾", id="asterisk-unchanged"),
        ],
    )
    def test_parentheses_and_operators(self, input_value: str, expected: str) -> None:
        """Convert parentheses and operators to superscript variants."""
        assert to_sup(input_value) == expected

    @pytest.mark.parametrize(
        "input_value, expected",
        [
            pytest.param("abcxyz", "ᵃᵇᶜˣʸᶻ", id="supported-lowercase"),
            pytest.param("faq", "ᶠᵃq", id="partial-faq"),
        ],
    )
    def test_partial_letter_support_lower(
        self, input_value: str, expected: str
    ) -> None:
        """Convert supported lowercase letters and pass unsupported ones."""
        assert to_sup(input_value) == expected

    @pytest.mark.parametrize(
        "input_value, expected",
        [
            pytest.param("CO2", "Cᴼ²", id="partial-CO2"),
            pytest.param("ABC", "ᴬᴮC", id="partial-ABC"),
            pytest.param("QWERTY", "QᵂᴱᴿᵀY", id="partial-QWERTY"),
        ],
    )
    def test_partial_letter_support_upper(
        self, input_value: str, expected: str
    ) -> None:
        """Convert supported uppercase letters and pass unsupported ones."""
        assert to_sup(input_value) == expected

    @pytest.mark.parametrize(
        "input_value, expected_contains",
        [
            pytest.param(3.1415, "³.¹⁴¹⁵", id="float-pi-like-no-dot"),
            pytest.param("0.001", "⁰.⁰⁰¹", id="str-leading-zeros-no-dot"),
        ],
    )
    def test_float_string_representation(
        self, input_value: float | str, expected_contains: str
    ) -> None:
        """Handle float string representation without errors."""
        result = to_sup(input_value)
        assert expected_contains in result

    @pytest.mark.parametrize(
        "input_value, expected",
        [
            pytest.param("α2β", "α²β", id="greek-with-digit"),
            pytest.param("你好3", "你好³", id="chinese-with-digit"),
            pytest.param("ñ + 1", "ñ ⁺ ¹", id="latin-extended-with-ops"),
        ],
    )
    def test_non_ascii_input_passthrough(self, input_value: str, expected: str) -> None:
        """Preserve non-ASCII characters outside mapping."""
        assert to_sup(input_value) == expected

    @pytest.mark.parametrize(
        "input_value, expected",
        [
            pytest.param("  x2  ", "  ˣ²  ", id="leading-trailing-spaces"),
            pytest.param("a b c 1 2", "ᵃ ᵇ ᶜ ¹ ²", id="multiple-spaces-preserved"),
        ],
    )
    def test_whitespace_preservation(self, input_value: str, expected: str) -> None:
        """Preserve spaces and surrounding whitespace."""
        assert to_sup(input_value) == expected

    @pytest.mark.parametrize(
        "repeat_count",
        [
            pytest.param(1000, id="1k"),
        ],
    )
    def test_large_number_performance_sanity(self, repeat_count: int) -> None:
        """Process long numeric string without degradation."""
        input_value: str = "1234567890" * repeat_count
        expected: str = "¹²³⁴⁵⁶⁷⁸⁹⁰" * repeat_count
        assert to_sup(input_value) == expected
