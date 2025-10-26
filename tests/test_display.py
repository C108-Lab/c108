#
# C108 - Display Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import math
from dataclasses import FrozenInstanceError
from inspect import stack

# Third-party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.dictify import dictify
from c108.display import DisplayFormat, DisplayValue, DisplayMode, MultSymbol, DisplaySymbols, DisplayScale, DisplayFlow
from c108.display import trimmed_digits, trimmed_round


# Tests ----------------------------------------------------------------------------------------------------------------

class TestDisplayFormat:
    """Core tests for DisplayFormat covering validation, formatting, and errors."""

    @pytest.mark.parametrize(
        "mult,expected",
        [
            pytest.param("caret", "10^3", id="caret_format"),
            pytest.param("latex", "10^{3}", id="latex_format"),
            pytest.param("python", "10**3", id="python_format"),
            pytest.param("unicode", "10³", id="unicode_format"),
        ],
    )
    def test_mult_exp_valid(self, mult, expected):
        """Format exponent according to selected style."""
        fmt = DisplayFormat(mult=mult)
        result = fmt.mult_exp(base=10, power=3)
        assert result == expected

    @pytest.mark.parametrize(
        "mult,base,power,expected",
        [
            pytest.param("caret", 2, 5, "2^5", id="caret_binary"),
            pytest.param("latex", 2, 5, "2^{5}", id="latex_binary"),
            pytest.param("python", 2, 5, "2**5", id="python_binary"),
            pytest.param("unicode", 2, 5, "2⁵", id="unicode_binary"),
        ],
    )
    def test_mult_exp_with_custom_base(self, mult, base, power, expected):
        """Format exponent for non-decimal bases."""
        fmt = DisplayFormat(mult=mult)
        assert fmt.mult_exp(base=base, power=power) == expected

    def test_mult_exp_zero_power(self):
        """Return empty string when power is zero."""
        fmt = DisplayFormat(mult="caret")
        assert fmt.mult_exp(base=10, power=0) == ""

    def test_invalid_mult_raises_valueerror(self):
        """Raise ValueError for unsupported mult format."""
        with pytest.raises(ValueError, match=r"(?i).*expected one of.*but found.*"):
            DisplayFormat(mult="invalid")

    @pytest.mark.parametrize(
        "base,power,err_type,match",
        [
            pytest.param("10", 3, TypeError, r"(?i).*base must be an int.*", id="nonint_base"),
            pytest.param(10, "3", TypeError, r"(?i).*power must be an int.*", id="nonint_power"),
        ],
    )
    def test_mult_exp_type_errors(self, base, power, err_type, match):
        """Raise TypeError when base or power is non-integer."""
        fmt = DisplayFormat(mult="python")
        with pytest.raises(err_type, match=match):
            fmt.mult_exp(base=base, power=power)

    @pytest.mark.parametrize(
        "initial,override,expected",
        [
            pytest.param("caret", "latex", "latex", id="override_latex"),
            pytest.param("latex", "python", "python", id="override_python"),
            pytest.param("python", "unicode", "unicode", id="override_unicode"),
            pytest.param("unicode", "caret", "caret", id="override_caret"),
        ],
    )
    def test_merge_override_mult(self, initial, override, expected):
        """Return new instance with overridden mult."""
        fmt = DisplayFormat(mult=initial)
        merged = fmt.merge(mult=override)
        assert merged.mult == expected
        assert merged is not fmt  # Ensure new instance is returned

    @pytest.mark.parametrize(
        "initial",
        [
            pytest.param("caret", id="keep_caret"),
            pytest.param("latex", id="keep_latex"),
            pytest.param("python", id="keep_python"),
            pytest.param("unicode", id="keep_unicode"),
        ],
    )
    def test_merge_inherit_mult(self, initial):
        """Inherit mult when UNSET is passed."""
        fmt = DisplayFormat(mult=initial)
        merged = fmt.merge()
        assert merged.mult == initial
        assert merged is not fmt


class TestDisplaySymbols:
    @pytest.mark.parametrize(
        ("attr", "expected"),
        [
            pytest.param("nan", "NaN", id="nan"),
            pytest.param("none", "None", id="none"),
            pytest.param("pos_infinity", "inf", id="pos_infinity"),
            pytest.param("neg_infinity", "-inf", id="neg_infinity"),
            pytest.param("pos_underflow", "0", id="pos_underflow"),
            pytest.param("neg_underflow", "-0", id="neg_underflow"),
            pytest.param("mult", MultSymbol.ASTERISK, id="mult"),
        ]
    )
    def test_ascii_values(self, attr: str, expected) -> None:
        """Verify ASCII factory returns expected symbols."""
        symbols = DisplaySymbols.ascii()
        assert getattr(symbols, attr) == expected

    @pytest.mark.parametrize(
        ("attr", "expected"),
        [
            pytest.param("nan", "NaN", id="nan"),
            pytest.param("none", "None", id="none"),
            pytest.param("pos_infinity", "+∞", id="pos_infinity"),
            pytest.param("neg_infinity", "−∞", id="neg_infinity"),
            pytest.param("pos_underflow", "≈0", id="pos_underflow"),
            pytest.param("neg_underflow", "≈0", id="neg_underflow"),
            pytest.param("mult", MultSymbol.CROSS, id="mult"),
        ])
    def test_unicode_values(self, attr: str, expected) -> None:
        """Verify Unicode factory returns expected symbols."""
        symbols = DisplaySymbols.unicode()
        assert getattr(symbols, attr) == expected

    def test_unicode_underflow_equal(self) -> None:
        """Ensure Unicode uses same underflow symbol for both signs."""
        symbols = DisplaySymbols.unicode()
        assert symbols.pos_underflow == "≈0"
        assert symbols.neg_underflow == "≈0"
        assert symbols.pos_underflow == symbols.neg_underflow

    def test_frozen_assign(self) -> None:
        """Enforce immutability by preventing attribute assignment."""
        symbols = DisplaySymbols.ascii()
        with pytest.raises(FrozenInstanceError, match=r"(?i).*assign.*"):
            symbols.nan = "changed"  # type: ignore[assignment]

    def test_factories_distinct(self) -> None:
        """Return distinct but equal instances for factory calls."""
        a1 = DisplaySymbols.ascii()
        a2 = DisplaySymbols.ascii()
        u1 = DisplaySymbols.unicode()
        u2 = DisplaySymbols.unicode()
        assert a1 is not a2
        assert u1 is not u2
        assert a1 == a2
        assert u1 == u2


class TestDisplayValueMode:
    @pytest.mark.parametrize(
        "mult_exp, unit_exp, expected_mode",
        [
            pytest.param(0, 0, DisplayMode.PLAIN, id="plain"),
            pytest.param(0, 3, DisplayMode.FIXED, id="0-3-fixed"),
            pytest.param(3, 0, DisplayMode.FIXED, id="3-0-fixed"),
            pytest.param(None, 0, DisplayMode.BASE_FIXED, id="base-fixed"),
            pytest.param(None, 3, DisplayMode.UNIT_FIXED, id="unit-fixed"),
            pytest.param(0, None, DisplayMode.UNIT_FLEX, id="exp-0-unit-flex"),
            pytest.param(3, None, DisplayMode.UNIT_FLEX, id="exp-3-unit-flex"),
            pytest.param(None, None, DisplayMode.BASE_FIXED, id="nones-base-fixed"),
        ],
    )
    def test_infer_display_mode(self, mult_exp, unit_exp, expected_mode):
        """Infer DisplayMode from exponents."""
        dv = DisplayValue(123, mult_exp=mult_exp, unit_exp=unit_exp)
        assert dv.mode == expected_mode


class TestDisplayValue__str__:
    @pytest.mark.parametrize(
        "value, mult_exp, unit_exp, expected_str",
        [
            pytest.param(123, 0, 0, "123 B", id="plain"),
            pytest.param(123, 0, 3, "0.123 kB", id="0-3-fixed"),
            pytest.param(123, 3, 0, "0.123×10³ B", id="3-0-fixed"),
            pytest.param(123000, None, 0, "123×10³ B", id="base-fixed"),
            pytest.param(123000, None, 3, "123 kB", id="unit-fixed"),
            pytest.param(123000, 0, None, "123 kB", id="exp-0-unit-flex"),
            pytest.param(123 * 10 ** 6, 3, None, "123×10³ kB", id="exp-3-unit-flex"),
            pytest.param(123, None, None, "123 B", id="nones-base-fixed"),
        ],
    )
    def test_display_value_decimal(self, value, mult_exp, unit_exp, expected_str):
        """Infer DisplayMode from mult_exp/unit_exp pair, return proper decimal-scale str."""
        dv = DisplayValue(value, unit="B", mult_exp=mult_exp, unit_exp=unit_exp, scale=DisplayScale(type="decimal"))
        assert str(dv) == expected_str

    @pytest.mark.parametrize(
        "value, mult_exp, unit_exp, expected_str",
        [
            pytest.param(123, 0, 0, "123 B",
                         id="0-0-plain"),
            pytest.param(123, 0, 10, "0.12 KiB",
                         id="0-10-fixed"),
            pytest.param(0.123 * 2 ** 30, 30, 0, "0.123×2³⁰ B",
                         id="30-0-fixed"),
            pytest.param(123 * 1024, None, 0, "123×2¹⁰ B",
                         id="base-fixed"),
            pytest.param(123 * 1024, None, 10, "123 KiB",
                         id="unit-fixed"),
            pytest.param(123 * 1024, 0, None, "123 KiB",
                         id="exp-0-unit-flex"),
            pytest.param(1 * 2 ** 40, 20, None, "1×2²⁰ MiB",
                         id="exp-20-unit-flex"),
            pytest.param(1 * 2 ** 40, 38, None, "4×2³⁸ B",
                         id="exp-38-unit-flex"),
            pytest.param(123, None, None, "123 B",
                         id="nones-base-fixed"),
        ],
    )
    def test_display_value_binary(self, value, mult_exp, unit_exp, expected_str):
        """Infer DisplayMode from mult_exp/unit_exp pair, return proper binary scale str."""
        dv = DisplayValue(
            value,
            unit="B",
            mult_exp=mult_exp,
            unit_exp=unit_exp,
            scale=DisplayScale(type="binary"),
        )
        assert str(dv) == expected_str


class TestDisplayValueNormalized:

    @pytest.mark.parametrize(
        "value, unit, expected",
        [
            pytest.param(10 ** -100, "B", 1e-76, id="tiny-underflow++"),
            pytest.param(-10 ** -100, "B", -1e-76, id="tiny-underflow--"),
            pytest.param(1, "B", 1, id="normal"),
            pytest.param(1e100, "B", 1e70, id="huge-overflow++"),
            pytest.param(-1e100, "B", -1e70, id="huge-overflow--"),
        ],
    )
    def test_normalized_unitflex(self, value, unit, expected):
        dv = DisplayValue(value, mult_exp=0, unit=unit)
        assert dv.normalized == pytest.approx(expected, rel=1e-9, abs=0.0)


class TestDisplayValueOverUnderflowFormatting:

    @pytest.mark.parametrize(
        "value, unit, expected_str",
        [
            pytest.param(1e-100, "B", "+0 yB", id="tiny-underflow++"),
            pytest.param(-1e-100, "B", "-0 yB", id="tiny-underflow--"),
            pytest.param(1, "B", "1 B", id="normal"),
            pytest.param(1e100, "B", "+inf QB", id="huge-overflow++"),
            pytest.param(-1e100, "B", "-inf QB", id="huge-overflow--"),
        ],
    )
    def test_overflow_format_unitflex(self, value, unit, expected_str):
        symbols = DisplaySymbols(pos_infinity="+inf", neg_infinity="-inf",
                                 pos_underflow="+0", neg_underflow="-0")
        dv = DisplayValue(value, mult_exp=0, unit=unit, symbols=symbols)
        assert str(dv) == expected_str


class TestTrimmedDigits:
    @pytest.mark.parametrize(
        "number, round_digits, expected",
        [
            pytest.param(123000, 15, 3, id="int_trim_trailing_zeros"),
            pytest.param(100, 15, 1, id="int_single_after_trim"),
            pytest.param(101, 15, 3, id="int_no_trailing_zeros"),
            pytest.param(0, 15, 1, id="int_zero_one_digit"),
            pytest.param(-456000, 15, 3, id="int_negative_ignored_sign"),
        ],
    )
    def test_int_cases(self, number, round_digits, expected):
        """Handle integers with trailing zero trimming."""
        assert trimmed_digits(number, round_digits=round_digits) == expected

    @pytest.mark.parametrize(
        "number, round_digits, expected",
        [
            pytest.param(0.456, 15, 3, id="float_simple"),
            pytest.param(123.456, 15, 6, id="float_all_significant"),
            pytest.param(123.450, 15, 5, id="float_trim_trailing_decimal_zeros"),
            pytest.param(1200.0, 15, 2, id="float_nonstandard_treat_trailing_zeros_non_sig"),
            pytest.param(0.00123, 15, 3, id="float_leading_zeros_not_counted"),
        ],
    )
    def test_float_cases(self, number, round_digits, expected):
        """Handle floats with non-standard trailing zero trimming."""
        assert trimmed_digits(number, round_digits=round_digits) == expected

    @pytest.mark.parametrize(
        "number, round_digits, expected",
        [
            pytest.param(0.1 + 0.2, 15, 1, id="float_artifact_rounded"),
            pytest.param(1 / 3, 15, 15, id="float0.33_rounded_to_ndigits"),
            pytest.param(1e100, 15, 1, id="float1e+100_rounded_to_ndigits"),
            pytest.param(1e-100, 15, 1, id="float1e-100_rounded_to_ndigits"),
        ],
    )
    def test_float_artifacts_with_rounding(self, number, round_digits, expected):
        """Round float artifacts before analysis."""
        assert trimmed_digits(number, round_digits=round_digits) == expected

    @pytest.mark.parametrize(
        "number, round_digits, expected",
        [
            pytest.param(0.1 + 0.2, None, 17, id="no_round_artifacts_kept"),
            pytest.param(1 / 3, 5, 5, id="custom_round_5"),
            pytest.param(1 / 3, 2, 2, id="custom_round_2"),
            pytest.param(1 / 3, 0, 1, id="custom_round_0"),
        ],
    )
    def test_custom_round_digits(self, number, round_digits, expected):
        """Apply custom rounding precision when provided."""
        assert trimmed_digits(number, round_digits=round_digits) == expected

    @pytest.mark.parametrize(
        "number, round_digits",
        [
            pytest.param(None, 15, id="none_input"),
            pytest.param(math.nan, 15, id="nan_input"),
            pytest.param(math.inf, 15, id="pos_inf_input"),
            pytest.param(-math.inf, 15, id="neg_inf_input"),
        ],
    )
    def test_non_numerics_return_none(self, number, round_digits):
        """Return None for non-displayable inputs."""
        assert trimmed_digits(number, round_digits=round_digits) is None

    @pytest.mark.parametrize(
        "number, round_digits, expected",
        [
            pytest.param(-0.0, 15, 1, id="neg_zero"),
            pytest.param(100, 2, 1, id="int_round_digits_ignored"),
        ],
    )
    def test_edge_cases(self, number, round_digits, expected):
        """Handle documented edge cases correctly."""
        assert trimmed_digits(number, round_digits=round_digits) == expected

    @pytest.mark.parametrize(
        "number, round_digits, expected_substring",
        [
            pytest.param("123", 15, "number", id="bad_number_type_str"),
            pytest.param([], 15, "number", id="bad_number_type_list"),
            pytest.param(123, "15", "round_digits", id="bad_round_digits_type_str"),
            pytest.param(1.23, 1.5, "round_digits", id="bad_round_digits_type_float"),
        ],
    )
    def test_type_errors(self, number, round_digits, expected_substring):
        """Raise TypeError for invalid parameter types."""
        with pytest.raises(TypeError, match=rf"(?i).*{expected_substring}.*"):
            trimmed_digits(number, round_digits=round_digits)


class TestTrimmedRound:
    """Test suite for trimmed_round function."""

    @pytest.mark.parametrize(
        "number,trimmed_digits,expected",
        [
            pytest.param(123.456, 3, 123, id="float_3_digits"),
            pytest.param(123.456, 2, 120, id="float_2_digits"),
            pytest.param(123.456, 1, 100, id="float_1_digit"),
            pytest.param(123.456, 5, 123.46, id="float_5_digits"),
            pytest.param(123.456, 6, 123.456, id="float_6_digits"),
            pytest.param(-123.456, 3, -123, id="neg_float_3_digits"),
            pytest.param(-123.456, 2, -120, id="neg_float_2_digits"),
            pytest.param(0.00123, 2, 0.0012, id="small_2_digits"),
            pytest.param(0.00123, 1, 0.001, id="small_1_digit"),
            pytest.param(9.99, 2, 10.0, id="rounds_up_9_99"),
            pytest.param(999, 2, 1000, id="rounds_up_999"),
            pytest.param(0, 1, 0, id="zero_int"),
            pytest.param(0.0, 5, 0.0, id="zero_float"),
            pytest.param(123000, 3, 123000, id="int_3_digits_no_change"),
            pytest.param(123000, 2, 120000, id="int_2_digits"),
            pytest.param(123000, 1, 100000, id="int_1_digit"),
        ],
    )
    def test_rounding_behavior(self, number, trimmed_digits, expected):
        """Round numbers to given significant digits."""
        result = trimmed_round(number=number, trimmed_digits=trimmed_digits)
        assert result == expected

    @pytest.mark.parametrize(
        "number,trimmed_digits,expected_type",
        [
            pytest.param(123.456, 3, float, id="float_to_float_when_no_decimals"),
            pytest.param(123.456, 5, float, id="float_remains_float_with_decimals"),
            pytest.param(100, 2, int, id="int_stays_int"),
        ],
    )
    def test_result_type(self, number, trimmed_digits, expected_type):
        """Preserve or coerce return type as per result precision."""
        result = trimmed_round(number=number, trimmed_digits=trimmed_digits)
        assert isinstance(result, expected_type)

    @pytest.mark.parametrize(
        "number,trimmed_digits,expected",
        [
            pytest.param(None, 3, None, id="number_none_passthrough"),
            pytest.param(123.456, None, 123.456, id="digits_none_passthrough_float"),
            pytest.param(100, None, 100, id="digits_none_passthrough_int"),
            pytest.param(float("inf"), 3, float("inf"), id="inf_passthrough"),
            pytest.param(float("-inf"), 4, float("-inf"), id="neg_inf_passthrough"),
            pytest.param(float("nan"), 2, float("nan"), id="nan_passthrough"),
        ],
    )
    def test_passthrough_values(self, number, trimmed_digits, expected):
        """Return None/NaN/Inf as-is or bypass when digits is None."""
        result = trimmed_round(number=number, trimmed_digits=trimmed_digits)
        if isinstance(expected, float) and math.isnan(expected):
            assert isinstance(result, float) and math.isnan(result)
        else:
            assert result == expected

    @pytest.mark.parametrize(
        "number,trimmed_digits,err,match",
        [
            pytest.param("123", 2, TypeError, r"(?i).*number.*", id="number_str"),
            pytest.param([123], 2, TypeError, r"(?i).*number.*", id="number_list"),
            pytest.param(123.456, "2", TypeError, r"(?i).*trimmed_digits.*", id="digits_str"),
            pytest.param(123.456, 1.5, TypeError, r"(?i).*trimmed_digits.*", id="digits_float"),
        ],
    )
    def test_type_errors(self, number, trimmed_digits, err, match):
        """Reject invalid argument types."""
        with pytest.raises(err, match=match):
            trimmed_round(number=number, trimmed_digits=trimmed_digits)

    @pytest.mark.parametrize(
        "number,trimmed_digits",
        [
            pytest.param(123.456, 0, id="zero_digits"),
            pytest.param(-10, -1, id="negative_digits"),
        ],
    )
    def test_value_errors_on_digits(self, number, trimmed_digits):
        """Reject trimmed_digits less than 1."""
        with pytest.raises(ValueError, match=r"(?i).*trimmed_digits.*"):
            trimmed_round(number=number, trimmed_digits=trimmed_digits)


# Private Methods Tests ------------------------------------------------------------------------------------------------

class Test_AutoMultEponent:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(123e-6, -6, id="0.123->-3"),
            pytest.param(123e-3, -3, id="0.123->-3"),
            pytest.param(123, 0, id="3-digit->0"),
            pytest.param(123456, 3, id="6-digit->3"),
            pytest.param(1234567, 6, id="7-digit->6"),
        ],
    )
    def test_decimal_auto_multiplier_exp(self, value: int, expected: int):
        """Verify decimal auto multiplier exponent."""
        dv = DisplayValue(value, unit_exp=0, scale=DisplayScale(type="decimal"))
        assert dv._mult_exp == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(123, 0, id="lt-1Ki->exp0"),
            pytest.param(2 ** 12, 10, id="ge-1Ki-lt-1Mi->exp10"),
            pytest.param(2 ** 21, 20, id="ge-1Mi->exp20"),
        ],
    )
    def test_binary_auto_multiplier_exp(self, value: int, expected: int):
        """Verify binary auto multiplier exponent with 2^(10N)."""
        dv = DisplayValue(value, unit_exp=0, scale=DisplayScale(type="binary"))
        assert dv._mult_exp == expected


class Test_AutoUnitExponent:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(123e-6, -6, id="0.123->-6"),
            pytest.param(0.123, -3, id="0.123->-3"),
            pytest.param(123, 0, id="3-digit->base"),
            pytest.param(1_234, 3, id="4-digit->k"),
            pytest.param(123_456, 3, id="6-digit->k"),
            pytest.param(1_234_567, 6, id="7-digit->M"),
            pytest.param(123_456_789, 6, id="9-digit->M"),
            pytest.param(1_234_567_890, 9, id="10-digit->G"),
        ],
    )
    def test_decimal_auto_unit_exp(self, value: int, expected: int):
        """Verify decimal auto unit exponent selection with standard SI prefixes."""
        dv = DisplayValue(value, mult_exp=0, scale=DisplayScale(type="decimal"))
        assert dv._unit_exp == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(512, 0, id="lt-1Ki->base"),
            pytest.param(2 ** 10, 10, id="exactly-1Ki->Ki"),
            pytest.param(2 ** 10 * 500, 10, id="500Ki->Ki"),
            pytest.param(2 ** 20, 20, id="exactly-1Mi->Mi"),
            pytest.param(2 ** 20 * 500, 20, id="500Mi->Mi"),
            pytest.param(2 ** 30, 30, id="exactly-1Gi->Gi"),
        ],
    )
    def test_binary_auto_unit_exp(self, value: int, expected: int):
        """Verify binary auto unit exponent selection with IEC prefixes."""
        dv = DisplayValue(value, mult_exp=0, scale=DisplayScale(type="binary"))
        assert dv._unit_exp == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(500, 0, id="500->base"),
            pytest.param(1_000, 3, id="1k->k"),
            pytest.param(999_000, 3, id="999k->k"),
            pytest.param(1_000_000, 6, id="exact-1M"),  # Within scale_step
            pytest.param(10_000_000, 6, id="10M->M"),  # Beyond scale_step from k
            pytest.param(999_000_000, 6, id="999M->M"),
            pytest.param(1_000_000_000, 9, id="exact-1G"),
        ],
    )
    def test_decimal_prefixes_no_gap(self, value: int, expected: int):
        """Verify behavior with gaps in custom unit_prefixes (decimal)."""
        # Custom scale with gap: only base, k, M, G (missing intermediate prefixes)
        custom_prefixes = {0: "", 3: "k", 6: "M", 9: "G"}
        dv = DisplayValue(value, mult_exp=0, unit_prefixes=custom_prefixes, scale=DisplayScale(type="decimal"))
        assert dv._unit_exp == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(-1e30, 9, id="-1e30->9"),
            pytest.param(-100, 0, id="-100->base-0"),
            pytest.param(100, 0, id="+100->base-0"),
            pytest.param(10_000, 0, id="gap-lower-0"),
            pytest.param(100_000, 9, id="gap-upper-1G"),
            pytest.param(1_000_000_000, 9, id="exact-1G"),
            pytest.param(1234567_000_000_000, 9, id="1G->G"),
        ],
    )
    def test_decimal_prefixes_large_gap(self, value: int, expected: int):
        """Verify behavior with large gaps in custom unit_prefixes (decimal)."""
        # Large gap: only base, M, G (missing m, k)
        custom_prefixes = {0: "", 9: "G"}
        dv = DisplayValue(value, mult_exp=0, unit_prefixes=custom_prefixes, scale=DisplayScale(type="decimal"))
        assert dv._unit_exp == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(512, 0, id="512->base"),
            pytest.param(2 ** 10, 10, id="1Ki->Ki"),
            pytest.param(2 ** 19, 10, id="512Ki->Ki"),
            pytest.param(2 ** 20, 20, id="exact-1Mi"),  # Within scale_step
            pytest.param(2 ** 25, 20, id="32Mi->Mi"),  # Beyond scale_step from Ki
            pytest.param(2 ** 30, 30, id="1Gi->Gi"),
        ],
    )
    def test_binary_gap_in_prefixes(self, value: int, expected: int):
        """Verify behavior with gaps in custom unit_prefixes (binary)."""
        # Custom scale with some prefixes
        custom_prefixes = {0: "", 10: "Ki", 20: "Mi", 30: "Gi"}
        dv = DisplayValue(value, mult_exp=0, unit_prefixes=custom_prefixes, scale=DisplayScale(type="binary"))
        assert dv._unit_exp == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(1e-30, 0, id="1e-30->0"),
            pytest.param(1_000, 0, id="1k->0"),
            pytest.param(10_000, 0, id="10k->0"),
            pytest.param(1_000_000, 9, id="1M->9"),
            pytest.param(123_000_000, 9, id="123M->9"),
            pytest.param(1e30, 9, id="1e30->9"),
        ],
    )
    def test_decimal_only_two_prefixes(self, value: int, expected: int):
        """Verify behavior with minimal custom unit_prefixes (only two options)."""
        # Minimal scale: only k and M
        custom_prefixes = {0: "", 9: "G"}
        dv = DisplayValue(value, mult_exp=0, unit_prefixes=custom_prefixes, scale=DisplayScale(type="decimal"))
        assert dv._unit_exp == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(0, 0, id="zero->base"),
            pytest.param(0.0, 0, id="zero-float->base"),
            pytest.param(float('nan'), 0, id="nan->base"),
            pytest.param(float('inf'), 0, id="inf->base"),
            pytest.param(float('-inf'), 0, id="neg-inf->base"),
            pytest.param(None, 0, id="none->base"),
        ],
    )
    def test_non_finite_values(self, value, expected: int):
        """Verify non-finite values always return base unit exponent."""
        dv = DisplayValue(value, mult_exp=0, scale=DisplayScale(type="decimal"))
        assert dv._unit_exp == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(-123, 0, id="neg-3-digit->base"),
            pytest.param(-123_456, 3, id="neg-6-digit->k"),
            pytest.param(-1_234_567, 6, id="neg-7-digit->M"),
        ],
    )
    def test_negative_values(self, value: int, expected: int):
        """Verify negative values use absolute value for unit selection."""
        dv = DisplayValue(value, mult_exp=0, scale=DisplayScale(type="decimal"))
        assert dv._unit_exp == expected


class Test_DisplayValueValidators:

    def test_validates_unit_exp(self):
        with pytest.raises(ValueError, match="unit_exp must be one of SI decimal powers"):
            DisplayValue(123, unit_exp=5, scale=DisplayScale(type="decimal"))
        with pytest.raises(ValueError, match="unit_exp must be one of IEC binary powers"):
            DisplayValue(123, unit_exp=5, scale=DisplayScale(type="binary"))
        with pytest.raises(ValueError, match="unit_exp must be one of decimal powers"):
            DisplayValue(123, mult_exp=0, scale=DisplayScale(type="decimal"), unit_prefixes={0: "", 5: "penta"})
        # Empty unit_prefixes map should fall back to default mapping
        dv = DisplayValue(123, mult_exp=0, scale=DisplayScale(type="decimal"), unit_prefixes={})


class TestOverflowUnderflowPredicates:

    @pytest.mark.parametrize(
        "value, mult_exp, unit, overflow_tolerance, underflow_tolerance, unit_prefixes, expected_overflow, expected_underflow",
        [
            pytest.param(10 ** -100, 0, "B", 5, 6,
                         {-24: "y", 24: "Y"}, False, True, id="tiny-underflow"),
            pytest.param(0.1, 0, "B", 5, 6,
                         {-24: "y", 24: "Y"}, False, False, id="gap-no-flags"),
            pytest.param(10 ** 100, 0, "B", 5, 6,
                         {-24: "y", 24: "Y"}, True, False, id="huge-overflow"),
        ],
    )
    def test_overflows_mode_unitflex(self, value, mult_exp, unit, overflow_tolerance, underflow_tolerance,
                                     unit_prefixes,
                                     expected_overflow, expected_underflow):
        """Parametrize overflow/underflow flags for extreme magnitudes."""
        dv = DisplayValue(value,
                          mult_exp=mult_exp,
                          unit=unit,
                          flow=DisplayFlow(
                              overflow_tolerance=overflow_tolerance,
                              underflow_tolerance=underflow_tolerance),
                          unit_prefixes=unit_prefixes,
                          )
        assert dv.flow.overflow is expected_overflow
        assert dv.flow.underflow is expected_underflow

    @pytest.mark.parametrize(
        "value, unit, overflow_tolerance, underflow_tolerance, expected_overflow, expected_underflow",
        [
            pytest.param(10 ** -100, "B", 5, 6, False, True, id="tiny-underflow"),
            pytest.param(1000, "B", 5, 6, False, False, id="normal-no-flags"),
            pytest.param(10 ** 100, "B", 5, 6, True, False, id="huge-overflow"),
        ],
    )
    def test_overflows_mode_fixed(self, value, unit, overflow_tolerance, underflow_tolerance,
                                  expected_overflow, expected_underflow):
        """Parametrize overflow/underflow flags for extreme magnitudes."""
        dv = DisplayValue(value,
                          mult_exp=3,
                          unit_exp=3,
                          unit=unit,
                          flow=DisplayFlow(
                              overflow_tolerance=overflow_tolerance,
                              underflow_tolerance=underflow_tolerance),
                          )
        print(dv)
        assert dv.flow.overflow is expected_overflow
        assert dv.flow.underflow is expected_underflow

    @pytest.mark.parametrize(
        "value, unit, overflow_tolerance, underflow_tolerance, expected_overflow, expected_underflow",
        [
            pytest.param(10 ** -100, "B", 5, 6, False, False, id="tiny"),
            pytest.param(1000, "B", 5, 6, False, False, id="normal"),
            pytest.param(10 ** 100, "B", 5, 6, False, False, id="huge"),
        ],
    )
    def test_no_overflows_mode_plain(self, value, unit, overflow_tolerance, underflow_tolerance,
                                     expected_overflow, expected_underflow):
        """Parametrize overflow/underflow flags for extreme magnitudes."""
        dv = DisplayValue(value,
                          mult_exp=0,
                          unit_exp=0,
                          unit=unit,
                          flow=DisplayFlow(
                              overflow_tolerance=overflow_tolerance,
                              underflow_tolerance=underflow_tolerance),
                          )
        print(dv)
        assert dv.flow.overflow is expected_overflow
        assert dv.flow.underflow is expected_underflow

    @pytest.mark.parametrize(
        "value, unit, overflow_tolerance, underflow_tolerance, expected_overflow, expected_underflow",
        [
            pytest.param(10 ** -100, "B", 5, 6, False, False, id="tiny"),
            pytest.param(1000, "B", 5, 6, False, False, id="normal"),
            pytest.param(10 ** 100, "B", 5, 6, False, False, id="huge"),
        ],
    )
    def test_no_overflows_mode_base_fixed(self, value, unit, overflow_tolerance, underflow_tolerance,
                                          expected_overflow, expected_underflow):
        """Parametrize overflow/underflow flags for extreme magnitudes."""
        dv = DisplayValue(value,
                          unit_exp=0,
                          unit=unit,
                          flow=DisplayFlow(
                              overflow_tolerance=overflow_tolerance,
                              underflow_tolerance=underflow_tolerance),
                          )
        print(dv)
        assert dv.flow.overflow is expected_overflow
        assert dv.flow.underflow is expected_underflow

    @pytest.mark.parametrize(
        "value, unit, overflow_tolerance, underflow_tolerance, expected_overflow, expected_underflow",
        [
            pytest.param(10 ** -100, "B", 5, 6, False, False, id="tiny"),
            pytest.param(1000, "B", 5, 6, False, False, id="normal"),
            pytest.param(10 ** 100, "B", 5, 6, False, False, id="huge"),
        ],
    )
    def test_no_overflows_mode_unit_fixed(self, value, unit, overflow_tolerance, underflow_tolerance,
                                          expected_overflow, expected_underflow):
        """Parametrize overflow/underflow flags for extreme magnitudes."""
        dv = DisplayValue(value,
                          unit_exp=3,
                          unit=unit,
                          flow=DisplayFlow(
                              overflow_tolerance=overflow_tolerance,
                              underflow_tolerance=underflow_tolerance),
                          )
        print(dv)
        assert dv.flow.overflow is expected_overflow
        assert dv.flow.underflow is expected_underflow


# DEMO-s ---------------------------------------------------------------------------------------------------------------

class Test_DEMO_DisplayValue:
    """
    Test Demos for tutorials
    """
    pass
