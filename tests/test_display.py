#
# C108 - Display Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import math
from inspect import stack

# Third-party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.dictify import dictify
from c108.display import DisplayValue, DisplayMode, MultSymbol, DisplaySymbols, DisplayScale, DisplayFlow
from c108.display import trimmed_digits, trimmed_round, _disp_power


# Tests ----------------------------------------------------------------------------------------------------------------

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


class TestDisplayValueAsStr:
    @pytest.mark.parametrize(
        "value, mult_exp, unit_exp, expected_str",
        [
            pytest.param(123, 0, 0, "123 B", id="plain"),
            pytest.param(123, 0, 3, "0.123 B", id="0-3-fixed"),
            pytest.param(123, 3, 0, "0.123×10³ B", id="3-0-fixed"),
            pytest.param(123000, None, 0, "123×10³ B", id="base-fixed"),
            pytest.param(123000, None, 3, "123 kB", id="unit-fixed"),
            pytest.param(123000, 0, None, "123 kB", id="exp-0-unit-flex"),
            pytest.param(123 * 10 ** 6, 3, None, "123×10³ kB", id="exp-3-unit-flex"),
            pytest.param(123, None, None, "123 B", id="nones-base-fixed"),
        ],
    )
    def test_infer_display_mode(self, value, mult_exp, unit_exp, expected_str):
        """Infer DisplayMode from exponents."""
        dv = DisplayValue(value, unit="B", mult_exp=mult_exp, unit_exp=unit_exp)
        print("\n", value, mult_exp, unit_exp, dv)
        # assert str(dv) == expected_str


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
        print("\n", value)
        print(dv, " | ", dv.normalized)
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


class Test_DispPower:
    @pytest.mark.parametrize(
        ("base", "power", "fmt", "expected"),
        [
            pytest.param(10, -6, "unicode", "10⁻⁶", id="unicode-neg"),
            pytest.param(2, 3, "caret", "2^3", id="caret-pos"),
            pytest.param(10, 3, "python", "10**3", id="python-pos"),
            pytest.param(2, 20, "latex", "2^{20}", id="latex-pos"),
        ],
    )
    def test_render_modes(self, base: int, power: int, fmt: str, expected: str) -> None:
        """Render power across formats."""
        result = _disp_power(base=base, power=power, format=fmt)
        assert result == expected

    @pytest.mark.parametrize(
        ("base", "fmt"),
        [
            pytest.param(10, "unicode", id="unicode"),
            pytest.param(2, "caret", id="caret"),
            pytest.param(3, "python", id="python"),
            pytest.param(5, "latex", id="latex"),
        ],
    )
    def test_zero_power(self, base: int, fmt: str) -> None:
        """Return empty string for zero power."""
        result = _disp_power(base=base, power=0, format=fmt)
        assert result == ""

    def test_invalid_power_format(self):
        with pytest.raises(ValueError, match="invalid power format"):
            _disp_power(power=123, format="unknown")


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
    pass

    def test_none(self):
        print_method()

        num_unit = DisplayValue(value=None)
        print("DisplayValue(value=None)")
        print("__str__", num_unit)
        print("__repr__", repr(num_unit))

    def test__str__repr__(self):
        print_method()

        num_unit = DisplayValue(value=123.456)
        print("DisplayValue(value=123.456)")
        print("__str__", num_unit)
        print("__repr__", repr(num_unit))
        print(dictify(num_unit, include_properties=True))

    def test_mode_plain(self):
        print_method()

        # @formatter:off
        num_unit = DisplayValue(value=123.456, mult_exp=0, unit_exp=0)
        print(    "DisplayValue(value=123.456, mult_exp=0, unit_exp=0)")
        print(num_unit)
        print(dictify(num_unit, include_properties=True))

        # @formatter:on
        # Check Fields
        assert num_unit.mode == DisplayMode.PLAIN
        assert num_unit.precision == None
        assert num_unit.trim_digits == 6
        assert num_unit.unit == None
        # Check Properties
        assert num_unit.normalized == 123.456
        assert num_unit.ref_value == 1
        assert num_unit._multiplier_str == ""
        assert num_unit.unit_prefix == ""
        assert num_unit.number == "123.456"
        assert num_unit.units == ""
        assert num_unit.__str__() == "123.456"

        # @formatter:off
        num_unit = DisplayValue(value=123.1e+21, mult_exp=0, unit_exp=0)
        print(    "DisplayValue(value=123.1e+21, mult_exp=0, unit_exp=0)")
        print(num_unit)
        print(dictify(num_unit, include_properties=True))
        # @formatter:on
        # Check Properties
        assert num_unit.mode == DisplayMode.PLAIN
        assert num_unit.normalized == 1.231e+23
        assert num_unit.ref_value == 1
        assert num_unit._multiplier_str == ""
        assert num_unit.unit_prefix == ""
        assert num_unit.number == "1.231e+23"
        assert num_unit.units == ""
        assert num_unit.__str__() == "1.231e+23"

    def test_mode_si_fixed(self):
        print_method()

        # @formatter:off
        num_unit = DisplayValue(value=123.456, unit_exp=-3, unit="s", trim_digits=4)
        print(    "DisplayValue(value=123.456, unit_exp=-3, unit='s', trim_digits=4)")
        print(num_unit)
        print(dictify(num_unit, include_properties=True))
        assert num_unit.__str__() == "123.5×10³ ms"

    def test_mode_si_flex(self):
        print_method()

        # @formatter:off
        num_unit = DisplayValue(value=123456, mult_exp=0, trim_digits=4)
        print(    "DisplayValue(value=123456, mult_exp=0, trim_digits=4)")
        print(num_unit)
        print(dictify(num_unit, include_properties=True))
        assert num_unit.__str__() == "123.5k"

    def test_mode_mutliplier(self):
        print_method()

        # @formatter:off
        num_unit = DisplayValue(value=123456, unit_exp=0, unit='s')
        print(    "DisplayValue(value=123456, unit_exp=0, unit='s')")
        print(num_unit)
        print(dictify(num_unit, include_properties=True))
        assert num_unit.__str__() == "123.456×10³ s"


    def test_exponents(self):
        print_method()

        # @formatter:off
        num_unit = DisplayValue(value=123.456, mult_exp=-3, unit='s')
        print(    "DisplayValue(value=123.456, mult_exp=-3, unit='s')")
        print(num_unit)
        print(dictify(num_unit, include_properties=True))

        # @formatter:on
        # Check Properties
        assert num_unit.__str__() == "123.456×10⁻³ ks"

        print()

        num_unit = DisplayValue(value=123456, unit_exp=3, unit='m')
        print("DisplayValue(value=123456789, unit_exp=3, unit='m')")
        print(num_unit)
        print(dictify(num_unit, include_properties=True))

        # @formatter:on
        # Check Properties
        assert num_unit.__str__() == "123.456 km"

    def test_precision(self):
        print_method()

        # @formatter:off
        num_unit = DisplayValue(value=123.67788, precision=2)
        print(    "DisplayValue(value=123.67788, precision=2)")
        print(num_unit)

        print(dictify(num_unit, include_properties=True))
        
        # @formatter:on
        # Check Properties
        assert num_unit.normalized == 123.67788
        assert num_unit.__str__() == "123.68"

        # @formatter:off
        num_unit = DisplayValue(value=123677.888, precision=1, mult_exp=3)
        print(    "DisplayValue(value=123677.888, precision=1, mult_exp=3)")
        print(num_unit)

        print(dictify(num_unit, include_properties=True))
        
        # @formatter:on
        # Check Properties
        assert num_unit.normalized == 123.677888
        assert num_unit.__str__() == "123.7×10³"

    def test_significant_digits_fixed(self):
        print_method()

        # @formatter:off
        num_unit = DisplayValue(value=123.67e6, trim_digits=4)
        print(    "DisplayValue(value=123.67e6, trim_digits=4)")
        # @formatter:on
        print(num_unit)
        print(dictify(num_unit, include_properties=True))
        # Check Properties
        assert num_unit.normalized == 123.7
        assert num_unit.__str__() == "123.7×10⁶"

        num_unit = DisplayValue(value=123.67e6, trim_digits=2)
        print("DisplayValue(value=123.67e6, trim_digits=2)")
        print(num_unit)
        print(dictify(num_unit, include_properties=True))
        # @formatter:on
        # Check Properties
        assert num_unit.normalized == 120
        assert num_unit.__str__() == "120×10⁶"

    def test_significant_digits_flex(self):
        print_method()

        # @formatter:off
        num_unit = DisplayValue(value=123.456)
        print(    "DisplayValue(value=123.456)")
        print(num_unit)
        print("num_unit.trim_digits    :", num_unit.trim_digits)
        
        print(dictify(num_unit, include_properties=True))
        
        # @formatter:on
        assert num_unit.trim_digits == 6

        # @formatter:off
        num_unit = DisplayValue(value=123000.0)
        print(    "DisplayValue(value=123000.0)")
        print(num_unit)
        print("num_unit.trim_digits    :", num_unit.trim_digits)

        print(dictify(num_unit, include_properties=True))
        
        # @formatter:on
        assert num_unit.trim_digits == 3

    def test_unit_pluralization(self):
        assert DisplayValue(value=0, unit="byte", pluralize=True).__str__() == "0 bytes"
        assert DisplayValue(value=1, unit="byte", pluralize=True).__str__() == "1 byte"
        assert DisplayValue(value=2, unit="byte", pluralize=True).__str__() == "2 bytes"
        assert DisplayValue(value=2, unit="plr", pluralize=True, unit_plurals={"plr": "PLR"}).__str__() == "2 PLR"
        # Non-pluralizable unit
        assert DisplayValue(value=2, unit="abc_def", pluralize=True).__str__() == "2 abc_def"

    def test_infinite_values(self):
        print_method()

        # @formatter:off
        num_unit = DisplayValue(None, unit="byte", mult_exp=3)
        print(    "DisplayValue(None, unit='byte', mult_exp=3)")
        print(num_unit)
        print(dictify(num_unit, include_properties=True))
        num_unit = DisplayValue(float("inf"), unit='bit', unit_exp=6)
        print(    "DisplayValue(float('inf'), unit='bit', unit_exp=6)")
        print(num_unit)
        num_unit = DisplayValue(float("-inf"), unit='bit', unit_exp=6)
        print(    "DisplayValue(float('-inf'), unit='bit', unit_exp=6)")
        print(num_unit)
        print(dictify(num_unit, include_properties=True))
        num_unit = DisplayValue(value=float("NaN"), unit='sec')
        print(    "DisplayValue(value=float('NaN'), unit='sec')")
        print(num_unit)
        print(dictify(num_unit, include_properties=True))

def print_method(prefix: str = "------- ",
                 suffix: str = " -------",
                 start: str = "\n\n",
                 end: str = "\n"):
    method_name = stack()[1][3]
    _print_title(title=method_name, prefix=prefix, suffix=suffix, start=start, end=end)


def _print_title(title,
                 prefix: str = "------- ",
                 suffix: str = " -------",
                 start: str = "\n",
                 end: str = "\n"):
    """
    Prints a formatted title to the console.

    Args:
        title (str): The main title string to be printed.
        prefix (str, optional): A string to prepend to the title. Defaults to "------- ".
        suffix (str, optional): A string to append to the title. Defaults to " -------".
        start (str, optional): A string to print before the entire formatted title. Defaults to "\n".
        end (str, optional): A string to print after the entire formatted title. Defaults to "\n".
    """
    print(f"{start}{prefix}{title}{suffix}{end}", end="")
