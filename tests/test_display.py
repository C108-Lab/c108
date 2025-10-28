#
# C108 - Display Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import math
from dataclasses import FrozenInstanceError
from decimal import Decimal
from fractions import Fraction

# Third-party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.display import DisplayFlow, DisplayFormat, DisplayValue, DisplayMode, MultSymbol, DisplaySymbols, DisplayScale
from c108.display import trimmed_digits, trimmed_round


# Tests ----------------------------------------------------------------------------------------------------------------


# Factory Methods Tests ---------------------------------------------------------------

class TestDisplayValueFactoryPlain:
    """Tests for DisplayValue.plain() factory method."""

    @pytest.mark.parametrize('value, unit, expected_str', [
        pytest.param(42, "byte", "42 bytes", id='int_plural'),
        pytest.param(1, "byte", "1 byte", id='int_singular'),
        pytest.param(0, "byte", "0 bytes", id='zero_plural'),
        pytest.param(-5, "meter", "-5 meters", id='negative'),
        pytest.param(123_000, "byte", "123000 bytes", id='large_no_scale'),
        pytest.param(3.14159, "meter", "3.14159 meters", id='float_e+0'),
        pytest.param(123.456e+123, "s", "1.23456e+125 s", id='float_e+125'),
    ])
    def test_basic_plain_display(self, value, unit, expected_str):
        """Plain mode displays values as-is without scaling."""
        dv = DisplayValue.plain(value, unit=unit)
        assert str(dv) == expected_str
        assert dv.mode == DisplayMode.PLAIN
        assert dv.mult_exp == 0
        assert dv.unit_exp == 0

    def test_plain_with_precision(self):
        """Precision controls decimal places in plain mode."""
        dv = DisplayValue.plain(3.14159, unit="meter", precision=2)
        assert "3.14" in str(dv)
        assert dv.mode == DisplayMode.PLAIN

    def test_plain_with_trim_digits(self):
        """Trim digits reduces significant figures in plain mode."""
        dv = DisplayValue.plain(123.456789, unit="second", trim_digits=5)
        assert str(dv) == "123.46 seconds"

    def test_plain_precision_precedence(self):
        """Precision takes precedence over trim_digits."""
        dv = DisplayValue.plain(1 / 3, unit="meter", precision=2, trim_digits=10)
        assert str(dv) == "0.33 meters"

    @pytest.mark.parametrize('value, expected_contains', [
        pytest.param(None, "None", id='none'),
        pytest.param(float('inf'), "+∞ bytes", id='inf'),
        pytest.param(float('nan'), "NaN", id='nan'),
    ])
    def test_plain_non_finite(self, value, expected_contains):
        """Plain mode handles non-finite values."""
        dv = DisplayValue.plain(value, unit="byte")
        assert expected_contains in str(dv)


class TestDisplayValueFactoryBaseFixed:
    """Tests for DisplayValue.base_fixed() factory method."""

    @pytest.mark.parametrize('value, unit, expected_pattern', [
        pytest.param(1_500_000, "byte", r"1\.5×10⁶ bytes", id='auto_scale_mega'),
        pytest.param(123, "byte", "123 bytes", id='no_scale_small'),
        pytest.param(0.000123, "second", r"123×10⁻⁶ seconds", id='auto_scale_micro'),
        pytest.param(42, "meter", "42 meters", id='moderate_no_scale'),
    ])
    def test_base_fixed_auto_scaling(self, value, unit, expected_pattern):
        """BASE_FIXED auto-scales multiplier to keep value compact."""
        dv = DisplayValue.base_fixed(value, unit=unit)
        assert dv.mode == DisplayMode.BASE_FIXED
        assert dv.mult_exp is None  # Auto-calculated
        assert dv.unit_exp == 0  # Always base units
        # Use regex for flexible matching
        import re
        assert re.search(expected_pattern, str(dv))

    def test_base_fixed_with_precision(self):
        """Precision formats normalized value in BASE_FIXED mode."""
        dv = DisplayValue.base_fixed(123_456_789, unit="byte", precision=2)
        result = str(dv)
        assert "123.46" in result or "123,46" in result  # Locale-independent
        assert "×10" in result
        assert "bytes" in result

    def test_base_fixed_binary_scale(self):
        """BASE_FIXED works with binary scale."""
        scale = DisplayScale(type="binary")
        dv = DisplayValue.base_fixed(2 ** 30, unit="B", trim_digits=3)
        dv_bin = DisplayValue(
            2 ** 30, unit="B", mult_exp=None, unit_exp=0,
            scale=scale, trim_digits=3
        )
        assert "2³⁰" in str(dv_bin) or "2^30" in str(dv_bin)
        assert dv_bin.mode == DisplayMode.BASE_FIXED


class TestDisplayValueFactoryUnitFlex:
    """Tests for DisplayValue.unit_flex() factory method."""

    @pytest.mark.parametrize('value, unit, expected_unit_suffix', [
        pytest.param(1_500_000, "byte", "Mbytes", id='mega_bytes'),
        pytest.param(2_500, "byte", "kbytes", id='kilo_bytes'),
        pytest.param(0.000123, "second", "µs", id='micro_seconds'),
        pytest.param(42, "meter", "meters", id='base_no_prefix'),
    ])
    def test_unit_flex_auto_prefix(self, value, unit, expected_unit_suffix):
        """UNIT_FLEX auto-selects SI prefix for optimal display."""
        dv = DisplayValue.unit_flex(value, unit=unit)
        assert dv.mode == DisplayMode.UNIT_FLEX
        assert dv.mult_exp == 0  # Default
        assert dv.unit_exp is None  # Auto-selected
        assert expected_unit_suffix in str(dv)

    def test_unit_flex_no_unit(self):
        """UNIT_FLEX works without unit (prefix only)."""
        dv = DisplayValue.unit_flex(1_500_000)
        assert str(dv) == "1.5M"  # Just prefix, no unit
        assert dv.mode == DisplayMode.UNIT_FLEX

    def test_unit_flex_with_mult_exp(self):
        """UNIT_FLEX allows explicit mult_exp."""
        dv = DisplayValue.unit_flex(123_000_000, unit="byte", mult_exp=3)
        assert "×10³" in str(dv) or "×10^3" in str(dv)
        assert dv.mode == DisplayMode.UNIT_FLEX

    def test_unit_flex_custom_prefixes(self):
        """UNIT_FLEX respects custom unit_prefixes."""
        custom_prefixes = {0: '', 9: 'G'}  # Only base and giga
        dv = DisplayValue.unit_flex(
            500_000_000, unit="byte",
            unit_prefixes=custom_prefixes
        )
        # Should select 'G' even though value is between k and M
        assert "Gbytes" in str(dv)


class TestDisplayValueFactoryUnitFixed:
    """Tests for DisplayValue.unit_fixed() factory method."""

    def test_unit_fixed_from_base_value(self):
        """Create from base units, fixed SI prefix."""
        dv = DisplayValue.unit_fixed(value=123_000_000, si_unit="Mbyte")
        assert dv.mode == DisplayMode.UNIT_FIXED
        assert "123" in str(dv)
        assert "Mbyte" in str(dv)

    def test_unit_fixed_from_si_value(self):
        """Create from SI-prefixed value."""
        dv = DisplayValue.unit_fixed(si_value=123, si_unit="Mbyte")
        assert dv.mode == DisplayMode.UNIT_FIXED
        assert "123" in str(dv)
        assert "Mbyte" in str(dv)
        # Internally converts to base: 123 * 10^6
        assert dv.value == 123_000_000

    def test_unit_fixed_mutual_exclusion(self):
        """Cannot specify both value and si_value."""
        with pytest.raises(ValueError, match="cannot specify both"):
            DisplayValue.unit_fixed(
                value=100, si_value=200, si_unit="Mbyte"
            )

    def test_unit_fixed_requires_one(self):
        """Must specify either value or si_value."""
        with pytest.raises(ValueError, match="must specify either"):
            DisplayValue.unit_fixed(si_unit="Mbyte")

    def test_unit_fixed_with_multiplier(self):
        """UNIT_FIXED adds multiplier when needed."""
        dv = DisplayValue.unit_fixed(
            value=123_000_000_000, si_unit="Mbyte"
        )
        assert "×10" in str(dv) or "×10^3" in str(dv)
        assert "Mbyte" in str(dv)


class TestDisplayValueFactoryFixed:
    """Tests for DisplayValue.fixed() factory method."""

    def test_fixed_basic(self):
        """FIXED mode with explicit exponents."""
        dv = DisplayValue.fixed(value=123)
        assert dv.mode == DisplayMode.FIXED
        # TODO: Add assertions when fixed() implementation is clarified


# Value Type Conversion Tests ---------------------------------------------------------------

class TestDisplayValueTypeConversion:
    """Tests for std_numeric() type conversion integration."""

    @pytest.mark.parametrize('value, expected_type', [
        pytest.param(42, int, id='int'),
        pytest.param(3.14, float, id='float'),
        pytest.param(Decimal("3.14"), float, id='decimal'),
        pytest.param(Fraction(22, 7), float, id='fraction'),
        pytest.param(None, type(None), id='none'),
    ])
    def test_stdlib_types(self, value, expected_type):
        """Accept and convert stdlib numeric types."""
        dv = DisplayValue(value, unit="meter")
        assert isinstance(dv.value, expected_type)

    def test_bool_rejection(self):
        """Reject boolean values explicitly."""
        with pytest.raises(TypeError, match="(?i)bool"):
            DisplayValue(True, unit="meter")
        with pytest.raises(TypeError, match="(?i)bool"):
            DisplayValue(False, unit="meter")

    @pytest.mark.parametrize('value', [
        pytest.param(float('inf'), id='inf'),
        pytest.param(float('-inf'), id='neg_inf'),
        pytest.param(float('nan'), id='nan'),
    ])
    def test_non_finite_values(self, value):
        """Accept non-finite float values."""
        dv = DisplayValue(value, unit="byte")
        assert not dv.is_finite
        assert dv.value == value or (math.isnan(dv.value) and math.isnan(value))

    # NumPy type tests (conditional on numpy availability)
    @pytest.mark.skipif(
        not hasattr(pytest, 'importorskip'),
        reason="Requires pytest.importorskip"
    )
    def test_numpy_types(self):
        """Convert NumPy types to stdlib equivalents."""
        np = pytest.importorskip('numpy')

        test_cases = [
            (np.int32(42), int),
            (np.int64(42), int),
            (np.float32(3.14), float),
            (np.float64(3.14), float),
            (np.array([42]).item(), int),
        ]

        for np_value, expected_type in test_cases:
            dv = DisplayValue(np_value, unit="meter")
            assert isinstance(dv.value, expected_type)

    # Pandas type tests (conditional)
    @pytest.mark.skipif(
        not hasattr(pytest, 'importorskip'),
        reason="Requires pytest.importorskip"
    )
    def test_pandas_types(self):
        """Convert Pandas types to stdlib equivalents."""
        pd = pytest.importorskip('pandas')

        # pd.NA converts to float('nan')
        dv = DisplayValue(pd.NA, unit="byte")
        assert math.isnan(dv.value)

        # Series.item() extracts scalar
        series = pd.Series([42])
        dv = DisplayValue(series.item(), unit="byte")
        assert dv.value == 42


# Display Modes & Formatting ---------------------------------------------------------------

class TestDisplayValueModeInference:
    """Tests for display mode inference from mult_exp/unit_exp."""

    @pytest.mark.parametrize('mult_exp, unit_exp, expected_mode', [
        pytest.param(0, 0, DisplayMode.PLAIN, id='plain'),
        pytest.param(3, 6, DisplayMode.FIXED, id='fixed_both'),
        pytest.param(0, 3, DisplayMode.FIXED, id='fixed_unit_only'),
        pytest.param(3, 0, DisplayMode.FIXED, id='fixed_mult_only'),
        pytest.param(None, 0, DisplayMode.BASE_FIXED, id='base_fixed'),
        pytest.param(None, 3, DisplayMode.UNIT_FIXED, id='unit_fixed'),
        pytest.param(0, None, DisplayMode.UNIT_FLEX, id='unit_flex_0'),
        pytest.param(3, None, DisplayMode.UNIT_FLEX, id='unit_flex_3'),
        pytest.param(None, None, DisplayMode.BASE_FIXED, id='both_none'),
    ])
    def test_mode_inference(self, mult_exp, unit_exp, expected_mode):
        """Correctly infer mode from exponent combination."""
        dv = DisplayValue(
            123, unit="byte",
            mult_exp=mult_exp, unit_exp=unit_exp
        )
        assert dv.mode == expected_mode


class TestDisplayValueStringFormatting:
    """Tests for __str__ output across modes and scales."""

    # PLAIN mode
    @pytest.mark.parametrize('value, unit, expected', [
        pytest.param(123, "B", "123 B", id='plain_int'),
        pytest.param(1, "byte", "1 byte", id='plain_singular'),
        pytest.param(2, "byte", "2 bytes", id='plain_plural'),
    ])
    def test_plain_mode_formatting(self, value, unit, expected):
        """PLAIN mode string formatting."""
        dv = DisplayValue.plain(value, unit=unit)
        assert str(dv) == expected

    # FIXED mode (decimal)
    @pytest.mark.parametrize('value, mult_exp, unit_exp, expected', [
        pytest.param(123, 0, 3, "0.123 kB", id='fixed_0_3'),
        pytest.param(123, 3, 0, "0.123×10³ B", id='fixed_3_0'),
        pytest.param(123456, 3, 6, "0.123456×10³ MB", id='fixed_both'),
    ])
    def test_fixed_mode_decimal(self, value, mult_exp, unit_exp, expected):
        """FIXED mode with decimal scale."""
        dv = DisplayValue(
            value, unit="B",
            mult_exp=mult_exp, unit_exp=unit_exp,
            format=DisplayFormat.unicode()
        )
        # Normalize expected string (remove superscripts for comparison)
        result = str(dv).replace('³', '3')
        expected_norm = expected.replace('³', '3')
        assert result == expected_norm or str(dv) == expected

    # BASE_FIXED mode
    def test_base_fixed_formatting(self):
        """BASE_FIXED mode string formatting."""
        dv = DisplayValue.base_fixed(123_000, unit="byte")
        result = str(dv)
        assert "123" in result
        assert "10" in result  # Multiplier present
        assert "bytes" in result
        assert dv.mode == DisplayMode.BASE_FIXED

    # UNIT_FLEX mode
    def test_unit_flex_formatting(self):
        """UNIT_FLEX mode string formatting."""
        dv = DisplayValue.unit_flex(1_500_000, unit="byte")
        assert str(dv) == "1.5 Mbytes"
        assert dv.mode == DisplayMode.UNIT_FLEX

    # Binary scale
    @pytest.mark.parametrize('value, mult_exp, unit_exp, expected_contains', [
        pytest.param(123, 0, 0, "123 B", id='binary_plain'),
        pytest.param(123, 0, 10, "KiB", id='binary_Ki'),
        pytest.param(2 ** 30 * 0.123, 30, 0, ["2³⁰", "B"], id='binary_mult'),
    ])
    def test_binary_scale_formatting(
            self, value, mult_exp, unit_exp, expected_contains
    ):
        """Binary scale string formatting."""
        scale = DisplayScale(type="binary")
        dv = DisplayValue(
            value, unit="B",
            mult_exp=mult_exp, unit_exp=unit_exp,
            scale=scale,
            format=DisplayFormat.unicode()
        )
        result = str(dv)
        if isinstance(expected_contains, list):
            for substr in expected_contains:
                assert substr in result or substr.replace('³⁰', '30') in result
        else:
            assert expected_contains in result


# Properties & Computed Values ---------------------------------------------------------------

class TestDisplayValueProperties:
    """Tests for computed properties."""

    @pytest.mark.parametrize('value, expected', [
        pytest.param(None, False, id='none'),
        pytest.param(math.inf, False, id='inf'),
        pytest.param(-math.inf, False, id='neg_inf'),
        pytest.param(math.nan, False, id='nan'),
        pytest.param(0, True, id='zero'),
        pytest.param(42, True, id='int'),
        pytest.param(3.14, True, id='float'),
    ])
    def test_is_finite(self, value, expected):
        """is_finite property correctness."""
        dv = DisplayValue(value, unit="byte")
        assert dv.is_finite == expected

    @pytest.mark.parametrize('scale_type, mult_exp, expected', [
        pytest.param('decimal', 3, 1000, id='dec_10_3'),
        pytest.param('decimal', 0, 1, id='dec_10_0'),
        pytest.param('decimal', 6, 1_000_000, id='dec_10_6'),
        pytest.param('binary', 10, 1024, id='bin_2_10'),
        pytest.param('binary', 20, 2 ** 20, id='bin_2_20'),
        pytest.param('binary', 0, 1, id='bin_2_0'),
    ])
    def test_mult_value(self, scale_type, mult_exp, expected):
        """mult_value computes correct multiplier."""
        scale = DisplayScale(type=scale_type)
        dv = DisplayValue(
            123, unit="B",
            mult_exp=mult_exp, unit_exp=0,
            scale=scale
        )
        assert dv.mult_value == expected

    @pytest.mark.parametrize('scale_type, unit_exp, expected', [
        pytest.param('decimal', 6, 1_000_000, id='dec_M'),
        pytest.param('decimal', 3, 1_000, id='dec_k'),
        pytest.param('decimal', 0, 1, id='dec_base'),
        pytest.param('binary', 20, 2 ** 20, id='bin_Mi'),
        pytest.param('binary', 10, 1024, id='bin_Ki'),
        pytest.param('binary', 0, 1, id='bin_base'),
    ])
    def test_unit_value(self, scale_type, unit_exp, expected):
        """unit_value computes correct unit prefix value."""
        scale = DisplayScale(type=scale_type)
        dv = DisplayValue(
            123, unit="B",
            mult_exp=0, unit_exp=unit_exp,
            scale=scale
        )
        assert dv.unit_value == expected

    def test_ref_value(self):
        """ref_value = mult_value * unit_value."""
        dv = DisplayValue(
            123_456_789, unit="B",
            mult_exp=3, unit_exp=6
        )
        assert dv.ref_value == 1000 * 1_000_000
        assert dv.ref_value == dv.mult_value * dv.unit_value

    def test_normalized_calculation(self):
        """normalized = value / ref_value."""
        dv = DisplayValue(
            123_000_000, unit="byte",
            mult_exp=3, unit_exp=6
        )
        expected = 123_000_000 / (1000 * 1_000_000)
        assert abs(dv.normalized - expected) < 0.001

    def test_unit_prefix_extraction(self):
        """unit_prefix extracts prefix from mapping."""
        dv = DisplayValue.unit_flex(1_500_000, unit="byte")
        assert dv.unit_prefix == "M"

    @pytest.mark.parametrize('value, pluralize, expected', [
        pytest.param(1, True, "byte", id='singular'),
        pytest.param(0, True, "bytes", id='zero_plural'),
        pytest.param(2, True, "bytes", id='two_plural'),
        pytest.param(1.0, True, "byte", id='one_float_singular'),
        pytest.param(1.5, True, "bytes", id='float_plural'),
        pytest.param(5, False, "byte", id='no_pluralize'),
    ])
    def test_units_pluralization(self, value, pluralize, expected):
        """units property handles pluralization."""
        dv = DisplayValue(
            value, unit="byte",
            mult_exp=0, unit_exp=0,
            pluralize=pluralize
        )
        assert dv.units == expected

    def test_units_with_prefix(self):
        """units includes SI prefix."""
        dv = DisplayValue.unit_flex(1_500_000, unit="byte")
        assert dv.units == "Mbytes"

    def test_units_prefix_only(self):
        """units shows prefix when unit is None."""
        dv = DisplayValue.unit_flex(1_500_000)
        assert dv.units == "M"

    def test_number_property(self):
        """number property includes multiplier."""
        dv = DisplayValue.base_fixed(123_000, unit="byte")
        number = dv.number
        assert "123" in number
        assert "10" in number  # Multiplier

    def test_parts_tuple(self):
        """parts returns (number, units) tuple."""
        dv = DisplayValue.unit_flex(1_500, unit="byte")
        number, units = dv.parts
        assert "1.5" in number
        assert units == "kbytes"


# Overflow/Underflow Behavior ---------------------------------------------------------------

class TestDisplayValueOverflowUnderflow:
    """Tests for overflow/underflow formatting behavior."""

    def test_unit_flex_overflow_formatting(self):
        """UNIT_FLEX shows inf for overflow."""
        dv = DisplayValue.unit_flex(1e100, unit="B")
        assert "+∞" in str(dv) or "inf" in str(dv)
        assert dv.flow.overflow

    def test_unit_flex_underflow_formatting(self):
        """UNIT_FLEX shows ≈0 for underflow."""
        dv = DisplayValue.unit_flex(1e-100, unit="B")
        result = str(dv)
        assert "≈0" in result or "+0" in result or "0" in result
        assert dv.flow.underflow

    def test_fixed_mode_overflow(self):
        """FIXED mode overflow behavior."""
        dv = DisplayValue(
            1e100, unit="B",
            mult_exp=3, unit_exp=6,
            flow=DisplayFlow(overflow_tolerance=5)
        )
        assert dv.flow.overflow
        result = str(dv)
        assert "∞" in result or "inf" in result

    def test_plain_no_overflow(self):
        """PLAIN mode never overflows."""
        dv = DisplayValue.plain(1e100, unit="B")
        assert not dv.flow.overflow
        assert not dv.flow.underflow

    def test_base_fixed_no_overflow(self):
        """BASE_FIXED scales multiplier, no overflow."""
        dv = DisplayValue.base_fixed(1e100, unit="B")
        assert not dv.flow.overflow
        result = str(dv)
        assert "10" in result  # Multiplier auto-scaled

    def test_custom_overflow_predicate(self):
        """Custom overflow predicate."""

        def custom_overflow(dv):
            return dv.value >= 1000

        flow = DisplayFlow(overflow_predicate=custom_overflow)
        dv = DisplayValue(
            2500, unit="meter",
            mult_exp=0, unit_exp=None,
            flow=flow
        )
        assert dv.flow.overflow

    def test_e_notation_mode(self):
        """Overflow with e_notation mode."""
        flow = DisplayFlow(mode='e_notation', overflow_tolerance=3)
        dv = DisplayValue(
            1e10, unit="B",
            mult_exp=3, unit_exp=6,
            flow=flow
        )
        result = str(dv)
        assert "e" in result.lower() or "E" in result


# Composition Tests ---------------------------------------------------------------

class TestDisplayValueComposition:
    """Composition tests with DisplayFlow, DisplayFormat, DisplayScale, DisplaySymbols."""

    def test_custom_display_format(self):
        """Custom DisplayFormat integration."""
        fmt = DisplayFormat(mult='latex', symbols='ascii')
        dv = DisplayValue.base_fixed(123_000, unit="byte", format=fmt)
        result = str(dv)
        assert "10^{" in result  # LaTeX format

    def test_ascii_symbols(self):
        """ASCII symbols integration."""
        symbols = DisplaySymbols.ascii()
        dv = DisplayValue(float('inf'), unit="byte", symbols=symbols)
        assert "inf" in str(dv)
        assert "∞" not in str(dv)

    def test_unicode_symbols(self):
        """Unicode symbols integration."""
        symbols = DisplaySymbols.unicode()
        dv = DisplayValue(float('inf'), unit="byte", symbols=symbols)
        assert "∞" in str(dv)

    def test_binary_scale_integration(self):
        """Binary scale full integration."""
        scale = DisplayScale(type="binary")
        fmt = DisplayFormat.unicode()
        dv = DisplayValue(
            2 ** 30, unit="B",
            mult_exp=20, unit_exp=10,
            scale=scale, format=fmt
        )
        result = str(dv)
        assert "KiB" in result
        assert "2²⁰" in result or "2^20" in result

    def test_custom_unit_plurals(self):
        """Custom pluralization mapping."""
        custom_plurals = {"datum": "data", "index": "indices"}
        dv = DisplayValue(
            5, unit="datum",
            mult_exp=0, unit_exp=0,
            unit_plurals=custom_plurals
        )
        assert "data" in str(dv)

    def test_custom_unit_prefixes(self):
        """Custom unit prefix mapping."""
        custom_prefixes = {0: '', 9: 'G'}  # Only base and giga
        dv = DisplayValue.unit_flex(
            500_000, unit="byte",
            unit_prefixes=custom_prefixes
        )
        # Should select closest available prefix
        assert "Gbytes" in str(dv)

    def test_flow_merge_with_owner(self):
        """DisplayFlow.merge() establishes owner backlink."""
        flow = DisplayFlow(overflow_tolerance=3)
        dv = DisplayValue(1e10, unit="B", mult_exp=3, unit_exp=6)
        merged_flow = flow.merge(owner=dv)

        # Flow should now evaluate predicates with dv as context
        assert merged_flow.overflow  # Should trigger based on dv's state


# Edge Cases & Validation ---------------------------------------------------------------

class TestDisplayValueEdgeCases:
    """Edge case and boundary condition tests."""

    @pytest.mark.parametrize('unit', [
        pytest.param("", id='empty_string'),
        pytest.param("  ", id='whitespace'),
        pytest.param("a" * 1000, id='very_long'),
    ])
    def test_unusual_unit_strings(self, unit):
        """Handle unusual but valid unit strings."""
        dv = DisplayValue(42, unit=unit, mult_exp=0, unit_exp=0)
        result = str(dv)
        assert "42" in result

    def test_unicode_unit_name(self):
        """Unicode characters in unit names."""
        dv = DisplayValue(42, unit="метр", mult_exp=0, unit_exp=0)
        assert "метр" in str(dv)

    def test_fractional_unit_with_slash(self):
        """Units with slashes (rates)."""
        dv = DisplayValue(100, unit="byte/s", mult_exp=0, unit_exp=0)
        assert "byte/s" in str(dv)

    def test_negative_zero(self):
        """Negative zero handling."""
        dv = DisplayValue(-0.0, unit="meter")
        result = str(dv)
        assert "0" in result

    def test_very_small_positive(self):
        """Very small positive values."""
        dv = DisplayValue.unit_flex(1e-50, unit="second")
        # Should not crash
        result = str(dv)
        assert "second" in result or "s" in result

    def test_very_large_negative(self):
        """Very large negative values."""
        dv = DisplayValue.unit_flex(-1e50, unit="meter")
        result = str(dv)
        assert "-" in result

    def test_zero_with_units(self):
        """Zero value with various units."""
        dv = DisplayValue(0, unit="byte")
        assert str(dv) == "0 bytes"

    def test_none_value_with_unit(self):
        """None value formatting."""
        dv = DisplayValue(None, unit="item")
        result = str(dv)
        assert "None" in result or "N/A" in result

    def test_precision_zero(self):
        """Precision=0 shows no decimals."""
        dv = DisplayValue.plain(3.7, unit="meter", precision=0)
        result = str(dv)
        assert "4" in result  # Rounded
        assert "." not in result

    def test_trim_digits_one(self):
        """trim_digits=1 minimal significant figures."""
        dv = DisplayValue.plain(123.456, unit="meter", trim_digits=1)
        result = str(dv)
        assert "100" in result or "1e2" in result.lower()

    def test_whole_as_int_true(self):
        """whole_as_int converts 3.0 to "3"."""
        dv = DisplayValue(3.0, unit="meter", mult_exp=0, unit_exp=0, whole_as_int=True)
        result = str(dv)
        # Should show "3" not "3.0"
        assert result == "3 meters" or "3.0" not in result

    def test_whole_as_int_false(self):
        """whole_as_int=False keeps 3.0 as "3.0"."""
        dv = DisplayValue(3.0, unit="meter", mult_exp=0, unit_exp=0, whole_as_int=False)
        result = str(dv)
        assert "3.0" in result


class TestDisplayValueValidation:
    """Input validation and error handling tests."""

    def test_invalid_value_type(self):
        """Reject invalid value types."""
        with pytest.raises(TypeError):
            DisplayValue("not_a_number", unit="meter")

    def test_invalid_unit_type(self):
        """Reject non-string units."""
        with pytest.raises(TypeError):
            DisplayValue(42, unit=123)

    def test_invalid_mult_exp_type(self):
        """Reject non-int mult_exp."""
        with pytest.raises(TypeError):
            DisplayValue(42, unit="byte", mult_exp="3")

    def test_invalid_unit_exp_type(self):
        """Reject non-int unit_exp."""
        with pytest.raises(TypeError):
            DisplayValue(42, unit="byte", unit_exp="3")

    def test_invalid_unit_exp_value(self):
        """Reject unit_exp not in prefix mapping."""
        with pytest.raises(ValueError, match="unit_exp"):
            DisplayValue(42, unit="byte", mult_exp=0, unit_exp=5)

    def test_negative_precision(self):
        """Reject negative precision."""
        with pytest.raises(ValueError):
            DisplayValue.plain(3.14, unit="meter", precision=-1)

    def test_negative_trim_digits(self):
        """Reject negative trim_digits."""
        with pytest.raises(ValueError):
            DisplayValue.plain(3.14, unit="meter", trim_digits=-1)

    def test_invalid_scale_type(self):
        """Reject invalid scale type."""
        with pytest.raises(ValueError):
            scale = DisplayScale(type="hexadecimal")
            DisplayValue(42, unit="byte", scale=scale)

    def test_frozen_immutability(self):
        """DisplayValue is immutable (frozen dataclass)."""
        dv = DisplayValue(42, unit="byte")
        with pytest.raises(FrozenInstanceError):
            dv.value = 100
        with pytest.raises(FrozenInstanceError):
            dv.unit = "meter"

    def test_unit_prefixes_wrong_type(self):
        """Reject non-mapping unit_prefixes."""
        with pytest.raises(TypeError):
            DisplayValue(42, unit="byte", unit_prefixes=[0, 3, 6])

    def test_unit_plurals_wrong_type(self):
        """Reject non-mapping unit_plurals."""
        with pytest.raises(TypeError):
            DisplayValue(42, unit="byte", unit_plurals=["byte", "bytes"])


# Formatting Pipeline Tests ---------------------------------------------------------------

class TestDisplayValueFormattingPipeline:
    """Tests for the formatting pipeline order and interactions."""

    def test_pipeline_non_finite_first(self):
        """Non-finite values bypass all formatting."""
        dv = DisplayValue(
            float('inf'), unit="byte",
            precision=2, trim_digits=5, whole_as_int=True
        )
        result = str(dv)
        assert "∞" in result or "inf" in result

    def test_pipeline_precision_over_trim(self):
        """Precision takes precedence over trim_digits."""
        dv = DisplayValue.plain(
            1 / 3, unit="meter",
            precision=2, trim_digits=10
        )
        result = str(dv)
        assert "0.33" in result
        assert len(result.split('.')[1].split()[0]) == 2  # Exactly 2 decimals

    def test_pipeline_trim_applied(self):
        """trim_digits reduces significant figures."""
        dv = DisplayValue.plain(
            123.456789, unit="second",
            trim_digits=4
        )
        result = str(dv)
        # Should have ~4 significant digits
        assert "123.5" in result or "123.4" in result

    def test_pipeline_whole_as_int_after_rounding(self):
        """whole_as_int applied after rounding."""
        dv = DisplayValue.plain(
            2.999, unit="meter",
            precision=0, whole_as_int=True
        )
        result = str(dv)
        # Rounds to 3, then converts to int display
        assert result == "3 meters"

    def test_pipeline_overflow_formatting_last(self):
        """Overflow formatting applied at end."""
        dv = DisplayValue(
            1e100, unit="B",
            mult_exp=3, unit_exp=6,
            precision=2,  # Should be ignored due to overflow
            flow=DisplayFlow(overflow_tolerance=5)
        )
        result = str(dv)
        assert "∞" in result or "inf" in result
        assert ".00" not in result  # Precision not applied to inf


# Normalized Value Tests ---------------------------------------------------------

class TestDisplayValueNormalized:
    """Tests for normalized property across modes."""

    def test_normalized_plain_mode(self):
        """PLAIN mode normalized equals value."""
        dv = DisplayValue.plain(123, unit="byte")
        assert dv.normalized == 123

    def test_normalized_base_fixed_mode(self):
        """BASE_FIXED mode normalized with auto multiplier."""
        dv = DisplayValue.base_fixed(123_000, unit="byte")
        # Should normalize to ~123 with auto multiplier
        assert 100 <= dv.normalized <= 999

    def test_normalized_unit_flex_mode(self):
        """UNIT_FLEX mode normalized with auto prefix."""
        dv = DisplayValue.unit_flex(1_500_000, unit="byte")
        # Should normalize to 1.5 with M prefix
        assert abs(dv.normalized - 1.5) < 0.01

    def test_normalized_fixed_mode(self):
        """FIXED mode normalized with both exponents."""
        dv = DisplayValue(
            123_000_000, unit="byte",
            mult_exp=3, unit_exp=6
        )
        # normalized = 123_000_000 / (10^3 * 10^6)
        expected = 123_000_000 / (1000 * 1_000_000)
        assert abs(dv.normalized - expected) < 0.001

    def test_normalized_with_trim_digits(self):
        """Normalized includes trimming."""
        dv = DisplayValue.plain(
            123.456789, unit="meter",
            trim_digits=4
        )
        # Normalized should be rounded to 4 sig figs
        assert abs(dv.normalized - 123.5) < 0.1

    def test_normalized_extreme_overflow(self):
        """Normalized for extreme overflow values."""
        dv = DisplayValue.unit_flex(1e100, unit="B")
        # Should still compute normalized (even if display shows inf)
        assert dv.normalized > 1e70  # Large but scaled

    def test_normalized_extreme_underflow(self):
        """Normalized for extreme underflow values."""
        dv = DisplayValue.unit_flex(1e-100, unit="B")
        # Should still compute normalized
        assert 0 < dv.normalized < 1


# Decimal Vs Binary Scale Comparison ---------------------------------------------------------------

class TestDisplayValueScaleComparison:
    """Compare decimal vs binary scale behavior side-by-side."""

    def test_scale_1024_decimal_vs_binary(self):
        """1024 bytes: decimal shows 1.024k, binary shows 1Ki."""
        dec = DisplayValue.unit_flex(1024, unit="byte")
        bin_scale = DisplayScale(type="binary")
        bin = DisplayValue(1024, unit="B", mult_exp=0, unit_exp=None, scale=bin_scale)

        assert "1.024" in str(dec) or "1.02" in str(dec)
        assert "kbytes" in str(dec)

        assert "1 KiB" in str(bin) or "1KiB" in str(bin)

    def test_scale_exponent_calculation(self):
        """Scale affects exponent calculation."""
        dec_scale = DisplayScale(type="decimal")
        bin_scale = DisplayScale(type="binary")

        # 1024 has different exponents in different scales
        assert dec_scale.value_exponent(1024) == 3  # log10(1024) = 3.01
        assert bin_scale.value_exponent(1024) == 10  # log2(1024) = 10

    def test_scale_mult_exp_auto(self):
        """Scale affects auto mult_exp selection."""
        value = 2 ** 20  # 1 MiB

        # Decimal: should get mult_exp around 6
        dec = DisplayValue.base_fixed(value, unit="byte")
        # Binary: should get mult_exp=20 or nearby
        bin = DisplayValue.base_fixed(
            value, unit="B",
            format=DisplayFormat.unicode(),
            scale=DisplayScale(type="binary")
        )

        result_dec = str(dec)
        result_bin = str(bin)

        assert "10⁶" in result_dec or "10^6" in result_dec
        assert "2²⁰" in result_bin or "2^20" in result_bin


# Factory Method Edge Cases ---------------------------------------------------------------

class TestDisplayValueFactoryEdgeCases:
    """Edge cases specific to factory methods."""

    def test_unit_fixed_fractional_unit(self):
        """unit_fixed with rate units (e.g., MB/s)."""
        dv = DisplayValue.unit_fixed(
            value=500_000_000, si_unit="Mbyte/s"
        )
        result = str(dv)
        assert "500" in result
        assert "Mbyte/s" in result

    def test_unit_fixed_parse_prefix_from_unit(self):
        """unit_fixed correctly extracts prefix from si_unit."""
        test_cases = [
            ("kbyte", 3),
            ("Mbyte", 6),
            ("Gbyte", 9),
            ("ms", -3),
            ("µs", -6),
            ("ns", -9),
        ]

        for si_unit, expected_exp in test_cases:
            dv = DisplayValue.unit_fixed(si_value=100, si_unit=si_unit)
            assert dv.unit_exp == expected_exp

    def test_base_fixed_none_value(self):
        """base_fixed with None value."""
        dv = DisplayValue.base_fixed(None, unit="byte")
        result = str(dv)
        assert "None" in result or "N/A" in result

    def test_unit_flex_zero(self):
        """unit_flex with zero value."""
        dv = DisplayValue.unit_flex(0, unit="byte")
        assert str(dv) == "0 bytes"

    def test_plain_with_non_finite(self):
        """plain factory with non-finite values."""
        test_cases = [
            (float('inf'), "inf"),
            (float('-inf'), "-inf"),
            (float('nan'), "NaN"),
        ]

        for value, expected in test_cases:
            dv = DisplayValue.plain(value, unit="meter")
            result = str(dv)
            assert expected in result or expected.replace('inf', '∞') in result


# Regression Tests (Based On Existing Test Patterns) ------------------------------------------

class TestDisplayValueRegression:
    """Regression tests to ensure existing behavior is preserved."""

    def test_mode_inference_all_combinations(self):
        """All documented mult_exp/unit_exp combinations infer correct mode."""
        test_matrix = [
            # (mult_exp, unit_exp, expected_mode)
            (0, 0, DisplayMode.PLAIN),
            (0, 3, DisplayMode.FIXED),
            (3, 0, DisplayMode.FIXED),
            (3, 6, DisplayMode.FIXED),
            (None, 0, DisplayMode.BASE_FIXED),
            (None, 3, DisplayMode.UNIT_FIXED),
            (None, None, DisplayMode.BASE_FIXED),
            (0, None, DisplayMode.UNIT_FLEX),
            (3, None, DisplayMode.UNIT_FLEX),
        ]

        for mult_exp, unit_exp, expected in test_matrix:
            dv = DisplayValue(
                123, unit="B",
                mult_exp=mult_exp, unit_exp=unit_exp
            )
            assert dv.mode == expected, \
                f"Failed for mult_exp={mult_exp}, unit_exp={unit_exp}"

    def test_overflow_predicates_all_modes(self):
        """Overflow predicates work correctly for all modes."""
        extreme_value = 1e100

        # UNIT_FLEX: should overflow
        dv_flex = DisplayValue.unit_flex(extreme_value, unit="B")
        assert dv_flex.flow.overflow

        # FIXED: should overflow
        dv_fixed = DisplayValue(
            extreme_value, unit="B",
            mult_exp=3, unit_exp=6
        )
        assert dv_fixed.flow.overflow

        # BASE_FIXED: should NOT overflow (auto-scales)
        dv_base = DisplayValue.base_fixed(extreme_value, unit="B")
        assert not dv_base.flow.overflow

        # PLAIN: should NOT overflow
        dv_plain = DisplayValue.plain(extreme_value, unit="B")
        assert not dv_plain.flow.overflow

    def test_pluralization_edge_cases_preserved(self):
        """All pluralization edge cases work as documented."""
        test_cases = [
            (1, True, "byte"),
            (0, True, "bytes"),
            (2, True, "bytes"),
            (1.0, True, "byte"),
            (1.5, True, "bytes"),
            (5, False, "byte"),
        ]

        for value, pluralize, expected in test_cases:
            dv = DisplayValue(
                value, unit="byte",
                mult_exp=0, unit_exp=0,
                pluralize=pluralize
            )
            assert dv.units == expected


# Performance & Stress Tests (OPTIONAL) ---------------------------------------------------------------

class TestDisplayValueStress:
    """Stress tests for extreme scenarios."""

    def test_very_long_unit_name(self):
        """Handle extremely long unit names."""
        long_unit = "x" * 10_000
        dv = DisplayValue(42, unit=long_unit, mult_exp=0, unit_exp=0)
        result = str(dv)
        assert len(result) > 10_000

    def test_many_decimal_places(self):
        """Handle values with many decimal places."""
        value = 1 / 7  # Repeating decimal
        dv = DisplayValue.plain(value, unit="meter", precision=50)
        result = str(dv)
        assert "meter" in result

    def test_extreme_precision(self):
        """Very high precision values."""
        dv = DisplayValue.plain(math.pi, unit="meter", precision=100)
        result = str(dv)
        assert "3.14159" in result


# Demo/Documentation Tests ---------------------------------------------------------------

class TestDisplayValueDocExamples:
    """Tests matching documentation examples exactly."""

    def test_readme_example_basic(self):
        """Basic usage example from docs."""
        dv = DisplayValue(42, unit="byte")
        assert str(dv) == "42 bytes"

    def test_readme_example_factory_methods(self):
        """Factory method examples from docs."""
        assert "1.5 Mbytes" in str(DisplayValue.unit_flex(1_500_000, unit="byte"))
        assert "1.5×10⁶" in str(DisplayValue.base_fixed(1_500_000, unit="byte")) or \
               "1.5×10^6" in str(DisplayValue.base_fixed(1_500_000, unit="byte"))
        assert "1500000 bytes" == str(DisplayValue.plain(1_500_000, unit="byte"))

    def test_readme_example_precision(self):
        """Precision examples from docs."""
        assert "0.33 s" == str(DisplayValue(1 / 3, unit="s", mult_exp=0, unit_exp=0, precision=2))
        assert "1.3 s" in str(DisplayValue(4 / 3, unit="s", mult_exp=0, unit_exp=0, trim_digits=2))


# TODO / Future Enhancement Markers ---------------------------------------------------------------

class TestDisplayValueTODO:
    """Placeholder tests for future enhancements."""

    @pytest.mark.skip(reason="merge() method not yet implemented")
    def test_merge_method(self):
        """Test DisplayValue.merge() when implemented."""
        dv1 = DisplayValue(42, unit="byte")
        dv2 = dv1.merge(precision=2)
        assert dv2.precision == 2
        assert dv1.precision is None  # Original unchanged

    @pytest.mark.skip(reason="Awaiting clarification on fixed() factory")
    def test_fixed_factory_full(self):
        """Complete tests for DisplayValue.fixed() factory."""
        pass
