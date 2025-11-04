#
# C108 - Display Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import math
from dataclasses import FrozenInstanceError
from decimal import Decimal
from fractions import Fraction
from typing import Literal

# Third-party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.display import (
    DisplayFlow,
    DisplayFormat,
    DisplayValue,
    DisplayMode,
    MultSymbol,
    DisplaySymbols,
    DisplayScale,
)
from c108.display import trimmed_digits, trimmed_round


# Tests ----------------------------------------------------------------------------------------------------------------


def _pred_true(_dv) -> bool:
    return True


def _pred_false(_dv) -> bool:
    return False


class TestDisplayFlow:
    def test_invalid_predicate_types(self) -> None:
        """Validate predicate type errors."""
        invalid_obj = object()

        with pytest.raises(
            ValueError, match=r"(?i).*overflow_predicate must be callable.*"
        ):
            DisplayFlow(
                mode="e_notation",
                overflow_predicate=invalid_obj,  # not callable
                underflow_predicate=_pred_false,
                overflow_tolerance=3,
                underflow_tolerance=2,
            )

        with pytest.raises(
            ValueError, match=r"(?i).*underflow_predicate must be callable.*"
        ):
            DisplayFlow(
                mode="infinity",
                overflow_predicate=_pred_true,
                underflow_predicate=invalid_obj,  # not callable
                overflow_tolerance=4,
                underflow_tolerance=1,
            )

    def test_invalid_mode_value(self) -> None:
        """Validate mode enum."""
        with pytest.raises(
            ValueError, match=r"(?i).*mode must be 'e_notation' or 'infinity'.*"
        ):
            DisplayFlow(
                mode="bad_mode",
                overflow_predicate=_pred_true,
                underflow_predicate=_pred_false,
                overflow_tolerance=5,
                underflow_tolerance=1,
            )

    @pytest.mark.parametrize(
        ("overflow_tolerance", "underflow_tolerance", "expect_exc", "match"),
        [
            pytest.param(
                "3",
                1,
                TypeError,
                r"(?i).*overflow_tolerance must be int \| None.*",
                id="overflow_tolerance_type_error",
            ),
            pytest.param(
                2,
                {},
                TypeError,
                r"(?i).*underflow_tolerance must be int \| None.*",
                id="underflow_tolerance_type_error",
            ),
        ],
    )
    def test_invalid_tolerance_types(
        self,
        overflow_tolerance: object,
        underflow_tolerance: object,
        expect_exc: type[Exception],
        match: str,
    ) -> None:
        """Validate tolerance type errors."""
        with pytest.raises(expect_exc, match=match):
            DisplayFlow(
                mode="e_notation",
                overflow_predicate=_pred_true,
                underflow_predicate=_pred_false,
                overflow_tolerance=overflow_tolerance,
                underflow_tolerance=underflow_tolerance,
            )

    def test_merge_owner_type(self) -> None:
        """Validate owner type in merge."""
        flow = DisplayFlow(
            mode="e_notation",
            overflow_predicate=_pred_true,
            underflow_predicate=_pred_false,
            overflow_tolerance=7,
            underflow_tolerance=3,
        )
        with pytest.raises(TypeError, match=r"(?i).*owner must be DisplayValue.*"):
            flow.merge(owner=object())

    def test_merge_unset_owner(self) -> None:
        """Merge and unset owner explicitly."""
        flow = DisplayFlow(
            mode="infinity",
            overflow_predicate=_pred_true,
            underflow_predicate=_pred_true,
            overflow_tolerance=9,
            underflow_tolerance=4,
        )
        merged = flow.merge(owner=None)
        assert merged.overflow is False
        assert merged.underflow is False

    def test_merge_overrides_and_immutability(self) -> None:
        """Override fields via merge and keep original intact."""

        def p_old_over(_dv) -> bool:
            return False

        def p_old_under(_dv) -> bool:
            return False

        def p_new_over(_dv) -> bool:
            return True

        def p_new_under(_dv) -> bool:
            return True

        base = DisplayFlow(
            mode="e_notation",
            overflow_predicate=p_old_over,
            underflow_predicate=p_old_under,
            overflow_tolerance=6,
            underflow_tolerance=2,
        )
        merged = base.merge(
            mode="infinity",
            overflow_predicate=p_new_over,
            underflow_predicate=p_new_under,
            overflow_tolerance=10,
            underflow_tolerance=5,
            owner=None,
        )

        # New instance with overrides applied
        assert merged is not base
        assert merged.mode == "infinity"
        assert merged.overflow_tolerance == 10
        assert merged.underflow_tolerance == 5
        assert merged._overflow_predicate is p_new_over
        assert merged._underflow_predicate is p_new_under

        # Original remains unchanged
        assert base.mode == "e_notation"
        assert base.overflow_tolerance == 6
        assert base.underflow_tolerance == 2
        assert base._overflow_predicate is p_old_over
        assert base._underflow_predicate is p_old_under


class TestDisplayFormat:
    """Core tests for DisplayFormat covering validation, formatting, and errors."""

    @pytest.mark.parametrize(
        "mult,base,power,expected",
        [
            pytest.param("caret", 10, 3, "10^3", id="caret_10_3"),
            pytest.param("latex", 10, 3, "10^{3}", id="latex_10_3"),
            pytest.param("python", 10, 3, "10**3", id="python_10_3"),
            pytest.param("unicode", 10, 3, "10³", id="unicode_10_3"),
            pytest.param("caret", 2, 5, "2^5", id="caret_2_5"),
            pytest.param("latex", 2, 5, "2^{5}", id="latex_2_5"),
            pytest.param("python", 2, 5, "2**5", id="python_2_5"),
            pytest.param("unicode", 2, 5, "2⁵", id="unicode_2_5"),
        ],
    )
    def test_mult_exp_formatting(
        self, mult: str, base: int, power: int, expected: str
    ) -> None:
        """Format exponent according to selected style and base."""
        fmt = DisplayFormat(mult=mult)
        assert fmt.mult_exp(base=base, power=power) == expected

    def test_mult_exp_zero_power(self) -> None:
        """Return empty string when power is zero."""
        fmt = DisplayFormat(mult="caret")
        assert fmt.mult_exp(base=10, power=0) == ""

    @pytest.mark.parametrize(
        "base,power,err_type,match",
        [
            pytest.param(
                "10", 3, TypeError, r"(?i).*base must be an int.*", id="nonint_base"
            ),
            pytest.param(
                10, "3", TypeError, r"(?i).*power must be an int.*", id="nonint_power"
            ),
        ],
    )
    def test_mult_exp_type_errors(self, base, power, err_type, match) -> None:
        """Raise TypeError when base or power is non-integer."""
        fmt = DisplayFormat(mult="python")
        with pytest.raises(err_type, match=match):
            fmt.mult_exp(base=base, power=power)

    def test_invalid_mult_raises_valueerror(self) -> None:
        """Raise ValueError for unsupported mult format."""
        with pytest.raises(ValueError, match=r"(?i).*expected one of.*but found.*"):
            DisplayFormat(mult="invalid")

    def test_invalid_symbols_raises_valueerror(self) -> None:
        """Raise ValueError for unsupported symbols preset."""
        with pytest.raises(
            ValueError, match=r"(?i).*symbols preset expected one of.*but found.*"
        ):
            DisplayFormat(symbols="bad")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "factory,exp_mult,exp_symbols",
        [
            pytest.param(DisplayFormat.ascii, "caret", "ascii", id="ascii_preset"),
            pytest.param(
                DisplayFormat.unicode, "unicode", "unicode", id="unicode_preset"
            ),
        ],
    )
    def test_factories_presets(self, factory, exp_mult: str, exp_symbols: str) -> None:
        """Return correct presets from factory constructors."""
        fmt = factory()
        assert isinstance(fmt, DisplayFormat)
        assert fmt.mult == exp_mult
        assert fmt.symbols == exp_symbols

    @pytest.mark.parametrize(
        "initial,override,expected",
        [
            pytest.param("caret", "latex", "latex", id="override_to_latex"),
            pytest.param("python", "unicode", "unicode", id="override_to_unicode"),
        ],
    )
    def test_merge_override_mult(
        self, initial: str, override: str, expected: str
    ) -> None:
        """Return new instance with overridden mult."""
        fmt = DisplayFormat(mult=initial)
        merged = fmt.merge(mult=override)
        assert merged.mult == expected
        assert merged.symbols == fmt.symbols
        assert merged is not fmt

    @pytest.mark.parametrize(
        "initial",
        [
            pytest.param("caret", id="keep_caret"),
            pytest.param("unicode", id="keep_unicode"),
        ],
    )
    def test_merge_inherit_mult_explicit_unset(self, initial: str) -> None:
        """Inherit mult when unset."""
        fmt = DisplayFormat(mult=initial)
        merged = fmt.merge()
        assert merged.mult == initial
        assert merged.symbols == fmt.symbols
        assert merged is not fmt

    @pytest.mark.parametrize(
        "initial_symbols,override,expected",
        [
            pytest.param("ascii", "unicode", "unicode", id="ascii_to_unicode"),
            pytest.param("unicode", "ascii", "ascii", id="unicode_to_ascii"),
        ],
    )
    def test_merge_override_symbols(
        self, initial_symbols: str, override: str, expected: str
    ) -> None:
        """Return new instance with overridden symbols."""
        fmt = DisplayFormat(symbols=initial_symbols)
        merged = fmt.merge(symbols=override)
        assert merged.symbols == expected
        assert merged.mult == fmt.mult
        assert merged is not fmt

    @pytest.mark.parametrize(
        "initial_symbols",
        [
            pytest.param("ascii", id="keep_ascii"),
        ],
    )
    def test_merge_inherit_symbols_explicit_unset(self, initial_symbols: str) -> None:
        """Inherit symbols when unset."""
        fmt = DisplayFormat(symbols=initial_symbols)
        merged = fmt.merge()
        assert merged.symbols == initial_symbols
        assert merged.mult == fmt.mult
        assert merged is not fmt

    @pytest.mark.parametrize(
        "initial_mult,initial_symbols,override_mult,override_symbols,exp_mult,exp_symbols",
        [
            pytest.param(
                "caret",
                "ascii",
                "python",
                "unicode",
                "python",
                "unicode",
                id="override_both",
            ),
        ],
    )
    def test_merge_override_both(
        self,
        initial_mult: str,
        initial_symbols: str,
        override_mult: str,
        override_symbols: str,
        exp_mult: str,
        exp_symbols: str,
    ) -> None:
        """Return new instance with both mult and symbols overridden."""
        fmt = DisplayFormat(mult=initial_mult, symbols=initial_symbols)
        merged = fmt.merge(mult=override_mult, symbols=override_symbols)
        assert merged.mult == exp_mult
        assert merged.symbols == exp_symbols
        assert merged is not fmt

    @pytest.mark.parametrize(
        "field,value",
        [
            pytest.param("mult", "python", id="immutable_mult"),
            pytest.param("symbols", "unicode", id="immutable_symbols"),
        ],
    )
    def test_frozen_immutability(self, field: str, value: str) -> None:
        """Raise FrozenInstanceError when trying to mutate fields."""
        fmt = DisplayFormat()
        with pytest.raises(FrozenInstanceError, match=r"(?i).*cannot assign.*"):
            setattr(fmt, field, value)


class TestDisplayScale:
    @pytest.mark.parametrize(
        ("val", "expected"),
        [
            pytest.param(0.00234, -3, id="small_fraction"),
            pytest.param(4.56, 0, id="unit_range"),
            pytest.param(86, 1, id="two_digits"),
            pytest.param(-450, 2, id="negative_abs"),
        ],
    )
    def test_decimal_exp(self, val: int | float | None, expected: int) -> None:
        """Compute exponent for decimal scale."""
        scale = DisplayScale(type="decimal")
        assert scale.value_exponent(val) == expected

    @pytest.mark.parametrize(
        ("val", "expected"),
        [
            pytest.param(1, 0, id="one"),
            pytest.param(1024, 10, id="pow2"),
            pytest.param(0.72, -1, id="fraction"),
            pytest.param(3, 1, id="between2and4"),
            pytest.param(-5, 2, id="negative_abs"),
        ],
    )
    def test_binary_exp(self, val: int | float | None, expected: int) -> None:
        """Compute exponent for binary scale."""
        scale = DisplayScale(type="binary")
        assert scale.value_exponent(val) == expected

    @pytest.mark.parametrize(
        ("scale_type", "val", "expected"),
        [
            pytest.param("decimal", 0, 0, id="decimal_zero"),
            pytest.param("binary", 0, 0, id="binary_zero"),
            pytest.param("decimal", None, None, id="decimal_none"),
            pytest.param("binary", None, None, id="binary_none"),
        ],
    )
    def test_zero_none(self, scale_type: str, val, expected) -> None:
        """Handle zero and None consistently."""
        scale = DisplayScale(type=scale_type)
        assert scale.value_exponent(val) == expected

    def test_bad_value_type(self) -> None:
        """Reject non-numeric value types."""
        scale = DisplayScale(type="decimal")
        with pytest.raises(
            TypeError, match=r"(?is).*type validation failed.*int.*float.*"
        ):
            scale.value_exponent("oops")  # type: ignore[arg-type]

    def test_bad_scale_type(self) -> None:
        """Reject invalid scale type at init."""
        with pytest.raises(
            ValueError,
            match=r"(?i).*scale type 'binary' or 'decimal' literal expected.*",
        ):
            DisplayScale(type="hex")  # type: ignore[arg-type]

    def test_base_not_int(self) -> None:
        """Reject non-int base at runtime."""
        scale = DisplayScale(type="decimal")
        object.__setattr__(scale, "base", "10")
        with pytest.raises(ValueError, match=r"(?i).*scale base must be 2 or 10.*"):
            scale.value_exponent(1)

    def test_base_not_supported(self) -> None:
        """Reject unsupported base values."""
        scale = DisplayScale(type="binary")
        object.__setattr__(scale, "base", 3)
        with pytest.raises(ValueError, match=r"(?i).*scale base must be 2 or 10.*"):
            scale.value_exponent(8)


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
        ],
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
        ],
    )
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

    @pytest.mark.parametrize(
        ("field_name", "bad_value"),
        [
            pytest.param("nan", 123, id="nan"),
            pytest.param("none", 123, id="none"),
            pytest.param("pos_infinity", 0, id="pos_infinity"),
            pytest.param("neg_infinity", 0, id="neg_infinity"),
            pytest.param("pos_underflow", 0, id="pos_underflow"),
            pytest.param("neg_underflow", 0, id="neg_underflow"),
            pytest.param("mult", 3.14, id="mult"),
            pytest.param("separator", 1, id="separator"),
            pytest.param("ellipsis", 1, id="ellipsis"),
        ],
    )
    def test_invalid_types(self, field_name: str, bad_value: object) -> None:
        """Raise TypeError and mention field name when invalid type is provided."""
        valid_kwargs = {
            "nan": "NaN",
            "none": "None",
            "pos_infinity": "inf",
            "neg_infinity": "-inf",
            "pos_underflow": "0",
            "neg_underflow": "-0",
            "mult": "*",
            "separator": " ",
            "ellipsis": "...",
        }
        # Override one field with an invalid value
        valid_kwargs[field_name] = bad_value
        with pytest.raises(TypeError, match=rf"(?i){field_name}"):
            DisplaySymbols(
                nan=valid_kwargs["nan"],
                none=valid_kwargs["none"],
                pos_infinity=valid_kwargs["pos_infinity"],
                neg_infinity=valid_kwargs["neg_infinity"],
                pos_underflow=valid_kwargs["pos_underflow"],
                neg_underflow=valid_kwargs["neg_underflow"],
                mult=valid_kwargs["mult"],
                separator=valid_kwargs["separator"],
                ellipsis=valid_kwargs["ellipsis"],
            )

    # Core tests for DisplaySymbols.merge() behavior ------------------------------------------

    @pytest.mark.parametrize(
        ("field", "override", "expected"),
        [
            pytest.param("nan", "N/A", "N/A", id="nan_override"),
            pytest.param("none", "NULL", "NULL", id="none_override"),
            pytest.param("pos_infinity", "+INF", "+INF", id="pos_inf_override"),
            pytest.param("neg_infinity", "-INF", "-INF", id="neg_inf_override"),
            pytest.param("pos_underflow", "~0", "~0", id="pos_underflow_override"),
            pytest.param("neg_underflow", "~0", "~0", id="neg_underflow_override"),
            pytest.param("separator", "_", "_", id="separator_override"),
            pytest.param("ellipsis", "…", "…", id="ellipsis_override"),
        ],
    )
    def test_override_single_field(
        self, field: str, override: str, expected: str
    ) -> None:
        """Override a single field and return new instance."""
        base = DisplaySymbols()
        kwargs = {field: override}
        merged = base.merge(**kwargs)
        assert getattr(merged, field) == expected
        assert merged is not base

    def test_override_mult_symbol(self) -> None:
        """Override mult symbol with a new MultSymbol."""
        base = DisplaySymbols(mult=MultSymbol.ASTERISK)
        merged = base.merge(mult=MultSymbol.CDOT)
        assert merged.mult == MultSymbol.CDOT
        assert merged is not base

    def test_inherit_unset_fields(self) -> None:
        """Inherit all fields when no overrides provided."""
        base = DisplaySymbols(nan="NaN", separator=" ")
        merged = base.merge()
        assert merged == base
        assert merged is not base

    def test_multiple_overrides(self) -> None:
        """Apply multiple overrides simultaneously."""
        base = DisplaySymbols()
        merged = base.merge(
            nan="N/A",
            none="NULL",
            mult=MultSymbol.CROSS,
            separator="_",
        )
        assert merged.nan == "N/A"
        assert merged.none == "NULL"
        assert merged.mult == MultSymbol.CROSS
        assert merged.separator == "_"

    def test_invalid_type_raises(self) -> None:
        """Raise TypeError when invalid type is passed."""
        base = DisplaySymbols()
        with pytest.raises(TypeError, match=r"(?i).*nan.*"):
            base.merge(nan=123)  # type: ignore[arg-type]

    def test_merge_returns_new_instance(self) -> None:
        """Ensure merge returns a distinct frozen instance."""
        base = DisplaySymbols()
        merged = base.merge(separator="_")
        assert merged is not base
        assert isinstance(merged, DisplaySymbols)
        assert merged.separator == "_"

    def test_merge_immutability_preserved(self) -> None:
        """Ensure merged instance remains frozen."""
        base = DisplaySymbols()
        merged = base.merge(separator="_")
        with pytest.raises(Exception, match=r"(?i).*assign.*"):
            merged.nan = "changed"  # type: ignore[assignment]


# Factory Methods Tests ---------------------------------------------------------------


class TestDisplayValueFactoryBaseFixed:
    """Tests for DisplayValue.base_fixed() factory method."""

    @pytest.mark.parametrize(
        "value, unit, expected",
        [
            pytest.param(1_500_000, "byte", "1.5×10⁶ bytes", id="auto_scale_mega"),
            pytest.param(123, "byte", "123 bytes", id="no_scale_small"),
            pytest.param(0.000123, "second", "123×10⁻⁶ seconds", id="auto_scale_micro"),
            pytest.param(42, "meter", "42 meters", id="moderate_no_scale"),
        ],
    )
    def test_base_fixed_auto_scaling(self, value, unit, expected):
        """BASE_FIXED auto-scales multiplier to keep value compact."""
        dv = DisplayValue.base_fixed(value, unit=unit)
        assert dv.mode == DisplayMode.BASE_FIXED
        assert dv.mult_exp is None  # Auto-calculated
        assert dv.unit_exp == 0  # Always base units
        assert str(dv) == expected

    def test_base_fixed_trim_and__precision(self):
        """Precision formats normalized value in BASE_FIXED mode."""
        dv = DisplayValue.base_fixed(
            123_456_789, unit="byte", trim_digits=123456, precision=2
        )
        result = str(dv)
        assert dv.mode == DisplayMode.BASE_FIXED
        assert dv.trim_digits == 123456
        assert dv.precision == 2
        assert "123.46" in result or "123,46" in result  # Locale-independent
        assert "×10" in result
        assert "bytes" in result

    @pytest.mark.parametrize(
        "fmt,format_factory",
        [
            pytest.param("ascii", DisplayFormat.ascii, id="ascii-format"),
            pytest.param("unicode", DisplayFormat.unicode, id="unicode-format"),
        ],
    )
    def test_base_fixed_format(self, fmt, format_factory):
        """Verify that base_fixed uses the correct DisplayFormat for each format type."""
        dv = DisplayValue.base_fixed(123, format=fmt)
        format_ = format_factory()
        assert dv.format == format_

    @pytest.mark.parametrize(
        "scale,scale_instance",
        [
            pytest.param("binary", DisplayScale(type="binary"), id="binary-scale"),
            pytest.param("decimal", DisplayScale(type="decimal"), id="decimal-scale"),
        ],
    )
    def test_base_fixed_scale(self, scale, scale_instance):
        """Verify that base_fixed uses the correct DisplayFormat for each format type."""
        dv = DisplayValue.base_fixed(123, scale=scale)
        assert dv.scale == scale_instance


class TestDisplayValueFactoryPlain:
    """Tests for DisplayValue.plain() factory method."""

    @pytest.mark.parametrize(
        "value, unit, expected_str",
        [
            pytest.param(42, "byte", "42 bytes", id="int_plural"),
            pytest.param(1, "byte", "1 byte", id="int_singular"),
            pytest.param(0, "byte", "0 bytes", id="zero_plural"),
            pytest.param(-5, "meter", "-5 meters", id="negative"),
            pytest.param(123_000, "byte", "123000 bytes", id="large_no_scale"),
            pytest.param(3.14159, "meter", "3.14159 meters", id="float_e+0"),
            pytest.param(123.456e123, "s", "1.23456e+125 s", id="float_e+125"),
        ],
    )
    def test_basic_plain_display(self, value, unit, expected_str):
        """Plain mode displays values as-is without scaling."""
        dv = DisplayValue.plain(value, unit=unit)
        assert dv.mode == DisplayMode.PLAIN
        assert dv.mult_exp == 0
        assert dv.unit_exp == 0
        assert str(dv) == expected_str

    def test_plain_with_precision(self):
        """Precision controls decimal places in plain mode."""
        dv = DisplayValue.plain(3.14159, unit="meter", precision=2)
        assert dv.mode == DisplayMode.PLAIN
        assert "3.14" in str(dv)

    def test_plain_with_trim_digits(self):
        """Trim digits reduces significant figures in plain mode."""
        dv = DisplayValue.plain(123.456789, unit="second", trim_digits=5)
        assert dv.mode == DisplayMode.PLAIN
        assert str(dv) == "123.46 seconds"

    def test_plain_with_precision_precedence(self):
        """Precision takes precedence over trim_digits."""
        dv = DisplayValue.plain(1 / 3, unit="meter", precision=2, trim_digits=10)
        assert dv.mode == DisplayMode.PLAIN
        assert str(dv) == "0.33 meters"

    @pytest.mark.parametrize(
        "value, expected_contains",
        [
            pytest.param(None, "None", id="none"),
            pytest.param(float("inf"), "+∞ bytes", id="inf"),
            pytest.param(float("nan"), "NaN", id="nan"),
        ],
    )
    def test_plain_non_finite(self, value, expected_contains):
        """Plain mode handles non-finite values."""
        dv = DisplayValue.plain(value, unit="byte")
        assert dv.mode == DisplayMode.PLAIN
        assert expected_contains in str(dv)

    @pytest.mark.parametrize(
        "fmt,format_factory",
        [
            pytest.param("ascii", DisplayFormat.ascii, id="ascii-format"),
            pytest.param("unicode", DisplayFormat.unicode, id="unicode-format"),
        ],
    )
    def test_plain_format(self, fmt, format_factory):
        """Verify that base_fixed uses the correct DisplayFormat for each format type."""
        dv = DisplayValue.plain(123, format=fmt)
        format_ = format_factory()
        assert dv.format == format_


class TestDisplayValueFactorySIFixed:
    """Tests for DisplayValue.si_fixed() factory method."""

    def test_si_fixed_from_base_value(self):
        """Create from base units, fixed SI prefix."""
        dv = DisplayValue.si_fixed(value=123_000_000, si_unit="Mbyte")
        assert dv.mode == DisplayMode.UNIT_FIXED
        assert dv.mult_exp is None  # Auto-selected
        assert dv.unit_exp == 6
        assert str(dv) == "123 Mbytes"

    def test_si_fixed_from_si_value(self):
        """Create from SI-prefixed value."""
        dv = DisplayValue.si_fixed(si_value=123, si_unit="Mbyte")
        assert dv.mode == DisplayMode.UNIT_FIXED
        assert "123" in str(dv)
        assert "Mbyte" in str(dv)
        # Internally converts to base: 123 * 10^6
        assert dv.value == 123_000_000

    def test_si_fixed_mutual_exclusion(self):
        """Cannot specify both value and si_value."""
        with pytest.raises(
            ValueError, match="only one of 'value' or 'si_value' allowed"
        ):
            DisplayValue.si_fixed(value=100, si_value=200, si_unit="Mbyte")

    def test_si_fixed_none_values(self):
        """Must specify either value or si_value."""
        dv = DisplayValue.si_fixed(value=None, si_value=None, si_unit="Mbyte")
        assert dv.mode == DisplayMode.UNIT_FIXED
        assert str(dv) == "None"

    def test_si_fixed_with_multiplier(self):
        """UNIT_FIXED adds multiplier when needed."""
        dv = DisplayValue.si_fixed(value=123_000_000_000, si_unit="Mbyte")
        assert dv.mode == DisplayMode.UNIT_FIXED
        assert "×10" in str(dv) or "×10^3" in str(dv)
        assert "Mbyte" in str(dv)

    @pytest.mark.parametrize(
        "fmt,format_factory",
        [
            pytest.param("ascii", DisplayFormat.ascii, id="ascii-format"),
            pytest.param("unicode", DisplayFormat.unicode, id="unicode-format"),
        ],
    )
    def test_si_fixed_format(self, fmt, format_factory):
        """Verify that base_fixed uses the correct DisplayFormat for each format type."""
        dv = DisplayValue.si_fixed(123, si_unit="u", format=fmt)
        format_ = format_factory()
        assert dv.format == format_

    @pytest.mark.parametrize(
        "overflow,flow_instance",
        [
            pytest.param(
                "e_notation", DisplayFlow(mode="e_notation"), id="e_notation-flow-mode"
            ),
            pytest.param(
                "infinity", DisplayFlow(mode="infinity"), id="infinity-flow-mode"
            ),
        ],
    )
    def test_si_fixed_overflow(self, overflow, flow_instance):
        """Verify that base_fixed uses the correct DisplayFormat for each format type."""
        dv = DisplayValue.si_fixed(123, si_unit="u", overflow=overflow)
        flow_ = flow_instance.merge(owner=dv)
        assert dv.flow == flow_


class TestDisplayValueFactorySIFlex:
    """Tests for DisplayValue.si_flex() factory method."""

    @pytest.mark.parametrize(
        "value, unit, expected_unit_suffix",
        [
            pytest.param(1_500_000, "byte", "Mbytes", id="mega_bytes"),
            pytest.param(2_500, "byte", "kbytes", id="kilo_bytes"),
            pytest.param(0.000123, "second", "µs", id="micro_seconds"),
            pytest.param(42, "meter", "meters", id="base_no_prefix"),
        ],
    )
    def test_si_flex_auto_prefix(self, value, unit, expected_unit_suffix):
        """UNIT_FLEX auto-selects SI prefix for optimal display."""
        dv = DisplayValue.si_flex(value, unit=unit)
        assert dv.mode == DisplayMode.UNIT_FLEX
        assert dv.mult_exp == 0  # Default multiplier is 10^0 = 1
        assert dv.unit_exp is None  # Auto-selected
        assert expected_unit_suffix in str(dv)

    def test_si_flex_no_unit(self):
        """UNIT_FLEX works without unit (prefix only)."""
        dv = DisplayValue.si_flex(1_500_000)
        assert dv.mode == DisplayMode.UNIT_FLEX
        assert str(dv) == "1.5M"  # Just prefix, no unit

    def test_si_flex_with_mult_exp(self):
        """UNIT_FLEX allows explicit mult_exp."""
        dv = DisplayValue.si_flex(123_000_000, unit="byte", mult_exp=3)
        assert dv.mode == DisplayMode.UNIT_FLEX
        assert "×10³" in str(dv) or "×10^3" in str(dv)

    def test_si_flex_custom_prefixes(self):
        """UNIT_FLEX respects custom unit_prefixes."""
        custom_prefixes = {0: "", 9: "G"}  # Only base and giga
        dv = DisplayValue.si_flex(
            500_000_000, unit="byte", unit_prefixes=custom_prefixes
        )
        # Should select 'G' even though value is between k and M
        assert dv.mode == DisplayMode.UNIT_FLEX
        assert "Gbytes" in str(dv)

    @pytest.mark.parametrize(
        "fmt,format_factory",
        [
            pytest.param("ascii", DisplayFormat.ascii, id="ascii-format"),
            pytest.param("unicode", DisplayFormat.unicode, id="unicode-format"),
        ],
    )
    def test_si_flex_format(self, fmt, format_factory):
        """Verify that base_fixed uses the correct DisplayFormat for each format type."""
        dv = DisplayValue.si_flex(123, unit="u", format=fmt)
        format_ = format_factory()
        assert dv.format == format_

    @pytest.mark.parametrize(
        "overflow,flow_instance",
        [
            pytest.param(
                "e_notation", DisplayFlow(mode="e_notation"), id="e_notation-flow-mode"
            ),
            pytest.param(
                "infinity", DisplayFlow(mode="infinity"), id="infinity-flow-mode"
            ),
        ],
    )
    def test_si_flex_overflow(self, overflow, flow_instance):
        """Verify that base_fixed uses the correct DisplayFormat for each format type."""
        dv = DisplayValue.si_flex(123, unit="u", overflow=overflow)
        flow_ = flow_instance.merge(owner=dv)
        assert dv.flow == flow_


# Value Type Conversion Tests ---------------------------------------------------------------


class TestDisplayValueTypeConversion:
    """Tests for std_numeric() type conversion integration."""

    @pytest.mark.parametrize(
        "value, expected_type",
        [
            pytest.param(42, int, id="int"),
            pytest.param(3.14, float, id="float"),
            pytest.param(Decimal("3.14"), float, id="decimal"),
            pytest.param(Fraction(22, 7), float, id="fraction"),
            pytest.param(None, type(None), id="None"),
            pytest.param(math.nan, float, id="NaN"),
            pytest.param(math.inf, float, id="inf"),
        ],
    )
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

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(float("inf"), id="inf"),
            pytest.param(float("-inf"), id="neg_inf"),
            pytest.param(float("nan"), id="nan"),
        ],
    )
    def test_non_finite_values(self, value):
        """Accept non-finite float values."""
        dv = DisplayValue(value, unit="byte")
        assert not dv.is_finite
        assert dv.value == value or (math.isnan(dv.value) and math.isnan(value))

    # NumPy type tests (conditional on numpy availability)
    @pytest.mark.skipif(
        not hasattr(pytest, "importorskip"), reason="Requires pytest.importorskip"
    )
    def test_numpy_types(self):
        """Convert NumPy types to stdlib equivalents."""
        np = pytest.importorskip("numpy")

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
        not hasattr(pytest, "importorskip"), reason="Requires pytest.importorskip"
    )
    def test_pandas_types(self):
        """Convert Pandas types to stdlib equivalents."""
        pd = pytest.importorskip("pandas")

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

    @pytest.mark.parametrize(
        "mult_exp, unit_exp, expected_mode",
        [
            pytest.param(0, 0, DisplayMode.PLAIN, id="plain"),
            pytest.param(3, 6, DisplayMode.FIXED, id="fixed_both"),
            pytest.param(0, 3, DisplayMode.FIXED, id="fixed_unit_only"),
            pytest.param(3, 0, DisplayMode.FIXED, id="fixed_mult_only"),
            pytest.param(None, 0, DisplayMode.BASE_FIXED, id="base_fixed"),
            pytest.param(None, 3, DisplayMode.UNIT_FIXED, id="unit_fixed"),
            pytest.param(0, None, DisplayMode.UNIT_FLEX, id="unit_flex_0"),
            pytest.param(3, None, DisplayMode.UNIT_FLEX, id="unit_flex_3"),
            pytest.param(None, None, DisplayMode.BASE_FIXED, id="both_none"),
        ],
    )
    def test_mode_inference(self, mult_exp, unit_exp, expected_mode):
        """Correctly infer mode from exponent combination."""
        dv = DisplayValue(123, unit="byte", mult_exp=mult_exp, unit_exp=unit_exp)
        assert dv.mode == expected_mode


class TestDisplayValueNormalisedTypeConversion:
    """Tests for std_numeric() type conversion integration."""

    @pytest.mark.parametrize(
        "value, expected_type",
        [
            pytest.param(42, int, id="int"),
            pytest.param(3.14, float, id="float"),
            pytest.param(1e100, float, id="float"),
            pytest.param(Decimal("3.14"), float, id="decimal"),
            pytest.param(Fraction(22, 7), float, id="fraction"),
            pytest.param(None, type(None), id="None"),
            pytest.param(math.nan, float, id="NaN"),
            pytest.param(math.inf, float, id="inf"),
        ],
    )
    def test_stdlib_types(self, value, expected_type):
        """Accept and convert stdlib numeric types."""
        dv = DisplayValue(value)
        assert isinstance(dv.normalized, expected_type)

    def test_rescaled_and_normalized(self):
        """unit_fixed with rate units (e.g., MB/s)."""
        dv = DisplayValue(value=500_000_000, unit_exp=6)
        result = str(dv)
        assert isinstance(dv.normalized, int)
        assert result == "500M"

        dv = DisplayValue(value=125.0e6, unit_exp=6)
        print(dv)
        assert isinstance(dv.normalized, float)


class TestDisplayValueStringFormatting:
    """Tests for __str__ output across modes and scales."""

    # PLAIN mode
    @pytest.mark.parametrize(
        "value, unit, expected",
        [
            pytest.param(123, "B", "123 B", id="plain_int"),
            pytest.param(1, "byte", "1 byte", id="plain_singular"),
            pytest.param(2, "byte", "2 bytes", id="plain_plural"),
        ],
    )
    def test_plain_mode_formatting(self, value, unit, expected):
        """PLAIN mode string formatting."""
        dv = DisplayValue.plain(value, unit=unit)
        assert str(dv) == expected

    # FIXED mode (decimal)
    @pytest.mark.parametrize(
        "value, mult_exp, unit_exp, expected",
        [
            pytest.param(123, 0, 3, "0.123 kB", id="fixed_0_3"),
            pytest.param(123, 3, 0, "0.123×10³ B", id="fixed_3_0"),
            pytest.param(123456, 3, 6, "0.000123456×10³ MB", id="fixed_both"),
        ],
    )
    def test_fixed_mode_decimal(self, value, mult_exp, unit_exp, expected):
        """FIXED mode with decimal scale."""
        dv = DisplayValue(
            value,
            unit="B",
            mult_exp=mult_exp,
            unit_exp=unit_exp,
            format=DisplayFormat.unicode(),
        )
        # Normalize expected string (remove superscripts for comparison)
        result = str(dv).replace("³", "3")
        expected_norm = expected.replace("³", "3")
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
    def test_si_flex_formatting(self):
        """UNIT_FLEX mode string formatting."""
        dv = DisplayValue.si_flex(1_500_000, unit="byte")
        assert str(dv) == "1.5 Mbytes"
        assert dv.mode == DisplayMode.UNIT_FLEX

    # Binary scale
    @pytest.mark.parametrize(
        "value, mult_exp, unit_exp, expected_contains",
        [
            pytest.param(123, 0, 0, "123 B", id="binary_plain"),
            pytest.param(123, 0, 10, "KiB", id="binary_Ki"),
            pytest.param(2**30 * 0.123, 30, 0, ["2³⁰", "B"], id="binary_mult"),
        ],
    )
    def test_binary_scale_formatting(
        self, value, mult_exp, unit_exp, expected_contains
    ):
        """Binary scale string formatting."""
        scale = DisplayScale(type="binary")
        dv = DisplayValue(
            value,
            unit="B",
            mult_exp=mult_exp,
            unit_exp=unit_exp,
            scale=scale,
            format=DisplayFormat.unicode(),
        )
        result = str(dv)
        if isinstance(expected_contains, list):
            for substr in expected_contains:
                assert substr in result or substr.replace("³⁰", "30") in result
        else:
            assert expected_contains in result


# Properties & Computed Values ---------------------------------------------------------------


class TestDisplayValueProperties:
    """Tests for computed properties."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            pytest.param(None, False, id="none"),
            pytest.param(math.inf, False, id="inf"),
            pytest.param(-math.inf, False, id="neg_inf"),
            pytest.param(math.nan, False, id="nan"),
            pytest.param(0, True, id="zero"),
            pytest.param(42, True, id="int"),
            pytest.param(3.14, True, id="float"),
        ],
    )
    def test_is_finite(self, value, expected):
        """is_finite property correctness."""
        dv = DisplayValue(value, unit="byte")
        assert dv.is_finite == expected

    @pytest.mark.parametrize(
        "scale_type, mult_exp, expected",
        [
            pytest.param("decimal", 3, 1000, id="dec_10_3"),
            pytest.param("decimal", 0, 1, id="dec_10_0"),
            pytest.param("decimal", 6, 1_000_000, id="dec_10_6"),
            pytest.param("binary", 10, 1024, id="bin_2_10"),
            pytest.param("binary", 20, 2**20, id="bin_2_20"),
            pytest.param("binary", 0, 1, id="bin_2_0"),
        ],
    )
    def test_mult_value(self, scale_type, mult_exp, expected):
        """mult_value computes correct multiplier."""
        scale = DisplayScale(type=scale_type)
        dv = DisplayValue(123, unit="B", mult_exp=mult_exp, unit_exp=0, scale=scale)
        assert dv.mult_value == expected

    @pytest.mark.parametrize(
        "scale_type, unit_exp, expected",
        [
            pytest.param("decimal", 6, 1_000_000, id="dec_M"),
            pytest.param("decimal", 3, 1_000, id="dec_k"),
            pytest.param("decimal", 0, 1, id="dec_base"),
            pytest.param("binary", 20, 2**20, id="bin_Mi"),
            pytest.param("binary", 10, 1024, id="bin_Ki"),
            pytest.param("binary", 0, 1, id="bin_base"),
        ],
    )
    def test_unit_value(self, scale_type, unit_exp, expected):
        """unit_value computes correct unit prefix value."""
        scale = DisplayScale(type=scale_type)
        dv = DisplayValue(123, unit="B", mult_exp=0, unit_exp=unit_exp, scale=scale)
        assert dv.unit_value == expected

    def test_ref_value(self):
        """ref_value = mult_value * unit_value."""
        dv = DisplayValue(123_456_789, unit="B", mult_exp=3, unit_exp=6)
        assert dv.ref_value == 1000 * 1_000_000
        assert dv.ref_value == dv.mult_value * dv.unit_value

    def test_normalized_calculation(self):
        """normalized = value / ref_value."""
        dv = DisplayValue(123_000_000, unit="byte", mult_exp=3, unit_exp=6)
        expected = 123_000_000 / (1000 * 1_000_000)
        assert abs(dv.normalized - expected) < 0.001

    def test_unit_prefix_extraction(self):
        """unit_prefix extracts prefix from mapping."""
        dv = DisplayValue.si_flex(1_500_000, unit="byte")
        assert dv.unit_prefix == "M"

    @pytest.mark.parametrize(
        "value, pluralize, expected",
        [
            pytest.param(1, True, "byte", id="singular"),
            pytest.param(0, True, "bytes", id="zero_plural"),
            pytest.param(2, True, "bytes", id="two_plural"),
            pytest.param(1.0, True, "byte", id="one_float_singular"),
            pytest.param(1.5, True, "bytes", id="float_plural"),
            pytest.param(5, False, "byte", id="no_pluralize"),
        ],
    )
    def test_units_pluralization(self, value, pluralize, expected):
        """units property handles pluralization."""
        dv = DisplayValue(
            value, unit="byte", mult_exp=0, unit_exp=0, pluralize=pluralize
        )
        assert dv.units == expected

    def test_units_with_prefix(self):
        """units includes SI prefix."""
        dv = DisplayValue.si_flex(1_500_000, unit="byte")
        assert dv.units == "Mbytes"

    def test_units_prefix_only(self):
        """units shows prefix when unit is None."""
        dv = DisplayValue.si_flex(1_500_000)
        assert dv.units == "M"

    def test_number_property(self):
        """number property includes multiplier."""
        dv = DisplayValue.base_fixed(123_000, unit="byte")
        number = dv.number
        assert "123" in number
        assert "10" in number  # Multiplier

    def test_parts_tuple(self):
        """parts returns (number, units) tuple."""
        dv = DisplayValue.si_flex(1_500, unit="byte")
        number, units = dv.parts
        assert "1.5" in number
        assert units == "kbytes"


# Overflow/Underflow Behavior ---------------------------------------------------------------


class TestDisplayValueOverflowUnderflow:
    """Tests for overflow/underflow formatting behavior."""

    def test_infinity_mode(self):
        """FIXED mode overflow behavior."""
        dv = DisplayValue(
            1e100,
            unit="B",
            mult_exp=3,
            unit_exp=6,
            flow=DisplayFlow(overflow_tolerance=5, mode="infinity"),
        )
        assert dv.flow.overflow
        assert str(dv) == "+∞ MB"

    def test_e_notation_mode(self):
        """FIXED mode overflow behavior."""
        dv = DisplayValue(
            1e100,
            unit="B",
            mult_exp=3,
            unit_exp=6,
            flow=DisplayFlow(overflow_tolerance=5, mode="e_notation"),
        )
        assert dv.flow.overflow
        assert str(dv) == "1.000000e+91 MB"

    def test_si_flex_overflow_formatting(self):
        """UNIT_FLEX shows inf for overflow."""
        dv = DisplayValue.si_flex(1e100, unit="B")
        assert "+∞" in str(dv) or "inf" in str(dv)
        assert dv.flow.overflow

    def test_si_flex_underflow_formatting(self):
        """UNIT_FLEX shows ≈0 for underflow."""
        dv = DisplayValue.si_flex(1e-100, unit="B")
        result = str(dv)
        assert "≈0" in result or "+0" in result or "0" in result
        assert dv.flow.underflow

    @pytest.mark.parametrize(
        "value, unit, expected_str",
        [
            pytest.param(1e-100, "byte", "+0 bytes", id="tiny-underflow++"),
            pytest.param(-1e-100, "byte", "-0 bytes", id="tiny-underflow--"),
            pytest.param(1, "B", "1 B", id="normal"),
            pytest.param(1e100, "B", "+inf B", id="huge-overflow++"),
            pytest.param(-1e100, "B", "-inf B", id="huge-overflow--"),
        ],
    )
    def test_si_flex_tiny_huge(self, value, unit, expected_str):
        symbols = DisplaySymbols(
            pos_infinity="+inf",
            neg_infinity="-inf",
            pos_underflow="+0",
            neg_underflow="-0",
        )
        dv = DisplayValue(value, mult_exp=0, unit=unit, symbols=symbols)
        assert str(dv) == expected_str

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
        dv = DisplayValue(2500, unit="meter", mult_exp=0, unit_exp=None, flow=flow)
        assert dv.flow.overflow


# Composition Tests ---------------------------------------------------------------


class TestDisplayValueComposition:
    """Composition tests with DisplayFlow, DisplayFormat, DisplayScale, DisplaySymbols."""

    def test_custom_display_format(self):
        """Custom DisplayFormat integration."""
        fmt = DisplayFormat(mult="latex", symbols="ascii")
        dv = DisplayValue(123_000, unit_exp=0, unit="byte", format=fmt)
        result = str(dv)
        assert "10^{" in result  # LaTeX format

    def test_ascii_symbols(self):
        """ASCII symbols integration."""
        symbols = DisplaySymbols.ascii()
        dv = DisplayValue(float("inf"), unit="byte", symbols=symbols)
        assert "inf" in str(dv)
        assert "∞" not in str(dv)

    def test_unicode_symbols(self):
        """Unicode symbols integration."""
        symbols = DisplaySymbols.unicode()
        dv = DisplayValue(float("inf"), unit="byte", symbols=symbols)
        assert "∞" in str(dv)

    def test_binary_scale_integration(self):
        """Binary scale full integration."""
        scale = DisplayScale(type="binary")
        fmt = DisplayFormat.unicode()
        dv = DisplayValue(
            2**30, unit="B", mult_exp=20, unit_exp=10, scale=scale, format=fmt
        )
        result = str(dv)
        assert "KiB" in result
        assert "2²⁰" in result or "2^20" in result

    def test_custom_unit_plurals(self):
        """Custom pluralization mapping."""
        custom_plurals = {"datum": "data", "index": "indices"}
        dv = DisplayValue(
            5, unit="datum", mult_exp=0, unit_exp=0, unit_plurals=custom_plurals
        )
        assert "data" in str(dv)

    def test_custom_unit_prefixes(self):
        """Custom unit prefix mapping."""
        custom_prefixes = {0: "", 9: "G"}  # Only base and giga
        dv = DisplayValue.si_flex(500_000, unit="byte", unit_prefixes=custom_prefixes)
        # Should select closest available prefix
        assert "Gbytes" in str(dv)

    def test_flow_merge_with_owner(self):
        """DisplayFlow.merge() establishes owner backlink."""
        flow = DisplayFlow(overflow_tolerance=3)
        dv = DisplayValue(1e10, unit="B", mult_exp=3, unit_exp=6)
        merged_flow = flow.merge(owner=dv)
        # Flow should now evaluate predicates with dv as context
        assert merged_flow._owner is dv  # Should trigger based on dv's state


# Edge Cases & Validation ---------------------------------------------------------------


class TestDisplayValueEdgeCases:
    """Edge case and boundary condition tests."""

    @pytest.mark.parametrize(
        "unit",
        [
            pytest.param("", id="empty_string"),
            pytest.param("  ", id="whitespace"),
            pytest.param("a" * 1000, id="very_long"),
        ],
    )
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
        dv = DisplayValue.si_flex(1e-50, unit="second")
        # Should not crash
        result = str(dv)
        assert "second" in result or "s" in result

    def test_very_large_negative(self):
        """Very large negative values."""
        dv = DisplayValue.si_flex(-1e50, unit="meter")
        assert str(dv) == "−∞ meters"

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
        assert str(dv) == "100.0 meters"

    def test_whole_as_int_true(self):
        """whole_as_int converts 3.0 to "3"."""
        dv = DisplayValue(3.0, unit="meter", mult_exp=0, unit_exp=0, whole_as_int=True)
        result = str(dv)
        # Should show "3" not "3.0"
        assert result == "3 meters" or "3.0" not in result

    def test_whole_as_int_false(self):
        """whole_as_int=False keeps 3.0 as "3.0"."""
        dv = DisplayValue(3.0, unit="meter", mult_exp=0, unit_exp=0, whole_as_int=False)
        assert str(dv) == "3.0 meters"


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
            float("inf"), unit="byte", precision=2, trim_digits=5, whole_as_int=True
        )
        result = str(dv)
        assert "∞" in result or "inf" in result

    def test_pipeline_precision_over_trim(self):
        """Precision takes precedence over trim_digits."""
        dv = DisplayValue.plain(1 / 3, unit="meter", precision=2, trim_digits=10)
        result = str(dv)
        assert "0.33" in result
        assert len(result.split(".")[1].split()[0]) == 2  # Exactly 2 decimals

    def test_pipeline_trim_applied(self):
        """trim_digits reduces significant figures."""
        dv = DisplayValue.plain(123.456789, unit="second", trim_digits=4)
        result = str(dv)
        # Should have ~4 significant digits
        assert "123.5" in result or "123.4" in result

    def test_pipeline_whole_as_int_after_rounding(self):
        """whole_as_int applied after rounding."""
        dv = DisplayValue(
            2.999, mult_exp=0, unit_exp=0, unit="meter", precision=0, whole_as_int=True
        )
        result = str(dv)
        # Rounds to 3, then converts to int display
        assert result == "3 meters"

    def test_pipeline_overflow_formatting_last(self):
        """Overflow formatting applied at end."""
        dv = DisplayValue(
            1e100,
            unit="B",
            mult_exp=3,
            unit_exp=6,
            precision=2,  # Should be ignored due to overflow
            flow=DisplayFlow(overflow_tolerance=5, mode="infinity"),
        )
        assert str(dv) == "+∞ MB"


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
        dv = DisplayValue.si_flex(1_500_000, unit="byte")
        # Should normalize to 1.5 with M prefix
        assert abs(dv.normalized - 1.5) < 0.01

    def test_normalized_fixed_mode(self):
        """FIXED mode normalized with both exponents."""
        dv = DisplayValue(123_000_000, unit="byte", mult_exp=3, unit_exp=6)
        # normalized = 123_000_000 / (10^3 * 10^6)
        expected = 123_000_000 / (1000 * 1_000_000)
        assert abs(dv.normalized - expected) < 0.001

    def test_normalized_with_trim_digits(self):
        """Normalized includes trimming."""
        dv = DisplayValue.plain(123.456789, unit="meter", trim_digits=4)
        # Normalized should be rounded to 4 sig figs
        assert abs(dv.normalized - 123.5) < 0.1

    def test_normalized_extreme_overflow(self):
        """Normalized for extreme overflow values."""
        dv = DisplayValue.si_flex(1e100, unit="B")
        # Should still compute normalized (even if display shows inf)
        assert dv.normalized > 1e29  # Large but scaled
        assert str(dv) == "+∞ B"  # Large but scaled

    def test_normalized_extreme_underflow(self):
        """Normalized for extreme underflow values."""
        dv = DisplayValue.si_flex(1e-100, unit="B")
        # Should still compute normalized
        assert 0 < dv.normalized < 1


# Decimal Vs Binary Scale Comparison ---------------------------------------------------------------


class TestDisplayValueScaleComparison:
    """Compare decimal vs binary scale behavior side-by-side."""

    def test_scale_1024_decimal_vs_binary(self):
        """1024 bytes: decimal shows 1.024k, binary shows 1Ki."""
        dec = DisplayValue.si_flex(1024, unit="byte")
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
        value = 2**20  # 1 MiB

        # Decimal: should get mult_exp around 6
        dec = DisplayValue.base_fixed(value, unit="byte")
        # Binary: should get mult_exp=20 or nearby
        bin = DisplayValue(
            value,
            unit_exp=0,
            unit="B",
            format=DisplayFormat.unicode(),
            scale=DisplayScale(type="binary"),
        )

        result_dec = str(dec)
        result_bin = str(bin)

        assert "10⁶" in result_dec or "10^6" in result_dec
        assert "2²⁰" in result_bin or "2^20" in result_bin


# Factory Method Edge Cases ---------------------------------------------------------------


class TestDisplayValueFactoryEdgeCases:
    """Edge cases specific to factory methods."""

    def test_si_fixed_fractional_unit(self):
        """unit_fixed with rate units (e.g., MB/s)."""
        dv = DisplayValue.si_fixed(value=500_000_000, si_unit="Mbyte/s")
        result = str(dv)
        assert result == "500 Mbyte/s"

    def test_si_fixed_parse_prefix_from_unit(self):
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
            dv = DisplayValue.si_fixed(si_value=100, si_unit=si_unit)
            assert dv.unit_exp == expected_exp

    def test_base_fixed_none_value(self):
        """base_fixed with None value."""
        dv = DisplayValue.base_fixed(None, unit="byte")
        result = str(dv)
        assert "None" in result or "N/A" in result

    def test_si_flex_zero(self):
        """unit_flex with zero value."""
        dv = DisplayValue.si_flex(0, unit="byte")
        assert str(dv) == "0 bytes"

    def test_plain_with_non_finite(self):
        """plain factory with non-finite values."""
        test_cases = [
            (float("inf"), "+∞ meters"),
            (float("-inf"), "−∞ meters"),
            (float("nan"), "NaN"),
        ]

        for value, expected in test_cases:
            dv = DisplayValue.plain(value, unit="meter")
            assert str(dv) == expected


# Regression Tests (Based On Existing Test Patterns) ------------------------------------------


class TestDisplayValueRegression:
    """Regression tests to ensure existing behavior is preserved."""

    def test_overflow_predicates_all_modes(self):
        """Overflow predicates work correctly for all modes."""
        extreme_value = 1e100

        # UNIT_FLEX: should overflow
        dv_flex = DisplayValue.si_flex(extreme_value, unit="B")
        assert dv_flex.flow.overflow

        # FIXED: should overflow
        dv_fixed = DisplayValue(extreme_value, unit="B", mult_exp=3, unit_exp=6)
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
                value, unit="byte", mult_exp=0, unit_exp=0, pluralize=pluralize
            )
            assert dv.units == expected


# Performance & Stress Tests (OPTIONAL) ---------------------------------------------------------------


class TestDisplayValueBigAndTiny:
    """Big int and tiny numbers"""

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(10**1000, id="10^1k"),
            pytest.param(-(10**1000), id="10^1k"),
        ],
    )
    def test_big_tiny_decimal(self, value):
        """Accept huge int and format in all possible display modes."""
        dv = DisplayValue(value, unit_exp=3)
        print(dv.mode)
        print(dv.normalized)

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(1 * 2**1024, id="123**1024"),
        ],
    )
    def test_big_tiny_binary(self, value):
        """Accept huge int and format in all possible display modes."""
        dv = DisplayValue(value, mult_exp=10, scale=DisplayScale(type="binary"))
        print("\n", dv)
        print(dv.mode)
        print(dv.normalized)

    #         '123 KiB'


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


class TestDisplayValueMerge:
    """Tests for DisplayValue.merge() functionality."""

    def test_merge_override_value(self):
        """Override value while keeping other attributes unchanged."""
        base = DisplayValue(value=10, unit="byte", mult_exp=2, unit_exp=3)
        merged = base.merge(value=99)
        assert merged.value == 99
        assert merged.unit == "byte"
        assert merged.mult_exp == 2
        assert merged.unit_exp == 3

    def test_merge_override_unit_and_exps(self):
        """Override unit, mult_exp, and unit_exp."""
        base = DisplayValue(value=5, unit="meter", mult_exp=1, unit_exp=2)
        merged = base.merge(unit="second", mult_exp=9, unit_exp=6)
        assert merged.unit == "second"
        assert merged.mult_exp == 9
        assert merged.unit_exp == 6
        assert merged.value == 5

    def test_merge_override_precision_and_trim(self):
        """Override precision and trim_digits."""
        base = DisplayValue(value=3.1415, precision=2, trim_digits=1)
        merged = base.merge(precision=5, trim_digits=3)
        assert merged.precision == 5
        assert merged.trim_digits == 3

    def test_merge_override_format_and_flow(self):
        """Override format and flow with new instances."""
        flow1 = DisplayFlow(mode="e_notation")
        flow2 = DisplayFlow(mode="infinity")
        fmt1 = DisplayFormat.unicode()
        fmt2 = DisplayFormat.ascii()
        base_dv = DisplayValue(value=1.23, flow=flow1, format=fmt1)
        merged_dv = base_dv.merge(flow=flow2, format=fmt2)
        assert merged_dv.flow == flow2.merge(owner=merged_dv)
        assert merged_dv.format == fmt2

    def test_merge_override_scale_and_symbols(self):
        """Override scale and symbols with explicit objects."""
        scale1 = DisplayScale(type="decimal")
        scale2 = DisplayScale(type="binary")
        symbols1 = DisplaySymbols()
        symbols2 = DisplaySymbols()
        base = DisplayValue(value=7, scale=scale1, symbols=symbols1)
        merged = base.merge(scale=scale2, symbols=symbols2)
        assert merged.scale == scale2
        assert merged.symbols == symbols2

    def test_merge_override_boolean_and_plurals(self):
        """Override pluralize, whole_as_int, unit_prefixes, and unit_plurals."""
        base = DisplayValue(
            value=10,
            pluralize=True,
            whole_as_int=True,
            unit_prefixes={3: "k"},
            unit_plurals={"byte": "bytes"},
        )
        merged = base.merge(
            pluralize=False,
            whole_as_int=False,
            unit_prefixes={6: "M"},
            unit_plurals={"second": "seconds"},
        )
        assert merged.pluralize is False
        assert merged.whole_as_int is False
        assert merged.unit_prefixes == {6: "M"}
        assert merged.unit_plurals == {"second": "seconds"}

    def test_merge_with_all_fields_unset_returns_equivalent_copy(self):
        """Return equivalent copy when all parameters are UNSET."""
        base = DisplayValue(
            value=42,
            unit="byte",
            mult_exp=3,
            unit_exp=6,
            pluralize=False,
            precision=5,
            trim_digits=2,
            whole_as_int=True,
        )
        merged = base.merge()
        assert merged == base

    @pytest.mark.parametrize(
        "field,val,key",
        [
            pytest.param("value", "invalid", "value", id="invalid_value"),
            pytest.param("mult_exp", "bad", "mult_exp", id="invalid_mult_exp"),
            pytest.param("unit_exp", "bad", "unit_exp", id="invalid_unit_exp"),
        ],
    )
    def test_merge_invalid_types_raise(self, field, val, key):
        """Raise TypeError when incompatible type is passed."""
        base = DisplayValue(value=1.0)
        kwargs = {field: val}
        with pytest.raises(TypeError, match=rf"(?i){key}"):
            base.merge(**kwargs)


from decimal import Decimal
from fractions import Fraction


class TestDisplayValueToStr:
    """Test DisplayValue.to_str() method with diverse formats and edge cases."""

    @pytest.mark.parametrize(
        "dv,expected",
        [
            pytest.param(DisplayValue(150, unit="byte"), "150 bytes", id="plain_int"),
            pytest.param(
                DisplayValue(1.5, unit="meter"), "1.5 meters", id="plain_float"
            ),
            pytest.param(DisplayValue(None, unit="item"), "None", id="none_value"),
        ],
    )
    def test_basic_values(self, dv, expected):
        """Return correct string for basic numeric and None values."""
        assert dv.to_str() == expected

    @pytest.mark.parametrize(
        "dv,format_str,expected",
        [
            pytest.param(
                DisplayValue(1500, unit="byte"), "{number}", "1.5×10³", id="number_only"
            ),
            pytest.param(
                DisplayValue(1500, unit="byte"), "{units}", "bytes", id="units_only"
            ),
            pytest.param(
                DisplayValue(1500, unit="byte"),
                "{number}_{units}",
                "1.5×10³_bytes",
                id="number_units",
            ),
            pytest.param(
                DisplayValue(1500, unit="byte"),
                "[{units}] {number}",
                "[bytes] 1.5×10³",
                id="custom_layout",
            ),
            pytest.param(
                DisplayValue(1500, unit="byte"), "{normalized}", "1.5", id="normalized"
            ),
            pytest.param(
                DisplayValue(1500, unit="byte"), "{value}", "1500", id="value"
            ),
            pytest.param(
                DisplayValue(1500, unit="byte"), "{separator}", " ", id="separator"
            ),
            pytest.param(
                DisplayValue(1500, unit="byte"), "{unit_prefix}", "", id="unit_prefix"
            ),
            pytest.param(DisplayValue(1500, unit="byte"), "{unit}", "byte", id="unit"),
        ],
    )
    def test_custom_format(self, dv, format_str, expected):
        """
        Format string applies placeholders correctly.

        Tested placeholders:
            - {number} - fully formatted number with multiplier
            - {units} - fully formatted units with prefix
            - {normalized} - normalized value only (no multiplier)
            - {value} - raw input value
            - {separator} - separator symbol
            - {unit_prefix} - SI/IEC prefix only
            - {unit} - base unit name only
        """
        assert dv.to_str(format=format_str) == expected

    def test_overflow_format_override(self):
        """Use custom overflow_format when value is infinity."""
        dv = DisplayValue(float("inf"), unit="byte")
        assert dv.to_str(overflow_format="MAX") == "MAX"
        assert dv.to_str(overflow_format="{symbols.pos_infinity} {units}") == "+∞ bytes"

    def test_underflow_format_override(self):
        """Use custom underflow_format when value is zero."""
        dv = DisplayValue.si_flex(1e-100, unit="byte")
        assert dv.to_str(underflow_format="MIN") == "MIN"

    @pytest.mark.parametrize(
        "value,unit,max_width,expected",
        [
            pytest.param(123456789, "byte", 8, "123456…", id="truncate_width"),
            pytest.param(123.456, "m", 6, "123.4…", id="truncate_float"),
        ],
    )
    def test_max_width_truncation(self, value, unit, max_width, expected):
        """Truncate output to max_width with ellipsis."""
        dv = DisplayValue.plain(value, unit=unit)
        result = dv.to_str(max_width=max_width)
        assert result.startswith(expected[:-1]) and result.endswith("…")

    @pytest.mark.parametrize(
        "dv,format_str,expected",
        [
            pytest.param(
                DisplayValue(1500, unit="byte"),
                "{normalized}",
                "1.5",
                id="normalized_placeholder_int",
            ),
            pytest.param(
                DisplayValue(1.234, unit="m"),
                "{normalized:.2f}",
                "1.23",
                id="normalized_placeholder_float",
            ),
        ],
    )
    def test_normalized_placeholder(self, dv, format_str, expected):
        """Return normalized value correctly with {normalized} placeholder."""
        assert dv.to_str(format=format_str) == expected

    @pytest.mark.parametrize(
        "dv,format_str,expected",
        [
            pytest.param(
                DisplayValue(123_456, unit="byte"),
                "{value}",
                "123456",
                id="raw_value_placeholder",
            ),
            pytest.param(
                DisplayValue(3.14, unit="m"), "{value}", "3.14", id="raw_value_float"
            ),
        ],
    )
    def test_value_placeholder(self, dv, format_str, expected):
        """Return raw input value correctly with {value} placeholder."""
        assert dv.to_str(format=format_str) == expected

    @pytest.mark.parametrize(
        "dv,format_str,expected",
        [
            pytest.param(
                DisplayValue.si_flex(1_500_000, unit="byte"),
                "{unit_prefix}",
                "M",
                id="unit_prefix_si",
            ),
        ],
    )
    def test_unit_prefix_placeholder(self, dv, format_str, expected):
        """Return correct SI/IEC unit prefix with {unit_prefix} placeholder."""
        print("dv       ", dv)
        print("dv_to_str", dv.to_str(format=format_str))
        assert dv.to_str(format=format_str) == expected


# Helper Methods --------------------------------------------------------------------------------


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
            pytest.param(
                1200.0, 15, 2, id="float_nonstandard_treat_trailing_zeros_non_sig"
            ),
            pytest.param(0.00123, 15, 3, id="float_leading_zeros_not_counted"),
        ],
    )
    def test_float_cases(self, number, round_digits, expected):
        """Handle floats with non-standardtrailing zero trimming."""
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
        "number,trim_digits,expected",
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
    def test_rounding_behavior(self, number, trim_digits, expected):
        """Round numbers to given significant digits."""
        result = trimmed_round(number=number, trim_digits=trim_digits)
        assert result == expected

    @pytest.mark.parametrize(
        "number,trim_digits,expected_type",
        [
            pytest.param(3.0, 1, float, id="3.0_to_float_when_no_decimals"),
            pytest.param(123.456, 3, float, id="big_to_float_when_no_decimals"),
            pytest.param(123.456, 5, float, id="float_remains_float_with_decimals"),
            pytest.param(100, 2, int, id="int_stays_int"),
        ],
    )
    def test_result_type(self, number, trim_digits, expected_type):
        """Preserve or coerce return type as per result precision."""
        result = trimmed_round(number=number, trim_digits=trim_digits)
        assert isinstance(result, expected_type)

    @pytest.mark.parametrize(
        "number,trim_digits,expected",
        [
            pytest.param(None, 3, None, id="number_none_passthrough"),
            pytest.param(123.456, None, 123.456, id="digits_none_passthrough_float"),
            pytest.param(100, None, 100, id="digits_none_passthrough_int"),
            pytest.param(float("inf"), 3, float("inf"), id="inf_passthrough"),
            pytest.param(float("-inf"), 4, float("-inf"), id="neg_inf_passthrough"),
            pytest.param(float("nan"), 2, float("nan"), id="nan_passthrough"),
        ],
    )
    def test_passthrough_values(self, number, trim_digits, expected):
        """Return None/NaN/Inf as-is or bypass when digits is None."""
        result = trimmed_round(number=number, trim_digits=trim_digits)
        if isinstance(expected, float) and math.isnan(expected):
            assert isinstance(result, float) and math.isnan(result)
        else:
            assert result == expected

    @pytest.mark.parametrize(
        "number,trim_digits,err,match",
        [
            pytest.param("123", 2, TypeError, r"(?i).*number.*", id="number_str"),
            pytest.param([123], 2, TypeError, r"(?i).*number.*", id="number_list"),
            pytest.param(
                123.456, "2", TypeError, r"(?i).*trim_digits.*", id="digits_str"
            ),
            pytest.param(
                123.456, 1.5, TypeError, r"(?i).*trim_digits.*", id="digits_float"
            ),
        ],
    )
    def test_type_errors(self, number, trim_digits, err, match):
        """Reject invalid argument types."""
        with pytest.raises(err, match=match):
            trimmed_round(number=number, trim_digits=trim_digits)

    @pytest.mark.parametrize(
        "number,trim_digits",
        [
            pytest.param(123.456, 0, id="zero_digits"),
            pytest.param(-10, -1, id="negative_digits"),
        ],
    )
    def test_value_errors_on_digits(self, number, trim_digits):
        """Reject trim_digits less than 1."""
        with pytest.raises(ValueError, match=r"(?i).*trim_digits.*"):
            trimmed_round(number=number, trim_digits=trim_digits)


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
            pytest.param(2**12, 10, id="ge-1Ki-lt-1Mi->exp10"),
            pytest.param(2**21, 20, id="ge-1Mi->exp20"),
        ],
    )
    def test_binary_auto_multiplier_exp(self, value: int, expected: int):
        """Verify binary auto multiplier exponent with 2^(10N)."""
        dv = DisplayValue(value, unit_exp=0, scale=DisplayScale(type="binary"))
        assert dv._mult_exp == expected


# Private Methods Tests ------------------------------------------------------------------------------------------------


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
        """Verify decimal auto unit exponent selection with standardSI prefixes."""
        dv = DisplayValue(value, mult_exp=0, scale=DisplayScale(type="decimal"))
        assert dv._unit_exp == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(512, 0, id="lt-1Ki->base"),
            pytest.param(2**10, 10, id="exactly-1Ki->Ki"),
            pytest.param(2**10 * 500, 10, id="500Ki->Ki"),
            pytest.param(2**20, 20, id="exactly-1Mi->Mi"),
            pytest.param(2**20 * 500, 20, id="500Mi->Mi"),
            pytest.param(2**30, 30, id="exactly-1Gi->Gi"),
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
        dv = DisplayValue(
            value,
            mult_exp=0,
            unit_prefixes=custom_prefixes,
            scale=DisplayScale(type="decimal"),
        )
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
        dv = DisplayValue(
            value,
            mult_exp=0,
            unit_prefixes=custom_prefixes,
            scale=DisplayScale(type="decimal"),
        )
        assert dv._unit_exp == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(512, 0, id="512->base"),
            pytest.param(2**10, 10, id="1Ki->Ki"),
            pytest.param(2**19, 10, id="512Ki->Ki"),
            pytest.param(2**20, 20, id="exact-1Mi"),  # Within scale_step
            pytest.param(2**25, 20, id="32Mi->Mi"),  # Beyond scale_step from Ki
            pytest.param(2**30, 30, id="1Gi->Gi"),
        ],
    )
    def test_binary_gap_in_prefixes(self, value: int, expected: int):
        """Verify behavior with gaps in custom unit_prefixes (binary)."""
        # Custom scale with some prefixes
        custom_prefixes = {0: "", 10: "Ki", 20: "Mi", 30: "Gi"}
        dv = DisplayValue(
            value,
            mult_exp=0,
            unit_prefixes=custom_prefixes,
            scale=DisplayScale(type="binary"),
        )
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
        dv = DisplayValue(
            value,
            mult_exp=0,
            unit_prefixes=custom_prefixes,
            scale=DisplayScale(type="decimal"),
        )
        assert dv._unit_exp == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(0, 0, id="zero->base"),
            pytest.param(0.0, 0, id="zero-float->base"),
            pytest.param(float("nan"), 0, id="nan->base"),
            pytest.param(float("inf"), 0, id="inf->base"),
            pytest.param(float("-inf"), 0, id="neg-inf->base"),
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
        with pytest.raises(
            ValueError, match="unit_exp must be one of SI decimal powers"
        ):
            DisplayValue(123, unit_exp=5, scale=DisplayScale(type="decimal"))
        with pytest.raises(
            ValueError, match="unit_exp must be one of IEC binary powers"
        ):
            DisplayValue(123, unit_exp=5, scale=DisplayScale(type="binary"))
        with pytest.raises(ValueError, match="unit_exp must be one of decimal powers"):
            DisplayValue(
                123,
                mult_exp=0,
                scale=DisplayScale(type="decimal"),
                unit_prefixes={0: "", 5: "penta"},
            )
        # Empty unit_prefixes map should fall back to default mapping
        dv = DisplayValue(
            123, mult_exp=0, scale=DisplayScale(type="decimal"), unit_prefixes={}
        )


class Test_MultiplyPreservingPrecision:
    """Test cases for _multiply_preserving_precision."""

    @pytest.mark.parametrize(
        "float_value,int_multiplier,expected",
        [
            pytest.param(1.5, 2, 3.0, id="simple_float_multiply"),
            pytest.param(2.0, 1, 2.0, id="multiplier_one"),
            pytest.param(0.0, 10, 0.0, id="zero_value"),
            pytest.param(-1.5, 4, -6.0, id="negative_value"),
        ],
    )
    def test_non_overflow_multiplications(
        self, float_value: float, int_multiplier: int, expected: float
    ):
        """Verify basic float and integer multiplications."""
        dv = DisplayValue(1.0)
        result = dv._multiply_preserving_precision(float_value, int_multiplier)
        assert result == pytest.approx(expected)

    def test_large_multiplier_switches_to_int(self):
        """Ensure large multiplier triggers integer arithmetic."""
        dv = DisplayValue(1.0)
        float_val = 1.23
        multiplier = 10**400
        exact = 123 * 10**398
        result = dv._multiply_preserving_precision(float_val, multiplier)
        diff = abs(result - exact) / exact
        assert isinstance(result, int)
        assert str(result).startswith("12")
        assert diff == pytest.approx(0, rel=1e-12)

    def test_small_multiplier_returns_float(self):
        """Ensure small multiplier stays as float."""
        dv = DisplayValue(1.0)
        result = dv._multiply_preserving_precision(1.23, 10)
        assert isinstance(result, float)
        assert math.isclose(result, 12.3, rel_tol=1e-12)

    def test_preserves_precision_near_float_limit(self):
        """Ensure precision is preserved near float max range."""
        dv = DisplayValue(1.0)
        near_limit = (10**307) / 10
        result = dv._multiply_preserving_precision(near_limit, 10)
        assert result == pytest.approx(10**307)
        assert result > 0

    def test_invalid_multiplier_type_raises(self):
        """Ensure invalid multiplier type raises TypeError."""
        dv = DisplayValue(1.0)
        with pytest.raises(TypeError, match=r"(?i).*int_multiplier.*"):
            dv._multiply_preserving_precision(1.23, "100")  # type: ignore[arg-type]

    def test_invalid_float_value_type_raises(self):
        """Ensure invalid float_value type raises TypeError."""
        dv = DisplayValue(1.0)
        with pytest.raises(TypeError, match=r"(?i).*float_value.*"):
            dv._multiply_preserving_precision("1.23", 10)  # type: ignore[arg-type]


class Test_OverflowUnderflowPredicates:
    @pytest.mark.parametrize(
        "value, mult_exp, unit, overflow_tolerance, underflow_tolerance, unit_prefixes, expected_overflow, expected_underflow",
        [
            pytest.param(
                10**-100,
                0,
                "B",
                5,
                6,
                {-24: "y", 24: "Y"},
                False,
                True,
                id="tiny-underflow",
            ),
            pytest.param(
                0.1, 0, "B", 5, 6, {-24: "y", 24: "Y"}, False, False, id="gap-no-flags"
            ),
            pytest.param(
                10**100,
                0,
                "B",
                5,
                6,
                {-24: "y", 24: "Y"},
                True,
                False,
                id="huge-overflow",
            ),
        ],
    )
    def test_overflows_mode_unitflex(
        self,
        value,
        mult_exp,
        unit,
        overflow_tolerance,
        underflow_tolerance,
        unit_prefixes,
        expected_overflow,
        expected_underflow,
    ):
        """Parametrize overflow/underflow flags for extreme magnitudes."""
        dv = DisplayValue(
            value,
            mult_exp=mult_exp,
            unit=unit,
            flow=DisplayFlow(
                overflow_tolerance=overflow_tolerance,
                underflow_tolerance=underflow_tolerance,
            ),
            unit_prefixes=unit_prefixes,
        )
        assert dv.flow.overflow is expected_overflow
        assert dv.flow.underflow is expected_underflow

    @pytest.mark.parametrize(
        "value, unit, overflow_tolerance, underflow_tolerance, expected_overflow, expected_underflow",
        [
            pytest.param(10**-100, "B", 5, 6, False, True, id="tiny-underflow"),
            pytest.param(1000, "B", 5, 6, False, False, id="normal-no-flags"),
            pytest.param(10**100, "B", 5, 6, True, False, id="huge-overflow"),
        ],
    )
    def test_overflows_mode_fixed(
        self,
        value,
        unit,
        overflow_tolerance,
        underflow_tolerance,
        expected_overflow,
        expected_underflow,
    ):
        """Parametrize overflow/underflow flags for extreme magnitudes."""
        dv = DisplayValue(
            value,
            mult_exp=3,
            unit_exp=3,
            unit=unit,
            flow=DisplayFlow(
                overflow_tolerance=overflow_tolerance,
                underflow_tolerance=underflow_tolerance,
            ),
        )
        assert dv.flow.overflow is expected_overflow
        assert dv.flow.underflow is expected_underflow

    @pytest.mark.parametrize(
        "value, unit, overflow_tolerance, underflow_tolerance, expected_overflow, expected_underflow",
        [
            pytest.param(10**-100, "B", 5, 6, False, False, id="tiny"),
            pytest.param(1000, "B", 5, 6, False, False, id="normal"),
            pytest.param(10**100, "B", 5, 6, False, False, id="huge"),
        ],
    )
    def test_no_overflows_mode_plain(
        self,
        value,
        unit,
        overflow_tolerance,
        underflow_tolerance,
        expected_overflow,
        expected_underflow,
    ):
        """Parametrize overflow/underflow flags for extreme magnitudes."""
        dv = DisplayValue(
            value,
            mult_exp=0,
            unit_exp=0,
            unit=unit,
            flow=DisplayFlow(
                overflow_tolerance=overflow_tolerance,
                underflow_tolerance=underflow_tolerance,
            ),
        )
        assert dv.flow.overflow is expected_overflow
        assert dv.flow.underflow is expected_underflow

    @pytest.mark.parametrize(
        "value, unit, overflow_tolerance, underflow_tolerance, expected_overflow, expected_underflow",
        [
            pytest.param(10**-100, "B", 5, 6, False, False, id="tiny"),
            pytest.param(1000, "B", 5, 6, False, False, id="normal"),
            pytest.param(10**100, "B", 5, 6, False, False, id="huge"),
        ],
    )
    def test_no_overflows_mode_base_fixed(
        self,
        value,
        unit,
        overflow_tolerance,
        underflow_tolerance,
        expected_overflow,
        expected_underflow,
    ):
        """Parametrize overflow/underflow flags for extreme magnitudes."""
        dv = DisplayValue(
            value,
            unit_exp=0,
            unit=unit,
            flow=DisplayFlow(
                overflow_tolerance=overflow_tolerance,
                underflow_tolerance=underflow_tolerance,
            ),
        )
        assert dv.flow.overflow is expected_overflow
        assert dv.flow.underflow is expected_underflow

    @pytest.mark.parametrize(
        "value, unit, overflow_tolerance, underflow_tolerance, expected_overflow, expected_underflow",
        [
            pytest.param(10**-100, "B", 5, 6, False, False, id="tiny"),
            pytest.param(1000, "B", 5, 6, False, False, id="normal"),
            pytest.param(10**100, "B", 5, 6, False, False, id="huge"),
        ],
    )
    def test_no_overflows_mode_unit_fixed(
        self,
        value,
        unit,
        overflow_tolerance,
        underflow_tolerance,
        expected_overflow,
        expected_underflow,
    ):
        """Parametrize overflow/underflow flags for extreme magnitudes."""
        dv = DisplayValue(
            value,
            unit_exp=3,
            unit=unit,
            flow=DisplayFlow(
                overflow_tolerance=overflow_tolerance,
                underflow_tolerance=underflow_tolerance,
            ),
        )
        assert dv.flow.overflow is expected_overflow
        assert dv.flow.underflow is expected_underflow
