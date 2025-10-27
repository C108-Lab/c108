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

def _pred_true(_dv) -> bool:
    return True


def _pred_false(_dv) -> bool:
    return False


class TestDisplayFlow:
    def test_invalid_predicate_types(self) -> None:
        """Validate predicate type errors."""
        invalid_obj = object()

        with pytest.raises(ValueError, match=r"(?i).*overflow_predicate must be callable.*"):
            DisplayFlow(mode="e_notation",
                        overflow_predicate=invalid_obj,  # not callable
                        underflow_predicate=_pred_false,
                        overflow_tolerance=3, underflow_tolerance=2, )

        with pytest.raises(ValueError, match=r"(?i).*underflow_predicate must be callable.*"):
            DisplayFlow(mode="infinity",
                        overflow_predicate=_pred_true, underflow_predicate=invalid_obj,  # not callable
                        overflow_tolerance=4, underflow_tolerance=1, )

    def test_invalid_mode_value(self) -> None:
        """Validate mode enum."""
        with pytest.raises(ValueError, match=r"(?i).*mode must be 'e_notation' or 'infinity'.*"):
            DisplayFlow(mode="bad_mode",
                        overflow_predicate=_pred_true, underflow_predicate=_pred_false,
                        overflow_tolerance=5, underflow_tolerance=1, )

    @pytest.mark.parametrize(
        ("overflow_tolerance", "underflow_tolerance", "expect_exc", "match"),
        [
            pytest.param("3", 1, TypeError,
                         r"(?i).*overflow_tolerance must be int \| None.*",
                         id="overflow_tolerance_type_error",
                         ),
            pytest.param(2, {}, TypeError,
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
            DisplayFlow(mode="e_notation",
                        overflow_predicate=_pred_true, underflow_predicate=_pred_false,
                        overflow_tolerance=overflow_tolerance, underflow_tolerance=underflow_tolerance, )

    def test_merge_owner_type(self) -> None:
        """Validate owner type in merge."""
        flow = DisplayFlow(mode="e_notation",
                           overflow_predicate=_pred_true, underflow_predicate=_pred_false,
                           overflow_tolerance=7, underflow_tolerance=3, )
        with pytest.raises(TypeError, match=r"(?i).*owner must be DisplayValue.*"):
            flow.merge(owner=object())

    def test_merge_unset_owner(self) -> None:
        """Merge and unset owner explicitly."""
        flow = DisplayFlow(mode="infinity",
                           overflow_predicate=_pred_true, underflow_predicate=_pred_true,
                           overflow_tolerance=9, underflow_tolerance=4, )
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

        base = DisplayFlow(mode="e_notation",
                           overflow_predicate=p_old_over, underflow_predicate=p_old_under,
                           overflow_tolerance=6, underflow_tolerance=2, )
        merged = base.merge(mode="infinity",
                            overflow_predicate=p_new_over, underflow_predicate=p_new_under,
                            overflow_tolerance=10, underflow_tolerance=5,
                            owner=None, )

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
    def test_mult_exp_formatting(self, mult: str, base: int, power: int, expected: str) -> None:
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
            pytest.param("10", 3, TypeError, r"(?i).*base must be an int.*", id="nonint_base"),
            pytest.param(10, "3", TypeError, r"(?i).*power must be an int.*", id="nonint_power"),
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
        with pytest.raises(ValueError, match=r"(?i).*symbols preset expected one of.*but found.*"):
            DisplayFormat(symbols="bad")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "factory,exp_mult,exp_symbols",
        [
            pytest.param(DisplayFormat.ascii, "caret", "ascii", id="ascii_preset"),
            pytest.param(DisplayFormat.unicode, "unicode", "unicode", id="unicode_preset"),
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
    def test_merge_override_mult(self, initial: str, override: str, expected: str) -> None:
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
    def test_merge_override_symbols(self, initial_symbols: str, override: str, expected: str) -> None:
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
            pytest.param("caret", "ascii", "python", "unicode", "python", "unicode", id="override_both"),
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
    def test_decimal_exp(self, val: float, expected: int) -> None:
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
    def test_binary_exp(self, val: float, expected: int) -> None:
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
        with pytest.raises(TypeError, match=r"(?i).*value must be int \| float.*"):
            scale.value_exponent("oops")  # type: ignore[arg-type]

    def test_bad_scale_type(self) -> None:
        """Reject invalid scale type at init."""
        with pytest.raises(ValueError, match=r"(?i).*scale type 'binary' or 'decimal' literal expected.*"):
            DisplayScale(type="hex")  # type: ignore[arg-type]

    def test_base_not_int(self) -> None:
        """Reject non-int base at runtime."""
        scale = DisplayScale(type="decimal")
        object.__setattr__(scale, "base", "10")
        with pytest.raises(ValueError, match=r"(?i).*int scale base required.*"):
            scale.value_exponent(1)

    def test_base_not_supported(self) -> None:
        """Reject unsupported base values."""
        scale = DisplayScale(type="binary")
        object.__setattr__(scale, "base", 3)
        with pytest.raises(ValueError, match=r"(?i).*scale base must binary or decimal.*"):
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


class TestDisplayValueExtendedValueValidation:
    def test_reject_bool(self):
        """Reject boolean values explicitly with TypeError."""
        with pytest.raises(TypeError, match=r"(?i)bool|boolean"):
            _ = DisplayValue(True)

    def test_decimal_and_fraction(self):
        """Accept Decimal and Fraction by converting to std numeric."""
        dv1 = DisplayValue(Decimal("3.5"))
        dv2 = DisplayValue(Fraction(1, 4))
        assert dv1.is_finite and dv2.is_finite
        assert dv1.normalized == pytest.approx(3.5)
        assert dv2.normalized == pytest.approx(250)

    def test_numpy_scalar(self):
        """Accept NumPy scalar by converting to std numeric."""
        np = pytest.importorskip("numpy")
        dv_i = DisplayValue(np.int64(42))
        dv_f = DisplayValue(np.float64(1.25))
        assert str(dv_i) == "42"
        assert dv_f.normalized == pytest.approx(1.25)

    def test_pandas_na(self):
        """Treat pandas NA as None for display."""
        pd = pytest.importorskip("pandas")
        dv = DisplayValue(pd.NA, unit="item")
        assert not dv.is_finite
        assert str(dv) == "NaN"

    def test_torch_tensor_scalar(self):
        """Accept PyTorch tensor scalar via .item()."""
        torch = pytest.importorskip("torch")
        dv = DisplayValue(torch.tensor(3.5))
        assert dv.is_finite
        assert dv.normalized == pytest.approx(3.5)

    # def test_astropy_quantity(self):
    #     """Accept Astropy Quantity by extracting .value and discarding units."""
    #     u = pytest.importorskip("astropy.units")
    #     dv = DisplayValue(5 * u.m)
    #     assert str(dv) == "5"

    @pytest.mark.parametrize(
        "scale_type",
        [
            pytest.param("ternary", id="ternary"),
            pytest.param("octal", id="octal"),
            pytest.param("weird", id="weird"),
        ],
    )
    def test_invalid_scale_type_rejection(self, scale_type):
        """Reject invalid scale type with ValueError."""
        with pytest.raises(ValueError, match=r"(?i)scale|type"):
            _ = DisplayScale(type=scale_type)

    @pytest.mark.parametrize(
        "scale_type, unit_exp",
        [
            pytest.param("decimal", 7, id="decimal-7"),
            pytest.param("binary", 7, id="binary-7"),
        ],
    )
    def test_invalid_unit_exp_rejection(self, scale_type, unit_exp):
        """Reject non-standard IEC/SI exponents for unit_exp."""
        with pytest.raises(ValueError, match=r"(?i)unit[_ ]?exp|exponent"):
            _ = DisplayValue(1, unit="B", unit_exp=unit_exp, scale=DisplayScale(type=scale_type))

    def test_negative_precision_rejection(self):
        """Reject negative precision with ValueError."""
        with pytest.raises(ValueError, match=r"(?i)precision"):
            _ = DisplayValue(1.23, precision=-1)

    def test_invalid_format_type_rejection(self):
        """Reject invalid format type with TypeError."""
        with pytest.raises(TypeError, match=r"(?i)format"):
            _ = DisplayValue(1, format="plain")  # type: ignore[arg-type]

    def test_invalid_mult_exp_type_rejection(self):
        """Reject non-int mult_exp with TypeError."""
        with pytest.raises(TypeError, match=r"(?i)mult[_ ]?exp"):
            _ = DisplayValue(1, mult_exp="3")  # type: ignore[arg-type]

    def test_frozen_dataclass_immutability(self):
        """Ensure dataclass is frozen (immutable)."""
        dv = DisplayValue(1)
        with pytest.raises(FrozenInstanceError):
            dv.value = 2  # type: ignore[misc, assignment]


class TestDisplayValueProperties:
    @pytest.mark.parametrize(
        "value, expected",
        [
            pytest.param(None, False, id="none"),
            pytest.param(math.inf, False, id="pos-inf"),
            pytest.param(-math.inf, False, id="neg-inf"),
            pytest.param(math.nan, False, id="nan"),
            pytest.param(0, True, id="zero"),
            pytest.param(1.5, True, id="float"),
        ],
    )
    def test_is_finite(self, value, expected):
        """Evaluate is_finite across non-finite and finite values."""
        dv = DisplayValue(value)
        assert dv.is_finite is expected

    @pytest.mark.parametrize(
        "scale_type, mult_exp, expected",
        [
            pytest.param("decimal", 3, 1000, id="dec-10^3"),
            pytest.param("decimal", 0, 1, id="dec-10^0"),
            pytest.param("binary", 10, 1024, id="bin-2^10"),
            pytest.param("binary", 0, 1, id="bin-2^0"),
        ],
    )
    def test_mult_value(self, scale_type, mult_exp, expected):
        """Compute multiplier numeric value across scales."""
        dv = DisplayValue(1, mult_exp=mult_exp, scale=DisplayScale(type=scale_type))
        assert dv.mult_value == expected

    @pytest.mark.parametrize(
        "scale_type, unit_exp, expected",
        [
            pytest.param("decimal", 6, 1_000_000, id="dec-10^6"),
            pytest.param("decimal", 0, 1, id="dec-10^0"),
            pytest.param("binary", 20, 1 << 20, id="bin-2^20"),
            pytest.param("binary", 0, 1, id="bin-2^0"),
        ],
    )
    def test_unit_value(self, scale_type, unit_exp, expected):
        """Compute unit prefix numeric value across scales."""
        dv = DisplayValue(1, unit="B", unit_exp=unit_exp, scale=DisplayScale(type=scale_type))
        assert dv.unit_value == expected

    def test_ref_value(self):
        """Calculate ref_value as mult_value × unit_value."""
        dv = DisplayValue(
            1,
            unit="B",
            mult_exp=3,
            unit_exp=6,
            scale=DisplayScale(type="decimal"),
        )
        assert dv.ref_value == 10 ** 9

    def test_unit_prefix_from_mapping(self):
        """Extract unit_prefix from custom mapping."""
        mapping = {3: "k", 6: "M"}
        dv = DisplayValue(1, unit="byte", unit_exp=3, unit_prefixes=mapping, scale=DisplayScale(type="decimal"))
        assert dv.unit_prefix == "k"

    @pytest.mark.parametrize(
        "value, pluralize, expected",
        [
            pytest.param(1, True, "byte", id="singular"),
            pytest.param(2, True, "bytes", id="plural"),
            pytest.param(1, False, "byte", id="no-pluralize-1"),
            pytest.param(2, False, "byte", id="no-pluralize-2"),
        ],
    )
    def test_units_pluralization(self, value, pluralize, expected):
        """Pluralize units properly for edge cases."""
        dv = DisplayValue(value, unit="byte", pluralize=pluralize)
        assert dv.units == expected

    def test_units_prefix_without_unit(self):
        """Expose unit prefix when unit is None."""
        dv = DisplayValue(1230, mult_exp=0, unit=None, unit_exp=3, scale=DisplayScale(type="decimal"))
        assert dv.units == "k"

    def test_number_with_and_without_multiplier(self):
        """Render number with/without multiplier part."""
        dv_no_mult = DisplayValue(123, unit="m", mult_exp=0, unit_exp=0, scale=DisplayScale(type="decimal"))
        dv_with_mult = DisplayValue(123, unit="m", mult_exp=3, unit_exp=0, scale=DisplayScale(type="decimal"))
        assert dv_no_mult.number == "123"
        assert dv_with_mult.number.endswith("×10³")

    def test_parts_tuple(self):
        """Return parts tuple as (number, units)."""
        dv = DisplayValue(123, unit="B", mult_exp=3, unit_exp=6, scale=DisplayScale(type="decimal"))
        assert dv.parts == (dv.number, dv.units)


class TestDVFormattingPipeline:
    def test_precision_precedence_over_trim_digits(self):
        """Apply precision when specified, ignoring trim_digits."""
        dv = DisplayValue(1 / 3, unit="s", precision=2, trim_digits=10)
        assert str(dv) == "333.33×10⁻³ s"

    def test_whole_as_int_conversion(self):
        """Convert whole float to int representation when enabled."""
        dv = DisplayValue(3.0, unit="s", whole_as_int=True)
        assert dv.number == "3"

    def test_trim_digits_bypass_when_precision_set(self):
        """Bypass trim_digits auto-calculation when precision is set."""
        dv = DisplayValue(1 / 3, unit="s", precision=4, trim_digits=1)
        assert str(dv) == "300.0000×10⁻³ s"

    @pytest.mark.parametrize(
        "value, expected_contains",
        [
            pytest.param(1e-100, "e-", id="underflow"),
            pytest.param(1e100, "e+", id="overflow"),
        ],
    )
    def test_overflow_underflow_e_notation_mode(self, value, expected_contains):
        """Format extreme magnitudes using scientific notation in e_notation mode."""
        dv = DisplayValue(value, unit="B", mult_exp=0, flow=DisplayFlow(mode="e_notation"))
        s = str(dv)
        print("\n", s)
        assert expected_contains in s.lower()
        assert "B" in s


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
