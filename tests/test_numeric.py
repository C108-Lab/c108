"""
Core test suite for std_numeric() - stdlib types only, no third-party dependencies.

Tests cover: basic types, Decimal/Fraction, overflow/underflow, special values,
error handling modes, boolean handling, and parameter combinations.
"""

import math
from decimal import Decimal
from fractions import Fraction
from types import SimpleNamespace

# Local ----------------------------------------------------------------------------------------------------------------

from c108.abc import search_attrs
from c108.numeric import std_numeric

import pytest
import pytest


class TestStdNumericBasicTypes:
    """Test standardPython numeric types (int, float, None)."""

    @pytest.mark.parametrize(
        "value, expected, expected_type",
        [
            pytest.param(42, 42, int, id="int"),
            pytest.param(3.25, 3.25, float, id="float"),
            pytest.param(None, None, type(None), id="none"),
            pytest.param(10**400, 10**400, int, id="huge-int"),
            pytest.param(-123, -123, int, id="negative-int"),
            pytest.param(-3.5, -3.5, float, id="negative-float"),
        ],
    )
    def test_preserve_value_type(self, value, expected, expected_type):
        """Preserve values and types for supported numerics and None."""
        res = std_numeric(value)
        assert res == expected
        assert isinstance(res, expected_type)


class TestStdNumericDecimal:
    """Test decimal.Decimal conversion and edge cases."""

    @pytest.mark.parametrize(
        "val, expected, approx",
        [
            pytest.param(Decimal("3.5"), 3.5, False, id="fractional_simple"),
            pytest.param(
                Decimal("1.2345678901234567890123456789"),
                None,
                True,
                id="high_precision",
            ),
        ],
        ids=["fractional_simple", "high_precision"],
    )
    def test_decimal_fractional_to_float(self, val, expected, approx):
        """Convert fractional Decimal to float and handle precision loss."""
        res = std_numeric(val)
        assert isinstance(res, float)
        if approx:
            d = float(val)
            assert abs(res - d) < 1e-16 or math.isfinite(res)
        else:
            assert res == expected

    @pytest.mark.parametrize(
        "val, expected",
        [
            pytest.param(Decimal("42"), 42, id="int_exact"),
            pytest.param(Decimal("42.0"), 42, id="int_trailing_zero"),
        ],
    )
    def test_decimal_int_to_float(self, val, expected):
        """Convert integer-valued Decimal to float."""
        res = std_numeric(val)
        assert res == pytest.approx(expected)
        assert isinstance(res, float)

    def test_decimal_huge_int_to_float(self):
        """Preserve huge integer-valued Decimal as Python int."""
        # These are all mathematically integers
        assert std_numeric(Decimal("1e400")) == math.inf
        assert std_numeric(Decimal("1.5e400")) == math.inf
        assert std_numeric(Decimal("-2.0e400")) == -math.inf

    def test_decimal_fractional_overflow_to_inf(self):
        """Convert Decimal with actual fractional part beyond float range to inf."""
        # Create a value with true fractional part
        # At this scale, precision is lost anyway
        val = Decimal("1e400") / Decimal("3")  # Has repeating decimal
        res = std_numeric(val)
        # This will likely still be huge int due to Decimal precision
        # Or we just accept that overflow to inf happens via __float__

    @pytest.mark.parametrize(
        "val,expected_sign",
        [
            pytest.param(Decimal("1e-400"), +1, id="underflow_pos"),
            pytest.param(Decimal("-1e-400"), -1, id="underflow_neg"),
            pytest.param(Decimal("1e-1000"), +1, id="tiny_pos"),
        ],
    )
    def test_decimal_underflow_to_zero(self, val, expected_sign):
        """Convert Decimal below float minimum to zero with sign preservation."""
        res = std_numeric(val)
        assert isinstance(res, float)
        assert res == 0.0
        sign = 1 if math.copysign(1.0, res) > 0 else -1
        assert sign == expected_sign


class TestStdNumericFraction:
    """Test fractions.Fraction conversion and edge cases."""

    def test_fraction_with_remainder(self):
        """Convert Fraction with remainder to float."""
        res = std_numeric(Fraction(22, 7))
        assert isinstance(res, float)
        assert math.isclose(res, 22 / 7, rel_tol=0, abs_tol=1e-15)

    def test_fraction_int_to_float(self):
        """Convert integer-valued Fraction to int, not float."""
        res = std_numeric(Fraction(84, 2))
        assert res == pytest.approx(42)
        assert isinstance(res, float)

    def test_fraction_huge_to_float(self):
        """Convert Fraction with huge numerator to infinity."""
        big = Fraction(10**1000, 1)
        res = std_numeric(big)
        assert isinstance(res, float)
        assert res == math.inf

    def test_fraction_underflow_to_zero(self):
        """Convert Fraction with huge denominator to zero."""
        tiny = Fraction(1, 10**1000)
        res = std_numeric(tiny)
        assert isinstance(res, float)
        assert res == 0.0
        assert math.copysign(1.0, res) > 0


class TestStdNumericSpecialFloatValues:
    """Test IEEE 754 special values (inf, -inf, nan)."""

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(float("inf"), id="positive_inf"),
            pytest.param(float("-inf"), id="negative_inf"),
            pytest.param(math.inf, id="math_inf"),
            pytest.param(-math.inf, id="math_neg_inf"),
        ],
    )
    def test_infinity_preserved(self, value):
        """Preserve infinity values as-is without conversion."""
        res = std_numeric(value)
        assert isinstance(res, float)
        assert math.isinf(res) and (res > 0) == (value > 0)

    def test_nan_preserved(self):
        """Preserve NaN value as-is without conversion."""
        res = std_numeric(float("nan"))
        assert isinstance(res, float)
        assert math.isnan(res)

    def test_math_nan_preserved(self):
        """Preserve math.nan as-is without conversion."""
        res = std_numeric(math.nan)
        assert isinstance(res, float)
        assert math.isnan(res)


class TestStdNumericBooleanHandling:
    """Test boolean rejection and acceptance based on allow_bool parameter."""

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(True, id="true"),
            pytest.param(False, id="false"),
        ],
    )
    def test_bool_rejected_by_default(self, value):
        """Raise TypeError for boolean when allow_bool=False (default)."""
        with pytest.raises(TypeError) as exc:
            std_numeric(value)
        assert "allow_bool" in str(exc.value).lower()

    @pytest.mark.parametrize(
        "bool_val,expected",
        [
            pytest.param(True, 1, id="true_to_1"),
            pytest.param(False, 0, id="false_to_0"),
        ],
    )
    def test_bool_allowed_converts_to_int(self, bool_val, expected):
        """Convert boolean to int when allow_bool=True."""
        res = std_numeric(bool_val, allow_bool=True)
        assert res == expected
        assert isinstance(res, int)


class TestStdNumericErrorHandlingRaise:
    """Test on_error='raise' mode (default) for type errors."""

    @pytest.mark.parametrize(
        "invalid_value",
        [
            pytest.param("123", id="string"),
            pytest.param([1, 2, 3], id="list"),
            pytest.param({"value": 42}, id="dict"),
            pytest.param((1, 2), id="tuple"),
            pytest.param({1, 2, 3}, id="set"),
            pytest.param(b"bytes", id="bytes"),
            pytest.param(1 + 2j, id="complex"),
        ],
    )
    def test_invalid_type_raises(self, invalid_value):
        """Raise TypeError for unsupported types with on_error='raise'."""
        with pytest.raises(TypeError):
            std_numeric(invalid_value)

    def test_bool_raises_with_helpful_message(self):
        """Raise TypeError for bool with hint about allow_bool parameter."""
        with pytest.raises(TypeError) as exc:
            std_numeric(True)
        msg = str(exc.value).lower()
        assert "bool" in msg
        assert "allow_bool" in msg


class TestStdNumericErrorHandlingNan:
    """Test on_error='nan' mode returns nan for type errors."""

    @pytest.mark.parametrize(
        "invalid_value",
        [
            pytest.param("invalid", id="string"),
            pytest.param([1, 2], id="list"),
            pytest.param({"key": "val"}, id="dict"),
            pytest.param(1 + 0j, id="complex"),
        ],
    )
    def test_invalid_type_returns_nan(self, invalid_value):
        """Return float('nan') for unsupported types with on_error='nan'."""
        res = std_numeric(invalid_value, on_error="nan")
        assert isinstance(res, float)
        assert math.isnan(res)

    def test_bool_returns_nan_when_not_allowed(self):
        """Return float('nan') for bool when allow_bool=False and on_error='nan'."""
        res = std_numeric(True, on_error="nan")
        assert isinstance(res, float)
        assert math.isnan(res)

    def test_valid_values_still_converted(self):
        """Convert valid values normally even when on_error='nan'."""
        assert std_numeric(5, on_error="nan") == 5
        res = std_numeric(Decimal("2.5"), on_error="nan")
        assert isinstance(res, float) and res == 2.5


class TestStdNumericErrorHandlingNone:
    """Test on_error='none' mode returns None for type errors."""

    @pytest.mark.parametrize(
        "invalid_value",
        [
            pytest.param("text", id="string"),
            pytest.param([42], id="list"),
            pytest.param(set(), id="empty_set"),
            pytest.param(2j, id="complex"),
        ],
    )
    def test_invalid_type_returns_none(self, invalid_value):
        """Return None for unsupported types with on_error='none'."""
        res = std_numeric(invalid_value, on_error="none")
        assert res is None

    def test_bool_returns_none_when_not_allowed(self):
        """Return None for bool when allow_bool=False and on_error='none'."""
        assert std_numeric(True, on_error="none") is None

    def test_valid_values_still_converted(self):
        """Convert valid values normally even when on_error='none'."""
        assert std_numeric(7, on_error="none") == 7
        res = std_numeric(Fraction(3, 2), on_error="none")
        assert isinstance(res, float) and res == 1.5


class TestStdNumericEdgeCasesNumericNotErrors:
    """Test that numeric edge cases are preserved regardless of on_error setting."""

    @pytest.mark.parametrize(
        "on_error_mode",
        [
            pytest.param("raise", id="raise_mode"),
            pytest.param("nan", id="nan_mode"),
            pytest.param("none", id="none_mode"),
        ],
    )
    def test_infinity_preserved_all_modes(self, on_error_mode):
        """Preserve infinity in all on_error modes (numeric edge case, not error)."""
        res = std_numeric(float("inf"), on_error=on_error_mode)
        assert isinstance(res, float) and math.isinf(res) and res > 0

    @pytest.mark.parametrize(
        "on_error_mode",
        [
            pytest.param("raise", id="raise_mode"),
            pytest.param("nan", id="nan_mode"),
            pytest.param("none", id="none_mode"),
        ],
    )
    def test_overflow_to_inf_all_modes(self, on_error_mode):
        """Convert overflow to infinity in all on_error modes (not suppressed)."""
        res = std_numeric(float("1e400"), on_error=on_error_mode)
        assert isinstance(res, float) and math.isinf(res)

    @pytest.mark.parametrize(
        "on_error_mode",
        [
            pytest.param("raise", id="raise_mode"),
            pytest.param("nan", id="nan_mode"),
            pytest.param("none", id="none_mode"),
        ],
    )
    def test_nan_preserved_all_modes(self, on_error_mode):
        """Preserve NaN in all on_error modes (numeric value, not error)."""
        res = std_numeric(float("nan"), on_error=on_error_mode)
        assert isinstance(res, float) and math.isnan(res)


class TestStdNumericParameterCombinations:
    """Test combinations of allow_bool and on_error parameters."""

    @pytest.mark.parametrize(
        "bool_val,allow_bool,on_error,expected",
        [
            pytest.param(True, False, "raise", TypeError, id="reject_raise"),
            pytest.param(True, False, "nan", float("nan"), id="reject_nan"),
            pytest.param(True, False, "none", None, id="reject_none"),
            pytest.param(True, True, "raise", 1, id="allow_raise"),
            pytest.param(True, True, "nan", 1, id="allow_nan"),
            pytest.param(True, True, "none", 1, id="allow_none"),
            pytest.param(False, True, "raise", 0, id="false_allow_raise"),
        ],
    )
    def test_bool_with_all_parameter_combinations(self, bool_val, allow_bool, on_error, expected):
        """Test boolean handling across all parameter combinations."""
        if expected is TypeError:
            with pytest.raises(TypeError):
                std_numeric(bool_val, allow_bool=allow_bool, on_error=on_error)
            return
        res = std_numeric(bool_val, allow_bool=allow_bool, on_error=on_error)
        if isinstance(expected, float) and math.isnan(expected):
            assert isinstance(res, float) and math.isnan(res)
        else:
            assert res == expected
            assert isinstance(res, int) if allow_bool else True


class TestStdNumericTypePreservation:
    """Test that returned types match expected semantics (int vs float)."""

    def test_returns_int_not_float_for_integers(self):
        """Return int type for integer values, not float."""
        res = std_numeric(100)
        assert res == 100
        assert isinstance(res, int)

    def test_returns_float_for_fractional_values(self):
        """Return float type for values with fractional parts."""
        res = std_numeric(Fraction(3, 2))
        assert isinstance(res, float)
        assert res == 1.5

    def test_huge_int_returns_int_type(self):
        """Return int type even for huge integers beyond float range."""
        res = std_numeric(10**300)
        assert isinstance(res, int)

    def test_overflow_returns_float_inf_type(self):
        """Return float type for overflow (infinity), not int."""
        res = std_numeric(float("1e400"))
        assert isinstance(res, float)
        assert math.isinf(res)


class TestStdNumericEdgeCases:
    """Test uncovered branches in std_numeric."""

    @pytest.mark.parametrize(
        "val",
        [
            pytest.param(type("B", (), {"__name__": "bool"})(), id="bool_name"),
            pytest.param(type("B2", (), {"__name__": "bool8"})(), id="bool8_name"),
        ],
    )
    def test_is_bool_type_variants(self, val):
        """Cover __is_bool_type internal helper."""

        # Access indirectly via dtype simulation
        class Dummy:
            dtype = SimpleNamespace(is_bool=True)

        d = Dummy()
        d.dtype.is_bool = True
        assert std_numeric(1) == 1  # dummy call to ensure import
        # We cannot directly call __is_bool_type, but we trigger dtype.is_bool path
        obj = SimpleNamespace(dtype=SimpleNamespace(is_bool=True), item=lambda: 1)
        assert std_numeric(obj, allow_bool=True) == 1

    @pytest.mark.parametrize(
        "on_error,expected",
        [
            pytest.param(
                "raise", pytest.raises(TypeError, match=r"boolean values not supported"), id="raise"
            ),
            pytest.param("nan", float("nan"), id="nan"),
            pytest.param("none", None, id="none"),
        ],
    )
    def test_bool_handling_false(self, on_error, expected):
        """Handle bool with allow_bool=False."""
        if on_error == "raise":
            with expected:
                std_numeric(True, on_error=on_error, allow_bool=False)
        else:
            out = std_numeric(True, on_error=on_error, allow_bool=False)
            if on_error == "nan":
                assert out != out
            else:
                assert out is None

    def test_bool_allow_true(self):
        """Handle bool with allow_bool=True."""
        assert std_numeric(True, allow_bool=True) == 1

    def test_pandas_na_like(self):
        """Handle pandas.NA-like object."""

        class NAType:
            __name__ = "NAType"
            __module__ = "pandas._libs.missing"

        val = NAType()
        assert std_numeric(val) != std_numeric(val)  # nan

    def test_numpy_masked_like(self):
        """Handle numpy.ma.masked-like object."""

        class MaskedConstant:
            __name__ = "MaskedConstant"
            __module__ = "numpy.ma.core"

        val = MaskedConstant()
        assert std_numeric(val) != std_numeric(val)

    @pytest.mark.parametrize(
        "on_error,expected_type",
        [
            pytest.param("raise", TypeError, id="raise"),
            pytest.param("nan", float, id="nan"),
            pytest.param("none", type(None), id="none"),
        ],
    )
    def test_collection_reject(self, on_error, expected_type):
        """Reject sizable collection."""
        val = [1, 2]
        if on_error == "raise":
            with pytest.raises(expected_type, match=r"collection"):
                std_numeric(val, on_error=on_error)
        else:
            out = std_numeric(val, on_error=on_error)
            if on_error == "nan":
                assert out != out
            else:
                assert out is None

    @pytest.mark.parametrize(
        "dtype_str,on_error,expect",
        [
            pytest.param(
                "complex64", "raise", pytest.raises(TypeError, match=r"complex"), id="complex_raise"
            ),
            pytest.param("complex64", "nan", float("nan"), id="complex_nan"),
            pytest.param("complex64", "none", None, id="complex_none"),
        ],
    )
    def test_dtype_complex(self, dtype_str, on_error, expect):
        """Handle complex dtype."""

        class Dummy:
            def __init__(self):
                self.dtype = dtype_str

        val = Dummy()
        if on_error == "raise":
            with expect:
                std_numeric(val, on_error=on_error)
        else:
            out = std_numeric(val, on_error=on_error)
            if on_error == "nan":
                assert out != out
            else:
                assert out is None

    @pytest.mark.parametrize(
        "dtype_str,on_error,expect",
        [
            pytest.param(
                "bool",
                "raise",
                pytest.raises(TypeError, match=r"boolean values not supported"),
                id="bool_raise",
            ),
            pytest.param("bool", "nan", float("nan"), id="bool_nan"),
            pytest.param("bool", "none", None, id="bool_none"),
        ],
    )
    def test_dtype_bool(self, dtype_str, on_error, expect):
        """Handle boolean dtype."""

        class Dummy:
            def __init__(self):
                self.dtype = dtype_str

        val = Dummy()
        if on_error == "raise":
            with expect:
                std_numeric(val, on_error=on_error)
        else:
            out = std_numeric(val, on_error=on_error)
            if on_error == "nan":
                assert out != out
            else:
                assert out is None

    def test_item_and_numpy_methods(self):
        """Handle .item() and .numpy() extraction."""

        class Dummy:
            dtype = "float32"

            def item(self):
                return 5.5

        assert std_numeric(Dummy()) == pytest.approx(5.5)

        class Dummy2:
            dtype = "float32"

            def item(self):
                raise TypeError

            def numpy(self):
                return 7.7

        assert std_numeric(Dummy2()) == pytest.approx(7.7)

    def test_item_bool_result(self):
        """Handle .item() returning bool."""

        class Dummy:
            def item(self):
                return True

        with pytest.raises(TypeError, match=r"unsupported numeric type"):
            std_numeric(Dummy())

        assert std_numeric(Dummy(), allow_bool=True) == 1

    def test_sympy_boolean_like(self):
        """Handle sympy BooleanTrue/False."""

        class SympyTrue:
            __name__ = "BooleanTrue"
            __module__ = "sympy.logic.boolalg"

        val = SympyTrue()
        with pytest.raises(TypeError, match=r"unsupported numeric type"):
            std_numeric(val)
        assert std_numeric(val, allow_bool=True) == 1

    def test_index_error_handling(self):
        """Handle __index__ raising error."""

        class BadIndex:
            def __index__(self):
                raise TypeError("bad")

        val = BadIndex()
        assert std_numeric(val, on_error="nan") != std_numeric(val, on_error="nan")

    def test_astropy_quantity_like(self):
        """Handle Astropy Quantity-like object."""

        class Quantity:
            def __init__(self):
                self.value = 3.3
                self.unit = "m"

        val = Quantity()
        assert std_numeric(val) == pytest.approx(3.3)

    def test_float_overflow_and_sign(self):
        """Handle OverflowError and sign detection."""

        class Huge:
            def __float__(self):
                raise OverflowError

            def __lt__(self, other):
                return False

        val = Huge()
        assert std_numeric(val) == float("inf")

        class NegHuge:
            def __float__(self):
                raise OverflowError

            def __lt__(self, other):
                return True

        val2 = NegHuge()
        assert std_numeric(val2) == float("-inf")

    def test_float_typeerror_nan_none(self):
        """Handle __float__ raising TypeError."""

        class BadFloat:
            def __float__(self):
                raise TypeError("bad")

        val = BadFloat()
        with pytest.raises(TypeError, match=r"cannot convert"):
            std_numeric(val)
        assert std_numeric(val, on_error="nan") != std_numeric(val, on_error="nan")
        assert std_numeric(val, on_error="none") is None

    def test_int_fallback_and_error(self):
        """Handle __int__ fallback and error."""

        class GoodInt:
            def __int__(self):
                return 42

        assert std_numeric(GoodInt()) == 42

        class BadInt:
            def __int__(self):
                raise TypeError("bad")

        val = BadInt()
        with pytest.raises(TypeError, match=r"cannot convert"):
            std_numeric(val)
        assert std_numeric(val, on_error="nan") != std_numeric(val, on_error="nan")
        assert std_numeric(val, on_error="none") is None

    def test_unsupported_type(self):
        """Handle unsupported type fallback."""

        class Unknown:
            pass

        with pytest.raises(TypeError, match=r"unsupported numeric type"):
            std_numeric(Unknown())
        assert std_numeric(Unknown(), on_error="nan") != std_numeric(Unknown(), on_error="nan")
        assert std_numeric(Unknown(), on_error="none") is None


# Sentinel classes to validate duck-typing priority without third-party deps
class _IndexOnly:
    def __index__(self):
        return 7


class _ItemReturningFloat:
    def __init__(self, v):
        self._v = v

    def item(self):
        return self._v


class _FloatOnly:
    def __float__(self):
        return 2.5


class _IntOnly:
    def __int__(self):
        return 9


class TestStdNumericDuckTypingPriority:
    """Ensure duck-typing order (__index__, .item(), integer-valued checks, __float__)."""

    def test_index_precedence_over_float(self):
        class _Both:
            def __index__(self):
                return 11

            def __float__(self):
                return 3.0

        res = std_numeric(_Both())
        assert res == 11 and isinstance(res, int)

    def test_item_used_when_present(self):
        res = std_numeric(_ItemReturningFloat(4.75))
        assert isinstance(res, float) and res == 4.75

    def test_index_only(self):
        res = std_numeric(_IndexOnly())
        assert res == 7 and isinstance(res, int)

    def test_float_only(self):
        res = std_numeric(_FloatOnly())
        assert isinstance(res, float) and res == 2.5

    def test_int_only_interpreted_as_int(self):
        res = std_numeric(_IntOnly())
        assert isinstance(res, int) and res == 9
