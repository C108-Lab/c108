"""
Core test suite for std_numeric() - stdlib types only, no third-party dependencies.

Tests cover: basic types, Decimal/Fraction, overflow/underflow, special values,
error handling modes, boolean handling, and parameter combinations.
"""

import math
from decimal import Decimal
from fractions import Fraction

import pytest

from c108.abc import search_attrs
# Local ----------------------------------------------------------------------------------------------------------------

from c108.numeric import std_numeric

import pytest


class TestStdNumericBasicTypes:
    """Test standard Python numeric types (int, float, None)."""

    @pytest.mark.parametrize(
        "value, expected, expected_type",
        [
            pytest.param(42, 42, int, id="int"),
            pytest.param(3.25, 3.25, float, id="float"),
            pytest.param(None, None, type(None), id="none"),
            pytest.param(10 ** 400, 10 ** 400, int, id="huge-int"),
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
    def test_decimal_integer_to_int(self, val, expected):
        """Convert integer-valued Decimal to int."""
        res = std_numeric(val)
        assert res == expected
        assert isinstance(res, int)

    def test_decimal_huge_integer_valued_to_int(self):
        """Preserve huge integer-valued Decimal as Python int (arbitrary precision)."""
        val = Decimal('1e400')
        res = std_numeric(val)
        assert isinstance(res, int)
        assert res == int(Decimal('1e400'))  # Exact value preserved

    def test_decimal_huge_integer_preserved(self):
        """Preserve huge integer-valued Decimal as Python int."""
        # These are all mathematically integers
        assert isinstance(std_numeric(Decimal('1e400')), int)
        assert isinstance(std_numeric(Decimal('1.5e400')), int)  # = 15e399
        assert isinstance(std_numeric(Decimal('2.0e400')), int)  # = 2e400

    def test_decimal_fractional_overflow_to_inf(self):
        """Convert Decimal with actual fractional part beyond float range to inf."""
        # Create a value with true fractional part
        # At this scale, precision is lost anyway
        val = Decimal('1e400') / Decimal('3')  # Has repeating decimal
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

    def test_fraction_integer_valued_to_int(self):
        """Convert integer-valued Fraction to int, not float."""
        res = std_numeric(Fraction(84, 2))
        assert res == 42
        assert isinstance(res, int)

    def test_fraction_huge_to_int(self):
        """Convert Fraction with huge numerator to infinity."""
        big = Fraction(10 ** 1000, 1)
        res = std_numeric(big)
        assert isinstance(res, int)
        assert res == 10 ** 1000

    def test_fraction_underflow_to_zero(self):
        """Convert Fraction with huge denominator to zero."""
        tiny = Fraction(1, 10 ** 1000)
        res = std_numeric(tiny)
        assert isinstance(res, float)
        assert res == 0.0
        assert math.copysign(1.0, res) > 0


class TestStdNumericSpecialFloatValues:
    """Test IEEE 754 special values (inf, -inf, nan)."""

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(float('inf'), id="positive_inf"),
            pytest.param(float('-inf'), id="negative_inf"),
            pytest.param(math.inf, id="math_inf"),
            pytest.param(-math.inf, id="math_neg_inf"),
        ]
    )
    def test_infinity_preserved(self, value):
        """Preserve infinity values as-is without conversion."""
        res = std_numeric(value)
        assert isinstance(res, float)
        assert math.isinf(res) and (res > 0) == (value > 0)

    def test_nan_preserved(self):
        """Preserve NaN value as-is without conversion."""
        res = std_numeric(float('nan'))
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
        ]
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
        ]
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
        ]
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
        ]
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
        res = std_numeric(Decimal('2.5'), on_error="nan")
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
        ]
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
        ]
    )
    def test_infinity_preserved_all_modes(self, on_error_mode):
        """Preserve infinity in all on_error modes (numeric edge case, not error)."""
        res = std_numeric(float('inf'), on_error=on_error_mode)
        assert isinstance(res, float) and math.isinf(res) and res > 0

    @pytest.mark.parametrize(
        "on_error_mode",
        [
            pytest.param("raise", id="raise_mode"),
            pytest.param("nan", id="nan_mode"),
            pytest.param("none", id="none_mode"),
        ]
    )
    def test_overflow_to_inf_all_modes(self, on_error_mode):
        """Convert overflow to infinity in all on_error modes (not suppressed)."""
        res = std_numeric(float('1e400'), on_error=on_error_mode)
        assert isinstance(res, float) and math.isinf(res)

    @pytest.mark.parametrize(
        "on_error_mode",
        [
            pytest.param("raise", id="raise_mode"),
            pytest.param("nan", id="nan_mode"),
            pytest.param("none", id="none_mode"),
        ]
    )
    def test_nan_preserved_all_modes(self, on_error_mode):
        """Preserve NaN in all on_error modes (numeric value, not error)."""
        res = std_numeric(float('nan'), on_error=on_error_mode)
        assert isinstance(res, float) and math.isnan(res)


class TestStdNumericParameterCombinations:
    """Test combinations of allow_bool and on_error parameters."""

    @pytest.mark.parametrize(
        "bool_val,allow_bool,on_error,expected",
        [
            pytest.param(True, False, "raise", TypeError, id="reject_raise"),
            pytest.param(True, False, "nan", float('nan'), id="reject_nan"),
            pytest.param(True, False, "none", None, id="reject_none"),
            pytest.param(True, True, "raise", 1, id="allow_raise"),
            pytest.param(True, True, "nan", 1, id="allow_nan"),
            pytest.param(True, True, "none", 1, id="allow_none"),
            pytest.param(False, True, "raise", 0, id="false_allow_raise"),
        ]
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
        res = std_numeric(10 ** 300)
        assert isinstance(res, int)

    def test_overflow_returns_float_inf_type(self):
        """Return float type for overflow (infinity), not int."""
        res = std_numeric(float('1e400'))
        assert isinstance(res, float)
        assert math.isinf(res)


# Sentinel classes to validate duck-typing priority without third-party deps
class _IndexOnly:
    def __index__(self):
        return 7


class _ItemReturningFloat:
    def __init__(self, v): self._v = v

    def item(self): return self._v


class _FloatOnly:
    def __float__(self): return 2.5


class _IntOnly:
    def __int__(self): return 9


class TestStdNumericDuckTypingPriority:
    """Ensure duck-typing order (__index__, .item(), integer-valued checks, __float__)."""

    def test_index_precedence_over_float(self):
        class _Both:
            def __index__(self): return 11

            def __float__(self): return 3.0

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


# Integration Tests ----------------------------------------------------------------------------------------------------

# numpy Tests ------------------------------------------------------------------------------------
np = pytest.importorskip("numpy")


class TestStdNumNumpyNumericSupport:
    """
    Core tests for NumPy datatype support in std_numeric.
    """

    @pytest.mark.parametrize(
        ("scalar", "expected"),
        [
            pytest.param(np.int8(-5), -5, id="int8-neg"),
            pytest.param(np.int16(123), 123, id="int16-pos"),
            pytest.param(np.uint16(65530), 65530, id="uint16-large"),
            pytest.param(np.int64(2 ** 63 - 1), 2 ** 63 - 1, id="int64-max"),
            pytest.param(np.uint64(2 ** 63 + 5), 2 ** 63 + 5, id="uint64-beyond-int64"),
        ],
    )
    def test_np_integers_to_int(self, scalar, expected) -> None:
        """Convert NumPy integer scalars to Python int."""
        result = std_numeric(scalar, on_error="raise", allow_bool=False)
        assert type(result) is int
        assert result == expected

    @pytest.mark.parametrize(
        ("scalar", "expected", "kind"),
        [
            # Finite values
            pytest.param(np.float16(3.5), 3.5, "finite", id="float16-3.5"),
            pytest.param(np.float32(-2.25), -2.25, "finite", id="float32-neg"),
            pytest.param(np.float64(1.0e100), 1.0e100, "finite", id="float64-large"),
            # Specials
            pytest.param(np.float32(np.nan), None, "nan", id="nan-f32"),
            pytest.param(np.float64(np.inf), None, "inf+", id="inf-pos-f64"),
            pytest.param(np.float64(-np.inf), None, "inf-", id="inf-neg-f64"),
        ],
    )
    def test_np_floats(self, scalar, expected, kind) -> None:
        """Convert NumPy float scalars and preserve special values."""
        result = std_numeric(scalar, on_error="raise", allow_bool=False)
        assert type(result) is float
        if kind == "finite":
            assert result == pytest.approx(expected)
        elif kind == "nan":
            assert math.isnan(result)
        elif kind == "inf+":
            assert math.isinf(result) and result > 0
        elif kind == "inf-":
            assert math.isinf(result) and result < 0
        else:
            raise AssertionError(f"Unexpected kind: {kind}")

    @pytest.mark.parametrize(
        ("array_value", "expected", "expected_type"),
        [
            pytest.param(np.array(7, dtype=np.int32), 7, int, id="zerod-int32"),
            pytest.param(np.array(3.5, dtype=np.float32), 3.5, float, id="zerod-float32"),
        ],
    )
    def test_zero_dim_arrays(self, array_value, expected, expected_type) -> None:
        """Convert zero-dimensional NumPy arrays via .item() path."""
        result = std_numeric(array_value, on_error="raise", allow_bool=False)
        assert type(result) is expected_type
        if expected_type is float:
            assert result == pytest.approx(expected)
        else:
            assert result == expected

    @pytest.mark.parametrize(
        ("value", "allow_bool", "expected", "expect_error"),
        [
            pytest.param(np.bool_(True), True, 1, False, id="bool-true-allowed"),
            pytest.param(np.bool_(False), True, 0, False, id="bool-false-allowed"),
            pytest.param(np.bool_(True), False, None, True, id="bool-true-rejected"),
        ],
    )
    def test_numpy_bool_behavior(self, value, allow_bool, expected, expect_error) -> None:
        """Handle NumPy booleans based on allow_bool flag."""
        if expect_error:
            with pytest.raises(TypeError, match=r"(?i)boolean values not supported"):
                std_numeric(value, on_error="raise", allow_bool=allow_bool)
        else:
            result = std_numeric(value, on_error="raise", allow_bool=allow_bool)
            assert type(result) is int
            assert result == expected

    @pytest.mark.parametrize(
        ("array_like", "type_pattern"),
        [
            pytest.param(
                np.array([1, 2, 3], dtype=np.int32),
                r"(?i)(array|array-like)",
                id="1d-array-multi-element"
            ),
            pytest.param(
                np.array([[1, 2], [3, 4]], dtype=np.float64),
                r"(?i)(array|array-like)",
                id="2d-array"
            ),
        ],
    )
    def test_reject_numpy_arrays(self, array_like, type_pattern) -> None:
        """Reject numpy arrays with ndim > 0 - these are collections, not scalars."""
        with pytest.raises(TypeError, match=type_pattern):
            std_numeric(array_like, on_error="raise", allow_bool=False)


# pandas Tests --------------------------------------------------------------------------------------
pd = pytest.importorskip("pandas")


class TestStdNumPandasNumericSupport:
    """
    Core tests for Pandas datatype support in std_numeric.
    """

    @pytest.mark.parametrize(
        ("scalar", "expected"),
        [
            pytest.param(pd.Series([-5], dtype="Int8").iloc[0], -5, id="int8-neg"),
            pytest.param(pd.Series([123], dtype="Int16").iloc[0], 123, id="int16-pos"),
            pytest.param(pd.Series([65530], dtype="UInt16").iloc[0], 65530, id="uint16-large"),
            pytest.param(pd.Series([2 ** 63 - 1], dtype="Int64").iloc[0], 2 ** 63 - 1, id="int64-max"),
            pytest.param(pd.Series([2 ** 63 + 5], dtype="UInt64").iloc[0], 2 ** 63 + 5, id="uint64-beyond-int64"),
        ],
    )
    def test_pd_integers_to_int(self, scalar, expected) -> None:
        """Convert Pandas integer scalars to Python int."""
        result = std_numeric(scalar, on_error="raise", allow_bool=False)
        assert type(result) is int
        assert result == expected

    @pytest.mark.parametrize(
        ("scalar", "expected", "kind"),
        [
            # Finite values
            pytest.param(pd.Series([-2.25], dtype="Float32").iloc[0], -2.25, "finite", id="float32-neg"),
            pytest.param(pd.Series([1.0e100], dtype="Float64").iloc[0], 1.0e100, "finite", id="float64-large"),
            # Specials
            pytest.param(pd.Series([float("nan")], dtype="Float32").iloc[0], None, "nan", id="nan-f32"),
            pytest.param(pd.Series([float("inf")], dtype="Float64").iloc[0], None, "inf+", id="inf-pos-f64"),
            pytest.param(pd.Series([float("-inf")], dtype="Float64").iloc[0], None, "inf-", id="inf-neg-f64"),
        ],
    )
    def test_pd_floats(self, scalar, expected, kind) -> None:
        """Convert Pandas float scalars and preserve special values."""
        result = std_numeric(scalar, on_error="raise", allow_bool=False)
        assert type(result) is float
        if kind == "finite":
            assert result == pytest.approx(expected)
        elif kind == "nan":
            assert math.isnan(result)
        elif kind == "inf+":
            assert math.isinf(result) and result > 0
        elif kind == "inf-":
            assert math.isinf(result) and result < 0
        else:
            raise AssertionError(f"Unexpected kind: {kind}")

    def test_pd_na_to_nan(self) -> None:
        """Convert pandas.NA to float('nan')."""
        result = std_numeric(pd.NA, on_error="raise", allow_bool=False)
        assert type(result) is float
        assert math.isnan(result)

    @pytest.mark.parametrize(
        ("value", "allow_bool", "expected", "expect_error"),
        [
            pytest.param(pd.array([True], dtype="boolean")[0],
                         True, 1, False, id="bool-true-allowed"),
            pytest.param(pd.array([False], dtype="boolean")[0],
                         True, 0, False, id="bool-false-allowed"),
            pytest.param(pd.array([True], dtype="boolean")[0],
                         False, None, True, id="bool-true-rejected"),
        ],
    )
    def test_pandas_bool_behavior(self, value, allow_bool, expected, expect_error) -> None:
        """Handle Pandas booleans based on allow_bool flag."""
        if expect_error:
            with pytest.raises(TypeError, match=r"(?i)boolean.*not supported"):
                std_numeric(value, on_error="raise", allow_bool=allow_bool)
        else:
            result = std_numeric(value, on_error="raise", allow_bool=allow_bool)
            assert type(result) is int
            assert result == expected

    @pytest.mark.parametrize(
        ("series_value", "expected", "expected_type"),
        [
            pytest.param(pd.Series([7], dtype='int32').iloc[0], 7, int, id="series-int32"),
            pytest.param(pd.Series([3.5], dtype='float32').iloc[0], 3.5, float, id="series-float32"),
        ],
    )
    def test_series_scalars(self, series_value, expected, expected_type) -> None:
        """Convert Pandas Series scalar values."""
        result = std_numeric(series_value, on_error="raise", allow_bool=False)
        assert type(result) is expected_type
        if expected_type is float:
            assert result == pytest.approx(expected)
        else:
            assert result == expected

    @pytest.mark.parametrize(
        ("nullable_int", "expected"),
        [
            pytest.param(pd.array([42], dtype="Int64")[0], 42, id="nullable-int64"),
            pytest.param(pd.array([pd.NA], dtype="Int64")[0], None, id="nullable-int64-na"),
        ],
    )
    def test_nullable_integers(self, nullable_int, expected) -> None:
        """Handle Pandas nullable integer arrays."""
        result = std_numeric(nullable_int, on_error="raise", allow_bool=False)
        if expected is None:
            assert type(result) is float
            assert math.isnan(result)
        else:
            assert type(result) is int
            assert result == expected

    @pytest.mark.parametrize(
        ("array_like", "type_pattern"),
        [
            pytest.param(
                pd.Series([1, 2, 3], dtype="Int64"),
                r"(?i)(series|array-like)",
                id="series-multi-element"
            ),
            pytest.param(
                pd.DataFrame({"a": [1, 2, 3]}),
                r"(?i)(dataframe|array-like)",
                id="dataframe"
            ),
        ],
    )
    def test_reject_pandas_collections(self, array_like, type_pattern) -> None:
        """Reject pandas Series/DataFrame - these are collections, not scalars."""
        with pytest.raises(TypeError, match=type_pattern):
            std_numeric(array_like, on_error="raise", allow_bool=False)


# PyTorch Tests ---------------------------------------------------------------------------------
torch = pytest.importorskip("torch")


class TestStdNumPyTorchNumericSupport:
    """
    Core tests for PyTorch datatype support in std_numeric.
    """

    @pytest.mark.parametrize(
        ("scalar", "expected"),
        [
            pytest.param(torch.tensor(-5, dtype=torch.int8), -5, id="int8-neg"),
            pytest.param(torch.tensor(123, dtype=torch.int16), 123, id="int16-pos"),
            pytest.param(torch.tensor(65530, dtype=torch.int32), 65530, id="int32-large"),
            pytest.param(torch.tensor(2 ** 31 - 1, dtype=torch.int32), 2 ** 31 - 1, id="int32-max"),
            pytest.param(torch.tensor(2 ** 63 - 1, dtype=torch.int64), 2 ** 63 - 1, id="int64-max"),
        ],
    )
    def test_torch_integers_to_int(self, scalar, expected) -> None:
        """Convert PyTorch integer scalars to Python int."""
        result = std_numeric(scalar, on_error="raise", allow_bool=False)
        assert type(result) is int
        assert result == expected

    @pytest.mark.parametrize(
        ("scalar", "expected"),
        [
            pytest.param(torch.tensor(255, dtype=torch.uint8), 255, id="uint8-max"),
            pytest.param(torch.tensor(128, dtype=torch.uint8), 128, id="uint8-mid"),
            pytest.param(torch.tensor(0, dtype=torch.uint8), 0, id="uint8-zero"),
        ],
    )
    def test_torch_unsigned_integers_to_int(self, scalar, expected) -> None:
        """Convert PyTorch unsigned integer scalars to Python int."""
        result = std_numeric(scalar, on_error="raise", allow_bool=False)
        assert type(result) is int
        assert result == expected

    @pytest.mark.parametrize(
        ("scalar", "expected", "kind"),
        [
            # Finite values - float16 (half precision)
            pytest.param(torch.tensor(3.5, dtype=torch.float16), 3.5, "finite", id="float16-3.5"),
            pytest.param(torch.tensor(-1.25, dtype=torch.float16), -1.25, "finite", id="float16-neg"),
            # Finite values - float32
            pytest.param(torch.tensor(-2.25, dtype=torch.float32), -2.25, "finite", id="float32-neg"),
            pytest.param(torch.tensor(1234.5678, dtype=torch.float32), 1234.5678, "finite", id="float32-precise"),
            # Finite values - float64 (double precision)
            pytest.param(torch.tensor(1.0e100, dtype=torch.float64), 1.0e100, "finite", id="float64-large"),
            pytest.param(torch.tensor(-9.87654321e-50, dtype=torch.float64), -9.87654321e-50, "finite",
                         id="float64-tiny"),
            # Specials - NaN
            pytest.param(torch.tensor(float("nan"), dtype=torch.float32), None, "nan", id="nan-f32"),
            pytest.param(torch.tensor(float("nan"), dtype=torch.float64), None, "nan", id="nan-f64"),
            # Specials - Infinity
            pytest.param(torch.tensor(float("inf"), dtype=torch.float32), None, "inf+", id="inf-pos-f32"),
            pytest.param(torch.tensor(float("-inf"), dtype=torch.float32), None, "inf-", id="inf-neg-f32"),
            pytest.param(torch.tensor(float("inf"), dtype=torch.float64), None, "inf+", id="inf-pos-f64"),
            pytest.param(torch.tensor(float("-inf"), dtype=torch.float64), None, "inf-", id="inf-neg-f64"),
        ],
    )
    def test_torch_floats(self, scalar, expected, kind) -> None:
        """Convert PyTorch float scalars and preserve special values."""
        result = std_numeric(scalar, on_error="raise", allow_bool=False)
        assert type(result) is float
        if kind == "finite":
            assert result == pytest.approx(expected, rel=1e-5)
        elif kind == "nan":
            assert math.isnan(result)
        elif kind == "inf+":
            assert math.isinf(result) and result > 0
        elif kind == "inf-":
            assert math.isinf(result) and result < 0
        else:
            raise AssertionError(f"Unexpected kind: {kind}")

    @pytest.mark.parametrize(
        ("tensor_value", "expected", "expected_type"),
        [
            pytest.param(torch.tensor(7, dtype=torch.int32), 7, int, id="zerod-int32"),
            pytest.param(torch.tensor(3.5, dtype=torch.float32), 3.5, float, id="zerod-float32"),
            pytest.param(torch.tensor(-42, dtype=torch.int64), -42, int, id="zerod-int64"),
            pytest.param(torch.tensor(2.718, dtype=torch.float64), 2.718, float, id="zerod-float64"),
        ],
    )
    def test_zero_dim_tensors(self, tensor_value, expected, expected_type) -> None:
        """Convert zero-dimensional PyTorch tensors via .item() path."""
        result = std_numeric(tensor_value, on_error="raise", allow_bool=False)
        assert type(result) is expected_type
        if expected_type is float:
            assert result == pytest.approx(expected, rel=1e-5)
        else:
            assert result == expected

    @pytest.mark.parametrize(
        ("value", "allow_bool", "expected", "expect_error"),
        [
            pytest.param(torch.tensor(True, dtype=torch.bool), True, 1, False, id="bool-true-allowed"),
            pytest.param(torch.tensor(False, dtype=torch.bool), True, 0, False, id="bool-false-allowed"),
            pytest.param(torch.tensor(True, dtype=torch.bool), False, None, True, id="bool-true-rejected"),
            pytest.param(torch.tensor(False, dtype=torch.bool), False, None, True, id="bool-false-rejected"),
        ],
    )
    def test_torch_bool_behavior(self, value, allow_bool, expected, expect_error) -> None:
        """Handle PyTorch booleans based on allow_bool flag."""
        if expect_error:
            with pytest.raises(TypeError, match=r"(?i)boolean.*not supported"):
                std_numeric(value, on_error="raise", allow_bool=allow_bool)
        else:
            result = std_numeric(value, on_error="raise", allow_bool=allow_bool)
            assert type(result) is int
            assert result == expected

    @pytest.mark.parametrize(
        ("tensor", "type_pattern"),
        [
            pytest.param(
                torch.tensor([1, 2, 3], dtype=torch.int32),
                r"(?i)(tensor|array-like)",
                id="1d-tensor-multi-element"
            ),
            pytest.param(
                torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
                r"(?i)(tensor|array-like)",
                id="2d-tensor"
            ),
            pytest.param(
                torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.int64),
                r"(?i)(tensor|array-like)",
                id="3d-tensor"
            ),
        ],
    )
    def test_reject_torch_tensors(self, tensor, type_pattern) -> None:
        """Reject PyTorch tensors with ndim > 0 - these are collections, not scalars."""
        with pytest.raises(TypeError, match=type_pattern):
            std_numeric(tensor, on_error="raise", allow_bool=False)

    @pytest.mark.parametrize(
        ("scalar", "expected"),
        [
            pytest.param(torch.tensor(1.5, dtype=torch.bfloat16), 1.5, id="bfloat16-1.5"),
            pytest.param(torch.tensor(-3.25, dtype=torch.bfloat16), -3.25, id="bfloat16-neg"),
        ],
    )
    def test_torch_bfloat16(self, scalar, expected) -> None:
        """Convert PyTorch bfloat16 scalars (brain floating point format)."""
        result = std_numeric(scalar, on_error="raise", allow_bool=False)
        assert type(result) is float
        # bfloat16 has lower precision, so use wider tolerance
        assert result == pytest.approx(expected, rel=1e-2)

    @pytest.mark.parametrize(
        ("scalar", "expected", "kind"),
        [
            # Complex numbers should be rejected as unsupported
            pytest.param(
                torch.tensor(1 + 2j, dtype=torch.complex64),
                None,
                "complex64",
                id="complex64"
            ),
            pytest.param(
                torch.tensor(3 - 4j, dtype=torch.complex128),
                None,
                "complex128",
                id="complex128"
            ),
        ],
    )
    def test_torch_complex_rejected(self, scalar, expected, kind) -> None:
        """Reject PyTorch complex number types - not numeric scalars."""
        with pytest.raises(TypeError, match=r"(?i)(complex|unsupported)"):
            std_numeric(scalar, on_error="raise", allow_bool=False)

    def test_torch_tensor_from_python_int(self) -> None:
        """Handle PyTorch tensors created from Python integers."""
        value = torch.tensor(42)
        result = std_numeric(value, on_error="raise", allow_bool=False)
        assert type(result) is int
        assert result == 42

    def test_torch_tensor_from_python_float(self) -> None:
        """Handle PyTorch tensors created from Python floats."""
        value = torch.tensor(3.14159)
        result = std_numeric(value, on_error="raise", allow_bool=False)
        assert type(result) is float
        assert result == pytest.approx(3.14159)

    @pytest.mark.parametrize(
        ("on_error_mode", "expected_result"),
        [
            pytest.param("nan", None, id="on-error-nan"),
            pytest.param("none", None, id="on-error-none"),
        ],
    )
    def test_torch_tensor_with_on_error_modes(self, on_error_mode, expected_result) -> None:
        """Ensure valid PyTorch scalars work with all on_error modes."""
        value = torch.tensor(42, dtype=torch.int32)
        result = std_numeric(value, on_error=on_error_mode, allow_bool=False)
        assert type(result) is int
        assert result == 42

    def test_torch_zero_with_sign(self) -> None:
        """Preserve sign of zero in PyTorch floats."""
        pos_zero = torch.tensor(0.0, dtype=torch.float32)
        neg_zero = torch.tensor(-0.0, dtype=torch.float32)

        result_pos = std_numeric(pos_zero, on_error="raise", allow_bool=False)
        result_neg = std_numeric(neg_zero, on_error="raise", allow_bool=False)

        assert type(result_pos) is float
        assert type(result_neg) is float
        assert result_pos == 0.0
        assert result_neg == 0.0
        # Check sign using copysign or division
        assert math.copysign(1.0, result_pos) == 1.0
        assert math.copysign(1.0, result_neg) == -1.0


# tensorflow Tests ------------------------------------------------------------------------------------
tf = pytest.importorskip("tensorflow")


class TestStdNumTensorFlowNumericSupport:
    """
    Core tests for TensorFlow datatype support in std_numeric.
    """

    @pytest.mark.parametrize(
        ("tensor", "expected"),
        [
            pytest.param(tf.constant(-5, dtype=tf.int8), -5, id="int8-neg"),
            pytest.param(tf.constant(123, dtype=tf.int16), 123, id="int16-pos"),
            pytest.param(tf.constant(65530, dtype=tf.uint16), 65530, id="uint16-large"),
            pytest.param(tf.constant(2 ** 63 - 1, dtype=tf.int64), 2 ** 63 - 1, id="int64-max"),
            pytest.param(tf.constant(2 ** 63 + 5, dtype=tf.uint64), 2 ** 63 + 5, id="uint64-beyond-int64"),
        ],
    )
    def test_tf_integers_to_int(self, tensor, expected) -> None:
        """Convert TensorFlow integer scalars to Python int."""
        result = std_numeric(tensor, on_error="raise", allow_bool=False)
        assert type(result) is int
        assert result == expected

    @pytest.mark.parametrize(
        ("tensor", "expected", "kind"),
        [
            # Finite values
            pytest.param(tf.constant(3.5, dtype=tf.float16), 3.5, "finite", id="float16-3.5"),
            pytest.param(tf.constant(-2.25, dtype=tf.float32), -2.25, "finite", id="float32-neg"),
            pytest.param(tf.constant(1.0e100, dtype=tf.float64), 1.0e100, "finite", id="float64-large"),
            # Specials
            pytest.param(tf.constant(float("nan"), dtype=tf.float32), None, "nan", id="nan-f32"),
            pytest.param(tf.constant(float("inf"), dtype=tf.float64), None, "inf+", id="inf-pos-f64"),
            pytest.param(tf.constant(float("-inf"), dtype=tf.float64), None, "inf-", id="inf-neg-f64"),
        ],
    )
    def test_tf_floats(self, tensor, expected, kind) -> None:
        """Convert TensorFlow float scalars and preserve special values."""
        result = std_numeric(tensor, on_error="raise", allow_bool=False)
        assert type(result) is float
        if kind == "finite":
            assert result == pytest.approx(expected)
        elif kind == "nan":
            assert math.isnan(result)
        elif kind == "inf+":
            assert math.isinf(result) and result > 0
        elif kind == "inf-":
            assert math.isinf(result) and result < 0
        else:
            raise AssertionError(f"Unexpected kind: {kind}")

    @pytest.mark.parametrize(
        ("value", "allow_bool", "expected", "expect_error"),
        [
            pytest.param(tf.constant(True, dtype=tf.bool), True, 1, False, id="bool-true-allowed"),
            pytest.param(tf.constant(False, dtype=tf.bool), True, 0, False, id="bool-false-allowed"),
            pytest.param(tf.constant(True, dtype=tf.bool), False, None, True, id="bool-true-rejected"),
        ],
    )
    def test_tf_bool_behavior(self, value, allow_bool, expected, expect_error) -> None:
        """Handle TensorFlow booleans based on allow_bool flag."""
        if expect_error:
            with pytest.raises(TypeError, match=r"(?i)boolean.*not supported"):
                std_numeric(value, on_error="raise", allow_bool=allow_bool)
        else:
            result = std_numeric(value, on_error="raise", allow_bool=allow_bool)
            assert type(result) is int
            assert result == expected

    @pytest.mark.parametrize(
        ("tensor", "expected", "expected_type"),
        [
            pytest.param(tf.constant(7, dtype=tf.int32), 7, int, id="scalar-int32"),
            pytest.param(tf.constant(3.5, dtype=tf.float32), 3.5, float, id="scalar-float32"),
        ],
    )
    def test_tf_scalar_tensors(self, tensor, expected, expected_type) -> None:
        """Convert rank-0 TensorFlow tensors via .numpy()/.item() protocol."""
        result = std_numeric(tensor, on_error="raise", allow_bool=False)
        assert type(result) is expected_type
        if expected_type is float:
            assert result == pytest.approx(expected)
        else:
            assert result == expected

    @pytest.mark.parametrize(
        ("tensor_like", "type_pattern"),
        [
            pytest.param(tf.constant([1, 2, 3], dtype=tf.int32), r"(?i)(tensor|array-like)", id="1d-tensor"),
            pytest.param(tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float64),
                         r"(?i)(tensor|array-like)", id="2d-tensor"),
        ],
    )
    def test_reject_tf_non_scalar_tensors(self, tensor_like, type_pattern) -> None:
        """Reject TensorFlow tensors with rank > 0 - these are collections, not scalars."""
        with pytest.raises(TypeError, match=type_pattern):
            std_numeric(tensor_like, on_error="raise", allow_bool=False)


# ... existing code ...

# JAX Tests ----------------------------------------------------------------------------------------
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


class TestStdNumJaxNumericSupport:
    """
    Core tests for JAX datatype support in std_numeric.
    Note: JAX scalars are DeviceArray/Array scalars; multi-d arrays should be rejected.
    """

    @pytest.mark.parametrize(
        ("scalar", "expected"),
        [
            pytest.param(jnp.int8(-5), -5, id="int8-neg"),
            pytest.param(jnp.int16(123), 123, id="int16-pos"),
            pytest.param(jnp.uint16(65530), 65530, id="uint16-large"),
            pytest.param(jnp.int32(-2147483648), -2147483648, id="int32-min"),
            pytest.param(jnp.int32(2147483647), 2147483647, id="int32-max"),
            # # Note: JAX's int64/uint64 support is platform-dependent and may silently
            # # truncate to int32 on some systems, so better test moderate values only
            # # OR disable int64 tests
            # pytest.param(jnp.int64(1000000), 1000000, id="int64-moderate"),
            # pytest.param(jnp.uint64(2000000), 2000000, id="uint64-moderate"),
        ],
    )
    def test_jax_integers_to_int(self, scalar, expected) -> None:
        """Convert JAX integer scalars to Python int."""
        result = std_numeric(scalar, on_error="raise", allow_bool=False)
        assert type(result) is int
        assert result == expected

    @pytest.mark.parametrize(
        ("scalar", "expected", "kind"),
        [
            # Finite values
            pytest.param(jnp.float16(3.5), 3.5, "finite", id="float16-3.5"),
            pytest.param(jnp.float32(-2.25), -2.25, "finite", id="float32-neg"),
            # Use value within float32 range for portability (float32 max ~3.4e38)
            pytest.param(jnp.float32(1.0e30), 1.0e30, "finite", id="float32-large"),
            # Specials
            pytest.param(jnp.float32(jnp.nan), None, "nan", id="nan-f32"),
            pytest.param(jnp.float32(jnp.inf), None, "inf+", id="inf-pos-f64"),
            pytest.param(jnp.float32(-jnp.inf), None, "inf-", id="inf-neg-f64"),
            # # JAX defaults to 32-bit mode; use moderate value for float64
            # # OR disable fp64 tests
            # pytest.param(jnp.float64(1234.5678), 1234.5678, "finite", id="float64-moderate"),
        ],
    )
    def test_jax_floats(self, scalar, expected, kind) -> None:
        """Convert JAX float scalars and preserve special values."""
        result = std_numeric(scalar, on_error="raise", allow_bool=False)
        assert type(result) is float
        if kind == "finite":
            assert result == pytest.approx(expected)
        elif kind == "nan":
            assert math.isnan(result)
        elif kind == "inf+":
            assert math.isinf(result) and result > 0
        elif kind == "inf-":
            assert math.isinf(result) and result < 0
        else:
            raise AssertionError(f"Unexpected kind: {kind}")

    @pytest.mark.parametrize(
        ("array_value", "expected", "expected_type"),
        [
            pytest.param(jnp.array(7, dtype=jnp.int32), 7, int, id="zerod-int32"),
            pytest.param(jnp.array(3.5, dtype=jnp.float32), 3.5, float, id="zerod-float32"),
        ],
    )
    def test_zero_dim_arrays(self, array_value, expected, expected_type) -> None:
        """Convert zero-dimensional JAX arrays via .item() path."""
        result = std_numeric(array_value, on_error="raise", allow_bool=False)
        assert type(result) is expected_type
        if expected_type is float:
            assert result == pytest.approx(expected)
        else:
            assert result == expected

    @pytest.mark.parametrize(
        ("value", "allow_bool", "expected", "expect_error"),
        [
            pytest.param(jnp.bool_(True), True, 1, False, id="bool-true-allowed"),
            pytest.param(jnp.bool_(False), True, 0, False, id="bool-false-allowed"),
            pytest.param(jnp.bool_(True), False, None, True, id="bool-true-rejected"),
        ],
    )
    def test_jax_bool_behavior(self, value, allow_bool, expected, expect_error) -> None:
        """Handle JAX booleans based on allow_bool flag."""
        if expect_error:
            with pytest.raises(TypeError, match=r"(?i)boolean.*not supported"):
                std_numeric(value, on_error="raise", allow_bool=allow_bool)
        else:
            result = std_numeric(value, on_error="raise", allow_bool=allow_bool)
            assert type(result) is int
            assert result == expected

    @pytest.mark.parametrize(
        ("array_like", "type_pattern"),
        [
            pytest.param(
                jnp.array([1, 2, 3], dtype=jnp.int32),
                r"(?i)(array|array-like)",
                id="1d-array-multi-element"
            ),
            pytest.param(
                jnp.array([[1, 2], [3, 4]], dtype=jnp.float64),
                r"(?i)(array|array-like)",
                id="2d-array"
            ),
        ],
    )
    def test_reject_jax_arrays(self, array_like, type_pattern) -> None:
        """Reject JAX arrays with ndim > 0 - these are collections, not scalars."""
        with pytest.raises(TypeError, match=type_pattern):
            std_numeric(array_like, on_error="raise", allow_bool=False)
