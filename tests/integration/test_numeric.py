"""
Core test suite for std_numeric() - stdlib types only, no third-party dependencies.

Tests cover: basic types, Decimal/Fraction, overflow/underflow, special values,
error handling modes, boolean handling, and parameter combinations.
"""

import math
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.numeric import std_numeric


# Integration Tests ----------------------------------------------------------------------------------------------------

pytestmark = pytest.mark.integration

# Optional imports -----------------------------------------------------------------------------------------------------
astropy = pytest.importorskip("astropy")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
sp = pytest.importorskip("sympy")
tf = pytest.importorskip("tensorflow")
torch = pytest.importorskip("torch")

from astropy import units as u

# numpy Tests ------------------------------------------------------------------------------------


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
            pytest.param(np.int64(2**63 - 1), 2**63 - 1, id="int64-max"),
            pytest.param(np.uint64(2**63 + 5), 2**63 + 5, id="uint64-beyond-int64"),
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
        ("value", "expected", "expected_type"),
        [
            pytest.param(np.float64(5.0), 5, float, id="numpy-int-valued-float-simple"),
            pytest.param(np.float32(100.0), 100, float, id="numpy-int-valued-float-hundred"),
            pytest.param(np.float64(-42.0), -42, float, id="numpy-int-valued-float-negative"),
            pytest.param(np.float32(0.0), 0, float, id="numpy-int-valued-float-zero"),
            pytest.param(
                np.float64(1e10),
                10**10,
                float,
                id="numpy-int-valued-scientific-notation",
            ),
            pytest.param(np.float64(3.5), 3.5, float, id="numpy-fractional-float"),
            pytest.param(np.float32(1.1), 1.1, float, id="numpy-fractional-float-small"),
        ],
    )
    def test_np_float_type_preserved(self, value, expected, expected_type) -> None:
        """Convert numpy float to float for int-like source values"""
        result = std_numeric(value, on_error="raise", allow_bool=False)
        assert result == pytest.approx(expected)
        assert type(result) is expected_type

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
                id="1d-array-multi-element",
            ),
            pytest.param(
                np.array([[1, 2], [3, 4]], dtype=np.float64),
                r"(?i)(array|array-like)",
                id="2d-array",
            ),
        ],
    )
    def test_reject_numpy_arrays(self, array_like, type_pattern) -> None:
        """Reject numpy arrays with ndim > 0 - these are collections, not scalars."""
        with pytest.raises(TypeError, match=type_pattern):
            std_numeric(array_like, on_error="raise", allow_bool=False)


# pandas Tests --------------------------------------------------------------------------------------


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
            pytest.param(pd.Series([2**63 - 1], dtype="Int64").iloc[0], 2**63 - 1, id="int64-max"),
            pytest.param(
                pd.Series([2**63 + 5], dtype="UInt64").iloc[0],
                2**63 + 5,
                id="uint64-beyond-int64",
            ),
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
            pytest.param(
                pd.Series([-2.25], dtype="Float32").iloc[0],
                -2.25,
                "finite",
                id="float32-neg",
            ),
            pytest.param(
                pd.Series([1.0e100], dtype="Float64").iloc[0],
                1.0e100,
                "finite",
                id="float64-large",
            ),
            # Specials
            pytest.param(
                pd.Series([float("nan")], dtype="Float32").iloc[0],
                None,
                "nan",
                id="nan-f32",
            ),
            pytest.param(
                pd.Series([float("inf")], dtype="Float64").iloc[0],
                None,
                "inf+",
                id="inf-pos-f64",
            ),
            pytest.param(
                pd.Series([float("-inf")], dtype="Float64").iloc[0],
                None,
                "inf-",
                id="inf-neg-f64",
            ),
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
            pytest.param(
                pd.array([True], dtype="boolean")[0],
                True,
                1,
                False,
                id="bool-true-allowed",
            ),
            pytest.param(
                pd.array([False], dtype="boolean")[0],
                True,
                0,
                False,
                id="bool-false-allowed",
            ),
            pytest.param(
                pd.array([True], dtype="boolean")[0],
                False,
                None,
                True,
                id="bool-true-rejected",
            ),
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
            pytest.param(pd.Series([7], dtype="int32").iloc[0], 7, int, id="series-int32"),
            pytest.param(
                pd.Series([3.5], dtype="float32").iloc[0],
                3.5,
                float,
                id="series-float32",
            ),
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
                id="series-multi-element",
            ),
            pytest.param(
                pd.DataFrame({"a": [1, 2, 3]}),
                r"(?i)(dataframe|array-like)",
                id="dataframe",
            ),
        ],
    )
    def test_reject_pandas_collections(self, array_like, type_pattern) -> None:
        """Reject pandas Series/DataFrame - these are collections, not scalars."""
        with pytest.raises(TypeError, match=type_pattern):
            std_numeric(array_like, on_error="raise", allow_bool=False)


# PyTorch Tests ---------------------------------------------------------------------------------


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
            pytest.param(torch.tensor(2**31 - 1, dtype=torch.int32), 2**31 - 1, id="int32-max"),
            pytest.param(torch.tensor(2**63 - 1, dtype=torch.int64), 2**63 - 1, id="int64-max"),
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
            pytest.param(
                torch.tensor(-1.25, dtype=torch.float16),
                -1.25,
                "finite",
                id="float16-neg",
            ),
            # Finite values - float32
            pytest.param(
                torch.tensor(-2.25, dtype=torch.float32),
                -2.25,
                "finite",
                id="float32-neg",
            ),
            pytest.param(
                torch.tensor(1234.5678, dtype=torch.float32),
                1234.5678,
                "finite",
                id="float32-precise",
            ),
            # Finite values - float64 (double precision)
            pytest.param(
                torch.tensor(1.0e100, dtype=torch.float64),
                1.0e100,
                "finite",
                id="float64-large",
            ),
            pytest.param(
                torch.tensor(-9.87654321e-50, dtype=torch.float64),
                -9.87654321e-50,
                "finite",
                id="float64-tiny",
            ),
            # Specials - NaN
            pytest.param(
                torch.tensor(float("nan"), dtype=torch.float32),
                None,
                "nan",
                id="nan-f32",
            ),
            pytest.param(
                torch.tensor(float("nan"), dtype=torch.float64),
                None,
                "nan",
                id="nan-f64",
            ),
            # Specials - Infinity
            pytest.param(
                torch.tensor(float("inf"), dtype=torch.float32),
                None,
                "inf+",
                id="inf-pos-f32",
            ),
            pytest.param(
                torch.tensor(float("-inf"), dtype=torch.float32),
                None,
                "inf-",
                id="inf-neg-f32",
            ),
            pytest.param(
                torch.tensor(float("inf"), dtype=torch.float64),
                None,
                "inf+",
                id="inf-pos-f64",
            ),
            pytest.param(
                torch.tensor(float("-inf"), dtype=torch.float64),
                None,
                "inf-",
                id="inf-neg-f64",
            ),
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
            pytest.param(
                torch.tensor(2.718, dtype=torch.float64),
                2.718,
                float,
                id="zerod-float64",
            ),
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
            pytest.param(
                torch.tensor(True, dtype=torch.bool),
                True,
                1,
                False,
                id="bool-true-allowed",
            ),
            pytest.param(
                torch.tensor(False, dtype=torch.bool),
                True,
                0,
                False,
                id="bool-false-allowed",
            ),
            pytest.param(
                torch.tensor(True, dtype=torch.bool),
                False,
                None,
                True,
                id="bool-true-rejected",
            ),
            pytest.param(
                torch.tensor(False, dtype=torch.bool),
                False,
                None,
                True,
                id="bool-false-rejected",
            ),
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
                id="1d-tensor-multi-element",
            ),
            pytest.param(
                torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
                r"(?i)(tensor|array-like)",
                id="2d-tensor",
            ),
            pytest.param(
                torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.int64),
                r"(?i)(tensor|array-like)",
                id="3d-tensor",
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
                id="complex64",
            ),
            pytest.param(
                torch.tensor(3 - 4j, dtype=torch.complex128),
                None,
                "complex128",
                id="complex128",
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
            pytest.param(tf.constant(2**63 - 1, dtype=tf.int64), 2**63 - 1, id="int64-max"),
            pytest.param(
                tf.constant(2**63 + 5, dtype=tf.uint64),
                2**63 + 5,
                id="uint64-beyond-int64",
            ),
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
            pytest.param(
                tf.constant(1.0e100, dtype=tf.float64),
                1.0e100,
                "finite",
                id="float64-large",
            ),
            # Specials
            pytest.param(tf.constant(float("nan"), dtype=tf.float32), None, "nan", id="nan-f32"),
            pytest.param(
                tf.constant(float("inf"), dtype=tf.float64),
                None,
                "inf+",
                id="inf-pos-f64",
            ),
            pytest.param(
                tf.constant(float("-inf"), dtype=tf.float64),
                None,
                "inf-",
                id="inf-neg-f64",
            ),
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
            pytest.param(
                tf.constant(False, dtype=tf.bool),
                True,
                0,
                False,
                id="bool-false-allowed",
            ),
            pytest.param(
                tf.constant(True, dtype=tf.bool),
                False,
                None,
                True,
                id="bool-true-rejected",
            ),
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
            pytest.param(
                tf.constant([1, 2, 3], dtype=tf.int32),
                r"(?i)(tensor|array-like)",
                id="1d-tensor",
            ),
            pytest.param(
                tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float64),
                r"(?i)(tensor|array-like)",
                id="2d-tensor",
            ),
        ],
    )
    def test_reject_tf_non_scalar_tensors(self, tensor_like, type_pattern) -> None:
        """Reject TensorFlow tensors with rank > 0 - these are collections, not scalars."""
        with pytest.raises(TypeError, match=type_pattern):
            std_numeric(tensor_like, on_error="raise", allow_bool=False)


# ... existing code ...

# JAX Tests ----------------------------------------------------------------------------------------


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
            # # JAX defaults to 32-bit mode;
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
                id="1d-array-multi-element",
            ),
            pytest.param(
                jnp.array([[1, 2], [3, 4]], dtype=jnp.float32),
                r"(?i)(array|array-like)",
                id="2d-array",
            ),
        ],
    )
    def test_reject_jax_arrays(self, array_like, type_pattern) -> None:
        """Reject JAX arrays with ndim > 0 - these are collections, not scalars."""
        with pytest.raises(TypeError, match=type_pattern):
            std_numeric(array_like, on_error="raise", allow_bool=False)


# Astropy Tests ---------------------------------------------------------------------------------


class TestStdNumAstropyNumericSupport:
    """
    Core tests for Astropy Quantity support in std_numeric.
    Quantities have physical units; std_numeric extracts the numeric magnitude.
    """

    @pytest.mark.parametrize(
        ("quantity", "expected"),
        [
            # Literal multiplication creates float64 by default in Astropy
            pytest.param(5 * u.m, 5.0, id="literal-meter"),
            pytest.param(-42 * u.s, -42.0, id="literal-neg-second"),
            pytest.param(100 * u.kg, 100.0, id="literal-kilogram"),
            pytest.param(0 * u.K, 0.0, id="literal-zero-kelvin"),
        ],
    )
    def test_astropy_quantity_from_literals(self, quantity, expected) -> None:
        """Astropy creates float64 from literal multiplication - preserve as float."""
        result = std_numeric(quantity, on_error="raise", allow_bool=False)
        assert type(result) is float
        assert result == expected

    @pytest.mark.parametrize(
        ("quantity", "expected"),
        [
            # Astropy Quantity converts integer dtypes to float64 internally
            pytest.param(u.Quantity(np.int32(5), u.m), 5.0, id="int32-meter"),
            pytest.param(u.Quantity(np.int64(-42), u.s), -42.0, id="int64-second"),
            pytest.param(u.Quantity(np.int64(100), u.kg), 100.0, id="int64-kilogram"),
        ],
    )
    def test_astropy_quantity_explicit_int(self, quantity, expected) -> None:
        """Astropy Quantity with integer input converts to float (Astropy behavior)."""
        result = std_numeric(quantity, on_error="raise", allow_bool=False)
        assert type(result) is float
        assert result == expected

    @pytest.mark.parametrize(
        ("quantity", "expected"),
        [
            pytest.param(3.5 * u.m, 3.5, id="float-meter"),
            pytest.param(-2.25 * u.s, -2.25, id="float-neg-second"),
            pytest.param(9.8 * (u.m / u.s**2), 9.8, id="float-acceleration"),
            pytest.param(1.602e-19 * u.C, 1.602e-19, id="float-tiny-coulomb"),
            pytest.param(299792458.123 * (u.m / u.s), 299792458.123, id="float-large-speed"),
        ],
    )
    def test_astropy_quantity_floats(self, quantity, expected) -> None:
        """Convert Astropy Quantity with float values to Python float."""
        result = std_numeric(quantity, on_error="raise", allow_bool=False)
        assert type(result) is float
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("quantity", "expected"),
        [
            pytest.param(5.0 * u.m, 5.0, id="int-valued-float-simple"),
            pytest.param(100.0 * u.kg, 100.0, id="int-valued-float-hundred"),
            pytest.param(-42.0 * u.s, -42.0, id="int-valued-float-negative"),
            pytest.param(0.0 * u.K, 0.0, id="int-valued-float-zero"),
            pytest.param(1e10 * u.m, 1e10, id="int-valued-scientific-notation"),
            pytest.param(3.5 * u.m, 3.5, id="fractional-float"),
            pytest.param(1.1 * u.kg, 1.1, id="fractional-float-small"),
        ],
    )
    def test_astropy_quantity_float_preserves_float(self, quantity, expected) -> None:
        """Integer-valued floats in Quantities stay float - no smart conversion."""
        result = std_numeric(quantity, on_error="raise", allow_bool=False)
        assert type(result) is float
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("quantity", "kind"),
        [
            pytest.param(float("nan") * u.m, "nan", id="nan-meter"),
            pytest.param(float("inf") * u.s, "inf+", id="inf-pos-second"),
            pytest.param(float("-inf") * u.K, "inf-", id="inf-neg-kelvin"),
        ],
    )
    def test_astropy_quantity_special_values(self, quantity, kind) -> None:
        """Handle Astropy Quantity with special IEEE 754 values (nan, inf)."""
        result = std_numeric(quantity, on_error="raise", allow_bool=False)
        assert type(result) is float
        if kind == "nan":
            assert math.isnan(result)
        elif kind == "inf+":
            assert math.isinf(result) and result > 0
        elif kind == "inf-":
            assert math.isinf(result) and result < 0
        else:
            raise AssertionError(f"Unexpected kind: {kind}")

    @pytest.mark.parametrize(
        ("quantity", "expected"),
        [
            pytest.param(u.Quantity(np.int32(42), unit=u.m), 42, id="quantity-np-int32"),
            pytest.param(
                u.Quantity(np.int64(2**40), unit=u.s),
                2**40,
                id="quantity-np-int64-large",
            ),
            pytest.param(u.Quantity(np.float32(3.14), unit=u.rad), 3.14, id="quantity-np-float32"),
            pytest.param(
                u.Quantity(np.float64(1.23456789), unit=u.kg),
                1.23456789,
                id="quantity-np-float64",
            ),
        ],
    )
    def test_astropy_quantity_numpy_dtypes_to_float(self, quantity, expected) -> None:
        """Handle Astropy Quantity wrapping NumPy scalar types - always float output."""
        result = std_numeric(quantity, on_error="raise", allow_bool=False)
        assert type(result) is float
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("quantity", "expected_type"),
        [
            pytest.param(u.Quantity([1, 2, 3], unit=u.m), None, id="1d-array"),
            pytest.param(u.Quantity([[1, 2], [3, 4]], unit=u.s), None, id="2d-array"),
            pytest.param(u.Quantity(np.arange(10), unit=u.kg), None, id="range-array"),
        ],
    )
    def test_reject_astropy_quantity_arrays(self, quantity, expected_type) -> None:
        """Reject Astropy Quantity arrays - these are collections, not scalars."""
        with pytest.raises(TypeError, match=r"(?i)(array|collection)"):
            std_numeric(quantity, on_error="raise", allow_bool=False)

    def test_astropy_quantity_zero_dimensional(self) -> None:
        """Convert zero-dimensional Astropy Quantity arrays."""
        # Zero-dimensional array from np.array(42) is int64 by default
        quantity = u.Quantity(np.array(42), unit=u.m)
        result = std_numeric(quantity, on_error="raise", allow_bool=False)
        assert type(result) is float
        assert result == pytest.approx(42)

    @pytest.mark.parametrize(
        ("quantity", "expected"),
        [
            pytest.param(1.234 * u.dimensionless_unscaled, 1.234, id="dimensionless-float"),
            # Literal multiplication creates float64
            pytest.param(50 * u.percent, 50.0, id="percent-literal"),
            pytest.param(2 * u.rad, 2.0, id="radian-literal"),
            # Explicit integer dtype
            pytest.param(u.Quantity(np.int32(50), u.percent), 50, id="percent-int32"),
        ],
    )
    def test_astropy_dimensionless_quantities(self, quantity, expected) -> None:
        """Handle dimensionless Astropy Quantities - dtype determines output type."""
        result = std_numeric(quantity, on_error="raise", allow_bool=False)
        assert type(result) is float
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize(
        "on_error_mode",
        [
            pytest.param("nan", id="on-error-nan"),
            pytest.param("none", id="on-error-none"),
        ],
    )
    def test_astropy_quantity_with_on_error_modes(self, on_error_mode) -> None:
        """Ensure valid Astropy Quantities work with all on_error modes."""
        quantity = 42.5 * u.m
        result = std_numeric(quantity, on_error=on_error_mode, allow_bool=False)
        assert type(result) is float
        assert result == pytest.approx(42.5)

    def test_astropy_quantity_extraction_via_value(self) -> None:
        """Verify Astropy Quantity uses .value attribute for extraction."""
        quantity = 123.456 * u.J
        result = std_numeric(quantity, on_error="raise", allow_bool=False)
        assert result == pytest.approx(quantity.value)
        assert type(result) is float

    @pytest.mark.parametrize(
        ("quantity", "allow_bool", "expected", "expect_error"),
        [
            # Astropy Quantity with boolean value
            pytest.param(u.Quantity(True), True, 1.0, False, id="bool-true-allowed"),
            pytest.param(u.Quantity(False), True, 0.0, False, id="bool-false-allowed"),
            pytest.param(u.Quantity(True), False, None, True, id="bool-true-rejected"),
            pytest.param(u.Quantity(False), False, None, True, id="bool-false-rejected"),
            # Astropy Quantity with numpy bool
            pytest.param(u.Quantity(np.bool_(True)), True, 1.0, False, id="np-bool-true-allowed"),
            pytest.param(
                u.Quantity(np.bool_(False)),
                True,
                0.0,
                False,
                id="np-bool-false-allowed",
            ),
            pytest.param(
                u.Quantity(np.bool_(True)),
                False,
                None,
                True,
                id="np-bool-true-rejected",
            ),
        ],
    )
    def test_astropy_quantity_bool_behavior(
        self, quantity, allow_bool, expected, expect_error
    ) -> None:
        """Handle Astropy Quantity with boolean values based on allow_bool flag."""
        if expect_error:
            with pytest.raises(TypeError, match=r"(?i)boolean values not supported"):
                std_numeric(quantity, on_error="raise", allow_bool=allow_bool)
        else:
            result = std_numeric(quantity, on_error="raise", allow_bool=allow_bool)
            # Astropy converts booleans to float64, so result will be float
            assert type(result) is int
            assert result == expected

    @pytest.mark.parametrize(
        ("array_like", "type_pattern"),
        [
            # Astropy non-scalar quantities (1-D arrays) should be rejected as array-like
            pytest.param(
                u.Quantity([1, 2, 3], u.m),
                r"(?i)(array-like|length)",
                id="quantity-list",
            ),
            pytest.param(
                u.Quantity(np.array([1, 2, 3], dtype=np.int64), u.s),
                r"(?i)(array-like|length)",
                id="quantity-numpy-array",
            ),
            pytest.param(
                u.Quantity(np.array([True, False]), u.one),
                r"(?i)(array-like|length)",
                id="quantity-bool-array",
            ),
            # Multi-dimensional Quantity
            pytest.param(
                u.Quantity(np.array([[1, 2], [3, 4]]), u.kg),
                r"(?i)(array-like|length)",
                id="quantity-2d",
            ),
        ],
    )
    def test_reject_astropy_collections(self, array_like, type_pattern) -> None:
        """Reject Astropy Quantity collections (non-scalars)."""
        with pytest.raises(TypeError, match=type_pattern):
            std_numeric(array_like, on_error="raise", allow_bool=False)


# Ensure SymPy is available for these tests, otherwise skip the whole class


class TestStdNumSympyNumericSupport:
    """
    Core tests for SymPy numeric support in std_numeric.
    """

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(sp.Integer(-5), -5, id="integer-neg"),
            pytest.param(sp.Integer(0), 0, id="integer-zero"),
            pytest.param(sp.Integer(2**100), 2**100, id="integer-big"),
        ],
    )
    def test_sympy_integer_to_int(self, value, expected) -> None:
        """SymPy Integer uses __index__ → int."""
        result = std_numeric(value, on_error="raise", allow_bool=False)
        assert type(result) is int
        assert result == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(sp.Rational(7, 2), 3.5, id="rational-simple"),
            pytest.param(sp.Rational(-9, 4), -2.25, id="rational-negative"),
        ],
    )
    def test_sympy_rational_to_float(self, value, expected) -> None:
        """SymPy Rational uses __float__ → float."""
        result = std_numeric(value, on_error="raise", allow_bool=False)
        assert type(result) is float
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(sp.Float(3.5), 3.5, id="float-plain"),
            pytest.param(sp.Float(-2.25), -2.25, id="float-negative"),
            pytest.param(sp.Float("1.0e100"), 1.0e100, id="float-large"),
        ],
    )
    def test_sympy_float_to_float(self, value, expected) -> None:
        """SymPy Float uses __float__ → float."""
        result = std_numeric(value, on_error="raise", allow_bool=False)
        assert type(result) is float
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("value", "kind"),
        [
            pytest.param(sp.nan, "nan", id="nan"),
            pytest.param(sp.oo, "inf+", id="pos-inf"),
            pytest.param(-sp.oo, "inf-", id="neg-inf"),
        ],
    )
    def test_sympy_specials(self, value, kind) -> None:
        """Pass through special values via float."""
        result = std_numeric(value, on_error="raise", allow_bool=False)
        assert type(result) is float
        if kind == "nan":
            assert math.isnan(result)
        elif kind == "inf+":
            assert math.isinf(result) and result > 0
        elif kind == "inf-":
            assert math.isinf(result) and result < 0
        else:
            raise AssertionError(f"Unexpected kind: {kind}")

    def test_sympy_boolean_behavior(self) -> None:
        """SymPy booleans are Python bool; respect allow_bool flag."""
        with pytest.raises(TypeError, match=r"(?i)boolean.*not supported"):
            std_numeric(sp.true, on_error="raise", allow_bool=False)
        assert std_numeric(sp.false, on_error="raise", allow_bool=True) == 0
        assert std_numeric(sp.true, on_error="raise", allow_bool=True) == 1

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(
                sp.Integer(10) / sp.Integer(3),
                float(sp.Integer(10) / sp.Integer(3)),
                id="expr-rational",
            ),
            pytest.param(sp.sin(sp.pi / 6), 0.5, id="expr-trig-evaluable"),
        ],
    )
    def test_sympy_expressions_floatable(self, value, expected) -> None:
        """Floatable SymPy expressions convert via __float__."""
        result = std_numeric(value, on_error="raise", allow_bool=False)
        assert type(result) is float
        assert result == pytest.approx(expected)

    def test_sympy_symbol_rejected(self) -> None:
        """Non-numeric SymPy objects should be rejected."""
        x = sp.Symbol("x")
        with pytest.raises(TypeError):
            std_numeric(x, on_error="raise", allow_bool=False)
