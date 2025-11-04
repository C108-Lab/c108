"""
Core test suite for validators.py methods with external deps
"""

import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.validators import validate_not_empty, validate_shape

# Integration Tests ----------------------------------------------------------------------------------------------------

pytestmark = pytest.mark.integration

# Optional imports -----------------------------------------------------------------------------------------------------

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
tf = pytest.importorskip("tensorflow")
torch = pytest.importorskip("torch")


class TestValidateNotEmptyIntegration:
    """Integration tests for validate_not_empty() external data types."""

    @pytest.mark.parametrize(
        "array, expected_size",
        [
            pytest.param(np.array([1, 2, 3]), 3, id="numpy_nonempty"),
        ],
    )
    def test_numpy_nonempty(self, array, expected_size):
        """Validate non-empty NumPy array."""
        result = validate_not_empty(array, name="np_array")
        assert result.size == pytest.approx(expected_size)

    def test_numpy_empty_raises(self):
        """Raise ValueError for empty NumPy array."""
        arr = np.array([])
        with pytest.raises(ValueError, match=r"(?i).*must not be empty.*"):
            validate_not_empty(arr, name="empty_np")

    def test_pandas_dataframe_nonempty(self):
        """Validate non-empty Pandas DataFrame."""
        df = pd.DataFrame({"a": [1, 2]})
        result = validate_not_empty(df, name="df")
        assert not result.empty

    def test_pandas_dataframe_empty_raises(self):
        """Raise ValueError for empty Pandas DataFrame."""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match=r"(?i).*must not be empty.*"):
            validate_not_empty(df, name="empty_df")

    def test_pandas_series_nonempty(self):
        """Validate non-empty Pandas Series."""
        s = pd.Series([10, 20])
        result = validate_not_empty(s, name="series")
        assert not result.empty

    def test_torch_tensor_nonempty(self):
        """Validate non-empty PyTorch tensor."""
        t = torch.tensor([1, 2, 3])
        result = validate_not_empty(t, name="tensor")
        assert result.numel() == 3

    def test_torch_tensor_empty_raises(self):
        """Raise ValueError for empty PyTorch tensor."""
        t = torch.tensor([])
        with pytest.raises(ValueError, match=r"(?i).*must not be empty.*"):
            validate_not_empty(t, name="empty_tensor")

    def test_tensorflow_tensor_nonempty(self):
        """Validate non-empty TensorFlow tensor."""
        t = tf.constant([1, 2, 3])
        result = validate_not_empty(t, name="tf_tensor")
        assert int(tf.size(result)) == 3

    def test_tensorflow_tensor_empty_raises(self):
        """Raise ValueError for empty TensorFlow tensor."""
        t = tf.constant([])
        with pytest.raises(ValueError, match=r"(?i).*must not be empty.*"):
            validate_not_empty(t, name="empty_tf_tensor")

    def test_jax_array_nonempty(self):
        """Validate non-empty JAX array."""
        arr = jnp.array([1, 2, 3])
        result = validate_not_empty(arr, name="jax_array")
        assert result.size == 3

    def test_jax_array_empty_raises(self):
        """Raise ValueError for empty JAX array."""
        arr = jnp.array([])
        with pytest.raises(ValueError, match=r"(?i).*must not be empty.*"):
            validate_not_empty(arr, name="empty_jax_array")


class TestValidateShapePandas:
    @pytest.mark.parametrize(
        "data,shape",
        [
            pytest.param(
                {"a": [1, 2], "b": [3, 4]},
                (2, 2),
                id="df_exact_2x2_strict",
            ),
            pytest.param(
                {"x": [10, 20, 30]},
                (3, 1),
                id="df_single_col_as_3x1",
            ),
        ],
    )
    def test_df_pass(self, data, shape):
        """Validate DataFrame shapes and pass."""
        df = pd.DataFrame(data)
        out = validate_shape(df, shape=shape)
        assert out is df

    @pytest.mark.parametrize(
        "data,shape,err_sub",
        [
            pytest.param(
                {"a": [1, 2], "b": [3, 4]},
                (2, 3),
                "Shape mismatch",
                id="df_wrong_cols",
            ),
            pytest.param(
                {"a": [1, 2], "b": [3, 4]},
                (3, 2),
                "Shape mismatch",
                id="df_wrong_rows",
            ),
        ],
    )
    def test_df_fail(self, data, shape, err_sub):
        """Raise on DataFrame shape mismatch."""
        df = pd.DataFrame(data)
        with pytest.raises(ValueError, match=rf"(?i).*{err_sub}.*"):
            validate_shape(df, shape=shape)

    def test_df_non_strict_trailing_match(self):
        """Allow leading batch dims for DataFrame in non-strict mode."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})  # shape (3, 2)
        out = validate_shape(df, shape=("any", 2), strict=False)
        assert out is df

    def test_series_pass(self):
        """Validate Series 1D shape."""
        s = pd.Series([1, 2, 3, 4])  # shape (4,)
        out = validate_shape(s, shape=(4,))
        assert out is s

    def test_series_fail_dims(self):
        """Raise on Series wrong length."""
        s = pd.Series([1, 2, 3])  # shape (3,)
        with pytest.raises(ValueError, match=rf"(?i).*Shape mismatch.*"):
            validate_shape(s, shape=(4,))

    def test_series_non_strict_requires_ndim(self):
        """Require at least ndim in non-strict mode."""
        s = pd.Series([1, 2, 3])  # shape (3,)
        with pytest.raises(
            ValueError, match=rf"(?i).*expected at least 2 dimensions.*"
        ):
            validate_shape(s, shape=("any", "any"), strict=False)

    def test_empty_df(self):
        """Validate empty DataFrame shape (0 rows)."""
        df = pd.DataFrame({"a": [], "b": []})  # shape (0, 2)
        out = validate_shape(df, shape=(0, 2))
        assert out is df

    def test_df_dimension_specific_mismatch_message(self):
        """Include dimension index in mismatch message."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})  # shape (3, 2)
        with pytest.raises(ValueError, match=rf"(?i).*dimension 1: expected 3.*"):
            validate_shape(df, shape=(3, 3))

    @pytest.mark.parametrize(
        "data,shape,err_sub",
        [
            pytest.param(
                {"a": [1, 2], "b": [3, 4]},
                (-1, 2),
                "must be non-negative",
                id="df_negative_rows",
            ),
            pytest.param(
                {"a": [1, 2], "b": [3, 4]},
                (2, -1),
                "must be non-negative",
                id="df_negative_cols",
            ),
            pytest.param(
                {"a": [1, 2], "b": [3, 4]},
                (-1, -1),
                "must be non-negative",
                id="df_both_negative",
            ),
        ],
    )
    def test_df_negative_dimensions(self, data, shape, err_sub):
        """Raise on negative dimensions in shape."""
        df = pd.DataFrame(data)
        with pytest.raises(ValueError, match=rf"(?i).*{err_sub}.*"):
            validate_shape(df, shape=shape)

    @pytest.mark.parametrize(
        "value,shape,allow_scalar,should_pass",
        [
            pytest.param(42, (), True, True, id="scalar_allowed"),
            pytest.param(42, (), False, False, id="scalar_disallowed"),
            pytest.param(42, (1,), True, False, id="scalar_wrong_shape"),
        ],
    )
    def test_scalar_allowed_and_disallowed(
        self, value, shape, allow_scalar, should_pass
    ):
        """Validate scalar acceptance and rejection."""
        if should_pass:
            out = validate_shape(value, shape=shape, allow_scalar=allow_scalar)
            assert out == value
        else:
            with pytest.raises(ValueError, match=r"(?i).*scalar.*"):
                validate_shape(value, shape=shape, allow_scalar=allow_scalar)

    @pytest.mark.parametrize(
        "data,shape,should_pass",
        [
            pytest.param([[1, 2], [3, 4]], (2, 2), True, id="list_2x2_pass"),
            pytest.param([[1, 2], [3, 4, 5]], (2, 2), False, id="list_irregular_fail"),
            pytest.param([], (0,), True, id="empty_list_1d_pass"),
            pytest.param([], ("any",), True, id="empty_list_any_1d_pass"),
            pytest.param([[]], (1, 0), True, id="empty_nested_list_pass"),
            pytest.param([[]], (1, "any"), True, id="empty_nested_list_any_pass"),
            pytest.param([[]], ("any", "any"), True, id="empty_nested_list_any_any_pass"),
            pytest.param([[1, 2], [3, 4]], ("any", "any"), True, id="list_2x2_any_any_pass"),
            pytest.param([[[1]], [[2]]], ("any", "any", "any"), True, id="list_3d_any_any_any_pass"),
        ],
    )
    def test_list_of_lists_shape(self, data, shape, should_pass):
        """Validate nested list shapes."""
        if should_pass:
            out = validate_shape(data, shape=shape)
            assert out is data
        else:
            with pytest.raises(ValueError, match=r"(?i)shape mismatch|inconsistent shapes"):
                validate_shape(data, shape=shape)

    @pytest.mark.parametrize(
        "array,shape,strict,should_pass",
        [
            pytest.param(np.zeros((3, 2)), (3, 2), True, True, id="numpy_exact_pass"),
            pytest.param(
                np.zeros((3, 2)), (2, 3), True, False, id="numpy_mismatch_fail"
            ),
            pytest.param(
                np.zeros((5, 3, 2)), (3, 2), False, True, id="numpy_non_strict_pass"
            ),
            pytest.param(
                np.zeros((5, 3, 2)), (3, 2), True, False, id="numpy_strict_fail"
            ),
        ],
    )
    def test_numpy_array_strict_and_non_strict(self, array, shape, strict, should_pass):
        """Validate numpy arrays with strict and non-strict modes."""
        if should_pass:
            out = validate_shape(array, shape=shape, strict=strict)
            assert np.array_equal(out, array)
        else:
            with pytest.raises(ValueError, match=r"(?i).*shape mismatch.*"):
                validate_shape(array, shape=shape, strict=strict)

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((3, "foo"), id="invalid_literal"),
            pytest.param((3.5, 2), id="float_dimension"),
            pytest.param(("any", -1), id="negative_with_any"),
        ],
    )
    def test_invalid_shape_spec(self, shape):
        """Raise on invalid shape specification."""
        arr = np.zeros((3, 2))
        with pytest.raises(
            (TypeError, ValueError), match=r"(?i).*invalid|non-negative.*"
        ):
            validate_shape(arr, shape=shape)

    @pytest.mark.parametrize(
        "obj",
        [
            pytest.param("not an array", id="string_input"),
            pytest.param({"a": 1}, id="dict_input"),
            pytest.param(object(), id="plain_object"),
        ],
    )
    def test_unsupported_type_raises(self, obj):
        """Raise TypeError for unsupported input types."""
        with pytest.raises(TypeError, match=r"(?i).*array-like.*"):
            validate_shape(obj, shape=(1,))
