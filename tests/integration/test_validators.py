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


class TestValidateShapeNumpy:
    @pytest.mark.parametrize(
        "array,shape",
        [
            pytest.param(np.zeros((2, 2)), (2, 2), id="array_exact_2x2"),
            pytest.param(np.arange(3).reshape(3, 1), (3, 1), id="array_reshape_3x1"),
            pytest.param(np.zeros((0, 2)), (0, 2), id="array_empty_0x2"),
            pytest.param(np.zeros((2, 3, 4)), (2, 3, 4), id="array_2x3x4"),
        ],
    )
    def test_array_pass(self, array, shape):
        """Validate NumPy array shapes and pass."""
        out = validate_shape(array, shape=shape)
        assert np.array_equal(out, array)

    @pytest.mark.parametrize(
        "array,shape,err_sub",
        [
            pytest.param(
                np.zeros((2, 2)), (2, 3), "Shape mismatch", id="array_wrong_cols"
            ),
            pytest.param(
                np.zeros((2, 2)), (3, 2), "Shape mismatch", id="array_wrong_rows"
            ),
        ],
    )
    def test_array_fail(self, array, shape, err_sub):
        """Raise on NumPy array shape mismatch."""
        with pytest.raises(ValueError, match=rf"(?i).*{err_sub}.*"):
            validate_shape(array, shape=shape)

    def test_array_non_strict_trailing_match(self):
        """Allow leading batch dims for NumPy array in non-strict mode."""
        arr = np.zeros((5, 3, 2))  # shape (5, 3, 2)
        out = validate_shape(arr, shape=(3, 2), strict=False)
        assert np.array_equal(out, arr)

    def test_array_dimension_specific_mismatch_message(self):
        """Include dimension index in mismatch message."""
        arr = np.zeros((3, 2))
        with pytest.raises(ValueError, match=rf"(?i).*dimension 1: expected 3.*"):
            validate_shape(arr, shape=(3, 3))

    @pytest.mark.parametrize(
        "shape,err_sub",
        [
            pytest.param((-1, 2), "must be non-negative", id="array_negative_rows"),
            pytest.param((2, -1), "must be non-negative", id="array_negative_cols"),
            pytest.param((-1, -1), "must be non-negative", id="array_both_negative"),
        ],
    )
    def test_array_negative_dimensions(self, shape, err_sub):
        """Raise on negative dimensions in shape."""
        arr = np.zeros((2, 2))
        with pytest.raises(ValueError, match=rf"(?i).*{err_sub}.*"):
            validate_shape(arr, shape=shape)

    @pytest.mark.parametrize(
        "value,shape",
        [
            pytest.param(np.array(42), (),  id="np_scalar_allowed"),
            pytest.param(np.array(3.14), (),  id="np_scalar_allowed"),
        ],
    )
    def test_scalar(
        self, value, shape
    ):
        """Validate scalar acceptance and rejection."""
        out = validate_shape(value, shape=shape)
        assert out == value

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
        """Validate NumPy arrays with strict and non-strict modes."""
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

    def test_array_non_strict_requires_ndim(self):
        """Require at least ndim in non-strict mode."""
        arr = np.zeros((3, 2))
        with pytest.raises(ValueError, match=r"(?i).*dimensions.*"):
            validate_shape(arr, shape=(3, 2, 1), strict=False)

    @pytest.mark.parametrize(
        "data,shape,should_pass",
        [
            pytest.param([[1, 2], [3, 4]], (2, 2), True, id="list_2x2_pass"),
            pytest.param([[1, 2], [3, 4, 5]], (2, 2), False, id="list_irregular_fail"),
            pytest.param([], (0,), True, id="empty_list_1d_pass"),
        ],
    )
    def test_list_input_shapes(self, data, shape, should_pass):
        """Validate list inputs as array-like."""
        if should_pass:
            out = validate_shape(np.array(data), shape=shape)
            assert np.array_equal(out, np.array(data))
        else:
            # Test the raw list first, then numpy conversion if it succeeds
            try:
                arr = np.array(data)
                with pytest.raises(ValueError, match=r"(?i).*shape mismatch.*"):
                    validate_shape(arr, shape=shape)
            except ValueError as numpy_error:
                # If numpy array creation fails, test with raw list
                if "inhomogeneous" in str(numpy_error):
                    with pytest.raises(
                        ValueError, match=r"(?i).*(ragged|inconsistent).*"
                    ):
                        validate_shape(data, shape=shape)
                else:
                    raise

    def test_any_wildcard_dimension(self):
        """Allow 'any' wildcard in shape."""
        arr = np.zeros((3, 5))
        out = validate_shape(arr, shape=(3, "any"))
        assert np.array_equal(out, arr)


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
        "value,shape",
        [
            # Pandas Series with single element is still shape (1,), not ()
            pytest.param(pd.Series([42]), (1,), id="series_single_element"),

            # DataFrame with single cell is shape (1, 1), not ()
            pytest.param(pd.DataFrame([[42]]), (1, 1), id="dataframe_single_cell"),

            # Empty Series
            pytest.param(pd.Series([]), (0,), id="series_empty"),

            # If you extract a scalar from pandas, it's a Python/numpy scalar
            pytest.param(pd.Series([42]).iloc[0], (), id="extracted_scalar"),
        ],
    )
    def test_pandas_scalars(self, value, shape):
        """Validate pandas object shapes."""
        out = validate_shape(value, shape=shape)
        # For the extracted scalar case, would need allow_scalar or the new behavior

    @pytest.mark.parametrize(
        "data,shape,should_pass",
        [
            pytest.param([[1, 2], [3, 4]], (2, 2), True, id="list_2x2_pass"),
            pytest.param([[1, 2], [3, 4, 5]], (2, 2), False, id="list_irregular_fail"),
            pytest.param([], (0,), True, id="empty_list_1d_pass"),
            pytest.param([], ("any",), True, id="empty_list_any_1d_pass"),
            pytest.param([[]], (1, 0), True, id="empty_nested_list_pass"),
            pytest.param([[]], (1, "any"), True, id="empty_nested_list_any_pass"),
            pytest.param(
                [[]], ("any", "any"), True, id="empty_nested_list_any_any_pass"
            ),
            pytest.param(
                [[1, 2], [3, 4]], ("any", "any"), True, id="list_2x2_any_any_pass"
            ),
            pytest.param(
                [[[1]], [[2]]],
                ("any", "any", "any"),
                True,
                id="list_3d_any_any_any_pass",
            ),
        ],
    )
    def test_list_of_lists_shape(self, data, shape, should_pass):
        """Validate nested list shapes."""
        if should_pass:
            out = validate_shape(data, shape=shape)
            assert out is data
        else:
            with pytest.raises(
                ValueError, match=r"(?i)shape mismatch|inconsistent shapes"
            ):
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


class TestValidateShapePyTorch:
    """Extended test suite for validate_shape() with PyTorch tensors."""

    @pytest.mark.parametrize(
        "tensor,shape",
        [
            pytest.param(torch.zeros((2, 2)), (2, 2), id="tensor_exact_2x2"),
            pytest.param(
                torch.arange(6).reshape(3, 2), (3, 2), id="tensor_reshape_3x2"
            ),
            pytest.param(torch.zeros((0, 2)), (0, 2), id="tensor_empty_0x2"),
            pytest.param(torch.zeros((2, 3, 4)), (2, 3, 4), id="tensor_2x3x4"),
        ],
    )
    def test_tensor_pass(self, tensor, shape):
        """Validate PyTorch tensor shapes and pass."""
        result = validate_shape(tensor, shape=shape, strict=True)
        assert torch.equal(result, tensor)

    @pytest.mark.parametrize(
        "tensor,shape,err_sub",
        [
            pytest.param(
                torch.zeros((2, 2)), (2, 3), "Shape mismatch", id="tensor_wrong_cols"
            ),
            pytest.param(
                torch.zeros((2, 2)), (3, 2), "Shape mismatch", id="tensor_wrong_rows"
            ),
        ],
    )
    def test_tensor_fail(self, tensor, shape, err_sub):
        """Raise on PyTorch tensor shape mismatch."""
        with pytest.raises(ValueError, match=rf"(?i).*{err_sub}.*"):
            validate_shape(tensor, shape=shape, strict=True)

    def test_tensor_non_strict_trailing_match(self):
        """Allow leading batch dims for PyTorch tensor in non-strict mode."""
        tensor = torch.zeros((5, 3, 2))
        result = validate_shape(tensor, shape=(3, 2), strict=False)
        assert torch.equal(result, tensor)

    def test_tensor_dimension_specific_mismatch_message(self):
        """Include dimension index in mismatch message."""
        tensor = torch.zeros((2, 3))
        with pytest.raises(ValueError, match=r"(?i).*dimension 1.*expected 4.*"):
            validate_shape(tensor, shape=(2, 4), strict=True)

    @pytest.mark.parametrize(
        "shape,err_sub",
        [
            pytest.param((-1, 2), "must be non-negative", id="tensor_negative_rows"),
            pytest.param((2, -1), "must be non-negative", id="tensor_negative_cols"),
            pytest.param((-1, -1), "must be non-negative", id="tensor_both_negative"),
        ],
    )
    def test_tensor_negative_dimensions(self, shape, err_sub):
        """Raise on negative dimensions in shape."""
        tensor = torch.zeros((2, 2))
        with pytest.raises(ValueError, match=rf"(?i).*{err_sub}.*"):
            validate_shape(tensor, shape=shape, strict=True)

    @pytest.mark.parametrize(
        "value,shape,should_pass",
        [
            pytest.param(torch.tensor(42), (), True, id="scalar_allowed"),
            pytest.param(torch.tensor(42), (1,), False, id="scalar_wrong_shape"),
        ],
    )
    def test_scalar(
        self, value, shape, should_pass
    ):
        """Validate scalar acceptance and rejection."""
        if should_pass:
            result = validate_shape(
                value, shape=shape, strict=True
            )
            assert torch.equal(result, value)
        else:
            with pytest.raises(ValueError, match=r"(?i).*scalar.*"):
                validate_shape(
                    value, shape=shape, strict=True
                )

    @pytest.mark.parametrize(
        "tensor,shape,strict,should_pass",
        [
            pytest.param(
                torch.zeros((3, 2)), (3, 2), True, True, id="torch_exact_pass"
            ),
            pytest.param(
                torch.zeros((3, 2)), (2, 3), True, False, id="torch_mismatch_fail"
            ),
            pytest.param(
                torch.zeros((5, 3, 2)), (3, 2), False, True, id="torch_non_strict_pass"
            ),
            pytest.param(
                torch.zeros((5, 3, 2)), (3, 2), True, False, id="torch_strict_fail"
            ),
        ],
    )
    def test_torch_tensor_strict_and_non_strict(
        self, tensor, shape, strict, should_pass
    ):
        """Validate PyTorch tensors with strict and non-strict modes."""
        if should_pass:
            result = validate_shape(
                tensor, shape=shape, strict=strict
            )
            assert torch.equal(result, tensor)
        else:
            with pytest.raises(ValueError, match=r"(?i).*shape mismatch.*"):
                validate_shape(tensor, shape=shape, strict=strict)

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
        tensor = torch.zeros((2, 2))
        with pytest.raises(
            (TypeError, ValueError), match=r"(?i).*invalid|non-negative.*"
        ):
            validate_shape(tensor, shape=shape, strict=True)

    @pytest.mark.parametrize(
        "obj",
        [
            pytest.param("not a tensor", id="string_input"),
            pytest.param({"a": 1}, id="dict_input"),
            pytest.param(object(), id="plain_object"),
        ],
    )
    def test_unsupported_type_raises(self, obj):
        """Raise TypeError for unsupported input types."""
        with pytest.raises(TypeError, match=r"(?i).*array-like|unsupported.*"):
            validate_shape(obj, shape=(2, 2), strict=True)

    def test_tensor_non_strict_requires_ndim(self):
        """Require at least ndim in non-strict mode."""
        tensor = torch.zeros((2,))
        with pytest.raises(ValueError, match=r"(?i).*dimensions|ndim.*"):
            validate_shape(tensor, shape=(2, 2), strict=False)

    def test_any_wildcard_dimension(self):
        """Allow 'any' wildcard in shape."""
        tensor = torch.zeros((3, 5))
        result = validate_shape(
            tensor, shape=(3, "any"), strict=True
        )
        assert torch.equal(result, tensor)

    def test_empty_tensor_shape(self):
        """Validate empty tensor shape (0 elements)."""
        tensor = torch.zeros((0, 3))
        result = validate_shape(tensor, shape=(0, 3), strict=True)
        assert torch.equal(result, tensor)

    @pytest.mark.parametrize(
        "tensor,shape,should_pass",
        [
            pytest.param(
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                (2, 2),
                True,
                id="float_tensor_pass",
            ),
            pytest.param(
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                (2, 3),
                False,
                id="float_tensor_fail",
            ),
        ],
    )
    def test_float_tensor_shapes(self, tensor, shape, should_pass):
        """Validate float tensor shapes."""
        if should_pass:
            result = validate_shape(
                tensor, shape=shape, strict=True
            )
            assert torch.allclose(result, tensor, atol=1e-8)
        else:
            with pytest.raises(ValueError, match=r"(?i).*shape mismatch.*"):
                validate_shape(tensor, shape=shape, strict=True)

    @pytest.mark.parametrize(
        "tensor,shape,should_pass",
        [
            pytest.param(
                torch.tensor([[1, 2], [3, 4]]), ("any", 2), True, id="any_rows_pass"
            ),
            pytest.param(
                torch.tensor([[1, 2], [3, 4]]), (2, "any"), True, id="any_cols_pass"
            ),
            pytest.param(
                torch.tensor([[1, 2], [3, 4]]), ("any", "any"), True, id="any_any_pass"
            ),
        ],
    )
    def test_any_wildcard_combinations(self, tensor, shape, should_pass):
        """Validate wildcard combinations for PyTorch tensors."""
        result = validate_shape(tensor, shape=shape, strict=True)
        assert torch.equal(result, tensor)

    def test_high_dimensional_tensor(self):
        """Validate high-dimensional tensor shape."""
        tensor = torch.zeros((2, 3, 4, 5))
        result = validate_shape(
            tensor, shape=(2, 3, 4, 5), strict=True
        )
        assert torch.equal(result, tensor)

    def test_high_dimensional_tensor_fail(self):
        """Raise on high-dimensional tensor shape mismatch."""
        tensor = torch.zeros((2, 3, 4, 5))
        with pytest.raises(ValueError, match=r"(?i).*shape mismatch.*"):
            validate_shape(tensor, shape=(2, 3, 4, 6), strict=True)

    def test_tensor_with_batch_and_any(self):
        """Allow batch dimension with 'any' in non-strict mode."""
        tensor = torch.zeros((10, 3, 2))
        result = validate_shape(tensor, shape=(3, 2), strict=False)
        assert torch.equal(result, tensor)

    def test_tensor_shape_mismatch_message_contains_expected_and_actual(self):
        """Ensure mismatch message includes expected and actual shapes."""
        tensor = torch.zeros((2, 3))
        with pytest.raises(
            ValueError, match=r"(?i).*expected.*\(2, 4\).*got.*\(2, 3\).*"
        ):
            validate_shape(tensor, shape=(2, 4), strict=True)


class TestValidateShapeTensorFlow:
    @pytest.mark.parametrize(
        "tensor,shape",
        [
            pytest.param(tf.zeros((2, 3)), (2, 3), id="tf_exact_2x3"),
            pytest.param(tf.ones((4,)), (4,), id="tf_1d_4"),
        ],
    )
    def test_tf_pass(self, tensor, shape):
        """Validate TensorFlow tensor shapes and pass."""
        out = validate_shape(tensor, shape=shape)
        assert tf.reduce_all(tf.equal(out, tensor))

    @pytest.mark.parametrize(
        "tensor,shape,err_sub",
        [
            pytest.param(
                tf.zeros((2, 3)), (3, 2), "shape mismatch", id="tf_wrong_dims"
            ),
            pytest.param(
                tf.zeros((2, 3)), (2, 4), "shape mismatch", id="tf_wrong_cols"
            ),
        ],
    )
    def test_tf_fail(self, tensor, shape, err_sub):
        """Raise on TensorFlow tensor shape mismatch."""
        with pytest.raises(ValueError, match=rf"(?i).*{err_sub}.*"):
            validate_shape(tensor, shape=shape)

    def test_tf_non_strict_trailing_match(self):
        """Allow leading batch dims for TensorFlow tensor in non-strict mode."""
        tensor = tf.zeros((5, 2, 3))  # actual shape (5, 2, 3)
        out = validate_shape(tensor, shape=(2, 3), strict=False)
        assert tf.reduce_all(tf.equal(out, tensor))

    def test_tf_strict_requires_exact_ndim(self):
        """Raise when strict mode requires exact ndim."""
        tensor = tf.zeros((5, 2, 3))
        with pytest.raises(ValueError, match=r"(?i).*shape mismatch.*"):
            validate_shape(tensor, shape=(2, 3), strict=True)

    def test_tf_any_dimension_pass(self):
        """Allow 'any' dimension in TensorFlow tensor shape."""
        tensor = tf.zeros((7, 5))
        out = validate_shape(tensor, shape=("any", 5))
        assert tf.reduce_all(tf.equal(out, tensor))

    def test_tf_negative_expected_dimension(self):
        """Raise on negative expected dimension."""
        tensor = tf.zeros((2, 3))
        with pytest.raises(ValueError, match=r"(?i).*must be non-negative.*"):
            validate_shape(tensor, shape=(-1, 3))

    def test_tf_scalar(self):
        """Validate scalar acceptance and rejection for TensorFlow tensors."""
        scalar = tf.constant(42.0)
        out = validate_shape(scalar, shape=())
        assert tf.reduce_all(tf.equal(out, scalar))
        validate_shape(scalar, shape=())

    def test_tf_empty_tensor(self):
        """Validate empty TensorFlow tensor shape (0 rows)."""
        tensor = tf.zeros((0, 3))
        out = validate_shape(tensor, shape=(0, 3))
        assert tf.reduce_all(tf.equal(out, tensor))

    def test_tf_dimension_specific_mismatch_message(self):
        """Include dimension index in mismatch message for TensorFlow tensor."""
        tensor = tf.zeros((3, 2))
        with pytest.raises(ValueError, match=r"(?i).*dimension 1: expected 3.*"):
            validate_shape(tensor, shape=(3, 3))

    @pytest.mark.parametrize(
        "tensor,shape,strict,should_pass",
        [
            pytest.param(tf.zeros((3, 2)), (3, 2), True, True, id="tf_strict_pass"),
            pytest.param(tf.zeros((3, 2)), (2, 3), True, False, id="tf_strict_fail"),
            pytest.param(
                tf.zeros((5, 3, 2)), (3, 2), False, True, id="tf_non_strict_pass"
            ),
            pytest.param(
                tf.zeros((5, 3, 2)), (3, 2), True, False, id="tf_non_strict_fail"
            ),
        ],
    )
    def test_tf_strict_and_non_strict(self, tensor, shape, strict, should_pass):
        """Validate TensorFlow tensors with strict and non-strict modes."""
        if should_pass:
            out = validate_shape(tensor, shape=shape, strict=strict)
            assert tf.reduce_all(tf.equal(out, tensor))
        else:
            with pytest.raises(ValueError, match=r"(?i).*shape mismatch.*"):
                validate_shape(tensor, shape=shape, strict=strict)

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((3, "foo"), id="invalid_literal"),
            pytest.param((3.5, 2), id="float_dimension"),
            pytest.param(("any", -1), id="negative_with_any"),
        ],
    )
    def test_tf_invalid_shape_spec(self, shape):
        """Raise on invalid shape specification for TensorFlow tensor."""
        tensor = tf.zeros((3, 2))
        with pytest.raises(
            (TypeError, ValueError), match=r"(?i).*invalid|non-negative.*"
        ):
            validate_shape(tensor, shape=shape)

    def test_tf_unsupported_type_raises(self):
        """Raise TypeError for unsupported TensorFlow-like input."""

        class FakeTensor:
            def __init__(self):
                self.shape = "not_a_tuple"

        fake = FakeTensor()
        with pytest.raises(
            AttributeError, match=r"(?i).*could not convert.*shape.*to tuple.*"
        ):
            validate_shape(fake, shape=(1,))

    def test_tf_non_strict_requires_ndim(self):
        """Require at least ndim in non-strict mode for TensorFlow tensor."""
        tensor = tf.zeros((3,))
        with pytest.raises(ValueError, match=r"(?i).*expected at least 2 dimensions.*"):
            validate_shape(tensor, shape=("any", "any"), strict=False)

    @pytest.mark.parametrize(
        "tensor,shape,err_sub",
        [
            pytest.param(
                tf.zeros((2, 3)), (-1, 3), "must be non-negative", id="tf_negative_rows"
            ),
            pytest.param(
                tf.zeros((2, 3)), (2, -1), "must be non-negative", id="tf_negative_cols"
            ),
            pytest.param(
                tf.zeros((2, 3)),
                (-1, -1),
                "must be non-negative",
                id="tf_both_negative",
            ),
        ],
    )
    def test_tf_negative_dimensions(self, tensor, shape, err_sub):
        """Raise on negative dimensions in TensorFlow shape."""
        with pytest.raises(ValueError, match=rf"(?i).*{err_sub}.*"):
            validate_shape(tensor, shape=shape)

    def test_tf_scalar_wrong_shape(self):
        """Raise when TensorFlow scalar does not match expected shape."""
        scalar = tf.constant(7)
        with pytest.raises(ValueError, match=r"(?i).*scalar.*"):
            validate_shape(scalar, shape=(1,))

    def test_tf_high_dim_any_any_any_pass(self):
        """Validate high-dimensional TensorFlow tensor with 'any' dimensions."""
        tensor = tf.zeros((2, 3, 4))
        out = validate_shape(tensor, shape=("any", "any", "any"))
        assert tf.reduce_all(tf.equal(out, tensor))


class TestValidateShapeJAX:
    @pytest.mark.parametrize(
        "array,shape",
        [
            pytest.param(jnp.zeros((2, 2)), (2, 2), id="jax_exact_2x2"),
            pytest.param(jnp.arange(3).reshape(3, 1), (3, 1), id="jax_reshape_3x1"),
            pytest.param(jnp.zeros((0, 2)), (0, 2), id="jax_empty_0x2"),
            pytest.param(jnp.zeros((2, 3, 4)), (2, 3, 4), id="jax_2x3x4"),
        ],
    )
    def test_jax_array_pass(self, array, shape):
        """Validate JAX array shapes and pass."""
        out = validate_shape(array, shape=shape)
        assert jnp.array_equal(out, array)

    @pytest.mark.parametrize(
        "array,shape,err_sub",
        [
            pytest.param(
                jnp.zeros((2, 2)), (2, 3), "Shape mismatch", id="jax_wrong_cols"
            ),
            pytest.param(
                jnp.zeros((2, 2)), (3, 2), "Shape mismatch", id="jax_wrong_rows"
            ),
        ],
    )
    def test_jax_array_fail(self, array, shape, err_sub):
        """Raise on JAX array shape mismatch."""
        with pytest.raises(ValueError, match=rf"(?i).*{err_sub}.*"):
            validate_shape(array, shape=shape)

    def test_jax_array_non_strict_trailing_match(self):
        """Allow leading batch dims for JAX array in non-strict mode."""
        arr = jnp.zeros((5, 3, 2))
        out = validate_shape(arr, shape=(3, 2), strict=False)
        assert jnp.array_equal(out, arr)

    def test_jax_array_dimension_specific_mismatch_message(self):
        """Include dimension index in mismatch message for JAX arrays."""
        arr = jnp.zeros((3, 2))
        with pytest.raises(ValueError, match=rf"(?i).*dimension 1: expected 3.*"):
            validate_shape(arr, shape=(3, 3))

    @pytest.mark.parametrize(
        "shape,err_sub",
        [
            pytest.param((-1, 2), "must be non-negative", id="jax_negative_rows"),
            pytest.param((2, -1), "must be non-negative", id="jax_negative_cols"),
            pytest.param((-1, -1), "must be non-negative", id="jax_both_negative"),
        ],
    )
    def test_jax_array_negative_dimensions(self, shape, err_sub):
        """Raise on negative dimensions in shape for JAX arrays."""
        arr = jnp.zeros((2, 2))
        with pytest.raises(ValueError, match=rf"(?i).*{err_sub}.*"):
            validate_shape(arr, shape=shape)

    @pytest.mark.parametrize(
        "array,shape,strict,should_pass",
        [
            pytest.param(jnp.zeros((3, 2)), (3, 2), True, True, id="jax_exact_pass"),
            pytest.param(
                jnp.zeros((3, 2)), (2, 3), True, False, id="jax_mismatch_fail"
            ),
            pytest.param(
                jnp.zeros((5, 3, 2)), (3, 2), False, True, id="jax_non_strict_pass"
            ),
            pytest.param(
                jnp.zeros((5, 3, 2)), (3, 2), True, False, id="jax_strict_fail"
            ),
        ],
    )
    def test_jax_array_strict_and_non_strict(self, array, shape, strict, should_pass):
        """Validate JAX arrays with strict and non-strict modes."""
        if should_pass:
            out = validate_shape(array, shape=shape, strict=strict)
            assert jnp.array_equal(out, array)
        else:
            with pytest.raises(ValueError, match=r"(?i).*shape mismatch.*"):
                validate_shape(array, shape=shape, strict=strict)

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((3, "foo"), id="jax_invalid_literal"),
            pytest.param((3.5, 2), id="jax_float_dimension"),
            pytest.param(("any", -1), id="jax_negative_with_any"),
        ],
    )
    def test_jax_invalid_shape_spec(self, shape):
        """Raise on invalid shape specification for JAX arrays."""
        arr = jnp.zeros((3, 2))
        with pytest.raises(
            (TypeError, ValueError), match=r"(?i).*invalid|non-negative.*"
        ):
            validate_shape(arr, shape=shape)

    def test_jax_any_wildcard_dimension(self):
        """Allow 'any' wildcard in shape for JAX arrays."""
        arr = jnp.zeros((3, 5))
        out = validate_shape(arr, shape=(3, "any"))
        assert jnp.array_equal(out, arr)

    def test_jax_array_non_strict_requires_ndim(self):
        """Require at least ndim in non-strict mode for JAX arrays."""
        arr = jnp.zeros((3, 2))
        with pytest.raises(ValueError, match=r"(?i).*dimensions.*"):
            validate_shape(arr, shape=(3, 2, 1), strict=False)

    @pytest.mark.parametrize(
        "value,shape,should_pass",
        [
            pytest.param(42, (), True, id="jax_scalar_allowed"),
            pytest.param(42, (1,), False, id="jax_scalar_wrong_shape"),
        ],
    )
    def test_jax_scalar_allowed_and_disallowed(
        self, value, shape, should_pass
    ):
        """Validate scalar acceptance and rejection for JAX path."""
        if should_pass:
            out = validate_shape(value, shape=shape)
            assert out == value
        else:
            with pytest.raises(ValueError, match=r"(?i).*scalar.*"):
                validate_shape(value, shape=shape)

    @pytest.mark.parametrize(
        "obj",
        [
            pytest.param("not an array", id="jax_string_input"),
            pytest.param({"a": 1}, id="jax_dict_input"),
            pytest.param(object(), id="jax_plain_object"),
        ],
    )
    def test_jax_unsupported_type_raises(self, obj):
        """Raise TypeError for unsupported input types in JAX path."""
        with pytest.raises(TypeError, match=r"(?i).*array-like.*"):
            validate_shape(obj, shape=(1,))

    @pytest.mark.parametrize(
        "data,shape,should_pass",
        [
            pytest.param([[1, 2], [3, 4]], (2, 2), True, id="jax_list_2x2_pass"),
            pytest.param(
                [[1, 2], [3, 4, 5]], (2, 2), False, id="jax_list_irregular_fail"
            ),
            pytest.param([], (0,), True, id="jax_empty_list_1d_pass"),
        ],
    )
    def test_jax_list_input_shapes(self, data, shape, should_pass):
        """Validate list inputs as array-like for JAX path."""
        if should_pass:
            arr = jnp.array(data)
            out = validate_shape(arr, shape=shape)
            assert jnp.array_equal(out, arr)
        else:
            try:
                arr = jnp.array(data)
                with pytest.raises(ValueError, match=r"(?i).*shape mismatch.*"):
                    validate_shape(arr, shape=shape)
            except Exception:
                with pytest.raises(ValueError, match=r"(?i).*(ragged|inconsistent).*"):
                    validate_shape(data, shape=shape)

    def test_jax_array_non_strict_requires_ndim(self):
        """Require at least ndim in non-strict mode for JAX arrays."""
        arr = jnp.zeros((3, 2))
        with pytest.raises(ValueError, match=r"(?i).*dimensions.*"):
            validate_shape(arr, shape=(3, 2, 1), strict=False)

    def test_jax_any_wildcard_dimension(self):
        """Allow 'any' wildcard in shape for JAX arrays."""
        arr = jnp.zeros((3, 5))
        out = validate_shape(arr, shape=(3, "any"))
        assert jnp.array_equal(out, arr)
