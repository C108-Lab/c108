"""
Core test suite for validators.py methods with external deps
"""

import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.validators import validate_not_empty


# Integration Tests ----------------------------------------------------------------------------------------------------

pytestmark = pytest.mark.integration

# Optional imports -----------------------------------------------------------------------------------------------------

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
sp = pytest.importorskip("sympy")
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
