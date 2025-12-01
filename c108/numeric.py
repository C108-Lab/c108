"""
Standardize numeric types from Python stdlib and third-party libraries.

This module provides robust numeric type conversion suitable for display
formatting, data processing, and cross-library interoperability.
"""

# Standard library -----------------------------------------------------------------------------------------------------

import operator
from typing import Any, Literal


# Local ----------------------------------------------------------------------------------------------------------------


def std_numeric(
    value: Any,
    *,
    on_error: Literal["raise", "nan", "none"] = "raise",
    allow_bool: bool = False,
) -> int | float | None:
    """
    Convert numeric types to standardPython int or float.

    Normalizes numeric values from Python stdlib and third-party libraries
    (NumPy, Pandas, Decimal, etc.) into standardPython types for display,
    serialization, and processing.

    Design Philosophy
    -----------------
    This function provides **transparent type normalization** without heuristics:
    - Types with __index__() → int (exact integer types)
    - Types with __float__() → float (all other numeric types)
    - No "smart" conversions based on numeric values

    The source type's __float__() method handles overflow/underflow:
    - Overflow: values too large become inf/-inf
    - Underflow: values too small become 0.0/-0.0
    - Special values (nan, inf) pass through unchanged

    Parameters
    ----------
    value : various
        Numeric value to convert. Supports Python int/float/None, Decimal,
        Fraction, and third-party types via __index__, __float__, .item(),
        or .value protocols.

    on_error : {"raise", "nan", "none"}, default "raise"
        How to handle TYPE ERRORS (unsupported types like str, list, dict):
        - "raise": Raise TypeError (default)
        - "nan": Return float('nan')
        - "none": Return None

        Note: Numeric edge cases (inf, nan) are valid values, not errors.

    allow_bool : bool, default False
        If True, convert bool to int (True→1, False→0). If False, treat
        bool as type error. Default False helps catch bugs since bool is
        a subclass of int in Python.

    Returns
    -------
    int
        For Python int (arbitrary precision) or types implementing __index__
        (NumPy integers, etc.).

    float
        For all other numeric types via __float__(), including special IEEE 754
        values (inf, -inf, nan, 0.0, -0.0).

    None
        For None input, pandas.NA, numpy.ma.masked, or type errors when
        on_error="none".

    Raises
    ------
    TypeError
        When on_error="raise" and value is unsupported type or bool when
        allow_bool=False.

    Type Conversion Rules
    ---------------------
    **Integer types** (via __index__):
        Python int, NumPy int8-64/uint8-64 → int (arbitrary precision)

    **Float types** (via __float__):
        Python float, Decimal, Fraction, NumPy float16-128 → float

    **Special values** (pass through as float):
        inf, -inf, nan → preserved unchanged

    **Overflow/underflow** (handled by source type's __float__):
        Decimal('1e400').__float__() → inf
        Decimal('1e-400').__float__() → 0.0

    **Array scalars** (via .item()):
        NumPy/PyTorch/TensorFlow/JAX tensor scalars extracted then converted

    **Physical quantities** (via .value):
        Astropy Quantity → extract .value, discard units, convert

    **Missing data**:
        None → None
        pandas.NA → nan
        numpy.ma.masked → nan

    Common Types Supported
    ----------------------
    - Python: int, float, None, Decimal, Fraction
    - NumPy: int/uint/float scalars, nan, inf, arrays via .item()
    - Pandas: numeric scalars, pd.NA
    - ML: PyTorch/TensorFlow/JAX scalars via .item()/.numpy()
    - Scientific: Astropy Quantity (via .value)
    - Any type with __float__() or __index__()

    Examples
    --------
    Basic types:
    >>> std_numeric(42)
    42
    >>> std_numeric(3.14)
    3.14
    >>> std_numeric(None) # Returns None

    Decimal and Fraction (always become float):
    >>> from decimal import Decimal
    >>> std_numeric(Decimal('42'))
    42.0
    >>> std_numeric(Decimal('3.14'))
    3.14

    Overflow and underflow (handled by __float__):
    >>> std_numeric(Decimal('1e400'))
    inf
    >>> std_numeric(Decimal('1e-400'))
    0.0

    Special values:
    >>> std_numeric(float('inf'))
    inf
    >>> std_numeric(float('nan'))
    nan

    Error handling:
    >>> std_numeric("invalid")
    Traceback (most recent call last):
        ...
    TypeError: unsupported numeric type: str
    >>> std_numeric("invalid", on_error="nan")
    nan

    Boolean handling:
    >>> std_numeric(True)
    Traceback (most recent call last):
        ...
    TypeError: boolean values not supported, got True. Set allow_bool=True to convert booleans to int
    >>> std_numeric(True, allow_bool=True)
    1

    NumPy scalars and arrays:
    >>> import numpy as np                          # doctest: +SKIP
    >>> std_numeric(np.int64(42))                   # doctest: +SKIP
    42
    >>> std_numeric(np.float32(123))                # doctest: +SKIP
    123.0
    >>> std_numeric(np.array([42]).item())          # doctest: +SKIP
    42

    Pandas types:
    >>> import pandas as pd                         # doctest: +SKIP
    >>> std_numeric(pd.Series([42]).iloc[0])        # doctest: +SKIP
    42
    >>> std_numeric(pd.NA, on_error="nan")          # doctest: +SKIP
    nan

    PyTorch tensors:
    >>> import torch                                # doctest: +SKIP
    >>> std_numeric(torch.tensor(42).item())        # doctest: +SKIP
    42
    >>> std_numeric(torch.tensor(3.14, dtype=torch.float32).item())     # doctest: +SKIP
    3.140000104904175

    TensorFlow tensors:
    >>> import tensorflow as tf                         # doctest: +SKIP
    >>> std_numeric(tf.constant(42).numpy().item())     # doctest: +SKIP
    42
    >>> std_numeric(tf.constant(123).numpy())           # doctest: +SKIP
    123

    JAX arrays:
    >>> import jax.numpy as jnp                         # doctest: +SKIP
    >>> std_numeric(jnp.array(42).item())               # doctest: +SKIP
    42
    >>> std_numeric(jnp.array(3.14))                    # doctest: +SKIP
    3.140000104904175

    Astropy quantities:
    >>> from astropy import units as u                  # doctest: +SKIP
    >>> std_numeric((123 * u.second).value)             # doctest: +SKIP
    123.0

    See Also
    --------
    float() : Python built-in for float conversion
    int() : Python built-in for integer conversion
    """

    def __is_bool_type(val: Any) -> bool:
        """Check if value is any kind of boolean."""
        val_type = type(val)
        val_type_name = val_type.__name__.lower()

        if val_type is bool:
            return True
        if "bool" in val_type_name:
            return True
        if isinstance(val, (bool, int)) and val_type_name in ("bool_", "bool8", "bool"):
            return True

        # Check for mock/custom __name__ attribute on the class
        if hasattr(val_type, "__name__"):
            custom_name = getattr(val_type, "__name__", "")
            if isinstance(custom_name, str) and "bool" in custom_name.lower():
                return True

        return False

    def __handle_error(error_type: str = "unsupported") -> int | float | None:
        """Handle errors based on on_error parameter."""
        if on_error == "raise":
            if error_type == "bool":
                raise TypeError("boolean values not supported")
            elif error_type == "complex":
                raise TypeError("complex numbers not supported")
            elif error_type == "collection":
                raise TypeError(
                    f"sizable collection not supported, got {type(value).__name__} "
                    f"with length {len(value)}. Extract scalar first"
                )
            else:  # unsupported
                raise TypeError(
                    f"unsupported numeric type: {type(value).__name__}. "
                    f"Expected int, float, or types with __index__, __float__, or .item()"
                )
        elif on_error == "nan":
            return float("nan")
        else:  # "none"
            return None

    def __handle_special_cases() -> int | float | None | bool:
        """Handle pandas.NA, numpy.ma.masked, and other special cases. Returns False if not handled."""
        # pandas.NA special case
        if hasattr(value, "__class__"):
            cls = value.__class__
            cls_name = getattr(cls, "__name__", "")
            cls_module = getattr(cls, "__module__", "")
            if cls_name == "NAType" and "pandas" in cls_module:
                return float("nan")

        # numpy.ma.masked special case
        if hasattr(value, "__class__"):
            cls = value.__class__
            if getattr(cls, "__name__", "") == "MaskedConstant" and getattr(
                cls, "__module__", ""
            ).startswith("numpy.ma"):
                return float("nan")

        return False

    def __extract_from_dtype() -> int | float | None | bool:
        """Extract and convert values from objects with dtype attribute. Returns False if not handled."""
        if not hasattr(value, "dtype"):
            return False

        try:
            dtype_str = str(value.dtype)

            # Reject complex types
            if "complex" in dtype_str:
                return __handle_error("complex")

            # Check for boolean dtype
            dtype_is_bool = False
            if hasattr(value.dtype, "is_bool"):
                try:
                    dtype_is_bool = value.dtype.is_bool
                except (AttributeError, TypeError):
                    pass
            if not dtype_is_bool and "bool" in dtype_str.lower():
                dtype_is_bool = True

            if dtype_is_bool and not allow_bool:
                return __handle_error("bool")
        except AttributeError:
            pass

        # Extract scalar from tensor
        result = None
        if hasattr(value, "item") and callable(value.item):
            try:
                result = value.item()
            except (TypeError, ValueError, AttributeError):
                pass

        if result is None and hasattr(value, "numpy") and callable(value.numpy):
            try:
                result = value.numpy()
            except (TypeError, ValueError, AttributeError):
                pass

        if result is not None:
            if __is_bool_type(result):
                if allow_bool:
                    return 1 if result else 0
                else:
                    return __handle_error("bool")

            if type(result) in (int, float) or result is None:
                return result
            elif isinstance(result, (int, float)):
                return int(result) if isinstance(result, int) else float(result)

        return False

    def __try_protocol_methods() -> int | float | None | bool:
        """Try various protocol methods (__index__, __float__, __int__). Returns False if not handled."""
        # __index__() for exact integer types
        if hasattr(value, "__index__"):
            try:
                return operator.index(value)
            except (TypeError, ValueError, AttributeError):
                pass

        # Astropy Quantity (has .value and .unit)
        if hasattr(value, "value") and hasattr(value, "unit"):
            try:
                magnitude = value.value
                return std_numeric(magnitude, on_error=on_error, allow_bool=allow_bool)
            except (TypeError, ValueError, AttributeError):
                pass

        # __float__() for all other numeric types
        if hasattr(value, "__float__"):
            try:
                return float(value)
            except OverflowError:
                try:
                    if value < 0:
                        return float("-inf")
                    else:
                        return float("inf")
                except (TypeError, AttributeError):
                    return float("inf")
            except (TypeError, ValueError) as e:
                if on_error == "raise":
                    raise TypeError(f"cannot convert {type(value).__name__} to float: {e}") from e
                elif on_error == "nan":
                    return float("nan")
                else:
                    return None

        # __int__() as last resort
        if hasattr(value, "__int__"):
            try:
                return int(value)
            except (TypeError, ValueError, OverflowError) as e:
                if on_error == "raise":
                    raise TypeError(f"cannot convert {type(value).__name__} to int: {e}") from e
                elif on_error == "nan":
                    return float("nan")
                else:
                    return None

        return False

    # ===== Main conversion logic =====

    # None passthrough
    if value is None:
        return None

    # Boolean handling
    if isinstance(value, bool):
        if allow_bool:
            return int(value)
        else:
            if on_error == "raise":
                raise TypeError(
                    f"boolean values not supported, got {value}. "
                    f"Set allow_bool=True to convert booleans to int"
                )
            elif on_error == "nan":
                return float("nan")
            else:
                return None

    # Standard Python types - fast path
    if type(value) in (int, float):
        return value

    # Handle special cases (pandas.NA, numpy.ma.masked)
    special_result = __handle_special_cases()
    if special_result is not False:
        return special_result

    # Reject array-like collections
    if hasattr(value, "__len__"):
        try:
            length = len(value)
        except TypeError:
            pass
        else:
            if on_error == "raise":
                if isinstance(value, str):
                    raise TypeError(f"unsupported numeric type: {type(value).__name__}")
                return __handle_error("collection")
            elif on_error == "nan":
                return float("nan")
            else:
                return None

    # Handle array/tensor scalars with dtype
    dtype_result = __extract_from_dtype()
    if dtype_result is not False:
        return dtype_result

    # Try .item() for non-tensor objects (e.g., pandas scalars)
    if hasattr(value, "item") and callable(value.item):
        try:
            result = value.item()
            if __is_bool_type(result):
                if allow_bool:
                    return 1 if result else 0
                else:
                    return __handle_error("bool")
            elif type(result) in (int, float) or result is None:
                return result
            elif isinstance(result, (int, float)):
                return int(result) if isinstance(result, int) else float(result)
        except (TypeError, ValueError, AttributeError):
            pass

    # SymPy Boolean handling (only when allow_bool=True)
    if allow_bool and hasattr(value, "__class__"):
        cls = value.__class__
        cls_name = cls.__dict__.get("__name__", cls.__name__ if hasattr(cls, "__name__") else "")
        cls_module = cls.__dict__.get(
            "__module__", cls.__module__ if hasattr(cls, "__module__") else ""
        )

        if "sympy" in cls_module and cls_name in ("BooleanTrue", "BooleanFalse"):
            try:
                return int(bool(value))
            except (TypeError, ValueError):
                return 1 if cls_name == "BooleanTrue" else 0

    # Try protocol methods (__index__, __float__, __int__)
    protocol_result = __try_protocol_methods()
    if protocol_result is not False:
        return protocol_result

    # Type not supported
    return __handle_error("unsupported")
