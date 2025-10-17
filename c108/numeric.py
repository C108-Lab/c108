"""
Standardize numeric types from Python stdlib and third-party libraries.

This module provides robust numeric type conversion suitable for display
formatting, data processing, and cross-library interoperability.
"""

# Standard library -----------------------------------------------------------------------------------------------------
import math
import operator
from typing import Literal, Protocol, runtime_checkable

# Local ----------------------------------------------------------------------------------------------------------------
from .tools import fmt_type


@runtime_checkable
class SupportsFloat(Protocol):
    """Protocol for duck-typed numeric conversion."""

    def __float__(self) -> float: ...


def std_numeric(
        value,
        *,
        on_error: Literal["raise", "nan", "none"] = "raise",
        allow_bool: bool = False
) -> int | float | None:
    """
    Convert numeric types to standard Python int, float, or None.

    Normalizes numeric values from Python stdlib and third-party libraries
    (NumPy, Pandas, Decimal, etc.) into standard Python types for display,
    serialization, and processing.

    Parameters
    ----------
    value : various
        Numeric value to convert. Supports Python int/float/None, Decimal,
        Fraction, and third-party types via __index__, __int__, __float__,
        .item(), or .value protocols.

    on_error : {"raise", "nan", "none"}, default "raise"
        How to handle TYPE ERRORS (unsupported types like str, list, dict):

        - "raise": Raise TypeError (default, safest for development)
        - "nan": Return float('nan') (useful for display/reporting)
        - "none": Return None (useful for filtering pipelines)

        **Important**: Numeric edge cases (inf, nan, overflow, underflow) are
        ALWAYS preserved as valid IEEE 754 values, regardless of this setting.

    allow_bool : bool, default False
        If True, convert bool to int (True→1, False→0). If False, treat
        bool as type error (respects on_error setting). Default False helps
        catch bugs since bool is subclass of int in Python.

    Returns
    -------
    int
        For Python int (arbitrary precision, never overflows), or types
        implementing __index__ (NumPy integers), or integer-valued
        Decimal/Fraction (e.g., Decimal('42.0') → 42), or types
        implementing only __int__ (without __float__).

    float
        For float values, including special IEEE 754 values:
        inf/-inf (overflow or explicit), nan, 0.0/-0.0 (underflow).

    None
        For None input, pandas.NA, numpy.ma.masked, or type errors
        when on_error="none".

    Raises
    ------
    TypeError
        When on_error="raise" and value is unsupported type (str, list, etc.)
        or bool when allow_bool=False.

    Behavior Notes
    --------------
    **Type Preservation:**
    - Python int → int (preserves arbitrary precision, never overflows)
    - Integer-valued Decimal/Fraction → int (e.g., Decimal('42'), Decimal('3.14e100'))
    - Decimal/Fraction with fractions → float (may overflow to inf/underflow to 0.0)
    - NumPy integers → int via __index__ (preserves exact values)
    - Types with only __int__ → int (no __float__ available)
    - Float-like types → float via __float__ (may overflow/underflow)

    **Overflow (value too large for float):**

    Integer types preserve exact values as Python int (arbitrary precision):
        10**400 → int (exact, 400 digits)
        Decimal('1e400') → int (exact, integer-valued)

    Float conversion overflows to inf when value exceeds ~±1.8e308:
        Decimal('3.14') → 3.14 (within range)
        float('1e400') → inf (already overflow in float literal)

    Note: Decimal('1e400') is mathematically an integer (10^400), so it's
        returned as int. Decimal('3.14e400') normalizes to 314e398 (also an
        integer internally), so also returned as int. Only Decimals with true
        fractional parts that exceed float range will overflow to inf via the
        __float__() path, but this is rare in practice.

    For display purposes where you want to show "∞" for huge values,
        check the magnitude explicitly:
        >>> result = std_numeric(value)
        >>> if isinstance(result, int) and abs(result) > 1e100:
        ...     display_as_infinity()

    **Underflow (value too small for float):**
    Values smaller than ~5e-324 silently become 0.0/-0.0 (sign preserved).

        Decimal('1e-400') → 0.0
        Decimal('-1e-400') → -0.0

    **Special Values (always preserved):**
    inf, -inf, nan are valid numeric values, not errors. They pass through
    unchanged regardless of on_error setting.

        float('inf') → inf
        float('nan') → nan
        pandas.NA → nan
        numpy.ma.masked → nan

    **Precision Loss:**
    High-precision Decimal values within float range lose precision when
    converted (float has ~15-17 significant digits). Use Python int or
    keep as Decimal if exact precision needed.

    **Detection Priority:**
    1. __index__() → int (NumPy integers, strictest)
    2. .item() → int or float (array scalars)
    3. .value (Astropy Quantity)
    4. Integer-valued check (Decimal/Fraction optimization)
    5. __int__() → int (when __float__ not available)
    6. __float__() → float (general fallback)

    Examples
    --------
    **Basic types:**

    >>> std_numeric(42)
    42
    >>> std_numeric(3.14)
    3.14
    >>> std_numeric(None)
    None
    >>> std_numeric(10**100)  # Arbitrary precision int
    10000000000000000...000  # (100 digits, exact int)

    **Decimal - integer-valued becomes int:**

    >>> from decimal import Decimal
    >>> std_numeric(Decimal('42'))
    42
    >>> std_numeric(Decimal('42.0'))
    42
    >>> std_numeric(Decimal('1e400'))  # Huge but integer-valued
    10000000000000000...000  # (400 digits, exact int)

    **Decimal - with fraction becomes float:**

    >>> std_numeric(Decimal('3.14'))
    3.14

    **Underflow to zero:**

    >>> std_numeric(Decimal('1e-400'))
    0.0

    **Special values preserved:**

    >>> std_numeric(float('inf'))
    inf
    >>> std_numeric(float('nan'))
    nan

    **Error handling:**

    >>> std_numeric("invalid")  # Default: raise
    Traceback (most recent call last):
        ...
    TypeError: unsupported numeric type: str

    >>> std_numeric("invalid", on_error="nan")
    nan
    >>> std_numeric("invalid", on_error="none")
    None

    **Boolean handling:**

    >>> std_numeric(True)  # Default: reject
    Traceback (most recent call last):
        ...
    TypeError: boolean values not supported

    >>> std_numeric(True, allow_bool=True)
    1

    **Numeric edge cases preserved regardless of on_error:**

    >>> std_numeric(Decimal('1e400'), on_error="none")
    10000000000000000...000  # Huge int, not suppressed!

    >>> std_numeric(float('nan'), on_error="raise")  # Still nan
    nan

    **Custom types with __int__ only:**

    >>> class MyInt:
    ...     def __int__(self):
    ...         return 42
    >>> std_numeric(MyInt())
    42

    See Also
    --------
    float() : Python built-in for float conversion
    int() : Python built-in for integer conversion
    math.isfinite() : Check if value is finite (not inf/nan)
    """

    # None passthrough
    if value is None:
        return None

    # Boolean handling
    if isinstance(value, bool):
        if allow_bool:
            return int(value)  # True → 1, False → 0
        else:
            # Respect on_error setting
            if on_error == "raise":
                raise TypeError(
                    f"boolean values not supported, got {value}. "
                    f"Set allow_bool=True to convert booleans to int (True→1, False→0)"
                )
            elif on_error == "nan":
                return float('nan')
            else:  # on_error == "none"
                return None

    # Standard Python numeric types - fast path
    # Python int has arbitrary precision, never overflows
    if isinstance(value, (int, float)):
        return value

    # Check for pandas.NA (singleton) - must check before duck typing
    # pandas.NA has __float__ but raises TypeError, so handle specially
    if hasattr(value, "__class__"):
        cls = value.__class__
        cls_name = getattr(cls, "__name__", "")
        cls_module = getattr(cls, "__module__", "")
        if cls_name == "NAType" and "pandas" in cls_module:
            return math.nan

    # Detect numpy.ma.masked sentinel (MaskedConstant) without importing numpy
    # Treat as NaN for numeric display purposes
    if hasattr(value, "__class__"):
        cls = value.__class__
        if getattr(cls, "__name__", "") == "MaskedConstant" and getattr(cls, "__module__", "").startswith("numpy.ma"):
            return math.nan

    # Priority 1: Check __index__() for "true integers"
    # This is the most reliable signal that a type represents an exact integer
    # NumPy integer types (int8-64, uint8-64) implement this
    # Returns Python int with arbitrary precision - no overflow possible
    if hasattr(value, '__index__'):
        try:
            return operator.index(value)
        except (TypeError, ValueError) as e:
            # __index__ exists but failed
            if on_error == "raise":
                raise TypeError(
                    f"cannot convert {fmt_type(value)} to int via __index__: {e}"
                ) from e
            elif on_error == "nan":
                return float('nan')
            else:  # on_error == "none"
                return None

    # Priority 2: Handle array/tensor scalars with .item() method
    # Common in NumPy, PyTorch, TensorFlow, JAX
    # .item() returns Python scalar - respect its type (int or float)
    if hasattr(value, 'item') and callable(value.item):
        try:
            result = value.item()
            # Check result type and recurse to handle it properly
            if isinstance(result, bool):
                # Respect allow_bool setting even from .item() results
                if allow_bool:
                    return int(result)
                else:
                    if on_error == "raise":
                        raise TypeError(
                            f"boolean values not supported (from .item()), got {value}"
                        )
                    elif on_error == "nan":
                        return float('nan')
                    else:  # on_error == "none"
                        return None
            elif isinstance(result, (int, float, type(None))):
                return result
            # If .item() returned something else, fall through to other methods
        except (TypeError, ValueError, AttributeError):
            # .item() failed, try other methods
            pass

    # Priority 3: Handle Astropy Quantity objects (physical quantities with units)
    # These have .value attribute containing the numeric magnitude
    if hasattr(value, 'value') and hasattr(value, 'unit'):
        # Heuristic: object has both .value and .unit attributes
        # Strong indicator of Astropy Quantity or similar
        try:
            magnitude = value.value
            # Recursively process the magnitude
            return std_numeric(
                magnitude,
                on_error=on_error,
                allow_bool=allow_bool
            )
        except (TypeError, ValueError, AttributeError):
            # Not actually a Quantity-like object, fall through
            pass

    # Priority 4: Check for integer-valued Decimal/Fraction (optimization)
    # These types implement both __int__ and __float__
    # If mathematically an integer, preserve as int for arbitrary precision
    type_name = type(value).__name__
    if type_name in ('Decimal', 'Fraction') and hasattr(value, '__int__'):
        try:
            # Try converting to int
            as_int = int(value)
            # Check if it's mathematically an integer by comparing back
            # This avoids truncation of fractional parts
            if value == type(value)(as_int):
                return as_int  # It's an exact integer, preserve as int
            # Has fractional part, fall through to __float__
        except (TypeError, ValueError, OverflowError):
            # Conversion failed, fall through to __float__
            pass

    # Priority 5: General __int__() support (when __float__ not available)
    # For types that only implement __int__ without __float__
    # This avoids ambiguity - types with both prefer __float__ (more general)
    if hasattr(value, '__int__') and not hasattr(value, '__float__'):
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError) as e:
            # __int__() failed - treat as type error
            if on_error == "raise":
                raise TypeError(
                    f"cannot convert {fmt_type(value)} to int via __int__: {e}"
                ) from e
            elif on_error == "nan":
                return float('nan')
            else:  # on_error == "none"
                return None

    # Priority 6: Duck typing via __float__
    # Handles: Decimal (with fractions), Fraction (with fractions),
    # NumPy float scalars, SymPy, mpmath, etc.
    # May overflow to inf/-inf or underflow to 0.0/-0.0
    if hasattr(value, '__float__'):
        try:
            result = float(value)
            # Overflow: values beyond ~±1.8e308 become inf/-inf
            # Underflow: values below ~5e-324 become 0.0/-0.0
            # Special values: numpy.nan → float('nan'), numpy.inf → float('inf')
            # These are all VALID numeric values, not errors
            return result
        except (TypeError, ValueError) as e:
            # Conversion failed - this is a type error
            if on_error == "raise":
                raise TypeError(
                    f"cannot convert {fmt_type(value)} to float: {e}"
                ) from e
            elif on_error == "nan":
                return float('nan')
            else:  # on_error == "none"
                return None

    # If we get here, type is not supported - this is a type error
    if on_error == "raise":
        raise TypeError(
            f"unsupported numeric type: {fmt_type(value)}. "
            f"Expected int, float, None, or types implementing __index__, __int__, "
            f"__float__, .item(), or having .value attribute (e.g., numpy scalars, "
            f"Decimal, Fraction, pandas scalars, array.item(), Quantity.value)"
        )
    elif on_error == "nan":
        return float('nan')
    else:  # on_error == "none"
        return None
