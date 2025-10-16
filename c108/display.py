"""
Numeric display formatting tools for terminal UI, progress bars, status displays, etc
"""

# Standard library -----------------------------------------------------------------------------------------------------
import math
from dataclasses import dataclass, InitVar, field
from enum import StrEnum, unique
from typing import Any, Self, Protocol, runtime_checkable

# Local ----------------------------------------------------------------------------------------------------------------

from .collections import BiDirectionalMap

from .tools import fmt_type, fmt_value


@runtime_checkable
class SupportsFloat(Protocol):
    """Protocol for duck-typed numeric conversion."""

    def __float__(self) -> float: ...


# @formatter:off

class DisplayConf:
    PLURALS = {
        "byte": "bytes", "step": "steps", "item": "items",
        "second": "seconds", "minute": "minutes", "hour": "hours",
        "day": "days", "week": "weeks", "month": "months",
        "year": "years", "meter": "meters", "gram": "grams"
    }


# TODO si_prefixes & value_multipliers customization
si_prefixes = BiDirectionalMap({
    -12: "p", -9: "n", -6: "µ", -3: "m", 0: "",
    3: "k", 6: "M", 9: "G", 12: "T", 15: "P", 18: "E", 21: "Z",
})

value_multipliers = BiDirectionalMap({
    -12: "10⁻¹²", -9: "10⁻⁹", -6: "10⁻⁶", -3: "10⁻³", 0: "",
    3: "10³", 6: "10⁶", 9: "10⁹", 12: "10¹²",
    15: "10¹⁵", 18: "10¹⁸", 21: "10²¹",
})

valid_exponents = tuple(si_prefixes.keys())
valid_orders = tuple([10**exp for exp in valid_exponents])
valid_si_prefixes = tuple(sorted(si_prefixes.values()))
# @formatter:on

# Classes --------------------------------------------------------------------------------------------------------------

# @formatter:off
@unique
class DisplayMode(StrEnum):
    """
    Modes for value-unit pair display.

    Attributes:
        BASE_FIXED (str) : Base units, flexible value multiplier - 123×10⁹ byte
        PLAIN (str)      : Base units, plain int, E-notation for floats - 1 byte, 2.2+e3 s
        SI_FIXED (str)   : Fixed SI units prefix, flexible value multiplier - 123×10³ Mbyte
        SI_FLEX (str)    : Flexible SI units prefix, no value multiplier - 123.4 ns
    """
    BASE_FIXED = "base_fixed"
    PLAIN = "plain"
    SI_FIXED = "si_fixed"
    SI_FLEX = "si_flex"
# @formatter:on


@unique
class MultiSymbol(StrEnum):
    ASTERISK = "*"
    CDOT = "⋅"
    CROSS = "×"


@dataclass(frozen=True)
class DisplayValue:
    """
    A numeric value with intelligent unit formatting for display.

    Supports: int, float, None, inf, -inf, nan, numpy scalars, pandas scalars.
    NumPy and Pandas types are detected via duck typing.

    This class can display numbers in plain format, with SI prefixes (flexible or
    fixed), or with an explicit power-of-ten multiplier. It automatically handles
    exponent calculation, trimmed digits, and unit pluralization.

    For most use cases, prefer the factory class methods:
        - DisplayValue.base_fixed() for base units with multiplier - 123×10⁹ byte
        - DisplayValue.plain() for plain int, E-notation for floats - 1 byte, 2.2+e3 s
        - DisplayValue.si_fixed() for fixed SI prefix - 123×10³ Mbyte
        - DisplayValue.si_flex() for auto-scaled SI prefix - 123.4 ns
    """

    value: int | float | None
    unit: str | None = None

    mult_exp: InitVar[int | str | None] = None
    unit_exp: InitVar[int | str | None] = None
    trim_digits: InitVar[int | None] = None

    multi_symbol: str = MultiSymbol.CROSS
    plural_units: dict[str, str] | bool = True
    precision: int | None = None
    separator: str = " "
    whole_as_int: bool | None = None

    _mult_exp: int | None = field(init=False, default=None, repr=False)
    _unit_exp: int | None = field(init=False, default=None, repr=False)
    _trim_digits: int | None = field(init=False, default=None, repr=False)

    def __post_init__(
            self,
            mult_exp: int | str | None,
            unit_exp: int | str | None,
            trim_digits: int | None
    ):
        # Value, trim and precision
        value_ = _std_numeric(self.value)
        object.__setattr__(self, 'value', value_)
        self._validate_trim_and_precision(trim_digits)

        # Parse exponents to int or None
        mult_exp = _parse_exponent_value(mult_exp)
        unit_exp = _parse_exponent_value(unit_exp)

        # Set exponents (auto-calculate if needed)
        self._set_exponents(mult_exp, unit_exp)

        # Set trimmed digits and whole_as_int
        object.__setattr__(self, '_trim_digits', trim_digits)
        if self.whole_as_int is None:
            object.__setattr__(self, 'whole_as_int', self.mode != DisplayMode.PLAIN)

    def merge(self, **kwargs) -> Self:
        """
        TODO Create new instance with updated formatting options.
        """

    @classmethod
    def si_flex(cls, value: int | float, *, unit: str) -> Self:
        """
        TODO Auto-scale to appropriate SI prefix (MB, GB, ms, µs, etc.).
        """

    @classmethod
    def si_fixed(
            cls,
            value: int | float | None = None,
            si_value: int | float | None = None,
            trim_digits: int | None = None,
            precision: int | None = None,
            *,
            si_unit: str
    ) -> Self:
        """Create with fixed SI prefix and flexible multiplier.

        The si_unit parameter determines both the unit and the fixed SI prefix.

        Args:
            value: Numeric value IN BASE UNITS. Mutually exclusive with si_value.
                   Use when you have data in base units (bytes, seconds, meters).
            si_value: Numeric value IN SI-PREFIXED UNITS. Mutually exclusive with value.
                      Use when you have data already in SI units (megabytes, milliseconds).
            si_unit: SI-prefixed unit string (e.g., "Mbyte", "ms", "km"). REQUIRED.
                     Specifies both the base unit and the fixed SI prefix.
            precision: Number of decimal digits for float values representation.
            trim_digits: Number of digits for rounding in representation.

        Raises:
            ValueError: If both value and si_value are provided, or if neither is provided.

        Examples:
            # From base units (123 million bytes):
            DisplayValue.si_fixed(value=123_000_000, si_unit="Mbyte")
            # → "123 Mbyte" or "123×10³ Mbyte" depending on actual magnitude

            # From SI units (123 megabytes):
            DisplayValue.si_fixed(si_value=123, si_unit="Mbyte")
            # → "123 Mbyte" (internally converts to 123_000_000 base units)

            # Fractional units:
            DisplayValue.si_fixed(si_value=500, si_unit="Mbyte/s")
            # → "500 Mbyte/s"
        """
        # Validation
        if value is not None and si_value is not None:
            raise ValueError("only one of 'value' or 'si_value' allowed, not both.")

        # Parse si_unit to extract prefix and base unit
        prefix, base_unit = _parse_si_unit_string(si_unit)
        exp = si_prefixes.get_key(prefix) if prefix else 0

        # Convert si_value to stdlib types
        si_value_ = _std_numeric(si_value)
        # Convert to base units if provided
        value = si_value_ * (10 ** exp) if _is_finite(si_value_) else si_value_

        return cls(
            value=value,
            trim_digits=trim_digits,
            precision=precision,
            unit=base_unit,
            unit_exp=exp,
        )

    def __str__(self):
        return self.as_str

    @property
    def as_str(self):
        """Number with units as a string."""
        if not self._units_str:
            return self._number_str

        # Don't use separator when _units_str is just an SI prefix (single character like 'k', 'M')
        if self.unit is None and len(self._units_str) <= 1:
            return f"{self._number_str}{self._units_str}"

        return f"{self._number_str}{self.separator}{self._units_str}"

    @property
    def display_digits(self) -> int | None:
        """
        Number of digits used for display formatting after trimming trailing zeros.

        Returns:
            int: Number of digits for display (minimum 1 for finite values).
            None: If value is None or non-finite (NaN, infinity).

        Note:
            Returns user-specified trim_digits if provided during initialization,
            otherwise auto-calculates by calling trimmed_digits() on the source value
            with the configured round_digits precision.

        See Also:
            trimmed_digits() - Function for calculating display digits from numbers
        """
        if self._trim_digits is not None:
            return self._trim_digits

        return self._src_trimmed_digits

    @property
    def exponent(self) -> int:
        """
        The total exponent at normalized value, equals to source value exponent or 0.

        exponent = multiplier_exponent + unit_exponent

        Example:
            Value 123.456×10³ byte, the exponent == 3;
            Value 123.456×10³ kbyte the exponent == 6, and multiplier_exponent == 3;
            Value 1.2e+3 s in PLAIN mode, the exponent == 0 as we do not display multiplier and unit exponents here.
        """
        if self._mult_exp is None and self._unit_exp is None:
            return 0
        elif self._mult_exp == 0 and self._unit_exp == 0:
            return 0
        else:
            return self._src_exponent

    @property
    def mode(self) -> DisplayMode:
        """Derive display mode from multiplier and unit exponents."""
        mult_exp = self._mult_exp
        unit_exp = self._unit_exp

        if mult_exp == 0 and unit_exp == 0:
            return DisplayMode.PLAIN
        elif mult_exp is None and isinstance(unit_exp, int):
            return DisplayMode.BASE_FIXED if unit_exp == 0 else DisplayMode.SI_FIXED
        elif isinstance(mult_exp, int) and unit_exp is None:
            return DisplayMode.SI_FLEX
        else:
            raise ValueError(
                f"Invalid exponents state: mult_exp={mult_exp}, unit_exp={unit_exp}. "
                f"At least one must be an integer."
            )

    @property
    def multiplier_exponent(self) -> int:
        """
        Display exponent in SI_FIXED display mode, multiplier_exponent = exponent - unit_exponent.
        Returns 0 for other modes.

        Example:
            123.456×10³ km has exponent = 6 and multiplier_exponent = 3.
        """
        if self._mult_exp is not None:
            return self._mult_exp
        return self.exponent - self._unit_exp

    @property
    def normalized(self) -> int | float | None:
        """
        Normalized value, mantissa without multiplier.

        normalized_value = value / ref_value, where ref_value = 10 ** exponent

        Includes rounding to trimmed digits and optional whole_as_int convertion.

        Example:
            displayed value 123.4×10³ ms has the normalized value 123.4
        """
        if not _is_finite(self.value):
            return self.value

        value_ = self.value / self.ref_value
        norm_number = _normalized_number(value_,
                                         trim_digits=self.display_digits,
                                         whole_as_int=self.whole_as_int)
        return norm_number

    @property
    def ref_value(self) -> int | float:
        """
        The reference value for scaling the normalized display number:
        normalized = value / ref_value, where ref_value = 10^exponent

        Example:
            Value is 123.456×10³ kbyte, the ref_value is 10⁶, exponent = 6
        """
        return 10 ** self.exponent

    @property
    def si_prefix(self) -> str:
        """
        The SI prefix in units of measurement, e.g., 'm' (milli-), 'k' (kilo-).
        """
        return si_prefixes[self.unit_exponent]

    @property
    def unit_exponent(self) -> int:
        """
        The SI Unit exponent used in SI display modes; 0 in other modes.
        """
        if self._unit_exp is not None:
            return self._unit_exp
        return self.exponent - self._mult_exp

    @property
    def _multiplier_str(self) -> str:
        """
        Numeric multiplier suffix.

        Example:
            displayed value 123×10³ byte has _multiplier_str of ×10³
        """
        if self.multiplier_exponent == 0:
            return ""

        return f"{self.multi_symbol}{value_multipliers[self.multiplier_exponent]}"

    @property
    def _number_str(self) -> str:
        """
        Numerical part of full str-representation including the multiplier if applicable.

        Example:
            The value 123.456×10³ km has _number_str 123.456×10³
        """
        display_number = self.normalized

        if not _is_finite(display_number):
            return _infinite_to_str(display_number)

        if self.precision is not None:
            return f"{display_number:.{self.precision}f}{self._multiplier_str}"

        if self.whole_as_int or isinstance(display_number, int):
            return f"{display_number}{self._multiplier_str}"
        else:
            return f"{display_number:.{self.display_digits}g}{self._multiplier_str}"

    @property
    def _units_str(self) -> str:
        """
        Units of measurement only from a full str-representation, includes SI prefix if applicable.

        Example:
            123 ms has _units_str = 'ms'.
            123.5 k (no unit) has _units_str = 'k'.
        """
        # Handle case where no unit is specified but SI prefix exists
        if not self.unit:
            if self.si_prefix:
                return self.si_prefix
            else:
                return ""

        if not _is_finite(self.normalized):
            if self._unit_exp is not None:
                return f"{self.si_prefix}{self.unit}"
            else:
                return f"{self.unit}"

        # Check if we should pluralize (normalized value != ±1)
        if abs(self.normalized) == 1:
            return f"{self.si_prefix}{self.unit}"

        plural_units = self._get_plural_units()
        if self.unit in plural_units:
            return f"{self.si_prefix}{plural_units[self.unit]}"
        else:
            return f"{self.si_prefix}{self.unit}"

    def _get_plural_units(self) -> dict[str, str]:
        """Returns the appropriate plural map based on the configuration."""
        if isinstance(self.plural_units, dict):
            return self.plural_units
        elif self.plural_units is True:
            return DisplayConf.PLURALS
        else:
            return {}

    def _validate_value(self):
        """Validate value based on Obj state, no properties involved"""
        if not isinstance(self.value, (int, float, type(None))):
            raise ValueError(f"The 'value' must be int | float | None: {type(self.value)}.")

        # Validate finite values
        if self.value is not None:
            if math.isnan(self.value):
                raise ValueError("NaN values are not supported")
            if math.isinf(self.value):
                raise ValueError("Infinite values are not supported")

    def _validate_trim_and_precision(self, trim_digits: int | None):
        """Validate initialization parameters"""
        if trim_digits is not None and trim_digits < 1:
            raise ValueError(f"trim_digits must be >= 1, got {trim_digits}")

        if self.precision is not None and self.precision < 0:
            raise ValueError(f"precision must be >= 0, got {self.precision}")

    def _set_exponents(self, mult_exp: int | None, unit_exp: int | None):
        """
        Determine Exponents if not provided. Uses object.__setattr__ for frozen dataclass.
        """
        if type(mult_exp) not in (int, type(None)):
            raise TypeError(f"mult_exp must be int or None, got {type(mult_exp)}")

        if type(unit_exp) not in (int, type(None)):
            raise TypeError(f"unit_exp must be int or None, got {type(unit_exp)}")

        if mult_exp is not None and unit_exp is not None:
            if mult_exp != 0 or unit_exp != 0:
                raise ValueError(
                    f"mult_exp and unit_exp must be 0 if specified both: {mult_exp}/{unit_exp} "
                    "Use two Nones or only one of mult_exp/unit_exp otherwise."
                )

        if mult_exp is None and unit_exp is None:
            mult_exp = self._src_exponent

        object.__setattr__(self, '_mult_exp', mult_exp)
        object.__setattr__(self, '_unit_exp', unit_exp)

    @property
    def _src_exponent(self) -> int:
        """
        Returns the exponent of source DisplayValue.value represented a product of normalized and ref_value OR 0

        The exponent is a multiple of 3, and the resulting normalized number is in the range [1, 1000):

        value = normalized * 10^_src_exponent
        """
        if self.value == 0:
            return 0
        elif _is_finite(self.value):
            magnitude = math.floor(math.log10(abs(self.value)))
            src_exponent = (magnitude // 3) * 3
            return src_exponent
        else:
            return 0

    @property
    def _src_trimmed_digits(self) -> int | None:
        """
        Calculate trimmed("significant") digits based on the source DisplayValue.value with trimmed_digits()

        Ignore trailing zeros in float as non-significant both before and after the decimal point
        """
        return trimmed_digits(self.value)


# Methods --------------------------------------------------------------------------------------------------------------

def _infinite_to_str(val: int | float | None):
    """Format stdlib infinite numerics: None, +/-inf, NaN."""

    if val is None:
        return "N/A"

    if math.isinf(val):
        return "∞" if val > 0 else "-∞"

    if math.isnan(val):
        return f"NaN"

    raise TypeError(f"cannot format as infinite value: {fmt_type(val)}")


def _is_finite(value: Any) -> bool:
    """
    Check if a value is a finite numeric value suitable for display.

    Args:
        value: The value to check.

    Returns:
        bool: True if value is a finite int or float (excluding bool).
              False for None, NaN, infinity, bool, or non-numeric types.

    Examples:
        _is_finite(123) is True
        _is_finite(123.456) is True
        _is_finite(float('inf')) is False
        _is_finite(float('nan')) is False
        _is_finite(None) is False
        _is_finite(True) is False  # Excludes booleans
    """
    # Exclude booleans (isinstance(True, int) is True in Python)
    if isinstance(value, bool):
        return False

    # Check for numeric type
    if not isinstance(value, (int, float)):
        return False

    # Check for finite value (excludes NaN, inf, -inf)
    # Propagates no exceptions - math.isfinite handles all numeric types
    return math.isfinite(value)


def _parse_exponent_value(exp: int | str | None) -> int | None:
    """
    Parse an exponent from a SI prefix string or validate and return its int power.

    Raises:
         ValueError if exponent is not a valid SI prefix or is invalid int power.

    Examples:
        >>> _parse_exponent_value("1000")
        1000
        >>> _parse_exponent_value("M")
        6
    """
    if exp is None:
        return None

    elif isinstance(exp, str):
        if exp not in valid_si_prefixes:
            raise ValueError(
                f"Invalid exponent string value: '{exp}', expected one of {valid_si_prefixes}"
            )
        return si_prefixes.get_key(exp)

    elif isinstance(exp, int):
        if exp not in valid_exponents:
            raise ValueError(
                f"Invalid exponent integer value: {exp}, expected one of {valid_exponents}"
            )
        return exp

    raise TypeError(f"Exponent value must be an int | str | None, got {fmt_type(exp)}")


def _parse_si_unit_string(si_unit: str) -> tuple[str, str]:
    """Parse SI unit string into (prefix, base_unit).

    Examples:
        "Mbyte" → ("M", "byte")
        "ms" → ("m", "s")
        "byte" → ("", "byte")
        "km/h" → ("k", "m/h")
    """
    if not isinstance(si_unit, str) or not si_unit:
        raise ValueError(f"si_unit must be a non-empty string, got: {fmt_value(si_unit)}")

    first_char = si_unit[0]

    # Check if first character is a valid SI prefix
    if first_char in valid_si_prefixes and len(si_unit) > 1:
        return first_char, si_unit[1:]
    else:
        # No SI prefix, entire string is the base unit
        return "", si_unit


def _normalized_number(
        value: int | float | None,
        trim_digits: int | None = None,
        whole_as_int: bool = False
) -> int | float | None:
    """Process value to normalized number by rounding and conditional int conversion."""
    if not _is_finite(value):
        return value

    if not isinstance(value, (int, float)):
        raise TypeError(f"Expected int | float, got {type(value)}")

    if trim_digits is not None:
        if value != 0:
            magnitude = math.floor(math.log10(abs(value)))
            factor_ = 10 ** (trim_digits - 1 - magnitude)
            value = round(value * factor_) / factor_

    if whole_as_int and isinstance(value, float) and value.is_integer():
        return int(value)

    return value


def _std_numeric(value: int | float | None | SupportsFloat) -> int | float | None:
    """
    Convert common numeric types to standard Python int or float/inf/nan or None.

    Handles:
    - Python int, float, None
    - (+/-)math.inf, math.nan
    - numpy.int64, numpy.float32, numpy.nan, numpy.inf
    - pandas.NA, pd.Series.item(), etc.

    Returns:
        Normalized int/float/None/inf/nan

    Raises:
        TypeError: If value cannot be converted to numeric
    """
    # None passthrough
    if value is None:
        return None

    # Standard Python types
    if isinstance(value, (int, float)):
        # Handle bool (subclass of int) - decide if you want to allow it
        if isinstance(value, bool):
            raise TypeError(f"boolean values not supported, but got {value}")
        return value

    # Check for pandas.NA (before duck typing, as it has __float__ but raises)
    # pandas.NA is a singleton
    if hasattr(value, '__class__') and value.__class__.__name__ == 'NAType':
        return math.nan

    # Duck typing: anything with __float__
    if hasattr(value, '__float__'):
        try:
            result = float(value)
            # numpy.nan converts to float('nan')
            # numpy.inf converts to float('inf')
            return result
        except (TypeError, ValueError) as e:
            raise TypeError(f"cannot convert {fmt_type(value)} to float: {e}")

    # If we get here, type is not supported
    raise TypeError(
        f"unsupported numeric type: {fmt_type(value)}. "
        f"Expected int, float, None, or types implementing __float__"
    )


def trimmed_digits(number: int | float | None, *, round_digits: int | None = 15) -> int | None:
    """
    Count significant digits for display by removing all trailing zeros.

    Used for compact display formatting (e.g., "123×10³" instead of "123000"). Removes trailing zeros
    from both integers and the decimal representation of floats to determine the minimum digits
    needed for display.

    Float values are rounded before analysis to eliminate floating-point precision artifacts
    (e.g., 0.30000000000000004 from 0.1 + 0.2).

    **⚠️ DISPLAY PURPOSE ONLY:** This function treats trailing zeros in floats (e.g., 1200.0)
    as non-significant, which violates standard scientific notation rules. Use this ONLY for
    UI display formatting, NOT for scientific calculations or significant figure analysis.

    Args:
        number: The number to analyze for display. Accepts int, float, or None.
        round_digits: Number of decimal places to round floats before analysis.
                     Default 15 eliminates common float artifacts while preserving
                     meaningful precision. Set to None to disable rounding (keeps
                     all float precision artifacts). Only affects float values.

    Returns:
        int: Number of significant digits after removing trailing zeros (minimum 1).
        None: If input is None, NaN, inf, or -inf.

    Raises:
        TypeError: If number is not int, float, or None.
                   If round_digits is not int, None, or missing.

    Examples:
        # Integers - trailing zeros removed for compact display
        trimmed_digits(123000) == 3           # Display as "123×10³"
        trimmed_digits(100) == 1              # Display as "1×10²"
        trimmed_digits(101) == 3              # Display as "101"
        trimmed_digits(0) == 1                # Zero has one digit
        trimmed_digits(-456000) == 3          # Sign ignored, "456×10³"

        # Floats - all trailing zeros removed (non-standard!)
        trimmed_digits(0.456) == 3            # No trailing zeros
        trimmed_digits(123.456) == 6          # All significant
        trimmed_digits(123.450) == 5          # Python may normalize to "123.45"
        trimmed_digits(1200.0) == 2           # ⚠️ Non-standard: "12×10²"
        trimmed_digits(0.00123) == 3          # Leading zeros don't count

        # Float precision artifacts - automatically handled with default round_digits=15
        trimmed_digits(0.1 + 0.2) == 1        # 0.30000000000000004 → 0.3 → 1 digit
        trimmed_digits(1/3) == 15             # 0.333... rounded to 15 digits
        trimmed_digits(0.1 + 0.2, round_digits=None) == 17  # Keep artifacts

        # Custom rounding precision
        trimmed_digits(1/3, round_digits=5) == 5    # 0.33333
        trimmed_digits(1/3, round_digits=2) == 2    # 0.33
        trimmed_digits(1/3, round_digits=0) == 1    # 0.0 → 1 digit (zero)

        # Scientific notation (Python's string conversion)
        trimmed_digits(1.23e5) == 3           # "123000.0" → rounded → 3 trimmed
        trimmed_digits(1.23e-4) == 3          # "0.000123" → 3 trimmed
        trimmed_digits(1e10) == 1             # "10000000000.0" → 1 trimmed

        # Special values - None for non-displayable numbers
        trimmed_digits(None) is None
        trimmed_digits(float('nan')) is None
        trimmed_digits(float('inf')) is None
        trimmed_digits(float('-inf')) is None

        # Edge cases
        trimmed_digits(-0.0) == 1             # Negative zero same as zero
        trimmed_digits(100, round_digits=2) == 1  # Rounding has no effect on ints

    Note:
        The round_digits parameter uses Python's built-in round() function, which
        uses "round half to even" (banker's rounding). For most display purposes,
        the default value of 15 provides excellent results.
    """
    # Type validation
    if not isinstance(number, (int, float, type(None))):
        raise TypeError(f"Expected int | float | None, got {fmt_type(number)}")

    if round_digits is not None and not isinstance(round_digits, int):
        raise TypeError(f"round_digits must be int or None, got {fmt_type(round_digits)}")

    # Handle None input
    if number is None:
        return None

    # Handle non-finite numbers (NaN, inf, -inf)
    # Propagates from math.isfinite()
    if not _is_finite(number):
        return None

    # Round floats to eliminate precision artifacts before string conversion
    # Only affects floats; integers pass through unchanged
    if round_digits is not None and isinstance(number, float):
        number = round(number, round_digits)

    # Handle zero (including -0.0) after rounding
    if number == 0:
        return 1

    # Convert to absolute value string
    str_number = str(abs(number))

    # Handle Python's scientific notation string format (e.g., "1.23e-10")
    if 'e' in str_number.lower():
        # Extract mantissa before 'e'
        mantissa = str_number.lower().split('e')[0]

        # Remove decimal point and trailing zeros from mantissa
        digits = mantissa.replace('.', '').rstrip('0')

        # Remove any leading zeros (defensive, shouldn't occur in mantissa)
        digits = digits.lstrip('0')

        return max(len(digits), 1)

    # Standard decimal representation
    # Remove decimal point to get all digits
    digits = str_number.replace('.', '')

    # Remove trailing zeros for display compactness
    digits = digits.rstrip('0')

    # Remove leading zeros (e.g., from "0.00123" → "000123" → "123")
    digits = digits.lstrip('0')

    # Ensure at least 1 digit for any finite number
    return max(len(digits), 1)


# Module Sanity Checks -------------------------------------------------------------------------------------------------

# Ensure the exponent keys are synchronized.
if set(si_prefixes.keys()) != set(value_multipliers.keys()):
    raise AssertionError(
        "Configuration Error: The exponent keys for si_prefixes and value_multipliers must be identical."
    )
