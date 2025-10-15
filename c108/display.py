"""
Display formatting tools for terminal UI, progress bars, status displays, etc
"""

# Standard library -----------------------------------------------------------------------------------------------------
import math
from dataclasses import dataclass, InitVar, field
from enum import StrEnum, unique
from typing import Any, Self, Protocol, runtime_checkable

# Local ----------------------------------------------------------------------------------------------------------------

from .collections import BiDirectionalMap

from .tools import fmt_type


@runtime_checkable
class SupportsFloat(Protocol):
    """Protocol for duck-typed numeric conversion."""

    def __float__(self) -> float: ...


# @formatter:off

class UnitsConf:
    PLURALS = {
        "byte": "bytes", "step": "steps", "item": "items",
        "second": "seconds", "minute": "minutes", "hour": "hours",
        "day": "days", "week": "weeks", "month": "months",
        "year": "years", "meter": "meters", "gram": "grams"
    }


units_conf = UnitsConf()

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
        - DisplayValue.base_fixed() for base units with multiplier
        - DisplayValue.plain() for plain number display
        - DisplayValue.si_fixed() for fixed SI prefix
        - DisplayValue.si_flex() for auto-scaled SI prefix
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

    def _asstr__TODO_(self):
        """Format value with unit, handling special cases."""

        # Handle None
        if self.value is None:
            return "N/A" if self.unit is None else f"N/A {self.unit}"

        # Handle infinity
        if math.isinf(self.value):
            sign = "" if self.value > 0 else "-"
            unit_str_ = f" {self._format_unit()}" if self.unit else ""
            return f"{sign}∞{unit_str_}"

        # Handle NaN
        if math.isnan(self.value):
            unit_str = f" {self._format_unit()}" if self.unit else ""
            return f"NaN{unit_str}"

        # Normal numeric formatting
        return self._format_normal_value()

    def merge(self, **kwargs) -> Self:
        """
        Create new instance with updated formatting options.
        """

    @classmethod
    def si_flex(cls, value: int | float, *, unit: str) -> Self:
        """
        Auto-scale to appropriate SI prefix (MB, GB, ms, µs, etc.).
        """

    @classmethod
    def si_fixed(
            cls,
            value: int | float | None = None,
            si_value: int | float | None = None,
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
            precision: Number of decimal digits for float display.
            trim_digits: Trimmed digits for rounding (auto-calculated if None).
            whole_as_int: Display whole floats as integers (default: True).
            multi_symbol: Multiplication operator symbol.
            separator: Separator between number and unit.
            plural_units: Plural map (dict) or True for default plurals, False to disable.

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
        prefix, base_unit = cls._parse_si_unit_string(si_unit)
        exp = si_prefixes.get_key(prefix) if prefix else 0

        # Convert si_value to base units if provided
        if si_value is not None:
            value = si_value * (10 ** exp)

        return cls(
            value=value,
            unit=base_unit,
            unit_exp=exp,
        )

    @staticmethod
    def _parse_si_unit_string(si_unit: str) -> tuple[str, str]:
        """Parse SI unit string into (prefix, base_unit).

        Examples:
            "Mbyte" → ("M", "byte")
            "ms" → ("m", "s")
            "byte" → ("", "byte")
            "km/h" → ("k", "m/h")
        """
        if not isinstance(si_unit, str) or not si_unit:
            raise ValueError(f"si_unit must be a non-empty string, got: {si_unit}")

        first_char = si_unit[0]

        # Check if first character is a valid SI prefix
        if first_char in valid_si_prefixes and len(si_unit) > 1:
            return first_char, si_unit[1:]
        else:
            # No SI prefix, entire string is the base unit
            return "", si_unit

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
        if self.value is None:
            return None

        value_ = self.value / self.ref_value
        norm_number = _normalized_number(value_,
                                         trim_digits=self.trimmed_digits,
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
    def trimmed_digits(self) -> int | None:
        """
        Number of trimmed digits for rounding in string representation.

        Returns:
            User-specified value if set via trim_digits parameter, otherwise
            auto-calculated from source value using trimmed_digits() function.

        See Also:
            trimmed_digits() - Function for calculating trimmed digits from a number
        """
        if self._trim_digits is not None:
            return self._trim_digits

        return self._src_significant_digits

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

        if display_number is None:
            return str(display_number)

        if self.precision is not None:
            return f"{display_number:.{self.precision}f}{self._multiplier_str}"

        if self.whole_as_int or isinstance(display_number, int):
            return f"{display_number}{self._multiplier_str}"
        else:
            return f"{display_number:.{self.trimmed_digits}g}{self._multiplier_str}"

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

        # Check if we should pluralize (normalized value != ±1)
        elif abs(self.normalized) == 1:
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
            return UnitsConf.PLURALS
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
    def _src_significant_digits(self) -> int | None:
        """
        Calculate trimmed("significant") digits based on the source DisplayValue.value with trimmed_digits()

        Ignore trailing zeros in float as non-significant both before and after the decimal point
        """
        return trimmed_digits(self.value)


# Methods --------------------------------------------------------------------------------------------------------------

def _is_finite(value: Any) -> bool:
    """Checks if a value is a Python stdlib finite int or float.

    Args:
        value: The value to check.

    Returns:
        True if the value is an int or a finite float, False otherwise.
        Excludes booleans, None, NaN, infinity, other numerics.
    """
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
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

    raise TypeError(f"Exponent value must be an int | str | None, got {type(exp)}")


def _normalized_number(
        value: int | float | None,
        trim_digits: int | None = None,
        whole_as_int: bool = False
) -> int | float | None:
    """Process value to normalized number by rounding and conditional int conversion."""
    if value is None:
        return None

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


def trimmed_digits(number: int | float | None) -> int | None:
    """
    Calculate trimmed digits by removing trailing zeros from integers and floats.
    All trailing zeros are treated as non-significant in float as well as integers.

    **⚠️ WARNING:** Ignoring trailing zeros BEFORE decimal point in float is non-standard engineering or scientific
    practice so in the context of strict definition trimmed_digits() method should not be used for significant
    digits calculation.

    Args:
        number: The number to analyze. Must be int, float, or None.

    Returns:
        Number of digits after removing trailing zeros, or None if input is None.
        Returns 1 for zero values.

    Raises:
        TypeError: If number is not int, float, or None.

    Examples:
        trimmed_digits(123000) == 3
        trimmed_digits(0.456) == 3
        trimmed_digits(123.456) == 6
        trimmed_digits(0) == 1
    """
    if number is None:
        return None
    elif not isinstance(number, (int, float)):
        raise TypeError(f"Expected int | float | None, got {type(number)}")

    if number == 0:
        return 1

    # Convert to string, removing sign
    str_number = str(abs(number))

    # Handle scientific notation
    if 'e' in str_number.lower():
        mantissa = str_number.lower().split('e')[0]
        digits = mantissa.replace('.', '').rstrip('0')
        return len(digits) if digits else 1

    # Remove decimal point and trailing zeros
    digits = str_number.replace('.', '').rstrip('0')

    # Remove leading zeros
    digits = digits.lstrip('0')

    return len(digits) if digits else 1


# Module Sanity Checks -------------------------------------------------------------------------------------------------

# Ensure the exponent keys are synchronized.
if set(si_prefixes.keys()) != set(value_multipliers.keys()):
    raise AssertionError(
        "Configuration Error: The exponent keys for si_prefixes and value_multipliers must be identical."
    )
