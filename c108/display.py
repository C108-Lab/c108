"""
Numeric display formatting tools for terminal UI, progress bars, status displays, etc
"""

# Standard library -----------------------------------------------------------------------------------------------------
import math
import collections.abc as abc
from dataclasses import dataclass, InitVar, field
from enum import StrEnum, unique
from functools import cached_property
from typing import Any, Mapping, Protocol, Self, runtime_checkable

# Local ----------------------------------------------------------------------------------------------------------------

from .collections import BiDirectionalMap

from .tools import fmt_type, fmt_value, dict_get, fmt_any, fmt_sequence


@runtime_checkable
class SupportsFloat(Protocol):
    """Protocol for duck-typed numeric conversion."""

    def __float__(self) -> float: ...


# @formatter:off

class DisplayConf:
    PLURAL_UNITS = {
        "byte": "bytes", "step": "steps", "item": "items",
        "second": "seconds", "minute": "minutes", "hour": "hours",
        "day": "days", "week": "weeks", "month": "months",
        "year": "years", "meter": "meters", "gram": "grams"
    }
    SI_PREFIXES = BiDirectionalMap({
        -12: "p", -9: "n", -6: "µ", -3: "m", 0: "",
        3: "k", 6: "M", 9: "G", 12: "T", 15: "P", 18: "E", 21: "Z",
    })
    VALUE_MULTIPLIERS = BiDirectionalMap({
        -12: "10⁻¹²", -9: "10⁻⁹", -6: "10⁻⁶", -3: "10⁻³", 0: "",
        3: "10³", 6: "10⁶", 9: "10⁹", 12: "10¹²",
        15: "10¹⁵", 18: "10¹⁸", 21: "10²¹",
    })


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

    Automatically handles value type conversion, exponent calculation, digit
    trimming, and unit pluralization for clean, readable numeric displays in
    terminal UIs, progress bars, and status indicators.

    Value Type Support: Accepts diverse numeric types through duck typing and heuristic detection:
        - *Python stdlib:* int, float, None, Decimal, Fraction, math.inf/nan
        - *NumPy:* int8-64, uint8-64, float16-128, numpy.nan/inf, array scalars
        - *Pandas:* numeric scalars, pd.NA (converted to nan)
        - *ML frameworks:* PyTorch/TensorFlow/JAX tensor scalars (via .item())
        - *Scientific:* Astropy Quantity (extracts .value, discards units)
        - *Any type with __float__():* SymPy expressions, mpmath, etc.

    All external types are normalized to Python int/float/None internally.
    Booleans are explicitly rejected to prevent confusion (True → 1).

    Display Modes: Four main display modes are inferred from init options:
        - BASE_FIXED: Base units with multipliers → "123×10⁹ bytes"
        - PLAIN: Raw values, no scaling → "123000000 bytes"
        - SI_FIXED: Fixed SI prefix + multipliers → "123×10³ Mbytes"
        - SI_FLEX: Auto-scaled SI prefix → "123 Mbytes"

    Formatting Pipeline:
        - Handle non-finite numerics
        - Apply trim rules (optional)
        - Apply whole_as_int rule (optional)
        - Apply precision formatting (optional)

    **Factory Methods (Recommended):**
        - `DisplayValue.base_fixed()` - Base units with multipliers
        - `DisplayValue.plain()` - Plain number display
        - `DisplayValue.si_fixed()` - Fixed SI prefix
        - `DisplayValue.si_flex()` - Auto-scaled SI prefix

    Attributes:
        value: Numeric value (int/float/None). Automatically converted from
               external types (NumPy, Pandas, Decimal, etc.) to stdlib types.
        unit: Base unit name (e.g., "byte", "second"). Auto-pluralized.
        precision: Fixed decimal places for floats. Takes precedence over trim_digits.
                   Use for consistent decimal display (e.g., "3.14" with precision=2).
        trim_digits: Override auto-calculated digit count for rounding. Used when
                    precision is None. Controls compact display digit count.
        multi_symbol: Multiplier symbol (×, ⋅, *) for scientific notation.
        plural_units: Enable auto-pluralization or provide custom mapping.
        separator: Separator between number and unit (default: space).
        whole_as_int: Display whole floats as integers (3.0 → "3").

    Examples:
        # Basic usage with different types
        DisplayValue(42, unit="byte")                    # → "42 bytes"
        DisplayValue(np.int64(42), unit="byte")          # → "42 bytes" (NumPy)
        DisplayValue(Decimal("3.14"), unit="meter")      # → "3.14 meters"

        # Precision vs trim_digits
        dv = DisplayValue(1/3, unit="s", precision=2)    # → "0.33 s" (fixed 2 decimals)
        dv = DisplayValue(1/3, unit="s", trim_digits=5)  # → "0.33333 s" (5 digits)
        dv = DisplayValue(1/3, unit="s")                 # → "0.333333333333333 s" (auto)

        # When both set, precision wins
        dv = DisplayValue(1/3, unit="s", precision=2, trim_digits=10)
        # → "0.33 s" (precision=2 takes precedence)

        # Factory methods (recommended)
        DisplayValue.si_flex(1_500_000, unit="byte")     # → "1.5 Mbytes"
        DisplayValue.base_fixed(1_500_000, unit="byte")  # → "1.5×10⁶ bytes"
        DisplayValue.plain(1_500_000, unit="byte")       # → "1500000 bytes"

    See Also:
        - trimmed_digits() - Function for auto-calculating display digits
        - _std_numeric() - Value type conversion implementation
    """

    value: int | float | None
    unit: str | None = None

    mult_exp: InitVar[int | str | None] = None
    unit_exp: InitVar[int | str | None] = None
    trim_digits: InitVar[int | None] = None

    multi_symbol: str = MultiSymbol.CROSS
    plural_units: Mapping[str, str] | None = None
    pluralize: bool = True
    precision: int | None = None
    separator: str = " "
    whole_as_int: bool | None = None

    si_prefixes: Mapping[int, str] | None = None
    value_multipliers: Mapping[int, str] | None = None

    _mult_exp: int | None = field(init=False, default=None, repr=False)
    _unit_exp: int | None = field(init=False, default=None, repr=False)
    _trim_digits: int | None = field(init=False, default=None, repr=False)

    def __post_init__(
            self,
            mult_exp: int | str | None,
            unit_exp: int | str | None,
            trim_digits: int | None
    ):
        # Validate SI prefixes and value multipliers
        self._validate_prefixes_multipliers()

        # Value, trim and precision
        value_ = _std_numeric(self.value)
        object.__setattr__(self, 'value', value_)
        self._validate_trim_and_precision(trim_digits)

        # Parse exponents to int or None
        mult_exp = self._parse_exponent_value(mult_exp)
        unit_exp = self._parse_exponent_value(unit_exp)

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
    def base_fixed(
            cls,
            value: int | float | None = None,
            trim_digits: int | None = None,
            precision: int | None = None,
            *,
            unit: str
    ) -> Self:
        """
        Create DisplayValue with base units and flexible value multiplier.

        Displays numbers in base units (byte, second, meter) with scientific notation
        multipliers (×10³, ×10⁶, etc.) when the value is large or small. The multiplier
        auto-scales to keep the normalized value compact (typically 1-999).

        Display mode: BASE_FIXED
        Format: `{normalized_value}×10ⁿ {base_unit}` or `{value} {base_unit}` if no scaling needed

        Formatting Pipeline:
            - Handle non-finite numerics
            - Apply trim rules (optional)
            - Apply precision formatting (optional)

        Args:
            value: Numeric value in base units. Accepts int, float, None, or any
                   type convertible via _std_numeric() (NumPy, Pandas, Decimal, etc.).
                   All external types are normalized to Python int/float/None.
            unit: Base unit name (e.g., "byte", "second", "meter"). REQUIRED.
                  Will be automatically pluralized for values != 1 if plural_units=True.
            trim_digits: Override auto-calculated display digits. If None, uses
                         trimmed_digits() to determine minimal representation.
            precision: Number of decimal places for float display. Use for consistent
                       decimal formatting (e.g., precision=2 always shows "X.XX" format).

        Returns:
            DisplayValue configured for base unit display with scientific multipliers.


        Examples:
            # Large values get multipliers
            DisplayValue.base_fixed(123_000_000_000, unit="byte")
            # → "123×10⁹ bytes"

            # NumPy/Pandas types auto-converted
            DisplayValue.base_fixed(np.int64(5_500_000), unit="byte")
            # → "5.5×10⁶ bytes"

            DisplayValue.base_fixed(pd.Series([1e9]).item(), unit="byte")
            # → "1×10⁹ bytes"

            # Precision takes precedence over trim_digits
            DisplayValue.base_fixed(123_456_789, unit="byte", precision=2)
            # → "123.46×10⁶ bytes" (exactly 2 decimal places)

            DisplayValue.base_fixed(123_456_789, unit="byte", trim_digits=5)
            # → "123.457×10⁶ bytes" (5 significant digits)

            DisplayValue.base_fixed(123_456_789, unit="byte", precision=2, trim_digits=10)
            # → "123.46×10⁶ bytes" (precision wins)

            # Decimal type support
            from decimal import Decimal
            DisplayValue.base_fixed(Decimal("1.5e9"), unit="byte")
            # → "1.5×10⁹ bytes"

            # Small values
            DisplayValue.base_fixed(0.000123, unit="second")
            # → "123×10⁻⁶ seconds"

            # No multiplier for moderate values
            DisplayValue.base_fixed(42, unit="byte")
            # → "42 bytes"

        See Also:
            - plain() - For plain number display without multipliers
            - si_flex() - For auto-scaled SI prefixes (KB, MB, GB)
            - si_fixed() - For fixed SI prefix with multipliers
        """
        return cls(
            value=value,
            trim_digits=trim_digits,
            precision=precision,
            unit=unit,
            unit_exp=0,
        )

    @classmethod
    def plain(
            cls,
            value: int | float | None = None,
            trim_digits: int | None = None,
            precision: int | None = None,
            *,
            unit: str
    ) -> Self:
        """
        Create DisplayValue with plain number display in base units.

        Displays integers as-is and floats in Python's default E-notation for very
        large or small values. No scientific notation multipliers (×10ⁿ) are added.
        This is the simplest, most straightforward display format.

        Display mode: PLAIN
        Format: `{value} {base_unit}` (ints) or `{value:e} {base_unit}` (floats with E-notation)

        Formatting Pipeline:
            - Handle non-finite numerics
            - Apply trim rules (optional)
            - Apply precision formatting (optional)

        Args:
            value: Numeric value in base units. Accepts int, float, None, or any
                   type convertible via _std_numeric() (NumPy, Pandas, Decimal, etc.).
                   All external types are normalized to Python int/float/None.
            unit: Base unit name (e.g., "byte", "second", "meter"). REQUIRED.
                  Will be automatically pluralized for values != 1 if plural_units=True.
            trim_digits: Override auto-calculated display digits. If None, uses
                         trimmed_digits() to determine minimal representation.
            precision: Number of decimal places for float display. Use for consistent
                       decimal formatting (e.g., precision=2 always shows "X.XX" format).

        Returns:
            DisplayValue configured for plain display without multipliers.

        Examples:
            # Integers display as-is
            DisplayValue.plain(123_000_000, unit="byte")
            # → "123000000 bytes"

            # NumPy/Pandas types auto-converted
            DisplayValue.plain(np.float64(42.5), unit="item")
            # → "42.5 items"

            DisplayValue.plain(pd.NA, unit="byte")
            # → "nan bytes" (pd.NA converted to float('nan'))

            # Precision control for floats
            DisplayValue.plain(3.14159, unit="meter", precision=2)
            # → "3.14 meters" (exactly 2 decimals)

            DisplayValue.plain(3.14159, unit="meter", trim_digits=4)
            # → "3.142 meters" (4 significant digits)

            # Precision takes precedence
            DisplayValue.plain(3.14159, unit="meter", precision=2, trim_digits=10)
            # → "3.14 meters" (precision wins)

            # Decimal/Fraction support
            from decimal import Decimal
            from fractions import Fraction
            DisplayValue.plain(Decimal("3.14159"), unit="meter", precision=2)
            # → "3.14 meters"

            DisplayValue.plain(Fraction(22, 7), unit="meter", precision=3)
            # → "3.143 meters"

            # Auto-trimming for clean display
            DisplayValue.plain(123.4560, unit="second")
            # → "123.456 seconds" (trailing zero auto-removed)

            # Singular/plural handling
            DisplayValue.plain(1, unit="step")
            # → "1 step" (singular)

            DisplayValue.plain(2, unit="step")
            # → "2 steps" (plural)

        See Also:
            - base_fixed() - For scientific multipliers (×10ⁿ) with base units
            - si_flex() - For human-readable SI prefixes (KB, MB, ms, µs)
            - si_fixed() - For fixed SI prefix display
        """
        return cls(
            value=value,
            trim_digits=trim_digits,
            precision=precision,
            unit=unit,
            mult_exp=0,
            unit_exp=0,
        )

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
        """
        Create DisplayValue with fixed SI prefix and flexible multiplier.

        The si_unit parameter determines both the unit and the fixed SI prefix.
        Value multipliers (×10ⁿ) are added when the magnitude requires additional scaling.

        Formatting Pipeline:
            - Handle non-finite numerics
            - Apply trim rules (optional)
            - Apply precision formatting (optional)

        Args:
            value: Numeric value IN BASE UNITS. Mutually exclusive with si_value.
                   Accepts int, float, None, or any type convertible via _std_numeric()
                   (NumPy, Pandas, Decimal, etc.). Use when you have data in base units
                   (bytes, seconds, meters).
            si_value: Numeric value IN SI-PREFIXED UNITS. Mutually exclusive with value.
                     Accepts same types as value. Use when you have data already in
                     SI units (megabytes, milliseconds).
            si_unit: SI-prefixed unit string (e.g., "Mbyte", "ms", "km"). REQUIRED.
                    Specifies both the base unit and the fixed SI prefix.
            trim_digits: Override auto-calculated display digits. If None, uses
                         trimmed_digits() to determine minimal representation.
            precision: Number of decimal places for float display. Use for consistent
                       decimal formatting (e.g., precision=2 always shows "X.XX" format).

        Returns:
            DisplayValue with fixed SI prefix and flexible multiplier if needed.

        Raises:
            ValueError: If both value and si_value are provided, or if neither is provided.
            TypeError: If value/si_value type cannot be converted to numeric.

        Examples:
            # From base units (123 million bytes):
            DisplayValue.si_fixed(value=123_000_000, si_unit="Mbyte")
            # → "123 Mbyte" or "123×10³ Mbyte" depending on magnitude

            # From SI units (123 megabytes):
            DisplayValue.si_fixed(si_value=123, si_unit="Mbyte")
            # → "123 Mbyte" (internally converts to 123_000_000 base units)

            # NumPy/Pandas types auto-converted
            DisplayValue.si_fixed(value=np.int64(500_000_000), si_unit="Mbyte")
            # → "500 Mbyte"

            DisplayValue.si_fixed(si_value=pd.Series([500]).item(), si_unit="Mbyte")
            # → "500 Mbyte"

            # Precision control
            DisplayValue.si_fixed(value=123_456_789, si_unit="Mbyte", precision=2)
            # → "123.46 Mbyte"

            DisplayValue.si_fixed(value=123_456_789, si_unit="Mbyte", trim_digits=4)
            # → "123.5 Mbyte" (4 significant digits)

            # Decimal/Fraction support
            from decimal import Decimal
            DisplayValue.si_fixed(si_value=Decimal("123.456"), si_unit="Mbyte")
            # → "123.456 Mbyte"

            # Fractional units
            DisplayValue.si_fixed(si_value=500, si_unit="Mbyte/s")
            # → "500 Mbyte/s"

            # Error handling
            DisplayValue.si_fixed(value=100, si_value=200, si_unit="Mbyte")
            # → ValueError: cannot specify both value and si_value

        See Also:
            - si_flex() - For automatically scaled SI prefixes
            - base_fixed() - For base units with value multipliers
            - _std_numeric() - Value type conversion details
        """
        # Validation
        if value is not None and si_value is not None:
            raise ValueError("only one of 'value' or 'si_value' allowed, not both.")

        # Parse si_unit to extract prefix and base unit
        prefix, base_unit = _parse_si_unit_string(si_unit)
        exp = DisplayConf.SI_PREFIXES.get_key(prefix) if prefix else 0

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

    @classmethod
    def si_flex(
            cls,
            value: int | float | None = None,
            trim_digits: int | None = None,
            precision: int | None = None,
            *,
            unit: str
    ) -> Self:
        """
        Create DisplayValue with automatically scaled SI prefix.

        Auto-scales to the most appropriate SI prefix (k, M, G, m, µ, n, etc.) to
        keep the displayed value compact and human-readable. This is the most
        user-friendly format for displaying sizes, durations, and measurements.

        No value multipliers (×10ⁿ) are shown - the SI prefix handles all scaling.

        Display mode: SI_FLEX
        Format: `{normalized_value} {SI_prefix}{base_unit}`

        Formatting Pipeline:
            - Handle non-finite numerics
            - Apply trim rules (optional)
            - Apply precision formatting (optional)

        Args:
            value: Numeric value IN BASE UNITS. Accepts int, float, None, or any
                   type convertible via _std_numeric() (NumPy, Pandas, Decimal, etc.).
                   All external types are normalized to Python int/float/None.
                   The function will automatically determine the best SI prefix.
            unit: Base unit name without SI prefix (e.g., "byte", "second", "meter").
                  REQUIRED. The SI prefix will be prepended automatically.
            trim_digits: Override auto-calculated display digits. If None, uses
                         trimmed_digits() to determine minimal representation.
            precision: Number of decimal places for float display. Use for consistent
                       decimal formatting (e.g., precision=2 always shows "X.XX" format).

        Returns:
            DisplayValue configured with optimal SI prefix for the value's magnitude.

        Examples:
            # Large byte values auto-scale
            DisplayValue.si_flex(1_500_000_000, unit="byte")
            # → "1.5 Gbytes" (giga = 10⁹)

            # NumPy/Pandas types auto-converted
            DisplayValue.si_flex(np.int64(250_000), unit="byte")
            # → "250 kbytes" (kilo = 10³)

            DisplayValue.si_flex(pd.Series([42]).item(), unit="byte")
            # → "42 bytes" (no prefix for moderate values)

            # Precision control - consistent decimals
            DisplayValue.si_flex(1_234_567_890, unit="byte", precision=2)
            # → "1.23 Gbytes" (exactly 2 decimal places)

            DisplayValue.si_flex(1_234_567_890, unit="byte", trim_digits=4)
            # → "1.235 Gbytes" (4 significant digits)

            DisplayValue.si_flex(1_234_567_890, unit="byte", precision=2, trim_digits=10)
            # → "1.23 Gbytes" (precision wins)

            # Time durations with appropriate prefixes
            DisplayValue.si_flex(0.000123, unit="second")
            # → "123 µs" (micro = 10⁻⁶)

            DisplayValue.si_flex(0.000000456, unit="second")
            # → "456 ns" (nano = 10⁻⁹)

            # Decimal/Fraction support
            from decimal import Decimal
            DisplayValue.si_flex(Decimal("1500"), unit="meter")
            # → "1.5 km" (kilo = 10³)

            from fractions import Fraction
            DisplayValue.si_flex(Fraction(25, 10), unit="meter")
            # → "2.5 m" or "2500 mm" depending on auto-scaling

            # Astropy Quantity (extracts .value, discards units)
            from astropy import units as u
            DisplayValue.si_flex(1500 * u.meter, unit="meter")
            # → "1.5 km" (extracts 1500, auto-scales, YOUR unit must match!)

            # ML framework tensors
            import torch
            DisplayValue.si_flex(torch.tensor(1500.0), unit="meter")
            # → "1.5 km"

        Note:
            The SI prefix is selected to keep the normalized value typically in the
            range 1-999 for optimal readability. Supported prefixes range from pico
            (10⁻¹²) to zetta (10²¹).

            For Astropy Quantity objects, only the numeric magnitude is extracted.
            Unit information is DISCARDED - ensure your Quantity's units are compatible
            with the specified 'unit' parameter before conversion.

        See Also:
            - si_fixed() - For fixed SI prefix with flexible multipliers
            - base_fixed() - For base units with value multipliers (×10ⁿ)
            - plain() - For plain display without any scaling
        """
        return cls(
            value=value,
            trim_digits=trim_digits,
            precision=precision,
            unit=unit,
            mult_exp=0,
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

        This property implements the precedence logic for determining how many
        digits to use when formatting the normalized value for display.

        **Precedence Logic:**
        1. If trim_digits was specified during initialization: Use that value
        2. Else: Auto-calculate using trimmed_digits(value, round_digits=15)

        Returns:
            int: Number of digits for display (minimum 1 for finite values).
            None: If value is None or non-finite (NaN, infinity).

        **Formatting Pipeline:**
            - Handle non-finite numerics
            - Apply trim rules (optional)
            - Apply whole_as_int rule (optional)
            - Apply precision formatting (optional)

        Examples:
            # Auto-calculated display_digits
            dv = DisplayValue(123000, unit="byte")
            dv.display_digits == 3  # trimmed_digits(123000) = 3

            # User-specified trim_digits
            dv = DisplayValue(123000, unit="byte", trim_digits=5)
            dv.display_digits == 5  # User override

            # With precision set (precision takes precedence in formatting)
            dv = DisplayValue(1/3, unit="s", precision=2, trim_digits=10)
            dv.display_digits == 10  # Returns trim_digits
            str(dv) == "0.33 s"      # But precision=2 used for actual formatting

            # Float precision artifacts handled automatically
            dv = DisplayValue(0.1 + 0.2, unit="m")
            dv.display_digits == 1  # Auto-rounds to 0.3, then trims to 1 digit

            # None for non-finite values
            dv = DisplayValue(float('inf'), unit="byte")
            dv.display_digits is None

        See Also:
            - trimmed_digits() - Function for calculating display digits from numbers
            - precision - Parameter for fixed decimal place formatting
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
        return self._si_prefixes[self.unit_exponent]

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

        return f"{self.multi_symbol}{self._value_multipliers[self.multiplier_exponent]}"

    @property
    def _number_str(self) -> str:
        """
        Numerical part of full str-representation including the multiplier if applicable.

        **Formatting Pipeline:**
            - Handle non-finite numerics
            - Apply trim rules (optional)
            - Apply whole_as_int rule (optional)
            - Apply precision formatting (optional)

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

        if not self.pluralize:
            return f"{self.si_prefix}{self.unit}"

        plural_units = self._get_plural_units()
        units_ = dict_get(plural_units, key=self.unit, default=self.unit)
        return f"{self.si_prefix}{units_}"

    @cached_property
    def _si_prefixes(self) -> BiDirectionalMap[int, str]:
        """Returns the appropriate SI prefix map based on the configuration."""
        return BiDirectionalMap(self.si_prefixes) if self.si_prefixes else DisplayConf.SI_PREFIXES

    @cached_property
    def _value_multipliers(self) -> BiDirectionalMap[int, str]:
        """Returns the appropriate SI prefix map based on the configuration."""
        return BiDirectionalMap(self.value_multipliers) if self.value_multipliers else DisplayConf.VALUE_MULTIPLIERS

    @cached_property
    def _valid_exponents(self) -> tuple[int, ...]:
        return tuple(self._si_prefixes.keys())

    @cached_property
    def _valid_si_prefixes(self) -> tuple[str, ...]:
        return tuple(self._si_prefixes.values())

    def _get_plural_units(self) -> Mapping[str, str]:
        """Returns the appropriate plural map based on the configuration."""
        if isinstance(self.plural_units, abc.Mapping):
            return self.plural_units
        else:
            return DisplayConf.PLURAL_UNITS

    def _parse_exponent_value(self, exp: int | str | None) -> int | None:
        """
        Parse an exponent from a SI prefix string or validate and return its int power.

        Raises:
             ValueError if exponent is not a valid SI prefix or is invalid int power.

        Examples:
            >>> DisplayValue._parse_exponent_value("1000")
            1000
            >>> DisplayValue._parse_exponent_value("M")
            6
        """
        if exp is None:
            return None

        elif isinstance(exp, str):
            if exp not in self._valid_si_prefixes:
                raise ValueError(
                    f"Invalid exponent str value: '{exp}', expected one of {self._valid_si_prefixes}"
                )
            return self._si_prefixes.get_key(exp)

        elif isinstance(exp, int):
            if exp not in self._valid_exponents:
                raise ValueError(
                    f"Invalid exponent int value: {exp}, expected one of {self._valid_exponents}"
                )
            return exp

        raise TypeError(f"Exponent value must be an int | str | None, got {fmt_type(exp)}")

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

    def _validate_prefixes_multipliers(self):
        """Validate keys for si_prefixes and value_multipliers mappings"""
        # Ensure the exponent keys are synchronized.
        if set(self._si_prefixes.keys()) != set(self._value_multipliers.keys()):
            raise AssertionError(
                f"mapping keys mismatch. The keys set in 'si_prefixes' and 'value_multipliers' "
                f"must be identical, but found: si_prefixes.keys {fmt_any(self._si_prefixes.keys())} "
                f"and value_multipliers.keys {fmt_any(self._value_multipliers.keys())}"
            )

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
    if first_char in DisplayConf.SI_PREFIXES.values() and len(si_unit) > 1:
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
    Convert common numeric types to standard Python int, float, or None.

    Normalizes various numeric representations from Python stdlib and popular
    external libraries into standard types suitable for display formatting.
    Preserves special float values (inf, -inf, nan).

    Supported types (via duck typing, no imports required):

    **Python stdlib:**
    - int, float, None
    - math.inf, math.nan (as float)
    - decimal.Decimal (via __float__)
    - fractions.Fraction (via __float__)

    **External libraries (detected heuristically):**
    - NumPy: int8-64, uint8-64, float16-128, numpy.nan, numpy.inf
    - Pandas: scalars, NA (converted to nan)
    - PyTorch: tensor.item() scalars
    - TensorFlow: tensor scalars
    - JAX: DeviceArray scalars
    - Astropy: Quantity.value (physical quantities with units)
    - SymPy: expressions that can evaluate to float
    - mpmath: arbitrary precision floats

    Args:
        value: Numeric value to convert. Accepts:
               - Python int, float, None
               - Any type implementing __float__()
               - Types with .item() method (array scalars)
               - Types with .value attribute (physical quantities)

    Returns:
        int: For Python int values (excluding bool)
        float: For float values, including inf, -inf, nan
        None: For None input or NA-like values

    Raises:
        TypeError: If value is bool or cannot be converted to numeric type.
                   If value has __float__ but raises during conversion.

    Examples:
        # Python stdlib types
        _std_numeric(42) == 42
        _std_numeric(3.14) == 3.14
        _std_numeric(None) is None
        _std_numeric(math.inf) == float('inf')
        _std_numeric(math.nan)  # returns nan (nan != nan)

        # Decimal and Fraction (stdlib)
        from decimal import Decimal
        from fractions import Fraction
        _std_numeric(Decimal('3.14')) == 3.14
        _std_numeric(Fraction(22, 7)) == 3.142857142857143

        # NumPy (if available)
        import numpy as np
        _std_numeric(np.int64(42)) == 42
        _std_numeric(np.float32(3.14)) == 3.14
        _std_numeric(np.nan)  # returns float('nan')
        _std_numeric(np.inf) == float('inf')

        # Pandas (if available)
        import pandas as pd
        _std_numeric(pd.NA) is math.nan  # False (nan != nan) but type is float

        # PyTorch tensor scalar (if available)
        import torch
        _std_numeric(torch.tensor(3.14)) == 3.14

        # Astropy Quantity (if available)
        from astropy import units as u
        _std_numeric(3.14 * u.meter) == 3.14  # Extracts numeric value

        # Error cases
        _std_numeric(True)  # raises TypeError: boolean values not supported
        _std_numeric("123")  # raises TypeError: unsupported numeric type
        _std_numeric([1, 2, 3])  # raises TypeError: unsupported numeric type

    Note:
        For array/tensor types, only scalars (0-dimensional or single-element)
        are supported. Multi-element arrays will raise TypeError.

        For Astropy Quantity objects, only the numeric magnitude is extracted.
        Unit information is discarded - ensure units are compatible with your
        DisplayValue.unit before conversion.
    """
    # None passthrough
    if value is None:
        return None

    # Reject boolean explicitly (bool is subclass of int in Python)
    if isinstance(value, bool):
        raise TypeError(f"boolean values not supported, got {value}")

    # Standard Python numeric types - fast path
    if isinstance(value, (int, float)):
        return value

    # Check for pandas.NA (singleton) - must check before duck typing
    # pandas.NA has __float__ but raises TypeError, so handle specially
    if hasattr(value, '__class__') and value.__class__.__name__ == 'NAType':
        return math.nan

    # Handle array/tensor scalars with .item() method
    # Common in NumPy, PyTorch, TensorFlow, JAX
    if hasattr(value, 'item') and callable(value.item):
        try:
            # .item() returns Python scalar (int or float)
            result = value.item()
            # Recursively process in case .item() returns non-standard type
            if isinstance(result, (int, float, type(None))):
                return result if not isinstance(result, bool) else None
            # If .item() returned something else, fall through to __float__
        except (TypeError, ValueError):
            # .item() failed, try other methods
            pass

    # Handle Astropy Quantity objects (physical quantities with units)
    # These have .value attribute containing the numeric magnitude
    if hasattr(value, 'value') and hasattr(value, 'unit'):
        # Heuristic: object has both .value and .unit attributes
        # Strong indicator of Astropy Quantity or similar
        try:
            magnitude = value.value
            # Recursively process the magnitude
            return _std_numeric(magnitude)
        except (TypeError, ValueError, AttributeError):
            # Not actually a Quantity-like object, fall through
            pass

    # Duck typing: anything with __float__
    # Handles: Decimal, Fraction, NumPy scalars, SymPy, mpmath, etc.
    if hasattr(value, '__float__'):
        try:
            result = float(value)
            # numpy.nan converts to float('nan')
            # numpy.inf converts to float('inf')
            # Decimal, Fraction, etc. convert normally
            return result
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"cannot convert {fmt_type(value)} to float: {e}"
            ) from e

    # If we get here, type is not supported
    raise TypeError(
        f"unsupported numeric type: {fmt_type(value)}. "
        f"Expected int, float, None, or types implementing __float__, .item(), "
        f"or having .value attribute (e.g., numpy scalars, Decimal, Fraction, "
        f"pandas scalars, array.item(), Quantity.value)"
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
if set(DisplayConf.SI_PREFIXES.keys()) != set(DisplayConf.VALUE_MULTIPLIERS.keys()):
    raise AssertionError(
        "Configuration Error: The exponent keys for si_prefixes and keys for value_multipliers must be identical."
    )
