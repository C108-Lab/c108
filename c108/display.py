"""
Numeric display formatting tools for terminal UI, progress bars, status displays, etc
"""

# ## Design Scope
#
# `DisplayValue` is designed for **one-way formatting** (numeric value → human-readable string).
# It is NOT designed for parsing strings back to values.
#
# If you need to serialize/deserialize `DisplayValue` objects:
#   - Store the original numeric value, not the formatted string
#   - Use JSON/pickle to serialize the entire DisplayValue object
#   - Don't rely on `str(dv)` for round-tripping

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
from .unicode import to_sup


@runtime_checkable
class SupportsFloat(Protocol):
    """Protocol for duck-typed numeric conversion."""

    def __float__(self) -> float: ...


# @formatter:off

class DisplayConf:
    BIN_POWER_SCALE = BiDirectionalMap({
        0: "",
        10: "2¹⁰",   # 1,024
        20: "2²⁰",   # ~1 million
        30: "2³⁰",   # ~1 billion
        40: "2⁴⁰",   # ~1 trillion
        50: "2⁵⁰",   # ~1 quadrillion
        60: "2⁶⁰",   # ~1 quintillion
        70: "2⁷⁰",   # ~1 sextillion
        80: "2⁸⁰",   # ~1 septillion
    })
    BIN_SCALE_BASE = 2  # Base in binary scale 2^0, 2^10, 2^20, ...
    BIN_SCALE_STEP = 10 # Step of power in binary scale
    PLURAL_UNITS = {
        "byte": "bytes", "step": "steps", "item": "items",
        "second": "seconds", "minute": "minutes", "hour": "hours",
        "day": "days", "week": "weeks", "month": "months",
        "year": "years", "meter": "meters", "gram": "grams"
    }
    SI_POWER_SCALE = BiDirectionalMap({
        -12: "10⁻¹²", -9: "10⁻⁹", -6: "10⁻⁶", -3: "10⁻³", 0: "",
        3: "10³", 6: "10⁶", 9: "10⁹", 12: "10¹²",
        15: "10¹⁵", 18: "10¹⁸", 21: "10²¹",
    })
    SI_PREFIXES_3N = BiDirectionalMap({
        -12: "p",   # pico
        -9: "n",    # nano
        -6: "µ",    # micro
        -3: "m",    # milli
        0: "",      # (no prefix)
        3: "k",     # kilo
        6: "M",     # mega
        9: "G",     # giga
        12: "T",    # tera
        15: "P",    # peta
        18: "E",    # exa
        21: "Z",    # zetta
        24: "Y",    # yotta
    })
    SI_PREFIXES = BiDirectionalMap({
        # Large (positive powers)
        24: "Y",    # yotta
        21: "Z",    # zetta
        18: "E",    # exa
        15: "P",    # peta
        12: "T",    # tera
        9: "G",     # giga
        6: "M",     # mega
        3: "k",     # kilo
        2: "h",     # hecto
        1: "da",    # deca (or deka)
        0: "",      # (no prefix)
        -1: "d",    # deci
        -2: "c",    # centi
        -3: "m",    # milli
        -6: "µ",    # micro
        -9: "n",    # nano
        -12: "p",   # pico
        -15: "f",   # femto
        -18: "a",   # atto
        -21: "z",   # zepto
        -24: "y",   # yocto
    })
    SI_SCALE_BASE = 10  # Base in SI scale 10^0, 10^10, 10^3, ...
    SI_SCALE_STEP = 3   # Step of power in SI scale


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
        UNIT_FIXED (str) : Fixed units prefix, flexible value multiplier - 123×10³ Mbyte
        UNIT_FLEX (str)  : Flexible units prefix, no value multiplier - 123.4 ns
    """
    BASE_FIXED = "base_fixed"
    PLAIN = "plain"
    UNIT_FIXED = "unit_fixed"
    UNIT_FLEX = "unit_flex"
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
        - *Pandas:* numeric scalars, pd.NA
        - *ML frameworks:* PyTorch/TensorFlow/JAX tensor scalars (via .item())
        - *Scientific:* Astropy Quantity (extracts .value, discards units)
        - *Any type with __float__():* SymPy expressions, mpmath, etc.

    All external types are normalized to Python int/float/None internally.
    Booleans are explicitly rejected to prevent confusion (True → 1).

    Display Modes: Four main display modes are inferred from exponent options:
        - BASE_FIXED: Base units with multipliers → "123×10⁹ bytes"
        - PLAIN: Raw values, no scaling → "123000000 bytes"
        - UNIT_FIXED: Fixed unit prefix + auto-scaled multipliers → "123×10³ Mbytes"
        - UNIT_FLEX: Auto-scaled unit prefix → "123 Mbytes"

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
        pluralize: Use plurals for units of mesurement if display value !=1
                   TODO check that we link to display value, not value in base units
        precision: Fixed decimal places for floats. Takes precedence over trim_digits.
                   Use for consistent decimal display (e.g., "3.14" with precision=2).
        trim_digits: Override auto-calculated digit count for rounding. Used when
                    precision is None. Controls compact display digit count.
        multi_symbol: Multiplier symbol (×, ⋅, *) for scientific notation.
        separator: Separator between number and unit (default: space).
        scale_base: 10 for SI, 2 for binary
        scale_step: 3 for SI, 10 for binary
        whole_as_int: Display whole floats as integers (3.0 → "3").
        unit_plurals: Unit pluralize mapping.

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

    mult_exp: InitVar[int | None] = None
    unit_exp: InitVar[int | None] = None
    trim_digits: InitVar[int | None] = None

    multi_symbol: str = MultiSymbol.CROSS
    pluralize: bool = True
    precision: int | None = None
    separator: str = " "
    whole_as_int: bool | None = None

    scale_base: int | None = None  # 10 for SI, 2 for binary
    scale_step: int | None = None  # 3 for SI, 10 for binary
    unit_prefixes: Mapping[int, str] | None = None
    unit_plurals: Mapping[str, str] | None = None
    value_multipliers: Mapping[int, str] | None = None

    _mult_exp: int | None = field(init=False, default=None, repr=False)
    _unit_exp: int | None = field(init=False, default=None, repr=False)
    _trim_digits: int | None = field(init=False, default=None, repr=False)

    def __post_init__(
            self,
            mult_exp: int | None,
            unit_exp: int | None,
            trim_digits: int | None
    ):

        # Validate and Set exponents (auto-calculate if needed)
        self._validate_and_set_exponents(mult_exp, unit_exp)

        # Validate SI prefixes and value multipliers
        self._validate_prefixes_and_multipliers(self._mult_exp, self._unit_exp)

        # Value
        value_ = _std_numeric(self.value)
        object.__setattr__(self, 'value', value_)

        # Trim
        self._validate_trim(trim_digits)

        # whole_as_int
        if self.whole_as_int is None:
            object.__setattr__(self, 'whole_as_int', self.mode != DisplayMode.PLAIN)

        # precision
        if self.precision is not None and self.precision < 0:
            raise ValueError(f"precision must be >= 0, got {self.precision}")

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
                  Will be automatically pluralized for values != 1 if unit_plurals=True.
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
                  Will be automatically pluralized for values != 1 if unit_plurals=True.
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
        prefix, base_unit = cls._parse_si_unit_string(si_unit)
        exp = DisplayConf.SI_PREFIXES_3N.get_key(prefix) if prefix else 0

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

        Display mode: UNIT_FLEX
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

    @classmethod
    def unit_fixed(
            cls,
            value: int | float | None = None,
            ref_value: int | float | None = None,
            trim_digits: int | None = None,
            precision: int | None = None,
            *,
            ref_unit: str
    ) -> Self:
        """
        Create DisplayValue in reference units with flexible multiplier.

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
            ref_value: Numeric value IN REFERENCE UNITS. Mutually exclusive with value.
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
        """

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
            return DisplayMode.BASE_FIXED if unit_exp == 0 else DisplayMode.UNIT_FIXED
        elif isinstance(mult_exp, int) and unit_exp is None:
            return DisplayMode.UNIT_FLEX
        else:
            raise ValueError(
                f"Invalid exponents state: mult_exp={mult_exp}, unit_exp={unit_exp}. "
                f"At least one must be an integer."
            )

    @property
    def multiplier_exponent(self) -> int:
        """
        Display exponent in UNIT_FIXED display mode, multiplier_exponent = exponent - unit_exponent.
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
    def unit_prefix(self) -> str:
        """
        The SI prefix in units of measurement, e.g., 'm' (milli-), 'k' (kilo-).
        """
        return self._unit_prefixes[self.unit_exponent]

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
    def _scale_base(self) -> int:
        return self.scale_base or DisplayConf.SI_SCALE_BASE

    @property
    def _scale_step(self) -> int:
        return self.scale_step or DisplayConf.SI_SCALE_STEP

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
            if self.unit_prefix:
                return self.unit_prefix
            else:
                return ""

        if not _is_finite(self.normalized):
            if self._unit_exp is not None:
                return f"{self.unit_prefix}{self.unit}"
            else:
                return f"{self.unit}"

        # Check if we should pluralize (normalized value != ±1)
        if abs(self.normalized) == 1:
            return f"{self.unit_prefix}{self.unit}"

        if not self.pluralize:
            return f"{self.unit_prefix}{self.unit}"

        plural_units = self._get_plural_units()
        units_ = dict_get(plural_units, key=self.unit, default=self.unit)
        return f"{self.unit_prefix}{units_}"

    @cached_property
    def _unit_prefixes(self) -> BiDirectionalMap[int, str]:
        """Returns the appropriate SI prefix map based on the configuration."""
        return BiDirectionalMap(self.unit_prefixes) if self.unit_prefixes else DisplayConf.SI_PREFIXES_3N

    @cached_property
    def _value_multipliers(self) -> BiDirectionalMap[int, str]:
        """Returns the appropriate SI prefix map based on the configuration."""

        return BiDirectionalMap(self.value_multipliers) if self.value_multipliers else DisplayConf.SI_POWER_SCALE

    @cached_property
    def _valid_exponents(self) -> tuple[int, ...]:
        return tuple(self._unit_prefixes.keys())

    @cached_property
    def _valid_unit_prefixes(self) -> tuple[str, ...]:
        return tuple(self._unit_prefixes.values())

    def _get_plural_units(self) -> Mapping[str, str]:
        """Returns the appropriate plural map based on the configuration."""
        if isinstance(self.unit_plurals, abc.Mapping):
            return self.unit_plurals
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
            if exp not in self._valid_unit_prefixes:
                raise ValueError(
                    f"Invalid exponent str value: '{exp}', expected one of {self._valid_unit_prefixes}"
                )
            return self._unit_prefixes.get_key(exp)

        elif isinstance(exp, int):
            if exp not in self._valid_exponents:
                raise ValueError(
                    f"Invalid exponent int value: {exp}, expected one of {self._valid_exponents}"
                )
            return exp

        raise TypeError(f"Exponent value must be an int | str | None, got {fmt_type(exp)}")

    @classmethod
    def _parse_si_unit_string(cls, si_unit: str) -> tuple[str, str]:
        """Parse SI unit string into (prefix, base_unit).

        Examples:
            "Mbyte" → ("M", "byte")
            "ms" → ("m", "s")
            "byte" → ("", "byte")
            "km/h" → ("k", "m/h")
        """
        if not isinstance(si_unit, str) or not si_unit:
            raise ValueError(f"si_unit must be a non-empty string, but got: {fmt_value(si_unit)}")

        first_char = si_unit[0]

        # Check if first character is a valid SI prefix
        if first_char in DisplayConf.SI_PREFIXES_3N.values() and len(si_unit) > 1:
            return first_char, si_unit[1:]
        else:
            # No SI prefix, entire string is the base unit
            return "", si_unit

    def _validate_prefixes_or_multipliers(
            self,
            mp: Mapping[int, str],
            name: str = "Mapping") -> BiDirectionalMap[int, str]:
        """
        Validate input of unit_prefixes or value_multiplier
        """
        try:
            bd_map = BiDirectionalMap(mp)
        except ValueError as exc:
            raise ValueError(
                f"cannot create BiDirectionalMap: invalid {name} {fmt_any(mp)}"
            ) from exc

        if len(bd_map) < 1:
            raise ValueError(f"at least one item required in {name}: {fmt_any(mp)}")

        return bd_map

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

    def _validate_prefixes_and_multipliers(self,
                                           mult_exp: int | str | None,
                                           unit_exp: int | str | None,
                                           ):
        """
        Validate unit_prefixes and value_multipliers

        Supported mult_exp/unit_exp pairs: 0/0, None/int, int/None, None/None
        """

        # ** Mapping Requirements for unit_prefixes and value_mulitpliers **
        #
        # * BASE_FIXED mode:.. value_multipliers required (auto-select)
        #                      mult_exp/unit_exp = None/0
        # * PLAIN mode:....... Needs neither
        #                      mult_exp/unit_exp = 0/0
        # * UNIT_FLEX mode:... unit_prefixes required (auto-select)
        #                      mult_exp/unit_exp = int/None
        # * UNIT_FIXED mode:.. value_multipliers required (auto-select), 1 item required from unit_prefixes
        #                      for the key=unit_exp
        #                      mult_exp/unit_exp = None/int

        # BASE_FIXED mode - no prefixes, auto-select mulitpliers
        if (mult_exp is None) and unit_exp == 0:
            if self.value_multipliers is not None:
                self._validate_prefixes_or_multipliers(self.value_multipliers, name="value_multipliers")
            return

        # PLAIN mode - NONE of prefixes or mulitpliers required
        if mult_exp == 0 and unit_exp == 0:
            return

        # UNIT_FLEX mode - unit_prefixes required (auto-select)
        if isinstance(mult_exp, int) and unit_exp is None:
            if self.unit_prefixes is not None:
                self._validate_prefixes_or_multipliers(self.unit_prefixes, name="unit_prefixes")
            return

        # UNIT_FIXED mode - value_multipliers required (auto-select)
        #                   1 item required in unit_prefixes mapped from key=unit_exp
        if mult_exp is None and isinstance(unit_exp, int):
            if self.value_multipliers is not None:
                self._validate_prefixes_or_multipliers(self.value_multipliers, name="value_multipliers")
            if self.unit_prefixes is not None:
                pfx = self._validate_prefixes_or_multipliers(self.unit_prefixes, name="unit_prefixes")
                if unit_exp in pfx:
                    # TODO we actually want an auto-generate from any fixed unit_exp value to unit-str display
                    # OR we could simply pass it as arguments (if they are not available in maps)
                    raise ValueError("XYZ ???")

            return

        raise ValueError(
            f"Invalid exponents state: mult_exp={mult_exp}, unit_exp={unit_exp}. "
            f"At least one must be an integer."
        )

    def _validate_trim(self, trim_digits: int | None):
        """Validate initialization parameters"""
        if not isinstance(trim_digits, (int, type(None))):
            raise ValueError(f"trim_digits must be int | None: {fmt_type(trim_digits)}")
        if isinstance(trim_digits, int) and trim_digits < 1:
            raise ValueError(f"trim_digits must be >= 1, got {trim_digits}")
        # Set trimmed digits
        object.__setattr__(self, '_trim_digits', trim_digits)

    def _validate_and_set_exponents(self, mult_exp: int | None, unit_exp: int | None):
        """
        Validate + set exponents if not provided.

        Supported mult_exp/unit_exp pairs: 0/0, None/int, int/None, None/None
        """
        if type(mult_exp) not in (int, type(None)):
            raise TypeError(f"mult_exp must be int or None, got {type(mult_exp)}")

        if type(unit_exp) not in (int, type(None)):
            raise TypeError(f"unit_exp must be int or None, got {type(unit_exp)}")

        if mult_exp is not None and unit_exp is not None:
            if mult_exp != 0 or unit_exp != 0:
                raise ValueError(
                    f"mult_exp and unit_exp must be 0 if specified both, but found: mult_exp={mult_exp}, unit_exp={unit_exp} "
                    "Supported mult_exp/unit_exp pairs: 0/0, None/int, int/None, None/None."
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

def _disp_power(base: int = 10, *, power: int, format: str = "unicode") -> str:
    """
    Format a power expression for display.

    Args:
        base: The base (typically 10 or 2)
        power: The exponent value
        format: Output format style:
            - "unicode" (default): "10³", "2²⁰" (superscript exponents)
            - "caret": "10^3", "2^20" (ASCII-safe)
            - "exp": "10**3", "2**20" (Python-style)
            - "e": "10e3", "2e20" (engineering notation style)
            - Custom template: ex. "{base}^{power}" with {base} and {power} placeholders

    Returns:
        Formatted power string, or empty string if power is 0.

    Examples:
        >>> _disp_power(3)
        '10³'
        >>> _disp_power(20, base=2)
        '2²⁰'
        >>> _disp_power(3, format="caret")
        '10^3'
        >>> _disp_power(3, format="exp")
        '10**3'
        >>> _disp_power(0)
        ''
        >>> _disp_power(-6, format="unicode")
        '10⁻⁶'
    """
    if power == 0:
        return ""

    if not isinstance(power, int):
        raise TypeError("power must be an int")
    if not isinstance(base, int):
        raise TypeError("base must be an int")
    if not isinstance(format, str):
        raise TypeError("format must be a str")

    formats = {
        "unicode": "{base}{sup_power}",
        "caret": "{base}^{power}",
        "exp": "{base}**{power}",
        "e": "{base}e{power}",
    }

    sup_power = to_sup(power)
    template = formats.get(format, format)

    try:
        return template.format(base=str(base), power=str(power), sup_power=sup_power)
    except KeyError as exc:
        missing = str(exc).strip("'")
        raise ValueError(
            f"Template references unknown placeholder {{{missing}}}. "
            "Allowed placeholders: base, power, sup_power."
        ) from exc
    except ValueError as exc:
        # Covers malformed templates (e.g., unmatched '{' or bad format spec)
        raise ValueError(f"Invalid format template: {exc}") from exc


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


import math
import operator


def _std_numeric(value: int | float | None | SupportsFloat) -> int | float | None:
    """
    Convert common numeric types to standard Python int, float, or None.

    Normalizes various numeric representations from Python stdlib and popular
    external libraries into standard types suitable for display formatting.
    Preserves special float values (inf, -inf, nan) and maintains semantic
    distinction between integer and floating-point types.

    Supported types (via duck typing, no imports required):

    **Python stdlib:**
    - int, float, None
    - math.inf, math.nan (as float)
    - decimal.Decimal (via __int__ or __float__)
    - fractions.Fraction (via __int__ or __float__)

    **External libraries (detected heuristically):**
    - NumPy: int8-64, uint8-64, float16-128, numpy.nan, numpy.inf
    - Pandas: scalars, NA (converted to nan)
    - PyTorch: tensor.item() scalars
    - TensorFlow: tensor scalars
    - JAX: DeviceArray scalars
    - Astropy: Quantity.value (physical quantities with units)
    - SymPy: expressions that can evaluate to float/int
    - mpmath: arbitrary precision floats

    **Overflow/Underflow Behavior:**

    This function preserves semantic type distinction (int vs float) to maintain
    information about the data's nature:

    - **Integer preservation**: Types detected as "true integers" (via __index__)
      are converted to Python int with arbitrary precision. No overflow occurs.
      Examples: numpy.int64, pandas.Int64 → Python int (exact, unlimited size)

    - **Float overflow**: Float-like types exceeding IEEE 754 double precision
      range (~±1.8e308) convert to float('inf') or float('-inf'). This is
      standard IEEE 754 behavior and clearly signals out-of-range values.
      Examples: Decimal('1e400') → inf, numpy.float128(1e5000) → inf

    - **Float underflow**: Float-like values smaller than minimum representable
      float (~5e-324) convert to 0.0 or -0.0 (preserving sign). This is silent
      but acceptable for display purposes.
      Examples: Decimal('1e-400') → 0.0, Fraction(1, 10**400) → 0.0

    - **Precision loss**: Large integers or high-precision decimals that fit
      within float range but exceed float precision (53 bits mantissa) lose
      precision when converted to float. This is expected for float-like types.
      Examples: Decimal with 100 digits → float (rounded to 15-17 significant digits)

    **Design rationale**: Preserving int vs float semantics allows downstream
    display logic to format appropriately (e.g., "1000" vs "1000.0", or showing
    infinity symbol for overflow). Converting everything to int would lose
    fractional information and make infinity unrepresentable.

    Args:
        value: Numeric value to convert. Accepts:
               - Python int, float, None
               - Types implementing __index__ (true integers)
               - Types implementing __int__ (integer conversion)
               - Types implementing __float__ (float conversion)
               - Types with .item() method (array scalars)
               - Types with .value attribute (physical quantities)

    Returns:
        int: For Python int values and "true integer" types (via __index__).
             These have arbitrary precision and never overflow.
        float: For float values, including inf, -inf, nan. May result from
               overflow (values > ~1.8e308) or underflow (values < ~5e-324).
        None: For None input or NA-like values (pandas.NA, numpy.ma.masked)

    Raises:
        TypeError: If value is bool or cannot be converted to numeric type.
                   If value has __float__/__int__ but raises during conversion.

    Examples:
        # Python stdlib types
        >>> _std_numeric(42)
        42
        >>> _std_numeric(3.14)
        3.14
        >>> _std_numeric(None)
        None
        >>> _std_numeric(math.inf)
        inf
        >>> _std_numeric(math.nan)
        nan

        # Arbitrary precision integers (no overflow)
        >>> _std_numeric(10 ** 400)  # Pure Python int
        100000000000...000  # (exact 400-digit integer)

        # Decimal and Fraction (stdlib)
        >>> from decimal import Decimal
        >>> from fractions import Fraction
        >>> _std_numeric(Decimal('3.14'))
        3.14
        >>> _std_numeric(Fraction(22, 7))
        3.142857142857143

        # Overflow to infinity (float-like types)
        >>> _std_numeric(Decimal('1e400'))
        inf
        >>> _std_numeric(Decimal('-1e400'))
        -inf

        # Underflow to zero (float-like types)
        >>> _std_numeric(Decimal('1e-400'))
        0.0
        >>> _std_numeric(Decimal('-1e-400'))
        -0.0

        # NumPy (if available)
        >>> import numpy as np
        >>> _std_numeric(np.int64(42))  # True integer via __index__
        42
        >>> _std_numeric(np.int64(10 ** 18))  # Huge int preserved
        1000000000000000000
        >>> _std_numeric(np.float32(3.14))
        3.14000...  # (slight precision difference)
        >>> _std_numeric(np.float128(1e400))  # Overflow
        inf

        # Pandas (if available)
        >>> import pandas as pd
        >>> _std_numeric(pd.NA)
        nan

        # PyTorch tensor scalar (if available)
        >>> import torch
        >>> _std_numeric(torch.tensor(3.14))
        3.14
        >>> _std_numeric(torch.tensor(42, dtype=torch.int64))
        42

        # Astropy Quantity (if available)
        >>> from astropy import units as u
        >>> _std_numeric(3.14 * u.meter)
        3.14  # Extracts numeric value, discards unit

        # Integer-valued Decimal preserves as int
        >>> _std_numeric(Decimal('42'))
        42
        >>> _std_numeric(Decimal('42.0'))
        42

        # Error cases
        >>> _std_numeric(True)
        Traceback: TypeError: boolean values not supported
        >>> _std_numeric("123")
        Traceback: TypeError: unsupported numeric type
        >>> _std_numeric([1, 2, 3])
        Traceback: TypeError: unsupported numeric type

    Note:
        **Array/tensor types**: Only scalars (0-dimensional or single-element)
        are supported. Multi-element arrays will raise TypeError.

        **Astropy Quantity**: Only the numeric magnitude is extracted. Unit
        information is discarded - ensure units are compatible with your
        DisplayValue.unit before conversion.

        **Integer detection priority**:
        1. __index__() - strictest, only "true" integers (NumPy int types)
        2. .item() returning Python int - array scalar unwrapping
        3. __int__() with integer-valued check - Decimal/Fraction integers
        4. __float__() - default for float-like types

        **Why not convert overflow to int?** While technically possible
        (e.g., int(Decimal('1e400'))), this would:
        - Lose fractional information (3.14e400 → 314...000)
        - Make infinity unrepresentable (overflow needs a sentinel)
        - Produce very long integers unsuitable for display
        Better to show "∞" or scientific notation in display layer.
    """
    #
    # TODO consider integration tests for _std_numeric against major third-party libs
    #

    # None passthrough
    if value is None:
        return None

    # Reject boolean explicitly (bool is subclass of int in Python)
    if isinstance(value, bool):
        raise TypeError(f"boolean values not supported, got {value}")

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
        except (TypeError, ValueError):
            # __index__ exists but failed, fall through to other methods
            pass

    # Priority 2: Handle array/tensor scalars with .item() method
    # Common in NumPy, PyTorch, TensorFlow, JAX
    # .item() returns Python scalar - respect its type (int or float)
    if hasattr(value, 'item') and callable(value.item):
        try:
            result = value.item()
            # Check result type and recurse to handle it properly
            if isinstance(result, bool):
                # Reject boolean (could come from boolean array)
                raise TypeError(f"boolean values not supported, got {value}")
            elif isinstance(result, (int, float, type(None))):
                return result
            # If .item() returned something else, fall through to other methods
        except (TypeError, ValueError):
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
            return _std_numeric(magnitude)
        except (TypeError, ValueError, AttributeError):
            # Not actually a Quantity-like object, fall through
            pass

    # Priority 4: Check for integer-valued Decimal/Fraction
    # These types implement both __int__ and __float__
    # If mathematically an integer, preserve as int to avoid overflow
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

    # Priority 5: Duck typing via __float__
    # Handles: Decimal (with fractions), Fraction (with fractions),
    # NumPy float scalars, SymPy, mpmath, etc.
    # May overflow to inf/-inf or underflow to 0.0/-0.0
    if hasattr(value, '__float__'):
        try:
            result = float(value)
            # Overflow: values beyond ~±1.8e308 become inf/-inf
            # Underflow: values below ~5e-324 become 0.0/-0.0
            # Special values: numpy.nan → float('nan'), numpy.inf → float('inf')
            return result
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"cannot convert {fmt_type(value)} to float: {e}"
            ) from e

    # If we get here, type is not supported
    raise TypeError(
        f"unsupported numeric type: {fmt_type(value)}. "
        f"Expected int, float, None, or types implementing __index__, __int__, "
        f"__float__, .item(), or having .value attribute (e.g., numpy scalars, "
        f"Decimal, Fraction, pandas scalars, array.item(), Quantity.value)"
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
