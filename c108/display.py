"""
Numeric display formatting tools for terminal UI, progress bars, status displays, etc
"""

# ## Scope
#
# `DisplayValue` is designed for **one-way formatting** (numeric value → human-readable string).
# It is NOT designed for parsing strings back to values.
#
# If you need to serialize/deserialize `DisplayValue` objects:
#   - Store the original numeric value, not the formatted string
#   - Use JSON/pickle to serialize the entire DisplayValue object
#   - Don't rely on `str(dv)` for round-tripping

#  Specific Use Cases for under/overflow:
#   - autoscale = False + BASE_FIXED or UNIT_FIXED - no power multiplier; number over-/underflow
#   - UNIT_FLEX (autoscale ignored)                - units map limited; number over-/underflow

# Standard library -----------------------------------------------------------------------------------------------------
import math
import collections.abc as abc
from dataclasses import dataclass, field
from enum import StrEnum, unique
from functools import cached_property
from typing import Any, Mapping, Protocol, Self, runtime_checkable, Literal

# Local ----------------------------------------------------------------------------------------------------------------

from .collections import BiDirectionalMap
from .numeric import std_numeric

from .tools import fmt_type, fmt_value, dict_get, fmt_any, fmt_sequence
from .unicode import to_sup


@runtime_checkable
class SupportsFloat(Protocol):
    """Protocol for duck-typed numeric conversion."""

    def __float__(self) -> float: ...


# @formatter:off

class DisplayConf:
    BINARY_SCALE = BiDirectionalMap({
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
    BIN_PREFIXES = {
        0: "",     # no prefix = 2^0 = 1
        10: "Ki",  # kibi = 2^10 = 1,024
        20: "Mi",  # mebi = 2^20 = 1,048,576
        30: "Gi",  # gibi = 2^30 = 1,073,741,824
        40: "Ti",  # tebi = 2^40 = 1,099,511,627,776
        50: "Pi",  # pebi = 2^50
        60: "Ei",  # exbi = 2^60
        70: "Zi",  # zebi = 2^70
        80: "Yi",  # yobi = 2^80
    }
    BIN_SCALE_BASE = 2  # Base in binary scale 2^0, 2^10, 2^20, ...
    BIN_SCALE_STEP = 10 # Step of power in binary scale
    PLURAL_UNITS = {
        "byte": "bytes", "step": "steps", "item": "items",
        "second": "seconds", "minute": "minutes", "hour": "hours",
        "day": "days", "week": "weeks", "month": "months",
        "year": "years", "meter": "meters", "gram": "grams"
    }
    DECIMAL_SCALE = BiDirectionalMap({
        -12: "10⁻¹²", -9: "10⁻⁹", -6: "10⁻⁶", -3: "10⁻³", 0: "",
        3: "10³", 6: "10⁶", 9: "10⁹", 12: "10¹²",
        15: "10¹⁵", 18: "10¹⁸", 21: "10²¹",
    })
    SI_PREFIXES_3N = BiDirectionalMap({
      -24: "y",   # yocto
      -21: "z",   # zepto
      -18: "a",   # atto
      -15: "f",   # femto
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
    SI_PREFIXES = BiDirectionalMap({ # All Si unit prefixes including 10^(+/-1), 10^(+/-2)
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
class MultSymbol(StrEnum):
    ASTERISK = "*"
    CDOT = "⋅"
    CROSS = "×"
    X = "x"


@dataclass(frozen=True)
class DisplayValue:
    """
    A numeric value with intelligent unit formatting for display.

    Automatically handles value type conversion, exponent calculation, digit
    trimming, and unit pluralization for clean, readable numeric displays in
    terminal UIs, progress bars, and status indicators.

    Value Type Support: Accepts diverse numeric types through std_numeric() duck typing and heuristic detection:
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
        precision: Fixed decimal places for floats. Takes precedence over trim_digits.
                   Use for consistent decimal display (e.g., "3.14" with precision=2).
        trim_digits: Override auto-calculated digit count for rounding. Used when
                    precision is None. Controls compact display digit count.
        mult_symbol: Multiplier symbol (×, ⋅, *) for scientific notation.
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
        - numeric.std_numeric() - Value type conversion function

    Raises:
          TypeError: on improper field types
          ValueError: on improper field values
    """
    value: int | float | None
    unit: str | None = None

    mult_exp: int | None = None
    unit_exp: int | None = None

    autoscale: bool = True  # Enable autoscale in BASE_FIXED and UNIT_FIXED modes TODO implement with overflow
    mult_format: Literal["unicode", "caret", "python", "latex"] = "unicode"
    mult_symbol: str = MultSymbol.CROSS
    overflow_mode: Literal["e_notation", "infinity"] = "e_notation"  # Overflow Display style TODO implement
    overflow_tolerance: int | None = None  # set None here to autoselect based on scale_type TODO implement
    pluralize: bool = True
    precision: int | None = None  # set None to disable precision formatting
    scale_type: Literal["binary", "decimal"] = "decimal"  # Mutliplier scale preset TODO implement
    separator: str = " "
    trim_digits: int | None = None
    underflow_tolerance: int | None = None  # set None here to autoselect based on scale_type TODO implement
    unit_plurals: Mapping[str, str] | None = None
    unit_prefixes: Mapping[int, str] | None = None
    whole_as_int: bool | None = None  # set None here to autoselect based on DisplayMode

    mode: DisplayMode = field(init=False)

    _mult_exp: int = field(init=False)  # Processed _mult_exp
    _unit_exp: int = field(init=False)  # Processed _unit_exp
    _scale_base: int = field(init=False)  # 10 for decimal/SI, 2 for binary
    _scale_step: int = field(init=False)  # 3 for decimal/SI, 10 for binary

    def __post_init__(self):
        """
        Validate and set fields
        """
        # value
        value_ = _std_numeric(self.value)
        object.__setattr__(self, 'value', value_)

        # unit
        if not isinstance(self.unit, (str, type(None))):
            raise TypeError(f"unit must be str or None, but got {fmt_type(self.unit)}")

        # mult_exp/unit_exp and DisplayMode
        self._validate_exponents_and_mode()

        # autoscale
        object.__setattr__(self, "autoscale", bool(self.autoscale))

        # mult_format
        if self.mult_format not in ("unicode", "caret", "python", "latex"):
            raise ValueError(f"mult_format must be one of 'unicode', 'caret', 'python', 'latex' "
                             f"but found {fmt_value(self.mult_format)}")

        # mult_symbol
        if not isinstance(self.mult_symbol, (str, type(None))):
            raise TypeError(f"mult_symbol must be str or None, but got {fmt_type(self.mult_symbol)}")

        # overflow_mode
        object.__setattr__(self, "overflow_mode", str(self.overflow_mode))

        # pluralize
        object.__setattr__(self, "pluralize", bool(self.pluralize))

        # precision
        self._validate_precision()

        # scale_type: Validate & process scale_type
        self._validate_scale_type()

        # separator
        object.__setattr__(self, "separator", str(self.separator))

        # trim_digits
        self._validate_trim_digits()

        # unit_plurals
        self._validate_unit_plurals()

        # unit_prefixes: based on known DisplayMode (inferred from mult_exp and unit_exp)
        self._validate_unit_prefixes()

        self._validate_whole_as_int()

        # TODO post-scale-init
        # overflow_tolerance auto  = 2*scale_step - 1 (i.e. +5 in SI units)
        # underflow_tolerance auto = 2*scale_step     (i.e. -6 in SI units)

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
        if not self.unit and self._units_str:
            return f"{self._number_str}{self._units_str}"

        return f"{self._number_str}{self.separator}{self._units_str}"

    @property
    def exponent(self) -> int:
        """
        The total exponent at normalized value, equals to source value exponent or 0.

        exponent = mult_exp + unit_exp

        Example:
            Value 123.456×10³ byte, the exponent == 3;
            Value 123.456×10³ kbyte the exponent == 6, and mult_exp == 3;
            Value 1.2e+3 s in PLAIN mode, the exponent == 0 as we do not display multiplier and unit exponents here.
        """
        return self._mult_exp + self._unit_exp

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
                                         trim_digits=self.trim_digits,
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
        if self.value in [None, math.nan]:
            return ""

        if self.mode == DisplayMode.UNIT_FIXED:
            return self.unit_prefixes[self._unit_exp]
        elif self.mode == DisplayMode.UNIT_FLEX:
            # TODO recheck and test what we should return here
            return self.unit_prefixes[self._unit_exp]
        return ""

    @property
    def _is_overflow(self):
        return self._unit_exp > self._unit_exp_max

    @property
    def _is_underflow(self):
        return self._unit_exp < self._unit_exp_min

    @property
    def _mult_exp_max(self) -> int:
        """mult_exp upper limit (inclusive), overflow mode is triggered if this limit is crossed"""
        return 0

    @property
    def _mult_exp_min(self) -> int:
        """mult_exp lower limit (inclusive), underflow mode is triggered if this limit is crossed"""
        return 0

    @property
    def _multiplier_str(self) -> str:
        """
        Numeric multiplier suffix.

        Example:
            displayed value 123×10³ byte has _multiplier_str of ×10³
        """
        if self._mult_exp == 0:
            return ""

        return f"{self.mult_symbol}{_disp_power(self._scale_base, power=self._mult_exp, format=self.mult_format)}"

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
            return _infinite_value_to_str(display_number)

        if self.precision is not None:
            return f"{display_number:.{self.precision}f}{self._multiplier_str}"

        if self.whole_as_int or isinstance(display_number, int):
            return f"{display_number}{self._multiplier_str}"
        else:
            return f"{display_number:.{self.trim_digits}g}{self._multiplier_str}"

    @property
    def _raw_exponent(self) -> int:
        """
        Returns the exponent of raw DisplayValue.value given in base units with stdlib E-notation float
        with 1 <= mantissa < 10.

        Returns 0 if the value is 0 or not a finite number (i.e. NaN, None or +/-infinity).

        The returned exponent is a multiple of 3.

        value = mantissa * 10^_raw_exponent

        """
        # TODO support self.scale_base + scale_step required
        if self.value == 0:
            return 0
        elif _is_finite(self.value):
            magnitude = math.floor(math.log10(abs(self.value)))
            src_exponent = (magnitude // 3) * 3
            return src_exponent
        else:
            return 0

    @property
    def _unit_exp_max(self) -> int | float:
        """
        unit_exp upper limit (inclusive), overflow mode is triggered if this limit is crossed

        This limit is a calculated int in UNIT_FLEX mode; +infinity in other modes
        """
        if self.mode != DisplayMode.UNIT_FLEX:
            return math.inf
        return max(self.unit_prefixes.keys()) + self.overflow_tolerance

    @property
    def _unit_exp_min(self) -> int | float:
        """
        unit_exp lower limit (inclusive), underflow mode is triggered if this limit is crossed

        This limit is a calculated int in UNIT_FLEX mode; -infinity in other modes
        """
        if self.mode != DisplayMode.UNIT_FLEX:
            return -math.inf
        return min(self.unit_prefixes.keys()) - self.underflow_tolerance

    @cached_property
    def _unit_prefixes(self) -> BiDirectionalMap[int, str]:
        """Returns the appropriate SI prefix map based on the configuration."""
        return BiDirectionalMap(self.unit_prefixes) if self.unit_prefixes else DisplayConf.SI_PREFIXES_3N

    @property
    def _units_str(self) -> str:
        """
        Units of measurement, includes SI prefix if applicable.

        Example:
            123 ms has _units_str = 'ms'.
            123.5 k (no unit) has _units_str = 'k'.
        """
        # Values which have NO units of measurement
        if self.value in [None, math.nan]:
            return ""

        unit_ = self.unit

        # Handle case where no unit is specified but unit_prefix defined
        # No pluralizetion should be applied
        if not unit_:
            return self.unit_prefix or ""

        if not self.pluralize:
            return f"{self.unit_prefix}{self.unit}"

        if abs(self.normalized) == 1:
            # Should be non-plural if == 1
            return f"{self.unit_prefix}{self.unit}"

        # Plurals for all normal numeric cases
        # including +/-infinity
        if self._unit_exp is not None:
            return f"{self.unit_prefix}{self._plural_unit}"
        else:
            return f"{self._plural_unit}"

    @cached_property
    def _valid_exponents(self) -> tuple[int, ...]:
        return tuple(self._unit_prefixes.keys())

    @cached_property
    def _valid_unit_prefixes(self) -> tuple[str, ...]:
        return tuple(self._unit_prefixes.values())

    @cached_property
    def _plural_unit(self) -> str:
        """
        Check for explicit plural rules for current unit. Return plural if found
        """
        if not self.unit:
            return ""

        if not self.pluralize:
            return self.unit

        unit_plurals = self.unit_plurals if isinstance(self.unit_plurals, abc.Mapping) \
            else DisplayConf.PLURAL_UNITS

        # Should NOT pluralize if not explicit pluralization rule found
        plural_unit = dict_get(unit_plurals, key=self.unit, default=self.unit)

        return plural_unit

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

    def _parse_exponent_value_OLD(self, exp: int | str | None) -> int | None:
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

    def _validate_exponents_and_mode(self):
        """
        Validate and set exponents mult_exp/unit_exp, auto-calculate if not set; set inferred mode.

        Supported input mult_exp/unit_exp pairs: 0/0, None/int, int/None, None/None

        Processed exponents have at least one of mult_exp/unit_exp set to int value

        Auto-scale and auto-unit behavour:
            - PLAIN mode:
                auto-scale and auto-unit not applicable;
                no overflows
            - BASE_FIX and UNIT_FIX modes + autoscale=True:
                autoscale finds max closest power of scale_base (powers allowed are scale_step*N, N >/=/< 0);
                no overflows
            - BASE_FIX and UNIT_FIX modes + autoscale=False:
                resets mult_exp=0, unit_exp is fixed; mantissa changes in over/under-flow tolerance range;
                overflow/underflow outside tolerance range
            - UNIT_FLEX mode - autoscale not applicable:
                auto-unit based on unit_prefixes mapping;
                overflow/underflow outside (unit_prefixes+tolerance) range
        """
        mult_exp = self.mult_exp
        unit_exp = self.unit_exp

        if type(mult_exp) not in (int, type(None)):
            raise TypeError(f"mult_exp must be int or None, got {fmt_type(mult_exp)}")

        if type(unit_exp) not in (int, type(None)):
            raise TypeError(f"unit_exp must be int or None, got {fmt_type(unit_exp)}")

        # # Should be completely fine to specify both; total exponent (mult_exp+unit_exp) diff vs raw exponent
        # # should be handled by residual exponent when required: raw_exponent = (mult_exp+unit_exp) + residual_exp
        # if mult_exp is not None and unit_exp is not None:
        #     if mult_exp != 0 or unit_exp != 0:
        #         raise ValueError(
        #             f"mult_exp and unit_exp must be 0 if specified both, but found: mult_exp={mult_exp}, unit_exp={unit_exp} "
        #             "Supported mult_exp/unit_exp pairs: 0/0, None/int, int/None, None/None."
        #         )

        if mult_exp is None and unit_exp is None:
            unit_exp = 0

        # Starting from this line whe should have at least one of mult_exp/unit_exp to be finite
        # i.e. the 0/0, None/int, int/None pairs only are passed below
        # which are unambiguously convertable to a DisplayMode

        if mult_exp == 0 and unit_exp == 0:
            object.__setattr__(self, "mode", DisplayMode.PLAIN)

        elif mult_exp is None and isinstance(unit_exp, int):
            total_exp = self._raw_exponent
            mult_exp = total_exp - unit_exp
            if unit_exp == 0:
                object.__setattr__(self, "mode", DisplayMode.BASE_FIXED)
            else:
                object.__setattr__(self, "mode", DisplayMode.UNIT_FIXED)

        elif isinstance(mult_exp, int) and unit_exp is None:
            total_exp = self._raw_exponent
            unit_exp = total_exp - mult_exp
            object.__setattr__(self, "mode", DisplayMode.UNIT_FLEX)

        if type(mult_exp) is not int or type(unit_exp) is not int:
            raise ValueError("improper intialization of mult_exp/unit_exp pair. Internal sanity condition vioated")

        object.__setattr__(self, '_mult_exp', mult_exp)
        object.__setattr__(self, '_unit_exp', unit_exp)

    def _validate_precision(self):
        precision = self.precision

        if not isinstance(precision, (int, type(None))):
            raise ValueError(f"precision must be int or None, but got: {fmt_type(precision)}")

        if isinstance(precision, int) and self.precision < 0:
            raise ValueError(f"precision must be int >= 0 or None, but got {fmt_value(precision)}")

    def _validate_trim_digits(self):
        """Validate initialization parameters"""
        trim_digits = self.trim_digits

        if not isinstance(trim_digits, (int, type(None))):
            raise ValueError(f"trim_digits must be int | None, but got: {fmt_type(trim_digits)}")

        if isinstance(trim_digits, int) and trim_digits < 1:
            raise ValueError(f"trim_digits must be >= 1, but got {trim_digits}")

        if isinstance(trim_digits, int):
            return

        trim_digits = trimmed_digits(self.value, round_digits=15)

        # Set trimmed digits
        object.__setattr__(self, "trim_digits", trim_digits)

    def _validate_scale_type(self):
        """
        Validate scale, set _scale_base and _scale_step
        """
        if self.scale_type == "binary":
            object.__setattr__(self, "_scale_base", 2)
            object.__setattr__(self, "_scale_step", 10)

        elif self.scale_type == "decimal":
            object.__setattr__(self, "_scale_base", 10)
            object.__setattr__(self, "_scale_step", 3)

        else:
            raise ValueError(f"scale_type 'binary' or 'decimal' literal expected, "
                             f"but {fmt_value(self.scale_type)} found")

    def _validate_unit_prefixes(self):
        """
        Provide unit prefixes mapping if required for current display mode

        Supported mult_exp/unit_exp pairs: 0/0, None/int, int/None, None/None
        """
        if self.mode == DisplayMode.BASE_FIXED:
            # Prefixes ignored
            return

        if self.mode == DisplayMode.PLAIN:
            # Prefixes ignored
            return

        if self.mode == DisplayMode.UNIT_FLEX:
            # Prefixes mapping required (at least 1 prefix in Mapping, auto-select)
            self._validate_unit_prefixes_raise()
            return

        if self.mode == DisplayMode.UNIT_FIXED:
            # Prefixes mapping required with current unix_exp in keys
            self._validate_unit_prefixes_raise(unit_exp=self._unit_exp)
            return

    def _validate_unit_prefixes_raise(self,
                                      unit_exp: int | None = None
                                      ):
        """
        Validate unit_prefixes
        """
        unit_prefixes_source = self.unit_prefixes if self.unit_prefixes is not None \
            else DisplayConf.SI_PREFIXES_3N

        for key, value in unit_prefixes_source.items():
            if not isinstance(key, int) or isinstance(key, bool):
                raise ValueError(f"unit_prefixes keys must be a valid int, "
                                 f"but got {fmt_any(unit_prefixes_source.keys())}")
            if not isinstance(value, str):
                raise ValueError(f"unit_prefixes values must be a non-empty str, "
                                 f"but got {fmt_any(unit_prefixes_source.values())}")
            if key == 0 and value != "":
                raise ValueError(f"unit_prefixes value must be an empty when exponent is 0, "
                                 f"but found key={key}, value={fmt_value(value)}")

        try:
            unit_prefixes = BiDirectionalMap(unit_prefixes_source)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"cannot create BiDirectionalMap: invalid unit_prefixes {fmt_any(unit_prefixes_source)}"
                             ) from exc

        if len(unit_prefixes) < 1:
            raise ValueError(f"non-empty mapping required in unit_prefixes: {fmt_any(unit_prefixes_source)}")

        if unit_exp is not None:
            if unit_exp not in unit_prefixes:
                raise ValueError(
                    f"Unit exponent {unit_exp} not found in unit_prefixes: {fmt_any(unit_prefixes_source)}")

        # Set self.unit_prefixes
        object.__setattr__(self, "unit_prefixes", unit_prefixes)

    def _validate_unit_plurals(self):
        """
        Provide unit plurals
        """
        unit_plurals = self.unit_plurals

        if not isinstance(unit_plurals, (abc.Mapping, type(None))):
            raise ValueError(f"unit_plurals must be a mapping or None, but got {fmt_type(unit_plurals)}")

        if isinstance(unit_plurals, abc.Mapping):
            unit_plurals = self.unit_plurals

        elif self.unit in DisplayConf.PLURAL_UNITS:
            unit_plurals = {self.unit: DisplayConf.PLURAL_UNITS[self.unit]}

        elif unit_plurals is None:
            unit_plurals = {}

        for key, value in unit_plurals.items():
            if not isinstance(key, str) or len(key) == 0:
                raise ValueError(f"unit_plurals keys must be non-empty strings, "
                                 f"but got {fmt_any(unit_plurals.keys())}")
            if not isinstance(value, str) or len(value) == 0:
                raise ValueError(f"unit_plurals values must be a non-empty strings, "
                                 f"but got {fmt_any(unit_plurals.values())}")

        # Set self.unit_plurals
        object.__setattr__(self, "unit_plurals", unit_plurals)

    def _validate_whole_as_int(self):
        whole_as_int = self.whole_as_int

        if not isinstance(whole_as_int, (int, type(None))):
            raise ValueError(f"whole_as_int must be int | None, but got: {fmt_type(whole_as_int)}")

        if self.whole_as_int is None:
            object.__setattr__(self, 'whole_as_int', self.mode != DisplayMode.PLAIN)
        else:
            object.__setattr__(self, 'whole_as_int', bool(self.whole_as_int))


# Methods --------------------------------------------------------------------------------------------------------------

def _disp_power(base: int = 10, *,
                power: int,
                format: Literal["unicode", "caret", "python", "latex"] = "unicode") -> str:
    """
    Format a power expression for display.

    Args:
        base: The base (typically 10 or 2)
        power: The exponent value
        format: Output format style:
            - "unicode" (default): "10³", "2²⁰" (superscript exponents)
            - "caret": "10^3", "2^20" (ASCII-safe, common in text)
            - "python": "10**3", "2**20" (Python operator syntax)
            - "latex": "10^{3}", "2^{20}" (LaTeX markup)

    Returns:
        Formatted power string, or empty string if power is 0.

    Examples:
        >>> _disp_power(power=3)
        '10³'
        >>> _disp_power(power=3, format="caret")
        '10^3'
        >>> _disp_power(power=0)
        ''
    """
    if power == 0:
        return ""

    if not isinstance(power, int):
        raise TypeError("power must be an int")
    if not isinstance(base, int):
        raise TypeError("base must be an int")
    if not isinstance(format, str):
        raise TypeError("format must be a str")

    # Handle built-in formats
    if format == "unicode":
        return f"{base}{to_sup(power)}"
    elif format == "caret":
        return f"{base}^{power}"
    elif format == "python":
        return f"{base}**{power}"
    elif format == "latex":
        return f"{base}^{{{power}}}"
    else:
        raise ValueError(f"invalid power format: {fmt_value(format)} "
                         f"Expected one of: 'unicode', 'caret', 'python', 'latex'")


def _infinite_value_to_str(val: int | float | None):
    """Format stdlib infinite numerics: None, +/-inf, NaN."""

    if val is None:
        return "N/A"  # TODO make this customizable

    if math.isinf(val):
        return "∞" if val > 0 else "-∞"  # TODO make these symbols customizable

    if math.isnan(val):
        return f"NaN"  # TODO make this customizable

    raise TypeError(f"cannot format as infinite value: {fmt_type(val)}")


def _is_finite(value: Any) -> bool:
    """
    Check if a value is a finite numeric suitable for display.

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


def _std_numeric(value: int | float | None | SupportsFloat) -> int | float | None:
    """
    Convert common numeric types to standard Python int, float, or None.

    Preserves special float values (inf, -inf, nan) and maintains semantic
    distinction between integer and floating-point types.

    Raises:
        TypeError: If value is not a supported numeric type.
    """
    try:
        num = std_numeric(value, on_error="raise", allow_bool=False)
    except TypeError as exc:
        raise TypeError(f"cannot convert {fmt_type(value)} to standard numeric types") from exc
    return num


def trimmed_digits(number: int | float | None, *, round_digits: int | None = 15) -> int | None:
    """
    Count significant digits for display by removing all trailing zeros.

    Used for compact display formatting (e.g., "123×10³" instead of "123000"). Removes trailing zeros
    from both integers and the decimal representation of floats to determine the minimum digits
    needed for display.

    Float values are rounded before analysis to eliminate floating-point precision artifacts
    (e.g., 0.30000000000000004 from 0.1 + 0.2).

    **⚠️ DISPLAY PURPOSE ONLY:** This function treats trailing zeros in floats (e.g., 1200.0)
    as non-significant, which violates standard significant-figure interpretation. Use this ONLY for
    UI display formatting, NOT for scientific or engineering calculations, significant-figure analysis.

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
