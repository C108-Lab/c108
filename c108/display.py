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

# ## Custom unit_prefix scales with 15+ orders gap
#
# For gaps > ~15 orders of magnitude between prefixes, floating-point precision limits
# may cause values to display in scientific notation (e.g., "1.234e20 Ps").
# Consider adding intermediate prefixes for better readability.
#
# Example:
#   dv = DisplayValue.unit_flex(1e20, unit="s", unit_prefixes={-15: "f", 15: "P"})
#   "1.0e5 Ps" (falls back to e-notation naturally)
#

# Standard library -----------------------------------------------------------------------------------------------------
import math
import collections.abc as abc
from dataclasses import dataclass, field, InitVar
from enum import StrEnum, unique
from functools import cached_property
from typing import Any, Mapping, Protocol, Self, runtime_checkable, Literal, List, Callable

# Local ----------------------------------------------------------------------------------------------------------------

from .collections import BiDirectionalMap
from .numeric import std_numeric
from .sentinels import UnsetType, UNSET
from .tools import fmt_type, fmt_value, dict_get, fmt_any, fmt_sequence
from .unicode import to_sup


@runtime_checkable
class SupportsFloat(Protocol):
    """Protocol for duck-typed numeric conversion."""

    def __float__(self) -> float: ...

# @formatter:off

class DisplayConf:
    """
    Default configuration constants for DisplayValue formatting.

    Provides standard unit prefixes (SI and IEC), pluralization rules,
    and tolerance thresholds for overflow/underflow formatting.

    These constants can be overridden in DisplayValue instances via
    the unit_prefixes and unit_plurals parameters.

    Attributes:
        BIN_PREFIXES: IEC binary prefixes (powers of 2) for binary units.
            Maps exponents to prefixes: 10→"Ki", 20→"Mi", etc.
            Used for bytes, bits, and other binary measurements.

        SI_PREFIXES_3N: SI decimal prefixes with 10^(3N) exponents only.
            Standard metric prefixes: k, M, G, T, etc.
            Excludes deci/deca/centi/hecto for cleaner display.
            Default for most unit scaling operations.

        SI_PREFIXES: Complete SI prefix set including all exponents.
            Includes 10^1 (deca), 10^2 (hecto), 10^-1 (deci), 10^-2 (centi).
            Use when fine-grained prefix control is needed.

        PLURAL_UNITS: Common unit pluralization mappings.
            Default rules for English plurals: "byte"→"bytes", etc.
            Override via DisplayValue.unit_plurals parameter.

        OVERFLOW_TOLERANCE: Maximum normalized value exponent before overflow formatting.
            When exponent > OVERFLOW_TOLERANCE, triggers overflow.
            Default: 5 (values ≥ 1_000_000 considered overflow on decimal scale).

        UNDERFLOW_TOLERANCE: Minimum normalized value exponent before underflow formatting.
            When exponent < UNDERFLOW_TOLERANCE, triggers underflow.
            Default: 6 (values < 0.000001 considered underflow on decimal scale).

    Examples:
        >>> # Using full SI set with centi/deci
        >>> DisplayValue(0.01, unit="meter", unit_prefixes=DisplayConf.SI_PREFIXES)

        >>> # Custom pluralization
        >>> custom_plurals = DisplayConf.PLURAL_UNITS | {"datum": "data"}
        >>> DisplayValue(5, unit="datum", unit_plurals=custom_plurals)

    Note:
        BiDirectionalMap allows lookup in both directions: exponent→prefix
        and prefix→exponent for efficient reverse lookups.
    """

    # IEC Binary Prefixes (powers of 2)
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

    # Overflow threshold: |value| > 10^5 triggers overflow formatting
    OVERFLOW_TOLERANCE = 5

    # Default unit pluralization rules
    PLURAL_UNITS = {
        "byte": "bytes", "step": "steps", "item": "items",
        "second": "seconds", "minute": "minutes", "hour": "hours",
        "day": "days", "week": "weeks", "month": "months",
        "year": "years", "meter": "meters", "gram": "grams"
    }

    # SI Prefixes: 10^(3N) exponents only (excludes deci/deca/centi/hecto)
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
        27: "R",    # ronna
        30: "Q",    # quetta
    })

    # SI Prefixes: Complete set including 10^1, 10^2, 10^-1, 10^-2
    SI_PREFIXES = BiDirectionalMap({
        -24: "y",  # yocto
        -21: "z",  # zepto
        -18: "a",  # atto
        -15: "f",  # femto
        -12: "p",  # pico
        -9: "n",   # nano
        -6: "µ",   # micro
        -3: "m",   # milli
        -2: "c",   # centi
        -1: "d",   # deci
        0: "",     # (no prefix)
        1: "da",   # deca (or deka)
        2: "h",    # hecto
        3: "k",    # kilo
        6: "M",    # mega
        9: "G",    # giga
        12: "T",   # tera
        15: "P",   # peta
        18: "E",   # exa
        21: "Z",   # zetta
        24: "Y",   # yotta
        27: "R",   # ronna
        30: "Q",   # quetta
    })

    # Underflow threshold: |value| < 10^-6 triggers underflow formatting
    UNDERFLOW_TOLERANCE = 6

# @formatter:on

# Classes --------------------------------------------------------------------------------------------------------------

# @formatter:off
@unique
class DisplayMode(StrEnum):
    """
    Display modes for formatting DisplayValue numbers and units.

    Modes are automatically inferred from DisplayValue's mult_exp and unit_exp
    parameters and cannot be set directly. Each mode determines how numeric
    values and unit prefixes are displayed.

    Attributes:
        BASE_FIXED: Base units with scientific notation multiplier
                    Example: "123×10⁹ bytes"
                    Inferred when: mult_exp=None, unit_exp=0

        FIXED: Fixed unit prefix and value multiplier
               Example: "123.46×10⁹ MB"
               Inferred when: mult_exp=int, unit_exp=int

        PLAIN: Raw numbers with base units, no scaling
               Example: "1 byte", "2200.0 seconds"
               Inferred when: mult_exp=0, unit_exp=0

        UNIT_FIXED: Fixed unit prefix with auto-scaled value multiplier
                    Example: "123×10³ Mbytes"
                    Inferred when: mult_exp=None, unit_exp=int

        UNIT_FLEX: Auto-scaled unit prefix without value multiplier
                   Example: "123.4 ns", "1.5 Mbytes"
                   Inferred when: mult_exp=int, unit_exp=None

    Note:
        See DisplayValue docs for complete mode inference rules and
        mult_exp/unit_exp combination details.
    """
    BASE_FIXED = "base_fixed"
    FIXED = "fixed"
    PLAIN = "plain"
    UNIT_FIXED = "unit_fixed"
    UNIT_FLEX = "unit_flex"

# @formatter:on

@unique
class MultSymbol(StrEnum):
    """
    Multiplier symbols for scientific notation (e.g., "1.5×10³ bytes").

    ASTERISK for ASCII compatibility, CROSS for standard math notation.
    """
    ASTERISK = "*"
    CDOT = "⋅"
    CROSS = "×"
    X = "x"


@dataclass(frozen=True)
class DisplaySymbols:
    """
    Symbols for formatting DisplayValue output.

    Controls the visual representation of non-finite values, mathematical
    operators, and spacing in formatted numeric displays.

    Attributes:
        nan: Symbol for Not-a-Number values.
        none: Symbol for None/null values.
        pos_infinity: Symbol for positive infinity.
        neg_infinity: Symbol for negative infinity.
        pos_underflow: Symbol for positive values too small to display.
        neg_underflow: Symbol for negative values too small to display.
        mult: Multiplier symbol for scientific notation (×, *, ⋅, or x).
        separator: String between numeric value and unit (default: single space).

    Examples:
        >>> # Default Unicode symbols
        >>> dv = DisplayValue(float('inf'), unit="byte", symbols=DisplaySymbols.unicode())
        >>> str(dv)  # "+∞ bytes"

        >>> # ASCII-safe for basic terminals
        >>> dv = DisplayValue(float('inf'), unit="byte", symbols=DisplaySymbols.ascii())
        >>> str(dv)  # "inf bytes"

        >>> # Custom symbols
        >>> symbols = DisplaySymbols(mult=MultSymbol.CDOT, separator="...")
        >>> dv = DisplayValue(1500, unit="byte", mult_exp=3, symbols=symbols)
        >>> str(dv)  # "1.5⋅10³...bytes"
    """
    # Non-finite values
    nan: str = "NaN"
    none: str = "None"

    pos_infinity: str = "inf"
    neg_infinity: str = "-inf"
    pos_underflow: str = "0"
    neg_underflow: str = "-0"

    # Multiplier symbol for scientific notation
    mult: MultSymbol = MultSymbol.ASTERISK

    # Separator between number and units
    separator: str = " "

    @classmethod
    def ascii(cls) -> 'DisplaySymbols':
        """
        ASCII-safe symbols for maximum compatibility.

        Use in environments with limited Unicode support (basic terminals,
        legacy systems, plain text logs, or when piping output).

        Returns:
            DisplaySymbols with ASCII-only characters (* for multiplication).
        """
        return cls(
            nan="NaN",
            none="None",
            pos_infinity="inf",
            neg_infinity="-inf",
            pos_underflow="0",
            neg_underflow="-0",
            mult=MultSymbol.ASTERISK)

    @classmethod
    def unicode(cls) -> 'DisplaySymbols':
        """
        Unicode mathematical symbols.

        Provides proper mathematical notation with infinity (∞), approximate
        equality (≈), and multiplication (×) symbols. Best for modern terminals
        and display contexts.

        Returns:
            DisplaySymbols with Unicode mathematical characters.
        """
        return cls(
            nan="NaN",
            none="None",
            pos_infinity="+∞",
            neg_infinity="−∞",
            pos_underflow="≈0",  # Note: Same symbol for both pos/neg
            neg_underflow="≈0",
            mult=MultSymbol.CROSS
        )


@dataclass(frozen=True)
class DisplayFlow:
    """
    Configures overflow and underflow display formatting behavior.

    Does not modify the actual value or normalized fields - only affects
    how values are formatted as strings. Raw numeric values remain accessible
    regardless of flow settings (except for non-finite cases like inf/nan).

    This class is intended to be nested inside DisplayValue as a configuration
    object. Predicates require a backlink to the owner DisplayValue instance
    for evaluation. This backlink is established via DisplayFlow.merge(owner=...)
    called by the DisplayValue instance.

    Attributes:
        mode: Formatting mode for overflow cases ('e_notation' or 'infinity').
        overflow_predicate: Optional callable to determine if normalized value
            should display as overflow. Receives DisplayValue instance as argument.
        overflow_tolerance: Maximum order of magnitude allowed in DisplayValue.normalized
            display before triggering overflow. If None, uses default from DisplayConf.
        underflow_predicate: Optional callable to determine if normalized value
            should display as underflow. Receives DisplayValue instance as argument.
        underflow_tolerance: Minimum order of magnitude allowed in DisplayValue.normalized
            display before triggering underflow. If None, uses default from DisplayConf.

    Examples:
        Basic usage with tolerance-based overflow:
        >>> flow = DisplayFlow(overflow_tolerance=3, mode='infinity')
        >>> dv = DisplayValue(1e20, unit="byte", flow=flow)
        >>> str(dv)  # Formatted with overflow handling
        'inf bytes'
        >>> dv.value  # Original value intact
        1e20
        >>> dv.normalized  # Normalized value intact; overflow affects str format only
        1e17

        Custom predicates for specific value thresholds:
        >>> def overflow_above_1000(dv):
        ...     return dv.value >= 1000
        >>> def underflow_below_0_001(dv):
        ...     return dv.value <= 0.001
        >>> flow = DisplayFlow(
        ...     overflow_predicate=overflow_above_1000,
        ...     underflow_predicate=underflow_below_0_001,
        ...     mode='infinity'
        ... )
        >>> dv = DisplayValue(2500, unit="meter", flow=flow)
        >>> str(dv)
        'inf meters'
        >>> dv = DisplayValue(0.0001, unit="meter", flow=flow)
        >>> str(dv)
        '0 meters'
    """
    # Formatting mode
    mode: Literal["e_notation", "infinity"] = "e_notation"

    # Overflow & underflow predicates
    overflow_predicate: InitVar[Callable[["DisplayValue"], bool] | None] = None
    underflow_predicate: InitVar[Callable[["DisplayValue"], bool] | None] = None

    # Overflow & underflow tolerances
    overflow_tolerance: int | None = None
    underflow_tolerance: int | None = None

    # Processed predicates
    _overflow_predicate: Callable[["DisplayValue"], bool] = field(init=False)
    _underflow_predicate: Callable[["DisplayValue"], bool] = field(init=False)

    # Backlink to wrapping DisplayValue instance
    _owner: "DisplayValue" = field(init=False, default=None)

    def __post_init__(self, overflow_predicate, underflow_predicate):
        """
        Validate parameters and initialize internal fields.

        Sets default predicates if not provided and applies default tolerance
        values from DisplayConf if tolerances are None.
        """
        # validate overflow_predicate/underflow_predicate
        if not isinstance(overflow_predicate, (abc.Callable, type(None))):
            raise ValueError(f"overflow_predicate must be callable, not {fmt_type(overflow_predicate)}")

        if not isinstance(underflow_predicate, (abc.Callable, type(None))):
            raise ValueError(f"underflow_predicate must be callable, not {fmt_type(underflow_predicate)}")

        overflow_predicate = overflow_predicate or _overflow_predicate
        underflow_predicate = underflow_predicate or _underflow_predicate
        object.__setattr__(self, '_overflow_predicate', overflow_predicate)
        object.__setattr__(self, '_underflow_predicate', underflow_predicate)

        # validate mode
        if self.mode not in ["e_notation", "infinity"]:
            raise ValueError(f"mode must be 'e_notation' or 'infinity', not {fmt_any(self.mode)}")

        # validate overflow_tolerance, underflow_tolerance
        if not isinstance(self.overflow_tolerance, (int, type(None))):
            raise TypeError(f"overflow_tolerance must be int | None, but got {fmt_type(self.overflow_tolerance)}")
        if not isinstance(self.underflow_tolerance, (int, type(None))):
            raise TypeError(f"underflow_tolerance must be int | None, but got {fmt_type(self.underflow_tolerance)}")

        overflow_tolerance = self.overflow_tolerance if self.overflow_tolerance is not None \
            else DisplayConf.OVERFLOW_TOLERANCE
        underflow_tolerance = self.underflow_tolerance if self.underflow_tolerance is not None \
            else DisplayConf.UNDERFLOW_TOLERANCE

        object.__setattr__(self, 'overflow_tolerance', overflow_tolerance)
        object.__setattr__(self, 'underflow_tolerance', underflow_tolerance)

    def merge(self,
              # Attrs override
              mode: Literal["e_notation", "infinity"] | UnsetType = UNSET,
              overflow_predicate: Callable[["DisplayValue"], bool] | UnsetType = UNSET,
              underflow_predicate: Callable[["DisplayValue"], bool] | UnsetType = UNSET,
              overflow_tolerance: int | UnsetType = UNSET,
              underflow_tolerance: int | UnsetType = UNSET,
              # Set owner
              owner: "DisplayValue | UnsetType" = UNSET,
              ) -> "DisplayFlow":
        """
        Create a new DisplayFlow instance with merged configuration options.

        Parameters not provided (UNSET) are inherited from the current instance.
        Use the owner parameter to establish a backlink to a DisplayValue instance,
        enabling predicate evaluation.

        Args:
            mode: Override formatting mode.
            overflow_predicate: Override overflow predicate function.
            underflow_predicate: Override underflow predicate function.
            overflow_tolerance: Override overflow tolerance value.
            underflow_tolerance: Override underflow tolerance value.
            owner: DisplayValue instance to link to. Pass None to explicitly unset owner.

        Returns:
            New DisplayFlow instance with merged configuration.

        Raises:
            TypeError: If owner is not a DisplayValue instance.
        """
        mode = self.mode if mode is UNSET else mode
        overflow_predicate = self._overflow_predicate if overflow_predicate is UNSET else overflow_predicate
        underflow_predicate = self._underflow_predicate if underflow_predicate is UNSET else underflow_predicate
        overflow_tolerance = self.overflow_tolerance if overflow_tolerance is UNSET else overflow_tolerance
        underflow_tolerance = self.underflow_tolerance if underflow_tolerance is UNSET else underflow_tolerance
        display_flow = DisplayFlow(mode=mode,
                                   overflow_predicate=overflow_predicate,
                                   underflow_predicate=underflow_predicate,
                                   overflow_tolerance=overflow_tolerance,
                                   underflow_tolerance=underflow_tolerance)
        owner = None if owner is UNSET else owner
        if not isinstance(owner, (DisplayValue, type(None))):
            raise TypeError(f"owner must be DisplayValue, not {fmt_type(owner)}")
        object.__setattr__(display_flow, '_owner', owner)
        return display_flow

    @property
    def overflow(self) -> bool:
        """
        Check if overflow condition is triggered on the owner DisplayValue instance.

        Evaluates the overflow predicate with the owner as argument. Returns False
        if no owner is assigned or predicate evaluation indicates no overflow.

        Returns:
            True if overflow condition is met, False otherwise.
        """
        # Predicate should be callable after post_init validation
        predicate = self._overflow_predicate
        if self._owner is not None:
            return predicate(self._owner)
        else:
            return False

    @property
    def underflow(self) -> bool:
        """
        Check if underflow condition is triggered on the owner DisplayValue instance.

        Evaluates the underflow predicate with the owner as argument. Returns False
        if no owner is assigned or predicate evaluation indicates no underflow.

        Returns:
            True if underflow condition is met, False otherwise.
        """
        # Predicate should be callable after post_init validation
        predicate = self._underflow_predicate
        if self._owner is not None:
            return predicate(self._owner)
        else:
            return False


@dataclass(frozen=True)
class DisplayFormat:
    """
    Formatting styles and options for display.

    TODO docs + should provide formatting methods? keep as is? _disp_power where should reside??
    """
    mult: Literal["unicode", "caret", "python", "latex"] = "unicode"

    def __post_init__(self):
        """Validate and set fields"""

        if self.mult not in ("unicode", "caret", "python", "latex"):
            raise ValueError(f"mult must be one of 'unicode', 'caret', 'python', 'latex' "
                             f"but found {fmt_value(self.mult)}")


@dataclass(frozen=True)
class DisplayScale:
    """
    Display Scale Controls
    TODO requires better docs
    """
    type: Literal["binary", "decimal"] = "decimal"  # TODO implement

    base: int | None = field(init=False, default=None)
    step: int | None = field(init=False, default=None)

    def __post_init__(self):
        """
        Validate and set attrs
        """
        if self.type == "binary":
            object.__setattr__(self, "base", 2)
            object.__setattr__(self, "step", 10)

        elif self.type == "decimal":
            object.__setattr__(self, "base", 10)
            object.__setattr__(self, "step", 3)

        else:
            raise ValueError(f"scale type 'binary' or 'decimal' literal expected, "
                             f"but {fmt_value(self.type)} found")

    def value_exponent(self, value: int | float | None) -> int | None:
        """
        Get integer value exponent based on current scale.

        For decimal (base 10): floor(log10(abs(value)))
        For binary (base 2): floor(log2(abs(value)))

        Examples:
            >>> # Decimal scale
            >>> scale = DisplayScale(type="decimal")  # base=10
            >>> scale.value_exponent(0.00234)  # -3,  0.00234 == 2.34 * 10^-3
            >>> scale.value_exponent(4.56)     # 0,      4.56 == 4.56 * 10^0
            >>> scale.value_exponent(86)       # 1,      86   == 8.6 * 10^1
            >>> scale.value_exponent(0.72)     # -1,     0.75  in [10^-1, 10^0)

            >>> # Binary scale
            >>> scale = DisplayScale(type="binary")  # base=2
            >>> scale.value_exponent(1)     # 0,  1 == 2^0
            >>> scale.value_exponent(1024)  # 10, 1024 == 2^10
            >>> scale.value_exponent(0.72)  # -1, 0.75  in [2^-1, 2^0)

            >>> # Edge cases
            >>> scale.value_exponent(0)
            0
            >>> scale.value_exponent(None) is None
            True
        """

        if not isinstance(value, (int, float, type(None))):
            raise TypeError(f"value must be int | float, but got {fmt_type(value)}")
        if not isinstance(self.base, int):
            raise ValueError(f"int scale base required, but got {fmt_value(self.base)}")
        if self.base not in [2, 10]:
            raise ValueError(f"scale base must binary or decimal, got {fmt_value(self.base)}")

        if value is None:
            return None

        if value == 0:
            return 0

        if self.base == 10:
            return math.floor(math.log10(abs(value)))
        elif self.base == 2:
            return math.floor(math.log2(abs(value)))
        else:
            raise NotImplementedError()


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

    **Factory Methods (Recommended)**:
        - All factory  methods return DisplayValue instances configured for specific display modes
        - `DisplayValue.base_fixed()` - Base units with multipliers;
        - `DisplayValue.plain()` - Plain number display;
        - `DisplayValue.unit_fixed()` - Fixed SI prefix;
        - `DisplayValue.unit_flex()` - Auto-scaled SI prefix.

    Display Modes: Inferred from mult_exp/unit_exp combination:
        - PLAIN (0, 0): Raw values → "123000000 bytes"
        - FIXED (int, int): Fixed multiplier + units → "123456.78×10⁹ MB"
        - BASE_FIXED (None, 0): Base units with multipliers → "123×10⁹ bytes"
        - UNIT_FIXED (None, int): Fixed prefix + auto multipliers → "123×10³ Mbytes"
        - UNIT_FLEX (int, None): Auto-scaled prefix → "123 Mbytes"

    Overflow Formatting: applied based on overflow and underflow predicates, by default they return:
        - BASE_FIXED: no overflow or underflow; value multiplier scales to required exponent;
        - FIXED: overflow or underflow when normalized value is outside the tolerance range;
        - PLAIN: no overflow or underflow; standard Pyhton int or float formatting;
        - UNIT_FIXED: no overflow or underflow; value multiplier scales to required exponent;
        - UNIT_FLEX: overflow or underflow on unit_prefix edges when normalized value is outside the tolerance range.

    Formatting Pipeline: Applied in order:
        1. Handle non-finite numerics (inf, nan, None)
        2. Apply trim_digits (if precision is None)
        3. Apply precision (if specified - takes precedence)
        4. Apply whole_as_int conversion (3.0 → "3")
        5. Apply overflow/underflow formatting per display mode

    Attributes:
        value: Numeric value (int/float/None). Automatically converted from
               external types (NumPy, Pandas, Decimal, etc.) to stdlib types.
        unit: Base unit name (e.g., "byte", "second"). Auto-pluralized.
        mult_exp: Value multiplier exponent (e.g. 3 in 1.23*10^3 Mbyte).
        unit_exp: Unit exponent (e.g. 6 in 1.23*10^3 Mbyte).
        pluralize: Use plurals for units of mesurement if display value !=1.
        precision: Fixed decimal places for floats. Takes precedence over trim_digits.
                   Use for consistent decimal display (e.g., "3.14" with precision=2).
        trim_digits: Override auto-calculated digit count for rounding. Used when
                     precision is None. Controls compact display digit count.
        unit_plurals: Unit pluralize mapping.
        unit_prefixes: Unit prefixes custom subset. Supported binary IEC and decimal SI prefixes only.
        whole_as_int: Display whole floats as integers (3.0 → "3").
        flow: Display flow configuration for overflow/underflow formatting behavior.
              Does not affect value or normalized properties.
        format:  Display Number formatting styles.
        mode:    Display mode inferred from mult_exp/unit_exp pair.
        scale:   Scale applied to exponents and unit prefixes; supported scales are "binary" and "decimal".
        symbols: DisplaySymbols = field(default_factory=DisplaySymbols.unicode).

    Scale Types:
        - decimal: mult_exp=5 → 10⁵ multiplier, unit_exp=6 → M (10⁶) prefix
        - binary: mult_exp=5 → 2⁵ multiplier, unit_exp=20 → Mi (2²⁰) prefix

    Examples:
        >>> # Basic usage - different types
        >>> str(DisplayValue(42))                          # "42"
        >>> str(DisplayValue(42, unit="byte"))             # "42 bytes"
        >>> str(DisplayValue(np.int64(42), unit="byte"))   # "42 bytes"

        >>> # Precision vs trim_digits
        >>> str(DisplayValue(1/3, unit="s", precision=2))      # "0.33 s"
        >>> str(DisplayValue(4/3, unit="s", trim_digits=2))    # "1.3 s"
        >>> str(DisplayValue(1/3, unit="s"))                   # "0.333333333333333 s"

        >>> # Precision takes precedence
        >>> str(DisplayValue(1/3, precision=2, trim_digits=10))  # "0.33 s"

        >>> # Factory methods
        >>> str(DisplayValue.unit_flex(1_500_000, unit="byte"))   # "1.5 Mbytes"
        >>> str(DisplayValue.base_fixed(1_500_000, unit="byte"))  # "1.5×10⁶ bytes"
        >>> str(DisplayValue.plain(1_500_000, unit="byte"))       # "1500000 bytes"

        >>> # Edge cases
        >>> str(DisplayValue(0, unit="byte"))        # "0 bytes"
        >>> str(DisplayValue(-42, unit="meter"))     # "-42 meters"
        >>> str(DisplayValue(None, unit="item"))     # "N/A items"  # ADD: Confirm actual behavior
        >>> str(DisplayValue(float('inf')))          # "∞"          # ADD: Confirm actual behavior

    See Also:
        - trimmed_digits(): Auto-calculate display digit count.
        - std_numeric(): Value type conversion function.
        - DisplayFlow:   Overflow/underflow configuration.
        - DisplayFormat: Number formatting configuration.
        - DisplayScale:  Binary/decimal scale configuration.

    Raises:
        TypeError: Invalid field types (e.g., string for value, bool for value).
        ValueError: Invalid field values (e.g., negative precision, invalid scale type).
    """
    value: int | float | None
    unit: str | None = None

    mult_exp: int | None = None
    unit_exp: int | None = None

    pluralize: bool = True
    precision: int | None = None  # set None to disable precision formatting
    trim_digits: int | None = None
    unit_plurals: Mapping[str, str] | None = None
    unit_prefixes: Mapping[int, str] | None = None
    whole_as_int: bool | None = None  # set None here to autoselect based on DisplayMode

    flow: DisplayFlow = field(default_factory=DisplayFlow)
    format: DisplayFormat = field(default_factory=DisplayFormat)
    mode: DisplayMode = field(init=False)
    scale: DisplayScale = field(default_factory=DisplayScale)
    symbols: DisplaySymbols = field(default_factory=DisplaySymbols.unicode)

    _mult_exp: int = field(init=False)  # Processed _mult_exp
    _unit_exp: int = field(init=False)  # Processed _unit_exp

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

        # mult_exp/unit_exp: check mult_exp/unit_exp, infer DisplayMode
        #                    The post-processing of mult_exp/unit_exp should be performed
        #                    in a dedicated companion method after unit_prefixes validation
        self._validate_exponents_and_mode()

        # # overflow_mode
        # object.__setattr__(self, "overflow_mode", str(self.flow.overflow_mode))

        # Flow control back-linking
        flow = self.flow.merge(owner=self)
        object.__setattr__(self, 'flow', flow)

        # pluralize
        object.__setattr__(self, "pluralize", bool(self.pluralize))

        # precision
        self._validate_precision()

        # trim_digits
        self._validate_trim_digits()

        # unit_plurals
        self._validate_unit_plurals()

        # unit_prefixes: based on known DisplayMode (inferred from mult_exp and unit_exp)
        self._validate_unit_prefixes()

        # whole_as_int
        self._validate_whole_as_int()

        # Process of mult_exp/unit_exp for multiplier autoscale and auto-units features
        self._process_exponents()

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
    def unit_fixed(  # TODO unit_fixed on decimal and binary scales
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
            DisplayValue.unit_fixed(value=123_000_000, si_unit="Mbyte")
            # → "123 Mbyte" or "123×10³ Mbyte" depending on magnitude

            # From SI units (123 megabytes):
            DisplayValue.unit_fixed(si_value=123, si_unit="Mbyte")
            # → "123 Mbyte" (internally converts to 123_000_000 base units)

            # NumPy/Pandas types auto-converted
            DisplayValue.unit_fixed(value=np.int64(500_000_000), si_unit="Mbyte")
            # → "500 Mbyte"

            DisplayValue.unit_fixed(si_value=pd.Series([500]).item(), si_unit="Mbyte")
            # → "500 Mbyte"

            # Precision control
            DisplayValue.unit_fixed(value=123_456_789, si_unit="Mbyte", precision=2)
            # → "123.46 Mbyte"

            DisplayValue.unit_fixed(value=123_456_789, si_unit="Mbyte", trim_digits=4)
            # → "123.5 Mbyte" (4 significant digits)

            # Decimal/Fraction support
            from decimal import Decimal
            DisplayValue.unit_fixed(si_value=Decimal("123.456"), si_unit="Mbyte")
            # → "123.456 Mbyte"

            # Fractional units
            DisplayValue.unit_fixed(si_value=500, si_unit="Mbyte/s")
            # → "500 Mbyte/s"

            # Error handling
            DisplayValue.unit_fixed(value=100, si_value=200, si_unit="Mbyte")
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
    def unit_flex(  # TODO unit_flex with decimal and binary scales
            cls,
            value: int | float | None = None,
            unit: str | None = None,
            mult_exp: int | None = 0,
            trim_digits: int | None = None,
            precision: int | None = None,
            unit_prefixes: Mapping[int, str] | None = None,
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
            # Large value auto-scale, no units
            DisplayValue.unit_flex(1_500_000_000)
            # → "1.5G" (giga = 10⁹)

            # Large byte values auto-scale
            DisplayValue.unit_flex(1_500_000_000, unit="byte")
            # → "1.5 Gbytes" (giga = 10⁹)

            # NumPy/Pandas types auto-converted
            DisplayValue.unit_flex(np.int64(250_000), unit="byte")
            # → "250 kbytes" (kilo = 10³)

            DisplayValue.unit_flex(pd.Series([42]).item(), unit="byte")
            # → "42 bytes" (no prefix for moderate values)

            # Precision control - consistent decimals
            DisplayValue.unit_flex(1_234_567_890, unit="byte", precision=2)
            # → "1.23 Gbytes" (exactly 2 decimal places)

            DisplayValue.unit_flex(1_234_567_890, unit="byte", trim_digits=4)
            # → "1.235 Gbytes" (4 significant digits)

            DisplayValue.unit_flex(1_234_567_890, unit="byte", precision=2, trim_digits=10)
            # → "1.23 Gbytes" (precision wins)

            # Time durations with appropriate prefixes
            DisplayValue.unit_flex(0.000123, unit="second")
            # → "123 µs" (micro = 10⁻⁶)

            DisplayValue.unit_flex(0.000000456, unit="second")
            # → "456 ns" (nano = 10⁻⁹)

            # Decimal/Fraction support
            from decimal import Decimal
            DisplayValue.unit_flex(Decimal("1500"), unit="meter")
            # → "1.5 km" (kilo = 10³)

            from fractions import Fraction
            DisplayValue.unit_flex(Fraction(25, 10), unit="meter")
            # → "2.5 m" or "2500 mm" depending on auto-scaling

            # Astropy Quantity (extracts .value, discards units)
            from astropy import units as u
            DisplayValue.unit_flex(1500 * u.meter, unit="meter")
            # → "1.5 km" (extracts 1500, auto-scales, YOUR unit must match!)

            # ML framework tensors
            import torch
            DisplayValue.unit_flex(torch.tensor(1500.0), unit="meter")
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
            unit=unit,
            trim_digits=trim_digits,
            precision=precision,
            mult_exp=mult_exp,
            unit_prefixes=unit_prefixes
        )

    @classmethod
    def fixed(  # TODO ...
            cls,
            value: int | float | None = None,
    ) -> Self:
        """
        Create DisplayValue in fixed units with fixed multiplier.
        """
        pass

    def __str__(self):
        """Number with units as a string."""
        if not self.units:
            return self.number

        # Don't use separator when units is just an SI prefix (single character like 'k', 'M')
        if not self.unit and self.units:
            return f"{self.number}{self.units}"

        return f"{self.number}{self.symbols.separator}{self.units}"

    @property
    def is_finite(self) -> bool:
        """True if value is not None, inf, or NaN."""
        return _is_finite(self.value)

    @property
    def mult_value(self) -> int | float:
        """
        The multiplier value as a number (e.g., 1000 for 10³, 1024 for 2¹⁰).

        Example:
            Value with mult_exp=3, scale.base=10 returns 1000
        """
        return self.scale.base ** self._mult_exp

    @property
    def normalized(self) -> int | float | None:
        """
        Normalized value.

        normalized_value = value / ref_value =  value / scale.base^(mult_exponent+unit_exponent)

        Includes rounding to trimmed digits and optional whole_as_int convertion.

        Example:
            displayed value 123.4×10³ ms has the normalized value 123.4
        """
        if not _is_finite(self.value):
            return self.value

        value_ = self.value / self.ref_value
        normalized = _normalized_number(value_,
                                        trim_digits=self.trim_digits,
                                        whole_as_int=self.whole_as_int)
        return normalized

    @property
    def number(self) -> str:
        """
        Fully formatted number including the multiplier if applicable.

        Example:
            The value 123.456×10³ km has number 123.456×10³
        """

        normalized = self.normalized

        if not _is_finite(normalized) or self._is_overflow or self._is_underflow:
            return self._over_value_str()

        if self.precision is not None:
            return f"{normalized:.{self.precision}f}{self._multiplier_str}"

        if self.whole_as_int or isinstance(normalized, int):
            return f"{normalized}{self._multiplier_str}"
        else:
            return f"{normalized:.{self.trim_digits}g}{self._multiplier_str}"

    @property
    def parts(self) -> tuple[str, str]:
        """Returns (number, units) as a tuple for unpacking."""
        return (self.number, self.units)

    @property
    def ref_value(self) -> int | float:
        """
        The reference value for scaling the normalized display number:
            - ref_value = mult_value * unit_value = scale.base ^ (mult_exponent+unit_exponent);
            - normalized = value / ref_value.

        Example:
            Value 123.456×10³ kbyte correspond to the ref_value = 10⁶
        """
        ref_exponent = self._mult_exp + self._unit_exp
        return self.scale.base ** ref_exponent

    @property
    def unit_prefix(self) -> str:
        """
        The SI prefix in units of measurement, e.g., 'm' (milli-), 'k' (kilo-).
        """
        # TODO re-check and test in various modes - sometimes we need bare units,
        #      sometimes need fixed prefix (also over/underflow test)

        if self.value in [None, math.nan]:
            return ""

        if self.mode == DisplayMode.UNIT_FIXED:
            return self.unit_prefixes[self._unit_exp]
        elif self.mode == DisplayMode.UNIT_FLEX:
            return self.unit_prefixes[self._unit_exp]
        return ""

    @property
    def unit_value(self) -> int | float:
        """
        The unit prefix value as a number (e.g., 1_000_000 for 'M' prefix).

        Example:
            Value with unit_exp=6, scale.base=10 returns 1000000
        """
        return self.scale.base ** self._unit_exp

    @property
    def units(self) -> str:
        """
        Fully formatted units including SI/IEC prefix and pluralization if applicable.

        Example:
            123 ms has units = 'ms'.
            123.5k (no unit) has units = 'k'.
        """
        # Values which have NO units of measurement
        if self.value in [None, math.nan]:
            return ""

        unit_ = self.unit

        # Handle case where no unit is specified but unit_prefix defined
        # No pluralizetion should be applied
        if not unit_:
            if self._is_overflow or self._is_underflow:
                return ""
            else:
                return self.unit_prefix or ""

        if not self.pluralize:
            return f"{self.unit_prefix}{self.unit}"

        if abs(self.normalized) == 1:
            # Should be non-plural if == 1
            return f"{self.unit_prefix}{self.unit}"

        # Plurals for numeric cases, overflow/underflow and +/-infinity
        return f"{self.unit_prefix}{self._plural_unit}"

    @property
    def _multiplier_str(self) -> str:
        """
        Numeric multiplier suffix.

        Example:
            displayed value 123×10³ byte has _multiplier_str of ×10³
        """
        if self._mult_exp == 0:
            return ""

        return f"{self.symbols.mult}{_disp_power(self.scale.base, power=self._mult_exp, format=self.format.mult)}"

    @property
    def _is_overflow(self) -> bool:
        """
        Returns true if OVERFLOW predicate defined and is True
        """
        return self.flow.overflow  # NO predicate parameters required

    @property
    def _is_underflow(self) -> bool:
        """
        Returns true if UNDERFLOW predicate defined and is True
        """
        return self.flow.underflow  # NO predicate parameters required

    def _over_unit_str(self):
        """
        Format units of measurement for stdlib infinite numerics (None, +/-inf, NaN) and overflow/underflow values.
        """
        val = self.value

        if val is None:
            return ""

        elif math.isnan(val):
            return ""

        # elif math.isinf(val):
        #     return self.unit
        #
        # elif self._is_overflow:
        #     return self.symbols.pos_infinity if val > 0 else self.symbols.neg_infinity
        #
        # elif self._is_underflow:
        #     return self.symbols.pos_underflow if val > 0 else self.symbols.neg_underflow

        else:
            raise ValueError(f"can not format value {fmt_value(val)} for overflow/underflow.")

    def _over_value_str(self):
        """
        Format stdlib infinite numerics (None, +/-inf, NaN) and overflow/underflow values.
        """
        val = self.value

        if val is None:
            return self.symbols.none

        elif math.isnan(val):
            return self.symbols.nan

        elif math.isinf(val):
            return self.symbols.pos_infinity if val > 0 else self.symbols.neg_infinity

        elif self._is_overflow:
            return self.symbols.pos_infinity if val > 0 else self.symbols.neg_infinity

        elif self._is_underflow:
            return self.symbols.pos_underflow if val > 0 else self.symbols.neg_underflow

        else:
            raise ValueError(f"can not format value {fmt_value(val)} for overflow/underflow.")

    @property
    def _raw_exponent(self) -> int:
        """
        Returns the exponent of raw DisplayValue.value given as:

        value = mantissa * scale.base^raw_exponent (with int/float mantissa 1 <= mantissa < 10)

        Returns 0 if the value is 0 or not a finite number (i.e. NaN, None or +/-infinity).
        """
        if self.value == 0:
            return 0

        elif _is_finite(self.value):
            if self.scale.type == "decimal":
                # For decimal scaling (SI), we use base 10
                raw_exponent = math.floor(math.log10(abs(self.value)))
            elif self.scale.type == "binary":
                # For binary scaling (IEC), we use base 2
                raw_exponent = math.floor(math.log2(abs(self.value)))
            else:
                raise NotImplementedError
            return raw_exponent

        else:
            return 0

    @property
    def _residual_exponent(self) -> int:
        """
        Returns the order of magnitude of normalized value:

            normalized = mantissa * scale.base^residual_exponent, with 1 <= mantissa < 10

        the order of magnitude of normalized value

        Returns 0 if the value is 0 or not a finite number (i.e. NaN, None or +/-infinity).
        """
        if self.value == 0:
            return 0

        elif _is_finite(self.value):
            return self.scale.value_exponent(self.normalized)

        else:
            return 0

    @cached_property
    def _unit_prefixes(self) -> BiDirectionalMap[int, str]:
        """Returns the appropriate SI prefix map based on the configuration."""
        return BiDirectionalMap(self.unit_prefixes) if self.unit_prefixes else DisplayConf.SI_PREFIXES_3N

    @cached_property
    def _valid_unit_exponents(self) -> tuple[int, ...]:
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

    def _auto_mult_exponent(self, unit_exp: int) -> int:
        """
        Returns the multiplier exponent from DisplayValue.value and unit_exp:

        value = mantissa * scale.base^mult_exponent * scale.base^unit_exponent
        (with int/float mantissa 1 <= mantissa < 1000)

        Returns 0 if the value is 0 or not a finite number (i.e. NaN, None or +/-infinity).
        """
        if not _is_finite(self.value):
            return 0

        elif self.value == 0:
            return 0

        else:
            value = self.value / (self.scale.base ** unit_exp)
            magnitude = self.scale.value_exponent(value)
            mult_exponent = (magnitude // self.scale.step) * self.scale.step
            return mult_exponent

    def _auto_unit_exponent(self, mult_exp: int) -> int:
        """
        Returns the multiplier exponent from DisplayValue.value and fixed mult_exp:

        value = mantissa * scale.base^mult_exp * scale.base^unit_exponent
        (with int/float mantissa 1 <= mantissa < not limited but we select closest unit_exponent from available in unit_prefixes)

        Returns 0 if the value is 0 or not a finite number (i.e. NaN, None or +/-infinity).
        """
        if not _is_finite(self.value):
            return 0

        elif self.value == 0:
            return 0

        else:
            value = self.value / (self.scale.base ** mult_exp)
            unit_exponents = sorted(self.unit_prefixes.keys())

            if self.scale.type in ["binary", "decimal"]:
                # For decimal scaling (SI), should use base 10 and step of 3 (i.e., 10^3 per step: k, M, G, ...)
                #   value_exp = math.log10(abs(value)) OR math.log2(abs(value))
                value_exp = self.scale.value_exponent(value)

                # Find largest prefix where value_exp >= exp (mantissa >= 1)
                unit_exponent = unit_exponents[0]
                for exp in unit_exponents:
                    if value_exp >= exp:
                        unit_exponent = exp

                # Check if we should switch to next higher prefix
                # Switch only if value_exp >= current_exp + scale.step
                current_index = unit_exponents.index(unit_exponent)
                if current_index < len(unit_exponents) - 1:
                    next_exp = unit_exponents[current_index + 1]
                    if value_exp >= unit_exponent + self.scale.step:
                        # Use closest between current and next
                        if abs(value_exp - next_exp) < abs(value_exp - unit_exponent):
                            unit_exponent = next_exp

            else:
                raise NotImplementedError()

            return unit_exponent

    def _process_exponents(self):
        """
        Process exponents mult_exp/unit_exp to auto-calculate multiplier and units when required.

        Auto-scale and auto-unit related behaviour:
            - BASE_FIX and UNIT_FIX modes:
                unit_exp is fixed
                multiplier autoscale finds max closest power of scale.base (scale.base^(scale.step*N), N >/=/< 0);
                no overflows
            - UNIT_FLEX mode:
                mult_exp/scale is fixed
                auto-unit finds closest unit_prefixes mapping;
                overflow/underflow display triggered when residual exponent is outside tolerance range

        Exponent equations:
            _ref_exponent = mult_exp + unit_exp
            raw_exponent = _ref_exponent + residual_exp
            mult_exp/unit_exp are used to format value multiplier (e.g. 3 in 10³) and unit-prefixes (e.g. M in Mbyte)
            residual_exponent is analyzed to switch between norma/overflow/underflow display
        """

        # Autoscale feature and auto-unit features calculators below require
        # processed scale.type and unit_prefixes

        mult_exp = self.mult_exp
        unit_exp = self.unit_exp

        if isinstance(mult_exp, int) and isinstance(unit_exp, int):
            object.__setattr__(self, '_mult_exp', mult_exp)
            object.__setattr__(self, '_unit_exp', unit_exp)
            return

        if mult_exp is None and unit_exp is None:
            unit_exp = 0

        if mult_exp is None and isinstance(unit_exp, int):
            mult_exp = self._auto_mult_exponent(unit_exp)

        elif isinstance(mult_exp, int) and unit_exp is None:
            unit_exp = self._auto_unit_exponent(mult_exp)

        else:
            raise ValueError("improper intialization of mult_exp/unit_exp pair. Internal sanity check not passed.")

        object.__setattr__(self, '_mult_exp', mult_exp)
        object.__setattr__(self, '_unit_exp', unit_exp)

    def _validate_exponents_and_mode(self):
        """
        Validate but do not process mult_exp/unit_exp exponents; set inferred DisplayMode.

        Supported input mult_exp/unit_exp pairs: 0/0, None/int, int/None, None/None

        Exponents compatibility: units_exp must be compatible with scale.type
            - SI decimal scale requires one of SI-prefixes exponents in unit_exp
            - IEC binary scale requires one of IEC binary exponents in unit_exp

        Exponents processing including auto-scale and auto-units see in a dedicated _process_exponents() method.

        Auto-scale and auto-unit behaviour:
            - PLAIN mode:
                no auto-scale, no auto-units;
                no overflows
            - BASE_FIX and UNIT_FIX modes:
                unit_exp is fixed
                autoscale finds max closest power of scale.base (powers allowed are scale.step*N, N >/=/< 0);
                no overflows
            - FIXED mode:
                unit_exp is fixed + mult_exp is fixed
                uses mult_exp=0 as default, accepts any pair of fixed mult_exp/unit_exp;
                mantissa changes in over/under-flow tolerance range;
                overflow/underflow display triggered when residual exponent is outside tolerance range
            - UNIT_FLEX mode:
                mult_exp/scale is fixed
                auto-unit based on unit_prefixes mapping;
                overflow/underflow display triggered when residual exponent is outside tolerance range

        Exponent equations  with all exponents using same scale.base, i.e. 2 or 10:
            _ref_exponent = mult_exp + unit_exp
            raw_exponent = _ref_exponent + residual_exp
            mult_exp/unit_exp are used to format multiplier (e.g. 3 in 10³) and unit-prefixes (e.g. M in Mbyte)
            residual_exponent is analyzed to switch between norma/overflow/underflow display
        """

        mult_exp = self.mult_exp
        unit_exp = self.unit_exp

        if type(mult_exp) not in (int, type(None)):
            raise TypeError(f"mult_exp must be int or None, got {fmt_type(mult_exp)}")

        if type(unit_exp) not in (int, type(None)):
            raise TypeError(f"unit_exp must be int or None, got {fmt_type(unit_exp)}")

        if mult_exp is None and unit_exp is None:
            unit_exp = 0

        if isinstance(unit_exp, int):
            self._validate_unit_exp_vs_scale_type(unit_exp)

        # Starting from this line whe should have at least one of mult_exp/unit_exp to be finite
        # i.e. the 0/0, None/int, int/None pairs only are passed below
        # which are unambiguously convertable to a DisplayMode

        if mult_exp == 0 and unit_exp == 0:
            object.__setattr__(self, "mode", DisplayMode.PLAIN)

        elif isinstance(mult_exp, int) and isinstance(unit_exp, int):
            object.__setattr__(self, "mode", DisplayMode.FIXED)

        elif mult_exp is None and isinstance(unit_exp, int):
            if unit_exp == 0:
                object.__setattr__(self, "mode", DisplayMode.BASE_FIXED)
            else:
                object.__setattr__(self, "mode", DisplayMode.UNIT_FIXED)

        elif isinstance(mult_exp, int) and unit_exp is None:
            object.__setattr__(self, "mode", DisplayMode.UNIT_FLEX)

        else:
            raise ValueError("improper processing of mult_exp/unit_exp pair. Internal sanity check not passed.")

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

    def _validate_unit_exp_vs_scale_type(self, unit_exp: int):
        if self.scale.type == "binary":
            valid_unit_exps = list(DisplayConf.BIN_PREFIXES.keys())
            if unit_exp not in valid_unit_exps:
                raise ValueError(f"when scale.type is binary, unit_exp must be one of IEC binary powers "
                                 f"{valid_unit_exps}, but got {unit_exp}")

        elif self.scale.type == "decimal":
            valid_unit_exps = list(DisplayConf.SI_PREFIXES.keys())
            if unit_exp not in valid_unit_exps:
                raise ValueError(f"when scale.type is decimal, unit_exp must be one of SI decimal powers "
                                 f"{valid_unit_exps}, but got {unit_exp}")

        else:
            raise ValueError(f"scale.type 'binary' or 'decimal' literal expected, ")

    def _validate_unit_exp_vs_unit_prefix(self, unit_exp: int, unit_prefix: str):
        if self.scale.type == "binary":
            valid_prefixes = DisplayConf.BIN_PREFIXES
            valid_unit_exps = list(DisplayConf.BIN_PREFIXES.keys())
        elif self.scale.type == "decimal":
            valid_prefixes = DisplayConf.SI_PREFIXES
            valid_unit_exps = list(DisplayConf.SI_PREFIXES.keys())
        else:
            raise ValueError(f"scale.type 'binary' or 'decimal' literal expected")

        if unit_exp not in valid_unit_exps:
            raise ValueError(
                f"unit_exp must be one of {self.scale.type} powers {valid_unit_exps}, but got {fmt_any(unit_exp)}")
        valid_prefix = valid_prefixes[unit_exp]
        if unit_prefix != valid_prefix:
            raise ValueError(
                f"expected unit prefix {valid_prefix} when unit_exp={unit_exp}, but got {fmt_type(unit_prefix)}")

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

        if self.mode in [DisplayMode.FIXED, DisplayMode.UNIT_FIXED]:
            # Prefixes mapping required with current unix_exp in keys
            self._validate_unit_prefixes_raise(unit_exp=self.unit_exp)
            return

    def _validate_unit_prefixes_raise(self,
                                      unit_exp: int | None = None
                                      ):
        """
        Validate unit_prefixes
        """
        if not self.unit_prefixes:
            if self.scale.type == "binary":
                prefixes = DisplayConf.BIN_PREFIXES
            elif self.scale.type == "decimal":
                prefixes = DisplayConf.SI_PREFIXES_3N
            else:
                raise ValueError(f"scale.type 'binary' or 'decimal' literal expected")
        else:
            prefixes = self.unit_prefixes

        # User provided unit prefix maps only should be checked
        if prefixes not in [DisplayConf.BIN_PREFIXES, DisplayConf.SI_PREFIXES, DisplayConf.SI_PREFIXES_3N]:

            for key, value in prefixes.items():
                if not isinstance(key, int) or isinstance(key, bool):
                    raise ValueError(f"unit_prefixes keys must be a valid int, "
                                     f"but got {fmt_any(prefixes.keys())}")
                if not isinstance(value, str):
                    raise ValueError(f"unit_prefixes values must be str, "
                                     f"but got {fmt_any(prefixes.values())}")
                self._validate_unit_exp_vs_unit_prefix(unit_exp=key, unit_prefix=value)

        try:
            unit_prefixes = BiDirectionalMap(prefixes)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"cannot create BiDirectionalMap: invalid unit_prefixes {fmt_any(prefixes)}"
                             ) from exc

        if len(unit_prefixes) < 1:
            raise ValueError(f"non-empty mapping required in unit_prefixes: {fmt_any(prefixes)}")

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
    """
    Process value to normalized number by rounding and conditional int conversion.
    """
    if not _is_finite(value):
        return value

    if not isinstance(value, (int, float)):
        raise TypeError(f"Expected int | float, got {fmt_type(value)}")

    if (trim_digits is not None) and value != 0:
        value = trimmed_round(value, trimmed_digits=trim_digits)

    if whole_as_int and isinstance(value, float) and value.is_integer():
        value = int(value)
        # We should apply trimmed rounding again here
        # to avoid float-to-int artifacts (e.g. 1e70 converted to int)
        value = trimmed_round(value, trimmed_digits=trim_digits)

    return value


def _overflow_predicate(dv: DisplayValue) -> bool:
    """Trigger display overflow formatting"""

    if not isinstance(dv, DisplayValue):
        raise TypeError(f"Expected DisplayValue, got {fmt_type(dv)}")

    if dv.mode in [DisplayMode.PLAIN, DisplayMode.BASE_FIXED, DisplayMode.UNIT_FIXED]:
        return False

    elif dv.mode == DisplayMode.UNIT_FLEX:
        # Overflow on unit scale edge only, no overflow in custom scale gaps
        if dv._raw_exponent > max(dv._valid_unit_exponents) + dv.flow.overflow_tolerance:
            return True
        else:
            return False

    elif dv.mode == DisplayMode.FIXED:
        # Overflow if upper tolerance limit crossed
        if dv._residual_exponent > dv.flow.overflow_tolerance:
            return True
        else:
            return False

    else:
        raise ValueError(f"overflow predicate not supported for DisplayMode '{dv.mode}'")


def _underflow_predicate(dv: DisplayValue) -> bool:
    """Trigger display underflow formatting"""

    if not isinstance(dv, DisplayValue):
        raise TypeError(f"Expected DisplayValue, got {fmt_type(dv)}")

    if dv.mode in [DisplayMode.PLAIN, DisplayMode.BASE_FIXED, DisplayMode.UNIT_FIXED]:
        return False

    elif dv.mode == DisplayMode.UNIT_FLEX:
        # Underflow on unit scale edge only, no underflow in custom scale gaps
        if dv._raw_exponent < min(dv._valid_unit_exponents) - dv.flow.underflow_tolerance:
            return True
        else:
            return False

    elif dv.mode == DisplayMode.FIXED:
        # Underflow if lower tolerance limit crossed
        if dv._residual_exponent < -dv.flow.underflow_tolerance:
            return True
        else:
            return False

    else:
        raise ValueError(f"underflow predicate not supported for DisplayMode '{dv.mode}'")


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


def trimmed_digits(number: int | float | None,
                   *,
                   round_digits: int | None = 15) -> int | None:
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
        raise TypeError(f"Expected number of int | float | None type, got {fmt_type(number)}")

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


def trimmed_round(number: int | float | None,
                  *,
                  trimmed_digits: int | None = None) -> int | float | None:
    """
    Round a number to a specified count of significant digits (trimmed digits).

    Companion method to trimmed_digits() that performs the actual rounding operation.
    Preserves original int or float type.

    Args:
        number: The number to round. Accepts int or float.
        trimmed_digits: Number of significant digits to keep (must be >= 1).

    Returns:
        int or float: Rounded number. Returns int if no decimal places remain,
                     otherwise returns float;
        None: Returns None as is.

    Returns infinity and NaN unprocessed. Returns number unprocessed if trimmed_digits is None.

    Raises:
        TypeError: If number is not int or float.
                   If trimmed_digits is not int.
        ValueError: If trimmed_digits < 1.
                    If number is NaN, inf, or -inf.

    Examples:
        # Basic rounding to significant digits
        trimmed_round(123.456, 3) == 123        # Keep 3 digits: 123
        trimmed_round(123.456, 2) == 120        # Keep 2 digits: 120
        trimmed_round(123.456, 1) == 100        # Keep 1 digit: 100
        trimmed_round(123.456, 5) == 123.46     # Keep 5 digits: 123.46
        trimmed_round(123.456, 6) == 123.456    # Keep 6 digits: 123.456

        # Integer inputs
        trimmed_round(123000, 3) == 123000      # Already 3 sig digits
        trimmed_round(123000, 2) == 120000      # Round to 2 sig digits
        trimmed_round(123000, 1) == 100000      # Round to 1 sig digit

        # Small numbers
        trimmed_round(0.00123, 2) == 0.0012     # Keep 2 digits: 0.0012
        trimmed_round(0.00123, 1) == 0.001      # Keep 1 digit: 0.001

        # Negative numbers
        trimmed_round(-123.456, 3) == -123      # Sign preserved
        trimmed_round(-123.456, 2) == -120      # Sign preserved

        # Edge cases
        trimmed_round(0, 1) == 0                # Zero
        trimmed_round(0.0, 5) == 0.0            # Zero float
        trimmed_round(9.99, 2) == 10.0          # Rounds up
        trimmed_round(999, 2) == 1000           # Rounds up to more digits
    """

    # Type checking
    if isinstance(number, type(None)):
        return number

    if math.isinf(number) or math.isnan(number):
        return number

    if isinstance(trimmed_digits, type(None)):
        return number

    if not isinstance(number, (int, float, type(None))):
        raise TypeError(f"number must be int | float | None, got {fmt_type(number)}")
    if not isinstance(trimmed_digits, int):
        raise TypeError(f"trimmed_digits must be int | None, got {fmt_type(trimmed_digits)}")

    # Value validation
    if trimmed_digits < 1:
        raise ValueError(f"trimmed_digits must be >= 1, got {trimmed_digits}")
    if math.isnan(number) or math.isinf(number):
        raise ValueError(f"number cannot be NaN or infinite, got {number}")

    # Handle zero specially
    if number == 0:
        return 0 if isinstance(number, int) else 0.0

    # Calculate the magnitude (order of magnitude) of the number
    magnitude = math.floor(math.log10(abs(number)))

    # Calculate how many decimal places we need
    # If magnitude is 2 (e.g., 123), and we want 3 sig digits, we need 0 decimal places
    # If magnitude is 2 (e.g., 123), and we want 5 sig digits, we need 2 decimal places
    decimal_places = trimmed_digits - magnitude - 1

    # Round to the calculated decimal places
    rounded = round(number, decimal_places)

    # Preserve original type: int stays int, float stays float
    if isinstance(number, int):
        return int(rounded)
    else:
        return rounded

# Module Sanity Checks -------------------------------------------------------------------------------------------------
