#
# C108 Units of Measurement Tools
#

# Standard library -----------------------------------------------------------------------------------------------------
import math
from dataclasses import dataclass, InitVar, field
from enum import StrEnum, unique

# Local ----------------------------------------------------------------------------------------------------------------
from .collections import BiDirectionalMap
from .tools import sequence_get


# @formatter:off

class UnitsConf:
    PLURALS_ = ("byte", "step", "item",
                          "second", "minute", "hour", "day", "week", "month", "year",
                          "meter", "gram")
    PLURALS = {"byte": "bytes",         "step": "steps",        "item": "items",
               "second": "seconds",     "minute": "minutes",    "hour": "hours",
                "day": "days",          "week": "weeks",        "month": "months",
                "year": "years",        "meter": "meters",      "gram": "grams"
    }



units_conf = UnitsConf()

si_prefixes = BiDirectionalMap({
    -12: "p",   -9: "n",  -6: "µ",  -3: "m",  0:   "",
    3: "k",    6: "M",   9: "G",  12: "T",  15: "P",  18:  "E",  21:  "Z",})

value_multipliers = BiDirectionalMap({
    -12: "10⁻¹²", -9: "10⁻⁹",  -6: "10⁻⁶",  -3: "10⁻³",    0: "",
    3: "10³",    6: "10⁶",    9: "10⁹",   12: "10¹²",
    15: "10¹⁵",  18: "10¹⁸",  21: "10²¹", })

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
class MultiOperator(StrEnum):
    ASTERISK = "*"
    CDOT = "⋅"
    CROSS = "×"


@dataclass(frozen=False)
class NumberUnit:
    """
    Formats a number with an optional unit, supporting various mode modes.

    This class can mode numbers in plain format, with SI prefixes (flexible or
    fixed), or with an explicit power-of-ten multiplier. It automatically handles
    exponent calculation, trimmed digits, and unit pluralization.

    Key Parameters:
      - value: Numerical value in base units (e.g., 0.001 for millisecond). Must be finite.
      - mode: Core mode mode (PLAIN, SI_FIXED, SI_FLEX, BASE_FIXED). Defaults to SI_FLEX.
      - precision: Number of decimal digits for float values in str. Must be >= 0 if specified.
      - trim_digits: Trimmed digits for display rounding (see trimmed_digits() function).
        Must be >= 1 if specified. Defaults to auto-calculated from source value.
      - mult_exp: Fixed multiplier exponent as int (e.g., 3) or SI prefix string (e.g., "M" for 10⁶).
        The unit exponent becomes flexible when this is set.
      - si_unit: Fixed SI unit with prefix (e.g., "kbyte" for kilobyte). Shorthand for setting
        both unit_exp and unit together. Cannot be combined with unit_exp or unit.
      - unit_exp: Fixed SI exponent as int (e.g., 3) or SI prefix string (e.g., 'k').
        The multiplier exponent becomes flexible when this is set. Example: 3 OR 'k' for kilo
        in '1.234 kbyte'. The corresponding base units can be set in 'unit'.
      - unit: The base units string (e.g., "s" for seconds).
      - whole_as_int: Display <float> whole numbers as integers (e.g., 10.0 -> 10).
        Defaults to True for SI modes, False for PLAIN mode.

    Exponent Specification (Three Ways):
      There are a few ways to specify exponents, each serving a different use case:

      1. unit_exp only:
         Use when working with a fixed SI prefix or base units and want auto-scaling multiplier.
         unit_exp == 0: BASE_FIXED mode. Base units, flexible value multiplier - 123×10⁹ byte
         unit_exp != 0: SI_FIXED mode. Fixed SI units prefix, flexible value multiplier - 123×10³ Mbyte
         The value argument is in base units.
         Example: NumberUnit(123456, unit_exp=0, unit="s")
         Output: "123.456×10³ s"

      2. mult_exp only:
         Use when you want explicit control over the scientific notation multiplier.
         mult_exp == 0: SI_FLEX mode. Flexible SI units prefix, no value multiplier - 123.4 ms
         mult_exp != 0: SI_FLEX mode. Flexible SI units prefix, with fixed value multiplier - 123.4×10³ ns
         The value argument is in base units.
         Example: NumberUnit(123456, mult_exp=0, unit="m")
         Output: "123.456 km"

      3. si_unit (convenience shorthand; SI_FIXED mode):
         Use as shorthand when you want to specify both SI prefix and unit together.
         The value argument is processed from SI units to base units.
         Example: NumberUnit(123456, si_unit="ms")
         Output: "123.456×10³ ms"

      4. mult_exp=0, unit_exp=0:
         Use when you want default stdlib representation for int and float.
         PLAIN mode. Base units, plain int, E-notation for floats - 1 byte, 2.2+e3 s
         Example: NumberUnit(123.1e+21, mult_exp=0, unit_exp=0, unit="s")
         Output: "1.231e+23 s"

      Note: Only ONE of these should be specified. Specifying multiple will raise ValueError.
            If neither is specified, exponents are auto-calculated from the value.

    Fractional units can be represented with denominator in the unit string
    and scaling the value accordingly. Example:

        bytes_per_s = 123_000 # source value in base units
        bytes_per_ms = bytes_per_s / 10**3
        speed_num = NumberUnit(value=bytes_per_ms, mult_exp=0, unit='byte/ms')

    Raises:
        ValueError: Invalid exponents, mode modes, parameter combinations, or non-finite values
        TypeError: Invalid types for value, exponents, si_unit, or other parameters
        KeyError: If BiDirectionalMap lookup fails (indicates invalid SI prefix in internal code)

    Notes:
        - Infinite and NaN values are not supported
        - Exponents must be multiples of 3 in range [-12, 21]
        - Only one of (mult_exp, unit_exp, si_unit) should be specified
        - Negative values are supported and pluralization checks abs(normalized) != 1

    Examples:
        >>> # BASE_FIXED mode
        >>> ...
        >>> # SI_FIXED mode
        >>> ...
        >>> # SI_FLEX mode
        >>> ...
        >>> # Convenience SI_FIXED from a value in SI units
        >>> ...
        >>> # PLAIN mode
        >>> ...
    """

    value: int | float | None
    unit: str | None = None

    mult_exp: InitVar[int | str] = None
    unit_exp: InitVar[int | str | None] = None
    si_unit: InitVar[str | None] = None
    trim_digits: InitVar[int | None] = None

    multi_operator: str = MultiOperator.CROSS
    plural_units: dict[str, str] | bool = True
    precision: int | None = None
    separator: str = " "
    whole_as_int: bool | None = None

    _mult_exp: int | None = field(init=False, default=None)
    _unit_exp: int | None = field(init=False, default=None)

    _trim_digits: int | None = field(init=False, default=None)

    def __post_init__(self, mult_exp: int | str, unit_exp: int | str | None,
                      si_unit: str | None, trim_digits: int | None):
        self._validate_value()
        self._validate_trim_and_precision(trim_digits)

        # 0) Parse unit and si_unit exponent to numerics or None.
        unit_exp = self._parse_unit_exp_unit_value(unit_exp=unit_exp, si_unit=si_unit)

        # 1) Parse exponents to numerics or None.
        mult_exp = _parse_exponent_value(mult_exp)
        unit_exp = _parse_exponent_value(unit_exp)

        # 2) Determine Exponents and Display Mode if not provided.
        self._set_exponents(mult_exp, unit_exp)

        # 3) Set trimmed digits for rounding in str representation.
        self._trim_digits = trim_digits

        # 4) Determine whole_as_int default.
        if self.whole_as_int is None:
            self.whole_as_int = self.mode != DisplayMode.PLAIN

        # 5) Validate final Config
        self._validate_exponents()

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
            Value 1.2e+3 s in PLAIN mode, the exponent == 0 as we do not mode multiplier and unit exponents here.
        """
        # The 'exponent' is the base for auto-calc of its compounds:
        #     exponent = multiplier_exponent + unit_exponent
        # so, the 'exponent' must the calculated from NumberUnit state, not from its properties
        self._validate_exponents()
        if self._mult_exp is None and self._unit_exp is None:
            return 0
        elif self._mult_exp == 0 and self._unit_exp == 0:
            return 0
        else:
            return self._src_exponent

    @property
    def mode(self) -> DisplayMode:
        """Derive mode mode from exponent state."""
        mult_exp = self._mult_exp
        unit_exp = self._unit_exp

        if mult_exp == 0 and unit_exp == 0:
            return DisplayMode.PLAIN

        elif mult_exp is None and isinstance(unit_exp, int):
            return DisplayMode.BASE_FIXED if unit_exp == 0 else DisplayMode.SI_FIXED

        elif isinstance(mult_exp, int) and unit_exp is None:
            return DisplayMode.SI_FLEX

        else:
            # This shouldn't happen if validation is correct
            raise ValueError(
                f"Invalid exponents, at least one should be int: mult_exp={mult_exp}, unit_exp={unit_exp}"
            )

    @property
    def multiplier_exponent(self) -> int:
        """
        Display exponent in SI_FIXED display mode, multiplier_exponent = exponent - unit_exponent.
        Returns 0 for other modes.

        Example:
            123.456×10³ km has exponent = 6 and multiplier_exponent = 3.
        """
        self._validate_exponents()
        if self._mult_exp is not None:
            return self._mult_exp
        return self.exponent - self._unit_exp

    @property
    def normalized(self) -> int | float | None:
        """
        Normalized value, mantissa without multiplier.

        normalized_value = value / ref_value, where ref_value = 10 ** exponent

        Includes rounding to trimmed digits if applicable.

        Example:
            displayed value 123.4×10³ ms has the normalized value 123.4
        """
        if self.value is None:
            return None

        norm_number = self.value / self.ref_value
        norm_number = _process_normalized_number(norm_number, significant_digits=self.trimmed_digits,
                                                 whole_as_int=self.whole_as_int)
        return norm_number

    @property
    def ref_value(self) -> int | float:
        """
        The reference value for scaling the normalized mode number:
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
        The SI Unit exponent used in SI mode modes; 0 in other modes.
        """
        self._validate_exponents()
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

        return f"{self.multi_operator}{value_multipliers[self.multiplier_exponent]}"

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

        if self.mode in DisplayMode:
            if self.whole_as_int or isinstance(display_number, int):
                return f"{display_number}{self._multiplier_str}"
            else:
                return f"{display_number:.{self.trimmed_digits}g}{self._multiplier_str}"
        else:
            raise ValueError(f"Invalid mode mode: {self.mode}")

    @property
    def _unit_order(self) -> int:
        """
        The SI Unit order in SI mode modes; 1 in other modes.

        Note: This property is kept for potential future extensions and testing.
        """
        return 10 ** self.unit_exponent

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

    def _validate_exponents(self):
        """Validate exponents based on Obj state, no properties involved"""
        mult_exp = self._mult_exp
        unit_exp = self._unit_exp

        if mult_exp is None and unit_exp is None:
            raise ValueError("At least one <int> value out of two '_mult_exp' or '_unit_exp' required")

        elif mult_exp == 0 and unit_exp == 0:
            return

        elif not isinstance(mult_exp, int) and not isinstance(unit_exp, int):
            raise ValueError("At least one <int> value out of two '_mult_exp' or '_unit_exp' "
                             "expected but none found.")

        if isinstance(mult_exp, int) and isinstance(unit_exp, int):
            if mult_exp != 0 or unit_exp != 0:
                raise ValueError(
                    "The <int> values '_mult_exp' and '_unit_exp' must be 0 if specified both")

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

    def _parse_si_exponent(self, exponent: int | str) -> int:
        """
        Parse valid SI prefix exponent (int like 3, or str like 'k') to its numeric value.

        Examples:
            _parse_si_exponent(3) == 3
            _parse_si_exponent("M") == 6
            _parse_si_exponent(2) raises ValueError
        """
        if isinstance(exponent, int):
            if exponent not in valid_exponents:
                raise ValueError(
                    f"Invalid unit_exponent integer value: {exponent}, expected one of {valid_exponents}"
                )
            return exponent
        if isinstance(exponent, str):
            if exponent not in valid_si_prefixes:
                raise ValueError(
                    f"Invalid unit_exponent string value: '{exponent}', expected one of {valid_si_prefixes}"
                )
            return si_prefixes.get_key(exponent)
        raise TypeError(f"Exponent must be an int or str with SI unit prefix, but got {type(exponent)}")

    def _parse_unit_exp_unit_value(self, unit_exp: int | None = None, si_unit: str | None = None) -> int | None:
        """
        Process unit_exp and si_unit to calculate corresponding self.unit and self.value
        """
        if self.unit is not None and si_unit is not None:
            raise ValueError("Cannot specify both 'unit' and 'si_unit' at the same time.")

        if unit_exp is not None and si_unit is not None:
            raise ValueError("Cannot specify both unit_exp and si_unit, use only one of them.")

        if unit_exp is not None:
            return unit_exp

        if si_unit is not None:
            # Should allow unit without SI Prefix too
            if not isinstance(si_unit, str):
                raise TypeError("si_unit must be a str | None.")
            si_prefix = sequence_get(si_unit, 0, default="")  # Should check the first letter
            if len(si_unit) > 1 and (si_prefix in valid_si_prefixes):
                # We are sure that both SI Prefix and Units are non-empty
                unit_exp = si_prefixes.get_key(si_prefix)
                self.unit = si_unit[1:]
                self.value = self.value * (10 ** unit_exp) if self.value is not None else None
            else:
                unit_exp = 0
                self.unit = si_unit
            return unit_exp

        return None

    def _set_exponents(self, mult_exp: int | None, unit_exp: int | None):
        """
        Determine Exponents if not provided.
        """
        # We should arrive here only with int | None exponents
        # When possible, we should get int exponents from _parse_exponent_value()
        # before entering this method
        if type(mult_exp) not in (int, type(None)):
            raise TypeError(f"mult_exp must be int or None, got {type(mult_exp)}")

        if type(unit_exp) not in (int, type(None)):
            raise TypeError(f"unit_exp must be int or None, got {type(unit_exp)}")

        if mult_exp is None and unit_exp is None:
            mult_exp = self._src_exponent or 0

        if mult_exp is not None and unit_exp is not None:
            # If Two int numbers given in exponents
            if mult_exp != 0 or unit_exp != 0:
                raise ValueError("'mult_exp' and 'unit_exp' must be 0 if specified both, "
                                 "use 2x None-s or only one of them otherwise.")
        self._mult_exp = mult_exp
        self._unit_exp = unit_exp

    @property
    def _src_exponent(self) -> int:
        """
        Returns the exponent of source NumberUnit.value represented a product of normalized and ref_value OR 0

        The exponent is a multiple of 3, and the resulting normalized number is in the range [1, 1000):

        value = normalized * 10^_src_exponent
        """
        if self.value in (0, None):
            return 0
        else:
            magnitude = math.floor(math.log10(abs(self.value)))
            src_exponent = (magnitude // 3) * 3
            return src_exponent

    @property
    def _src_significant_digits(self) -> int | None:
        """
        Calculate trimmed("significant") digits based on the source NumberUnit.value with trimmed_digits()

        Ignore trailing zeros in float as non-significant both before and after the decimal point
        """
        return trimmed_digits(self.value)


# Methods --------------------------------------------------------------------------------------------------------------

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
                f"Invalid exponent string value: '{exp}', expected one of "
                f"{valid_si_prefixes}"
            )
        return si_prefixes.get_key(exp)

    elif isinstance(exp, int):
        if exp not in valid_exponents:
            raise ValueError(
                f"Invalid exponent integer value: {exp}, expected one of {valid_exponents}"
            )
        return exp

    raise TypeError(f"Exponent value must be an int | str | None, got {type(exp)}")


def _process_normalized_number(norm_number: int | float | None,
                               significant_digits: int | None = None,
                               whole_as_int: bool = False) -> int | float | None:
    if norm_number is None:
        return None

    if not isinstance(norm_number, (int, float)):
        raise TypeError(f"Expected int | float, got {type(norm_number)}")

    if significant_digits is not None:
        if norm_number != 0:
            magnitude = math.floor(math.log10(abs(norm_number)))
            factor_ = 10 ** (significant_digits - 1 - magnitude)
            norm_number = round(norm_number * factor_) / factor_

    if whole_as_int and isinstance(norm_number, float) and norm_number.is_integer():
        return int(norm_number)

    return norm_number


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
