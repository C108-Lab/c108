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


class UnitsConf:
    PLURALIZABLE_UNITS = ("byte", "step", "item",
                          "second", "minute", "hour", "day", "week", "month", "year",
                          "meter", "gram")


units_conf = UnitsConf()

# @formatter:off
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
class NumDisplay(StrEnum):
    """
    Styles of numerical display.

    Attributes:
        PLAIN (str)      : Base units, E-notation for floats - 1 byte, 2.2+e3 s
        SI_FIXED (str)   : Fixed SI units prefix with a flexible value multiplier - 123×10³ Mbyte
        SI_FLEXIBLE (str): Flexible SI units prefix, no value multiplier - 123.4 ns
        MULTIPLIER (str) : Flexible value multiplier, base units without SI prefixes - 123×10⁹ byte
    """
    PLAIN = "plain"
    SI_FIXED = "si_fixed"
    SI_FLEXIBLE = "si_flexible"
    MULTIPLIER = "multiplier"
# @formatter:on

@unique
class MultiOperator(StrEnum):
    ASTERISK = "*"
    CDOT = "⋅"
    CROSS = "×"


@dataclass
class NumberUnit:
    """
    Formats a number with an optional unit, supporting various display modes.

    This class can display numbers in plain format, with SI prefixes (flexible or
    fixed), or with an explicit power-of-ten multiplier. It automatically handles
    exponent calculation, significant digits, and unit pluralization.

    Key Parameters:
      - value: Numerical value in base units (e.g., 0.001 for millisecond)
      - display: Core display mode (PLAIN, SI_FLEXIBLE, SI_FIXED, MULTIPLIER).
      - precision: Number of decimal digits for float values in str.
      - sig_digits: Significant digits for rounding in numeric str.
      - si_unit: Fixed SI units prefix (e.g., "k" for kbyte) in numerator.
      - mult_exp: Multiplier exponent in numeric str (e.g., 3 in 1.2×10³)
      - unit_exp: Fixed SI units exponent, e.g., 3 or 'k' for kbyte.
      - unit: The unit string (e.g., "s" for seconds).
      - whole_as_int: Display <float> whole numbers as integers (e.g., 10.0 -> 10).

    Fractional units can be represented with denominator in the unit string
    and scaling the value accordingly. Example:

        bytes_per_s = 123_000 # source value in base units
        bytes_per_ms = bytes_per_s / 10**3
        speed_num = NumberUnit(value=bytes_per_ms, mult_exp=0, unit='byte/ms')
    """

    value: int | float | None
    display: NumDisplay | None = None

    mult_exp: InitVar[int | str] = None
    unit_exp: InitVar[int | str | None] = None
    si_unit: InitVar[str | None] = None
    sig_digits: InitVar[int | None] = None

    precision: int | None = None
    whole_as_int: bool | None = None

    multi_operator: str = MultiOperator.CROSS
    separator: str = " "

    unit: str | None = None
    pluralize_units: bool | None = None

    _multiplier_exponent: int | None = field(init=False, default=None)
    _significant_digits: int | None = field(init=False, default=None)
    _unit_exponent: int | None = field(init=False, default=None)

    def __post_init__(self, mult_exp: int | str, unit_exp: int | str | None,
                      si_unit: str | None, sig_digits: int | None):
        self.validate_value()

        # 0) Parse unit and si_unit exponent to numerics or None.
        unit_exp = self._parse_units(unit_exp=unit_exp, si_unit=si_unit)

        # 1) Parse exponents to numerics or None.
        mult_exp = parse_exponent_value(mult_exp)
        unit_exp = parse_exponent_value(unit_exp)

        # 2) Determine Exponents and Display Mode if not provided.
        self._set_exponents(mult_exp, unit_exp)
        self._set_display_mode_if_None()

        # 3) Set significant digits for rounding in str representation.
        self._significant_digits = sig_digits

        # 4) Determine whole_as_int default.
        if self.whole_as_int is None:
            self.whole_as_int = self.display != NumDisplay.PLAIN

        # 5) Validate final Config
        self.validate_display_mode()
        self.validate_exponents()

    def __str__(self):
        return self.as_str

    @property
    def as_str(self):
        """Number with units as a string."""
        if not self.units_str:
            return self.number_str

        return f"{self.number_str}{self.separator}{self.units_str}"

    @property
    def exponent(self) -> int:
        """
        The total exponent at normalized value, equals to source value exponent or 0.

        exponent = multiplier_exponent + unit_exponent

        Example:
            Value is 123.456×10³ byte, the exponent = 3;
            Value is 123.456×10³ kbyte the exponent = 6, and multiplier_exponent = 3;
            Value is 1.2e+3 s in PLAIN mode, the exponent = 0 as we do not display multiplier and unit exponents here.
        """
        # The 'exponent' is the base for auto-calc of its compounds:
        #     exponent = multiplier_exponent + unit_exponent
        # so, the 'exponent' must the calculated from NumberUnit state, not from its properties
        self.validate_exponents()
        if self._multiplier_exponent is None and self._unit_exponent is None:
            return 0
        elif self._multiplier_exponent == 0 and self._unit_exponent == 0:
            return 0
        else:
            return self._src_exponent

    @property
    def multiplier_exponent(self) -> int:
        """
        Display exponent in SI_FIXED display mode, multiplier_exponent = exponent - unit_exponent.
        Returns 0 for other modes.

        Example:
            123.456×10³ km has exponent = 6 and multiplier_exponent = 3.
        """
        self.validate_exponents()
        if self._multiplier_exponent is not None:
            return self._multiplier_exponent
        return self.exponent - self.unit_exponent

    @property
    def multiplier_str(self) -> str:
        """
        Numeric multiplier suffix.

        Example:
            displayed value 123×10³ byte has multiplier_str of ×10³
        """
        self.validate_display_mode()
        if self.multiplier_exponent == 0:
            return ""

        return f"{self.multi_operator}{value_multipliers[self.multiplier_exponent]}"

    @property
    def normalized(self) -> int | float | None:
        """
        Normalized value, mantissa without multiplier.

        normalized_value = value / ref_value, where ref_value = 10 ** exponent

        Includes rounding to significant digits if applicable.

        Example:
            displayed value 123.4×10³ ms has the normalized value 123.4
        """
        if self.value is None:
            return None

        norm_number = self.value / self.ref_value
        norm_number = process_normalized_number(norm_number, significant_digits=self.significant_digits,
                                                whole_as_int=self.whole_as_int)
        return norm_number

    @property
    def number_str(self) -> str:
        """
        Number as a string including the multiplier if applicable.

        Example:
            The value 123.456×10³ km has number_str 123.456×10³
        """
        self.validate_display_mode()
        display_number = self.normalized

        if display_number is None:
            return str(display_number)

        if self.precision is not None:
            return f"{display_number:.{self.precision}f}{self.multiplier_str}"

        if self.display in NumDisplay:
            if self.whole_as_int or isinstance(display_number, int):
                return f"{display_number}{self.multiplier_str}"
            else:
                return f"{display_number:.{self.significant_digits}g}{self.multiplier_str}"
        else:
            raise ValueError(f"Invalid display mode: {self.display}")

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
    def significant_digits(self) -> int | None:

        if self._significant_digits is not None:
            return self._significant_digits

        return self._src_significant_digits

    @property
    def unit_exponent(self) -> int:
        """The SI Unit exponent used in SI display modes; 0 in other modes."""
        self.validate_exponents()
        if self._unit_exponent is not None:
            return self._unit_exponent
        return self.exponent - self._multiplier_exponent

    @property
    def unit_order(self) -> int:
        """The SI Unit order in SI display modes; 1 in other modes."""
        return 10 ** self.unit_exponent

    @property
    def units_str(self) -> str:
        """
        Units of measurement as a string including SI prefix if applicable.

        Example:
            123 ms has units_str = 'ms'.
        """
        self.validate_display_mode()
        if not self.unit:
            if self.si_prefix:
                return f"{self.si_prefix}1"
            else:
                return ""
        elif self.normalized == 1:
            return f"{self.si_prefix}{self.unit}"

        if self.pluralize_units is True:
            return f"{self.si_prefix}{self.unit}s"
        elif self.pluralize_units is False:
            # ! We should NOT process this case if pluralize_units is None
            return f"{self.si_prefix}{self.unit}"
        elif self.pluralize_units is None:
            if self.unit in units_conf.PLURALIZABLE_UNITS:
                return f"{self.si_prefix}{self.unit}s"
            else:
                return f"{self.si_prefix}{self.unit}"
        raise ValueError(f"Invalid pluralize_units value: {self.pluralize_units}")

    def validate_display_mode(self):
        """Validate display mode based on Obj state, no properties involved"""
        if self.value is None:
            # None as source value should be acceptable in any display mode.
            return
        # if self.display in (NumDisplay.SI_FIXED, NumDisplay.SI_FLEXIBLE) and not self.unit:
        #     raise ValueError("The 'unit' is required for SI display mode.")
        if self.display == NumDisplay.PLAIN and (self._multiplier_exponent != 0 or self._unit_exponent != 0):
            raise ValueError(f"Cannot use PLAIN display mode with a non-zero exponents, both expected to be 0 "
                             f"mult_exp/unit_exp: {self._multiplier_exponent}/{self._unit_exponent}")
        if self.display == NumDisplay.MULTIPLIER:
            if not isinstance(self._multiplier_exponent, int):
                raise ValueError("The 'multiplier_exponent' of int type required for MULTIPLIER display mode.")

    def validate_exponents(self):
        """Validate exponents based on Obj state, no properties involved"""
        mult_exp = self._multiplier_exponent
        unit_exp = self._unit_exponent

        if mult_exp is None and unit_exp is None:
            raise ValueError("At least one <int> value out of two '_multiplier_exponent' or '_unit_exponent' required")

        elif mult_exp == 0 and unit_exp == 0:
            return

        elif not isinstance(mult_exp, int) and not isinstance(unit_exp, int):
            raise ValueError("At least one <int> value out of two '_multiplier_exponent' or '_unit_exponent' "
                             "expected but none found.")

        if isinstance(mult_exp, int) and isinstance(unit_exp, int):
            if mult_exp != 0 or unit_exp != 0:
                raise ValueError(
                    "The <int> values '_multiplier_exponent' and '_unit_exponent' must be 0 if specified both")

    def validate_value(self):
        """Validate value based on Obj state, no properties involved"""
        if not isinstance(self.value, (int, float, type(None))):
            raise ValueError(f"The 'value' must be int | float | None: {type(self.value)}.")

    def _parse_si_exponent(self, exponent: int | str) -> int:
        """Parse the fixed SI prefix exponent (int like 3, or str like 'k')."""
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
        raise TypeError(f"Exponent must be an int or str with SI unit prefix, but got {type(value)}")

    def _parse_units(self, unit_exp: int | None = None, si_unit: str | None = None) -> int | None:
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

    def _set_display_mode_if_None(self):
        if self.display is None:
            if self._multiplier_exponent == 0 and self._unit_exponent == 0:
                self.display = NumDisplay.PLAIN
            elif isinstance(self._multiplier_exponent, int):
                self.display = NumDisplay.MULTIPLIER
            elif isinstance(self._unit_exponent, int):
                self.display = NumDisplay.SI_FIXED
            else:
                raise ValueError(
                    f"Invalid Exponents. Cannot determine display mode for mult_exp/unit_exp: "
                    f"{self._multiplier_exponent}/{self._unit_exponent}")

    def _set_exponents(self, mult_exp: int | None, unit_exp: int | None):
        # We should arrive here only with int | None exponents
        # Normally we get numerics with parse_exponent_value()
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
        self._multiplier_exponent = mult_exp
        self._unit_exponent = unit_exp

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
        Calculate significant digits based on the source NumberUnit.value

        Ignore trailing zeros in float as non-significant both before and after the decimal point
        """
        return trimmed_digits(self.value)


# Methods --------------------------------------------------------------------------------------------------------------

def parse_exponent_value(exp: int | str | None) -> int | None:
    """Parse an exponent from an integer or SI prefix string."""
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


def process_normalized_number(norm_number: int | float | None,
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
    Calculate significant digits by removing trailing zeros from both integers and floats.
    All trailing zeros are treated as non-significant in float as well as integers.

    NOTE: Ignoring trailing zeros BEFORE decimal point in float is not standard engineering or scientific
    practice so in this regard this method is not suitable for significant digits calculation.
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
