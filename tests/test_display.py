#
# C108 - Display Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
from inspect import stack

# Third-party ----------------------------------------------------------------------------------------------------------
from pytest import raises

# Local ----------------------------------------------------------------------------------------------------------------
from c108.dictify import dictify
from c108.display import DisplayValue, DisplayMode, MultiSymbol


# Tests ----------------------------------------------------------------------------------------------------------------

class TestDisplayValueDEMO:
    pass

    def test_none(self):
        print_method()

        num_unit = DisplayValue(value=None)
        print("DisplayValue(value=None)")
        print("__str__", num_unit)
        print("__repr__", repr(num_unit))

    def test__str__repr__(self):
        print_method()

        num_unit = DisplayValue(value=123.456)
        print("DisplayValue(value=123.456)")
        print("__str__", num_unit)
        print("__repr__", repr(num_unit))
        print(dictify(num_unit, include_properties=True))

    def test_mode_plain(self):
        print_method()

        # @formatter:off
        num_unit = DisplayValue(value=123.456, mult_exp=0, unit_exp=0)
        print(    "DisplayValue(value=123.456, mult_exp=0, unit_exp=0)")
        print(num_unit)
        print(dictify(num_unit, include_properties=True))

        # @formatter:on
        # Check Fields
        assert num_unit.mode == DisplayMode.PLAIN
        assert num_unit.precision == None
        assert num_unit.trimmed_digits == 6
        assert num_unit.unit == None
        # Check Properties
        assert num_unit.normalized == 123.456
        assert num_unit.ref_value == 1
        assert num_unit._multiplier_str == ""
        assert num_unit.si_prefix == ""
        assert num_unit._number_str == "123.456"
        assert num_unit._units_str == ""
        assert num_unit.as_str == "123.456"

        # @formatter:off
        num_unit = DisplayValue(value=123.1e+21, mult_exp=0, unit_exp=0)
        print(    "DisplayValue(value=123.1e+21, mult_exp=0, unit_exp=0)")
        print(num_unit)
        print(dictify(num_unit, include_properties=True))
        # @formatter:on
        # Check Properties
        assert num_unit.mode == DisplayMode.PLAIN
        assert num_unit.normalized == 1.231e+23
        assert num_unit.ref_value == 1
        assert num_unit._multiplier_str == ""
        assert num_unit.si_prefix == ""
        assert num_unit._number_str == "1.231e+23"
        assert num_unit._units_str == ""
        assert num_unit.as_str == "1.231e+23"

    def test_mode_si_fixed(self):
        print_method()

        # @formatter:off
        num_unit = DisplayValue(value=123.456, unit_exp="m", unit="s", trim_digits=4)
        print(    "DisplayValue(value=123.456, unit_exp='m', unit='s', trim_digits=4)")
        print(num_unit)
        print(dictify(num_unit, include_properties=True))
        assert num_unit.as_str == "123.5×10³ ms"

    def test_mode_si_flexible(self):
        print_method()

        # @formatter:off
        num_unit = DisplayValue(value=123456, mult_exp=0, trim_digits=4)
        print(    "DisplayValue(value=123456, mult_exp=0, trim_digits=4)")
        print(num_unit)
        print(dictify(num_unit, include_properties=True))
        assert num_unit.as_str == "123.5k"

    def test_mode_mutliplier(self):
        print_method()

        # @formatter:off
        num_unit = DisplayValue(value=123456, unit_exp=0, unit='s')
        print(    "DisplayValue(value=123456, unit_exp=0, unit='s')")
        print(num_unit)
        print(dictify(num_unit, include_properties=True))
        assert num_unit.as_str == "123.456×10³ s"


    def test_exponents(self):
        print_method()

        # @formatter:off
        num_unit = DisplayValue(value=123.456, mult_exp=-3, unit='s')
        print(    "DisplayValue(value=123.456, mult_exp=-3, unit='s')")
        print(num_unit)
        print(dictify(num_unit, include_properties=True))

        # @formatter:on
        # Check Properties
        assert num_unit.as_str == "123.456×10⁻³ ks"

        print()

        num_unit = DisplayValue(value=123456, unit_exp='k', unit='m', multi_symbol=MultiSymbol.CDOT)
        print("DisplayValue(value=123456789, unit_exp='k', unit='m', multi_symbol=MultiSymbol.CDOT)")
        print(num_unit)
        print(dictify(num_unit, include_properties=True))

        # @formatter:on
        # Check Properties
        assert num_unit.as_str == "123.456 km"

    def test_precision(self):
        print_method()

        # @formatter:off
        num_unit = DisplayValue(value=123.67788, precision=2)
        print(    "DisplayValue(value=123.67788, precision=2)")
        print(num_unit)

        print(dictify(num_unit, include_properties=True))
        
        # @formatter:on
        # Check Properties
        assert num_unit.normalized == 123.67788
        assert num_unit.as_str == "123.68"

        # @formatter:off
        num_unit = DisplayValue(value=123677.888, precision=1, mult_exp=3)
        print(    "DisplayValue(value=123677.888, precision=1, mult_exp=3)")
        print(num_unit)

        print(dictify(num_unit, include_properties=True))
        
        # @formatter:on
        # Check Properties
        assert num_unit.normalized == 123.677888
        assert num_unit.as_str == "123.7×10³"

    def test_significant_digits_fixed(self):
        print_method()

        # @formatter:off
        num_unit = DisplayValue(value=123.67e6, trim_digits=4)
        print(    "DisplayValue(value=123.67e6, trim_digits=4)")
        # @formatter:on
        print(num_unit)
        print(dictify(num_unit, include_properties=True))
        # Check Properties
        assert num_unit.normalized == 123.7
        assert num_unit.as_str == "123.7×10⁶"

        num_unit = DisplayValue(value=123.67e6, trim_digits=2)
        print("DisplayValue(value=123.67e6, trim_digits=2)")
        print(num_unit)
        print(dictify(num_unit, include_properties=True))
        # @formatter:on
        # Check Properties
        assert num_unit.normalized == 120
        assert num_unit.as_str == "120×10⁶"

    def test_significant_digits_flex(self):
        print_method()

        # @formatter:off
        num_unit = DisplayValue(value=123.456)
        print(    "DisplayValue(value=123.456)")
        print(num_unit)
        print("num_unit.trimmed_digits    :", num_unit.trimmed_digits)
        
        print(dictify(num_unit, include_properties=True))
        
        # @formatter:on
        assert num_unit.trimmed_digits == 6

        # @formatter:off
        num_unit = DisplayValue(value=123000.0)
        print(    "DisplayValue(value=123000.0)")
        print(num_unit)
        print("num_unit.trimmed_digits    :", num_unit.trimmed_digits)

        print(dictify(num_unit, include_properties=True))
        
        # @formatter:on
        assert num_unit.trimmed_digits == 3

    def test_unit_pluralization(self):
        assert DisplayValue(value=0, unit="byte", plural_units=True).as_str == "0 bytes"
        assert DisplayValue(value=1, unit="byte", plural_units=True).as_str == "1 byte"
        assert DisplayValue(value=2, unit="byte", plural_units=True).as_str == "2 bytes"
        assert DisplayValue(value=2, unit="plr", plural_units={"plr": "PLR"}).as_str == "2 PLR"
        # Non-pluralizable unit
        assert DisplayValue(value=2, unit="abc", plural_units=True).as_str == "2 abc"

    def test_invalid_inputs(self):
        # Should fail if mode is PLAIN but an exponent is given
        with raises(ValueError, match="must be 0 if specified both"):
            DisplayValue(value=123, mult_exp=3, unit_exp=0)

        with raises(ValueError, match="must be 0 if specified both"):
            DisplayValue(value=123, mult_exp=0, unit_exp=3)

        # Should fail on an invalid exponent key
        with raises(ValueError, match="Invalid exponent integer value"):
            DisplayValue(value=1, mult_exp=1)

    def test_infinite_values(self):
        print_method()

        # @formatter:off
        num_unit = DisplayValue(None, unit="byte", mult_exp=3)
        print(    "DisplayValue(None, unit='byte', mult_exp=3)")
        print(num_unit)
        print(dictify(num_unit, include_properties=True))
        num_unit = DisplayValue(float("inf"), unit='bit', unit_exp=6)
        print(    "DisplayValue(float('inf'), unit='bit', unit_exp=6)")
        print(num_unit)
        num_unit = DisplayValue(float("-inf"), unit='bit', unit_exp=6)
        print(    "DisplayValue(float('-inf'), unit='bit', unit_exp=6)")
        print(num_unit)
        print(dictify(num_unit, include_properties=True))
        num_unit = DisplayValue(value=float("NaN"), unit='sec')
        print(    "DisplayValue(value=float('NaN'), unit='sec')")
        print(num_unit)
        print(dictify(num_unit, include_properties=True))



def print_method(prefix: str = "------- ",
                 suffix: str = " -------",
                 start: str = "\n\n",
                 end: str = "\n"):
    method_name = stack()[1][3]
    _print_title(title=method_name, prefix=prefix, suffix=suffix, start=start, end=end)


def _print_title(title,
                 prefix: str = "------- ",
                 suffix: str = " -------",
                 start: str = "\n",
                 end: str = "\n"):
    """
    Prints a formatted title to the console.

    Args:
        title (str): The main title string to be printed.
        prefix (str, optional): A string to prepend to the title. Defaults to "------- ".
        suffix (str, optional): A string to append to the title. Defaults to " -------".
        start (str, optional): A string to print before the entire formatted title. Defaults to "\n".
        end (str, optional): A string to print after the entire formatted title. Defaults to "\n".
    """
    print(f"{start}{prefix}{title}{suffix}{end}", end="")
