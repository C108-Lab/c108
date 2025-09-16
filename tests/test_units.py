#
# C108 - Units Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
from inspect import stack

# Third-party ----------------------------------------------------------------------------------------------------------
from pytest import raises

# Local ----------------------------------------------------------------------------------------------------------------
from c108.dictify import dictify
from c108.tools import print_title
from c108.units import NumberUnit, NumDisplay, MultiOperator


# Tests ----------------------------------------------------------------------------------------------------------------

class TestNumUnits:

    def test_none(self):
        print_method()

        num_unit = NumberUnit(value=None)
        print("NumberUnit(value=None)")
        print("__str__", num_unit)
        print("__repr__", repr(num_unit))

    def test__str__repr__(self):
        print_method()

        num_unit = NumberUnit(value=123.456)
        print("NumberUnit(value=123.456)")
        print("__str__", num_unit)
        print("__repr__", repr(num_unit))
        print(to_dict(num_unit, inc_property=True))

    def test_mode_plain(self):
        print_method()

        # @formatter:off
        num_unit = NumberUnit(value=123.456, mult_exp=0, unit_exp=0)
        print(    "NumberUnit(value=123.456, mult_exp=0, unit_exp=0)")
        print(num_unit)
        print(to_dict(num_unit, inc_property=True))

        # @formatter:on
        # Check Fields
        assert num_unit.precision == None
        assert num_unit.significant_digits == 6
        assert num_unit.unit == None
        # Check Properties
        assert num_unit.normalized == 123.456
        assert num_unit.ref_value == 1
        assert num_unit.multiplier_str == ""
        assert num_unit.si_prefix == ""
        assert num_unit.number_str == "123.456"
        assert num_unit.units_str == ""
        assert num_unit.as_str == "123.456"

        # @formatter:off
        num_unit = NumberUnit(value=123.1e+21, mult_exp=0, unit_exp=0)
        print(    "NumberUnit(value=123.1e+21, mult_exp=0, unit_exp=0)")
        print(num_unit)
        # @formatter:on
        # Check Properties
        assert num_unit.normalized == 1.231e+23
        assert num_unit.ref_value == 1
        assert num_unit.multiplier_str == ""
        assert num_unit.si_prefix == ""
        assert num_unit.number_str == "1.231e+23"
        assert num_unit.units_str == ""
        assert num_unit.as_str == "1.231e+23"

    def test_mode_si_fixed(self):
        print_method()

        # @formatter:off
        num_unit = NumberUnit(value=123.456, unit_exp="m", unit="s", sig_digits=4)
        print(    "NumberUnit(value=123.456, unit_exp='m', unit='s', sig_digits=4)")
        print(num_unit)
        assert num_unit.as_str == "123.5×10³ ms"

    def test_mode_si_flexible(self):
        print_method()

        # @formatter:off
        num_unit = NumberUnit(value=123456, mult_exp=0, sig_digits=4)
        print(    "NumberUnit(value=123456, mult_exp=0, sig_digits=4)")
        print(num_unit)
        assert num_unit.as_str == "123.5 k1"

    def test_mode_mutliplier(self):
        print_method()

        # @formatter:off
        num_unit = NumberUnit(value=123456, unit_exp=0, unit='s')
        print(    "NumberUnit(value=123456, unit_exp=0, unit='s')")
        print(num_unit)
        assert num_unit.as_str == "123.456×10³ s"


    def test_exponents(self):
        print_method()

        # @formatter:off
        num_unit = NumberUnit(value=123.456, mult_exp=-3, unit='s')
        print(    "NumberUnit(value=123.456, mult_exp=-3, unit='s')")
        print(num_unit)
        print("num_unit.normalized         ", num_unit.normalized)
        print("num_unit.multiplier_exponent", num_unit.multiplier_exponent)
        print("num_unit.unit_exponent      ", num_unit.unit_exponent)

        # @formatter:on
        # Check Properties
        assert num_unit.as_str == "123.456×10⁻³ ks"

        print()

        num_unit = NumberUnit(value=123456, unit_exp='k', unit='m', multi_operator=MultiOperator.CDOT)
        print("NumberUnit(value=123456, unit_exp='k', unit='m', multi_operator=MultiOperator.CDOT)")
        print(num_unit)
        print("num_unit.normalized         ", num_unit.normalized)
        print("num_unit.multiplier_exponent", num_unit.multiplier_exponent)
        print("num_unit.unit_exponent      ", num_unit.unit_exponent)
        # @formatter:on
        # Check Properties
        assert num_unit.as_str == "123.456 km"

    def test_si_unit(self):
        print_method()

        # @formatter:off
        num_unit = NumberUnit(value=123456, si_unit='')
        print(    "NumberUnit(value=123456, si_unit='')")
        # @formatter:on
        print(num_unit)
        print("num_unit.unit_exponent:", num_unit.unit_exponent)
        print("num_unit.unit         :", num_unit.unit)

        assert num_unit.unit == ""
        assert num_unit.unit_exponent == 0
        assert num_unit.unit_order == 10 ** 0

        # @formatter:off
        num_unit = NumberUnit(value=123456, si_unit='ms')
        print(    "NumberUnit(value=123456, si_unit='ms')")
        # @formatter:on
        print(num_unit)
        print("num_unit.unit_exponent:", num_unit.unit_exponent)
        print("num_unit.unit         :", num_unit.unit)

        assert num_unit.unit == "s"
        assert num_unit.unit_exponent == -3
        assert num_unit.unit_order == 10 ** -3

        # @formatter:off
        num_unit = NumberUnit(value=0.001234, si_unit='kbyte')
        print(    "NumberUnit(value=0.001234, si_unit='kbyte')")
        # @formatter:on
        print(num_unit)
        print("num_unit.unit_exponent:", num_unit.unit_exponent)
        print("num_unit.unit         :", num_unit.unit)

        assert num_unit.unit == "byte"
        assert num_unit.unit_exponent == 3

        # @formatter:off
        num_unit = NumberUnit(value=1, si_unit='m')
        print(    "NumberUnit(value=1, si_unit='m')")
        # @formatter:on
        print(num_unit)
        print("num_unit.unit_exponent:", num_unit.unit_exponent)
        print("num_unit.unit         :", num_unit.unit)

        assert num_unit.unit == "m"
        assert num_unit.unit_exponent == 0

        # @formatter:off
        num_unit = NumberUnit(value=1, si_unit='mm')

        print(    "NumberUnit(value=1, si_unit='mm')")
        # @formatter:on
        print(num_unit)
        print("num_unit.unit_exponent:", num_unit.unit_exponent)
        print("num_unit.unit         :", num_unit.unit)

        assert num_unit.unit == "m"
        assert num_unit.unit_exponent == -3

    def test_unit_order(self):
        print_method()

        # @formatter:off
        num_unit = NumberUnit(value=123.456, unit_exp='m', unit='s')
        print(    "NumberUnit(value=123.456, unit_exp='m', unit='s')")
        print(num_unit)
        print("num_unit.unit_exponent:", num_unit.unit_exponent)
        print("num_unit.unit_order   :", num_unit.unit_order)

        assert num_unit.unit_exponent == -3
        assert num_unit.unit_order == 10 ** -3


    def test_precision(self):
        print_method()

        # @formatter:off
        num_unit = NumberUnit(value=123.67788, precision=2)
        print(    "NumberUnit(value=123.67788, precision=2)")
        print(num_unit)
        # @formatter:on
        # Check Properties
        assert num_unit.normalized == 123.67788
        assert num_unit.as_str == "123.68"

        # @formatter:off
        num_unit = NumberUnit(value=123677.888, precision=1, mult_exp=3)
        print(    "NumberUnit(value=123677.888, precision=1, mult_exp=3)")
        print(num_unit)
        # @formatter:on
        # Check Properties
        assert num_unit.normalized == 123.677888
        assert num_unit.as_str == "123.7×10³"

    def test_significant_digits_fixed(self):
        print_method()

        # @formatter:off
        num_unit = NumberUnit(value=123.67e6, sig_digits=4)
        print(    "NumberUnit(value=123.67e6, sig_digits=4)")
        # @formatter:on
        print(num_unit)
        print(to_dict(num_unit, inc_property=True))
        # Check Properties
        assert num_unit.normalized == 123.7
        assert num_unit.as_str == "123.7×10⁶"

        num_unit = NumberUnit(value=123.67e6, sig_digits=2)
        print("NumberUnit(value=123.67e6, sig_digits=2)")
        print(num_unit)
        # @formatter:on
        # Check Properties
        assert num_unit.normalized == 120
        assert num_unit.as_str == "120×10⁶"

    def test_significant_digits_flex(self):
        print_method()

        # @formatter:off
        num_unit = NumberUnit(value=123.456)
        print(    "NumberUnit(value=123.456)")
        print(num_unit)
        print("num_unit.significant_digits    :", num_unit.significant_digits)
        # @formatter:on
        assert num_unit.significant_digits == 6

        # @formatter:off
        num_unit = NumberUnit(value=123000.0)
        print(    "NumberUnit(value=123000.0)")
        print(num_unit)
        print("num_unit.significant_digits    :", num_unit.significant_digits)
        # @formatter:on
        assert num_unit.significant_digits == 3

    def test_unit_pluralization(self):
        assert NumberUnit(value=0, unit="byte").as_str == "0 bytes"
        assert NumberUnit(value=1, unit="byte").as_str == "1 byte"
        assert NumberUnit(value=2, unit="byte").as_str == "2 bytes"
        assert NumberUnit(value=2, unit="abc", pluralize_units=True).as_str == "2 abcs"
        # Non-pluralizable unit
        assert NumberUnit(value=5, unit="s").as_str == "5 s"
        assert NumberUnit(value=2, unit="abc", pluralize_units=False).as_str == "2 abc"

    def test_invalid_inputs(self):
        # Should fail if display is PLAIN but an exponent is given
        with raises(ValueError, match="Cannot use PLAIN display mode with a non-zero exponent"):
            NumberUnit(value=123, mult_exp=3, display=NumDisplay.PLAIN)

        with raises(ValueError, match="Cannot use PLAIN display mode with a non-zero exponent"):
            NumberUnit(value=123, unit_exp=3, display=NumDisplay.PLAIN)

        # Should fail on an invalid exponent key
        with raises(ValueError, match="Invalid exponent integer value"):
            NumberUnit(value=1, mult_exp=1)


def print_method(prefix: str = "------- ",
                 suffix: str = " -------",
                 start: str = "\n\n",
                 end: str = "\n"):
    method_name = stack()[1][3]
    print_title(title=method_name, prefix=prefix, suffix=suffix, start=start, end=end)
