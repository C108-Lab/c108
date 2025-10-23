#
# C108 - Display Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
from inspect import stack

# Third-party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.dictify import dictify
from c108.display import DisplayValue, DisplayMode, MultSymbol
from c108.display import trimmed_digits, _disp_power


# Tests ----------------------------------------------------------------------------------------------------------------


class TestDisplayValueValidators:

    def test_validates_unit_exp(self):
        with pytest.raises(ValueError, match="unit_exp must be one of SI decimal powers"):
            DisplayValue(123, unit_exp=5, scale_type="decimal")
        with pytest.raises(ValueError, match="unit_exp must be one of IEC binary powers"):
            DisplayValue(123, unit_exp=5, scale_type="binary")
        with pytest.raises(ValueError, match="unit_exp must be one of decimal powers"):
            DisplayValue(123, mult_exp=0, scale_type="decimal", unit_prefixes={0: "", 5:"penta"})
        # Empty unit_prefixes map should fall back to default mapping
        dv = DisplayValue(123, mult_exp=0, scale_type="decimal", unit_prefixes={})

class Test_DispPower:
    @pytest.mark.parametrize(
        ("base", "power", "fmt", "expected"),
        [
            pytest.param(10, -6, "unicode", "10⁻⁶", id="unicode-neg"),
            pytest.param(2, 3, "caret", "2^3", id="caret-pos"),
            pytest.param(10, 3, "python", "10**3", id="python-pos"),
            pytest.param(2, 20, "latex", "2^{20}", id="latex-pos"),
        ],
    )
    def test_render_modes(self, base: int, power: int, fmt: str, expected: str) -> None:
        """Render power across formats."""
        result = _disp_power(base=base, power=power, format=fmt)
        assert result == expected

    @pytest.mark.parametrize(
        ("base", "fmt"),
        [
            pytest.param(10, "unicode", id="unicode"),
            pytest.param(2, "caret", id="caret"),
            pytest.param(3, "python", id="python"),
            pytest.param(5, "latex", id="latex"),
        ],
    )
    def test_zero_power(self, base: int, fmt: str) -> None:
        """Return empty string for zero power."""
        result = _disp_power(base=base, power=0, format=fmt)
        assert result == ""

    def test_invalid_power_format(self):
        with pytest.raises(ValueError, match="invalid power format"):
            _disp_power(power=123, format="unknown")

class Test_IsOverflowUnderflow:
    def test_render_modes(self):
        """Get proper unit_exp limits and is_overflow/underflow flag."""
        dv = DisplayValue(123*10**23, mult_exp=0, unit="B", overflow_tolerance=5, underflow_tolerance=6)
        # TODO unit_exp should get closest unit_prefixes key instead of raising exc at 123*10**30 OR 123*10**-30
        print(dv)
        print(dv._unit_exp_min)
        print(dv._unit_exp_max)
        print(dv._is_overflow)
        print(dv._is_underflow)
        # assert result == expected



# DEMO-s ---------------------------------------------------------------------------------------------------------------

class Test_DEMO_DisplayValue:
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
        assert num_unit.trim_digits == 6
        assert num_unit.unit == None
        # Check Properties
        assert num_unit.normalized == 123.456
        assert num_unit.ref_value == 1
        assert num_unit._multiplier_str == ""
        assert num_unit.unit_prefix == ""
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
        assert num_unit.unit_prefix == ""
        assert num_unit._number_str == "1.231e+23"
        assert num_unit._units_str == ""
        assert num_unit.as_str == "1.231e+23"

    def test_mode_si_fixed(self):
        print_method()

        # @formatter:off
        num_unit = DisplayValue(value=123.456, unit_exp=-3, unit="s", trim_digits=4)
        print(    "DisplayValue(value=123.456, unit_exp=-3, unit='s', trim_digits=4)")
        print(num_unit)
        print(dictify(num_unit, include_properties=True))
        assert num_unit.as_str == "123.5×10³ ms"

    def test_mode_si_flex(self):
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

        num_unit = DisplayValue(value=123456, unit_exp=3, unit='m', mult_symbol=MultSymbol.CDOT)
        print("DisplayValue(value=123456789, unit_exp=3, unit='m', mult_symbol=MultSymbol.CDOT)")
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
        print("num_unit.trim_digits    :", num_unit.trim_digits)
        
        print(dictify(num_unit, include_properties=True))
        
        # @formatter:on
        assert num_unit.trim_digits == 6

        # @formatter:off
        num_unit = DisplayValue(value=123000.0)
        print(    "DisplayValue(value=123000.0)")
        print(num_unit)
        print("num_unit.trim_digits    :", num_unit.trim_digits)

        print(dictify(num_unit, include_properties=True))
        
        # @formatter:on
        assert num_unit.trim_digits == 3

    def test_unit_pluralization(self):
        assert DisplayValue(value=0, unit="byte", pluralize=True).as_str == "0 bytes"
        assert DisplayValue(value=1, unit="byte", pluralize=True).as_str == "1 byte"
        assert DisplayValue(value=2, unit="byte", pluralize=True).as_str == "2 bytes"
        assert DisplayValue(value=2, unit="plr", pluralize=True, unit_plurals={"plr": "PLR"}).as_str == "2 PLR"
        # Non-pluralizable unit
        assert DisplayValue(value=2, unit="abc_def", pluralize=True).as_str == "2 abc_def"

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
