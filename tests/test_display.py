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


# Private Methods Tests ------------------------------------------------------------------------------------------------

class Test_AutoMultEponent:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(123e-6, -6, id="0.123->-3"),
            pytest.param(123e-3, -3, id="0.123->-3"),
            pytest.param(123, 0, id="3-digit->0"),
            pytest.param(123456, 3, id="6-digit->3"),
            pytest.param(1234567, 6, id="7-digit->6"),
        ],
    )
    def test_decimal_auto_multiplier_exp(self, value: int, expected: int):
        """Verify decimal auto multiplier exponent."""
        dv = DisplayValue(value, unit_exp=0, scale_type="decimal")
        assert dv._mult_exp == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(123, 0, id="lt-1Ki->exp0"),
            pytest.param(2 ** 12, 10, id="ge-1Ki-lt-1Mi->exp10"),
            pytest.param(2 ** 21, 20, id="ge-1Mi->exp20"),
        ],
    )
    def test_binary_auto_multiplier_exp(self, value: int, expected: int):
        """Verify binary auto multiplier exponent with 2^(10N)."""
        dv = DisplayValue(value, unit_exp=0, scale_type="binary")
        assert dv._mult_exp == expected


class Test_AutoUnitExponent:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(123e-6, -6, id="0.123->-6"),
            pytest.param(0.123, -3, id="0.123->-3"),
            pytest.param(123, 0, id="3-digit->base"),
            pytest.param(1_234, 3, id="4-digit->k"),
            pytest.param(123_456, 3, id="6-digit->k"),
            pytest.param(1_234_567, 6, id="7-digit->M"),
            pytest.param(123_456_789, 6, id="9-digit->M"),
            pytest.param(1_234_567_890, 9, id="10-digit->G"),
        ],
    )
    def test_decimal_auto_unit_exp(self, value: int, expected: int):
        """Verify decimal auto unit exponent selection with standard SI prefixes."""
        dv = DisplayValue(value, mult_exp=0, scale_type="decimal")
        assert dv._unit_exp == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(512, 0, id="lt-1Ki->base"),
            pytest.param(2 ** 10, 10, id="exactly-1Ki->Ki"),
            pytest.param(2 ** 10 * 500, 10, id="500Ki->Ki"),
            pytest.param(2 ** 20, 20, id="exactly-1Mi->Mi"),
            pytest.param(2 ** 20 * 500, 20, id="500Mi->Mi"),
            pytest.param(2 ** 30, 30, id="exactly-1Gi->Gi"),
        ],
    )
    def test_binary_auto_unit_exp(self, value: int, expected: int):
        """Verify binary auto unit exponent selection with IEC prefixes."""
        dv = DisplayValue(value, mult_exp=0, scale_type="binary")
        assert dv._unit_exp == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(500, 0, id="500->base"),
            pytest.param(1_000, 3, id="1k->k"),
            pytest.param(999_000, 3, id="999k->k"),
            pytest.param(1_000_000, 6, id="exact-1M"),  # Within scale_step
            pytest.param(10_000_000, 6, id="10M->M"),  # Beyond scale_step from k
            pytest.param(999_000_000, 6, id="999M->M"),
            pytest.param(1_000_000_000, 9, id="exact-1G"),
        ],
    )
    def test_decimal_prefixes_no_gap(self, value: int, expected: int):
        """Verify behavior with gaps in custom unit_prefixes (decimal)."""
        # Custom scale with gap: only base, k, M, G (missing intermediate prefixes)
        custom_prefixes = {0: "", 3: "k", 6: "M", 9: "G"}
        dv = DisplayValue(value, mult_exp=0, unit_prefixes=custom_prefixes, scale_type="decimal")
        assert dv._unit_exp == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(-1e30, 9, id="-1e30->9"),
            pytest.param(-100, 0, id="-100->base-0"),
            pytest.param(100, 0, id="+100->base-0"),
            pytest.param(10_000, 0, id="gap-lower-0"),
            pytest.param(100_000, 9, id="gap-upper-1G"),
            pytest.param(1_000_000_000, 9, id="exact-1G"),
            pytest.param(1234567_000_000_000, 9, id="1G->G"),
        ],
    )
    def test_decimal_prefixes_large_gap(self, value: int, expected: int):
        """Verify behavior with large gaps in custom unit_prefixes (decimal)."""
        # Large gap: only base, M, G (missing m, k)
        custom_prefixes = {0: "", 9: "G"}
        dv = DisplayValue(value, mult_exp=0, unit_prefixes=custom_prefixes, scale_type="decimal")
        assert dv._unit_exp == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(512, 0, id="512->base"),
            pytest.param(2 ** 10, 10, id="1Ki->Ki"),
            pytest.param(2 ** 19, 10, id="512Ki->Ki"),
            pytest.param(2 ** 20, 20, id="exact-1Mi"),  # Within scale_step
            pytest.param(2 ** 25, 20, id="32Mi->Mi"),  # Beyond scale_step from Ki
            pytest.param(2 ** 30, 30, id="1Gi->Gi"),
        ],
    )
    def test_binary_gap_in_prefixes(self, value: int, expected: int):
        """Verify behavior with gaps in custom unit_prefixes (binary)."""
        # Custom scale with some prefixes
        custom_prefixes = {0: "", 10: "Ki", 20: "Mi", 30: "Gi"}
        dv = DisplayValue(value, mult_exp=0, unit_prefixes=custom_prefixes, scale_type="binary")
        assert dv._unit_exp == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(1e-30, 0, id="1e-30->0"),
            pytest.param(1_000, 0, id="1k->0"),
            pytest.param(10_000, 0, id="10k->0"),
            pytest.param(1_000_000, 9, id="1M->9"),
            pytest.param(123_000_000, 9, id="123M->9"),
            pytest.param(1e30, 9, id="1e30->9"),
        ],
    )
    def test_decimal_only_two_prefixes(self, value: int, expected: int):
        """Verify behavior with minimal custom unit_prefixes (only two options)."""
        # Minimal scale: only k and M
        custom_prefixes = {0: "", 9: "G"}
        dv = DisplayValue(value, mult_exp=0, unit_prefixes=custom_prefixes, scale_type="decimal")
        assert dv._unit_exp == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(0, 0, id="zero->base"),
            pytest.param(0.0, 0, id="zero-float->base"),
            pytest.param(float('nan'), 0, id="nan->base"),
            pytest.param(float('inf'), 0, id="inf->base"),
            pytest.param(float('-inf'), 0, id="neg-inf->base"),
            pytest.param(None, 0, id="none->base"),
        ],
    )
    def test_non_finite_values(self, value, expected: int):
        """Verify non-finite values always return base unit exponent."""
        dv = DisplayValue(value, mult_exp=0, scale_type="decimal")
        assert dv._unit_exp == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(-123, 0, id="neg-3-digit->base"),
            pytest.param(-123_456, 3, id="neg-6-digit->k"),
            pytest.param(-1_234_567, 6, id="neg-7-digit->M"),
        ],
    )
    def test_negative_values(self, value: int, expected: int):
        """Verify negative values use absolute value for unit selection."""
        dv = DisplayValue(value, mult_exp=0, scale_type="decimal")
        assert dv._unit_exp == expected


class Test_DisplayValueValidators:

    def test_validates_unit_exp(self):
        with pytest.raises(ValueError, match="unit_exp must be one of SI decimal powers"):
            DisplayValue(123, unit_exp=5, scale_type="decimal")
        with pytest.raises(ValueError, match="unit_exp must be one of IEC binary powers"):
            DisplayValue(123, unit_exp=5, scale_type="binary")
        with pytest.raises(ValueError, match="unit_exp must be one of decimal powers"):
            DisplayValue(123, mult_exp=0, scale_type="decimal", unit_prefixes={0: "", 5: "penta"})
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
    @pytest.mark.parametrize(
        ("value", "mult_exp", "unit", "overflow_tolerance", "underflow_tolerance"),
        [
            pytest.param(123 * 10 ** 30, 0, "B", 5, 6, id="exp-30"),
            pytest.param(123 * 10 ** 33, 0, "B", 5, 6, id="exp-33"),
        ],
    )
    def test_overflow(self, value, mult_exp, unit, overflow_tolerance, underflow_tolerance):
        """Print unit exponent limits and overflow/underflow flags."""
        dv = DisplayValue(
            value,
            mult_exp=mult_exp,
            unit=unit,
            overflow_tolerance=overflow_tolerance,
            underflow_tolerance=underflow_tolerance,
        )
        print()
        print("_as_str                                ", dv)
        print("mult_exp/unit_exp                      ", f"{dv._mult_exp}/{dv._unit_exp}")
        print("residual_exp                           ", dv._residual_exponent)
        print(
            "overflow_tolerance/underflow_tolerance ",
            f"{dv.overflow_tolerance}/{dv.underflow_tolerance}",
        )
        print("overflow/underflow                     ", f"{dv._is_overflow}/{dv._is_underflow}")


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
        assert num_unit._as_str == "123.456"

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
        assert num_unit._as_str == "1.231e+23"

    def test_mode_si_fixed(self):
        print_method()

        # @formatter:off
        num_unit = DisplayValue(value=123.456, unit_exp=-3, unit="s", trim_digits=4)
        print(    "DisplayValue(value=123.456, unit_exp=-3, unit='s', trim_digits=4)")
        print(num_unit)
        print(dictify(num_unit, include_properties=True))
        assert num_unit._as_str == "123.5×10³ ms"

    def test_mode_si_flex(self):
        print_method()

        # @formatter:off
        num_unit = DisplayValue(value=123456, mult_exp=0, trim_digits=4)
        print(    "DisplayValue(value=123456, mult_exp=0, trim_digits=4)")
        print(num_unit)
        print(dictify(num_unit, include_properties=True))
        assert num_unit._as_str == "123.5k"

    def test_mode_mutliplier(self):
        print_method()

        # @formatter:off
        num_unit = DisplayValue(value=123456, unit_exp=0, unit='s')
        print(    "DisplayValue(value=123456, unit_exp=0, unit='s')")
        print(num_unit)
        print(dictify(num_unit, include_properties=True))
        assert num_unit._as_str == "123.456×10³ s"


    def test_exponents(self):
        print_method()

        # @formatter:off
        num_unit = DisplayValue(value=123.456, mult_exp=-3, unit='s')
        print(    "DisplayValue(value=123.456, mult_exp=-3, unit='s')")
        print(num_unit)
        print(dictify(num_unit, include_properties=True))

        # @formatter:on
        # Check Properties
        assert num_unit._as_str == "123.456×10⁻³ ks"

        print()

        num_unit = DisplayValue(value=123456, unit_exp=3, unit='m', mult_symbol=MultSymbol.CDOT)
        print("DisplayValue(value=123456789, unit_exp=3, unit='m', mult_symbol=MultSymbol.CDOT)")
        print(num_unit)
        print(dictify(num_unit, include_properties=True))

        # @formatter:on
        # Check Properties
        assert num_unit._as_str == "123.456 km"

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
        assert num_unit._as_str == "123.68"

        # @formatter:off
        num_unit = DisplayValue(value=123677.888, precision=1, mult_exp=3)
        print(    "DisplayValue(value=123677.888, precision=1, mult_exp=3)")
        print(num_unit)

        print(dictify(num_unit, include_properties=True))
        
        # @formatter:on
        # Check Properties
        assert num_unit.normalized == 123.677888
        assert num_unit._as_str == "123.7×10³"

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
        assert num_unit._as_str == "123.7×10⁶"

        num_unit = DisplayValue(value=123.67e6, trim_digits=2)
        print("DisplayValue(value=123.67e6, trim_digits=2)")
        print(num_unit)
        print(dictify(num_unit, include_properties=True))
        # @formatter:on
        # Check Properties
        assert num_unit.normalized == 120
        assert num_unit._as_str == "120×10⁶"

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
        assert DisplayValue(value=0, unit="byte", pluralize=True)._as_str == "0 bytes"
        assert DisplayValue(value=1, unit="byte", pluralize=True)._as_str == "1 byte"
        assert DisplayValue(value=2, unit="byte", pluralize=True)._as_str == "2 bytes"
        assert DisplayValue(value=2, unit="plr", pluralize=True, unit_plurals={"plr": "PLR"})._as_str == "2 PLR"
        # Non-pluralizable unit
        assert DisplayValue(value=2, unit="abc_def", pluralize=True)._as_str == "2 abc_def"

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
