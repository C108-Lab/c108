#
# C108 - CLI Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
from pathlib import Path

# Third party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.cli import clify


# Tests ----------------------------------------------------------------------------------------------------------------

class TestClify:
    @pytest.mark.parametrize(
        "cmd,expected",
        [
            ('git commit -m "Initial commit"', ['git', 'commit', '-m', 'Initial commit']),
            ("echo 'hello world'", ['echo', 'hello world']),
            ("python -c 'print(1)'", ['python', '-c', 'print(1)']),
        ],
    )
    def test_string_shell_split(self, cmd, expected):
        assert clify(cmd) == expected

    def test_string_no_split_single_arg(self):
        assert clify("python -c 'print(1)'", shlex_split=False) == ["python -c 'print(1)'"]

    @pytest.mark.parametrize("cmd", [None, "", b"", bytearray()])
    def test_none_and_empty_inputs(self, cmd):
        assert clify(cmd) == []

    def test_iterable_mixed_types_and_pathlike_and_bytes(self):
        args = ['echo', 123, True, Path('some'), b'x']
        out = clify(args)
        assert out == ['echo', '123', 'True', 'some', 'x']

    def test_limits_max_items_iterable(self):
        gen = (str(i) for i in range(300))
        with pytest.raises(ValueError) as excinfo:
            clify(gen, max_items=256)
        assert "too many arguments" in str(excinfo.value)

    def test_limits_max_items_string_split(self):
        with pytest.raises(ValueError) as excinfo:
            clify("a b c", max_items=2)
        assert "too many arguments" in str(excinfo.value)

    def test_max_arg_length_violation(self):
        long_arg = "a" * 10
        with pytest.raises(ValueError) as excinfo:
            clify([long_arg], max_arg_length=5)
        assert "argument exceeds" in str(excinfo.value)

    def test_unsupported_type_raises_typeerror(self):
        with pytest.raises(TypeError) as excinfo:
            clify(object)  # not a string/bytes/bytearray/iterable/None
        assert "must be a string" in str(excinfo.value)

    def test_dict_iterable_yields_keys(self):
        d = {"a": 1, "b": 2}
        assert clify(d) == list(d.keys())

    def test_shlex_quotes_and_escapes(self):
        cmd = r'cmd --opt="a b" --path=\"/tmp/x\"'
        # shlex should keep "a b" as a single token and unescape quotes correctly
        out = clify(cmd)
        assert out[0] == "cmd"
        assert "--opt=a b" in out
        # The escaped quotes around /tmp/x become a literal quote in most shlex behaviors;
        # ensure the argument still contains /tmp/x
        assert any("/tmp/x" in part for part in out)

    def test_int_input_converts_to_string(self):
        """Test that int input is converted to a single string argument."""
        assert clify(42) == ["42"]
        assert clify(0) == ["0"]
        assert clify(-123) == ["-123"]

    def test_float_input_converts_to_string(self):
        """Test that float input is converted to a single string argument."""
        assert clify(3.14) == ["3.14"]
        assert clify(0.0) == ["0.0"]
        assert clify(-2.5) == ["-2.5"]
        assert clify(1e6) == ["1000000.0"]
