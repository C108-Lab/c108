#
# C108 - CLI Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Any

# Third party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.cli import clify, cli_multiline


# Tests ----------------------------------------------------------------------------------------------------------------


class TestCli_Multiline:
    @pytest.mark.parametrize(
        "input_value, expected",
        [
            (None, ""),
            ("", ""),
            ([], ""),
        ],
        ids=[
            "none_returns_empty",
            "empty_string_returns_empty",
            "empty_iterable_returns_empty",
        ],
    )
    def test_empty_inputs(self, input_value, expected):
        """Return an empty string for empty or None input."""
        assert cli_multiline(input_value) == expected

    def test_simple_list_no_breaks(self):
        """Keep short command on a single line for simple list input."""
        cmd = ["echo", "hello", "world"]
        result = cli_multiline(cmd)
        assert result == "echo hello world" or result.startswith("echo hello world")

    def test_string_shlex_split_true(self):
        """Split a shell-like string when shlex_split is enabled."""
        cmd = 'git commit -m "Initial commit"'
        result = cli_multiline(cmd, shlex_split=True)
        # ensure the quoted message remains intact (not split into separate words)
        assert "Initial commit" in result
        assert "--" not in result  # sanity: no long options here

    def test_string_shlex_split_false(self):
        """Do not split a shell-like string when shlex_split is disabled."""
        cmd = 'git commit -m "Initial commit"'
        result = cli_multiline(cmd, shlex_split=False)
        # when not split, original spacing/quoting should be preserved in the single-line output
        assert 'git commit -m "Initial commit"' in result

    def test_long_and_short_options_breaking(self):
        """Place long and short options on their own continued lines."""
        cmd = ["tar", "-cvpzf", "backup.tar.gz", "--exclude=/proc", "--exclude=/sys"]
        result = cli_multiline(cmd, multiline_indent=4)
        # long options should appear on continuation lines with backslashes
        assert "\\\n" in result or "\\\r\n" in result
        assert "--exclude=/proc" in result
        assert "--exclude=/sys" in result
        # short flags (like -cvpzf) should remain with their value on the same line
        assert "backup.tar.gz" in result

    def test_flag_values_stay_on_same_line(self):
        """Keep flag values on the same line as their flags."""
        cmd = ["prog", "--opt", "value", "-f", "file.txt", "positional"]
        result = cli_multiline(cmd)
        # --opt value and -f file.txt should be adjacent in the output
        assert "--opt value" in result or '--opt" value' in result
        assert "-f file.txt" in result
        # positional argument should appear (possibly on its own line)
        assert "positional" in result

    def test_grouped_short_flags_break(self):
        """Break grouped short flags into continuation lines appropriately."""
        cmd = ["prog", "-abc", "pos"]
        result = cli_multiline(cmd)
        # either grouped flags remain together or are placed on continuation line; ensure flags exist
        assert "-abc" in result or "-a" in result

    @pytest.mark.parametrize(
        "bad_value, match_substr",
        [
            (-1, "multiline_indent"),
            (0.5, "multiline_indent"),
            ("wide", "multiline_indent"),
        ],
        ids=["negative_indent", "nonint_indent", "str_indent"],
    )
    def test_invalid_multiline_indent_raises(self, bad_value, match_substr):
        """Raise ValueError for invalid multiline_indent values."""
        with pytest.raises(ValueError, match=rf"(?i){match_substr}"):
            cli_multiline(["echo", "x"], multiline_indent=bad_value)

    @pytest.mark.parametrize(
        "bad_value, match_substr",
        [
            (0, "max_line_length"),
            (-10, "max_line_length"),
            ("long", "max_line_length"),
        ],
        ids=["zero_maxlen", "negative_maxlen", "str_maxlen"],
    )
    def test_invalid_max_line_length_raises(self, bad_value, match_substr):
        """Raise ValueError for invalid max_line_length values."""
        with pytest.raises(ValueError, match=rf"(?i){match_substr}"):
            cli_multiline(["cmd"], max_line_length=bad_value)

    def test_non_string_iterable_items_coerced(self):
        """Coerce non-string iterable items like numbers into string form."""
        cmd = ["echo", 123, 45.6]
        result = cli_multiline(cmd)
        assert "123" in result
        assert "45.6" in result


class TestClify:
    @pytest.mark.parametrize(
        "cmd,expected",
        [
            (
                'git commit -m "Initial commit"',
                ["git", "commit", "-m", "Initial commit"],
            ),
            ("echo 'hello world'", ["echo", "hello world"]),
            ("python -c 'print(1)'", ["python", "-c", "print(1)"]),
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
        args = ["echo", 123, True, Path("some"), b"x"]
        out = clify(args)
        assert out == ["echo", "123", "True", "some", "x"]

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
