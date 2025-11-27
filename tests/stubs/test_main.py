#
# Tests for python -m c108.stubs CLI functionality
# uses in-process for coverage
#

import sys
import runpy
import pytest

PYTHON = sys.executable
MODULE = "c108.stubs"


class TestCliHelp:
    @pytest.mark.parametrize(
        "args, expect_returncode, expect_stdout_substr, expect_stderr_substr",
        [
            pytest.param(
                ["-m", MODULE, "--help"],
                0,
                "Generate stubs for c108 decorators",
                "",
                id="help-flag",
            ),
            pytest.param(
                ["-m", MODULE], 0, "Available commands:", "", id="no-args-shows-available-commands"
            ),
            pytest.param(
                ["-m", MODULE, "merge", "--sentinel", "INVALID", "somefile.py"],
                2,
                "",
                "invalid choice",
                id="merge-invalid-sentinel-fails",
            ),
        ],
    )
    def test_cli_outputs(
        self, args, expect_returncode, expect_stdout_substr, expect_stderr_substr, capsys
    ):
        """
        Run the stubs module with given arguments and assert basic behavior.

        This runs the module in-process (via runpy.run_module with run_name="__main__")
        and adjusts sys.argv to emulate running `python -m <module> ...`. Doing it
        in-process lets pytest-cov collect coverage for c108/stubs/__main__.py.
        """
        # args come in the form ["-m", "c108.stubs", ...]
        if len(args) < 2 or args[0] != "-m":
            raise ValueError("test expects args like ['-m', MODULE, ...']")

        module_name = args[1]
        new_argv = [module_name] + args[2:]

        original_argv = sys.argv[:]
        try:
            sys.argv = new_argv
            returncode = 0
            try:
                runpy.run_module(module_name, run_name="__main__")
            except SystemExit as se:
                # SystemExit.code can be None or an int
                returncode = 0 if se.code is None else int(se.code)

            captured = capsys.readouterr()

            assert returncode == expect_returncode

            if expect_stdout_substr:
                assert expect_stdout_substr in captured.out

            if expect_stderr_substr:
                assert expect_stderr_substr.lower() in captured.err.lower()
        finally:
            sys.argv = original_argv
