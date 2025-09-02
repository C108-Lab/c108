#
# C108 ArgsCLI Tools
#

# Standard library -----------------------------------------------------------------------------------------------------
import shlex
from typing import Any, Iterable

# Local ----------------------------------------------------------------------------------------------------------------
from .abc import listify


# Methods --------------------------------------------------------------------------------------------------------------

def cli_multiline(args: str | Iterable[str], multiline_indent: int = 8) -> str:
    """
    Return args list as a multi-line string in shell CLI format
    """
    if not args:
        return ""
    args = listify(args)
    args = [str(a) for a in args]

    # --option | --option=value
    is_long_option = [(len(opt) > 1) and opt[:2] == "--" for opt in args]

    # -h | -xyz
    is_flag_name = [(len(opt) > 1) and opt[:2] != "--" and opt[:1] == "-" for opt in args]

    # from -h <value> but without the '-h'
    is_flag_value = [False] * len(args)
    for i in range(1, len(args)):
        is_flag_value[i] = (is_flag_name[i - 1]
                            and not is_flag_name[i]
                            and not is_long_option[i])

    spaces = " " * multiline_indent  # Whitespaces before option name
    args_multiline = ""
    for i in range(len(args)):
        args_multiline += (("" if i == 0 else " ") +
                           (f"\\\n{spaces}{args[i]}" if not is_flag_value[i] else f"{args[i]}"))
    return args_multiline


def clify(command: str | list | tuple | Any, shlex_split: bool = False) -> list[str]:
    """
    Return a list of command-line command with arguments for Subprocess run

    Args:
        command: Command to run
        shlex_split: Whether to split topmost <str>-command. Has no effect on iterable command items
    """
    cli_command = [] if not command \
        else shlex.split(command) if isinstance(command, str) and shlex_split \
        else listify(command, as_type=str)
    return cli_command
