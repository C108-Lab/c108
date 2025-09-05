"""
C108 CLI Tools
"""

# Standard library -----------------------------------------------------------------------------------------------------
import shlex
from typing import Any, Iterable, List, Sequence

# Local ----------------------------------------------------------------------------------------------------------------
from .tools import listify, fmt_value


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


def clify(command: str | Sequence[Any]) -> List[str]:
    """Composes a command for execution with `subprocess`.

    This function normalizes a command, provided as either a shell-like string
    or a sequence of arguments, into a list of strings suitable for functions
    like `subprocess.run()`.

    - If the command is a string, it is safely split into arguments using
      `shlex.split()` to correctly handle quotes and escaped characters.
    - If the command is a sequence (e.g., list, tuple), each element is
      converted to a string.
    - If the command is `None` or an empty container, an empty list is returned.

    Args:
        command: The command to process, either as a single string or a
            sequence of arguments.

    Returns:
        A list of strings representing the command and its arguments.

    Raises:
        TypeError: If the input `command` is not a string or a sequence.

    Examples:
        >>> prepare_command('git commit -m "Initial commit"')
        ['git', 'commit', '-m', 'Initial commit']

        >>> prepare_command(['ls', '-l', '/home/user'])
        ['ls', '-l', '/home/user']

        >>> prepare_command(('echo', 123, True))
        ['echo', '123', 'True']

        >>> prepare_command("")
        []
    """
    if not command:
        return []

    if isinstance(command, str):
        return shlex.split(command)

    # Process non-str sequences (list, tuple, etc.).
    if isinstance(command, Sequence):
        return [str(arg) for arg in command]

    raise TypeError(
        f"Command must be a string or a sequence of arguments:{fmt_value(command)}"
    )
