"""
C108 CLI Tools
"""

# Standard library -----------------------------------------------------------------------------------------------------

import os, shlex
import collections.abc as abc
from typing import Any, Iterable

# Local ----------------------------------------------------------------------------------------------------------------
from .tools import listify, fmt_any


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


def clify(
        command: str | int | float | Iterable[Any] | None,
        shlex_split: bool = True,
        *,
        max_items: int = 256,
        max_arg_length: int = 4096,
) -> list[str]:
    """Normalize a command into a subprocess-ready argv list with sanity checks.

    This function composes a command—provided as a shell-like string or an iterable
    of arguments—into a list[str] suitable for subprocess APIs (e.g., subprocess.run).

    Rules:
    - None → [].
    - String input:
      - shlex_split=True (default): shell-parse using shlex.split; quotes/escapes respected.
      - shlex_split=False: treat the entire string as a single argument.
      - Empty string → [].
    - Int/float input: converted to string as a single argument.
    - Iterable input: each item is converted to text for argv:
      - Path-like objects via os.fspath.
      - Everything else via str.
      - The iterable is not recursively flattened; nested iterables are stringified.

    Sanity checks:
    - max_items: maximum number of arguments allowed.
    - max_arg_length: maximum length (characters) for any single argument.
    - Violations raise ValueError describing the problem.

    Args:
        command: Shell string, int, float, or an iterable of arguments (e.g., list/tuple/generator), or None.
        shlex_split: Whether to shell-split string input. Ignored for non-strings.
        max_items: Upper bound on argv length.
        max_arg_length: Upper bound on each argument length (len in characters).

    Returns:
        list[str]: The argv vector.

    Raises:
        TypeError: If command is of an unsupported type.
        ValueError: If max_items/max_arg_length are invalid, or limits are exceeded.

    Examples:
        >>> clify('git commit -m "Initial commit"')
        ['git', 'commit', '-m', 'Initial commit']

        >>> clify("python -c 'print(1)'", shlex_split=False)
        ["python -c 'print(1)'"]

        >>> clify(['echo', 123, True])
        ['echo', '123', 'True']

        >>> from pathlib import Path
        >>> clify(['ls', Path('/tmp')])
        ['ls', '/tmp']

        >>> clify(42)
        ['42']

        >>> clify(3.14)
        ['3.14']

        >>> clify(None)
        []
    """
    if not isinstance(max_items, int) or max_items <= 0:
        raise ValueError("max_items must be a positive integer")
    if not isinstance(max_arg_length, int) or max_arg_length <= 0:
        raise ValueError("max_arg_length must be a positive integer")

    def ensure_len(arg: str) -> str:
        if len(arg) > max_arg_length:
            raise ValueError(f"argument exceeds max_arg_length {max_arg_length}: {fmt_any(arg)}")
        return arg

    def to_text(x: Any) -> str:
        # Path-like support
        try:
            p = os.fspath(x)  # str or bytes for path-like; raises TypeError otherwise
            s = p if isinstance(p, str) else os.fsdecode(p)
        except TypeError:
            # Everything else via str
            s = str(x)
        return ensure_len(s)

    if command is None:
        return []

    if isinstance(command, str):
        if command == "":
            return []
        if shlex_split:
            parts = [ensure_len(p) for p in shlex.split(command)]
            if len(parts) > max_items:
                raise ValueError(f"too many arguments: {len(parts)} > max_items={max_items}")
            return parts
        else:
            # Single-argument string
            return [ensure_len(command)]

    if isinstance(command, (int, float)):
        return [ensure_len(str(command))]

    if isinstance(command, abc.Iterable):
        argv: list[str] = []
        for idx, item in enumerate(command, start=1):
            if idx > max_items:
                raise ValueError(f"too many arguments: {idx} > max_items={max_items}")
            argv.append(to_text(item))
        return argv

    raise TypeError(f"command must be a string, int, float, an iterable of arguments, or None, "
                    f"but found {fmt_any(command)}")
