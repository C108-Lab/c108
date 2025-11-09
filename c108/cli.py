"""
CLI command formatting and normalization tools.
"""

# Standard library -----------------------------------------------------------------------------------------------------

import os, re, shlex
import collections.abc as abc
from typing import Any, Iterable

# Local ----------------------------------------------------------------------------------------------------------------
from .tools import listify
from .formatters import fmt_any


# Methods --------------------------------------------------------------------------------------------------------------


def cli_multiline(
    command: str | int | float | Iterable[Any] | None,
    *,
    shlex_split: bool = True,
    multiline_indent: int = 8,
    max_line_length: int = 120,
) -> str:
    """Format a command as a readable multi-line POSIX shell string with line continuations.

    This function takes a command (in the same formats as clify()) and formats it
    as a multi-line string suitable for POSIX-compatible shells (bash, zsh, sh, etc.).
    Long options (--option) and flags (-f) start new lines for better readability.

    **Platform Support:**
    - ✓ Linux, macOS, Unix (bash, zsh, sh, fish, etc.)
    - ✓ Windows with WSL, Git Bash, MSYS2, Cygwin
    - ✗ Windows Command Prompt (cmd.exe) - uses `^` for continuation
    - ✗ Windows PowerShell - uses backtick (`) for continuation

    **Output Format:**
    Uses POSIX shell line continuation syntax with backslash (\\) at end of lines.
    The resulting string can be copied directly into POSIX shell scripts or terminals.

    Rules:
    - First argument stays on the first line
    - Long options (--option, --option=value) start new lines
    - Short flags (-f, -abc) start new lines
    - Flag values stay on the same line as their flag
    - Positional arguments after options start new lines
    - Line continuations use backslash (\\)
    - Each continued line is indented by multiline_indent spaces

    Args:
        command: Same input types as clify() - string, int, float, iterable, or None.
        shlex_split: Whether to shell-split string input (same as clify).
        multiline_indent: Number of spaces to indent continuation lines, int >=0 required.
        max_line_length: Hint for when to break lines (not strictly enforced), int > 0 required.

    Returns:
        str: Multi-line formatted POSIX shell command string, or empty string if no command.

    Raises:
         ValueError: If multiline_indent or max_line_length is invalid.

    Examples:
        >>> cmd = cli_multiline(['tar', '-cvpzf', 'backup.tar.gz', '--exclude=/proc', '--exclude=/sys'])
        >>> # 'tar -cvpzf backup.tar.gz \\
        >>> #         --exclude=/proc \\
        >>> #         --exclude=/sys'

        >>> cmd = cli_multiline('git commit -m "Initial commit" --author="John Doe"')
        >>> # 'git commit -m "Initial commit" \\
        >>> #         --author="John Doe"'

        >>> # The output can be used directly in bash scripts:
        >>> cmd = cli_multiline(['docker', 'run', '--rm', '-v', '/data:/data', 'ubuntu:latest'])
        >>> # docker run --rm \\
        >>> #         -v /data:/data \\
        >>> #         ubuntu:latest

    Note:
        For Windows cmd.exe or PowerShell compatibility, consider using clify()
        to get the argument list and format it according to the target shell's
        line continuation syntax.
    """

    if not isinstance(multiline_indent, int) or multiline_indent < 0:
        raise ValueError(
            f"multiline_indent must be non-negative int >= 0, but found {fmt_any(multiline_indent)}"
        )
    if not isinstance(max_line_length, int) or max_line_length < 1:
        raise ValueError(f"max_line_length must be int >= 1, but found {fmt_any(max_line_length)}")

    # Use clify to normalize the input
    args = clify(command, shlex_split=shlex_split)

    if not args:
        return ""

    if len(args) == 1:
        return args[0]

    # Format as multi-line
    indent = " " * multiline_indent
    lines = [args[0]]  # First argument always on first line
    seen_option = False  # Track if we've seen any options/flags

    i = 1
    while i < len(args):
        arg = args[i]

        # Check if this is an option/flag (but not negative numbers like -123, -1.5)
        is_option = (
            arg.startswith("--")  # Long options
            or (arg.startswith("-") and len(arg) > 1 and not re.match(r"^-\d*\.?\d+$", arg))
        )

        if is_option:
            seen_option = True
            # Start new line for this option
            current_line = arg

            # Check if next arg is a value for this option (not another flag/option)
            if (
                i + 1 < len(args)
                and not args[i + 1].startswith("--")  # Not a long option
                and not (
                    args[i + 1].startswith("-")
                    and len(args[i + 1]) > 1
                    and not re.match(r"^-\d*\.?\d+$", args[i + 1])
                )  # Not a short flag (but allow negative numbers)
                and "=" not in arg
            ):  # Only if current arg doesn't have embedded value
                i += 1
                current_line += f" {args[i]}"

            lines.append(current_line)
        else:
            # This is a positional argument
            if seen_option:
                # If we've seen options before, put positional args on new lines
                lines.append(arg)
            else:
                # No options seen yet, add to current line
                lines[-1] += f" {arg}"

        i += 1

    # Join with line continuations
    if len(lines) == 1:
        return lines[0]

    result = lines[0]
    for line in lines[1:]:
        result += f" \\\n{indent}{line}"

    return result


def clify(
    command: str | int | float | Iterable[Any] | None,
    shlex_split: bool = True,
    *,
    max_items: int = 256,
    max_arg_length: int = 4096,
) -> list[str]:
    """
    Normalize a command into a subprocess-ready argv list with sanity checks.

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
        raise ValueError(f"max_items must be a positive integer: {fmt_any(max_items)}")
    if not isinstance(max_arg_length, int) or max_arg_length <= 0:
        raise ValueError(f"max_arg_length must be a positive integer: {fmt_any(max_arg_length)}")

    def ensure_len(arg: str) -> str:
        if len(arg) > max_arg_length:
            raise ValueError(f"argument exceeds max_arg_length={max_arg_length}: {fmt_any(arg)}")
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

    raise TypeError(
        f"command must be a string, int, float, an iterable of arguments, or None, "
        f"but found {fmt_any(command)}"
    )
