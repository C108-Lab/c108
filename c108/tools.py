#
# C108 Tools & Utilites
#

# Standard library -----------------------------------------------------------------------------------------------------
import inspect


# Methods --------------------------------------------------------------------------------------------------------------

def ascii_string(s: str) -> str:
    return ''.join([i if ord(i) < 128 else '_' for i in s])


def method_name():
    return inspect.stack()[1][3]


def print_method(prefix: str = "------- ",
                 suffix: str = " -------",
                 start: str = "\n\n",
                 end: str = "\n"):
    method_name = inspect.stack()[1][3]
    print_title(title=method_name, prefix=prefix, suffix=suffix, start=start, end=end)


def print_title(title,
                prefix: str = "------- ",
                suffix: str = " -------",
                start: str = "\n",
                end: str = "\n"):
    print(f"{start}{prefix}{title}{suffix}{end}")
