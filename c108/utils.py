"""
C108 Utilities shared across the package.

Contains functions used by multiple modules to avoid circular imports.
"""

# Standard library -----------------------------------------------------------------------------------------------------
from typing import Any


# Methods --------------------------------------------------------------------------------------------------------------


def class_name(
    obj: Any,
    fully_qualified: bool = False,
    fully_qualified_builtins: bool = False,
) -> str:
    """
    Get the class name of an object or a class.

    Returns class name whether given an instance or the class itself.
    For example, both `class_name(10)` and `class_name(int)` return 'int'.

    Parameters:
        obj (Any): An object or a class.
        fully_qualified (bool): If true, returns the fully qualified name for user objects or classes.
        fully_qualified_builtins (bool): If true, returns the fully qualified name for builtin objects or classes.
        start (str): Optional prefix string to add at start of name.
        end (str): Optional suffix string to add at end of name.

    Returns:
        str: The class name.

    Examples:
        Basic usage with a builtin instance:
            >>> class_name(10)
            'int'

        Fully qualified name for a builtin (when enabled):
            >>> class_name(10, fully_qualified_builtins=True)
            'builtins.int'

        User-defined class: instance and class object:
            >>> class C: ...
            >>> class_name(C())
            'C'
            >>> class_name(C, fully_qualified=True)
            'c108.utils.C'
    """

    # Check if the obj is an instance or a class
    obj_is_class = isinstance(obj, type)

    # If obj is an instance
    if not obj_is_class:
        # Get the class of the instance
        cls = obj.__class__
    else:
        cls = obj

    # If class is builtin
    if cls.__module__ == "builtins":
        if fully_qualified_builtins:
            # Return the fully qualified name
            return cls.__module__ + "." + cls.__name__
        else:
            # Return only the class name
            return cls.__name__
    else:
        if fully_qualified:
            # Return the fully qualified name
            return cls.__module__ + "." + cls.__name__
        else:
            # Return only the class name
            return cls.__name__
