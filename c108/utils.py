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

    This function safely handles edge cases including objects without standard
    attributes (__class__, __name__, __module__) by falling back to str()
    representation. Special handling for typing module constructs returns
    readable representations like 'List[int]' or 'Union[int, str]'.

    Parameters:
        obj (Any): An object or a class.
        fully_qualified (bool): If true, returns the fully qualified name for user objects or classes.
        fully_qualified_builtins (bool): If true, returns the fully qualified name for builtin objects or classes.

    Returns:
        str: The class name, or a string representation if standard attributes are unavailable.

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

        Typing constructs with readable representations:
            >>> from typing import List, Union
            >>> class_name(List[int])
            'List[int]'
            >>> class_name(Union[int, str])
            'Union[int, str]'
    """
    # Check if the obj is an instance or a class
    obj_is_class = isinstance(obj, type)

    # Get the class
    if not obj_is_class:
        try:
            cls = obj.__class__
        except AttributeError:
            # Fallback for objects without __class__
            return str(obj)
    else:
        cls = obj

    # Get the class name safely
    try:
        name = cls.__name__
    except AttributeError:
        # Fallback for types without __name__
        return str(cls)

    # Get the module safely
    try:
        module = cls.__module__
    except AttributeError:
        # No module info available, just return the name
        return name

    # Special handling for typing module constructs
    if module == "typing":
        # These are typing constructs with internal names like _GenericAlias
        str_repr = str(obj)
        # Strip all 'typing.' prefixes for cleaner output
        str_repr = str_repr.replace("typing.", "")
        return str_repr

    # If class is builtin
    if module == "builtins":
        if fully_qualified_builtins:
            return f"{module}.{name}"
        else:
            return name
    else:
        if fully_qualified:
            return f"{module}.{name}"
        else:
            return name
