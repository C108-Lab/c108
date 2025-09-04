"""
C108 Utilities shared across the package.

Contains fundamental functions used by multiple modules to avoid
circular imports and provide a stable foundation.
"""

# Standard library -----------------------------------------------------------------------------------------------------
from typing import Any


# Methods --------------------------------------------------------------------------------------------------------------

def class_name(obj: Any, fully_qualified=True, fully_qualified_builtins=False,
               start: str = "", end: str = "") -> str:
    """Get the class name from the object. Optionally get the fully qualified class name

    Parameters:
        obj (Any): An object or a class
        fully_qualified (bool): If true, returns the fully qualified name for user objects or classes
        fully_qualified_builtins (bool): If true, returns the fully qualified name for builtin objects or classes
        start (str): Optional prefix string to add at start of name
        end (str): Optional suffix string to add at end of name

    Returns:
        str: The class name
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
    if cls.__module__ == 'builtins':
        if fully_qualified_builtins:
            # Return the fully qualified name
            return start + cls.__module__ + "." + cls.__name__ + end
        else:
            # Return only the class name
            return start + cls.__name__ + end
    else:
        if fully_qualified:
            # Return the fully qualified name
            return start + cls.__module__ + "." + cls.__name__ + end
        else:
            # Return only the class name
            return start + cls.__name__ + end
