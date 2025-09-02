#
# C108 Packaging Tools
#

# Standard library -----------------------------------------------------------------------------------------------------
import os, sys
from importlib.metadata import version as metadata_version
from types import ModuleType
from typing import Any

from packaging.version import InvalidVersion, Version


# Methods --------------------------------------------------------------------------------------------------------------

def py_basename(file_name: str = None) -> str:
    file_name = sys.argv[0] if file_name is None else file_name
    py_name = os.path.basename(file_name)
    py_name = py_name.removesuffix('.py')
    return py_name


def py_package_version(package: ModuleType | str) -> str:
    if package is None:
        raise ValueError("Type mismatch for package: <str> or module type <ModuleType> required")
    module_version = getattr(package, '__version__', None)
    if module_version:
        return module_version
    package_name = getattr(package, '__name__', str(package))
    try:
        py_package_version = metadata_version(package_name)
    except:
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        if isinstance(package, ModuleType) and package.__name__ in sys.builtin_module_names:
            return python_version
        elif isinstance(package, str) and str(package) in sys.builtin_module_names:
            # Handling if 'package' is str and is part of built-in module names
            return python_version
        raise ValueError(f"py_package_version: {package} does not have a version defined")
    return py_package_version


def is_numbered_version(version: Any,
                        min_depth: int = 1,
                        max_depth: int = 2,
                        raise_exception: bool = False) -> bool:
    """
    Checks if an object represents a version with number-convertable values on each version level

    Example: version 1234a.5b.6c is convertable to 1234.5.6, so this function returns True.

    Args:
        version  : The string to check.
        min_depth: The minimum allowed depth of the version (number of dots). Ex: 2024 requires `min_depth=0`
        max_depth: The maximum allowed depth of the version. Defaults to 2.

    Returns:
        bool: True if the string is a valid semantic version with the
              specified depth, False otherwise.
    """

    def _is_numbered_item(src: int | str) -> bool:
        if isinstance(src, int):
            return True
        elif isinstance(src, str) and src and src[0].isdigit():
            return True
        else:
            return False

    version = str(version)

    # Check all items are convertable to number
    as_list = version.split('.')
    if not all(_is_numbered_item(item) for item in as_list):
        if raise_exception:
            raise ValueError(f"Invalid Numbered version '{version}'. Version items should start with digit(s)")
        return False

    # Check version depth within limits
    num_dots = version.count(".")
    if min_depth <= num_dots <= max_depth:
        return True
    else:
        if raise_exception:
            raise InvalidVersion(f"Version '{version}' depth is out of range [{min_depth}, {max_depth}]")
        return False


def is_pep440_version(version: Any,
                      min_depth: int = 0,
                      max_depth: int = None,
                      raise_exception: bool = False) -> bool:
    """
    Checks if an object represents a valid PEP440 version number

    Args:
        version  : The string to check.
        min_depth: The minimum allowed depth of the version (number of dots). Ex: 2024 requires `min_depth=0`
        max_depth: The maximum allowed depth of the version. Defaults to 2.

    Returns:
        bool: True if the string is a valid semantic version with the
              specified depth, False otherwise.
    """
    version = str(version)
    max_depth = max_depth if isinstance(max_depth, int) else 108
    try:
        normalized_version = str(Version(version))
        if normalized_version != version:
            if raise_exception:
                raise InvalidVersion(f"Version '{version}' is not PEP440 compliant. "
                                     f"Normalized version can be {normalized_version}")
            return False
    except InvalidVersion:
        if raise_exception:
            raise InvalidVersion(f"Version '{version}' is not PEP440 compliant")
        return False

    # Check version depth within limits
    num_dots = version.count(".")
    if min_depth <= num_dots <= max_depth:
        return True
    else:
        if raise_exception:
            raise InvalidVersion(f"Version '{version}' depth is out of range [{min_depth}, {max_depth}]")
        return False


def is_semantic_version(version: Any,
                        min_depth: int = 1,
                        max_depth: int = 2,
                        allow_meta: bool = False,
                        raise_exception: bool = False):
    """
    Checks if a string represents a semantic version with specified depth.

    Note that PEP440 versions with post-release specifiers like '4.3.2.post2' are NOT supported

    Args:
        version  : The string to check.
        min_depth: The minimum allowed depth of the version (number of dots). Ex: 2024 requires `min_depth=0`
        max_depth: The maximum allowed depth of the version.
                                  Defaults to 2. Ex: 2024.0.0

    Returns:
        bool: True if the string is a valid semantic version with the
              specified depth, False otherwise.
    """
    version = str(version)

    # If metadata not allowed, version must be only digits and dots
    if not allow_meta and not version.replace(".", "").isdigit():
        if raise_exception:
            raise ValueError(f"Version '{version}' cannot contain meta characters but only digits. ")
        return False

    # If metadata allowed, only check the parts before "-"
    if allow_meta and '-' in version:
        version = version.split("-")[0]

    # Check version parts per min_depth and max_depth
    num_dots = version.count(".")
    if min_depth <= num_dots <= max_depth:
        return True
    else:
        if raise_exception:
            raise ValueError(f"Version '{version}' depth is out of range [{min_depth}, {max_depth}]")
        return False
