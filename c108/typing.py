"""
Runtime type validation utilities.
"""

# Standard library -----------------------------------------------------------------------------------------------------
import functools
import inspect
import sys
import typing

from dataclasses import is_dataclass, fields as dc_fields
from types import UnionType
from typing import Any, Callable, Literal, TypeVar, Union
from typing import get_type_hints, get_origin, get_args

# Local imports ------------------------------------------------------------------------------------
from c108.abc import search_attrs
from c108.formatters import fmt_type

# Public API -----------------------------------------------------------------------------------------------------------
__all__ = ["valid_types", "validate_attr_types", "validate_param_types"]

# Classes --------------------------------------------------------------------------------------------------------------

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")


# Python 3.10+ has types.UnionType for X | Y syntax
if sys.version_info >= (3, 10):
    import types

    UNION_TYPES = (typing.Union, types.UnionType)
else:
    UNION_TYPES = (typing.Union,)


def valid_types(func=None, *, skip: typing.Iterable[str] = None, only: typing.Iterable[str] = None):
    """
    Decorator to validate function arguments against type hints at runtime.

    Creates a wrapper around the function that checks argument types before each call.
    By default, validates all annotated parameters. Use `skip` or `only` to control
    which parameters are checked.

    Args:
        func: The function to wrap (automatically provided when used as @valid_types).
        skip: Parameter names to exclude from validation. Cannot be used with `only`.
        only: Parameter names to validate (all others ignored). Cannot be used with `skip`.

    Returns:
        A wrapper function that performs type validation before calling the original function.

    Raises:
        TypeError: If a validated argument doesn't match its type hint at call time.
        ValueError: If both `skip` and `only` are provided (mutually exclusive),
                    or if invalid parameter names are specified.

    Examples:
        Validate all parameters:

            @valid_types
            def connect(host: str, port: int) -> None:
                pass

        Validate only specific parameters (gradual adoption):

            @valid_types(only=("host",))
            def connect(host: str, port: int, options: dict) -> None:
                pass

        Skip validation for specific parameters:

            @valid_types(skip=("payload",))
            def send(id: int, payload: dict) -> None:
                pass

    Notes:
        - This is a function decorator that wraps the original function
        - Validation happens once per function call with minimal overhead
        - Only validates parameters with type hints; unannotated params are ignored
        - ``typing.Any`` hints are never validated
        - Generic types (e.g., ``List[int]``) validate the container type only, not contents
        - Works with both positional and keyword arguments
        - ``Union`` and ``Optional`` types validate against all union members
        - Supports both ``Union[X, Y]`` and ``X | Y`` syntax (Python 3.10+)
    """

    if func is None:
        return functools.partial(valid_types, skip=skip, only=only)

    # 0. VALIDATION: Mutual Exclusivity
    if skip is not None and only is not None:
        raise ValueError(
            f"@valid_types: Cannot use both 'skip' and 'only' on function '{func.__name__}'."
        )

    # 1. SETUP PHASE (runs once at decoration time)
    annotations = typing.get_type_hints(func)
    sig = inspect.signature(func)
    params = sig.parameters

    # Convert iterables to sets for O(1) lookups, handle None
    skip_set = set(skip) if skip else set()
    only_set = set(only) if only is not None else None

    # Validate that requested parameter names actually exist
    param_names = set(params.keys())
    if only_set is not None:
        invalid = only_set - param_names
        if invalid:
            raise ValueError(
                f"@valid_types on '{func.__name__}': 'only' contains invalid parameter names: {invalid}"
            )

    if skip_set:
        invalid = skip_set - param_names
        if invalid:
            raise ValueError(
                f"@valid_types on '{func.__name__}': 'skip' contains invalid parameter names: {invalid}"
            )

    # Build list of (index, name, type) tuples for runtime checking
    arg_checks = []

    for i, (name, param) in enumerate(params.items()):
        # Always skip 'return' annotation
        if name == "return":
            continue

        # LOGIC: Determine if we should check this argument
        should_check = False

        if only_set is not None:
            # Mode: ONLY (allow-list)
            # We only check if it is explicitly in the set
            if name in only_set:
                should_check = True
        else:
            # Mode: SKIP (block-list) or DEFAULT
            # We check if it is NOT in the skip set
            if name not in skip_set:
                should_check = True

        # Final gate: Does it actually have a type hint?
        if should_check and name in annotations:
            hint = annotations[name]

            # Type Normalization: Handle generic types and Union
            origin = typing.get_origin(hint)

            # Check if this is a Union type (typing.Union or types.UnionType from X | Y)
            if origin in UNION_TYPES:
                # Union[str, None] or Optional[str] or str | None -> check against tuple of types
                check_type = typing.get_args(hint)
            elif origin is not None:
                # List[int] -> list, Dict[str, int] -> dict, etc.
                check_type = origin
            else:
                # Plain type like str, int, etc.
                check_type = hint

            # Skip typing.Any (no runtime checking possible)
            if check_type is typing.Any:
                continue

            arg_checks.append((i, name, check_type))

    # 2. RUNTIME WRAPPER (fast path - just a simple loop)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for i, name, type_ in arg_checks:
            # Check positional argument
            if i < len(args):
                value = args[i]
                if not isinstance(value, type_):
                    # Format type name(s) nicely for error message
                    if isinstance(type_, tuple):
                        expected = " | ".join(
                            t.__name__ if hasattr(t, "__name__") else str(t) for t in type_
                        )
                    else:
                        expected = type_.__name__ if hasattr(type_, "__name__") else str(type_)

                    raise TypeError(
                        f"argument '{name}' expected {expected}, got {type(value).__name__}"
                    )
            # Check keyword argument
            elif name in kwargs:
                value = kwargs[name]
                if not isinstance(value, type_):
                    # Format type name(s) nicely for error message
                    if isinstance(type_, tuple):
                        expected = " | ".join(
                            t.__name__ if hasattr(t, "__name__") else str(t) for t in type_
                        )
                    else:
                        expected = type_.__name__ if hasattr(type_, "__name__") else str(type_)

                    raise TypeError(
                        f"argument '{name}' expected {expected}, got {type(value).__name__}"
                    )

        return func(*args, **kwargs)

    return wrapper


def validate_attr_types(
    obj: Any,
    *,
    attrs: list[str] | None = None,
    include_inherited: bool = True,
    include_private: bool = False,
    pattern: str | None = None,
    strict_unions: bool = True,
    strict_none: bool = True,
    strict_missing: bool = True,
) -> None:
    """
    Validate that object attributes match their type annotations.

    Supports dataclasses, attrs classes, and regular Python classes with
    type annotations. Performance-optimized with automatic fast path for dataclasses.

    This function validates the types of object attributes. For validating
    function parameters, see validate_param_types() (inline) or @valid_param_types
    (decorator).

    Args:
        obj: Object instance to validate
        attrs: Optional list of specific attribute names to validate.
               If None, validates all annotated attributes.
        include_inherited: If True, validates inherited attributes with type hints
        include_private: If True, validates private attributes (starting with '_')
        pattern: Optional regex pattern to filter which attributes to validate
        strict_unions: If True (default), raise TypeError when encountering Union types
                       that cannot be validated with isinstance() (e.g., list[int] | dict[str, int],
                       Callable[[int], str] | Callable[[str], int]). If False, silently skip
                       such unions. Simple unions like int | str | None are always validated
                       regardless of this flag.
        strict_none: If True (default), None values only pass validation when explicitly
                     allowed in the type hint via Optional or Union with None (strict
                     enforcement mode). If False, None values pass validation for ANY type
                     hint (lenient mode, useful for development/migration).
        strict_missing: If True (default), raise TypeError when an attribute has a type
                        annotation but is missing from the object (AttributeError when
                        accessing it). If False, silently skip missing attributes. This is
                        important for production validation to catch incomplete objects.

    Raises:
        TypeError: If attribute type doesn't match annotation, or if strict_unions=True
                   and a truly unvalidatable Union type is encountered
        ValueError: If obj has no type annotations
        RuntimeError: If Python version < 3.11

    🚀 Performance:
        Automatic optimization:
            - Fast path (~5-10µs): Used for dataclasses when attrs=None, pattern=None,
              and include_private=False
            - Slow path (~30-70µs): Used for all other cases (non-dataclasses, pattern
              matching, custom attr lists, private attrs)
            - You don't need to configure anything - the function automatically chooses
              the optimal path based on your parameters

        The fast path is 5-10x faster and recommended for high-throughput scenarios
        like validation in __post_init__ or API request handlers.

    Validation Modes:
        Strict (default - strict_none=True, strict_missing=True):
            >>> class Config:
            ...     timeout: int = None  # None fails - not in type hint
            >>> validate_attr_types(Config())  # doctest: +SKIP
            >>> # ❌ Raises TypeError

            >>> class Config:
            ...     timeout: int | None = None  # None passes - explicitly in hint
            >>> validate_attr_types(Config())  # ✅ Passes

            >>> class Config:
            ...     timeout: int  # Annotated but not set
            >>> validate_attr_types(Config())  # doctest: +SKIP
            >>> # ❌ Raises TypeError (missing attribute)

        Lenient (strict_none=False, strict_missing=False):
            >>> class Config:
            ...     timeout: int = None  # None passes despite int hint
            >>> validate_attr_types(Config(), strict_none=False)  # ✅ Passes

            >>> class Config:
            ...     timeout: int  # Missing attribute ignored
            >>> validate_attr_types(Config(), strict_missing=False)  # ✅ Passes

    See Also:
        validate_param_types(): Validate function parameter types (inline call)
        valid_param_types: Decorator for automatic parameter type validation

    Examples:
        >>> from dataclasses import dataclass
        >>>
        >>> @dataclass
        ... class ImageData:
        ...     id: str = "qwkfjkqfjhkgwdjhg349893874"
        ...     width: int = 1080
        ...     height: int = 1080
        ...
        ...     def __post_init__(self):
        ...         validate_attr_types(self)  # Auto-uses fast path
        >>>
        >>> obj = ImageData()
        >>>
        >>> # Validate with default settings (strict mode)
        >>> validate_attr_types(obj)
        >>>
        >>> # Lenient None checking
        >>> validate_attr_types(obj, strict_none=False)
        >>>
        >>> # Allow missing attributes
        >>> validate_attr_types(obj, strict_missing=False)
        >>>
        >>> # Use pattern matching (automatically uses slow path)
        >>> validate_attr_types(obj, pattern=r"^api_.*")
        >>>
        >>> # Validate after mutations
        >>> obj.width = "invalid"
        >>> validate_attr_types(obj)  # Raises TypeError
        Traceback (most recent call last):
        ...
        TypeError: type validation failed in <ImageData>:
          Attribute 'width' must be <int>, got <str>

        >>> # For function parameters, use validate_param_types() or @valid_param_types
        >>> def process(x: int, y: str):
        ...     validate_param_types()  # Inline validation
        ...     # ... or use @valid_param_types decorator
    """
    # Determine if we can use fast path for a dataclass
    is_dc = is_dataclass(obj)
    can_use_fast = is_dc and attrs is None and pattern is None and not include_private

    # Automatically choose the optimal path
    if can_use_fast:
        _validate_attr_dataclass_fast(
            obj,
            strict_unions=strict_unions,
            strict_none=strict_none,
            strict_missing=strict_missing,
        )
    else:
        _validate_attr_with_search(
            obj,
            attrs=attrs,
            include_inherited=include_inherited,
            include_private=include_private,
            pattern=pattern,
            strict_unions=strict_unions,
            strict_none=strict_none,
            strict_missing=strict_missing,
        )


def _validate_attr_dataclass_fast(
    obj: Any,
    *,
    strict_unions: bool,
    strict_none: bool,
    strict_missing: bool,
) -> None:
    """
    Fast path validation for dataclasses without filtering.

    Performance: ~5-10µs per validation

    Optimizations:
    - Uses cached dataclass fields() metadata (no dir() calls)
    - No regex compilation or pattern matching
    - No property detection or callable checks
    - Direct field access only
    - Minimal function calls

    Args:
        obj: Dataclass instance to validate
        strict_unions: If True, raise on unvalidatable unions; if False, skip them
        strict_none: If True, None must be in type hint; if False, None passes for any type
        strict_missing: If True, raise on missing attributes; if False, skip them
    """
    validation_errors = []

    # fields() returns cached metadata - very fast
    for field in dc_fields(obj):
        attr_name = field.name
        expected_type = field.type

        # Direct attribute access
        try:
            value = getattr(obj, attr_name)
        except AttributeError:
            if strict_missing:
                validation_errors.append(
                    f"Attribute '{attr_name}' is annotated but missing from object"
                )
            continue

        # Validate type (None values handled inside based on strict_none)
        error = _validate_attr_obj_type(
            name=attr_name,
            name_prefix="attribute",
            value=value,
            expected_type=expected_type,
            strict_none=strict_none,
            strict_unions=strict_unions,
        )

        if error:
            validation_errors.append(error)

    if validation_errors:
        raise TypeError(
            f"type validation failed in {fmt_type(obj)}:\n  " + "\n  ".join(validation_errors)
        )


def _validate_attr_obj_type(
    name: str,
    name_prefix: Literal["attribute", "parameter"],
    value: Any,
    expected_type: Any,
    strict_none: bool,
    strict_unions: bool,
) -> str | None:
    """
    Validate a single attribute type. Returns error message or None.

    Extracted to avoid code duplication between fast/slow paths.
    Optimized for hot path performance.

    Args:
        name: Name of the attribute or parameter being validated
        name_prefix: Name prefix ('attribute' or 'parameter')
        value: The actual value to validate
        expected_type: The type annotation to validate against
        strict_none: If False, None passes for any type; if True, None must be in type hint
        strict_unions: If True, raise on unvalidatable unions; if False, skip them

    Returns:
        Error message string if validation fails, None if validation passes

    Validation logic:
        1. String annotations: handled based on strict_unions
        2. Union types: special handling for Optional and multi-type unions
        3. None values: controlled by strict_none parameter
        4. Generic types: validated by origin (list, dict, etc.)
        5. Simple types: direct isinstance check
    """
    # Handle string annotations (should be rare in modern Python)
    if isinstance(expected_type, str):
        if strict_unions:
            return f"{name_prefix.capitalize()} '{name}' has string annotation which cannot be validated"
        return None

    # Get origin for generic/union types
    origin = get_origin(expected_type)

    # Handle Union types (both T | None and Optional[T])
    # UnionType: modern syntax (int | None)
    # Union: old syntax from typing module (Optional[int] or Union[int, None])
    if origin is UnionType or origin is Union:
        args = get_args(expected_type)
        is_optional = type(None) in args
        non_none_types = tuple(t for t in args if t is not type(None))

        # Handle None values based on strict_none mode
        if value is None:
            if not strict_none:
                # Lenient mode: None passes for ANY type
                return None
            elif is_optional:
                # Strict mode: None passes only if explicitly in type hint
                return None
            else:
                # Strict mode: None not in type hint, so it fails
                type_names = " | ".join(fmt_type(t) for t in non_none_types)
                return (
                    f"{name_prefix.capitalize()} '{name}' must be {type_names}, "
                    f"got None (strict_none=True requires None to be in type hint)"
                )

        # For non-None values, validate against non-None types
        if is_optional:
            # Union includes None (e.g., int | None, int | str | None)
            try:
                if not isinstance(value, non_none_types):
                    # Format union members nicely
                    type_names = " | ".join(fmt_type(t) for t in non_none_types)
                    type_names += " | None"
                    return (
                        f"{name_prefix.capitalize()} '{name}' must be {type_names}, "
                        f"got {fmt_type(value)}"
                    )
                return None  # Validation passed
            except TypeError:
                # isinstance failed - truly complex union (e.g., list[int] | dict[str, int])
                if strict_unions:
                    return (
                        f"{name_prefix.capitalize()} '{name}' has complex Union type "
                        f"which cannot be validated with isinstance()"
                    )
                return None  # Skip in non-strict mode
        else:
            # Non-Optional Union (e.g., int | str | float)
            try:
                if not isinstance(value, non_none_types):
                    # Format union members nicely
                    type_names = " | ".join(fmt_type(t) for t in non_none_types)
                    return (
                        f"{name_prefix.capitalize()} '{name}' must be {type_names}, "
                        f"got {fmt_type(value)}"
                    )
                return None  # Validation passed
            except TypeError:
                # isinstance failed - truly complex union
                if strict_unions:
                    return (
                        f"{name_prefix.capitalize()} '{name}' has complex Union type "
                        f"which cannot be validated with isinstance()"
                    )
                return None  # Skip in non-strict mode

    # Handle None for non-union types
    if value is None:
        if not strict_none:
            # Lenient mode: None passes for any type
            return None
        else:
            # Strict mode: None fails for non-Optional types
            return (
                f"{name_prefix.capitalize()} '{name}' must be {fmt_type(expected_type)}, "
                f"got None (strict_none=True requires None to be explicitly in type hint)"
            )

    # Handle other generic types (list[T], dict[K,V])
    if origin is not None:
        expected_type = origin

    # Final isinstance check for simple types
    try:
        if not isinstance(value, expected_type):
            return (
                f"{name_prefix.capitalize()} '{name}' must be {fmt_type(expected_type)}, "
                f"got {fmt_type(value)}"
            )
    except TypeError:
        # isinstance failed for non-union type
        if strict_unions:
            return f"Cannot validate {name_prefix.lower()} '{name}' with complex type"
        return None  # Skip

    return None  # Valid


def _validate_attr_with_search(
    obj: Any,
    *,
    attrs: list[str] | None,
    include_inherited: bool,
    include_private: bool,
    pattern: str | None,
    strict_unions: bool,
    strict_none: bool,
    strict_missing: bool,
) -> None:
    """
    Slower path using search_attrs for complex filtering.

    Performance: ~30-70µs per validation

    Used when:
    - Not a dataclass
    - Custom attrs list provided
    - Pattern filtering needed
    - Private attribute inclusion needed

    Args:
        obj: Object instance to validate
        attrs: Optional list of specific attributes to validate
        include_inherited: If True, include inherited attributes
        include_private: If True, include private attributes
        pattern: Optional regex pattern for filtering
        strict_unions: If True, raise on unvalidatable unions; if False, skip them
        strict_none: If True, None must be in type hint; if False, None passes for any type
        strict_missing: If True, raise on missing attributes; if False, skip them
    """

    # Get type hints
    try:
        if is_dataclass(obj):
            type_hints = {f.name: f.type for f in dc_fields(obj)}
        else:
            # Try get_type_hints first (resolves forward refs)
            try:
                type_hints = get_type_hints(obj.__class__)
            except Exception:
                # Fallback to __annotations__ (doesn't resolve forward refs)
                type_hints = getattr(obj.__class__, "__annotations__", {}).copy()
    except Exception:
        type_hints = {}

    if not type_hints:
        raise ValueError(
            f"Cannot validate {fmt_type(obj)}: no type annotations found. "
            f"Add type hints to class attributes."
        )

    # Determine which attributes to validate
    if attrs is not None:
        # User explicitly specified which attributes to validate
        attrs_to_validate = attrs
    else:
        # Decide based on filtering needs
        needs_filtering = (pattern is not None) or (not include_inherited) or include_private

        if needs_filtering:
            # Use search_attrs to filter existing attributes
            existing_attrs = set(
                search_attrs(
                    obj,
                    format="list",
                    exclude_none=False,
                    include_inherited=include_inherited,
                    include_methods=False,
                    include_private=include_private,
                    include_properties=False,
                    pattern=pattern,
                    skip_errors=True,
                )
            )

            # Start with type-hinted attributes that pass the filter
            # For missing attributes: include them if they would pass the filter
            attrs_to_validate = []
            for name in type_hints:
                # Check if this attribute would pass the filter criteria
                passes_filter = True

                if not include_private and name.startswith("_"):
                    passes_filter = False

                if pattern is not None:
                    import re

                    if not re.match(pattern, name):
                        passes_filter = False

                # Include if: (passes filter AND exists) OR (passes filter AND strict_missing)
                if passes_filter:
                    if name in existing_attrs or strict_missing:
                        attrs_to_validate.append(name)
        else:
            # No filtering - validate all type-hinted attributes
            attrs_to_validate = list(type_hints.keys())

    validation_errors = []

    for attr_name in attrs_to_validate:
        if attr_name not in type_hints:
            continue

        try:
            value = getattr(obj, attr_name)
        except AttributeError:
            if strict_missing:
                validation_errors.append(
                    f"Attribute '{attr_name}' is annotated but missing from object"
                )
            continue

        expected_type = type_hints[attr_name]

        # Validate type (None values handled inside based on strict_none)
        error = _validate_attr_obj_type(
            name=attr_name,
            name_prefix="attribute",
            value=value,
            expected_type=expected_type,
            strict_none=strict_none,
            strict_unions=strict_unions,
        )

        if error:
            validation_errors.append(error)

    if validation_errors:
        raise TypeError(
            f"type validation failed in {fmt_type(obj)}:\n  " + "\n  ".join(validation_errors)
        )


def validate_param_types(
    *,
    skip: typing.Iterable[str] | None = None,
    only: typing.Iterable[str] | None = None,
) -> None:
    """
    Validate function parameters against their type hints (inline validation).

    Must be called from within a function to inspect its parameters and annotations.
    Uses the calling frame to automatically detect the function and its arguments.

    This is the inline validation approach. For automatic validation via decorator,
    use @valid_types instead.

    Args:
        skip: Parameter names to exclude from validation. Cannot be used with `only`.
        only: Parameter names to validate (all others ignored). Cannot be used with `skip`.

    Raises:
        TypeError: If a validated parameter doesn't match its type hint
        ValueError: If both `skip` and `only` are provided (mutually exclusive),
                    or if invalid parameter names are specified
        RuntimeError: If called outside a function context or function cannot be found

    Union Type Support:
        **Supported (always validated):**
            - Simple unions: int | str | float
            - Optional types: str | None, int | None
            - Union of basic types: list | dict | tuple

        **Unsupported (silently skipped):**
            - Parameterized generic unions: list[int] | dict[str, int]
            - Callable unions with different signatures: Callable[[int], str] | Callable[[str], int]

    🚀 Performance:
        ~50-100µs first call, ~10-20µs subsequent calls
        For hot paths, consider using @valid_types decorator instead (~5-15µs)

    See Also:
        valid_types: Decorator for automatic parameter type validation (faster)
        validate_attr_types(): Validate object attribute types

    Examples:
        >>> # Basic usage
        >>> def process_data(user_id: int, name: str | None, score: float = 0.0):
        ...     validate_param_types()
        ...     return f"{user_id}: {name} ({score})"
        ...
        >>> process_data(101, "Alice", 98.5)
        '101: Alice (98.5)'

        >>> process_data("invalid", "Alice", 98.5)  # ❌ Raises TypeError
        Traceback (most recent call last):
        ...
        TypeError: type validation failed in process_data():
          Parameter 'user_id' must be <int>, got <str>

        >>> # Validate only specific parameters
        >>> def api_endpoint(user_id: int, token: str, debug: bool = False):
        ...     validate_param_types(only=["user_id", "token"])  # Skip 'debug'
        ...     # ... rest of function
        ...
        >>> # Skip certain parameters
        >>> def send_message(id: int, payload: dict, metadata: dict):
        ...     validate_param_types(skip=["metadata"])  # Skip metadata validation
        ...     # ... rest of function
        ...
        >>> # Works with instance methods (automatically skips 'self')
        >>> class DataProcessor:
        ...     def process(self, data: int | str, strict_mode: bool = False):
        ...         validate_param_types()  # Skips 'self' automatically
        ...         # ... rest of method
        ...
        >>> processor = DataProcessor()
        >>> processor.process(42)  # ✅ Passes
        >>> processor.process(3.14)  # ❌ Raises TypeError
        Traceback (most recent call last):
        ...
        TypeError: type validation failed in process():
          Parameter 'data' must be <int> | <str>, got <float>

        >>> # Conditional validation (advantage over decorator)
        >>> def handle_request(data: dict, mode: str):
        ...     if mode == "strict":
        ...         validate_param_types()
        ...     # ... rest of function
        ...
        >>> # For standard cases, decorator is cleaner:
        >>> @valid_types
        ... def process(data: int | str):
        ...     return f"Processed {data}"
        ...
        >>> process(42)
        'Processed 42'

    Note:
        Use @valid_types decorator when possible for better performance (~3x faster).
        Use this inline version when:
        - You want validation hidden from function signature (cleaner public API)
        - You need conditional validation based on runtime logic
        - You're retrofitting validation into existing code without changing signatures
    """
    # 0. VALIDATION: Mutual Exclusivity
    if skip is not None and only is not None:
        raise ValueError("validate_param_types: Cannot use both 'skip' and 'only' parameters.")

    # 1. GET CALLING CONTEXT
    frame = inspect.currentframe()
    if frame is None:
        raise RuntimeError("Cannot get current frame")

    caller_frame = frame.f_back
    if caller_frame is None:
        raise RuntimeError("validate_param_types() must be called from within a function")

    try:
        func_name = caller_frame.f_code.co_name
        local_vars = caller_frame.f_locals.copy()

        # 2. FIND THE FUNCTION OBJECT
        func = _validate_param_get_fn_from_frame(caller_frame, func_name, local_vars)

        if func is None or not callable(func):
            raise RuntimeError(
                f"Cannot find function '{func_name}' to inspect its signature. "
                f"validate_param_types() may not work with:\n"
                f"  - Lambdas (use @valid_types decorator instead)\n"
                f"  - Functions created via exec() or eval()\n"
                f"  - Dynamically generated functions\n"
                f"For these cases, use the @valid_types decorator for automatic validation."
            )

        # 3. GET TYPE HINTS AND SIGNATURE
        try:
            type_hints = get_type_hints(func)
        except Exception:
            # Fallback to annotations if get_type_hints fails
            type_hints = getattr(func, "__annotations__", {}).copy()

        if not type_hints:
            # No type hints - nothing to validate
            return

        sig = inspect.signature(func)
        param_names = set(sig.parameters.keys())

        # 4. CONVERT AND VALIDATE skip/only PARAMETERS
        skip_set = set(skip) if skip else set()
        only_set = set(only) if only is not None else None

        # Validate that requested parameter names actually exist
        if only_set is not None:
            invalid = only_set - param_names
            if invalid:
                raise ValueError(
                    f"validate_param_types in '{func_name}': 'only' contains invalid parameter names: {invalid}"
                )

        if skip_set:
            invalid = skip_set - param_names
            if invalid:
                raise ValueError(
                    f"validate_param_types in '{func_name}': 'skip' contains invalid parameter names: {invalid}"
                )

        # 5. DETERMINE WHICH PARAMETERS TO VALIDATE
        params_to_validate = []

        for param_name in sig.parameters.keys():
            # Skip if no type hint
            if param_name not in type_hints:
                continue

            # Skip 'self' and 'cls' automatically
            if param_name in ("self", "cls"):
                continue

            # Skip if parameter wasn't passed (not in local_vars)
            if param_name not in local_vars:
                continue

            # Apply skip/only logic
            should_validate = False

            if only_set is not None:
                # Mode: ONLY (allow-list)
                should_validate = param_name in only_set
            else:
                # Mode: SKIP (block-list) or DEFAULT
                should_validate = param_name not in skip_set

            if should_validate:
                params_to_validate.append(param_name)

        # 6. VALIDATE EACH PARAMETER
        validation_errors = []

        for param_name in params_to_validate:
            value = local_vars[param_name]
            expected_type = type_hints[param_name]

            # Use shared validation logic
            error = _validate_param_single_value(
                name=param_name,
                name_prefix="parameter",
                value=value,
                expected_type=expected_type,
            )

            if error:
                validation_errors.append(error)

        if validation_errors:
            raise TypeError(
                f"type validation failed in {func_name}():\n  " + "\n  ".join(validation_errors)
            )

    finally:
        # Clean up frame references to avoid reference cycles
        del frame
        del caller_frame


def _validate_param_get_fn_from_frame(
    caller_frame: Any,
    func_name: str,
    local_vars: dict[str, Any],
) -> Callable[..., Any] | None:
    """
    Find the function object from the calling frame.

    Tries multiple strategies:
    1. Caller's globals
    2. Method from self/cls (with descriptor unwrapping)
    3. Locals search by code object
    4. Enclosing frames search
    5. Check class __dict__ for descriptors

    Returns:
        The function object, or None if not found
    """
    func = None

    # Strategy 1: Try caller's globals
    if func_name in caller_frame.f_globals:
        candidate = caller_frame.f_globals[func_name]
        if callable(candidate) and hasattr(candidate, "__code__"):
            func = candidate

    # Strategy 2: For methods, try to get from self/cls
    # Need to check class __dict__ for descriptors (staticmethod, classmethod, property)
    if func is None and "self" in local_vars:
        obj_class = type(local_vars["self"])

        # First check class __dict__ for descriptors
        for cls in obj_class.__mro__:
            if func_name in cls.__dict__:
                attr = cls.__dict__[func_name]
                if isinstance(attr, staticmethod):
                    func = attr.__func__
                    break
                elif isinstance(attr, classmethod):
                    func = attr.__func__
                    break
                elif isinstance(attr, property):
                    # For property, we need to determine if we're in getter or setter
                    # Check if 'self' and one other param (the value being set)
                    func = attr.fset if attr.fset else attr.fget
                    break
                elif callable(attr):
                    func = attr
                    break

    if func is None and "cls" in local_vars:
        obj_class = local_vars["cls"]

        # Check class __dict__ for descriptors
        for cls in obj_class.__mro__:
            if func_name in cls.__dict__:
                attr = cls.__dict__[func_name]
                if isinstance(attr, staticmethod):
                    func = attr.__func__
                    break
                elif isinstance(attr, classmethod):
                    func = attr.__func__
                    break
                elif callable(attr):
                    func = attr
                    break

    # Strategy 3: Search locals for matching code object
    if func is None:
        for obj in caller_frame.f_locals.values():
            if callable(obj) and hasattr(obj, "__code__") and obj.__code__ is caller_frame.f_code:
                func = obj
                break

    # Strategy 4: Search enclosing frames
    if func is None:
        search_frame = caller_frame.f_back
        while search_frame is not None:
            for obj in search_frame.f_locals.values():
                if (
                    callable(obj)
                    and hasattr(obj, "__code__")
                    and obj.__code__ is caller_frame.f_code
                ):
                    func = obj
                    break
            if func is not None:
                break
            search_frame = search_frame.f_back

    # Strategy 5: Try to find in enclosing class definitions
    # Look for the function in nested class definitions
    if func is None:
        search_frame = caller_frame.f_back
        while search_frame is not None:
            # Check if we're defining a class (look for __qualname__ being set)
            for name, obj in search_frame.f_locals.items():
                if inspect.isclass(obj):
                    # Check this class's __dict__ for our function
                    for cls in obj.__mro__:
                        if func_name in cls.__dict__:
                            attr = cls.__dict__[func_name]
                            if isinstance(attr, staticmethod):
                                func = attr.__func__
                                break
                            elif isinstance(attr, classmethod):
                                func = attr.__func__
                                break
                            elif isinstance(attr, property):
                                func = attr.fset if attr.fset else attr.fget
                                break
                            elif callable(attr):
                                func = attr
                                break
                    if func is not None:
                        break
            if func is not None:
                break
            search_frame = search_frame.f_back

    return func


def _validate_param_single_value(
    name: str,
    name_prefix: Literal["attribute", "parameter"],
    value: Any,
    expected_type: Any,
) -> str | None:
    """
    Validate a single value against its expected type.

    Shared validation logic for both decorator and inline approaches.

    Args:
        name: Name of the parameter/attribute
        name_prefix: Either "attribute" or "parameter"
        value: The actual value to validate
        expected_type: The type annotation to validate against

    Returns:
        Error message string if validation fails, None if validation passes
    """
    # Handle string annotations
    if isinstance(expected_type, str):
        return None  # Cannot validate string annotations at runtime

    # Skip typing.Any
    if expected_type is typing.Any:
        return None

    # Get origin for generic/union types
    origin = get_origin(expected_type)

    # Handle Union types (both T | None and Union[T, ...])
    if origin in UNION_TYPES:
        args = get_args(expected_type)

        try:
            if not isinstance(value, args):
                # Format union members nicely
                type_names = " | ".join(fmt_type(t) for t in args)
                return (
                    f"{name_prefix.capitalize()} '{name}' must be {type_names}, "
                    f"got {fmt_type(value)}"
                )
            return None  # Validation passed
        except TypeError:
            # isinstance failed - complex union that can't be validated
            return None  # Silently skip (consistent with decorator behavior)

    # Handle other generic types (list[T], dict[K,V], etc.)
    if origin is not None:
        expected_type = origin

    # Final isinstance check for simple types
    try:
        if not isinstance(value, expected_type):
            return (
                f"{name_prefix.capitalize()} '{name}' must be {fmt_type(expected_type)}, "
                f"got {fmt_type(value)}"
            )
    except TypeError:
        # isinstance failed for non-union type
        return None  # Silently skip

    return None  # Valid
