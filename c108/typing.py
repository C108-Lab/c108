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
    Validate function arguments against their type hints at runtime.

    By default, validates all annotated parameters. Use `skip` or `only` to
    control which parameters are checked. Validation happens once per call
    with minimal overhead.

    Args:
        skip: Parameter names to exclude from validation. Cannot be used with `only`.
        only: Parameter names to validate (all others ignored). Cannot be used with `skip`.

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
    exclude_none: bool = False,
    include_inherited: bool = True,
    include_private: bool = False,
    pattern: str | None = None,
    strict: bool = True,
    allow_none: bool = True,
    fast: bool | Literal["auto"] = "auto",
) -> None:
    """
    Validate that object attributes match their type annotations.

    Supports dataclasses, attrs classes, and regular Python classes with
    type annotations. Performance-optimized with a fast path for dataclasses.

    This function validates the types of object attributes. For validating
    function parameters, see validate_param_types() (inline) or @valid_param_types
    (decorator).

    Args:
        obj: Object instance to validate
        attrs: Optional list of specific attribute names to validate.
               If None, validates all annotated attributes.
        exclude_none: If True, skip validation for attributes with None values
        include_inherited: If True, validates inherited attributes with type hints
        include_private: If True, validates private attributes (starting with '_')
        pattern: Optional regex pattern to filter which attributes to validate
        strict: If True (default), raise TypeError when encountering Union types that
                cannot be validated with isinstance() (e.g., list[int] | dict[str, int],
                Callable[[int], str] | Callable[[str], int]). If False, silently skip
                such unions. Simple unions like int | str | None are always validated
                regardless of this flag.
        allow_none: If True, None values pass validation for Optional types (T | None).
                    If False, None values must explicitly match the type hint.
        fast: Performance mode:
              - "auto" (default): Automatically use fast path when possible
              - True: Force fast path, raise ValueError if incompatible options provided
              - False: Force slow path using search_attrs()

    Raises:
        TypeError: If attribute type doesn't match annotation, or if strict=True
                   and a truly unvalidatable Union type is encountered
        ValueError: If obj has no type annotations, or if fast=True with incompatible options
        RuntimeError: If Python version < 3.11

    🚀 Performance:
        Fast path (dataclasses only, ~5-10µs):
            - Requires: is_dataclass(obj)=True, attrs=None, pattern=None, include_private=False
            - 5-10x faster than slow path
            - Recommended for high-throughput production scenarios

        Slow path (all classes, ~30-70µs):
            - Uses search_attrs() for flexible filtering
            - Supports pattern matching, private attrs, custom attr lists
            - Suitable for validation at API boundaries, config loading

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
        >>> # Validate with default settings
        >>> validate_attr_types(obj)
        >>>
        >>> # Force fast path (raises if incompatible)
        >>> validate_attr_types(obj, fast=True)
        >>>
        >>> # Use slow path with pattern matching
        >>> validate_attr_types(obj, pattern=r"^api_.*", fast=False)
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

    # Validate fast mode compatibility
    if fast is True and not can_use_fast:
        incompatible = []
        if not is_dc:
            incompatible.append("obj is not a dataclass")
        if attrs is not None:
            incompatible.append("attrs parameter is set")
        if pattern is not None:
            incompatible.append("pattern parameter is set")
        if include_private:
            incompatible.append("include_private=True")

        raise ValueError(
            f"cannot use fast=True with current options. "
            f"Fast path is only available for dataclasses without filtering. "
            f"Incompatible settings: {', '.join(incompatible)}. "
            f"Either remove these options or use fast=False or fast='auto'."
        )

    # Choose path
    use_fast = (fast is True) or (fast == "auto" and can_use_fast)

    if use_fast:
        _validate_attr_dataclass_fast(
            obj,
            exclude_none=exclude_none,
            strict=strict,
            allow_none=allow_none,
        )
    else:
        _validate_attr_with_search(
            obj,
            attrs=attrs,
            exclude_none=exclude_none,
            include_inherited=include_inherited,
            include_private=include_private,
            pattern=pattern,
            strict=strict,
            allow_none=allow_none,
        )


def _validate_attr_dataclass_fast(
    obj: Any,
    *,
    exclude_none: bool,
    strict: bool,
    allow_none: bool,
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
    """
    validation_errors = []

    # fields() returns cached metadata - very fast
    for field in dc_fields(obj):
        attr_name = field.name
        expected_type = field.type

        # Direct attribute access
        value = getattr(obj, attr_name)

        # Skip None if requested
        if exclude_none and value is None:
            continue

        # Validate type
        error = _validate_attr_obj_type(
            name=attr_name,
            name_prefix="attribute",
            value=value,
            expected_type=expected_type,
            allow_none=allow_none,
            strict=strict,
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
    allow_none: bool,
    strict: bool,
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
        allow_none: Whether None is acceptable for Optional types
        strict: Whether to raise errors for truly unvalidatable unions

    Returns:
        Error message string if validation fails, None if validation passes
    """
    # Handle string annotations (should be rare in modern Python)
    if isinstance(expected_type, str):
        if strict:
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

        if is_optional:
            # Union includes None (e.g., int | None, int | str | None)
            if allow_none and value is None:
                return None  # None is explicitly allowed

            # Validate against non-None types (whether single or multiple)
            try:
                if not isinstance(value, non_none_types):
                    # Format union members nicely
                    type_names = " | ".join(fmt_type(t) for t in non_none_types)
                    if allow_none:
                        type_names += " | None"
                    return (
                        f"{name_prefix.capitalize()} '{name}' must be {type_names}, "
                        f"got {fmt_type(value)}"
                    )
                return None  # Validation passed
            except TypeError:
                # isinstance failed - truly complex union (e.g., list[int] | dict[str, int])
                if strict:
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
                if strict:
                    return (
                        f"{name_prefix.capitalize()} '{name}' has complex Union type "
                        f"which cannot be validated with isinstance()"
                    )
                return None  # Skip in non-strict mode

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
        if strict:
            return f"Cannot validate {name_prefix.lower()} '{name}' with complex type"
        return None  # Skip

    return None  # Valid


def _validate_attr_with_search(
    obj: Any,
    *,
    attrs: list[str] | None,
    exclude_none: bool,
    include_inherited: bool,
    include_private: bool,
    pattern: str | None,
    strict: bool,
    allow_none: bool,
) -> None:
    """
    Slower path using search_attrs for complex filtering.

    Performance: ~30-70µs per validation

    Used when:
    - Not a dataclass
    - Custom attrs list provided
    - Pattern filtering needed
    - Private attribute inclusion needed
    - Non-inherited attributes only
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
        attrs_to_validate = attrs
    else:
        attrs_to_validate = search_attrs(
            obj,
            format="list",
            exclude_none=exclude_none,
            include_inherited=include_inherited,
            include_methods=False,
            include_private=include_private,
            include_properties=False,
            pattern=pattern,
            skip_errors=True,
        )

    validation_errors = []

    for attr_name in attrs_to_validate:
        if attr_name not in type_hints:
            continue

        try:
            value = getattr(obj, attr_name)
        except AttributeError:
            continue

        if exclude_none and value is None:
            continue

        expected_type = type_hints[attr_name]

        error = _validate_attr_obj_type(
            name=attr_name,
            name_prefix="attribute",
            value=value,
            expected_type=expected_type,
            allow_none=allow_none,
            strict=strict,
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
