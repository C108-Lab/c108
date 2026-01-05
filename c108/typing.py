"""
Runtime type validation utilities.
"""

import functools
import inspect
import sys
import typing

# Python 3.10+ has types.UnionType for X | Y syntax
if sys.version_info >= (3, 10):
    import types

    UNION_TYPES = (typing.Union, types.UnionType)
else:
    UNION_TYPES = (typing.Union,)


def validate_types(
    func=None, *, skip: typing.Iterable[str] = None, only: typing.Iterable[str] = None
):
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

            @validate_types
            def connect(host: str, port: int) -> None:
                pass

        Validate only specific parameters (gradual adoption):

            @validate_types(only=("host",))
            def connect(host: str, port: int, options: dict) -> None:
                pass

        Skip validation for specific parameters:

            @validate_types(skip=("payload",))
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
        return functools.partial(validate_types, skip=skip, only=only)

    # 0. VALIDATION: Mutual Exclusivity
    if skip is not None and only is not None:
        raise ValueError(
            f"@validate_types: Cannot use both 'skip' and 'only' on function '{func.__name__}'."
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
                f"@validate_types on '{func.__name__}': 'only' contains invalid parameter names: {invalid}"
            )

    if skip_set:
        invalid = skip_set - param_names
        if invalid:
            raise ValueError(
                f"@validate_types on '{func.__name__}': 'skip' contains invalid parameter names: {invalid}"
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
                        f"Argument '{name}' expected {expected}, got {type(value).__name__}"
                    )

        return func(*args, **kwargs)

    return wrapper
