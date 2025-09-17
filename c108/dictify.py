"""
C108 Dictify Tools
"""

# Standard library -----------------------------------------------------------------------------------------------------
import collections.abc as abc
import inspect

from copy import copy
from enum import Enum, unique
from dataclasses import dataclass, replace as dataclasses_replace
from typing import Any, Iterable, Callable, Mapping

# Local ----------------------------------------------------------------------------------------------------------------
from .abc import is_builtin, attrs_search, attr_is_property, ObjectInfo
from .tools import fmt_any, fmt_type
from .utils import class_name


# Classes --------------------------------------------------------------------------------------------------------------

@unique
class HookMode(str, Enum):
    """
    Object conversion strategy:
        - "dict": try object's to_dict() method, then fallback
        - "dict_strict": require to_dict() method
        - "none": skip object hooks
    """
    DICT = "dict"
    DICT_STRICT = "dict_strict"
    NONE = "none"


@dataclass
class DictifyOptions:
    """
    Configuration options for object-to-dict conversion.

    Controls how objects are converted to dictionaries, including recursion depth,
    attribute filtering, size limits, and type handling.

    Attributes:
        max_depth: Maximum recursion depth for nested objects (default: 3)

        include_class_name: Include class name in dict representation of user objects with '__class__' as key
        inject_class_name: Inject class name into to_dict() results with '__class__' as key
        include_none_attrs: Include attributes with None values from user objects
        include_none_items: Include dictionary items with None values
        include_private: Include private attributes (starting with _) from user classes
        include_properties: Include instance properties with assigned values

        max_items: Maximum number of items in collections (sequences, mappings, and sets)
        max_string_length: Maximum length for string values (truncated if exceeded)
        max_bytes: Maximum size for 'bytes' object (truncated if exceeded)
        sort_keys: Mappings key ordering

        hook_mode: Object conversion strategy - "dict" (try to_dict() then fallback),
                  "dict_strict" (require to_dict() method), or "none" (skip object hooks)
        fully_qualified_names: Use fully qualified class names (module.Class vs Class)
        to_str: Types to convert to string representation instead of dict
        always_filter: Types that are always processed through filtering
        never_filter: Types that skip filtering and preserve original values

    Examples:
        >>> # Basic usage with defaults
        >>> options = DictifyOptions()

        >>> # Debugging configuration
        >>> debug_opts = DictifyOptions(
        ...     max_depth=1,
        ...     include_private=True,
        ...     max_items=50
        ... )

        >>> # Serialization configuration
        >>> serial_opts = DictifyOptions(
        ...     include_class_name=True,
        ...     include_none_attrs=False,
        ...     max_string_length=100
        ... )
    """

    max_depth: int = 3

    include_class_name: bool = False
    inject_class_name: bool = False
    inject_none_attrs: bool = False
    include_none_attrs: bool = True
    include_none_items: bool = True
    include_private: bool = False
    include_properties: bool = False

    # Size limits
    max_items: int = 1000
    max_string_length: int = 240
    max_bytes: int = 1024

    # Mapping keys ordering
    sort_keys: bool = False

    # Advanced
    hook_mode: str = HookMode.DICT
    fully_qualified_names: bool = False
    to_str: tuple[type, ...] = ()
    always_filter: tuple[type, ...] = ()
    never_filter: tuple[type, ...] = ()


# Methods --------------------------------------------------------------------------------------------------------------

def core_dictify(obj: Any,
                 *,
                 options: DictifyOptions | None = None,
                 fn_raw: Callable[[Any], Any] = lambda x: x,
                 fn_terminal: Callable[[Any], Any] | None = None) -> Any:
    """
    Advanced object-to-dict conversion with full configurability and custom processing hooks.

    This is the core engine powering dictify() and serial_dictify(), offering complete control
    over conversion behavior through DictifyOptions and custom processing functions. Converts
    objects to dictionaries while preserving primitives, sequences, and sets in their original
    forms, with configurable depth limits, filtering rules, size constraints, and recursion
    depth handlers.

    Args:
        obj: Object to convert to dictionary
        options: DictifyOptions instance controlling conversion behavior
        fn_raw: Handler for raw/minimal processing mode (max_depth < 0). Called as fallback
               when obj.to_dict() is not available; defaults to identity function.
        fn_terminal: Handler for when recursion depth is exhausted (max_depth = 0) or one of
                    always_filter types encountered. Takes precedence over obj.to_dict() if provided. Defaults to None
                    (uses fallback chain obj.to_dict() → identity function).

    Returns:
        Human-readable data representation of the object

    Raises:
        TypeError: If options, fn_raw, or fn_terminal have invalid types
        TypeError: If hook_mode is 'dict_strict' and object lacks to_dict() method
        ValueError: If hook_mode contains invalid value

    Examples:
        # Basic usage with custom options
        opts = DictifyOptions(max_depth=5, include_private=True)
        result = core_dictify(obj, options=opts)

        # With custom processing functions
        def custom_terminal(x):
            return f"<truncated: {type(x).__name__}>"

        result = core_dictify(obj, options=opts, fn_terminal=custom_terminal)

    Note:
        - max_depth < 0: Returns obj.to_dict() or fn_raw() (raw/minimal processing)
        - max_depth = 0: Returns fn_terminal(), obj.to_dict(), or identity (terminal processing)
        - max_depth = N: Recurses N levels deep into collections; objects expand to dicts with attribute values
        processed at depth N-1
        - Builtins, which are never filtered: int, float, bool, complex, None, range
        - Builtins, which are always filtered: str, bytes, bytearray, memoryview
        - Never-filtered objects are returned as-is, custom handlers not applicable
        - Properties that raise exceptions are automatically skipped
        - Class name include (if enabled) only affects main recursive processing and optionally
        obj.to_dict() results injection; fn_raw() and fn_terminal() outputs are never modified
        - Mappings keys sorting (if enabled) applies only to main recursive processing and obj.to_dict() results
    """
    if not isinstance(options, (DictifyOptions, type(None))):
        raise TypeError(f"options must be a DictifyOptions instance, but found {fmt_type(options)}")
    if not isinstance(fn_raw, Callable):
        raise TypeError(f"fn_raw must be a Callable, but found {fmt_type(fn_raw)}")
    if not isinstance(fn_terminal, (Callable, type(None))):
        raise TypeError(f"fn_terminal must be a Callable or None, but found {fmt_type(fn_terminal)}")

    # Use defaults if no options provided
    opt = options or DictifyOptions()

    if not isinstance(opt.max_depth, int):
        raise TypeError(f"Recursion depth must be int but found: {fmt_any(opt.max_depth)}")

    def __core_dictify(obj, recursion_depth: int, opt: DictifyOptions):
        opt_inner = copy(opt)
        opt_inner.max_depth = recursion_depth
        return core_dictify(obj, options=opt_inner, fn_raw=fn_raw, fn_terminal=fn_terminal)

    def __fn_terminal(obj: Any, opt: DictifyOptions) -> Any:
        opt = opt or DictifyOptions()
        if fn_terminal is not None:
            return fn_terminal(obj)
        elif dict_ := _get_from_to_dict(obj, opt):
            return dict_
        else:
            return obj  # Final fallback

    def __process_collection(obj: abc.Collection[Any],
                             rec_depth: int,
                             opt: DictifyOptions,
                             from_object: bool) -> Any:
        """Process items recursively in a collection with __len__ method"""
        if not _implements_len(obj):
            raise TypeError(f"obj must be a collection which implements __len__ method, but found {fmt_type(obj)}")

        if rec_depth < 0:
            raise OverflowError(f"Collection recursion depth out of range: {rec_depth}")

        if rec_depth == 0 or len(obj) > opt.max_items:
            return __fn_terminal(obj, opt=opt)

        if isinstance(obj, (abc.Sequence, abc.Set)):
            # Other sequence types should be handled individually,
            # see str, bytes, bytearray, memoryview in serialize_object
            return type(obj)(__core_dictify(item, recursion_depth=rec_depth - 1, opt=opt) for item in obj)

        elif isinstance(obj, (dict, abc.Mapping)):
            inc_nones = opt.include_none_attrs if from_object else opt.include_none_items
            dict_ = {k: __core_dictify(v, recursion_depth=(rec_depth - 1), opt=opt) for k, v in obj.items()
                     if (v is not None) or inc_nones}
            if opt.include_class_name:
                dict_["__class__"] = _class_name(obj, options=opt)
            if opt.sort_keys:
                dict_ = dict(sorted(dict_.items()))
            return dict_
        else:
            raise TypeError(f"Items container must be a Sequence, Mapping or Set but found: {fmt_type(obj)}")

    # core_dictify() body Start ---------------------------------------------------------------------------

    # Return never-filtered objects
    if _is_never_filtered(obj, options=opt):
        return obj

    # Start depth Edge Cases processing -----------------------------------
    if opt.max_depth < 0:
        # Raw mode
        if dict_ := _get_from_to_dict(obj, opt):
            return dict_
        else:
            return fn_raw(obj)  # Always defined, no None check needed

    elif opt.max_depth == 0:
        # Terminal condition when recursion exhausted
        return __fn_terminal(obj, opt=opt)
    # End depth Edge Cases processing -----------------------------------

    # Check and replace always-filtered types which should include
    # expandable to large-size builtins and known third-party large types
    builtins_always_filter = [str, bytes, bytearray, memoryview]
    always_filter = [*builtins_always_filter, *list(opt.always_filter)]

    if isinstance(obj, tuple(always_filter)):
        return __fn_terminal(obj, opt=opt)

    if isinstance(obj, tuple(opt.to_str)):
        return _to_str(obj, fully_qualified=opt.fully_qualified_names)

    # Should check item-based Instances for recursion: list, tuple, set, dict, etc
    if isinstance(obj, (abc.Sequence, abc.Mapping, abc.Set)):
        if not _implements_len(obj):
            return __fn_terminal(obj, opt=opt)
        else:
            return __process_collection(obj, rec_depth=opt.max_depth, opt=opt, from_object=False)

    # We should arrive here only when max_depth > 0 (recursion not exhausted)
    # Should expand obj 1 level into deep.
    dict_ = _shallow_to_dict(obj, opt=opt)

    return __process_collection(dict_, rec_depth=opt.max_depth, opt=opt, from_object=True)
    # core_dictify() body End ---------------------------------------------------------------------------


def _class_name(obj: Any, options: DictifyOptions) -> str:
    return class_name(obj,
                      fully_qualified=options.fully_qualified_names,
                      fully_qualified_builtins=False,
                      start="", end="")


def _get_from_to_dict(obj, options: DictifyOptions | None = None) -> abc.Mapping[Any, Any] | None:
    """Returns obj.to_dict() value if the method is available and hook mode allows"""

    opt = options or DictifyOptions()

    # Process HookMode ------------------------------------------

    if opt.hook_mode == HookMode.DICT:
        fn = getattr(obj, "to_dict", None)
        dict_ = fn() if callable(fn) else None

    elif opt.hook_mode == HookMode.DICT_STRICT:
        fn = getattr(obj, "to_dict", None)
        if not callable(fn):
            raise TypeError(f"Class {fmt_type(obj)} must implement to_dict() when hook_mode='{HookMode.DICT_STRICT}'")
        dict_ = fn()

    elif opt.hook_mode == HookMode.NONE:
        dict_ = None

    else:
        valid = ", ".join([f"'{v.value}'" for v in HookMode])
        raise ValueError(f"Unknown hook_mode value: {fmt_any(opt.hook_mode)}. Expected: {valid}")

    # Check teh returned type -----------------------------------
    if dict_ is not None:
        if not isinstance(dict_, abc.Mapping):
            raise TypeError(f"Object's to_dict() must return a Mapping, but got {fmt_type(dict_)}")

    # dict_ should be injectable
    dict_ = {**dict_}
    if opt.include_class_name:
        dict_["__class__"] = _class_name(obj, options=opt)
    if opt.sort_keys:
        dict_ = dict(sorted(dict_.items()))

    return dict_


def _implements_len(obj: abc.Collection[Any]) -> bool:
    """Returns True if obj implements __len__"""
    try:
        len(obj)
        return True
    except TypeError:
        return False


def _is_never_filtered(obj: Any, options: DictifyOptions) -> bool:
    """
    Check obj against never-filtered types, which should include
    plain-to-display types plus opt.never_filter types
    """
    if isinstance(obj, (int, float, bool, complex, type(None), range)):
        return True
    elif isinstance(obj, tuple(options.never_filter)):
        return True
    else:
        return False


def _count_positional_args(fn: abc.Callable[..., Any]) -> int:
    """
    Return the number of positional parameters a callable exposes, or -1 when not applicable.

    Parameters
    ----------
    fn: Callable[..., Any]
        The object to inspect; may be a function, bound method, or callable instance.

    Returns:
        The number of positional parameters; -1 for non-callables
        or for callables that include a VAR_POSITIONAL parameter (*args).
    """
    if not callable(fn):
        return -1

    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    # If *args present, the callable can accept multiple positional arguments — reject.
    for p in params:
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
            return -1

    # Count only positional-like parameters (positional-only and positional-or-keyword).
    positional = [p for p in params if p.kind == inspect.Parameter.POSITIONAL_ONLY]

    # Return the number of positional parameters.
    return len(positional)


def _merge_options(options: DictifyOptions | None, **kwargs) -> DictifyOptions:
    opt = DictifyOptions(**kwargs) if options is None else dataclasses_replace(options, **kwargs)
    return opt


def _shallow_to_dict(obj: Any, *, opt: DictifyOptions = None) -> dict[str, Any]:
    """
    Shallow object to dict converter for user objects.

    This method generates a dictionary with attributes and their corresponding values for a given object or class.
    No recursion supported.

    Method is NOT intended for primitives or builtins processing (list, tuple, dict, etc)

    Returns:
        dict[str, Any] - dictionary containing attributes and their values

    Note:
        inc_none_attrs: Include attributes with None values
        inc_private: Include private attributes of user classes
        inc_property: Include instance properties with assigned values, has no effect if obj is a class
    """
    dict_ = {}

    attributes = attrs_search(obj, inc_private=opt.include_private, inc_property=opt.include_properties,
                              inc_none_attrs=opt.include_none_attrs)
    is_class_or_dataclass = inspect.isclass(obj)

    for attr_name in attributes:
        if attr_name.startswith('__') and attr_name.endswith('__'):
            continue  # Skip dunder methods

        is_obj_property = attr_is_property(attr_name, obj)

        if not is_class_or_dataclass and is_obj_property:
            if not opt.include_properties:
                continue  # Skip properties

            try:
                attr_value = getattr(obj, attr_name)
            except Exception:
                continue  # Skip if instance property getter raises exception

        else:
            attr_value = getattr(obj, attr_name)

        if callable(attr_value):
            continue  # Skip methods

        dict_[attr_name] = attr_value

    if opt.include_class_name:
        dict_["__class__"] = _class_name(obj, options=opt)
    if opt.sort_keys:
        dict_ = dict(sorted(dict_.items()))

    return dict_


def _to_str(obj: Any, fully_qualified: bool = True, fully_qualified_builtins: bool = False) -> str:
    """
    Returns custom <str> value of object.

    If custom __str__ method not found, overrides the stdlib __str__ with optionally Fully Qualified Class Name
    """
    has_default_str = obj.__str__ == object.__str__
    if not has_default_str:
        as_str = str(obj)
    else:
        cls_name = class_name(obj, fully_qualified=fully_qualified, fully_qualified_builtins=fully_qualified_builtins,
                              start="", end="")
        as_str = f'<class {cls_name}>'
    return as_str


def dictify(obj: Any, *,
            max_depth: int = 3,
            include_private: bool = False,
            include_class_name: bool = False,
            max_items: int = 50,
            options: DictifyOptions | None = None) -> Any:
    """
    Simple object-to-dict conversion with common customizations.

    Converts Python objects to human-readable dictionary representations, preserving
    built-in types while converting custom objects to dictionaries. This is a simplified
    interface to core_dictify() with sensible defaults for common use cases like debugging,
    logging, and data inspection.

    Args:
        obj: Object to convert to dictionary representation
        max_depth: Maximum recursion depth for nested objects
        include_private: Include private attributes starting with underscore
        include_class_name: Include class name in object representations
        max_items: Maximum number of items in collections
        options: DictifyOptions instance for advanced configuration; individual parameters
                passed to dictify() override corresponding options fields

    Returns:
        Human-readable data representation preserving built-in types and converting
        objects to dictionaries

    Raises:
        TypeError: If max_depth or max_items are not integers
        TypeError: If options is not a DictifyOptions instance or None

    Examples:
        # Basic object conversion
        >>> class Person:
        ...     def __init__(self, name, age):
        ...         self.name = name
        ...         self.age = age
        >>> p = Person("Alice", 30)
        >>> dictify(p)
        {'name': 'Alice', 'age': 30}

        # Include class information for debugging
        >>> dictify(p, include_class_name=True)
        {'__class__': 'Person', 'name': 'Alice', 'age': 30}

        # Control recursion depth
        >>> nested = {'users': [p], 'count': 1}
        >>> dictify(nested, max_depth=1)
        {'users': [{'name': 'Alice', 'age': 30}], 'count': 1}

        # Advanced configuration with options
        >>> from c108.dictify import DictifyOptions
        >>> opts = DictifyOptions(include_none_attrs=False, max_string_length=20)
        >>> dictify(p, max_depth=2, options=opts)

    Note:
        - Built-in types (int, str, list, dict, etc.) are preserved as-is
        - Custom objects are converted to shallow dictionaries at processing boundaries
        - Collections are recursively processed up to max_depth levels
        - Effective inspection depth may be max_depth + 1 for object attributes due to
          shallow dictionary conversion
        - Properties that raise exceptions are automatically skipped
        - For advanced control, use core_dictify() directly

    See Also:
        core_dictify: Advanced conversion with full configurability
        serial_dictify: Serialization-focused conversion with string handling
    """

    if not isinstance(max_depth, int):
        raise TypeError(f"max_depth must be int but found: {fmt_any(max_depth)}")
    if not isinstance(max_items, int):
        raise TypeError(f"max_items must be int but found: {fmt_any(max_items)}")
    if not isinstance(options, (DictifyOptions, type(None))):
        raise TypeError(f"options must be a DictifyOptions instance, but found {fmt_type(options)}")

    include_private = bool(include_private)
    include_class_name = bool(include_class_name)
    opt = _merge_options(options,
                         max_depth=max_depth,
                         include_private=include_private,
                         include_class_name=include_class_name,
                         max_items=max_items,
                         )

    def __dictify_process(obj: Any) -> Any:
        # Return never-filtered objects
        if _is_never_filtered(obj, options=opt):
            return obj
        # Should convert to dict the topmost obj level but keep its inner objects as-is
        dict_ = _shallow_to_dict(obj, opt=opt)
        return dict_

    return core_dictify(obj,
                        fn_raw=lambda x: x,
                        fn_terminal=__dictify_process,
                        options=opt)


def to_dict_OLD(obj: Any,
                inc_class_name: bool = False,
                inc_none_attrs: bool = True,
                inc_none_items: bool = False,
                inc_private: bool = False,
                inc_property: bool = False,
                max_items: int = 10 ** 21,
                fq_names: bool = True,
                recursion_depth=0,
                hook_mode: str = "flexible") -> dict[str, Any]:
    def __process_obj(obj: Any) -> Any:

        if is_builtin(obj):
            return obj

        # Should convert to dict the topmost obj level but keep its inner objects as-is
        dict_ = _shallow_to_dict(
            obj, inc_class_name=inc_class_name,
            inc_none_attrs=inc_none_attrs,
            inc_private=inc_private, inc_property=inc_property,
            fq_names=fq_names)
        return dict_

    # Should return builtins as is
    if is_builtin(obj):
        return obj

    return core_to_dict_OLD(obj,
                            # fn_raw specifies what to do if recursion impossible
                            fn_plain=lambda x: x,
                            # fn_terminal is applied on final recursion step and on always_filter types
                            fn_process=__process_obj,
                            inc_class_name=inc_class_name,
                            inc_none_attrs=inc_none_attrs,
                            inc_none_items=inc_none_items,
                            inc_private=inc_private,
                            inc_property=inc_property,
                            max_items=max_items,
                            fq_names=fq_names,
                            recursion_depth=recursion_depth,
                            hook_mode=hook_mode)


# TODO any sense to keep it public at all? Whats the essential diff from core_dictify which proves existence
#  of this method in public API? -- serialization safe limits or what?? The sense is that we always filter terminal
#  attrs when depth is reached or what? If we use it in YAML package only, maybe keep it there as private method?
def serialize_object_OLD(obj: Any,
                         inc_class_name: bool = False,
                         inc_none_attrs: bool = True,
                         inc_none_items: bool = False,
                         inc_private: bool = False,
                         inc_property: bool = False,
                         max_bytes: int = 108,
                         max_items: int = 14,
                         max_str_len: int = 108,
                         max_str_prefix: int = 28,
                         always_filter: Iterable[type] = (),
                         never_filter: Iterable[type] = (),
                         to_str: Iterable[type] = (),
                         fq_names: bool = True,
                         recursion_depth=0,
                         hook_mode: str = "flexible") -> dict[str, Any]:
    """
    Prepare objects for serialization to YAML, JSON, or other formats.

    Converts problematic types to string representations and applies
    size limits suitable for human-readable output and debugging.

    Primitive data returned as is, iterables are processed recursively with empty collections handled,
    user classes and instances are returned as data-only dict.

    Recursion of level N filters all objects from top level 0 deep to level N. At its deepest level
    attrs are shown as a single primitive value or a stats string.

    The ``serialize_object()`` is a sibling method to ``as_dict()``, both of them derive from ``core_dict()`` utility

    Parameters:
        obj: The object for which attributes need to be processed
        inc_class_name : Include class name into dict-s created from user objects
        inc_none_attrs : Include attributes with None value in dictionaries from user objects
        inc_none_items : Include items with None value in dictionaries
        inc_private    : Include private attributes of user classes
        inc_property   : Include instance properties with assigned values, has no effect if obj is a class
        max_bytes      : Size limit for byte types, i.e. bytes, bytearray, memoryview
        max_items      : Length limit for sequence, set and mapping types including list, tuple, dict, set, frozenset
        max_str_len    : Length limit for str type
        max_str_prefix : Length limit for long str prefix showed after filtering
        always_filter  : Collection of types to be always filtered. Note that always_filter=[int] >> =[int, boolean]
        never_filter   : Collection of types to skip filtering and preserve original values
        to_str         : User types of attributes to be converted to <str>
        fq_names       : Use Fully Qualified class names
        recursion_depth: Recursion depth for item containers (sequence, set, mapping) and as_dict expandable objects

    Returns:
        dict[str, Any] - dictionary containing attributes and their values

    See also: ``print_as_yaml()``

    Note:
        - recursion_depth < 0  returns obj without filtering
        - recursion_depth = 0 returns unfiltered obj for primitives, object info for iterables, and custom classes
        - recursion_depth = N iterates by N levels of recursion on iterables and objects expandable with ``as_dict()``
        - inc_property = True: the method always skips properties which raise exception

    Examples:
        # For YAML serialization
        yaml_ready = serialize_object(obj, to_str=(datetime, UUID))

        # With size limits for debugging
        debug_data = serialize_object(obj, max_items=20, max_depth=2)

    """

    def __object_info(obj,
                      max_bytes: int = max_bytes,
                      max_str_len: int = max_str_len,
                      max_str_prefix: int = max_str_prefix,
                      fq_names: bool = fq_names) -> str:

        _info = ObjectInfo.from_object(obj, fq_name=fq_names).as_str
        # Should check non-recursive large-size builtins
        if isinstance(obj, str):
            stp = obj.strip()
            str_prefix = stp[:max_str_prefix].strip()
            return _info + f" | {str_prefix}..." if len(obj) > max_str_len else obj.strip()
        elif isinstance(obj, (bytes, bytearray, memoryview)):
            # memoryview is similar to bytes but does NOT support .strip() which is present in str and bytes
            #            it can be a view to any object, including bytes
            str_prefix = bytes(obj[:max_str_prefix])
            return _info + f" | {str_prefix}..." if len(obj) > max_bytes else obj
        else:
            return _info

    return core_to_dict_OLD(obj,
                            fn_plain=lambda x: x,
                            fn_process=__object_info,
                            inc_class_name=inc_class_name,
                            inc_none_attrs=inc_none_attrs,
                            inc_none_items=inc_none_items,
                            inc_private=inc_private,
                            inc_property=inc_property,
                            max_items=max_items,
                            always_filter=always_filter,
                            never_filter=never_filter,
                            to_str=to_str,
                            fq_names=fq_names,
                            recursion_depth=recursion_depth,
                            hook_mode=hook_mode)
