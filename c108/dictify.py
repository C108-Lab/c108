"""
C108 Dictify Tools
"""

# Standard library -----------------------------------------------------------------------------------------------------
import collections.abc as abc
import inspect

from copy import copy
from enum import Enum, unique
from dataclasses import dataclass, field, replace as dataclasses_replace
from typing import Any, Dict, Callable, Iterable, Type

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


def _default_type_handlers() -> Dict[Type, Callable]:
    """
    Get default type handlers for common filtered types.

    These handlers implement the "always filtered" behavior mentioned in the docs
    for str, bytes, bytearray, and memoryview types.

    Returns:
        Dictionary mapping types to their default handler functions
    """
    return {
        str: _handle_str,
        bytes: _handle_bytes,
        bytearray: _handle_bytearray,
        memoryview: _handle_memoryview,
        # TODO range: _handle_range,
    }


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
        max_str_length: Maximum length for string values (truncated if exceeded)
        max_bytes: Maximum size for 'bytes' object (truncated if exceeded)
        sort_keys: Mappings key ordering

        hook_mode: Object conversion strategy - "dict" (try to_dict() then fallback),
                   "dict_strict" (require to_dict() method), or "none" (skip object hooks)
        fully_qualified_names: Use fully qualified class names (module.Class vs Class)
        skip_types: Types that skip filtering and preserve original values
        type_handlers: Custom handlers for specific types. If None, uses default handlers
                       for str, bytes, bytearray, memoryview. Handlers receive `(obj, options: DictifyOptions)`
                       and return the processed result.

    Examples:
        >>> # Basic usage with defaults
        >>> options = DictifyOptions()

        >>> # Debugging configuration
        >>> debug_opts = DictifyOptions(max_depth=1, include_private=True, max_items=50)

        >>> # Serialization configuration
        >>> serial_opts = DictifyOptions(include_class_name=True, include_none_attrs=False, max_str_length=100)

        >>> # Custom type handlers with defaults
        >>> import socket, threading
        >>> options = (
        ...     DictifyOptions()
        ...     .add_type_handler(socket.socket, (lambda s, opts: {"type": "socket", "closed": s._closed}))
        ...     .add_type_handler(threading.Thread, (lambda t, opts: {"name": t.name, "alive": t.is_alive()}))
        ... )

        >>> # Completely custom handlers (no defaults)
        >>> custom_only = (
        ...     DictifyOptions(type_handlers={})
        ...     .add_type_handler(str, lambda s, opts: s.upper())
        ...     .add_type_handler(dict, lambda d, opts: f"<dict with {len(d)} items>")
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
    max_str_length: int = 240
    max_bytes: int = 1024

    # Mapping keys ordering
    sort_keys: bool = False

    # Advanced
    hook_mode: str = HookMode.DICT
    fully_qualified_names: bool = False
    skip_types: tuple[type, ...] = (int, float, bool, complex, type(None))

    # Type-specific handlers
    type_handlers: Dict[Type, Callable[[Any, 'DictifyOptions'], Any]] = field(
        default_factory=_default_type_handlers
    )

    def add_type_handler(
            self,
            typ: type,
            handler: Callable[[Any, "DictifyOptions"], Any],
    ) -> "DictifyOptions":
        """
        Register or override a handler for a specific type.

        Args:
            typ: The concrete type to process.
            handler: A callable receiving (obj, options) and returning processed value.

        Returns:
            Self, to allow chaining.

        Raises:
            TypeError: If typ is not a type or handler is not callable.
            ValueError: If internal type_handlers is not a dict.
        """
        if not isinstance(self.type_handlers, dict):
            raise ValueError(f"type_handlers must be a dict but {fmt_type(self.type_handlers)} found")
        if not isinstance(typ, type):
            raise TypeError(f"typ must be a type, got {fmt_type(typ)}")
        if not callable(handler):
            raise TypeError(f"handler must be callable, got {fmt_type(handler)}")

        # Register or override handler for the given type
        self.type_handlers[typ] = handler
        return self

    def get_type_handlers(self) -> dict[type, Callable[[Any, 'DictifyOptions'], Any]]:
        """
        Retrieves a copy of the defined type handlers.

        Raises:
            TypeError: If `type_handlers` is not a dictionary.

        Returns:
            dict: A shallow copy of the `type_handlers` dictionary.
        """
        if isinstance(self.type_handlers, dict):
            return {**self.type_handlers}
        else:
            raise TypeError(f"type_handlers must be a dict but {fmt_type(self.type_handlers)} found")


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
        fn_raw: Handler for raw/minimal processing mode (max_depth < 0). Defaults to None,
                uses fallback chain obj.to_dict() → identity function.
        fn_terminal: Handler for when recursion depth is exhausted (max_depth = 0). Defaults to None,
                     uses fallback chain type_handlers → obj.to_dict() → identity function.

    Returns:
        Human-readable data representation of the object

    Raises:
        TypeError: If options, fn_raw, or fn_terminal have invalid types
        TypeError: If hook_mode is 'dict_strict' and object lacks to_dict() method
        ValueError: If hook_mode contains invalid value

    Examples:
        >>> # Basic usage with custom options
        >>> opts = DictifyOptions(max_depth=5, include_private=True)
        >>> result = core_dictify(obj, options=opts)

        >>> # With custom processing functions
        >>> def custom_terminal(x):
        ...     return f"<truncated: {type(x).__name__}>"
        >>> result = core_dictify(obj, options=opts, fn_terminal=custom_terminal)

    TODO update tests accordingly
    Precedence of handling rules:
        - Raw mode (max_depth < 0): fn_raw() > obj.to_dict() > identity function
        - Terminal mode due to depth exhaustion (max_depth == 0):
          fn_terminal() > type_handlers > obj.to_dict() > identity
        - Normal recursion (max_depth > 0): type_handlers > obj.to_dict() > recursive expansion

    Note:
        - max_depth < 0: Returns fn_raw() or fallback chain results
        - max_depth = 0: Returns fn_terminal() or fallback chain results
        - max_depth = N: Recurses N levels deep into collections; objects expand to dicts with attrs processed at depth N-1.
        - Builtins, which are never filtered: int, float, bool, complex, None, range
        - Builtins filtered with default type_handlers: str, bytes, bytearray, memoryview
        - Never-filtered objects are returned as-is, custom handlers not applicable
        - The type_handlers precede over object's to_dict() method
        - Properties that raise exceptions are automatically skipped
        - Class name include (if enabled) only affects main recursive processing, and optionally
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

    def __fn_raw_chain(obj: Any, opt: DictifyOptions) -> Any:
        """
        fn_raw chain of object processors with priority order
        fn_raw() > obj.to_dict() > identity function
        """
        opt = opt or DictifyOptions()
        if fn_raw is not None:
            return fn_raw(obj)
        if dict_ := _get_from_to_dict(obj, opt=opt) is not None:
            return dict_
        return object  # Final fallback

    def __fn_terminal_chain(obj: Any, opt: DictifyOptions) -> Any:
        """
        fn_terminal chain of object processors with priority order
        fn_terminal() > type_handlers > obj.to_dict() > identity function
        """
        opt = opt or DictifyOptions()
        if fn_terminal is not None:
            return fn_terminal(obj)
        if type_handler := _get_type_handler(obj, opt):
            return type_handler(obj, opt)
        if dict_ := _get_from_to_dict(obj, opt=opt) is not None:
            return dict_
        return obj  # Final fallback

    def __process_max_items(obj: abc.Collection[Any], opt: DictifyOptions) -> Any:
        """
        Processor for collection which have crossed max_items limit, falls back
        to str representation
        """
        if not _implements_iter(obj):
            raise TypeError(f"obj must implement __iter__ method, but found {fmt_type(obj)}")
        if not _implements_len(obj):
            raise TypeError(f"obj must implement __len__ method, but found {fmt_type(obj)}")

        opt = opt or DictifyOptions()
        items_count = len(obj)
        as_str = f"{_to_str(obj, opt=opt)} {items_count} items"
        return as_str  # Final fallback

    def __process_collection(obj: abc.Collection[Any],
                             rec_depth: int,
                             opt: DictifyOptions,
                             source_object: Any = None) -> Any:
        """
        Process items recursively in a collection which implements __len__ method
        TODO MappingView support?
        """
        if not _implements_iter(obj):
            raise TypeError(f"obj must implement __iter__ method, but found {fmt_type(obj)}")
        if not _implements_len(obj):
            raise TypeError(f"obj must implement __len__ method, but found {fmt_type(obj)}")

        if rec_depth < 0:
            raise OverflowError(f"Collection recursion depth out of range: {rec_depth}")

        if rec_depth == 0:
            return __fn_terminal_chain(obj, opt=opt)

        if len(obj) > opt.max_items:
            return __process_max_items(obj, opt=opt)

        if isinstance(obj, (abc.Sequence, abc.Set)):
            return type(obj)(__core_dictify(item, recursion_depth=rec_depth - 1, opt=opt) for item in obj)

        elif isinstance(obj, (dict, abc.Mapping)):
            inc_nones = opt.include_none_attrs if source_object else opt.include_none_items
            dict_ = {k: __core_dictify(v, recursion_depth=(rec_depth - 1), opt=opt) for k, v in obj.items()
                     if (v is not None) or inc_nones}
            if opt.include_class_name:
                if source_object is None:
                    dict_["__class__"] = _class_name(obj, options=opt)
                else:
                    dict_["__class__"] = _class_name(source_object, options=opt)
            if opt.sort_keys:
                dict_ = dict(sorted(dict_.items()))
            return dict_
        else:
            raise TypeError(f"Items container must be a Sequence, Mapping or Set but found: {fmt_type(obj)}")

    # core_dictify() body Start ---------------------------------------------------------------------------

    # Return skip_type objects as is -----------------
    if _is_skip_type(obj, options=opt):
        return obj

    # Edge Cases processing --------------------------
    if opt.max_depth < 0:
        return __fn_raw_chain(obj, opt=opt)

    if opt.max_depth == 0:
        return __fn_terminal_chain(obj, opt=opt)

    # Type handling and obj.to_dict() processors -----
    if type_handler := _get_type_handler(obj, opt=opt):
        return type_handler(obj, opt)

    if dict_ := _get_from_to_dict(obj, opt=opt):
        return dict_

    # Should check item-based Instances for recursion: list, tuple, set, dict, etc
    if isinstance(obj, (abc.Sequence, abc.Mapping, abc.Set)):
        if not _implements_len(obj):
            return __fn_terminal_chain(obj, opt=opt)
        else:
            return __process_collection(obj, rec_depth=opt.max_depth, opt=opt, source_object=None)

    # We should arrive here only when max_depth > 0 (recursion not exhausted)
    # Should expand obj 1 level into deep.
    dict_ = _shallow_to_dict(obj, opt=opt)
    return __process_collection(dict_, rec_depth=opt.max_depth, opt=opt, source_object=obj)

    # core_dictify() body End ---------------------------------------------------------------------------


def _get_type_handler(obj: Any, opt: DictifyOptions) -> abc.Callable[[Any, DictifyOptions], Any] | None:
    """
    Get the handler function for the object's type (exact or via inheritance).

    Prefers the closest matching type handler via MRO; fallback to mapping order for ABCs.

    Args:
        obj: Object to potentially handle.
        options: DictifyOptions instance.

    Returns:
        The handler function if found; otherwise None.

    Raises:
        ValueError: If options or options.type_handlers are of incorrect types.
    """
    if not isinstance(opt, DictifyOptions):
        raise TypeError(f"options must be a DictifyOptions but found {fmt_type(opt)}")
    if not isinstance(opt.type_handlers, abc.Mapping):
        raise TypeError(f"type_handlers must be a Mapping but found {fmt_type(opt.type_handlers)}")

    obj_type = type(obj)
    type_handlers = opt.type_handlers

    # Fast path: exact type match
    if obj_type in type_handlers:
        return type_handlers[obj_type]

    # Build candidates that are supertypes of obj_type (robust to non-type keys)
    handler_type_keys = [k for k in type_handlers.keys() if isinstance(k, type)]
    candidates: list[type] = []
    for handler_type in handler_type_keys:
        try:
            if handler_type is not obj_type and issubclass(obj_type, handler_type):
                candidates.append(handler_type)
        except TypeError:
            # Skip keys that aren't valid types
            continue

    # Prefer the nearest ancestor using the MRO
    if candidates:
        for base in obj_type.__mro__[1:]:
            if base in candidates:
                return type_handlers[base]

        # Fallback: mapping order for candidates not present in MRO (e.g., ABC registrations)
        for k in type_handlers.keys():
            if k in candidates:
                return type_handlers[k]

    # Final defensive pass: isinstance over all keys (supports tuple-of-types keys)
    for key, handler in type_handlers.items():
        try:
            if isinstance(obj, key):
                return handler
        except TypeError:
            continue

    return None


def _class_name(obj: Any, options: DictifyOptions) -> str:
    return class_name(obj,
                      fully_qualified=options.fully_qualified_names,
                      fully_qualified_builtins=False,
                      start="", end="")


def _get_from_to_dict(obj, opt: DictifyOptions | None = None) -> dict[Any, Any] | None:
    """Returns obj.to_dict() value if the method is available and hook mode allows"""

    opt = opt or DictifyOptions()

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

    # Check the returned type -----------------------------------
    if dict_ is not None:
        if not isinstance(dict_, abc.Mapping):
            raise TypeError(f"Object's to_dict() must return a Mapping, but got {fmt_type(dict_)}")

        # returned mapping should be of dict type
        dict_ = {**dict_}
        if opt.inject_class_name:
            dict_["__class__"] = _class_name(obj, options=opt)
        if opt.sort_keys:
            dict_ = dict(sorted(dict_.items()))

    return dict_


def _implements_iter(obj: Any) -> bool:
    """Return True if `obj` can be iterated with iter()."""
    try:
        iter(obj)
    except TypeError:
        return False
    else:
        return True


def _implements_len(obj: abc.Collection[Any]) -> bool:
    """Returns True if obj implements __len__"""
    try:
        len(obj)
        return True
    except TypeError:
        return False


def _is_skip_type(obj: Any, options: DictifyOptions) -> bool:
    """
    Check obj against types skipped in processing
    """
    if isinstance(obj, tuple(options.skip_types)):
        return True
    else:
        return False


def _handle_str(obj: str, options: DictifyOptions) -> str:
    """Default handler for str objects - applies max_str_length limit."""
    if len(obj) > options.max_str_length:
        return obj[:options.max_str_length] + "..."
    return obj


def _handle_bytes(obj: bytes, options: DictifyOptions) -> bytes:
    """Default handler for bytes objects - applies max_bytes limit."""
    if len(obj) > options.max_bytes:
        return obj[:options.max_bytes] + b"..."
    return obj


def _handle_bytearray(obj: bytearray, options: DictifyOptions) -> bytearray:
    """Default handler for bytearray objects - applies max_bytes limit."""
    if len(obj) > options.max_bytes:
        truncated = obj[:options.max_bytes]
        return bytearray(truncated + b"...")
    return obj


def _handle_memoryview(obj: memoryview, options: DictifyOptions) -> dict[str, Any]:
    """Default handler for memoryview objects - converts to descriptive dictionary."""
    result = {
        'type': 'memoryview',
        'nbytes': len(obj),
        'format': obj.format,
        'readonly': obj.readonly,
        'itemsize': obj.itemsize,
    }

    # Add shape info if available (for multi-dimensional views)
    if hasattr(obj, 'shape') and obj.shape is not None:
        result['shape'] = obj.shape
        result['ndim'] = obj.ndim
        result['strides'] = obj.strides if hasattr(obj, 'strides') else None

    # Add data preview if reasonably sized
    if len(obj) <= options.max_bytes:
        result['data'] = obj.tobytes()
    elif options.max_bytes > 0:
        preview_size = min(options.max_bytes // 2, 64)
        try:
            preview_data = obj[:preview_size].tobytes()
            result['data_preview'] = preview_data
            result['data_truncated'] = True
        except (ValueError, TypeError):
            result['data_preview'] = None
            result['data_truncated'] = True

    return result


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


def _to_str(obj: Any, opt: DictifyOptions) -> str:
    """
    Returns custom <str> value of object.

    If custom __str__ method not found, overrides the stdlib __str__ with optionally Fully Qualified Class Name
    """
    has_default_str = obj.__str__ == object.__str__
    if not has_default_str:
        as_str = str(obj)
    else:
        cls_name = class_name(obj, fully_qualified=opt.fully_qualified_names,
                              fully_qualified_builtins=False,
                              start="", end="")
        as_str = f"<class {cls_name}>"
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
        >>> p = Person("Alice", 7)
        >>> dictify(p)
        {'name': 'Alice', 'age': 7}

        # Include class information for debugging
        >>> dictify(p, include_class_name=True)
        {'__class__': 'Person', 'name': 'Alice', 'age': 30}

        # Control recursion depth
        >>> nested = {'users': [p], 'count': 1}
        >>> dictify(nested, max_depth=1)
        {'users': [{'name': 'Alice', 'age': 30}], 'count': 1}

        # Advanced configuration with options
        >>> from c108.dictify import DictifyOptions
        >>> opts = DictifyOptions(include_none_attrs=False, max_str_length=20)
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
                         inject_class_name=include_class_name,
                         max_items=max_items,
                         )

    def __dictify_process(obj: Any) -> Any:
        # Return never-filtered objects
        if _is_skip_type(obj, options=opt):
            return obj
        # Should convert to dict the topmost obj level but keep its inner objects as-is
        dict_ = _shallow_to_dict(obj, opt=opt)
        return dict_

    return core_dictify(obj,
                        fn_raw=lambda x: x,
                        fn_terminal=__dictify_process,
                        options=opt)


# ---------------------------------------------------------------

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
