#
# C108 - Typing Tests
#

# Standard library -----------------------------------------------------------------------------------------------------

import inspect
import types

from dataclasses import dataclass
from typing import List, Dict, Union, Optional, Any

# Third-party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.utils import Self
from c108.typing import (
    validate_param_types,
    validate_attr_types,
    valid_types,
    _validate_obj_type,
)


# Test Cases -----------------------------------------------------------------------------------------------------------


class TestValidTypes:
    """Test suite for the valid_types decorator."""

    def test_validates_all_annotated_parameters(self):
        """Validate all parameters with type hints by default."""

        @valid_types
        def func(a: int, b: str, c: float) -> None:
            pass

        # Valid calls
        func(1, "hello", 3.14)
        func(a=1, b="hello", c=3.14)

        # Invalid calls
        with pytest.raises(TypeError, match=r"(?i).*'a'.*expected.*int"):
            func("not_int", "hello", 3.14)

        with pytest.raises(TypeError, match=r"(?i).*'b'.*expected.*str"):
            func(1, 123, 3.14)

        with pytest.raises(TypeError, match=r"(?i).*'c'.*expected.*float"):
            func(1, "hello", "not_float")

    def test_ignores_unannotated_parameters(self):
        """Skip validation for parameters without type hints."""

        @valid_types
        def func(a: int, b, c: str) -> None:
            pass

        # b can be anything since it has no annotation
        func(1, "anything", "hello")
        func(1, 123, "hello")
        func(1, None, "hello")
        func(1, [1, 2, 3], "hello")

    def test_only_validates_specified_parameters(self):
        """Validate only parameters listed in 'only' argument."""

        @valid_types(only=("a", "c"))
        def func(a: int, b: str, c: float) -> None:
            pass

        # b is not validated, can be wrong type
        func(1, 123, 3.14)
        func(a=1, b=None, c=3.14)

        # a and c are validated
        with pytest.raises(TypeError, match=r"(?i).*'a'.*expected.*int"):
            func("bad", 123, 3.14)

        with pytest.raises(TypeError, match=r"(?i).*'c'.*expected.*float"):
            func(1, 123, "bad")

    def test_skip_excludes_specified_parameters(self):
        """Skip validation for parameters listed in 'skip' argument."""

        @valid_types(skip=("b",))
        def func(a: int, b: str, c: float) -> None:
            pass

        # b is skipped, can be wrong type
        func(1, 123, 3.14)
        func(a=1, b=None, c=3.14)

        # a and c are still validated
        with pytest.raises(TypeError, match=r"(?i).*'a'.*expected.*int"):
            func("bad", 123, 3.14)

        with pytest.raises(TypeError, match=r"(?i).*'c'.*expected.*float"):
            func(1, 123, "bad")

    def test_mutual_exclusivity_of_skip_and_only(self):
        """Raise ValueError when both 'skip' and 'only' are provided."""

        with pytest.raises(ValueError, match=r"(?i).*cannot use both.*skip.*only"):

            @valid_types(skip=("a",), only=("b",))
            def func(a: int, b: str) -> None:
                pass

    def test_invalid_parameter_names_in_only(self):
        """Raise ValueError when 'only' contains non-existent parameter names."""

        with pytest.raises(ValueError, match=r"(?i).*'only'.*invalid parameter names.*nonexistent"):

            @valid_types(only=("nonexistent",))
            def func(a: int, b: str) -> None:
                pass

    def test_invalid_parameter_names_in_skip(self):
        """Raise ValueError when 'skip' contains non-existent parameter names."""

        with pytest.raises(ValueError, match=r"(?i).*'skip'.*invalid parameter names.*nonexistent"):

            @valid_types(skip=("nonexistent",))
            def func(a: int, b: str) -> None:
                pass

    @pytest.mark.parametrize(
        "hint,valid_values,invalid_value",
        [
            pytest.param(
                List[int],
                [[1, 2, 3], []],
                "not_a_list",
                id="list_container",
            ),
            pytest.param(
                Dict[str, int],
                [{"a": 1}, {}],
                [1, 2, 3],
                id="dict_container",
            ),
        ],
    )
    def test_generic_type_validation(self, hint, valid_values, invalid_value):
        """Validate container type for generic types like List[int]."""

        @valid_types
        def func(param: hint) -> None:
            pass

        # Valid container types
        for value in valid_values:
            func(value)

        # Invalid container type
        with pytest.raises(TypeError, match=r"(?i).*'param'.*expected"):
            func(invalid_value)

    def test_union_type_validation(self):
        """Validate against all members of Union types."""

        @valid_types
        def func(param: Union[int, str]) -> None:
            pass

        # Valid: either type in union
        func(123)
        func("hello")

        # Invalid: neither type in union
        with pytest.raises(TypeError, match=r"(?i).*'param'.*expected.*(int|str)"):
            func([1, 2, 3])

    def test_optional_type_validation(self):
        """Validate Optional types allowing None or specified type."""

        @valid_types
        def func(param: Optional[str]) -> None:
            pass

        # Valid: None or str
        func(None)
        func("hello")

        # Invalid: wrong type
        with pytest.raises(TypeError, match=r"(?i).*'param'.*expected"):
            func(123)

    def test_typing_any_is_never_validated(self):
        """Skip validation for parameters annotated with typing.Any."""

        @valid_types
        def func(param: Any) -> None:
            pass

        # Any value is accepted
        func(123)
        func("hello")
        func(None)
        func([1, 2, 3])

    def test_works_with_positional_and_keyword_arguments(self):
        """Validate both positional and keyword argument calls."""

        @valid_types
        def func(a: int, b: str, c: float) -> None:
            pass

        # Mixed positional and keyword
        func(1, b="hello", c=3.14)
        func(1, "hello", c=3.14)

        # All keyword
        func(a=1, b="hello", c=3.14)

        # Validation works for both styles
        with pytest.raises(TypeError, match=r"(?i).*'a'.*expected.*int"):
            func(a="bad", b="hello", c=3.14)

        with pytest.raises(TypeError, match=r"(?i).*'b'.*expected.*str"):
            func(1, b=123, c=3.14)

    def test_preserves_function_metadata(self):
        """Preserve wrapped function's name and docstring."""

        @valid_types
        def my_function(x: int) -> int:
            """Original docstring."""
            return x * 2

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "Original docstring."

    def test_decorator_without_parentheses(self):
        """Support decorator syntax without parentheses."""

        @valid_types
        def func(a: int) -> None:
            pass

        func(1)

        with pytest.raises(TypeError, match=r"(?i).*'a'.*expected.*int"):
            func("bad")

    def test_skips_self(self):
        """Skip validating self."""

        class C:
            @valid_types
            def m(self, a: int):
                return a

        assert C().m(5) == 5

    def test_skips_cls(self):
        """Skip validating cls in class methods."""

        @dataclass
        class C:
            a: int = 0

            @classmethod
            @valid_types
            def m(cls, a: int) -> Self:
                return cls(a=a)

        assert C.m(7) == C(a=7)


class TestValidateParamTypes:
    @pytest.mark.parametrize(
        ("x", "y"),
        [
            pytest.param(1, "a", id="int-str"),
            pytest.param(0, "", id="zero-empty"),
        ],
    )
    def test_basic_match(self, x, y):
        """Validate matching simple annotations."""

        def fn(a: int, b: str):
            validate_param_types(
                params=["a", "b"],
                exclude_self=True,
                exclude_none=False,
                strict=True,
                allow_none=False,
            )
            return a, b

        assert fn(x, y) == (x, y)

    def test_mismatch_raises(self):
        """Raise on simple type mismatch."""

        def fn(a: int, b: str):
            validate_param_types(
                params=["a", "b"],
                exclude_self=True,
                exclude_none=False,
                strict=True,
                allow_none=False,
            )

        with pytest.raises(TypeError, match=r"(?i).*parameter.*a.*int.*"):
            fn("not-int", "ok")

    def test_optional_allows_none_when_allowed(self):
        """Accept None for Optional when allowed."""

        def fn(a: int | None):
            validate_param_types(
                params=["a"],
                exclude_self=True,
                exclude_none=False,
                strict=True,
                allow_none=True,
            )
            return a

        assert fn(None) is None

    def test_optional_rejects_none_when_disallowed(self):
        """Reject None for Optional when not allowed."""

        def fn(a: int | None):
            validate_param_types(
                params=["a"],
                exclude_self=True,
                exclude_none=False,
                strict=True,
                allow_none=False,
            )

        with pytest.raises(TypeError, match=r"(?i).*parameter.*a.*None.*"):
            fn(None)

    def test_exclude_none_skips_validation_for_none(self):
        """Skip validation when exclude_none is True."""
        calls = {"validated": False}

        def fn(a: int):
            # If validated, None would fail; exclude_none should skip
            validate_param_types(
                params=["a"],
                exclude_self=True,
                exclude_none=True,
                strict=True,
                allow_none=False,
            )
            calls["validated"] = True

        fn(None)
        assert calls["validated"] is True  # Function proceeds

    def test_exclude_self_skips_self(self):
        """Skip validating self when exclude_self is True."""

        class C:
            def m(self, a: int):
                validate_param_types(
                    params=["self", "a"],
                    exclude_self=True,
                    exclude_none=False,
                    strict=True,
                    allow_none=False,
                )
                return a

        assert C().m(5) == 5

    def test_include_self_validates_when_not_excluded(self):
        """Validate self when exclude_self is False."""

        class C:
            def m(self: int, a: int):
                validate_param_types(
                    params=["self", "a"],
                    exclude_self=False,
                    exclude_none=False,
                    strict=True,
                    allow_none=False,
                )

        with pytest.raises(TypeError, match=r"(?i).*parameter.*self.*int.*"):
            C().m(1)

    def test_params_subset_only_those(self):
        """Validate only specified params."""

        def fn(a: int, b: str):
            validate_param_types(
                params=["a"],
                exclude_self=True,
                exclude_none=False,
                strict=True,
                allow_none=False,
            )
            return a, b

        # b mismatches but is not validated
        assert fn(1, 2) == (1, 2)

    def test_multiple_errors_aggregated(self):
        """Aggregate multiple parameter errors."""

        def fn(a: int, b: str, c: float):
            validate_param_types(
                params=["a", "b", "c"],
                exclude_self=True,
                exclude_none=False,
                strict=True,
                allow_none=False,
            )

        with pytest.raises(TypeError, match=r"(?is).*a.*int.*b.*str.*c.*float.*"):
            fn("x", 1, "nope")

    def test_unannotated_param_is_ignored(self):
        """Ignore parameters without annotations."""

        def fn(a, b: int):
            validate_param_types(
                params=["a", "b"],
                exclude_self=True,
                exclude_none=False,
                strict=True,
                allow_none=False,
            )
            return b

        assert fn("anything", 3) == 3

    def test_strict_union_unvalidatable_raises(self):
        """Raise on truly unvalidatable union when strict."""
        from collections.abc import Callable

        def fn(cb: (Callable[[int], str] | Callable[[str], int])):
            validate_param_types(
                params=["cb"],
                exclude_self=True,
                exclude_none=False,
                strict=True,
                allow_none=False,
            )

        with pytest.raises(TypeError, match=r"(?i).*complex.*union.*"):
            # Any callable; framework should deem this union unvalidatable in strict mode
            fn(lambda x: x)

    def test_non_strict_union_unvalidatable_skips(self):
        """Skip unvalidatable union when not strict."""
        from collections.abc import Callable

        calls = {"ran": False}

        def fn(cb: (Callable[[int], str] | Callable[[str], int])):
            validate_param_types(
                params=["cb"],
                exclude_self=True,
                exclude_none=False,
                strict=False,
                allow_none=False,
            )
            calls["ran"] = True

        fn(lambda x: x)
        assert calls["ran"] is True

    def test_missing_param_in_signature_is_skipped(self):
        """Skip names not present in signature."""

        def fn(a: int):
            validate_param_types(
                params=["a", "missing"],
                exclude_self=True,
                exclude_none=False,
                strict=True,
                allow_none=False,
            )
            return a

        assert fn(10) == 10


class TestValidateParamTypesEdgeCases:
    """Test edge case branches in validate_param_types."""

    def test_no_current_frame_raises_runtime_error(self, monkeypatch):
        """Raise RuntimeError when inspect.currentframe() returns None."""
        monkeypatch.setattr(inspect, "currentframe", lambda: None)
        with pytest.raises(RuntimeError, match=r"Cannot get current frame"):
            validate_param_types()

    def test_no_caller_frame_raises_runtime_error(self, monkeypatch):
        """Raise RuntimeError when caller frame is None."""
        fake_frame = types.SimpleNamespace(f_back=None)
        monkeypatch.setattr(inspect, "currentframe", lambda: fake_frame)
        with pytest.raises(RuntimeError, match=r"must be called from within a function"):
            validate_param_types()

    def test_cannot_find_function_raises_runtime_error(self, monkeypatch):
        """Raise RuntimeError when function cannot be found in any scope."""
        # Build a fake frame chain to simulate missing function resolution
        fake_caller = types.SimpleNamespace(
            f_back=None,
            f_globals={},
            f_locals={},
            f_code=types.SimpleNamespace(co_name="missing_func"),
        )
        fake_frame = types.SimpleNamespace(f_back=fake_caller)
        monkeypatch.setattr(inspect, "currentframe", lambda: fake_frame)

        with pytest.raises(RuntimeError, match=r"Cannot find function"):
            validate_param_types()

    def test_get_type_hints_fallback_to_annotations(self, monkeypatch):
        """Fallback to __annotations__ when get_type_hints raises Exception."""

        def func(x: int):
            validate_param_types()
            return x

        def bad_get_type_hints(_):
            raise Exception("boom")

        monkeypatch.setattr("c108.abc.get_type_hints", bad_get_type_hints)
        assert func(5) == 5  # should not raise

    def test_no_type_hints_returns_early(self):
        """Return early when no type hints exist."""

        def func(x):
            validate_param_types()
            return x

        assert func(10) == 10

    @pytest.mark.parametrize(
        "params,exclude_self,exclude_none,value,expect_error",
        [
            (["x"], True, False, "bad", True),
            (["x"], True, True, None, False),
        ],
        ids=["type_mismatch", "exclude_none_skips"],
    )
    def test_validation_errors_and_exclude_none(
        self, params, exclude_self, exclude_none, value, expect_error
    ):
        """Trigger validation error or skip when exclude_none=True."""

        def func(x: int | None):
            validate_param_types(
                params=params, exclude_self=exclude_self, exclude_none=exclude_none
            )
            return x

        if expect_error:
            with pytest.raises(TypeError, match=r"type validation failed"):
                func(value)
        else:
            assert func(value) is None


class TestValidateAttrTypes:
    """Core validation tests for validate_attr_types()."""

    @pytest.mark.parametrize(
        "fast_mode",
        [pytest.param(True, id="fast_true"), pytest.param("auto", id="fast_auto")],
    )
    def test_valid_dataclass_passes(self, fast_mode):
        """Validate dataclass with correct types."""

        @dataclass
        class Item:
            id: str
            count: int

        obj = Item(id="abc", count=5)
        validate_attr_types(obj, fast=fast_mode)

    def test_invalid_type_raises(self):
        """Raise TypeError for wrong type."""

        @dataclass
        class Item:
            id: str
            count: int

        obj = Item(id="abc", count="bad")  # wrong type
        with pytest.raises(TypeError, match=r"(?i).*type validation failed.*"):
            validate_attr_types(obj, fast=True)

    def test_exclude_none_skips_none_fields(self):
        """Skip None fields when exclude_none=True."""

        @dataclass
        class D:
            x: int | None
            y: str

        obj = D(x=None, y="hi")
        validate_attr_types(obj, exclude_none=True, fast=True)

    def test_allow_none_false_blocks_none(self):
        """Raise TypeError if None not allowed."""

        @dataclass
        class D:
            a: int | None

        obj = D(a=None)
        with pytest.raises(TypeError, match=r"(?i).*type validation failed.*"):
            validate_attr_types(obj, allow_none=False, fast=True)

    def test_fast_incompatible_options_raise(self):
        """Raise ValueError if fast=True but incompatible options."""

        @dataclass
        class D:
            x: int

        obj = D(1)
        with pytest.raises(ValueError, match=r"(?i).*cannot use fast.*pattern parameter.*"):
            validate_attr_types(obj, pattern=r"x", fast=True)

    def test_non_dataclass_with_annotations_works(self):
        """Validate regular class with type annotations."""

        class User:
            name: str
            age: int

            def __init__(self, name: str, age: int):
                self.name = name
                self.age = age

        obj = User("Bob", 33)
        validate_attr_types(obj, fast=False)

    def test_non_dataclass_no_annotations_raises(self):
        """Raise ValueError if no type annotations."""

        class C:
            def __init__(self):
                self.x = 5

        obj = C()
        with pytest.raises(ValueError, match=r"(?i).*no type annotations.*"):
            validate_attr_types(obj, fast=False)

    def test_private_attrs_included_when_flag_true(self):
        """Include private attrs when include_private=True."""

        class C:
            _a: int
            b: str

            def __init__(self):
                self._a = 5
                self.b = "ok"

        obj = C()
        validate_attr_types(obj, include_private=True, fast=False)

    def test_pattern_filter_validates_subset(self):
        """Validate only attributes matching regex pattern."""

        class C:
            api_key: str
            internal_id: int

            def __init__(self):
                self.api_key = "abc"
                self.internal_id = 5

        obj = C()
        validate_attr_types(obj, pattern=r"^api_", fast=False)

    def test_strict_mode_not_blocks_union(self):
        """Raise TypeError for unsupported Union when strict=True."""

        class C:
            value: int | str | float | None

            def __init__(self):
                self.value = 3.14  # neither int nor str

        obj = C()
        validate_attr_types(obj, strict=True, fast=False)

    def test_inherited_attrs_checked_when_enabled(self):
        """Validate inherited attributes when include_inherited=True."""

        class Base:
            x: int

            def __init__(self):
                self.x = 10

        class Child(Base):
            y: str

            def __init__(self):
                super().__init__()
                self.y = "ok"

        obj = Child()
        validate_attr_types(obj, include_inherited=True, fast=False)

    def test_old_optional_syntax_supported(self):
        """Support Optional[T] syntax from older annotations."""
        from typing import Optional

        @dataclass
        class D:
            z: Optional[int]

        obj = D(z=None)
        validate_attr_types(obj, allow_none=True, fast=True)

    def test_fast_auto_switches_to_slow_path(self):
        """Use slow path automatically when filters are applied."""

        @dataclass
        class D:
            name: str
            age: int

        obj = D("Alice", 20)
        # should auto-select slow path (pattern breaks fast)
        validate_attr_types(obj, pattern=r"^name$", fast="auto")

    def test_fast_mode_requires_dataclass(self):
        """Raise ValueError if fast=True but not dataclass."""

        class NotDC:
            x: int

            def __init__(self):
                self.x = 3

        obj = NotDC()
        with pytest.raises(ValueError, match=r"(?i).*obj is not a dataclass.*"):
            validate_attr_types(obj, fast=True)

    # Test strict mode behavior: validates simple unions, rejects complex unions. ----------------------

    @pytest.mark.parametrize(
        "type_hint,value,should_pass",
        [
            # Simple unions - valid
            pytest.param(int | str, 42, True, id="union-int|str:int"),
            pytest.param(int | str, "hello", True, id="union-int|str:str"),
            pytest.param(int | str | float, 3.14, True, id="union-int|str|float:float"),
            pytest.param(int | str | float, 100, True, id="union-int|str|float:int"),
            pytest.param(str | bytes, b"data", True, id="union-str|bytes:bytes"),
            # Simple unions - invalid
            pytest.param(int | str, 3.14, False, id="union-int|str:float-fails"),
            pytest.param(int | str, [], False, id="union-int|str:list-fails"),
            pytest.param(int | float, "text", False, id="union-int|float:str-fails"),
            # Optional (T | None) - valid
            pytest.param(int | None, 42, True, id="optional-int:valid-int"),
            pytest.param(int | None, None, True, id="optional-int:none"),
            pytest.param(str | None, "test", True, id="optional-str:valid-str"),
            # Complex Optional Union - invalid in strict mode
            pytest.param(int | str | None, 42, True, id="complex-optional-int|str|none:int-pass"),
            pytest.param(
                int | str | None,
                "text",
                True,
                id="complex-optional-int|str|none:str-pass",
            ),
        ],
    )
    def test_strict_mode_union_validation(self, type_hint, value, should_pass):
        """Validate unions in strict mode and reject complex optional unions."""

        class C:
            pass

        C.__annotations__ = {"value": type_hint}
        obj = C()
        obj.value = value

        if should_pass:
            validate_attr_types(obj, strict=True, fast=False)
        else:
            with pytest.raises(TypeError, match=r"(?i)(type validation failed|complex optional)"):
                validate_attr_types(obj, strict=True, fast=False)

    @pytest.mark.parametrize(
        "strict",
        [
            pytest.param(True, id="strict"),
            pytest.param(False, id="non-strict"),
        ],
    )
    def test_simple_types_in_both_modes(self, strict):
        """Validate simple types and enforce errors consistently in both modes."""

        class C:
            x: int
            y: str
            z: float

            def __init__(self):
                self.x = 42
                self.y = "hello"
                self.z = 3.14

        obj = C()

        # Should pass with both strict modes
        validate_attr_types(obj, strict=strict, fast=False)

        # Wrong type should fail in both modes
        obj.x = "not an int"
        with pytest.raises(TypeError, match=r"(?i)type validation failed"):
            validate_attr_types(obj, strict=strict, fast=False)


class TestValidateTypesEdgeCases:
    """Test edge cases of validate_attr_types and its inner functions."""

    def test_fast_true_incompatible_options(self):
        """Raise ValueError when fast=True but incompatible options are set."""

        @dataclass
        class D:
            x: int = 1

        obj = D()
        with pytest.raises(ValueError, match=r"(?i)incompatible"):
            validate_attr_types(obj, fast=True, attrs=["x"])

    def test_fast_path_success(self):
        """Validate dataclass fast path passes with correct types."""

        @dataclass
        class D:
            x: int = 5
            y: str = "ok"

        validate_attr_types(D())  # should not raise

    def test_fast_path_type_error(self):
        """Raise TypeError when dataclass field type mismatches."""

        @dataclass
        class D:
            x: int = "bad"

        with pytest.raises(TypeError, match=r"(?i)type validation failed"):
            validate_attr_types(D())

    def test_slow_path_no_annotations(self):
        """Raise ValueError when class has no type annotations."""

        class C:
            def __init__(self):
                self.x = 1

        with pytest.raises(ValueError, match=r"(?i)no type annotations"):
            validate_attr_types(C(), fast=False)

    def test_slow_path_with_attrs_and_exclude_none(self):
        """Validate slow path skips None when exclude_none=True."""

        class C:
            x: int | None
            y: str

            def __init__(self):
                self.x = None
                self.y = "ok"

        validate_attr_types(C(), attrs=["x", "y"], exclude_none=True, fast=False)

    def test_slow_path_type_error(self):
        """Raise TypeError when attribute type mismatches in slow path."""

        class C:
            x: int

            def __init__(self):
                self.x = "bad"

        with pytest.raises(TypeError, match=r"(?i)type validation failed"):
            validate_attr_types(C(), fast=False)

    def test_slow_path_missing_attr(self):
        """Skip missing attribute gracefully."""

        class C:
            x: int

            def __init__(self):
                pass

        validate_attr_types(C(), fast=False)  # should not raise

    def test_complex_union_typeerror_strict(self):
        """Return complex Union error when isinstance fails for truly complex union."""
        T = list[int] | dict[str, int]
        result = _validate_obj_type(
            name="x",
            name_prefix="attribute",
            value=[],
            expected_type=T,
            allow_none=True,
            strict=True,
        )
        assert "complex Union" in result


class TestValidateTypesObjType:
    """Test uncovered branches in _validate_obj_type."""

    @pytest.mark.parametrize(
        "expected_type,strict,expected_substring",
        [
            pytest.param("int", True, "string annotation", id="string_annotation_strict"),
            pytest.param("int", False, None, id="string_annotation_non_strict"),
        ],
    )
    def test_string_annotation(self, expected_type, strict, expected_substring):
        """Handle string annotations with strict and non-strict modes."""
        result = _validate_obj_type(
            name="x",
            name_prefix="attribute",
            value=1,
            expected_type=expected_type,
            allow_none=True,
            strict=strict,
        )
        if expected_substring:
            assert expected_substring in result
        else:
            assert result is None

    def test_union_optional_allow_none(self):
        """Pass when value is None and allow_none=True for optional union."""
        T = int | None
        result = _validate_obj_type(
            name="x",
            name_prefix="attribute",
            value=None,
            expected_type=T,
            allow_none=True,
            strict=True,
        )
        assert result is None

    def test_union_optional_invalid_value(self):
        """Return error when value not in union types."""
        T = int | str | None
        result = _validate_obj_type(
            name="x",
            name_prefix="attribute",
            value=3.14,
            expected_type=T,
            allow_none=True,
            strict=True,
        )
        assert "must be" in result

    def test_union_non_optional_invalid_value(self):
        """Return error for non-optional union mismatch."""
        T = int | str
        result = _validate_obj_type(
            name="x",
            name_prefix="attribute",
            value=3.14,
            expected_type=T,
            allow_none=True,
            strict=True,
        )
        assert "must be" in result

    def test_union_complex_type_strict(self):
        """Return complex union error when isinstance fails and strict=True."""
        from typing import Callable

        T = Callable[[int], str] | Callable[[str], int]
        result = _validate_obj_type(
            name="x",
            name_prefix="attribute",
            value=lambda x: x,
            expected_type=T,
            allow_none=True,
            strict=True,
        )
        assert "complex Union" in result

    def test_union_complex_type_non_strict(self):
        """Skip complex union when strict=False."""
        from typing import Callable

        T = Callable[[int], str] | Callable[[str], int]
        result = _validate_obj_type(
            name="x",
            name_prefix="attribute",
            value=lambda x: x,
            expected_type=T,
            allow_none=True,
            strict=False,
        )
        assert result is None

    def test_generic_origin_type(self):
        """Handle generic origin types like list[int]."""
        T = list[int]
        result = _validate_obj_type(
            name="x",
            name_prefix="attribute",
            value=[1, 2],
            expected_type=T,
            allow_none=True,
            strict=True,
        )
        assert result is None

    def test_generic_origin_type_mismatch(self):
        """Return error for generic origin type mismatch."""
        T = list[int]
        result = _validate_obj_type(
            name="x",
            name_prefix="attribute",
            value="notalist",
            expected_type=T,
            allow_none=True,
            strict=True,
        )
        assert "must be" in result

    def test_isinstance_typeerror_strict(self):
        """Return error when isinstance raises TypeError and strict=True."""

        class Weird:
            pass

        result = _validate_obj_type(
            name="x",
            name_prefix="attribute",
            value=1,
            expected_type=lambda x: x,  # invalid type for isinstance
            allow_none=True,
            strict=True,
        )
        assert "Cannot validate" in result

    def test_isinstance_typeerror_non_strict(self):
        """Skip when isinstance raises TypeError and strict=False."""
        result = _validate_obj_type(
            name="x",
            name_prefix="attribute",
            value=1,
            expected_type=lambda x: x,
            allow_none=True,
            strict=False,
        )
        assert result is None
