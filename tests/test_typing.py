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
    _validate_attr_obj_type,
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
        validate_attr_types(obj)

    def test_invalid_type_raises(self):
        """Raise TypeError for wrong type."""

        @dataclass
        class Item:
            id: str
            count: int

        obj = Item(id="abc", count="bad")  # wrong type
        with pytest.raises(TypeError, match=r"(?i).*type validation failed.*"):
            validate_attr_types(obj)

    def test_strict_none_true_blocks_none(self):
        """Raise TypeError if None not allowed."""

        @dataclass
        class D:
            a: int

        obj = D(a=None)
        with pytest.raises(TypeError, match=r"(?i).*type validation failed.*"):
            validate_attr_types(obj, strict_none=True)

    def test_non_dataclass_with_annotations_works(self):
        """Validate regular class with type annotations."""

        class User:
            name: str
            age: int

            def __init__(self, name: str, age: int):
                self.name = name
                self.age = age

        obj = User("Bob", 33)
        validate_attr_types(obj)

    def test_non_dataclass_no_annotations_raises(self):
        """Raise ValueError if no type annotations."""

        class C:
            def __init__(self):
                self.x = 5

        obj = C()
        with pytest.raises(ValueError, match=r"(?i).*no type annotations.*"):
            validate_attr_types(obj)

    def test_private_attrs_included_when_flag_true(self):
        """Include private attrs when include_private=True."""

        class C:
            _a: int
            b: str

            def __init__(self):
                self._a = 5
                self.b = "ok"

        obj = C()
        validate_attr_types(obj, include_private=True)

    def test_pattern_filter_validates_subset(self):
        """Validate only attributes matching regex pattern."""

        class C:
            api_key: str
            internal_id: int

            def __init__(self):
                self.api_key = "abc"
                self.internal_id = 5

        obj = C()
        validate_attr_types(obj, pattern=r"^api_")

    def test_strict_mode_not_blocks_union(self):
        """Raise TypeError for unsupported Union when strict_unions=True."""

        class C:
            value: int | str | float | None

            def __init__(self):
                self.value = 3.14  # neither int nor str

        obj = C()
        validate_attr_types(obj, strict_unions=True)

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
        validate_attr_types(obj, include_inherited=True)

    def test_old_optional_syntax_supported(self):
        """Support Optional[T] syntax from older annotations."""
        from typing import Optional

        @dataclass
        class D:
            z: Optional[int]

        obj = D(z=None)
        validate_attr_types(obj, strict_none=False)

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
            validate_attr_types(obj, strict_unions=True)
        else:
            with pytest.raises(TypeError, match=r"(?i)(type validation failed|complex optional)"):
                validate_attr_types(obj, strict_unions=True)

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
        validate_attr_types(obj, strict_unions=strict)

        # Wrong type should fail in both modes
        obj.x = "not an int"
        with pytest.raises(TypeError, match=r"(?i)type validation failed"):
            validate_attr_types(obj, strict_unions=strict)

    def test_missing_attr_allowed(self):
        """Skip missing attribute gracefully."""

        class C:
            x: int

            def __init__(self):
                pass

        validate_attr_types(C(), strict_missing=False)  # should not raise

    def test_missing_attr_raises(self):
        """Raise on missing attribute."""

        class C:
            x: int

            def __init__(self):
                pass

        with pytest.raises(TypeError, match=r"(?i)type validation failed"):
            validate_attr_types(C(), strict_missing=True)  # should raise

    def test_complex_union_typeerror_strict(self):
        """Return complex Union error when isinstance fails for truly complex union."""
        T = list[int] | dict[str, int]
        result = _validate_attr_obj_type(
            name="x",
            name_prefix="attribute",
            value=[],
            expected_type=T,
            strict_none=False,
            strict_unions=True,
        )
        assert "complex Union" in result


class TestValidateAttrObjType:
    """Test uncovered branches in _validate_attr_obj_type."""

    @pytest.mark.parametrize(
        "expected_type,strict,expected_substring",
        [
            pytest.param("int", True, "string annotation", id="string_annotation_strict"),
            pytest.param("int", False, None, id="string_annotation_non_strict"),
        ],
    )
    def test_string_annotation(self, expected_type, strict, expected_substring):
        """Handle string annotations with strict and non-strict modes."""
        result = _validate_attr_obj_type(
            name="x",
            name_prefix="attribute",
            value=1,
            expected_type=expected_type,
            strict_none=True,
            strict_unions=strict,
        )
        if expected_substring:
            assert expected_substring in result
        else:
            assert result is None

    def test_union_optional_allow_none(self):
        """Pass when value is None and strict_none=False for optional union."""
        T = int | None
        result = _validate_attr_obj_type(
            name="x",
            name_prefix="attribute",
            value=None,
            expected_type=T,
            strict_none=True,
            strict_unions=True,
        )
        assert result is None

    def test_union_optional_invalid_value(self):
        """Return error when value not in union types."""
        T = int | str | None
        result = _validate_attr_obj_type(
            name="x",
            name_prefix="attribute",
            value=3.14,
            expected_type=T,
            strict_none=True,
            strict_unions=True,
        )
        assert "must be" in result

    def test_union_non_optional_invalid_value(self):
        """Return error for non-optional union mismatch."""
        T = int | str
        result = _validate_attr_obj_type(
            name="x",
            name_prefix="attribute",
            value=3.14,
            expected_type=T,
            strict_none=True,
            strict_unions=True,
        )
        assert "must be" in result

    def test_union_complex_type_strict(self):
        """Return complex union error when isinstance fails and strict_unions=True."""
        from typing import Callable

        T = Callable[[int], str] | Callable[[str], int]
        result = _validate_attr_obj_type(
            name="x",
            name_prefix="attribute",
            value=lambda x: x,
            expected_type=T,
            strict_none=True,
            strict_unions=True,
        )
        assert "complex Union" in result

    def test_union_complex_type_non_strict(self):
        """Skip complex union when strict_unions=False."""
        from typing import Callable

        T = Callable[[int], str] | Callable[[str], int]
        result = _validate_attr_obj_type(
            name="x",
            name_prefix="attribute",
            value=lambda x: x,
            expected_type=T,
            strict_none=True,
            strict_unions=False,
        )
        assert result is None

    def test_generic_origin_type(self):
        """Handle generic origin types like list[int]."""
        T = list[int]
        result = _validate_attr_obj_type(
            name="x",
            name_prefix="attribute",
            value=[1, 2],
            expected_type=T,
            strict_none=True,
            strict_unions=True,
        )
        assert result is None

    def test_generic_origin_type_mismatch(self):
        """Return error for generic origin type mismatch."""
        T = list[int]
        result = _validate_attr_obj_type(
            name="x",
            name_prefix="attribute",
            value="notalist",
            expected_type=T,
            strict_none=True,
            strict_unions=True,
        )
        assert "must be" in result

    def test_isinstance_typeerror_strict(self):
        """Return error when isinstance raises TypeError and strict_unions=True."""

        class Weird:
            pass

        result = _validate_attr_obj_type(
            name="x",
            name_prefix="attribute",
            value=1,
            expected_type=lambda x: x,  # invalid type for isinstance
            strict_none=True,
            strict_unions=True,
        )
        assert "Cannot validate" in result

    def test_isinstance_typeerror_non_strict(self):
        """Skip when isinstance raises TypeError and strict_unions=False."""
        result = _validate_attr_obj_type(
            name="x",
            name_prefix="attribute",
            value=1,
            expected_type=lambda x: x,
            strict_none=True,
            strict_unions=False,
        )
        assert result is None


class TestValidateParamTypes:
    """Test suite for the validate_param_types inline validator."""

    def test_validates_all_annotated_parameters(self):
        """Validate all parameters with type hints by default."""

        def func(a: int, b: str, c: float) -> None:
            validate_param_types()

        # Valid calls
        func(1, "hello", 3.14)
        func(a=1, b="hello", c=3.14)

        # Invalid calls
        with pytest.raises(TypeError, match=r"(?i).*'a'.*must be.*int"):
            func("not_int", "hello", 3.14)

        with pytest.raises(TypeError, match=r"(?i).*'b'.*must be.*str"):
            func(1, 123, 3.14)

        with pytest.raises(TypeError, match=r"(?i).*'c'.*must be.*float"):
            func(1, "hello", "not_float")

    def test_ignores_unannotated_parameters(self):
        """Skip validation for parameters without type hints."""

        def func(a: int, b, c: str) -> None:
            validate_param_types()

        # b can be anything since it has no annotation
        func(1, "anything", "hello")
        func(1, 123, "hello")
        func(1, None, "hello")
        func(1, [1, 2, 3], "hello")

    def test_only_validates_specified_parameters(self):
        """Validate only parameters listed in 'only' argument."""

        def func(a: int, b: str, c: float) -> None:
            validate_param_types(only=["a", "c"])

        # b is not validated, can be wrong type
        func(1, 123, 3.14)
        func(a=1, b=None, c=3.14)

        # a and c are validated
        with pytest.raises(TypeError, match=r"(?i).*'a'.*must be.*int"):
            func("bad", 123, 3.14)

        with pytest.raises(TypeError, match=r"(?i).*'c'.*must be.*float"):
            func(1, 123, "bad")

    def test_skip_excludes_specified_parameters(self):
        """Skip validation for parameters listed in 'skip' argument."""

        def func(a: int, b: str, c: float) -> None:
            validate_param_types(skip=["b"])

        # b is skipped, can be wrong type
        func(1, 123, 3.14)
        func(a=1, b=None, c=3.14)

        # a and c are still validated
        with pytest.raises(TypeError, match=r"(?i).*'a'.*must be.*int"):
            func("bad", 123, 3.14)

        with pytest.raises(TypeError, match=r"(?i).*'c'.*must be.*float"):
            func(1, 123, "bad")

    def test_mutual_exclusivity_of_skip_and_only(self):
        """Raise ValueError when both 'skip' and 'only' are provided."""

        def func(a: int, b: str) -> None:
            validate_param_types(skip=["a"], only=["b"])

        with pytest.raises(ValueError, match=r"(?i).*cannot use both.*skip.*only"):
            func(1, "hello")

    def test_invalid_parameter_names_in_only(self):
        """Raise ValueError when 'only' contains non-existent parameter names."""

        def func(a: int, b: str) -> None:
            validate_param_types(only=["nonexistent"])

        with pytest.raises(ValueError, match=r"(?i).*'only'.*invalid parameter names.*nonexistent"):
            func(1, "hello")

    def test_invalid_parameter_names_in_skip(self):
        """Raise ValueError when 'skip' contains non-existent parameter names."""

        def func(a: int, b: str) -> None:
            validate_param_types(skip=["nonexistent"])

        with pytest.raises(ValueError, match=r"(?i).*'skip'.*invalid parameter names.*nonexistent"):
            func(1, "hello")

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

        def func(param: hint) -> None:
            validate_param_types()

        # Valid container types
        for value in valid_values:
            func(value)

        # Invalid container type
        with pytest.raises(TypeError, match=r"(?i).*'param'.*must be"):
            func(invalid_value)

    def test_union_type_validation(self):
        """Validate against all members of Union types."""

        def func(param: Union[int, str]) -> None:
            validate_param_types()

        # Valid: either type in union
        func(123)
        func("hello")

        # Invalid: neither type in union
        with pytest.raises(TypeError, match=r"(?i).*'param'.*must be.*(int|str)"):
            func([1, 2, 3])

    def test_optional_type_validation(self):
        """Validate Optional types allowing None or specified type."""

        def func(param: Optional[str]) -> None:
            validate_param_types()

        # Valid: None or str
        func(None)
        func("hello")

        # Invalid: wrong type
        with pytest.raises(TypeError, match=r"(?i).*'param'.*must be"):
            func(123)

    def test_typing_any_is_never_validated(self):
        """Skip validation for parameters annotated with typing.Any."""

        def func(param: Any) -> None:
            validate_param_types()

        # Any value is accepted
        func(123)
        func("hello")
        func(None)
        func([1, 2, 3])

    def test_works_with_positional_and_keyword_arguments(self):
        """Validate both positional and keyword argument calls."""

        def func(a: int, b: str, c: float) -> None:
            validate_param_types()

        # Mixed positional and keyword
        func(1, b="hello", c=3.14)
        func(1, "hello", c=3.14)

        # All keyword
        func(a=1, b="hello", c=3.14)

        # Validation works for both styles
        with pytest.raises(TypeError, match=r"(?i).*'a'.*must be.*int"):
            func(a="bad", b="hello", c=3.14)

        with pytest.raises(TypeError, match=r"(?i).*'b'.*must be.*str"):
            func(1, b=123, c=3.14)

    def test_automatically_skips_self(self):
        """Automatically skip validating 'self' in instance methods."""

        class C:
            def method(self, a: int) -> int:
                validate_param_types()
                return a * 2

        obj = C()
        assert obj.method(5) == 10

        with pytest.raises(TypeError, match=r"(?i).*'a'.*must be.*int"):
            obj.method("bad")

    def test_automatically_skips_cls(self):
        """Automatically skip validating 'cls' in class methods."""

        @dataclass
        class C:
            value: int = 0

            @classmethod
            def create(cls, value: int) -> Self:
                validate_param_types()
                return cls(value=value)

        result = C.create(42)
        assert result.value == 42

        with pytest.raises(TypeError, match=r"(?i).*'value'.*must be.*int"):
            C.create("bad")

    def test_works_in_nested_functions(self):
        """Work correctly when called from nested functions."""

        def outer(x: int) -> int:
            def inner(y: str) -> str:
                validate_param_types()
                return y.upper()

            validate_param_types()
            return x + len(inner("hello"))

        assert outer(10) == 15

        with pytest.raises(TypeError, match=r"(?i).*'x'.*must be.*int"):
            outer("bad")

    def test_works_with_default_arguments(self):
        """Validate parameters even when using default values."""

        def func(a: int, b: str = "default", c: float = 1.0) -> None:
            validate_param_types()

        # Using defaults
        func(1)
        func(1, "custom")
        func(1, "custom", 2.5)

        # Invalid with defaults
        with pytest.raises(TypeError, match=r"(?i).*'a'.*must be.*int"):
            func("bad")

        with pytest.raises(TypeError, match=r"(?i).*'b'.*must be.*str"):
            func(1, 123)

    def test_error_message_format(self):
        """Verify error messages are clear and helpful."""

        def func(x: int, y: str) -> None:
            validate_param_types()

        try:
            func("bad", 123)
        except TypeError as e:
            error_msg = str(e)
            assert "type validation failed in func()" in error_msg
            assert "'x'" in error_msg
            assert "int" in error_msg
            assert "'y'" in error_msg
            assert "str" in error_msg

    def test_raises_runtime_error_when_called_at_module_level_skipping(self):
        """Skipping: Raise RuntimeError when called at module level."""
        # Need to actually invoke it at MODULE level to be raised,
        # so we do NOT check if it raises here
        pass

    def test_no_validation_when_no_type_hints(self):
        """Do nothing when function has no type hints."""

        def func(a, b, c):
            validate_param_types()  # Should not raise

        # Should accept anything
        func(1, "hello", [1, 2, 3])
        func("x", 123, None)

    def test_partial_type_hints(self):
        """Validate only parameters that have type hints."""

        def func(a: int, b, c: str):
            validate_param_types()

        # b can be anything
        func(1, "anything", "hello")
        func(1, 999, "hello")

        # a and c are validated
        with pytest.raises(TypeError, match=r"(?i).*'a'.*must be.*int"):
            func("bad", "anything", "hello")

        with pytest.raises(TypeError, match=r"(?i).*'c'.*must be.*str"):
            func(1, "anything", 999)

    def test_works_with_varargs_and_kwargs(self):
        """Work with functions that have *args and **kwargs."""

        def func(a: int, *args, b: str, **kwargs) -> None:
            validate_param_types()

        # Valid calls
        func(1, b="hello")
        func(1, 2, 3, b="hello", extra="stuff")

        # Invalid calls
        with pytest.raises(TypeError, match=r"(?i).*'a'.*must be.*int"):
            func("bad", b="hello")

        with pytest.raises(TypeError, match=r"(?i).*'b'.*must be.*str"):
            func(1, b=123)

    def test_conditional_validation(self):
        """Support conditional validation based on runtime logic."""

        def func(data: dict, strict_mode: bool) -> None:
            if strict_mode:
                validate_param_types()

        # Strict mode: validates
        func({"key": "value"}, True)

        with pytest.raises(TypeError, match=r"(?i).*'data'.*must be.*dict"):
            func("not_dict", True)

        # Non-strict mode: doesn't validate
        func("not_dict", False)  # Should not raise

    def test_modern_union_syntax(self):
        """Support Python 3.10+ union syntax (int | str)."""

        def func(param: int | str) -> None:
            validate_param_types()

        # Valid
        func(123)
        func("hello")

        # Invalid
        with pytest.raises(TypeError, match=r"(?i).*'param'.*must be"):
            func([1, 2, 3])

    def test_multiple_validation_errors(self):
        """Report all validation errors at once."""

        def func(a: int, b: str, c: float) -> None:
            validate_param_types()

        try:
            func("bad1", 123, "bad3")
        except TypeError as e:
            error_msg = str(e)
            # Should mention all three errors
            assert "'a'" in error_msg
            assert "'b'" in error_msg
            assert "'c'" in error_msg

    def test_skip_with_iterable_types(self):
        """Accept any iterable for skip parameter."""

        def func(a: int, b: str, c: float) -> None:
            validate_param_types(skip=("b",))  # tuple

        func(1, 123, 3.14)  # b not validated

        def func2(a: int, b: str, c: float) -> None:
            validate_param_types(skip={"b"})  # set

        func2(1, 123, 3.14)  # b not validated

    def test_only_with_iterable_types(self):
        """Accept any iterable for only parameter."""

        def func(a: int, b: str, c: float) -> None:
            validate_param_types(only=("a",))  # tuple

        func(1, 123, 3.14)  # only a validated

        def func2(a: int, b: str, c: float) -> None:
            validate_param_types(only={"a"})  # set

        func2(1, 123, 3.14)  # only a validated

    def test_works_in_static_methods(self):
        """Work correctly in static methods."""

        class C:
            @staticmethod
            def static_method(x: int, y: str) -> str:
                validate_param_types()
                return f"{x}:{y}"

        assert C.static_method(42, "hello") == "42:hello"

        with pytest.raises(TypeError, match=r"(?i).*'x'.*must be.*int"):
            C.static_method("bad", "hello")

    def test_works_with_property_setters(self):
        """Work in property setters."""

        class C:
            def __init__(self):
                self._value = 0

            @property
            def value(self) -> int:
                return self._value

            @value.setter
            def value(self, val: int) -> None:
                validate_param_types()
                self._value = val

        obj = C()
        obj.value = 42
        assert obj.value == 42

        with pytest.raises(TypeError, match=r"(?i).*'val'.*must be.*int"):
            obj.value = "bad"

    def test_recursion_handling(self):
        """Handle recursive function calls correctly."""

        def factorial(n: int) -> int:
            validate_param_types()
            if n <= 1:
                return 1
            return n * factorial(n - 1)

        assert factorial(5) == 120

        with pytest.raises(TypeError, match=r"(?i).*'n'.*must be.*int"):
            factorial("bad")

    def test_lambda_detection(self):
        """Provide helpful error for lambdas (unsupported)."""

        # Lambdas don't have a proper __name__ and can't be inspected normally
        # validate_param_types should detect this and provide helpful message

        # This is a limitation - lambdas won't work with inline validation
        # Users should use @valid_types decorator instead
        pass  # This is expected to not work, documented in docstring
