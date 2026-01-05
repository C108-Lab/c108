#
# C108 - Typing Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
from typing import List, Dict, Union, Optional, Any

# Third-party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.typing import validate_types


# Test Cases -----------------------------------------------------------------------------------------------------------


class TestValidateTypes:
    """Test suite for the validate_types decorator."""

    def test_validates_all_annotated_parameters(self):
        """Validate all parameters with type hints by default."""

        @validate_types
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

        @validate_types
        def func(a: int, b, c: str) -> None:
            pass

        # b can be anything since it has no annotation
        func(1, "anything", "hello")
        func(1, 123, "hello")
        func(1, None, "hello")
        func(1, [1, 2, 3], "hello")

    def test_only_validates_specified_parameters(self):
        """Validate only parameters listed in 'only' argument."""

        @validate_types(only=("a", "c"))
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

        @validate_types(skip=("b",))
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

            @validate_types(skip=("a",), only=("b",))
            def func(a: int, b: str) -> None:
                pass

    def test_invalid_parameter_names_in_only(self):
        """Raise ValueError when 'only' contains non-existent parameter names."""

        with pytest.raises(ValueError, match=r"(?i).*'only'.*invalid parameter names.*nonexistent"):

            @validate_types(only=("nonexistent",))
            def func(a: int, b: str) -> None:
                pass

    def test_invalid_parameter_names_in_skip(self):
        """Raise ValueError when 'skip' contains non-existent parameter names."""

        with pytest.raises(ValueError, match=r"(?i).*'skip'.*invalid parameter names.*nonexistent"):

            @validate_types(skip=("nonexistent",))
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

        @validate_types
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

        @validate_types
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

        @validate_types
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

        @validate_types
        def func(param: Any) -> None:
            pass

        # Any value is accepted
        func(123)
        func("hello")
        func(None)
        func([1, 2, 3])

    def test_works_with_positional_and_keyword_arguments(self):
        """Validate both positional and keyword argument calls."""

        @validate_types
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

        @validate_types
        def my_function(x: int) -> int:
            """Original docstring."""
            return x * 2

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "Original docstring."

    def test_decorator_without_parentheses(self):
        """Support decorator syntax without parentheses."""

        @validate_types
        def func(a: int) -> None:
            pass

        func(1)

        with pytest.raises(TypeError, match=r"(?i).*'a'.*expected.*int"):
            func("bad")
