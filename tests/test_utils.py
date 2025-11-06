#
# C108 - Utils Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
import sys
import types
import uuid
from dataclasses import dataclass, field

# Third party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.utils import class_name


# Tests ----------------------------------------------------------------------------------------------------------------


class TestClassName:
    @pytest.mark.parametrize(
        "obj, fully_qualified_builtins, expected",
        [
            pytest.param(int, False, "int", id="builtin-class-no-fq"),
            pytest.param(10, False, "int", id="builtin-instance-no-fq"),
            pytest.param(int, True, "builtins.int", id="builtin-class-fq"),
            pytest.param(10, True, "builtins.int", id="builtin-instance-fq"),
            pytest.param(str, False, "str", id="builtin-str-no-fq"),
            pytest.param("abc", True, "builtins.str", id="builtin-str-fq"),
            pytest.param(float, False, "float", id="builtin-float-no-fq"),
            pytest.param(3.14, True, "builtins.float", id="builtin-float-fq"),
            pytest.param(bool, False, "bool", id="builtin-bool-no-fq"),
            pytest.param(True, True, "builtins.bool", id="builtin-bool-fq"),
        ],
    )
    def test_builtin_names(self, obj, fully_qualified_builtins, expected):
        """Return correct builtin class names with and without full qualification."""
        assert class_name(obj, fully_qualified_builtins=fully_qualified_builtins) == expected

    @pytest.mark.parametrize(
        "fully_qualified_builtins, expected",
        [
            pytest.param(False, "NoneType", id="none-no-fq"),
            pytest.param(True, "builtins.NoneType", id="none-fq"),
        ],
    )
    def test_none_type(self, fully_qualified_builtins, expected):
        """Resolve None to NoneType and respect fully_qualified_builtins flag."""
        assert class_name(None, fully_qualified_builtins=fully_qualified_builtins) == expected

    @pytest.mark.parametrize(
        "as_class, fully_qualified",
        [
            pytest.param(True, True, id="class-fq"),
            pytest.param(False, True, id="instance-fq"),
            pytest.param(True, False, id="class-no-fq"),
            pytest.param(False, False, id="instance-no-fq"),
        ],
    )
    def test_user_class_fq_toggle(self, as_class, fully_qualified):
        """Return user class name respecting fully_qualified flag for class and instance."""

        class Custom:
            pass

        target = Custom if as_class else Custom()
        expected = f"{Custom.__module__}.{Custom.__name__}" if fully_qualified else Custom.__name__
        assert class_name(target, fully_qualified=fully_qualified) == expected

    def test_class_and_instance_same_base_name(self):
        """Return same base name for class and its instance."""

        class Custom:
            pass

        assert class_name(Custom) == class_name(Custom())

    def test_inherited_class_name(self):
        """Return correct name for subclass and instance."""

        class Base:
            pass

        class Sub(Base):
            pass

        assert class_name(Sub) == "Sub"
        assert class_name(Sub()) == "Sub"
        assert class_name(Sub, fully_qualified=True) == f"{Sub.__module__}.Sub"


class TestClassNameSafetyPaths:
    def test_object_without_class_attribute(self):
        """Fallback to str() for objects without __class__ attribute."""

        class NoClassAttr:
            def __getattribute__(self, name):
                if name == "__class__":
                    raise AttributeError("No __class__ attribute")
                return object.__getattribute__(self, name)

            def __str__(self):
                return "NoClassAttrObject"

        obj = NoClassAttr()
        result = class_name(obj)
        assert result == "NoClassAttrObject"

    def test_type_without_name_attribute(self):
        """Fallback to str() for types without __name__ attribute."""

        class NoNameType(type):
            @property
            def __name__(self):
                raise AttributeError("No __name__ attribute")

            def __str__(cls):
                return "CustomNoNameClass"

        class CustomClass(metaclass=NoNameType):
            pass

        result = class_name(CustomClass)
        assert result == "CustomNoNameClass"

    def test_type_without_module_attribute(self):
        """Return just the name for types without __module__ attribute."""

        class NoModuleType(type):
            @property
            def __module__(self):
                raise AttributeError("No __module__ attribute")

        class CustomClass(metaclass=NoModuleType):
            pass

        # Should return just the name without module
        result = class_name(CustomClass)
        assert result == "CustomClass"

        # fully_qualified should also just return the name
        result_fq = class_name(CustomClass, fully_qualified=True)
        assert result_fq == "CustomClass"

    def test_generic_type_list(self):
        """Handle generic list types with readable representation."""
        from typing import List

        generic_type = List[int]
        result = class_name(generic_type)
        assert result == "List[int]"

    def test_union_type(self):
        """Handle Union types with readable representation."""
        from typing import Union

        union_type = Union[int, str]
        result = class_name(union_type)
        assert result == "Union[int, str]"

    def test_optional_type(self):
        """Handle Optional types with readable representation."""
        from typing import Optional

        optional_type = Optional[int]
        result = class_name(optional_type)
        # Optional[int] is Union[int, None]
        assert result == "Optional[int]"

    def test_dict_generic_type(self):
        """Handle generic dict types with readable representation."""
        from typing import Dict

        generic_type = Dict[str, int]
        result = class_name(generic_type)
        assert result == "Dict[str, int]"

    def test_nested_generic_type(self):
        """Handle nested generic types with readable representation."""
        from typing import List, Dict

        nested_type = List[Dict[str, int]]
        result = class_name(nested_type)
        assert result == "List[Dict[str, int]]"

    def test_mock_object(self):
        """Handle mock objects correctly."""
        from unittest.mock import Mock

        mock_obj = Mock()
        result = class_name(mock_obj)
        assert result == "Mock"

    def test_c_extension_array_type(self):
        """Handle C extension array type."""
        import array

        arr = array.array("i", [1, 2, 3])
        result = class_name(arr)
        assert result == "array"

        result_fq = class_name(arr, fully_qualified=True)
        assert result_fq == "array.array"
