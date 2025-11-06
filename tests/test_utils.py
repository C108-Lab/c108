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
