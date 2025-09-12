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
        "obj",
        [int, 10],
        ids=["builtin-class", "builtin-instance"],
    )
    def test_builtin_default(self, obj):
        """Return short name for builtins by default."""
        assert class_name(obj) == "int"

    @pytest.mark.parametrize(
        "obj",
        [int, 10],
        ids=["builtin-class", "builtin-instance"],
    )
    def test_builtin_fully_qualified_when_enabled(self, obj):
        """Return fully qualified name for builtins when enabled."""
        assert class_name(obj, fully_qualified_builtins=True) == "builtins.int"

    @pytest.mark.parametrize(
        "as_class,fully_qualified",
        [
            (True, True),
            (False, True),
            (True, False),
            (False, False),
        ],
        ids=["class-fq", "instance-fq", "class-no-fq", "instance-no-fq"],
    )
    def test_user_class_fq_toggle(self, as_class, fully_qualified):
        """Return user class name respecting fully_qualified flag for class and instance."""

        # Note: Custom is defined inside the test function; its __qualname__ will include
        # the test function name. The implementation returns module + class __name__,
        # so we assert against module + __name__ when fully_qualified is requested.
        class Custom:
            pass

        target = Custom if as_class else Custom()
        expected = (
            f"{Custom.__module__}.{Custom.__name__}"
            if fully_qualified
            else Custom.__name__
        )
        assert class_name(target, fully_qualified=fully_qualified) == expected

    def test_start_and_end_wrapping(self):
        """Wrap the resolved name with provided start and end strings."""

        class Custom:
            pass

        name = class_name(Custom, fully_qualified=False, start="<", end=">")
        assert name == f"<{Custom.__name__}>"

    @pytest.mark.parametrize(
        "fully_qualified_builtins, expected",
        [
            (False, "NoneType"),
            (True, "builtins.NoneType"),
        ],
        ids=["builtin-default", "builtin-fq"],
    )
    def test_none_type_behaviour(self, fully_qualified_builtins, expected):
        """Resolve None to NoneType and respect fully_qualified_builtins."""
        assert class_name(None, fully_qualified_builtins=fully_qualified_builtins) == expected

    def test_class_and_instance_produce_same_base_name(self):
        """Return the same base name for a class and its instance."""

        class Custom:
            pass

        assert class_name(Custom) == class_name(Custom())
