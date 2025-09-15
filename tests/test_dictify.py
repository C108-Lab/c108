#
# C108 - Dictify Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
from dataclasses import dataclass, field

# Third-party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.utils import class_name
from c108.dictify import core_dictify, DictifyOptions, HookMode
from c108.tools import print_title


# Classes --------------------------------------------------------------------------------------------------------------


# Tests ----------------------------------------------------------------------------------------------------------------


class TestCoreDictifyBasic:
    class Foo:
        def __init__(self, x=1):
            self.x = x

        def to_dict(self):
            return {"x": self.x}

    @pytest.mark.parametrize(
        "mode",
        [HookMode.DICT, HookMode.DICT_STRICT],
        ids=["hook_dict", "hook_dict_strict"],
    )
    def test_obj_to_dict_modes(self, mode):
        """Convert object with to_dict under both hook modes."""
        obj = self.Foo(7)
        opts = DictifyOptions(hook_mode=mode)
        result = core_dictify(obj, options=opts)
        assert result == {"x": 7}

    def test_list_of_objs(self):
        """Convert list of objects using to_dict hook."""
        items = [self.Foo(1), self.Foo(2), self.Foo(3)]
        result = core_dictify(items, options=DictifyOptions())
        assert result == [{"x": 1}, {"x": 2}, {"x": 3}]

    def test_depth_negative_returns_same(self):
        """Return object unchanged when max_depth is negative."""
        obj = self.Foo(5)
        opts = DictifyOptions(max_depth=-1)
        result = core_dictify(obj, options=opts)
        assert result is obj
