#
# C108 - Tools Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
from dataclasses import dataclass, field

# Local ----------------------------------------------------------------------------------------------------------------
from c108.abc import listify
from c108.cli import cli_multiline, clify
from c108.pack import is_numbered_version, is_pep440_version, is_semantic_version
from c108.tools import print_method


# Tests ----------------------------------------------------------------------------------------------------------------

class Obj:
    a = 0
    to_dict = {"a": "zero"}


@dataclass
class DataClass:
    a = 0  # !!! <-- without type this is a class attr but NOT a dataclass field
    b: int = 1
    c: int = field(default=2)
    d: Obj = field(default_factory=Obj)


class TestTools:

    def test_cli_multiline(self):
        print_method()

        cmd = "cmd sub-cmd"
        args = ["SRC", "DEST", "-h", 1, "-q", "-xyz", "--opt=2", "--is-flag"]
        args_str = cli_multiline(args, multiline_indent=4)
        print(f"ARGS:\n'{args_str}'")
        print()
        print(f"CMD+ARGS:\n'{cmd} {args_str}'")

    def test_clify(self):
        assert clify("") == []
        assert clify(1) == ["1"]

        assert clify("abc") == ["abc"]
        assert clify("abc --help", shlex_split=True) == ["abc", "--help"]
        assert clify((1, 2, 3)) == ["1", "2", "3"]

    def test_listify(self):
        assert listify(1) == [1]
        assert listify(1, as_type=str) == ["1"]

        assert listify("abc") == ["abc"]

        assert listify([1, 2, 3]) == [1, 2, 3]
        assert listify((1, 2, 3), as_type=str) == ["1", "2", "3"]
        assert listify({1, 2, 3}) == [1, 2, 3]

    def test_is_numbered_version(self):
        assert is_numbered_version(0, min_depth=0)
        assert is_numbered_version("1", min_depth=0)
        assert is_numbered_version("1x.2y", min_depth=1)
        assert is_numbered_version("1.2rc1", min_depth=1)
        assert is_numbered_version("1.2.5-alpha", max_depth=2)
        assert not is_numbered_version("1.abc23", min_depth=1)
        assert not is_numbered_version("v1", max_depth=0)

    def test_is_pep440_version(self):
        assert is_pep440_version(0, min_depth=0)
        assert is_pep440_version("25a1")
        assert is_pep440_version("1b2")
        assert is_pep440_version("1.2.3")
        assert is_pep440_version("1.2.3.4.5")
        assert is_pep440_version("1.2.3a4")
        assert is_pep440_version("1.2.3rc4")
        assert is_pep440_version("1.2.3.post4")
        assert not is_pep440_version("1.xyz23", min_depth=1)
        assert not is_pep440_version("v1", max_depth=0)

    def test_is_semantic_version(self):
        assert is_semantic_version("1", min_depth=0)
        assert is_semantic_version("1.2", min_depth=1)
        assert is_semantic_version("1.2.3", min_depth=2)
        assert is_semantic_version("1.2.5-alpha", max_depth=7, allow_meta=True)
        assert not is_semantic_version("1.2", min_depth=2)
        assert not is_semantic_version("1.2.3.4", max_depth=2)
        assert not is_semantic_version("1.2.5-alpha", max_depth=7)

    def test_print_method(self):
        print_method()
