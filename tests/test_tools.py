#
# C108 - Tools Tests
#

# Standard library -----------------------------------------------------------------------------------------------------
from dataclasses import dataclass, field

# Third-party ----------------------------------------------------------------------------------------------------------
import pytest

# Local ----------------------------------------------------------------------------------------------------------------
from c108.cli import cli_multiline, clify
from c108.pack import is_numbered_version, is_pep440_version, is_semantic_version
from c108.tools import fmt_mapping, fmt_sequence, fmt_value
from c108.tools import print_method, listify, dict_get, dict_set


# Classes --------------------------------------------------------------------------------------------------------------

class Obj:
    a = 0
    to_dict = {"a": "zero"}


@dataclass
class DataClass:
    a = 0  # !!! <-- without type this is a class attr but NOT a dataclass field
    b: int = 1
    c: int = field(default=2)
    d: Obj = field(default_factory=Obj)


# Tests ----------------------------------------------------------------------------------------------------------------

class TestFormatSuite:

    # ---------- fmt_mapping ----------

    def test_fmt_mapping_basic(self):
        mp = {"a": 1, 2: "b"}
        out = fmt_mapping(mp, style="ascii")
        # Insertion order preserved by dicts
        assert out == "{<str: 'a'>: <int: 1>, <int: 2>: <str: 'b'>}"

    def test_fmt_mapping_with_nested_sequence(self):
        mp = {"k": [1, 2]}
        out = fmt_mapping(mp, style="unicode-angle")
        assert out == "{⟨str: 'k'⟩: [⟨int: 1⟩, ⟨int: 2⟩]}"

    @pytest.mark.parametrize(
        "style, expected_more",
        [
            ("ascii", "..."),
            ("unicode-angle", "…"),
        ],
    )
    def test_fmt_mapping_max_items_appends_ellipsis(self, style, expected_more):
        mp = {i: i for i in range(5)}
        out = fmt_mapping(mp, style=style, max_items=3)
        assert out.endswith(expected_more + "}")

    def test_fmt_mapping_custom_ellipsis(self):
        mp = {i: i for i in range(4)}
        out = fmt_mapping(mp, style="ascii", max_items=2, ellipsis="~more~")
        assert out.endswith("~more~}")

    def test_fmt_mapping_textual_values_are_atomic(self):
        mp = {"s": "xyz", "b": b"ab"}
        out = fmt_mapping(mp, style="paren")
        assert out == "{str('s'): str('xyz'), str('b'): bytes(b'ab')}"

    # ---------- fmt_sequence ----------

    @pytest.mark.parametrize(
        "seq, style, expected",
        [
            ([1, "a"], "ascii", "<int: 1>, <str: 'a'>"),
            ((1, "a"), "ascii", "<int: 1>, <str: 'a'>"),
        ],
    )
    def test_fmt_sequence_delimiters_list_vs_tuple(self, seq, style, expected):
        out = fmt_sequence(seq, style=style)
        if isinstance(seq, list):
            assert out == f"[{expected}]"
        else:
            assert out == f"({expected})"

    def test_fmt_sequence_singleton_tuple_trailing_comma(self):
        out = fmt_sequence((1,), style="ascii")
        assert out == "(<int: 1>,)"

    def test_fmt_sequence_nesting_depth_1(self):
        seq = [1, [2, 3]]
        out = fmt_sequence(seq, style="unicode-angle", depth=1)
        assert out == "[⟨int: 1⟩, [⟨int: 2⟩, ⟨int: 3⟩]]"

    def test_fmt_sequence_nesting_depth_0_treats_inner_as_atomic(self):
        seq = [1, [2, 3]]
        out = fmt_sequence(seq, style="paren", depth=0)
        # Inner list is formatted as a single value by fmt_value
        assert out == "[int(1), list([2, 3])]"

    @pytest.mark.parametrize(
        "style, expected_more",
        [
            ("ascii", "..."),
            ("unicode-angle", "…"),
        ],
    )
    def test_fmt_sequence_max_items_appends_ellipsis(self, style, expected_more):
        out = fmt_sequence(list(range(5)), style=style, max_items=3)
        # Expect 3 items then the ellipsis token
        assert out.endswith(expected_more + "]") or out.endswith(expected_more + ")")
        assert "<int: 0>" in out or "⟨int: 0⟩" in out

    def test_fmt_sequence_custom_ellipsis_propagates(self):
        out = fmt_sequence(list(range(5)), style="ascii", max_items=2, ellipsis=" [more] ")
        assert out.endswith(" [more] ]")

    def test_fmt_sequence_string_is_atomic(self):
        out = fmt_sequence("abc", style="colon")
        assert out == "str: 'abc'"

    # ---------- fmt_value ----------

    @pytest.mark.parametrize(
        "style, value, expected",
        [
            ("equal", 5, "int=5"),
            ("paren", 5, "int(5)"),
            ("colon", 5, "int: 5"),
            ("unicode-angle", 5, "⟨int: 5⟩"),
            ("ascii", 5, "<int: 5>"),
        ],
    )
    def test_fmt_value_basic_styles(self, style, value, expected):
        assert fmt_value(value, style=style) == expected

    def test_fmt_value_ascii_escapes_inner_gt(self):
        s = "X>Y"
        out = fmt_value(s, style="ascii")
        assert out == "<str: 'X\\>Y'>"

    @pytest.mark.parametrize(
        "style, ellipsis_expected",
        [
            ("ascii", "..."),
            ("unicode-angle", "…"),
        ],
    )
    def test_fmt_value_truncation_default_ellipsis_per_style(self, style, ellipsis_expected):
        long = "x" * 50
        out = fmt_value(long, style=style, max_repr=10)
        assert ellipsis_expected in out
        # Ensure the result ends with the chosen ellipsis inside the wrapper
        out_wo_closer = out.rstrip(">\u27e9")
        assert out_wo_closer.endswith(ellipsis_expected)

    def test_fmt_value_truncation_custom_ellipsis(self):
        out = fmt_value("abcdefghij", style="ascii", max_repr=5, ellipsis="<<more>>")
        assert out == "<str: 'a'<<more>>>"

    def test_fmt_value_type_name_for_user_class(self):
        class Foo:  # local user-defined type
            def __repr__(self):
                return "Foo()"

        f = Foo()
        out = fmt_value(f, style="equal")
        assert out.startswith("Foo=")

    def test_fmt_value_bytes_is_textual(self):
        b = b"abc"
        out = fmt_value(b, style="unicode-angle")
        assert out == "⟨bytes: b'abc'⟩"


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


class TestDictGetSet:
    def test_dict_get(self):
        d = {"a": 1,
             "b": {"c": 2},
             "e": {"empty": None}
             }
        assert dict_get(d, dot_key="a") == 1, "Should return d['a']"
        assert dict_get(d, keys=["a"]) == 1, "Should return d['a']"
        assert dict_get(d, dot_key="b.c") == 2, "Should return d['b']['c']"
        assert dict_get(d, keys=["b", "c"]) == 2, "Should return d['b']['c']"
        assert dict_get(d, dot_key="e.empty") == "", "Should return ''"
        assert dict_get(d, dot_key="e.empty", default=None) is None, "Should return None"

    def test_dict_set(self):
        d = {"a": 0,
             "b": {"c": 0}
             }

        dict_set(d, dot_key="a", value=1)
        dict_set(d, dot_key="b.c", value=2)
        dict_set(d, dot_key="i.j.q", value=3)
        assert d["a"] == 1
        assert d["b"]["c"] == 2
        assert d["i"]["j"]["q"] == 3
