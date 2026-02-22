"""
Populate docs/badges/badges.toml with the latest coverage numbers.
Assumes coverage XML report already generated earlier in the workflow.

NOTE: Run via `uv run python ...` after `uv sync --extra badges`.
DEPS: This script depends on extras from the pyproject.toml badges group
"""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import toml

DEFAULT_BADGES_DIR = Path("docs/badges")
DEFAULT_COVERAGE_XML = Path("coverage-unit.xml")


def extract_line_rate(xml_root) -> float:
    value = float(xml_root.attrib["line-rate"])
    return round(value * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description="Update badges.toml from coverage XML.")
    parser.add_argument(
        "--badges-dir",
        type=Path,
        default=DEFAULT_BADGES_DIR,
        help="Directory where badges.toml should be written.",
    )
    parser.add_argument(
        "--coverage-xml",
        type=Path,
        default=DEFAULT_COVERAGE_XML,
        help="Path to coverage XML file generated earlier.",
    )
    args = parser.parse_args()

    coverage_xml = args.coverage_xml.resolve()
    badges_dir = args.badges_dir.resolve()
    badge_toml = badges_dir / "badges.toml"

    if not coverage_xml.exists():
        raise FileNotFoundError(f"{coverage_xml} missing. Run coverage before this step.")

    tree = ET.parse(coverage_xml)
    percent = extract_line_rate(tree.getroot())

    data = {"pytest": {"coverage": {"unit": percent}}}
    badges_dir.mkdir(parents=True, exist_ok=True)
    badge_toml.write_text(toml.dumps(data), encoding="utf-8")
    print(f"Updated {badge_toml} with unit coverage = {percent}%")


if __name__ == "__main__":
    main()
