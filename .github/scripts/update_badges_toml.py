#!/usr/bin/env python3
"""
Populate docs/badges/badges.toml with the latest coverage numbers.
Assumes coverage XML report already generated earlier in the workflow.
"""

from pathlib import Path
import toml
import xml.etree.ElementTree as ET

BADGE_TOML = Path("docs/badges/badges.toml")
COVERAGE_XML = Path("coverage-unit.xml")


def extract_line_rate(xml_root) -> float:
    """Return line-rate percentage from coverage.py XML."""
    value = float(xml_root.attrib["line-rate"])
    return round(value * 100)


def main():
    if not COVERAGE_XML.exists():
        raise FileNotFoundError(f"{COVERAGE_XML} missing. Run coverage before this step.")
    tree = ET.parse(COVERAGE_XML)
    percent = extract_line_rate(tree.getroot())

    data = {"pytest": {"coverage": {"unit": percent}}}
    BADGE_TOML.parent.mkdir(parents=True, exist_ok=True)
    BADGE_TOML.write_text(toml.dumps(data), encoding="utf-8")
    print(f"Updated {BADGE_TOML} with unit coverage = {percent}%")


if __name__ == "__main__":
    main()
