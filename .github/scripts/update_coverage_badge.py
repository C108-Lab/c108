#!/usr/bin/env python3
"""
Populate docs/badges/coverage-badge.json based on docs/badges/badges.toml data
"""

import json
import toml
from pathlib import Path

BADGE_TOML = Path("docs/badges/badges.toml")
JSON_BADGE = Path("docs/badges/coverage-badge.json")


def pick_color(pct: float) -> str:
    if pct < 70:
        return "red"
    if pct < 80:
        return "yellow"
    return "green"


def main():
    data = toml.load(BADGE_TOML)
    coverage = float(data["pytest"]["coverage"]["unit"])
    badge = {
        "schemaVersion": 1,
        "label": "Coverage",
        "message": f"{coverage:.1f}%",
        "color": pick_color(coverage),
        "logo": "pytest",
        "logoColor": "lightgreen",
    }
    JSON_BADGE.write_text(json.dumps(badge, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
