"""
Populate docs/badges/coverage-badge.json based on docs/badges/badges.toml data.

NOTE: Run via `uv run python ...` after `uv sync --extra badges`.
DEPS: This script depends on extras from the pyproject.toml badges group
"""

import argparse
import json
from pathlib import Path

import toml

DEFAULT_BADGES_DIR = Path("docs/badges")


def pick_color(pct: float) -> str:
    if pct < 70:
        return "red"
    if pct < 80:
        return "yellow"
    return "green"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate coverage badge JSON.")
    parser.add_argument(
        "--badges-dir",
        type=Path,
        default=DEFAULT_BADGES_DIR,
        help="Directory containing badges.toml and where JSON should be written.",
    )
    args = parser.parse_args()

    badges_dir = args.badges_dir.resolve()
    badge_toml = badges_dir / "badges.toml"
    json_badge = badges_dir / "coverage-badge.json"

    data = toml.load(badge_toml)
    coverage = float(data["pytest"]["coverage"]["unit"])
    badge = {
        "schemaVersion": 1,
        "label": "Coverage",
        "message": f"{coverage:.1f}%",
        "color": pick_color(coverage),
        "logo": "pytest",
        "logoColor": "lightgreen",
    }
    json_badge.write_text(json.dumps(badge, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {json_badge} ({badge['message']}, {badge['color']})")


if __name__ == "__main__":
    main()
