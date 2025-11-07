"""
CLI interface for stub generators.

Usage:
    python -m c108.stubs mergeable my_file.py
    python -m c108.stubs merge my_file.py --sentinel UNSET
    python -m c108.stubs --help
"""

import sys
import argparse
from pathlib import Path
from .mergeable import main as mergeable_main
from .merge import main as merge_main


def main():
    """Main CLI entry point for stub generators."""
    parser = argparse.ArgumentParser(
        description="Generate stubs for c108 decorators", prog="python -m c108.stubs"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available stub generators")

    # Mergeable stub generator
    mergeable_parser = subparsers.add_parser(
        "mergeable", help="Generate merge() method stubs for @mergeable decorator"
    )
    mergeable_parser.add_argument("files", nargs="+", help="Python files containing dataclasses")
    mergeable_parser.add_argument(
        "--sentinel", default="UNSET", help="Sentinel value name to use (default: UNSET)"
    )
    mergeable_parser.add_argument("--output", "-o", help="Output file (default: print to stdout)")
    mergeable_parser.add_argument(
        "--no-docs", action="store_true", help="Generate stubs without docstrings"
    )
    mergeable_parser.add_argument(
        "--no-color", action="store_true", help="Disable syntax highlighting"
    )

    # Merge implementation generator
    merge_parser = subparsers.add_parser(
        "merge", help="Generate complete merge() method implementations for dataclasses"
    )
    merge_parser.add_argument("files", nargs="+", help="Python files containing dataclasses")
    merge_parser.add_argument(
        "--sentinel",
        default="UNSET",
        choices=["None", "DEFAULT", "MISSING", "NOT_FOUND", "STOP", "UNSET"],
        help="Sentinel value to use (default: UNSET)",
    )
    merge_parser.add_argument(
        "--include", nargs="+", help="Only include these fields (whitelist mode)"
    )
    merge_parser.add_argument("--exclude", nargs="+", help="Exclude these fields (blacklist mode)")
    merge_parser.add_argument(
        "--exclude-private",
        action="store_true",
        default=True,
        help="Exclude private fields starting with _ (default: True)",
    )
    merge_parser.add_argument(
        "--include-private",
        action="store_true",
        help="Include private fields (overrides --exclude-private)",
    )
    merge_parser.add_argument("--output", "-o", help="Output file (default: print to stdout)")
    merge_parser.add_argument(
        "--no-docs", action="store_true", help="Generate implementations without docstrings"
    )
    merge_parser.add_argument("--no-color", action="store_true", help="Disable syntax highlighting")

    merge_parser.add_argument(
        "--classes", nargs="+", help="Specific dataclass names to generate merge() for"
    )
    merge_parser.add_argument(
        "--all", action="store_true", help="Generate merge() for all dataclasses in file"
    )

    args = parser.parse_args()

    # Handle --include-private flag for merge command
    if args.command == "merge" and args.include_private:
        args.exclude_private = False

    if args.command == "mergeable":
        mergeable_main(args)
    elif args.command == "merge":
        merge_main(args)
    elif args.command is None:
        parser.print_help()
        print("\nAvailable commands:")
        print("  mergeable    Generate merge() method stubs for @mergeable decorator")
        print("  merge        Generate complete merge() method implementations")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
