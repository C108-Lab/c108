"""
CLI interface for stub generators.

Usage:
    python -m c108.stubs mergeable my_file.py
    python -m c108.stubs mergeable --help
"""

import sys
import argparse
from pathlib import Path
from .mergeable import main as mergeable_main


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

    args = parser.parse_args()

    if args.command == "mergeable":
        mergeable_main(args)
    elif args.command is None:
        parser.print_help()
        print("\nAvailable commands:")
        print("  mergeable    Generate merge() method stubs")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
