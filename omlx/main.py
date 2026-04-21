#!/usr/bin/env python3
"""omlx - A command-line tool for managing and launching applications.

Fork of jundot/omlx with additional features and improvements.
"""

import argparse
import sys
import os
from pathlib import Path

__version__ = "0.1.0"
__author__ = "omlx contributors"


def get_config_dir() -> Path:
    """Return the configuration directory for omlx."""
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config:
        return Path(xdg_config) / "omlx"
    return Path.home() / ".config" / "omlx"


def get_data_dir() -> Path:
    """Return the data directory for omlx."""
    xdg_data = os.environ.get("XDG_DATA_HOME")
    if xdg_data:
        return Path(xdg_data) / "omlx"
    return Path.home() / ".local" / "share" / "omlx"


def cmd_list(args: argparse.Namespace) -> int:
    """List all registered entries."""
    data_dir = get_data_dir()
    if not data_dir.exists():
        print("No entries found. Data directory does not exist.")
        return 0

    entries = list(data_dir.glob("*.json"))
    if not entries:
        print("No entries registered.")
        return 0

    for entry in sorted(entries):
        print(entry.stem)
    return 0


def cmd_version(args: argparse.Namespace) -> int:
    """Print version information."""
    print(f"omlx version {__version__}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="omlx",
        description="omlx - command-line application manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # list subcommand
    list_parser = subparsers.add_parser("list", help="List all registered entries")
    list_parser.set_defaults(func=cmd_list)

    # version subcommand
    version_parser = subparsers.add_parser("version", help="Show version information")
    version_parser.set_defaults(func=cmd_version)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the omlx CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        # Show help when no command is given — listing silently felt confusing
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
