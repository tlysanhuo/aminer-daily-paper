from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from scripts.recommend import main as recommend_main


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aminer-rec",
        description="AMiner personalized paper recommendation tools.",
    )
    parser.add_argument(
        "command",
        choices=("recommend",),
        help="Command to run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _build_parser().print_help()
        return 0
    command = args[0]
    if command == "recommend":
        return recommend_main(args[1:], prog="aminer-rec recommend")
    parser = _build_parser()
    parser.parse_args(args[:1])
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
