"""Evaluate a saved champion and both baselines without retraining."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from ai_sandbox.saved_champion import evaluate_saved_champion, print_report, write_report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True, help="new JSON file; parent must exist")
    args = parser.parse_args(argv)
    if args.count <= 0:
        parser.error("--count must be positive")
    if args.output.exists() or args.output.is_symlink():
        parser.error(f"output already exists: {args.output}")
    if not args.output.parent.is_dir():
        parser.error(f"output parent directory does not exist: {args.output.parent}")
    try:
        report = evaluate_saved_champion(args.model, args.seed_start, args.count)
        write_report(args.output, report)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print_report(report)
    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
