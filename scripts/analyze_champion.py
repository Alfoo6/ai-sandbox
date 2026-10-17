"""Analyze one saved champion episode and controlled observations without training."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from ai_sandbox.behavioral_analysis import analyze_champion
from ai_sandbox.saved_champion import write_report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scenario", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new JSON file; parent must exist")
    parser.add_argument("--no-food", action="store_true", help="construct an isolated empty scenario")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error(f"output already exists: {args.output}")
    if not args.output.parent.is_dir():
        parser.error(f"output parent directory does not exist: {args.output.parent}")
    try:
        report = analyze_champion(args.model, args.scenario, args.no_food)
        write_report(args.output, report)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
