"""Train with reproducibly changing scenario batches and save experiment artifacts."""

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from ai_sandbox.diverse_artifacts import run_diverse_experiment
from ai_sandbox.diverse_evolution import DiverseConfig


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--generations", type=int, default=5, help="number of evaluated generations")
    parser.add_argument("--population", type=int, default=12, help="population size")
    parser.add_argument("--elites", type=int, default=3, help="unchanged elites per generation")
    parser.add_argument("--scenarios-per-generation", type=int, default=5, help="shared training batch size")
    parser.add_argument("--mutation-scale", type=float, default=0.05, help="Gaussian mutation scale")
    parser.add_argument("--seed", type=int, default=42, help="evolution RNG seed")
    parser.add_argument("--scenario-schedule-seed", type=int, default=123, help="independent scenario RNG seed")
    parser.add_argument("--validation-interval", type=int, default=2,
                        help="validate every N completed generations and at the end; 0 disables")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results") / ("diverse-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")),
                        help="new directory for metrics and models (must not exist)")
    args = parser.parse_args(argv)
    try:
        args.config = DiverseConfig(
            population_size=args.population, elite_count=args.elites,
            scenarios_per_generation=args.scenarios_per_generation,
            mutation_scale=args.mutation_scale, generations=args.generations,
            seed=args.seed, scenario_schedule_seed=args.scenario_schedule_seed,
            validation_interval=args.validation_interval,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_diverse_experiment(args.config, args.output_dir)


if __name__ == "__main__":
    main()
