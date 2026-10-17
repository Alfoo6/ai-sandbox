"""Reconstruct the first evolved champion and watch it on a selected scenario."""

import argparse
from collections.abc import Sequence

from ai_sandbox.controllers import NeuralController
from ai_sandbox.neural_evolution import evolve
from ai_sandbox.neural_network import TinyNeuralNetwork
from ai_sandbox.scenarios import create_random_episode
from ai_sandbox.visual_demo import run_visual_episode


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario", type=int, default=0,
        help="scenario seed to watch (default: 0; training seeds: 0–4)",
    )
    return parser.parse_args(argv)


def reproduce_champion() -> TinyNeuralNetwork:
    """Train with the fixed original configuration, independent of viewer seed."""
    result = evolve(
        population_size=40,
        elite_count=8,
        scenario_seeds=(0, 1, 2, 3, 4),
        mutation_scale=0.05,
        generations=30,
        seed=42,
    )
    fitness = result.generation_summaries[-1].best_fitness
    if fitness != 125.0:
        raise RuntimeError(f"Expected final training best fitness 125.0, got {fitness}")
    return result.best_network


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print("Reconstructing champion on training seeds 0–4; please wait...", flush=True)
    champion = reproduce_champion()
    print(f"Final training best fitness: 125.0 | Scenario: {args.scenario}", flush=True)
    run_visual_episode(
        create_random_episode(args.scenario), NeuralController(champion),
        scenario_seed=args.scenario,
    )


if __name__ == "__main__":
    main()
