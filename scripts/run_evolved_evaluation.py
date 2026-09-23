"""Train once, then compare the champion with baselines on unseen seeds."""

from collections.abc import Iterable

from ai_sandbox.controllers import NeuralController, RandomController, RuleBasedController
from ai_sandbox.experiments import (
    ExperimentResult, compare_paired_metric, compare_resources, run_experiment,
    summarize_results,
)
from ai_sandbox.neural_evolution import evolve
from ai_sandbox.neural_network import TinyNeuralNetwork


TRAINING_SEEDS = (0, 1, 2, 3, 4)
HELD_OUT_SEEDS = tuple(range(100, 200))


def evaluate_held_out(
    champion: TinyNeuralNetwork,
    scenario_seeds: Iterable[int],
) -> tuple[tuple[ExperimentResult, ...], ...]:
    """Run three controllers on the same seeds using fresh scenario worlds."""
    seeds = tuple(scenario_seeds)
    return (
        run_experiment(seeds, RandomController),
        run_experiment(seeds, RuleBasedController),
        run_experiment(seeds, lambda _seed: NeuralController(champion)),
    )


def _print_summary(results: tuple[ExperimentResult, ...]) -> None:
    summary = summarize_results(results)
    print(f"{results[0].controller_name} ({summary.episodes} episodes)")
    print(f"{'metric':<22} {'mean':>7} {'median':>7} {'min':>5} {'max':>5}")
    for name, metric in (
        ("ticks survived", summary.ticks_survived),
        ("resources collected", summary.resources_collected),
        ("distance travelled", summary.distance_travelled),
        ("final energy", summary.final_energy),
    ):
        print(
            f"{name:<22} {metric.mean:>7.2f} {metric.median:>7.2f} "
            f"{metric.minimum:>5} {metric.maximum:>5}"
        )
    print(
        f"Collected food: {sum(result.resources_collected > 0 for result in results)}"
        f"/{summary.episodes} scenarios"
    )
    print(
        f"Survived past 100 ticks: "
        f"{sum(result.ticks_survived > 100 for result in results)}"
        f"/{summary.episodes} scenarios"
    )
    print()


def _print_comparison(
    name: str,
    evolved: tuple[ExperimentResult, ...],
    baseline: tuple[ExperimentResult, ...],
) -> None:
    resources = compare_resources(evolved, baseline)
    survival = compare_paired_metric(
        evolved, baseline, lambda result: result.ticks_survived
    )
    print(
        f"Evolved vs {name} resources: "
        f"evolved higher {resources.first_wins}, "
        f"{name.lower()} higher {resources.second_wins}, ties {resources.ties}"
    )
    print(
        f"Evolved vs {name} ticks: "
        f"evolved higher {survival.first_wins}, "
        f"{name.lower()} higher {survival.second_wins}, ties {survival.ties}"
    )


def main() -> None:
    training = evolve(
        population_size=40,
        elite_count=8,
        scenario_seeds=TRAINING_SEEDS,
        mutation_scale=0.05,
        generations=30,
        seed=42,
    )
    champion = training.best_network
    print(f"Training seeds: {TRAINING_SEEDS}")
    print(f"Final training fitness: {training.generation_summaries[-1].best_fitness:.1f}")
    print(f"Held-out seeds: {HELD_OUT_SEEDS[0]}..{HELD_OUT_SEEDS[-1]}")
    print()

    random_results, rule_results, evolved_results = evaluate_held_out(
        champion, HELD_OUT_SEEDS
    )
    for results in (random_results, rule_results, evolved_results):
        _print_summary(results)
    _print_comparison("Random", evolved_results, random_results)
    _print_comparison("RuleBased", evolved_results, rule_results)


if __name__ == "__main__":
    main()
