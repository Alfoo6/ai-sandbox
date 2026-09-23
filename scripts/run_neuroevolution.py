"""Run the first reproducible neuroevolution demonstration."""

from ai_sandbox.neural_evolution import evolve
from ai_sandbox.neural_fitness import evaluate_network


SCENARIO_SEEDS = (0, 1, 2, 3, 4)


def main() -> None:
    result = evolve(
        population_size=40,
        elite_count=8,
        scenario_seeds=SCENARIO_SEEDS,
        mutation_scale=0.05,
        generations=30,
        seed=42,
    )
    for summary in result.generation_summaries:
        print(
            f"Generation {summary.generation:02d} | "
            f"best {summary.best_fitness:.1f} | mean {summary.mean_fitness:.1f}"
        )
    final = evaluate_network(result.best_network, SCENARIO_SEEDS)
    print(f"Final best fitness: {final.fitness:.1f}")
    print(f"Per-scenario ticks: {final.ticks_survived}")


if __name__ == "__main__":
    main()
