"""Run the fixed initial v0.3.8 study; edit CONFIG to explore other settings."""

from time import perf_counter

from ai_sandbox.evolution_study import StudyConfig, ValidationResult, run_study


CONFIG = StudyConfig(
    population_size=40,
    elite_count=8,
    training_seeds=(0, 1, 2, 3, 4),
    mutation_scale=0.05,
    generations=30,
    evolution_seeds=(42, 43, 44),
    validation_seeds=tuple(range(200, 250)),
)


def _print_validation(label: str, training: str, result: ValidationResult) -> None:
    summary = result.summary
    print(
        f"{label:<22} {training:>10} {summary.ticks_survived.mean:>10.2f} "
        f"{summary.ticks_survived.median:>12.2f} "
        f"{summary.resources_collected.mean:>14.2f} "
        f"{result.episodes_collecting_resource:>4}/{summary.episodes:<3} "
        f"{result.episodes_past_100_ticks:>4}/{summary.episodes}"
    )


def main() -> None:
    print(f"Configuration: {CONFIG}", flush=True)
    started = perf_counter()
    study = run_study(CONFIG)
    print("\nValidation episodes: food = collected >=1 resource; >100 = survived past 100 ticks")
    print(
        f"{'Evolution seed':<22} {'Train best':>10} {'Mean ticks':>10} "
        f"{'Median ticks':>12} {'Mean resources':>14} {'Food':>8} {'>100':>8}"
    )
    for run in study.runs:
        _print_validation(str(run.evolution_seed), f"{run.final_training_best_fitness:.2f}", run.validation)
    print("\nReference baselines (same validation scenarios):")
    for baseline in study.reference_baselines:
        _print_validation(baseline.episodes[0].controller_name, "—", baseline)
    aggregate = study.aggregate
    print("\nAcross evolution runs (population SD, ddof=0, of per-run validation means):")
    print(f"Survival ticks: mean={aggregate.mean_validation_survival:.12g}, SD={aggregate.std_validation_survival:.12g}")
    print(f"Resources: mean={aggregate.mean_validation_resources:.12g}, SD={aggregate.std_validation_resources:.12g}")
    print("These SDs describe run variation, not episode variation. No statistical significance claim.")
    print("Validation is not final testing; scenario seeds 1000–1099 remain unused.")
    print(f"Runtime: {perf_counter() - started:.2f} seconds")


if __name__ == "__main__":
    main()
