"""Compare baseline controllers over the same 100 random scenarios."""

from ai_sandbox.controllers import RandomController, RuleBasedController
from ai_sandbox.experiments import run_experiment, summarize_results


def main() -> None:
    print(
        f"{'controller':<20} {'episodes':>8} {'ticks':>8} "
        f"{'resources':>10} {'distance':>9} {'energy':>8}"
    )
    for controller_type in (RandomController, RuleBasedController):
        results = run_experiment(range(100), controller_type)
        summary = summarize_results(results)
        print(
            f"{results[0].controller_name:<20} {summary.episodes:>8} "
            f"{summary.mean_ticks_survived:>8.2f} "
            f"{summary.mean_resources_collected:>10.2f} "
            f"{summary.mean_distance_travelled:>9.2f} "
            f"{summary.mean_final_energy:>8.2f}"
        )


if __name__ == "__main__":
    main()
