"""Compare baseline controllers over the same 100 random scenarios."""

from ai_sandbox.controllers import RandomController, RuleBasedController
from ai_sandbox.experiments import compare_resources, run_experiment, summarize_results


def main() -> None:
    random_results = run_experiment(range(100), RandomController)
    rule_results = run_experiment(range(100), RuleBasedController)
    for results in (random_results, rule_results):
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
        print()

    comparison = compare_resources(random_results, rule_results)
    print(
        f"Resources by scenario ({comparison.scenarios}): "
        f"{random_results[0].controller_name} wins {comparison.first_wins}, "
        f"{rule_results[0].controller_name} wins {comparison.second_wins}, "
        f"ties {comparison.ties}"
    )


if __name__ == "__main__":
    main()
