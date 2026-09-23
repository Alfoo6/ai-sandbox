"""Reproducible headless experiments over seeded scenarios."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from statistics import fmean, median

from ai_sandbox.contracts import Controller
from ai_sandbox.runner import run_episode
from ai_sandbox.scenarios import create_random_episode


@dataclass(frozen=True)
class ExperimentResult:
    scenario_seed: int
    controller_name: str
    ticks_survived: int
    resources_collected: int
    distance_travelled: int
    final_energy: int


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    median: float
    minimum: int
    maximum: int


@dataclass(frozen=True)
class ExperimentSummary:
    episodes: int
    ticks_survived: MetricSummary
    resources_collected: MetricSummary
    distance_travelled: MetricSummary
    final_energy: MetricSummary


@dataclass(frozen=True)
class ResourceComparison:
    scenarios: int
    first_wins: int
    second_wins: int
    ties: int


@dataclass(frozen=True)
class PairedComparison:
    scenarios: int
    first_wins: int
    second_wins: int
    ties: int


def run_experiment(
    scenario_seeds: Iterable[int],
    controller_factory: Callable[[int], Controller],
) -> tuple[ExperimentResult, ...]:
    """Run a fresh episode and controller for every scenario seed."""
    results = []
    for seed in scenario_seeds:
        episode = create_random_episode(seed)
        controller = controller_factory(seed)
        metrics = run_episode(episode, controller)
        results.append(ExperimentResult(
            scenario_seed=seed,
            controller_name=type(controller).__name__,
            ticks_survived=metrics.ticks_survived,
            resources_collected=metrics.resources_collected,
            distance_travelled=metrics.distance_travelled,
            final_energy=metrics.final_energy,
        ))
    return tuple(results)


def summarize_results(results: Iterable[ExperimentResult]) -> ExperimentSummary:
    """Compute mean, median, minimum, and maximum for each episode metric."""
    episodes = tuple(results)
    if not episodes:
        raise ValueError("cannot summarize an empty experiment")
    return ExperimentSummary(
        episodes=len(episodes),
        ticks_survived=_summarize_metric(result.ticks_survived for result in episodes),
        resources_collected=_summarize_metric(result.resources_collected for result in episodes),
        distance_travelled=_summarize_metric(result.distance_travelled for result in episodes),
        final_energy=_summarize_metric(result.final_energy for result in episodes),
    )


def _summarize_metric(values: Iterable[int]) -> MetricSummary:
    numbers = tuple(values)
    return MetricSummary(fmean(numbers), median(numbers), min(numbers), max(numbers))


def compare_resources(
    first_results: Iterable[ExperimentResult],
    second_results: Iterable[ExperimentResult],
) -> ResourceComparison:
    """Count resource wins and ties for matching scenario seeds."""
    comparison = compare_paired_metric(
        first_results, second_results, lambda result: result.resources_collected
    )
    return ResourceComparison(
        comparison.scenarios,
        comparison.first_wins,
        comparison.second_wins,
        comparison.ties,
    )


def compare_paired_metric(
    first_results: Iterable[ExperimentResult],
    second_results: Iterable[ExperimentResult],
    metric: Callable[[ExperimentResult], int],
) -> PairedComparison:
    """Count wins and ties by scenario seed for one integer episode metric."""
    first = tuple(first_results)
    second = tuple(second_results)
    first_by_seed = {result.scenario_seed: result for result in first}
    second_by_seed = {result.scenario_seed: result for result in second}
    if len(first_by_seed) != len(first) or len(second_by_seed) != len(second):
        raise ValueError("duplicate scenario seeds in comparison")
    if first_by_seed.keys() != second_by_seed.keys():
        raise ValueError("comparison requires matching scenario seeds")

    first_wins = second_wins = ties = 0
    for seed, first_result in first_by_seed.items():
        first_value = metric(first_result)
        second_value = metric(second_by_seed[seed])
        if first_value > second_value:
            first_wins += 1
        elif first_value < second_value:
            second_wins += 1
        else:
            ties += 1
    return PairedComparison(len(first_by_seed), first_wins, second_wins, ties)
