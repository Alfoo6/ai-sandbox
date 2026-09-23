"""Reproducible headless experiments over seeded scenarios."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from statistics import fmean

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
class ExperimentSummary:
    episodes: int
    mean_ticks_survived: float
    mean_resources_collected: float
    mean_distance_travelled: float
    mean_final_energy: float


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
    """Compute arithmetic means over individual episode results."""
    episodes = tuple(results)
    if not episodes:
        raise ValueError("cannot summarize an empty experiment")
    return ExperimentSummary(
        episodes=len(episodes),
        mean_ticks_survived=fmean(result.ticks_survived for result in episodes),
        mean_resources_collected=fmean(result.resources_collected for result in episodes),
        mean_distance_travelled=fmean(result.distance_travelled for result in episodes),
        mean_final_energy=fmean(result.final_energy for result in episodes),
    )
