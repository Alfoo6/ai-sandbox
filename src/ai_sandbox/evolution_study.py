"""Repeated independent training runs with shared validation scenarios."""

from dataclasses import dataclass
from math import isfinite
from statistics import fmean, pstdev

from ai_sandbox.controllers import NeuralController, RandomController, RuleBasedController
from ai_sandbox.experiments import (
    ExperimentResult, ExperimentSummary, run_experiment, summarize_results,
)
from ai_sandbox.neural_evolution import EvolutionResult, evolve


@dataclass(frozen=True)
class StudyConfig:
    population_size: int
    elite_count: int
    training_seeds: tuple[int, ...]
    mutation_scale: float
    generations: int
    evolution_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in ("population_size", "elite_count", "generations"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.elite_count > self.population_size:
            raise ValueError("elite_count must not exceed population_size")
        if not isfinite(self.mutation_scale) or self.mutation_scale < 0:
            raise ValueError("mutation_scale must be finite and non-negative")
        for name in ("training_seeds", "evolution_seeds", "validation_seeds"):
            seeds = tuple(getattr(self, name))
            if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
                raise ValueError(f"{name} must contain non-negative integers")
            if len(set(seeds)) != len(seeds):
                raise ValueError(f"{name} must contain unique seeds")
            object.__setattr__(self, name, seeds)
        if not set(self.training_seeds).isdisjoint(self.validation_seeds):
            raise ValueError("training_seeds and validation_seeds must be disjoint")
        if any(1000 <= seed <= 1099 for seed in self.training_seeds + self.validation_seeds):
            raise ValueError("scenario seeds 1000–1099 are reserved for final testing")


@dataclass(frozen=True)
class ValidationResult:
    episodes: tuple[ExperimentResult, ...]
    summary: ExperimentSummary

    @property
    def episodes_collecting_resource(self) -> int:
        return sum(episode.resources_collected >= 1 for episode in self.episodes)

    @property
    def episodes_past_100_ticks(self) -> int:
        return sum(episode.ticks_survived > 100 for episode in self.episodes)


@dataclass(frozen=True)
class StudyRun:
    evolution_seed: int
    training: EvolutionResult
    validation: ValidationResult

    @property
    def final_training_best_fitness(self) -> float:
        return self.training.generation_summaries[-1].best_fitness


@dataclass(frozen=True)
class StudyAggregate:
    """Mean and population SD (ddof=0) of per-run validation means."""

    mean_validation_survival: float
    std_validation_survival: float
    mean_validation_resources: float
    std_validation_resources: float


@dataclass(frozen=True)
class StudyResult:
    config: StudyConfig
    runs: tuple[StudyRun, ...]
    aggregate: StudyAggregate
    reference_baselines: tuple[ValidationResult, ...]


def aggregate_runs(runs: tuple[StudyRun, ...]) -> StudyAggregate:
    """Describe variation across runs, not across individual scenario episodes.

    Population SD describes the observed runs; one run has SD zero.
    """
    if not runs:
        raise ValueError("cannot aggregate an empty study")
    survival = tuple(run.validation.summary.ticks_survived.mean for run in runs)
    resources = tuple(run.validation.summary.resources_collected.mean for run in runs)
    return StudyAggregate(fmean(survival), pstdev(survival), fmean(resources), pstdev(resources))


def run_study(config: StudyConfig) -> StudyResult:
    """Train from scratch per evolution seed; validation never selects champions.

    Retain each unchanged champion in its EvolutionResult. Baseline controller
    RNG seeds equal scenario seeds, following the existing experiment convention.
    """
    runs = []
    for seed in config.evolution_seeds:
        training = evolve(
            population_size=config.population_size,
            elite_count=config.elite_count,
            scenario_seeds=config.training_seeds,
            mutation_scale=config.mutation_scale,
            generations=config.generations,
            seed=seed,
        )
        episodes = run_experiment(
            config.validation_seeds, lambda _seed: NeuralController(training.best_network)
        )
        runs.append(StudyRun(seed, training, ValidationResult(episodes, summarize_results(episodes))))
    baselines = []
    for controller in (RandomController, RuleBasedController):
        episodes = run_experiment(config.validation_seeds, controller)
        baselines.append(ValidationResult(episodes, summarize_results(episodes)))
    completed = tuple(runs)
    return StudyResult(config, completed, aggregate_runs(completed), tuple(baselines))
