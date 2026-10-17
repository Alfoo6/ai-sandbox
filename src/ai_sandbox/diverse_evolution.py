"""Changing-batch evolution, separate from the original fixed-scenario loop."""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from statistics import fmean

import numpy as np

from ai_sandbox.controllers import NeuralController
from ai_sandbox.evolution_study import ValidationResult
from ai_sandbox.experiments import run_experiment, summarize_results
from ai_sandbox.neural_evolution import (
    GenerationSummary, _unique_seed, evaluate_population, next_generation,
)
from ai_sandbox.neural_network import TinyNeuralNetwork


TRAINING_SEED_START = 10_000
GENERATION_SEED_SPAN = 1_000_000
VALIDATION_SEEDS = tuple(range(200, 250))


def _integer(name: str, value: int, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def scenario_batch(generation: int, count: int, schedule_seed: int) -> tuple[int, ...]:
    """Sample without replacement in a distinct seed block per generation.

    Generation g owns [10000 + g*1000000, 10000 + (g+1)*1000000).
    The schedule depends on neither evolution RNG consumption nor run length.
    """
    _integer("generation", generation, 0)
    _integer("count", count, 1)
    _integer("schedule_seed", schedule_seed, 0)
    if count > GENERATION_SEED_SPAN:
        raise ValueError("count exceeds generation seed block size")
    rng = np.random.default_rng(np.random.SeedSequence([schedule_seed, generation]))
    offsets = rng.choice(GENERATION_SEED_SPAN, size=count, replace=False)
    start = TRAINING_SEED_START + generation * GENERATION_SEED_SPAN
    return tuple(start + int(offset) for offset in offsets)


@dataclass(frozen=True)
class DiverseConfig:
    population_size: int = 12
    elite_count: int = 3
    scenarios_per_generation: int = 5
    mutation_scale: float = 0.05
    generations: int = 5
    seed: int = 42
    scenario_schedule_seed: int = 123
    validation_interval: int = 2
    validation_seeds: tuple[int, ...] = VALIDATION_SEEDS

    def __post_init__(self) -> None:
        for name in ("population_size", "elite_count", "scenarios_per_generation", "generations"):
            _integer(name, getattr(self, name), 1)
        for name in ("seed", "scenario_schedule_seed", "validation_interval"):
            _integer(name, getattr(self, name), 0)
        if self.elite_count > self.population_size:
            raise ValueError("elite_count must not exceed population_size")
        if self.scenarios_per_generation > GENERATION_SEED_SPAN:
            raise ValueError("scenarios_per_generation exceeds generation seed block size")
        if not isfinite(self.mutation_scale) or self.mutation_scale < 0:
            raise ValueError("mutation_scale must be finite and non-negative")
        seeds = tuple(self.validation_seeds)
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("validation_seeds must be nonempty and unique")
        if any(type(seed) is not int or not 200 <= seed < 1000 for seed in seeds):
            raise ValueError("validation_seeds must be in the validation namespace [200, 1000)")
        object.__setattr__(self, "validation_seeds", seeds)


@dataclass(frozen=True)
class DiverseGeneration:
    training: GenerationSummary
    scenario_seeds: tuple[int, ...]
    validation: ValidationResult | None
    selected_for_validation_champion: bool


@dataclass(frozen=True)
class ValidationChampion:
    generation: int
    network: TinyNeuralNetwork
    validation: ValidationResult


@dataclass(frozen=True)
class DiverseResult:
    generations: tuple[DiverseGeneration, ...]
    final_batch_champion: TinyNeuralNetwork
    best_validation_champion: ValidationChampion | None


def evolve_diverse(
    config: DiverseConfig,
    on_generation: Callable[[DiverseGeneration, ValidationChampion | None], None] | None = None,
) -> DiverseResult:
    """Breed using only current-batch scores; reevaluate everyone next generation.

    Validate the current batch winner after each interval of completed generations
    and at the final generation. Interval zero disables validation. Training ties
    use population order; validation ties retain the earliest candidate. Only a
    newly selected validation champion is passed to the optional reporting hook.
    The hook must not mutate networks. It consumes no evolution or scenario RNG.
    """
    rng = np.random.default_rng(config.seed)
    used_seeds: set[int] = set()
    population = tuple(
        TinyNeuralNetwork.random(26, 12, 5, seed=_unique_seed(rng, used_seeds))
        for _ in range(config.population_size)
    )
    records = []
    best_validation = None
    for generation in range(config.generations):
        seeds = scenario_batch(generation, config.scenarios_per_generation, config.scenario_schedule_seed)
        evaluations = evaluate_population(population, seeds)
        fitnesses = tuple(result.fitness for result in evaluations)
        winner = population[max(range(len(population)), key=fitnesses.__getitem__)]
        summary = GenerationSummary(generation, max(fitnesses), fmean(fitnesses), fitnesses)
        validation = None
        selected = None
        if config.validation_interval and (
            (generation + 1) % config.validation_interval == 0
            or generation == config.generations - 1
        ):
            episodes = run_experiment(config.validation_seeds, lambda _seed: NeuralController(winner))
            validation = ValidationResult(episodes, summarize_results(episodes))
            if best_validation is None or (
                validation.summary.ticks_survived.mean
                > best_validation.validation.summary.ticks_survived.mean
            ):
                # Mutation returns new children; this retained network is never mutated.
                selected = ValidationChampion(generation, winner, validation)
                best_validation = selected
        record = DiverseGeneration(summary, seeds, validation, selected is not None)
        records.append(record)
        if on_generation is not None:
            on_generation(record, selected)
        if generation < config.generations - 1:
            mutation_seeds = tuple(
                _unique_seed(rng, used_seeds)
                for _ in range(config.population_size - config.elite_count)
            )
            # Existing selection preserves elite objects unchanged. Their old fitness
            # is discarded: all individuals are evaluated on the next batch above.
            population = next_generation(
                population, fitnesses, config.elite_count, config.mutation_scale, mutation_seeds
            )
    return DiverseResult(tuple(records), winner, best_validation)
