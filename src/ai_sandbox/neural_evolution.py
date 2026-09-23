"""Small, reproducible generational neuroevolution loop."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import fmean

import numpy as np

from ai_sandbox.neural_fitness import NeuralFitnessResult, evaluate_network
from ai_sandbox.neural_mutation import mutate_network
from ai_sandbox.neural_network import TinyNeuralNetwork


@dataclass(frozen=True)
class GenerationSummary:
    generation: int
    best_fitness: float
    mean_fitness: float
    individual_fitnesses: tuple[float, ...]


@dataclass(frozen=True)
class EvolutionResult:
    best_network: TinyNeuralNetwork
    generation_summaries: tuple[GenerationSummary, ...]


def evaluate_population(
    population: Sequence[TinyNeuralNetwork],
    scenario_seeds: Iterable[int],
) -> tuple[NeuralFitnessResult, ...]:
    """Evaluate every network on the same ordered scenario seeds."""
    seeds = tuple(scenario_seeds)
    return tuple(evaluate_network(network, seeds) for network in population)


def next_generation(
    population: Sequence[TinyNeuralNetwork],
    fitnesses: Sequence[float],
    elite_count: int,
    mutation_scale: float,
    mutation_seeds: Iterable[int],
) -> tuple[TinyNeuralNetwork, ...]:
    """Keep ranked elites, then mutate them round-robin using explicit seeds.

    Python's stable sort preserves population order when fitnesses tie.
    """
    if not population or len(population) != len(fitnesses):
        raise ValueError("population and fitnesses must have the same nonzero length")
    if not 0 < elite_count <= len(population):
        raise ValueError("elite_count must be between 1 and population_size")
    if not np.isfinite(mutation_scale) or mutation_scale < 0:
        raise ValueError("mutation_scale must be finite and non-negative")

    seeds = tuple(mutation_seeds)
    child_count = len(population) - elite_count
    if len(seeds) != child_count or len(set(seeds)) != child_count:
        raise ValueError("mutation_seeds must contain one unique seed per child")

    ranked_indexes = sorted(range(len(population)), key=lambda index: -fitnesses[index])
    elites = tuple(population[index] for index in ranked_indexes[:elite_count])
    children = tuple(
        mutate_network(elites[index % elite_count], mutation_scale, seed)
        for index, seed in enumerate(seeds)
    )
    return elites + children


def evolve(
    population_size: int,
    elite_count: int,
    scenario_seeds: Iterable[int],
    mutation_scale: float,
    generations: int,
    seed: int,
) -> EvolutionResult:
    """Evaluate generation 0, then produce and evaluate later generations."""
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if not 0 < elite_count <= population_size:
        raise ValueError("elite_count must be between 1 and population_size")
    if generations <= 0:
        raise ValueError("generations must be positive")
    if not np.isfinite(mutation_scale) or mutation_scale < 0:
        raise ValueError("mutation_scale must be finite and non-negative")
    seeds = tuple(scenario_seeds)
    if not seeds:
        raise ValueError("scenario_seeds must not be empty")

    rng = np.random.default_rng(seed)
    used_seeds: set[int] = set()
    population = tuple(
        TinyNeuralNetwork.random(26, 12, 5, seed=_unique_seed(rng, used_seeds))
        for _ in range(population_size)
    )
    summaries = []
    best_network = population[0]
    for generation in range(generations):
        evaluations = evaluate_population(population, seeds)
        fitnesses = tuple(result.fitness for result in evaluations)
        best_index = max(range(population_size), key=fitnesses.__getitem__)
        best_network = population[best_index]
        summaries.append(GenerationSummary(
            generation=generation,
            best_fitness=fitnesses[best_index],
            mean_fitness=fmean(fitnesses),
            individual_fitnesses=fitnesses,
        ))
        if generation < generations - 1:
            mutation_seeds = tuple(
                _unique_seed(rng, used_seeds)
                for _ in range(population_size - elite_count)
            )
            population = next_generation(
                population, fitnesses, elite_count, mutation_scale, mutation_seeds
            )

    return EvolutionResult(best_network, tuple(summaries))


def _unique_seed(rng: np.random.Generator, used_seeds: set[int]) -> int:
    """Draw a distinct non-negative seed from the local training generator."""
    while True:
        seed = int(rng.integers(0, np.iinfo(np.int64).max))
        if seed not in used_seeds:
            used_seeds.add(seed)
            return seed
