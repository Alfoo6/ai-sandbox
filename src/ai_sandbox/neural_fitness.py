"""Reproducible survival fitness for one neural network."""

from collections.abc import Iterable
from dataclasses import dataclass
from statistics import fmean

from ai_sandbox.controllers.neural_controller import NeuralController
from ai_sandbox.neural_network import TinyNeuralNetwork
from ai_sandbox.runner import run_episode
from ai_sandbox.scenarios import create_random_episode


@dataclass(frozen=True)
class NeuralFitnessResult:
    fitness: float
    scenario_seeds: tuple[int, ...]
    ticks_survived: tuple[int, ...]


def evaluate_network(
    network: TinyNeuralNetwork,
    scenario_seeds: Iterable[int],
) -> NeuralFitnessResult:
    """Return mean ticks survived across fresh episodes for the given seeds."""
    seeds = tuple(scenario_seeds)
    if not seeds:
        raise ValueError("scenario_seeds must not be empty")

    ticks = tuple(
        run_episode(create_random_episode(seed), NeuralController(network)).ticks_survived
        for seed in seeds
    )
    return NeuralFitnessResult(fmean(ticks), seeds, ticks)
