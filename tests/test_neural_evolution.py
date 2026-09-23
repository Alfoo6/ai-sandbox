"""Tests for reproducible, elitist neuroevolution."""

import random
from statistics import fmean
import unittest
from unittest.mock import patch

import numpy as np

from ai_sandbox.neural_evolution import evolve, next_generation
from ai_sandbox.neural_fitness import NeuralFitnessResult, evaluate_network
from ai_sandbox.neural_mutation import mutate_network
from ai_sandbox.neural_network import TinyNeuralNetwork


class NeuralEvolutionTests(unittest.TestCase):
    def test_initial_population_is_independent_and_uses_shared_scenarios(self) -> None:
        calls = []

        def record_evaluation(network, seeds):
            calls.append((network, seeds))
            return evaluate_network(network, seeds)

        with patch("ai_sandbox.neural_evolution.evaluate_network", side_effect=record_evaluation):
            result = evolve(3, 1, iter((2, 0)), 0.05, 1, seed=42)

        self.assertEqual(len(calls), 3)
        self.assertEqual(len({id(network) for network, _ in calls}), 3)
        self.assertTrue(all(
            (network.input_size, network.hidden_size, network.output_size) == (26, 12, 5)
            for network, _ in calls
        ))
        self.assertEqual([seeds for _, seeds in calls], [(2, 0)] * 3)
        self.assertEqual(len(result.generation_summaries), 1)
        self.assertEqual(result.generation_summaries[0].generation, 0)

    def test_transition_keeps_ranked_elites_and_round_robins_new_children(self) -> None:
        population = tuple(TinyNeuralNetwork.random(seed=seed) for seed in range(5))
        calls = []

        def record_mutation(parent, scale, seed):
            calls.append((parent, scale, seed))
            return mutate_network(parent, scale, seed)

        with patch("ai_sandbox.neural_evolution.mutate_network", side_effect=record_mutation):
            following = next_generation(population, (5, 5, 4, 3, 2), 2, 0.05, (11, 12, 13))

        self.assertEqual(len(following), 5)
        self.assertIs(following[0], population[0])
        self.assertIs(following[1], population[1])
        self.assertEqual(calls, [
            (population[0], 0.05, 11),
            (population[1], 0.05, 12),
            (population[0], 0.05, 13),
        ])
        for child in following[2:]:
            self.assertTrue(all(child is not parent for parent in population))
        for original, current in zip(population[0].parameters(), following[0].parameters()):
            np.testing.assert_array_equal(original, current)

    def test_generation_zero_indexes_and_mean_use_individual_fitnesses(self) -> None:
        def evaluated(*fitnesses):
            return tuple(NeuralFitnessResult(float(value), (0,), (int(value),)) for value in fitnesses)

        with patch("ai_sandbox.neural_evolution.evaluate_population", side_effect=(
            evaluated(100, 120, 110),
            evaluated(120, 130, 90),
            evaluated(130, 110, 100),
        )):
            result = evolve(3, 1, (0,), 0.05, 3, seed=42)

        summaries = result.generation_summaries
        self.assertEqual([summary.generation for summary in summaries], [0, 1, 2])
        self.assertEqual([summary.best_fitness for summary in summaries], [120, 130, 130])
        self.assertEqual([summary.mean_fitness for summary in summaries], [110, 340 / 3, 340 / 3])
        self.assertEqual(summaries[0].individual_fitnesses, (100.0, 120.0, 110.0))

    def test_same_seed_reproduces_summaries_parameters_and_mutation_seeds(self) -> None:
        calls = []

        def record_mutation(parent, scale, seed):
            calls.append(seed)
            return mutate_network(parent, scale, seed)

        with patch("ai_sandbox.neural_evolution.mutate_network", side_effect=record_mutation):
            first = evolve(4, 2, (0, 1), 0.05, 3, seed=42)
            first_seeds = tuple(calls)
            calls.clear()
            second = evolve(4, 2, (0, 1), 0.05, 3, seed=42)

        self.assertEqual(first.generation_summaries, second.generation_summaries)
        self.assertEqual(tuple(calls), first_seeds)
        self.assertEqual(len(set(first_seeds)), len(first_seeds))
        for left, right in zip(first.best_network.parameters(), second.best_network.parameters()):
            np.testing.assert_array_equal(left, right)

    def test_different_training_seeds_can_change_final_parameters(self) -> None:
        first = evolve(3, 1, (0,), 0.05, 1, seed=42)
        second = evolve(3, 1, (0,), 0.05, 1, seed=43)
        self.assertTrue(any(
            not np.array_equal(left, right)
            for left, right in zip(first.best_network.parameters(), second.best_network.parameters())
        ))

    def test_elitism_keeps_best_fitness_nondecreasing(self) -> None:
        result = evolve(5, 2, (0, 1), 0.05, 4, seed=42)
        summaries = result.generation_summaries
        self.assertEqual(len(summaries), 4)
        self.assertTrue(all(
            later.best_fitness >= earlier.best_fitness
            for earlier, later in zip(summaries, summaries[1:])
        ))
        for summary in summaries:
            self.assertEqual(len(summary.individual_fitnesses), 5)
            self.assertEqual(summary.mean_fitness, fmean(summary.individual_fitnesses))
            self.assertEqual(summary.best_fitness, max(summary.individual_fitnesses))

    def test_invalid_configuration_is_rejected(self) -> None:
        defaults = dict(population_size=3, elite_count=1, scenario_seeds=(0,),
                        mutation_scale=0.05, generations=1, seed=42)
        invalid = (
            ("population_size", 0),
            ("elite_count", 0),
            ("elite_count", 4),
            ("generations", 0),
            ("scenario_seeds", ()),
            ("mutation_scale", -0.1),
            ("mutation_scale", float("nan")),
            ("mutation_scale", float("inf")),
        )
        for name, value in invalid:
            with self.subTest(parameter=name, value=value), self.assertRaisesRegex(ValueError, name):
                evolve(**(defaults | {name: value}))

    def test_global_python_and_numpy_rng_states_are_unchanged(self) -> None:
        python_before = random.getstate()
        numpy_before = np.random.get_state()

        evolve(3, 1, (0, 1), 0.05, 2, seed=42)

        self.assertEqual(random.getstate(), python_before)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])


if __name__ == "__main__":
    unittest.main()
