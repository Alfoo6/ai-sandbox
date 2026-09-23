"""Tests for seeded neural survival fitness."""

import random
import unittest
from unittest.mock import patch

import numpy as np

from ai_sandbox.neural_fitness import NeuralFitnessResult, evaluate_network
from ai_sandbox.neural_network import TinyNeuralNetwork
from ai_sandbox.runner import EpisodeResult
from ai_sandbox.scenarios import create_random_episode


class NeuralFitnessTests(unittest.TestCase):
    def test_preserves_each_seed_in_order_and_uses_arithmetic_mean(self) -> None:
        network = TinyNeuralNetwork.random(seed=42)
        results = (
            EpisodeResult(100, 10, 0, 0),
            EpisodeResult(150, 0, 0, 0),
            EpisodeResult(200, 2, 0, 0),
        )
        with patch("ai_sandbox.neural_fitness.run_episode", side_effect=results) as run:
            result = evaluate_network(network, iter((7, 2, 7)))

        self.assertEqual(result, NeuralFitnessResult(150.0, (7, 2, 7), (100, 150, 200)))
        self.assertEqual(run.call_count, 3)

    def test_same_network_and_seeds_give_same_result_without_mutating_network(self) -> None:
        network = TinyNeuralNetwork.random(seed=42)
        before = network.parameters()
        first = evaluate_network(network, (0, 1, 2))
        second = evaluate_network(network, (0, 1, 2))

        self.assertEqual(first, second)
        for original, current in zip(before, network.parameters()):
            np.testing.assert_array_equal(original, current)

    def test_each_seed_gets_fresh_episode_and_world(self) -> None:
        episodes = []

        def make_episode(seed: int):
            episode = create_random_episode(seed)
            episodes.append(episode)
            return episode

        network = TinyNeuralNetwork.random(seed=42)
        with patch("ai_sandbox.neural_fitness.create_random_episode", side_effect=make_episode):
            result = evaluate_network(network, (4, 4))

        self.assertEqual(result.scenario_seeds, (4, 4))
        self.assertEqual(len(episodes), 2)
        self.assertIsNot(episodes[0], episodes[1])
        self.assertIsNot(episodes[0].world, episodes[1].world)
        self.assertEqual(episodes[0].world.food_positions, episodes[1].world.food_positions)
        self.assertTrue(all(episode.terminated for episode in episodes))

    def test_empty_seeds_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "scenario_seeds must not be empty"):
            evaluate_network(TinyNeuralNetwork.random(seed=42), iter(()))

    def test_global_python_and_numpy_rng_states_are_unchanged(self) -> None:
        network = TinyNeuralNetwork.random(seed=42)
        python_before = random.getstate()
        numpy_before = np.random.get_state()

        evaluate_network(network, (0, 1, 2))

        self.assertEqual(random.getstate(), python_before)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])

    def test_seed_42_network_integrates_over_three_scenarios(self) -> None:
        network = TinyNeuralNetwork.random(26, 12, 5, seed=42)
        result = evaluate_network(network, (0, 1, 2))
        self.assertEqual(result.scenario_seeds, (0, 1, 2))
        self.assertEqual(len(result.ticks_survived), 3)
        self.assertEqual(result.fitness, sum(result.ticks_survived) / 3)


if __name__ == "__main__":
    unittest.main()
