"""Small checks for the held-out evaluation composition."""

from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

import numpy as np

from ai_sandbox.experiments import ExperimentResult
from ai_sandbox.neural_evolution import EvolutionResult, GenerationSummary
from ai_sandbox.neural_network import TinyNeuralNetwork
from ai_sandbox.scenarios import create_random_episode
from scripts.run_evolved_evaluation import (
    HELD_OUT_SEEDS, TRAINING_SEEDS, evaluate_held_out, main,
)


class EvolvedEvaluationTests(unittest.TestCase):
    def test_held_out_seeds_are_disjoint_from_training_seeds(self) -> None:
        self.assertEqual(TRAINING_SEEDS, (0, 1, 2, 3, 4))
        self.assertEqual(HELD_OUT_SEEDS, tuple(range(100, 200)))
        self.assertTrue(set(TRAINING_SEEDS).isdisjoint(HELD_OUT_SEEDS))

    def test_same_seeds_fresh_worlds_reproducibility_and_champion_unchanged(self) -> None:
        champion = TinyNeuralNetwork.random(seed=42)
        before = champion.parameters()
        episodes = []

        def make_episode(seed):
            episode = create_random_episode(seed)
            episodes.append((seed, episode, episode.world.food_positions))
            return episode

        with patch("ai_sandbox.experiments.create_random_episode", side_effect=make_episode):
            first = evaluate_held_out(champion, (100, 101))
            second = evaluate_held_out(champion, (100, 101))

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertTrue(all(
            tuple(result.scenario_seed for result in results) == (100, 101)
            for results in first
        ))
        self.assertEqual(len(episodes), 12)
        for seed in (100, 101):
            matching = [
                (episode, initial_food)
                for seen_seed, episode, initial_food in episodes if seen_seed == seed
            ]
            self.assertEqual(len(matching), 6)
            self.assertEqual(len({id(episode) for episode, _ in matching}), 6)
            self.assertEqual(len({id(episode.world) for episode, _ in matching}), 6)
            self.assertEqual(len({initial_food for _, initial_food in matching}), 1)
        for original, current in zip(before, champion.parameters()):
            np.testing.assert_array_equal(original, current)

    def test_main_trains_once_and_passes_one_champion_to_held_out_test(self) -> None:
        champion = TinyNeuralNetwork.random(seed=42)
        training = EvolutionResult(
            champion,
            (GenerationSummary(29, 125.0, 110.3, (125.0,)),),
        )
        results = (
            (ExperimentResult(100, "RandomController", 100, 0, 70, 0),),
            (ExperimentResult(100, "RuleBasedController", 100, 0, 90, 0),),
            (ExperimentResult(100, "NeuralController", 125, 1, 90, 0),),
        )
        with (
            patch("scripts.run_evolved_evaluation.evolve", return_value=training) as train,
            patch("scripts.run_evolved_evaluation.evaluate_held_out", return_value=results) as test,
            redirect_stdout(StringIO()),
        ):
            main()

        train.assert_called_once_with(
            population_size=40,
            elite_count=8,
            scenario_seeds=TRAINING_SEEDS,
            mutation_scale=0.05,
            generations=30,
            seed=42,
        )
        test.assert_called_once_with(champion, HELD_OUT_SEEDS)


if __name__ == "__main__":
    unittest.main()
