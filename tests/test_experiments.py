"""Focused tests for seeded experiment runs and aggregation."""

import random
import unittest
from unittest.mock import patch

from ai_sandbox.contracts import Action, Observation
from ai_sandbox.controllers import RandomController, RuleBasedController
from ai_sandbox.experiments import (
    ExperimentResult, ExperimentSummary, run_experiment, summarize_results,
)
from ai_sandbox.scenarios import create_random_episode


class StayController:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def choose_action(self, observation: Observation) -> Action:
        return Action.STAY


class ExperimentTests(unittest.TestCase):
    def test_one_result_per_requested_seed_in_order(self) -> None:
        results = run_experiment([7, 2, 7], StayController)
        self.assertIsInstance(results, tuple)
        self.assertEqual(len(results), 3)
        self.assertEqual([result.scenario_seed for result in results], [7, 2, 7])
        self.assertEqual([result.controller_name for result in results], ["StayController"] * 3)

    def test_result_name_comes_from_controller_instance(self) -> None:
        def differently_named_factory(seed: int) -> StayController:
            return StayController(seed)

        result, = run_experiment([3], differently_named_factory)
        self.assertEqual(result.controller_name, "StayController")

    def test_every_run_uses_fresh_episodes_worlds_and_controllers(self) -> None:
        episodes = []
        controllers = []

        def make_episode(seed: int):
            episode = create_random_episode(seed)
            episodes.append(episode)
            return episode

        def make_controller(seed: int) -> StayController:
            controller = StayController(seed)
            controllers.append(controller)
            return controller

        with patch("ai_sandbox.experiments.create_random_episode", side_effect=make_episode):
            results = run_experiment([4, 4], make_controller)

        self.assertEqual(len(episodes), 2)
        self.assertIsNot(episodes[0], episodes[1])
        self.assertIsNot(episodes[0].world, episodes[1].world)
        self.assertIsNot(controllers[0], controllers[1])
        self.assertEqual([controller.seed for controller in controllers], [4, 4])
        self.assertEqual([result.controller_name for result in results], ["StayController"] * 2)
        self.assertEqual(episodes[0].world.food_positions, episodes[1].world.food_positions)
        self.assertTrue(all(episode.terminated for episode in episodes))

    def test_same_inputs_produce_identical_results(self) -> None:
        first = run_experiment(range(3), RandomController)
        second = run_experiment(range(3), RandomController)
        self.assertEqual(first, second)

    def test_both_baselines_use_the_same_experiment_function(self) -> None:
        scenarios = []

        def make_episode(seed: int):
            episode = create_random_episode(seed)
            scenarios.append((episode.world.agent_position, episode.world.food_positions))
            return episode

        with patch("ai_sandbox.experiments.create_random_episode", side_effect=make_episode):
            for controller_type in (RandomController, RuleBasedController):
                with self.subTest(controller=controller_type.__name__):
                    results = run_experiment([0, 1], controller_type)
                    self.assertEqual(len(results), 2)
                    self.assertEqual([result.scenario_seed for result in results], [0, 1])
                    self.assertEqual(
                        [result.controller_name for result in results],
                        [controller_type.__name__] * 2,
                    )
        self.assertEqual(scenarios[:2], scenarios[2:])

    def test_aggregate_arithmetic_means(self) -> None:
        results = (
            ExperimentResult(0, "Test", 100, 1, 80, 0),
            ExperimentResult(1, "Test", 150, 3, 120, 10),
        )
        self.assertEqual(
            summarize_results(results),
            ExperimentSummary(2, 125.0, 2.0, 100.0, 5.0),
        )

    def test_empty_aggregate_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            summarize_results(())

    def test_experiment_does_not_change_module_random_state(self) -> None:
        before = random.getstate()
        run_experiment(range(3), RandomController)
        self.assertEqual(random.getstate(), before)


if __name__ == "__main__":
    unittest.main()
