"""Focused tests for seeded experiment runs and aggregation."""

import random
import unittest
from unittest.mock import patch

from ai_sandbox.contracts import Action, Observation
from ai_sandbox.controllers import RandomController, RuleBasedController
from ai_sandbox.experiments import (
    ExperimentResult, ExperimentSummary, MetricSummary, PairedComparison,
    ResourceComparison, compare_paired_metric, compare_resources, run_experiment,
    summarize_results,
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

    def test_aggregate_mean_median_minimum_and_maximum_with_even_count(self) -> None:
        results = (
            ExperimentResult(0, "Test", 100, 1, 80, 0),
            ExperimentResult(1, "Test", 150, 3, 120, 10),
        )
        self.assertEqual(
            summarize_results(results),
            ExperimentSummary(
                episodes=2,
                ticks_survived=MetricSummary(125.0, 125.0, 100, 150),
                resources_collected=MetricSummary(2.0, 2.0, 1, 3),
                distance_travelled=MetricSummary(100.0, 100.0, 80, 120),
                final_energy=MetricSummary(5.0, 5.0, 0, 10),
            ),
        )

    def test_aggregate_median_with_odd_count(self) -> None:
        results = (
            ExperimentResult(0, "Test", 100, 0, 80, 0),
            ExperimentResult(1, "Test", 200, 5, 180, 20),
            ExperimentResult(2, "Test", 120, 1, 100, 10),
        )
        summary = summarize_results(results)
        self.assertEqual(summary.ticks_survived.median, 120)
        self.assertEqual(summary.resources_collected.median, 1)
        self.assertEqual(summary.distance_travelled.median, 100)
        self.assertEqual(summary.final_energy.median, 10)

    def test_resource_comparison_matches_seeds_not_positions(self) -> None:
        first = (
            ExperimentResult(0, "First", 100, 2, 0, 0),
            ExperimentResult(1, "First", 100, 1, 0, 0),
            ExperimentResult(2, "First", 100, 3, 0, 0),
            ExperimentResult(3, "First", 100, 0, 0, 0),
        )
        second = (
            ExperimentResult(3, "Second", 100, 0, 0, 0),
            ExperimentResult(2, "Second", 100, 1, 0, 0),
            ExperimentResult(1, "Second", 100, 2, 0, 0),
            ExperimentResult(0, "Second", 100, 1, 0, 0),
        )
        self.assertEqual(compare_resources(first, second), ResourceComparison(4, 2, 1, 1))

    def test_resource_comparison_rejects_mismatched_seed_sets(self) -> None:
        first = (ExperimentResult(0, "First", 100, 1, 0, 0),)
        second = (ExperimentResult(1, "Second", 100, 1, 0, 0),)
        with self.assertRaisesRegex(ValueError, "matching scenario seeds"):
            compare_resources(first, second)

    def test_resource_comparison_rejects_duplicate_seeds_on_either_side(self) -> None:
        first = (ExperimentResult(0, "First", 100, 1, 0, 0),)
        second = (ExperimentResult(0, "Second", 100, 1, 0, 0),)
        for duplicate_first, duplicate_second in (
            (first + first, second),
            (first, second + second),
        ):
            with self.subTest(duplicate_first=len(duplicate_first) == 2):
                with self.assertRaisesRegex(ValueError, "duplicate scenario seeds"):
                    compare_resources(duplicate_first, duplicate_second)

    def test_paired_metric_counts_wins_losses_and_ties_by_seed(self) -> None:
        first = (
            ExperimentResult(2, "First", 150, 0, 0, 0),
            ExperimentResult(0, "First", 100, 0, 0, 0),
            ExperimentResult(1, "First", 125, 0, 0, 0),
        )
        second = (
            ExperimentResult(0, "Second", 90, 0, 0, 0),
            ExperimentResult(1, "Second", 150, 0, 0, 0),
            ExperimentResult(2, "Second", 150, 0, 0, 0),
        )
        self.assertEqual(
            compare_paired_metric(first, second, lambda result: result.ticks_survived),
            PairedComparison(3, 1, 1, 1),
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
