"""Tiny simulations and mocked fitness checks; never run a real experiment."""

from dataclasses import replace
import random
import unittest
from unittest.mock import patch

import numpy as np

from ai_sandbox.diverse_evolution import (
    DiverseConfig, GENERATION_SEED_SPAN, evolve_diverse, scenario_batch,
)
from ai_sandbox.experiments import ExperimentResult
from ai_sandbox.neural_evolution import evolve, next_generation
from ai_sandbox.neural_fitness import NeuralFitnessResult


TINY = DiverseConfig(population_size=3, elite_count=1, scenarios_per_generation=2,
                     generations=3, validation_interval=1, validation_seeds=(200, 201))


class DiverseEvolutionTests(unittest.TestCase):
    def assert_parameters_equal(self, first, second):
        for a, b in zip(first.parameters(), second.parameters()):
            np.testing.assert_array_equal(a, b)

    def test_schedule_reproducible_unique_disjoint_and_independent_of_order(self):
        first = tuple(scenario_batch(g, 5, 123) for g in range(4))
        second = tuple(scenario_batch(g, 5, 123) for g in reversed(range(4)))
        self.assertEqual(first, second[::-1])
        self.assertNotEqual(first[0], scenario_batch(0, 5, 124))
        flattened = sum(first, ())
        self.assertEqual(len(set(flattened)), 20)
        self.assertTrue(all(seed >= 10_000 for seed in flattened))
        self.assertTrue(set(flattened).isdisjoint(range(200, 1100)))

    def test_invalid_schedules_and_configurations(self):
        for args in ((-1, 2, 1), (0, 0, 1), (0, 1, -1), (0, GENERATION_SEED_SPAN + 1, 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                scenario_batch(*args)
        for name, value in (
            ("population_size", 0), ("elite_count", 4), ("elite_count", 0),
            ("generations", 0), ("generations", 1.5), ("generations", True),
            ("seed", -1), ("scenario_schedule_seed", -1), ("validation_interval", -1),
            ("scenarios_per_generation", 0), ("scenarios_per_generation", GENERATION_SEED_SPAN + 1),
            ("mutation_scale", -0.1), ("mutation_scale", float("nan")),
            ("mutation_scale", float("inf")), ("validation_seeds", ()),
            ("validation_seeds", (200, 200)), ("validation_seeds", (199,)),
            ("validation_seeds", (1000,)), ("validation_seeds", (1099,)),
            ("validation_seeds", (10_000,)),
        ):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                replace(TINY, **{name: value})

    def test_same_batch_for_all_individuals_and_reevaluation_of_unchanged_elites(self):
        calls = []
        populations = []
        scores = iter((300, 200, 100, 100, 90, 150, 70, 60, 50))

        def evaluate(network, seeds):
            calls.append((network, seeds, network.parameters()))
            fitness = next(scores)
            return NeuralFitnessResult(fitness, seeds, (fitness,) * len(seeds))

        def breed(population, fitnesses, *args):
            populations.append((population, fitnesses))
            return next_generation(population, fitnesses, *args)

        with (
            patch("ai_sandbox.neural_evolution.evaluate_network", side_effect=evaluate),
            patch("ai_sandbox.diverse_evolution.next_generation", side_effect=breed),
        ):
            result = evolve_diverse(replace(TINY, validation_interval=0))
        self.assertEqual(len(calls), 9)
        for generation in range(3):
            expected = scenario_batch(generation, 2, TINY.scenario_schedule_seed)
            self.assertEqual([seeds for _, seeds, _ in calls[generation * 3:generation * 3 + 3]], [expected] * 3)
        # Elite from generation 0 is preserved but reevaluated at 100, not reused at 300.
        self.assertIs(calls[0][0], calls[3][0])
        for before, after in zip(calls[0][2], calls[3][2]):
            np.testing.assert_array_equal(before, after)
        self.assertEqual(populations[1][1], (100, 90, 150))
        self.assertIs(calls[5][0], calls[6][0])
        self.assertEqual([r.training.best_fitness for r in result.generations], [300, 150, 70])
        self.assertIs(result.final_batch_champion, calls[6][0])
        self.assertIsNone(result.best_validation_champion)

    def test_determinism_and_global_rng_independence(self):
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        first = evolve_diverse(TINY)
        second = evolve_diverse(TINY)
        self.assertEqual(first.generations, second.generations)
        self.assert_parameters_equal(first.final_batch_champion, second.final_batch_champion)
        self.assertEqual(first.best_validation_champion.generation, second.best_validation_champion.generation)
        self.assert_parameters_equal(first.best_validation_champion.network, second.best_validation_champion.network)
        self.assertEqual(random.getstate(), python_before)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])

    def test_validation_does_not_change_breeding_or_scenario_schedule(self):
        enabled = evolve_diverse(TINY)
        disabled = evolve_diverse(replace(TINY, validation_interval=0))
        self.assertEqual([r.training for r in enabled.generations], [r.training for r in disabled.generations])
        self.assertEqual([r.scenario_seeds for r in enabled.generations], [r.scenario_seeds for r in disabled.generations])
        self.assert_parameters_equal(enabled.final_batch_champion, disabled.final_batch_champion)

    def test_validation_selection_improvements_ties_and_retention(self):
        validation_scores = iter((100, 150, 150, 125))
        evaluated_networks = []
        snapshots = []
        selected_generations = []

        def validate(seeds, factory):
            network = factory(200)._network
            evaluated_networks.append(network)
            snapshots.append(network.parameters())
            score = next(validation_scores)
            self.assertEqual(seeds, TINY.validation_seeds)
            return tuple(ExperimentResult(seed, "NeuralController", score, 0, 0, 0) for seed in seeds)

        def report(record, selected):
            if selected is not None:
                selected_generations.append(selected.generation)

        with patch("ai_sandbox.diverse_evolution.run_experiment", side_effect=validate):
            result = evolve_diverse(replace(TINY, generations=4), report)
        self.assertEqual(selected_generations, [0, 1])
        self.assertEqual(result.best_validation_champion.generation, 1)
        self.assertIs(result.best_validation_champion.network, evaluated_networks[1])
        for network, snapshot in zip(evaluated_networks, snapshots):
            for before, after in zip(snapshot, network.parameters()):
                np.testing.assert_array_equal(before, after)

    def test_interval_includes_final_generation_and_zero_disables(self):
        for interval, expected in ((2, [1, 2]), (10, [2]), (0, [])):
            with self.subTest(interval=interval):
                result = evolve_diverse(replace(TINY, validation_interval=interval))
                self.assertEqual([r.training.generation for r in result.generations if r.validation], expected)
                if interval == 0:
                    self.assertIsNone(result.best_validation_champion)

    def test_fixed_batch_matches_original_evolve_including_parameters(self):
        # Tiny regression: with a constant schedule, the reused learning logic must
        # exactly match the untouched original loop, including RNG consumption.
        original = evolve(3, 1, (0, 1), 0.05, 3, 42)
        with patch("ai_sandbox.diverse_evolution.scenario_batch", return_value=(0, 1)):
            diverse = evolve_diverse(replace(TINY, validation_interval=0))
        self.assertEqual(original.generation_summaries, tuple(r.training for r in diverse.generations))
        self.assert_parameters_equal(original.best_network, diverse.final_batch_champion)


if __name__ == "__main__":
    unittest.main()
