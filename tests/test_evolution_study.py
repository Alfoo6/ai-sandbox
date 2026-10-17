"""Fast integration and arithmetic checks for repeated evolution studies."""

from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError, replace
from io import StringIO
import unittest
from unittest.mock import call, patch

import numpy as np

from ai_sandbox.controllers import RandomController, RuleBasedController
from ai_sandbox.evolution_study import (
    StudyAggregate, StudyConfig, StudyRun, ValidationResult, aggregate_runs, run_study,
)
from ai_sandbox.experiments import ExperimentResult, run_experiment, summarize_results
from ai_sandbox.neural_evolution import evolve
from scripts.run_evolution_study import CONFIG, main


SMALL = StudyConfig(3, 1, (0, 1), 0.05, 2, (42, 43), (200, 201))


class EvolutionStudyTests(unittest.TestCase):
    def test_initial_configuration(self) -> None:
        self.assertEqual(CONFIG, StudyConfig(
            40, 8, (0, 1, 2, 3, 4), 0.05, 30, (42, 43, 44), tuple(range(200, 250))
        ))

    def test_configuration_freezes_seed_collections(self) -> None:
        seeds = [200, 201]
        config = replace(SMALL, validation_seeds=seeds)
        seeds.append(202)
        self.assertEqual(config.validation_seeds, (200, 201))
        with self.assertRaises(FrozenInstanceError):
            config.generations = 5

    def test_invalid_configurations(self) -> None:
        invalid = (
            {"population_size": 0}, {"population_size": 1.5},
            {"elite_count": 0}, {"elite_count": 4}, {"generations": 0},
            {"generations": True}, {"mutation_scale": -0.1},
            {"mutation_scale": float("nan")}, {"mutation_scale": float("inf")},
            {"validation_seeds": (1, 200)},
        )
        for name in ("training_seeds", "validation_seeds", "evolution_seeds"):
            invalid += tuple({name: value} for value in ((), (2, 2), (-1,), (1.5,)))
        for name in ("training_seeds", "validation_seeds"):
            invalid += tuple({name: (seed,)} for seed in (1000, 1050, 1099))
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(SMALL, **change)

    def test_repeated_studies_are_deterministic(self) -> None:
        first, second = run_study(SMALL), run_study(SMALL)
        self.assertEqual(first.aggregate, second.aggregate)
        self.assertEqual(first.reference_baselines, second.reference_baselines)
        for left, right in zip(first.runs, second.runs):
            self.assertEqual(left.evolution_seed, right.evolution_seed)
            self.assertEqual(left.training.generation_summaries, right.training.generation_summaries)
            self.assertEqual(left.validation, right.validation)
            for a, b in zip(left.training.best_network.parameters(), right.training.best_network.parameters()):
                np.testing.assert_array_equal(a, b)

    def test_independent_training_shared_validation_and_unchanged_champions(self) -> None:
        snapshots = []

        def train(**kwargs):
            result = evolve(**kwargs)
            snapshots.append((result, result.best_network.parameters()))
            return result

        with (
            patch("ai_sandbox.evolution_study.evolve", side_effect=train) as training,
            patch("ai_sandbox.evolution_study.run_experiment", wraps=run_experiment) as evaluation,
        ):
            study = run_study(SMALL)
        self.assertEqual(training.call_args_list, [
            call(population_size=3, elite_count=1, scenario_seeds=(0, 1),
                 mutation_scale=0.05, generations=2, seed=seed)
            for seed in SMALL.evolution_seeds
        ])
        self.assertEqual(evaluation.call_count, 4)
        self.assertTrue(all(args.args[0] == SMALL.validation_seeds for args in evaluation.call_args_list))
        self.assertEqual(
            [args.args[1] for args in evaluation.call_args_list[-2:]],
            [RandomController, RuleBasedController],
        )
        for run, (original, parameters) in zip(study.runs, snapshots):
            self.assertIs(run.training, original)
            self.assertEqual(run.final_training_best_fitness, original.generation_summaries[-1].best_fitness)
            self.assertEqual(tuple(e.scenario_seed for e in run.validation.episodes), SMALL.validation_seeds)
            for before, after in zip(parameters, run.training.best_network.parameters()):
                np.testing.assert_array_equal(before, after)
        first, second = study.runs
        self.assertIsNot(first.training.best_network, second.training.best_network)
        self.assertTrue(any(
            not np.array_equal(a, b) for a, b in zip(
                first.training.best_network.parameters(), second.training.best_network.parameters()
            )
        ))
        # Running seed 43 alone must reproduce it, independent of the prior seed 42 run.
        alone = run_study(replace(SMALL, evolution_seeds=(43,))).runs[0]
        self.assertEqual(second.validation, alone.validation)
        self.assertEqual(second.training.generation_summaries, alone.training.generation_summaries)
        for a, b in zip(second.training.best_network.parameters(), alone.training.best_network.parameters()):
            np.testing.assert_array_equal(a, b)
        for baseline, controller in zip(study.reference_baselines, (RandomController, RuleBasedController)):
            self.assertEqual(baseline.episodes, run_experiment(SMALL.validation_seeds, controller))

    def test_aggregation_uses_run_means_and_population_sd(self) -> None:
        def run(seed, ticks, resources):
            episodes = tuple(
                ExperimentResult(200 + i, "NeuralController", t, r, 0, 0)
                for i, (t, r) in enumerate(zip(ticks, resources))
            )
            return StudyRun(seed, None, ValidationResult(episodes, summarize_results(episodes)))

        first = run(42, (100, 200), (0, 4))
        second = run(43, (100, 100), (0, 0))
        self.assertEqual(aggregate_runs((first, second)), StudyAggregate(125, 25, 1, 1))
        self.assertEqual(aggregate_runs((first,)), StudyAggregate(150, 0, 2, 0))
        self.assertEqual(aggregate_runs((first, first)), StudyAggregate(150, 0, 2, 0))
        self.assertEqual(first.validation.summary.ticks_survived.median, 150)
        self.assertEqual(first.validation.episodes_collecting_resource, 1)
        self.assertEqual(first.validation.episodes_past_100_ticks, 1)
        with self.assertRaises(ValueError):
            aggregate_runs(())

    def test_script_prints_study_and_reference_baselines(self) -> None:
        study = run_study(SMALL)
        output = StringIO()
        with patch("scripts.run_evolution_study.run_study", return_value=study) as run, redirect_stdout(output):
            main()
        run.assert_called_once_with(CONFIG)
        for label in ("Train best", "Median ticks", "Reference baselines", "RandomController",
                      "RuleBasedController", "ddof=0", "not episode variation", "Runtime:"):
            self.assertIn(label, output.getvalue())


if __name__ == "__main__":
    unittest.main()
