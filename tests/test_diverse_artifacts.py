"""Persistence and CLI tests use at most two tiny generations."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from ai_sandbox.controllers import NeuralController
from ai_sandbox.diverse_artifacts import MODEL_METADATA, load_network, run_diverse_experiment, save_network
from ai_sandbox.diverse_evolution import DiverseConfig, evolve_diverse
from ai_sandbox.experiments import run_experiment
from ai_sandbox.neural_network import TinyNeuralNetwork
from scripts.run_diverse_evolution import main, parse_args


TINY = DiverseConfig(population_size=2, elite_count=1, generations=2,
                     scenarios_per_generation=1, validation_interval=1,
                     validation_seeds=(200, 201))


class DiverseArtifactsTests(unittest.TestCase):
    def test_model_round_trip_preserves_parameters_outputs_and_metadata(self):
        network = TinyNeuralNetwork.random(seed=7)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            save_network(path, network)
            loaded = load_network(path)
            for original, current in zip(network.parameters(), loaded.parameters()):
                np.testing.assert_array_equal(original, current)
            self.assertEqual(network.forward(np.arange(26) / 26), loaded.forward(np.arange(26) / 26))
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(json.loads(archive["metadata"].item()), MODEL_METADATA)
                self.assertEqual(set(archive.files), {"metadata", "w1", "b1", "w2", "b2"})
            self.assertFalse(path.with_suffix(".npz.tmp").exists())

    def test_invalid_models_are_rejected(self):
        network = TinyNeuralNetwork.random()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            with self.assertRaises(ValueError):
                save_network(path, TinyNeuralNetwork.random(hidden_size=3))
            parameters = dict(zip(("w1", "b1", "w2", "b2"), network.parameters()))
            for metadata, changes in (
                (MODEL_METADATA | {"format_version": 2}, {}),
                (MODEL_METADATA, {"w1": np.zeros((1, 1))}),
                (MODEL_METADATA, {"b2": np.full(5, np.nan)}),
            ):
                np.savez(path, metadata=json.dumps(metadata), **(parameters | changes))
                with self.assertRaises(ValueError):
                    load_network(path)

    def test_artifacts_match_result_and_loaded_champion_validation(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()) as output:
            destination = Path(directory) / "run"
            result = run_diverse_experiment(TINY, destination)
            config = json.loads((destination / "config.json").read_text())
            self.assertEqual(config["configuration"], json.loads(json.dumps(asdict(TINY))))
            self.assertEqual(config["architecture"], MODEL_METADATA)
            history = [json.loads(line) for line in (destination / "generations.jsonl").read_text().splitlines()]
            self.assertEqual(len(history), 2)
            for recorded, expected in zip(history, result.generations):
                self.assertGreaterEqual(recorded.pop("elapsed_seconds"), 0)
                self.assertEqual(recorded, json.loads(json.dumps(asdict(expected))))
            final = json.loads((destination / "result.json").read_text())
            self.assertEqual(final["status"], "complete")
            self.assertEqual(final["completed_generations"], 2)
            self.assertEqual(final["best_validation_generation"], result.best_validation_champion.generation)
            self.assertFalse(final["resumable_checkpoint"])
            self.assertGreaterEqual(final["elapsed_seconds"], 0)
            champion = load_network(destination / final["champion_file"])
            actual = run_experiment(TINY.validation_seeds, lambda _: NeuralController(champion))
            self.assertEqual(actual, result.best_validation_champion.validation.episodes)
            self.assertEqual(final["best_validation"]["episodes"], [asdict(e) for e in actual])
            self.assertEqual(len(list(destination.glob("*.npz"))), sum(
                r.selected_for_validation_champion for r in result.generations
            ))
            with self.assertRaises(FileExistsError):
                run_diverse_experiment(TINY, destination)
            self.assertIn("Generation 2/2", output.getvalue())
            self.assertIn("validation mean ticks=", output.getvalue())
            self.assertIn("elapsed=", output.getvalue())

    def test_disabled_validation_produces_no_champion(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            destination = Path(directory) / "run"
            run_diverse_experiment(replace(TINY, validation_interval=0), destination)
            final = json.loads((destination / "result.json").read_text())
            self.assertIsNone(final["best_validation"])
            self.assertIsNone(final["champion_file"])
            self.assertEqual(list(destination.glob("*.npz")), [])

    def test_interrupted_run_keeps_completed_history_and_model_without_final_summary(self):
        tiny_result = evolve_diverse(replace(TINY, generations=1))

        def interrupt(config, on_generation):
            on_generation(tiny_result.generations[0], tiny_result.best_validation_champion)
            raise KeyboardInterrupt

        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            destination = Path(directory) / "run"
            with patch("ai_sandbox.diverse_artifacts.evolve_diverse", side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    run_diverse_experiment(TINY, destination)
            self.assertTrue((destination / "config.json").exists())
            self.assertEqual(len((destination / "generations.jsonl").read_text().splitlines()), 1)
            self.assertEqual(len(list(destination.glob("*.npz"))), 1)
            self.assertFalse((destination / "result.json").exists())

    def test_cli_defaults_and_wiring_without_training(self):
        args = parse_args([])
        self.assertEqual(args.config, DiverseConfig())
        self.assertEqual(args.output_dir.parent, Path("results"))
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "run"
            argv = ["--generations", "2", "--population", "2", "--elites", "1",
                    "--scenarios-per-generation", "1", "--mutation-scale", "0",
                    "--seed", "4", "--scenario-schedule-seed", "9", "--validation-interval", "0",
                    "--output-dir", str(destination)]
            with patch("scripts.run_diverse_evolution.run_diverse_experiment") as run:
                main(argv)
            expected = DiverseConfig(population_size=2, elite_count=1, generations=2,
                                     scenarios_per_generation=1, mutation_scale=0,
                                     seed=4, scenario_schedule_seed=9, validation_interval=0)
            run.assert_called_once_with(expected, destination)
            self.assertFalse(destination.exists())

    def test_cli_invalid_arguments_never_start_training(self):
        with TemporaryDirectory() as directory:
            cases = (
                ["--generations", "0"], ["--generations", "1.5"],
                ["--population", "0"], ["--elites", "0"], ["--elites", "13"],
                ["--scenarios-per-generation", "0"], ["--mutation-scale", "-1"],
                ["--mutation-scale", "nan"], ["--mutation-scale", "inf"],
                ["--seed", "-1"], ["--scenario-schedule-seed", "-1"],
                ["--validation-interval", "-1"], ["--output-dir", directory],
            )
            for argv in cases:
                with self.subTest(argv=argv), redirect_stderr(StringIO()):
                    with patch("scripts.run_diverse_evolution.run_diverse_experiment") as run:
                        with self.assertRaises(SystemExit) as error:
                            main(argv)
                        self.assertEqual(error.exception.code, 2)
                        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
