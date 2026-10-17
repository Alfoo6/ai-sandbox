"""Saved inference wiring uses synthetic results; never trains or opens a GUI."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from ai_sandbox.controllers import NeuralController, RandomController, RuleBasedController
from ai_sandbox.diverse_artifacts import MODEL_METADATA, save_network
from ai_sandbox.experiments import ExperimentResult
from ai_sandbox.neural_network import TinyNeuralNetwork
from ai_sandbox.saved_champion import evaluate_saved_champion, load_champion, write_report
from ai_sandbox.scenarios import create_random_episode
from scripts.evaluate_saved_champion import main as evaluate_main
from scripts.watch_saved_champion import main as watch_main


class SavedChampionTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(TemporaryDirectory()))
        self.model = self.directory / "model.npz"
        self.network = TinyNeuralNetwork.random(seed=7)
        save_network(self.model, self.network)
        for target in ("ai_sandbox.neural_evolution.evolve",
                       "ai_sandbox.diverse_evolution.evolve_diverse",
                       "ai_sandbox.diverse_artifacts.evolve_diverse"):
            self.stack.enter_context(patch(target, side_effect=AssertionError("retraining forbidden")))

    def fake_experiment(self, seeds, factory):
        results = []
        for index, seed in enumerate(seeds):
            controller = factory(seed)
            name = type(controller).__name__
            if isinstance(controller, NeuralController):
                ticks, food = ((150, 2), (100, 0), (100, 0))[index]
            else:
                # Verify RNG seeding through a sequence of observable actions.
                reference = type(controller)(seed)
                observation = create_random_episode(seed).observe()
                self.assertEqual([controller.choose_action(observation) for _ in range(20)],
                                 [reference.choose_action(observation) for _ in range(20)])
                ticks, food = ((100, 0), (125, 1), (100, 0))[index]
            results.append(ExperimentResult(seed, name, ticks, food, 0, 0))
        return tuple(results)

    def test_report_seed_consistency_pairs_hash_json_and_reproducibility(self):
        before = self.model.read_bytes()
        with patch("ai_sandbox.saved_champion.run_experiment", side_effect=self.fake_experiment) as run:
            first = evaluate_saved_champion(self.model, 7, 3)
            second = evaluate_saved_champion(self.model, 7, 3)
        self.assertEqual(first, second)
        self.assertEqual([call.args[0] for call in run.call_args_list], [(7, 8, 9)] * 6)
        self.assertIs(run.call_args_list[1].args[1], RandomController)
        self.assertIs(run.call_args_list[2].args[1], RuleBasedController)
        self.assertEqual(first["model_sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(first["model_path"], str(self.model.resolve()))
        self.assertEqual(first["scenario_seeds"], [7, 8, 9])
        self.assertEqual(len(first["paired_comparisons"]), 4)
        for comparison in first["paired_comparisons"].values():
            self.assertEqual(comparison, dict(scenarios=3, first_wins=1, second_wins=1, ties=1))
        neural = first["controllers"]["neural"]
        self.assertEqual(neural["summary"]["ticks_survived"]["mean"], 350 / 3)
        self.assertEqual(neural["summary"]["ticks_survived"]["median"], 100)
        self.assertEqual(neural["summary"]["resources_collected"]["mean"], 2 / 3)
        self.assertEqual(neural["episodes_collecting_resource"], 1)
        self.assertEqual(neural["episodes_past_100_ticks"], 1)
        output = self.directory / "report.json"
        write_report(output, first)
        self.assertEqual(json.loads(output.read_text()), first)
        other = self.directory / "other.json"
        write_report(other, second)
        self.assertEqual(output.read_bytes(), other.read_bytes())
        with self.assertRaises(FileExistsError):
            write_report(output, second)
        with self.assertRaises(FileExistsError):
            write_report(self.model, second)
        self.assertEqual(self.model.read_bytes(), before)

    def test_loading_copies_parameters_and_inference_preserves_file(self):
        before = self.model.read_bytes()
        loaded = load_champion(self.model)
        for original, current in zip(self.network.parameters(), loaded.parameters()):
            np.testing.assert_array_equal(original, current)
        loaded.parameters()[0][:] = 99
        self.assertEqual(loaded.forward(np.zeros(26)), self.network.forward(np.zeros(26)))
        self.assertEqual(self.model.read_bytes(), before)

    def test_invalid_model_errors_include_path(self):
        path = self.directory / "bad.npz"
        parameters = dict(zip(("w1", "b1", "w2", "b2"), self.network.parameters()))
        with self.assertRaisesRegex(ValueError, "Cannot load model.*bad.npz"):
            load_champion(path)
        for metadata, changes in (
            (MODEL_METADATA | {"hidden_size": 3}, {}),
            (MODEL_METADATA, {"w1": np.zeros((1, 1))}),
            (MODEL_METADATA, {"b2": np.full(5, np.inf)}),
            (MODEL_METADATA, {"b2": np.full(5, "bad")}),
        ):
            np.savez(path, metadata=json.dumps(metadata), **(parameters | changes))
            with self.assertRaisesRegex(ValueError, "Cannot load model"):
                load_champion(path)
        np.savez(path, metadata=json.dumps(MODEL_METADATA))
        with self.assertRaisesRegex(ValueError, "Cannot load model"):
            load_champion(path)
        path.write_bytes(b"not a model")
        with self.assertRaisesRegex(ValueError, "Cannot load model"):
            load_champion(path)

    def test_evaluation_cli_wiring_and_no_overwrite(self):
        output = self.directory / "report.json"
        args = ["--model", str(self.model), "--seed-start", "7", "--count", "3",
                "--output", str(output)]
        with patch("ai_sandbox.saved_champion.run_experiment", side_effect=self.fake_experiment), redirect_stdout(StringIO()):
            evaluate_main(args)
        self.assertEqual(json.loads(output.read_text())["scenario_seeds"], [7, 8, 9])
        before = output.read_bytes()
        with patch("scripts.evaluate_saved_champion.evaluate_saved_champion") as evaluate, redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                evaluate_main(args)
            evaluate.assert_not_called()
        self.assertEqual(output.read_bytes(), before)

    def test_invalid_cli_inputs_do_not_run_episodes_or_viewer(self):
        output = self.directory / "report.json"
        with patch("ai_sandbox.saved_champion.run_experiment") as run, patch("scripts.watch_saved_champion.run_visual_episode") as view, redirect_stderr(StringIO()):
            for extra in (["--count", "0"], ["--count", "-1"], ["--seed-start", "x"]):
                with self.assertRaises(SystemExit):
                    evaluate_main(["--model", str(self.model), "--output", str(output), *extra])
            with self.assertRaises(SystemExit):
                evaluate_main(["--model", str(self.directory / "missing"), "--output", str(output)])
            with self.assertRaises(SystemExit):
                watch_main(["--model", str(self.directory / "missing")])
            run.assert_not_called()
            view.assert_not_called()
        self.assertFalse(output.exists())

    def test_viewer_loads_selected_scenario_and_existing_hud_without_training(self):
        before = self.model.read_bytes()
        with patch("scripts.watch_saved_champion.run_visual_episode") as view, patch("scripts.watch_saved_champion.create_random_episode", wraps=create_random_episode) as create:
            watch_main(["--model", str(self.model), "--scenario", "17"])
        create.assert_called_once_with(17)
        episode, controller = view.call_args.args
        self.assertIsInstance(controller, NeuralController)
        self.assertEqual(episode.observe(), create_random_episode(17).observe())
        self.assertEqual(view.call_args.kwargs, {"scenario_seed": 17})
        self.assertEqual(self.model.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
