"""Champion viewer wiring checks without opening a display or running training."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

from ai_sandbox.controllers import NeuralController
from ai_sandbox.neural_evolution import EvolutionResult, GenerationSummary
from ai_sandbox.neural_network import TinyNeuralNetwork
from ai_sandbox.scenarios import create_random_episode
from scripts.watch_evolved import main, parse_args, reproduce_champion


class WatchEvolvedTests(unittest.TestCase):
    def test_default_and_explicit_integer_scenarios(self) -> None:
        self.assertEqual(parse_args([]).scenario, 0)
        for seed in (0, 3, 100, 150, -1):
            self.assertEqual(parse_args(["--scenario", str(seed)]).scenario, seed)

    def test_non_integer_scenario_is_rejected(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
            parse_args(["--scenario", "1.5"])
        self.assertEqual(error.exception.code, 2)

    def test_reconstruction_uses_exact_configuration_and_returns_champion(self) -> None:
        champion = TinyNeuralNetwork.random(seed=42)
        result = EvolutionResult(champion, (GenerationSummary(29, 125.0, 110.0, (125.0,)),))
        with patch("scripts.watch_evolved.evolve", return_value=result) as train:
            self.assertIs(reproduce_champion(), champion)
        train.assert_called_once_with(
            population_size=40, elite_count=8, scenario_seeds=(0, 1, 2, 3, 4),
            mutation_scale=0.05, generations=30, seed=42,
        )

    def test_unexpected_fitness_is_rejected(self) -> None:
        result = EvolutionResult(
            TinyNeuralNetwork.random(seed=42),
            (GenerationSummary(29, 100.0, 100.0, (100.0,)),),
        )
        with patch("scripts.watch_evolved.evolve", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "125.0, got 100.0"):
                reproduce_champion()

    def test_main_wires_selected_scenario_and_one_champion_to_existing_viewer(self) -> None:
        champion = TinyNeuralNetwork.random(seed=42)
        for seed in (0, 3, 100, 150):
            with (
                self.subTest(seed=seed),
                patch("scripts.watch_evolved.reproduce_champion", return_value=champion) as train,
                patch("scripts.watch_evolved.create_random_episode", wraps=create_random_episode) as create,
                patch("scripts.watch_evolved.NeuralController", wraps=NeuralController) as controller,
                patch("scripts.watch_evolved.run_visual_episode") as view,
                redirect_stdout(StringIO()),
            ):
                main(["--scenario", str(seed)])
                train.assert_called_once_with()
                create.assert_called_once_with(seed)
                controller.assert_called_once_with(champion)
                episode, policy = view.call_args.args
                self.assertIsInstance(policy, NeuralController)
                reference = create_random_episode(seed)
                self.assertEqual(episode.world.food_positions, reference.world.food_positions)
                self.assertEqual(episode.observe(), reference.observe())
                self.assertEqual(episode.completed_ticks, 0)
                self.assertEqual(view.call_args.kwargs, {"scenario_seed": seed})


if __name__ == "__main__":
    unittest.main()
