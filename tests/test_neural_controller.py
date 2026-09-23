"""Tests for the neural controller's observation-to-action boundary."""

import unittest

import numpy as np

from ai_sandbox.contracts import Action, Cell, Observation
from ai_sandbox.controllers import NeuralController
from ai_sandbox.controllers.neural_controller import ACTION_ORDER
from ai_sandbox.episode import Episode
from ai_sandbox.neural_network import TinyNeuralNetwork
from ai_sandbox.runner import run_episode
from ai_sandbox.world import Position, World


EMPTY_CELLS = ((Cell.EMPTY,) * 5,) * 5


def network_with_scores(scores: tuple[float, ...], hidden_size: int = 1) -> TinyNeuralNetwork:
    """Use the output bias to set exact scores regardless of observation."""
    return TinyNeuralNetwork(
        26, hidden_size, len(scores),
        w1=np.zeros((26, hidden_size)),
        b1=np.zeros(hidden_size),
        w2=np.zeros((hidden_size, len(scores))),
        b2=scores,
    )


class NeuralControllerTests(unittest.TestCase):
    def test_planned_architecture_is_accepted_and_returns_action(self) -> None:
        controller = NeuralController(TinyNeuralNetwork.random(26, 12, 5, seed=42))
        self.assertIsInstance(controller.choose_action(Observation(EMPTY_CELLS, 100)), Action)

    def test_wrong_input_and_output_sizes_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "input_size must be 26"):
            NeuralController(TinyNeuralNetwork.random(25, 12, 5))
        with self.assertRaisesRegex(ValueError, "output_size must be 5"):
            NeuralController(TinyNeuralNetwork.random(26, 12, 4))

    def test_hidden_size_is_unrestricted(self) -> None:
        for hidden_size in (1, 3, 12):
            with self.subTest(hidden_size=hidden_size):
                controller = NeuralController(network_with_scores((0, 0, 0, 0, 1), hidden_size))
                self.assertIs(controller.choose_action(Observation(EMPTY_CELLS, 50)), Action.STAY)

    def test_every_output_position_maps_to_explicit_action(self) -> None:
        expected_order = (
            Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST, Action.STAY
        )
        self.assertEqual(ACTION_ORDER, expected_order)
        self.assertEqual(set(ACTION_ORDER), set(Action))
        observation = Observation(EMPTY_CELLS, 50)
        for index, expected in enumerate(expected_order):
            with self.subTest(index=index):
                scores = tuple(10.0 if position == index else -1.0 for position in range(5))
                self.assertIs(NeuralController(network_with_scores(scores)).choose_action(observation), expected)

    def test_first_maximum_wins_ties(self) -> None:
        observation = Observation(EMPTY_CELLS, 50)
        controller = NeuralController(network_with_scores((0.0, 2.0, 0.0, 2.0, -1.0)))
        self.assertIs(controller.choose_action(observation), Action.SOUTH)
        all_tied = NeuralController(network_with_scores((0.0,) * 5))
        self.assertIs(all_tied.choose_action(observation), Action.NORTH)

    def test_observation_is_encoded_without_mutation_or_world_context(self) -> None:
        # This network reads only encoded energy at input index 25.
        w1 = np.zeros((26, 1))
        w1[25, 0] = 1.0
        w2 = np.zeros((1, 5))
        w2[0, 2] = 1.0
        network = TinyNeuralNetwork(26, 1, 5, w1, [0.0], w2, [0.0] * 5)
        controller = NeuralController(network)
        observation = Observation(EMPTY_CELLS, 100)
        before = Observation(observation.cells, observation.energy)
        parameters_before = network.parameters()

        self.assertEqual([controller.choose_action(observation) for _ in range(10)], [Action.EAST] * 10)
        self.assertEqual(observation, before)
        for original, current in zip(parameters_before, network.parameters()):
            np.testing.assert_array_equal(original, current)
        self.assertIs(controller.choose_action(Observation(EMPTY_CELLS, 0)), Action.NORTH)

    def test_random_network_runs_with_existing_episode_runner(self) -> None:
        controller = NeuralController(TinyNeuralNetwork.random(26, 12, 5, seed=42))
        episode = Episode(World(Position(12, 12)))
        result = run_episode(episode, controller)
        self.assertTrue(episode.terminated)
        self.assertEqual(result.ticks_survived, 100)
        self.assertEqual(result.final_energy, 0)


if __name__ == "__main__":
    unittest.main()
