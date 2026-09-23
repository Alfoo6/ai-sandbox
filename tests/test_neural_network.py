"""Focused tests for the numeric network math and ownership."""

import math
import unittest

import numpy as np

from ai_sandbox.neural_network import TinyNeuralNetwork


class TinyNeuralNetworkTests(unittest.TestCase):
    def test_planned_architecture_and_output_size(self) -> None:
        network = TinyNeuralNetwork.random()
        self.assertEqual((network.input_size, network.hidden_size, network.output_size), (26, 12, 5))
        self.assertEqual(tuple(p.shape for p in network.parameters()), (
            (26, 12), (12,), (12, 5), (5,)
        ))
        scores = network.forward((0.0,) * 26)
        self.assertIsInstance(scores, tuple)
        self.assertEqual(len(scores), 5)
        self.assertTrue(all(isinstance(score, float) for score in scores))

    def test_random_seed_reproducibility_and_difference(self) -> None:
        first = TinyNeuralNetwork.random(seed=42)
        same = TinyNeuralNetwork.random(seed=42)
        different = TinyNeuralNetwork.random(seed=43)
        for left, right in zip(first.parameters(), same.parameters()):
            np.testing.assert_array_equal(left, right)
        self.assertTrue(any(
            not np.array_equal(left, right)
            for left, right in zip(first.parameters(), different.parameters())
        ))

    def test_forward_is_repeatable_and_does_not_change_parameters(self) -> None:
        network = TinyNeuralNetwork.random(seed=7)
        before = network.parameters()
        inputs = tuple(index / 26 for index in range(26))
        self.assertEqual(network.forward(inputs), network.forward(inputs))
        for left, right in zip(before, network.parameters()):
            np.testing.assert_array_equal(left, right)

    def test_random_construction_does_not_change_numpy_global_rng(self) -> None:
        before = np.random.get_state()
        TinyNeuralNetwork.random(seed=42)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_known_parameters_use_tanh_and_linear_output(self) -> None:
        network = TinyNeuralNetwork(
            2, 2, 1,
            w1=[[1.0, 0.0], [0.0, 1.0]],
            b1=[0.0, 0.0],
            w2=[[2.0], [3.0]],
            b2=[0.5],
        )
        # hidden = (tanh(1), tanh(-1)); score = 0.5 - tanh(1).
        self.assertAlmostEqual(network.forward([1.0, -1.0])[0], 0.5 - math.tanh(1.0))

    def test_invalid_parameter_shapes_are_rejected(self) -> None:
        valid = dict(w1=np.zeros((2, 2)), b1=np.zeros(2),
                     w2=np.zeros((2, 1)), b2=np.zeros(1))
        bad_shapes = {
            "w1": np.zeros((2, 3)),
            "b1": np.zeros(1),
            "w2": np.zeros((1, 1)),
            "b2": np.zeros(2),
        }
        for name, value in bad_shapes.items():
            with self.subTest(parameter=name), self.assertRaisesRegex(ValueError, name):
                TinyNeuralNetwork(2, 2, 1, **(valid | {name: value}))

    def test_invalid_forward_input_shape_is_rejected(self) -> None:
        network = TinyNeuralNetwork.random(2, 2, 1)
        for inputs in ([1.0], [1.0, 2.0, 3.0], [[1.0, 2.0]]):
            with self.subTest(inputs=inputs), self.assertRaisesRegex(ValueError, "inputs must have shape"):
                network.forward(inputs)

    def test_caller_arrays_and_parameter_copies_cannot_change_network(self) -> None:
        w1 = np.eye(2)
        b1 = np.zeros(2)
        w2 = np.ones((2, 1))
        b2 = np.zeros(1)
        network = TinyNeuralNetwork(2, 2, 1, w1, b1, w2, b2)
        expected = network.forward([1.0, 2.0])
        for array in (w1, b1, w2, b2):
            array.fill(99.0)
        for array in network.parameters():
            array.fill(-99.0)
        self.assertEqual(network.forward([1.0, 2.0]), expected)


if __name__ == "__main__":
    unittest.main()
