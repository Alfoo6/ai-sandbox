"""Tests for seeded, additive neural-network mutation."""

import random
import unittest

import numpy as np

from ai_sandbox.neural_mutation import mutate_network
from ai_sandbox.neural_network import TinyNeuralNetwork


class NeuralMutationTests(unittest.TestCase):
    def test_child_is_distinct_and_preserves_architecture(self) -> None:
        parent = TinyNeuralNetwork.random(26, 12, 5, seed=42)
        child = mutate_network(parent, 0.05, seed=17)

        self.assertIsNot(child, parent)
        self.assertIsInstance(child, TinyNeuralNetwork)
        self.assertEqual(
            (child.input_size, child.hidden_size, child.output_size),
            (parent.input_size, parent.hidden_size, parent.output_size),
        )
        self.assertEqual(
            tuple(parameter.shape for parameter in child.parameters()),
            tuple(parameter.shape for parameter in parent.parameters()),
        )

    def test_parent_unchanged_and_all_four_groups_change(self) -> None:
        parent = TinyNeuralNetwork.random(26, 12, 5, seed=42)
        before = parent.parameters()
        child = mutate_network(parent, 0.05, seed=17)

        for name, original, current, changed in zip(
            ("W1", "b1", "W2", "b2"), before, parent.parameters(), child.parameters()
        ):
            with self.subTest(parameter=name):
                np.testing.assert_array_equal(current, original)
                self.assertTrue(np.any(changed != original))

    def test_same_seed_repeats_and_different_seed_changes_child(self) -> None:
        parent = TinyNeuralNetwork.random(26, 12, 5, seed=42)
        first = mutate_network(parent, 0.05, seed=17)
        same = mutate_network(parent, 0.05, seed=17)
        different = mutate_network(parent, 0.05, seed=18)

        for left, right in zip(first.parameters(), same.parameters()):
            np.testing.assert_array_equal(left, right)
        self.assertTrue(any(
            not np.array_equal(left, right)
            for left, right in zip(first.parameters(), different.parameters())
        ))

    def test_zero_scale_makes_independent_identical_child(self) -> None:
        parent = TinyNeuralNetwork.random(2, 3, 1, seed=42)
        child = mutate_network(parent, 0.0, seed=17)

        self.assertIsNot(child, parent)
        for original, copied in zip(parent.parameters(), child.parameters()):
            np.testing.assert_array_equal(copied, original)

    def test_negative_scale_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "mutation_scale"):
            mutate_network(TinyNeuralNetwork.random(), -0.01, seed=17)

    def test_global_python_and_numpy_rng_states_are_unchanged(self) -> None:
        parent = TinyNeuralNetwork.random(seed=42)
        python_before = random.getstate()
        numpy_before = np.random.get_state()

        mutate_network(parent, 0.05, seed=17)

        self.assertEqual(random.getstate(), python_before)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])

    def test_noise_is_additive_with_approximately_requested_scale(self) -> None:
        # A fixed seed and 10,000+ scalars keep this sanity check stable.
        parent = TinyNeuralNetwork.random(100, 100, 20, seed=42)
        child = mutate_network(parent, 0.2, seed=17)
        deltas = np.concatenate([
            (changed - original).ravel()
            for original, changed in zip(parent.parameters(), child.parameters())
        ])

        self.assertGreater(len(deltas), 10_000)
        self.assertLess(abs(float(np.mean(deltas))), 0.01)
        self.assertAlmostEqual(float(np.std(deltas)), 0.2, delta=0.01)


if __name__ == "__main__":
    unittest.main()
