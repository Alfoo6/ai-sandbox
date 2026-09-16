"""Reproducibility and isolation tests for the random controller."""

import random
import unittest

from ai_sandbox.contracts import Action, Cell, LocalCells, Observation
from ai_sandbox.controllers import RandomController


EMPTY_CELLS: LocalCells = ((Cell.EMPTY,) * 5,) * 5


class RandomControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.observation = Observation(cells=EMPTY_CELLS, energy=100)

    def test_returns_valid_actions(self) -> None:
        for seed in (0, 42, -7):
            with self.subTest(seed=seed):
                controller = RandomController(seed)
                for _ in range(100):
                    action = controller.choose_action(self.observation)
                    self.assertIsInstance(action, Action)
                    self.assertIn(action, tuple(Action))

    def test_identical_seeds_produce_identical_sequences(self) -> None:
        for seed in (0, 42, -7):
            with self.subTest(seed=seed):
                first = RandomController(seed)
                second = RandomController(seed)
                expected = [first.choose_action(self.observation) for _ in range(100)]
                actual = [second.choose_action(self.observation) for _ in range(100)]
                self.assertEqual(actual, expected)

    def test_instances_keep_independent_rng_state(self) -> None:
        reference = RandomController(42)
        expected = [reference.choose_action(self.observation) for _ in range(100)]
        first = RandomController(42)
        second = RandomController(42)
        # Advance one instance before and between draws from the other.
        for _ in range(17):
            first.choose_action(self.observation)
        actual = []
        for _ in range(100):
            first.choose_action(self.observation)
            first.choose_action(self.observation)
            actual.append(second.choose_action(self.observation))
        self.assertEqual(actual, expected)

    def test_supplied_observation_is_unchanged(self) -> None:
        before = Observation(cells=self.observation.cells, energy=self.observation.energy)
        controller = RandomController(42)
        for _ in range(100):
            controller.choose_action(self.observation)
        self.assertEqual(self.observation, before)

    def test_observation_contents_do_not_affect_choices(self) -> None:
        cells: LocalCells = (
            (Cell.OUT_OF_BOUNDS,) * 5,
            (Cell.FOOD,) * 5,
            (Cell.EMPTY,) * 5,
            (Cell.EMPTY,) * 5,
            (Cell.EMPTY,) * 5,
        )
        other_observation = Observation(cells=cells, energy=1)
        first = RandomController(42)
        second = RandomController(42)
        self.assertEqual(
            [first.choose_action(self.observation) for _ in range(100)],
            [second.choose_action(other_observation) for _ in range(100)],
        )

    def test_module_random_state_is_independent(self) -> None:
        original_state = random.getstate()
        try:
            controller = RandomController(42)
            expected = [controller.choose_action(self.observation) for _ in range(100)]
            self.assertEqual(random.getstate(), original_state)

            random.seed(123)
            independent = RandomController(42)
            actual = []
            for _ in range(100):
                random.random()
                actual.append(independent.choose_action(self.observation))
            self.assertEqual(actual, expected)
        finally:
            random.setstate(original_state)


if __name__ == "__main__":
    unittest.main()
