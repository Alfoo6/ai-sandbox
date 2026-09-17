"""Focused behavior tests for the perception-aware baseline controller."""

import random
import unittest

from ai_sandbox.contracts import Action, Cell, LocalCells, Observation
from ai_sandbox.controllers import RuleBasedController


def observation_with(
    cells: dict[tuple[int, int], Cell] | None = None,
    *,
    energy: int = 100,
) -> Observation:
    contents = cells or {}
    grid = tuple(
        tuple(contents.get((row, column), Cell.EMPTY) for column in range(5))
        for row in range(5)
    )
    return Observation(cells=grid, energy=energy)  # type: ignore[arg-type]


class RuleBasedControllerTests(unittest.TestCase):
    def test_moves_toward_visible_food_in_each_direction(self) -> None:
        cases = (
            ((0, 2), Action.NORTH),
            ((4, 2), Action.SOUTH),
            ((2, 4), Action.EAST),
            ((2, 0), Action.WEST),
        )
        controller = RuleBasedController(42)
        for position, expected in cases:
            with self.subTest(position=position):
                observation = observation_with({position: Cell.FOOD})
                self.assertIs(controller.choose_action(observation), expected)

    def test_chooses_nearest_of_multiple_food_cells(self) -> None:
        observation = observation_with({
            (0, 2): Cell.FOOD,
            (2, 3): Cell.FOOD,
            (4, 4): Cell.FOOD,
        })
        self.assertIs(
            RuleBasedController(42).choose_action(observation),
            Action.EAST,
        )

    def test_equal_distance_food_targets_use_row_major_order(self) -> None:
        observation = observation_with({
            (2, 1): Cell.FOOD,
            (1, 2): Cell.FOOD,
            (2, 3): Cell.FOOD,
            (3, 2): Cell.FOOD,
        })
        self.assertIs(
            RuleBasedController(42).choose_action(observation),
            Action.NORTH,
        )

    def test_diagonal_food_uses_horizontal_axis_first(self) -> None:
        observation = observation_with({(0, 4): Cell.FOOD})
        self.assertIs(
            RuleBasedController(42).choose_action(observation),
            Action.EAST,
        )

    def test_explores_cardinal_directions_when_no_food_is_visible(self) -> None:
        observation = observation_with()
        controller = RuleBasedController(42)
        actions = [controller.choose_action(observation) for _ in range(100)]
        self.assertTrue(set(actions) <= {
            Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST,
        })
        self.assertGreater(len(set(actions)), 1)

    def test_equal_seeds_produce_equal_exploration_sequences(self) -> None:
        observation = observation_with()
        first = RuleBasedController(17)
        second = RuleBasedController(17)
        self.assertEqual(
            [first.choose_action(observation) for _ in range(100)],
            [second.choose_action(observation) for _ in range(100)],
        )

    def test_instances_keep_independent_rng_state(self) -> None:
        observation = observation_with()
        reference = RuleBasedController(17)
        expected = [reference.choose_action(observation) for _ in range(100)]
        first = RuleBasedController(17)
        second = RuleBasedController(17)
        for _ in range(13):
            first.choose_action(observation)
        actual = []
        for _ in range(100):
            first.choose_action(observation)
            actual.append(second.choose_action(observation))
        self.assertEqual(actual, expected)

    def test_module_random_state_is_independent(self) -> None:
        observation = observation_with()
        original_state = random.getstate()
        try:
            reference = RuleBasedController(42)
            expected = [reference.choose_action(observation) for _ in range(100)]
            self.assertEqual(random.getstate(), original_state)

            random.seed(123)
            controller = RuleBasedController(42)
            actual = []
            for _ in range(100):
                random.random()
                actual.append(controller.choose_action(observation))
            self.assertEqual(actual, expected)
        finally:
            random.setstate(original_state)

    def test_does_not_explore_toward_adjacent_out_of_bounds_cells(self) -> None:
        blocked = {
            (1, 2): Action.NORTH,
            (3, 2): Action.SOUTH,
            (2, 3): Action.EAST,
            (2, 1): Action.WEST,
        }
        for position, forbidden in blocked.items():
            with self.subTest(position=position):
                observation = observation_with({position: Cell.OUT_OF_BOUNDS})
                controller = RuleBasedController(42)
                for _ in range(100):
                    self.assertIsNot(controller.choose_action(observation), forbidden)

    def test_never_stays_during_exploration(self) -> None:
        observation = observation_with({
            (1, 2): Cell.OUT_OF_BOUNDS,
            (2, 1): Cell.OUT_OF_BOUNDS,
        })
        controller = RuleBasedController(42)
        for _ in range(100):
            self.assertIsNot(controller.choose_action(observation), Action.STAY)

    def test_energy_does_not_affect_behavior(self) -> None:
        high_energy = observation_with(energy=100)
        low_energy = observation_with(energy=1)
        first = RuleBasedController(42)
        second = RuleBasedController(42)
        self.assertEqual(
            [first.choose_action(high_energy) for _ in range(100)],
            [second.choose_action(low_energy) for _ in range(100)],
        )

    def test_supplied_observation_is_unchanged(self) -> None:
        cells: LocalCells = (
            (Cell.EMPTY,) * 5,
            (Cell.EMPTY, Cell.FOOD, Cell.EMPTY, Cell.FOOD, Cell.EMPTY),
            (Cell.EMPTY,) * 5,
            (Cell.EMPTY,) * 5,
            (Cell.OUT_OF_BOUNDS,) * 5,
        )
        observation = Observation(cells=cells, energy=73)
        before = Observation(cells=observation.cells, energy=observation.energy)
        controller = RuleBasedController(42)
        for _ in range(100):
            controller.choose_action(observation)
        self.assertEqual(observation, before)


if __name__ == "__main__":
    unittest.main()
