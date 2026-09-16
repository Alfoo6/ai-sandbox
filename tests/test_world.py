"""Focused tests for world initialization and movement."""

from dataclasses import FrozenInstanceError
import unittest

from ai_sandbox.contracts import Action
from ai_sandbox.world import Position, World


class PositionTests(unittest.TestCase):
    def test_position_is_an_immutable_value(self) -> None:
        position = Position(3, 7)
        self.assertEqual(position.x, 3)
        self.assertEqual(position.y, 7)
        self.assertEqual(position, Position(3, 7))
        self.assertNotEqual(position, Position(7, 3))
        self.assertEqual({position, Position(3, 7)}, {position})
        with self.assertRaises(FrozenInstanceError):
            position.x = 4  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            position.y = 4  # type: ignore[misc]


class WorldTests(unittest.TestCase):
    def test_initial_state(self) -> None:
        food = {Position(0, 0), Position(24, 24)}
        world = World(Position(12, 12), food)
        self.assertEqual(World.SIZE, 25)
        self.assertEqual(world.agent_position, Position(12, 12))
        self.assertEqual(world.food_positions, food)
        self.assertIsInstance(world.food_positions, frozenset)
        food.add(Position(-1, 0))
        self.assertNotIn(Position(-1, 0), world.food_positions)

    def test_food_defaults_to_empty(self) -> None:
        self.assertEqual(World(Position(0, 0)).food_positions, frozenset())

    def test_corners_are_valid_positions(self) -> None:
        for corner in (Position(0, 0), Position(0, 24), Position(24, 0), Position(24, 24)):
            with self.subTest(corner=corner):
                self.assertEqual(World(corner).agent_position, corner)
                self.assertIn(corner, World(Position(12, 12), {corner}).food_positions)

    def test_rejects_out_of_bounds_initial_positions(self) -> None:
        for position in (Position(-1, 12), Position(25, 12), Position(12, -1), Position(12, 25)):
            with self.subTest(agent=position), self.assertRaises(ValueError):
                World(position)
            with self.subTest(food=position), self.assertRaises(ValueError):
                World(Position(12, 12), {Position(1, 1), position})

    def test_rejects_initial_food_overlap(self) -> None:
        with self.assertRaises(ValueError):
            World(Position(12, 12), {Position(12, 12)})

    def test_moves_one_cell_in_each_direction(self) -> None:
        destinations = {
            Action.NORTH: Position(12, 11),
            Action.SOUTH: Position(12, 13),
            Action.EAST: Position(13, 12),
            Action.WEST: Position(11, 12),
        }
        for action, destination in destinations.items():
            with self.subTest(action=action):
                world = World(Position(12, 12))
                self.assertIs(world.apply_action(action), True)
                self.assertEqual(world.agent_position, destination)

    def test_stay_reports_no_movement(self) -> None:
        for position in (Position(12, 12), Position(0, 0), Position(24, 24)):
            with self.subTest(position=position):
                world = World(position)
                self.assertIs(world.apply_action(Action.STAY), False)
                self.assertEqual(world.agent_position, position)

    def test_boundaries_block_without_wrapping(self) -> None:
        for coordinate in range(25):
            boundaries = (
                (Position(coordinate, 0), Action.NORTH),
                (Position(coordinate, 24), Action.SOUTH),
                (Position(0, coordinate), Action.WEST),
                (Position(24, coordinate), Action.EAST),
            )
            for position, action in boundaries:
                with self.subTest(position=position, action=action):
                    world = World(position)
                    self.assertIs(world.apply_action(action), False)
                    self.assertEqual(world.agent_position, position)

    def test_can_reach_boundary_and_move_back(self) -> None:
        world = World(Position(1, 0))
        self.assertIs(world.apply_action(Action.WEST), True)
        self.assertEqual(world.agent_position, Position(0, 0))
        self.assertIs(world.apply_action(Action.WEST), False)
        self.assertIs(world.apply_action(Action.EAST), True)
        self.assertEqual(world.agent_position, Position(1, 0))

    def test_moving_onto_food_leaves_food_unchanged(self) -> None:
        food = {Position(13, 12), Position(0, 0)}
        world = World(Position(12, 12), food)
        self.assertIs(world.apply_action(Action.EAST), True)
        self.assertEqual(world.agent_position, Position(13, 12))
        self.assertEqual(world.food_positions, food)

    def test_rejects_invalid_action_without_changing_position(self) -> None:
        world = World(Position(12, 12))
        with self.assertRaises(TypeError):
            world.apply_action("north")  # type: ignore[arg-type]
        self.assertEqual(world.agent_position, Position(12, 12))


if __name__ == "__main__":
    unittest.main()
