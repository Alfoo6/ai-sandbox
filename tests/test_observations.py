"""Tests of local observation contents and snapshot isolation."""

from dataclasses import FrozenInstanceError, fields
import unittest

from ai_sandbox.contracts import Action, Cell, Observation
from ai_sandbox.episode import Episode
from ai_sandbox.world import Position, World


class ObservationTests(unittest.TestCase):
    def test_empty_view_in_middle(self) -> None:
        observation = Episode(World(Position(12, 12))).observe()
        self.assertIsInstance(observation, Observation)
        self.assertEqual(observation.cells, ((Cell.EMPTY,) * 5,) * 5)
        self.assertEqual(observation.energy, 100)
        self.assertEqual({field.name for field in fields(observation)}, {"cells", "energy"})

    def test_food_at_each_relative_position(self) -> None:
        # Test separately so transposed or mirrored directions cannot look correct.
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                with self.subTest(dx=dx, dy=dy):
                    episode = Episode(World(
                        Position(12, 12), {Position(12 + dx, 12 + dy)}
                    ))
                    cells = episode.observe().cells
                    self.assertIs(cells[dy + 2][dx + 2], Cell.FOOD)
                    self.assertIs(cells[2][2], Cell.EMPTY)
                    self.assertEqual(sum(row.count(Cell.FOOD) for row in cells), 1)

    def test_corners_and_edges(self) -> None:
        # '.' is an empty cell; '#' is outside the world.
        cases = (
            (Position(0, 0), ("#####", "#####", "##...", "##...", "##...")),
            (Position(24, 0), ("#####", "#####", "...##", "...##", "...##")),
            (Position(0, 24), ("##...", "##...", "##...", "#####", "#####")),
            (Position(24, 24), ("...##", "...##", "...##", "#####", "#####")),
            (Position(12, 0), ("#####", "#####", ".....", ".....", ".....")),
            (Position(12, 24), (".....", ".....", ".....", "#####", "#####")),
            (Position(0, 12), ("##...",) * 5),
            (Position(24, 12), ("...##",) * 5),
            (Position(1, 1), ("#####", "#....", "#....", "#....", "#....")),
            (Position(23, 23), ("....#", "....#", "....#", "....#", "#####")),
        )
        symbols = {".": Cell.EMPTY, "#": Cell.OUT_OF_BOUNDS}
        for position, pattern in cases:
            with self.subTest(position=position):
                expected = tuple(tuple(symbols[symbol] for symbol in row) for row in pattern)
                self.assertEqual(Episode(World(position)).observe().cells, expected)

    def test_food_outside_vision_does_not_appear(self) -> None:
        food = {
            Position(x, y)
            for x in range(25)
            for y in range(25)
            if abs(x - 12) > 2 or abs(y - 12) > 2
        }
        observation = Episode(World(Position(12, 12), food)).observe()
        self.assertEqual(observation.cells, ((Cell.EMPTY,) * 5,) * 5)

    def test_energy_tracks_current_episode_state(self) -> None:
        episode = Episode(World(Position(12, 12), {Position(13, 12)}))
        for _ in range(30):
            episode.step(Action.STAY)
        self.assertEqual(episode.observe().energy, episode.energy)
        self.assertEqual(episode.observe().energy, 70)
        episode.step(Action.EAST)
        self.assertEqual(episode.observe().energy, episode.energy)
        self.assertEqual(episode.observe().energy, 94)
        for _ in range(94):
            episode.step(Action.STAY)
        self.assertEqual(episode.observe().energy, 0)

    def test_observation_is_immutable_and_survives_world_changes(self) -> None:
        episode = Episode(World(Position(12, 12), {Position(13, 12), Position(14, 12)}))
        observation = episode.observe()
        with self.assertRaises(FrozenInstanceError):
            observation.energy = 0  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            observation.cells = observation.cells  # type: ignore[misc]
        with self.assertRaises(TypeError):
            observation.cells[0] = observation.cells[1]  # type: ignore[index]
        with self.assertRaises(TypeError):
            observation.cells[2][3] = Cell.EMPTY  # type: ignore[index]

        episode.step(Action.EAST)
        current = episode.observe()
        self.assertEqual(observation.energy, 100)
        self.assertIs(observation.cells[2][3], Cell.FOOD)
        self.assertIs(observation.cells[2][4], Cell.FOOD)
        self.assertEqual(current.energy, 99)
        self.assertIs(current.cells[2][2], Cell.EMPTY)
        self.assertIs(current.cells[2][3], Cell.FOOD)
        self.assertIs(current.cells[2][4], Cell.EMPTY)

    def test_observing_does_not_change_episode_state(self) -> None:
        episode = Episode(World(Position(0, 0), {Position(1, 0)}))
        first = episode.observe()
        self.assertEqual(episode.observe(), first)
        self.assertEqual(episode.world.agent_position, Position(0, 0))
        self.assertEqual(episode.world.food_positions, {Position(1, 0)})
        self.assertEqual(episode.energy, 100)
        self.assertEqual(episode.completed_ticks, 0)
        self.assertEqual(episode.resources_collected, 0)
        self.assertEqual(episode.distance_travelled, 0)
        self.assertFalse(episode.terminated)


if __name__ == "__main__":
    unittest.main()
