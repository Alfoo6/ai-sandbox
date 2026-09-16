"""Tests of complete ticks and episode termination through the public API."""

import unittest

from ai_sandbox.contracts import Action
from ai_sandbox.episode import Episode
from ai_sandbox.world import Position, World


class EpisodeTests(unittest.TestCase):
    def test_initial_state(self) -> None:
        world = World(Position(12, 12))
        episode = Episode(world)
        self.assertIs(episode.world, world)
        self.assertEqual(episode.energy, 100)
        self.assertEqual(episode.completed_ticks, 0)
        self.assertEqual(episode.resources_collected, 0)
        self.assertEqual(episode.distance_travelled, 0)
        self.assertFalse(episode.terminated)

    def test_movement_costs_one_energy_and_tick(self) -> None:
        episode = Episode(World(Position(12, 12)))
        episode.step(Action.NORTH)
        self.assertEqual(episode.energy, 99)
        self.assertEqual(episode.completed_ticks, 1)
        self.assertEqual(episode.world.agent_position, Position(12, 11))

    def test_food_restores_energy_and_is_removed_once(self) -> None:
        episode = Episode(World(Position(0, 0), {Position(1, 0), Position(2, 0)}))
        for _ in range(30):
            episode.step(Action.STAY)
        episode.step(Action.EAST)
        self.assertEqual(episode.energy, 94)
        self.assertEqual(episode.resources_collected, 1)
        self.assertEqual(episode.world.food_positions, {Position(2, 0)})
        episode.step(Action.STAY)
        episode.step(Action.WEST)
        episode.step(Action.EAST)
        self.assertEqual(episode.energy, 91)
        self.assertEqual(episode.resources_collected, 1)
        self.assertEqual(episode.world.food_positions, {Position(2, 0)})

    def test_food_cap_is_applied_before_tick_cost(self) -> None:
        episode = Episode(World(Position(0, 0), {Position(1, 0)}))
        episode.step(Action.EAST)
        self.assertEqual(episode.energy, 99)
        self.assertEqual(episode.resources_collected, 1)

    def test_blocked_movement_and_stay_cost_energy_and_ticks(self) -> None:
        for action in (Action.NORTH, Action.WEST, Action.STAY):
            with self.subTest(action=action):
                episode = Episode(World(Position(0, 0)))
                episode.step(action)
                self.assertEqual(episode.energy, 99)
                self.assertEqual(episode.completed_ticks, 1)
                self.assertEqual(episode.distance_travelled, 0)
                self.assertEqual(episode.world.agent_position, Position(0, 0))

    def test_distance_counts_successful_moves_including_return_moves(self) -> None:
        episode = Episode(World(Position(0, 0)))
        for action in (Action.WEST, Action.STAY, Action.EAST, Action.SOUTH,
                       Action.NORTH, Action.WEST):
            episode.step(action)
        self.assertEqual(episode.distance_travelled, 4)
        self.assertEqual(episode.completed_ticks, 6)
        self.assertEqual(episode.world.agent_position, Position(0, 0))

    def test_zero_energy_terminates_and_fatal_tick_counts(self) -> None:
        episode = Episode(World(Position(0, 0)))
        for _ in range(99):
            episode.step(Action.STAY)
        self.assertEqual(episode.energy, 1)
        self.assertFalse(episode.terminated)
        episode.step(Action.EAST)
        self.assertEqual(episode.energy, 0)
        self.assertEqual(episode.completed_ticks, 100)
        self.assertEqual(episode.distance_travelled, 1)
        self.assertEqual(episode.world.agent_position, Position(1, 0))
        self.assertTrue(episode.terminated)

    def test_food_is_collected_before_energy_termination_check(self) -> None:
        episode = Episode(World(Position(0, 0), {Position(1, 0)}))
        for _ in range(99):
            episode.step(Action.STAY)
        episode.step(Action.EAST)
        self.assertEqual(episode.energy, 25)
        self.assertEqual(episode.completed_ticks, 100)
        self.assertFalse(episode.terminated)

    def episode_at_tick_limit(self) -> Episode:
        episode = Episode(World(Position(0, 0), {Position(x, 0) for x in range(1, 21)}))
        # Collect food every 25 ticks to stay alive without altering state directly.
        for tick in range(1, 500):
            episode.step(Action.EAST if tick % 25 == 0 else Action.STAY)
        self.assertEqual(episode.completed_ticks, 499)
        self.assertFalse(episode.terminated)
        episode.step(Action.EAST)
        return episode

    def test_terminates_at_500_completed_ticks_with_energy_remaining(self) -> None:
        episode = self.episode_at_tick_limit()
        self.assertEqual(episode.completed_ticks, 500)
        self.assertEqual(episode.energy, 99)
        self.assertEqual(episode.resources_collected, 20)
        self.assertEqual(episode.distance_travelled, 20)
        self.assertEqual(episode.world.food_positions, frozenset())
        self.assertTrue(episode.terminated)

    def test_steps_after_either_termination_leave_all_state_unchanged(self) -> None:
        exhausted = Episode(World(Position(0, 0), {Position(1, 0)}))
        for _ in range(100):
            exhausted.step(Action.STAY)
        for episode in (exhausted, self.episode_at_tick_limit()):
            def snapshot() -> tuple[object, ...]:
                return (
                    episode.energy, episode.completed_ticks,
                    episode.resources_collected, episode.distance_travelled,
                    episode.terminated, episode.world.agent_position,
                    episode.world.food_positions,
                )

            before = snapshot()
            for action in (Action.EAST, Action.STAY):
                with self.subTest(ticks=episode.completed_ticks, action=action):
                    with self.assertRaises(RuntimeError):
                        episode.step(action)
                    self.assertEqual(snapshot(), before)


if __name__ == "__main__":
    unittest.main()
