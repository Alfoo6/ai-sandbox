"""Tests for fixed and seeded scenario factories."""

import random
import unittest

from ai_sandbox.contracts import Action
from ai_sandbox.controllers import RandomController
from ai_sandbox.scenarios import create_demo_episode, create_random_episode
from ai_sandbox.world import Position, World


class ScenarioTests(unittest.TestCase):
    def test_fixed_demo_scenario_is_unchanged(self) -> None:
        episode = create_demo_episode()
        self.assertEqual(episode.world.agent_position, Position(12, 12))
        self.assertEqual(episode.world.food_positions, {
            Position(11, 12), Position(13, 12), Position(12, 11),
            Position(12, 13), Position(10, 10), Position(14, 14),
        })

    def test_same_seed_produces_same_scenario(self) -> None:
        first = create_random_episode(42)
        second = create_random_episode(42)
        self.assertEqual(first.world.agent_position, second.world.agent_position)
        self.assertEqual(first.world.food_positions, second.world.food_positions)

    def test_selected_different_seeds_produce_different_scenarios(self) -> None:
        self.assertNotEqual(
            create_random_episode(0).world.food_positions,
            create_random_episode(1).world.food_positions,
        )

    def test_food_positions_are_unique_valid_and_do_not_overlap_agent(self) -> None:
        for seed in (0, 1, 42, -7):
            with self.subTest(seed=seed):
                world = create_random_episode(seed).world
                food = world.food_positions
                self.assertEqual(world.agent_position, Position(12, 12))
                self.assertEqual(len(food), 6)
                self.assertEqual(len(set(food)), 6)
                self.assertNotIn(world.agent_position, food)
                self.assertTrue(all(
                    0 <= position.x < World.SIZE and 0 <= position.y < World.SIZE
                    for position in food
                ))

    def test_generation_does_not_change_module_random_state(self) -> None:
        before = random.getstate()
        create_random_episode(42)
        self.assertEqual(random.getstate(), before)

    def test_generation_is_independent_of_controller_randomness(self) -> None:
        expected = create_random_episode(42).world.food_positions
        controller = RandomController(seed=42)
        observation = create_demo_episode().observe()
        for _ in range(20):
            controller.choose_action(observation)
        self.assertEqual(create_random_episode(42).world.food_positions, expected)

    def test_calls_return_independent_worlds_and_episodes(self) -> None:
        first = create_random_episode(42)
        second = create_random_episode(42)
        self.assertIsNot(first, second)
        self.assertIsNot(first.world, second.world)
        first.step(Action.NORTH)
        self.assertEqual(second.world.agent_position, Position(12, 12))
        self.assertEqual(second.completed_ticks, 0)
        self.assertEqual(second.energy, 100)


if __name__ == "__main__":
    unittest.main()
