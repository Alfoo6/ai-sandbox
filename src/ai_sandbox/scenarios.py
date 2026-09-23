"""Scenario factories shared by visual and headless entrypoints."""

from random import Random

from ai_sandbox.episode import Episode
from ai_sandbox.world import Position, World


def create_demo_episode() -> Episode:
    """Create a fresh episode using the fixed v0.1 demo scenario."""
    return Episode(World(
        agent_position=Position(12, 12),
        food_positions={
            Position(11, 12), Position(13, 12), Position(12, 11),
            Position(12, 13), Position(10, 10), Position(14, 14),
        },
    ))


def create_random_episode(seed: int) -> Episode:
    """Create a fresh episode with six seeded, uniformly sampled food cells."""
    agent_position = Position(12, 12)
    eligible_positions = [
        Position(x, y)
        for y in range(World.SIZE)
        for x in range(World.SIZE)
        if Position(x, y) != agent_position
    ]
    food_positions = set(Random(seed).sample(eligible_positions, 6))
    return Episode(World(agent_position=agent_position, food_positions=food_positions))
