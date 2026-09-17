"""Fixed scenarios shared by visual and headless entrypoints."""

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
