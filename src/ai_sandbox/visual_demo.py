"""Watch one manually defined episode with the random controller."""

import pygame

from ai_sandbox.controllers import RandomController
from ai_sandbox.episode import Episode
from ai_sandbox.rendering.pygame_renderer import PygameRenderer, WINDOW_SIZE
from ai_sandbox.runner import episode_ticks
from ai_sandbox.world import Position, World


TICK_INTERVAL_MS = 200


def create_demo_episode() -> Episode:
    return Episode(World(
        agent_position=Position(12, 12),
        food_positions={
            Position(11, 12), Position(13, 12), Position(12, 11),
            Position(12, 13), Position(10, 10), Position(14, 14),
        },
    ))


def run_visual_episode(episode: Episode, controller: RandomController) -> None:
    """Pace the shared simulation loop and keep the window open until dismissed."""
    pygame.display.init()
    pygame.font.init()
    try:
        surface = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("AI Sandbox")
        renderer = PygameRenderer(surface)
        clock = pygame.time.Clock()
        ticks = episode_ticks(episode, controller)
        paused = False
        next_tick = pygame.time.get_ticks() + TICK_INTERVAL_MS

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                        next_tick = pygame.time.get_ticks() + TICK_INTERVAL_MS

            now = pygame.time.get_ticks()
            if not paused and not episode.terminated and now >= next_tick:
                next(ticks)
                # No catch-up bursts: wall time never changes the tick's mechanics.
                next_tick = now + TICK_INTERVAL_MS

            renderer.draw(episode, type(controller).__name__, paused)
            pygame.display.flip()
            clock.tick(60)
    finally:
        pygame.quit()


def main() -> None:
    run_visual_episode(create_demo_episode(), RandomController(seed=42))


if __name__ == "__main__":
    main()
