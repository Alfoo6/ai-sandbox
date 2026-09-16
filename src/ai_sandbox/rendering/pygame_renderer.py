"""Simple grid rendering; no simulation updates or random choices."""

import pygame

from ai_sandbox.episode import Episode
from ai_sandbox.world import Position, World


CELL_SIZE = 24
GRID_PIXELS = World.SIZE * CELL_SIZE
WINDOW_SIZE = (GRID_PIXELS, GRID_PIXELS + 130)


class PygameRenderer:
    def __init__(self, surface: pygame.Surface) -> None:
        self._surface = surface
        self._font = pygame.font.Font(None, 24)

    def draw(self, episode: Episode, controller_name: str, paused: bool) -> None:
        """Draw current state without changing it, including a clipped vision area."""
        surface = self._surface
        surface.fill((22, 26, 32))
        agent = episode.world.agent_position
        food = episode.world.food_positions
        for y in range(World.SIZE):
            for x in range(World.SIZE):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                in_vision = abs(x - agent.x) <= 2 and abs(y - agent.y) <= 2
                color = (43, 57, 70) if in_vision else (29, 34, 41)
                pygame.draw.rect(surface, color, rect)
                pygame.draw.rect(surface, (51, 58, 67), rect, 1)
                if Position(x, y) in food:
                    pygame.draw.circle(surface, (110, 200, 120), rect.center, 6)

        agent_rect = pygame.Rect(
            agent.x * CELL_SIZE + 5, agent.y * CELL_SIZE + 5,
            CELL_SIZE - 10, CELL_SIZE - 10,
        )
        pygame.draw.rect(surface, (245, 190, 80), agent_rect)

        status = "Finished" if episode.terminated else "Paused" if paused else "Running"
        lines = (
            f"{controller_name}  |  {status}",
            f"Ticks: {episode.completed_ticks}   Energy: {episode.energy}",
            f"Resources: {episode.resources_collected}   Distance: {episode.distance_travelled}",
            "SPACE: pause/resume   ESC: exit   |   5 ticks/sec",
        )
        for index, line in enumerate(lines):
            text = self._font.render(line, True, (224, 228, 234))
            surface.blit(text, (12, GRID_PIXELS + 12 + index * 27))
