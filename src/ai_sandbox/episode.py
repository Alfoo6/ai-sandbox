"""Core v0.1 episode mechanics, independent of how actions are chosen."""

from typing import cast

from ai_sandbox.constants import MAX_ENERGY
from ai_sandbox.contracts import Action, Cell, LocalCells, Observation
from ai_sandbox.world import Position, World


INITIAL_ENERGY = MAX_ENERGY
FOOD_ENERGY = 25
TICK_ENERGY_COST = 1
MAX_TICKS = 500


class Episode:
    """Own a world and advance its episode by one action at a time.

    Once handed to an episode, the world should be advanced through step()
    so movement and food changes are accounted for in the episode metrics.
    """

    def __init__(self, world: World) -> None:
        self._world = world
        self._energy = INITIAL_ENERGY
        self._completed_ticks = 0
        self._resources_collected = 0
        self._distance_travelled = 0
        self._terminated = False

    @property
    def world(self) -> World:
        return self._world

    @property
    def energy(self) -> int:
        return self._energy

    @property
    def completed_ticks(self) -> int:
        return self._completed_ticks

    @property
    def resources_collected(self) -> int:
        return self._resources_collected

    @property
    def distance_travelled(self) -> int:
        return self._distance_travelled

    @property
    def terminated(self) -> bool:
        return self._terminated

    def observe(self) -> Observation:
        """Return a radius-2 snapshot, with north at the top and east at right."""
        agent = self._world.agent_position
        food = self._world.food_positions

        def cell_at(dx: int, dy: int) -> Cell:
            x, y = agent.x + dx, agent.y + dy
            if not (0 <= x < self._world.SIZE and 0 <= y < self._world.SIZE):
                return Cell.OUT_OF_BOUNDS
            if dx == 0 and dy == 0:
                return Cell.EMPTY
            if Position(x, y) in food:
                return Cell.FOOD
            return Cell.EMPTY

        # These fixed ranges produce exactly five rows of five cells.
        cells = cast(LocalCells, tuple(
            tuple(cell_at(dx, dy) for dx in range(-2, 3))
            for dy in range(-2, 3)
        ))
        return Observation(cells=cells, energy=self._energy)

    def step(self, action: Action) -> None:
        """Complete one tick; reject further steps after termination."""
        if self._terminated:
            raise RuntimeError("cannot step a terminated episode")

        if self._world.apply_action(action):
            self._distance_travelled += 1
        if self._world.collect_food():
            self._resources_collected += 1
            self._energy = min(MAX_ENERGY, self._energy + FOOD_ENERGY)
        self._energy -= TICK_ENERGY_COST
        self._completed_ticks += 1
        self._terminated = self._energy <= 0 or self._completed_ticks >= MAX_TICKS
