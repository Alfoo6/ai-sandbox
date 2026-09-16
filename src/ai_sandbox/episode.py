"""Core v0.1 episode mechanics, independent of how actions are chosen."""

from ai_sandbox.contracts import Action
from ai_sandbox.world import World


INITIAL_ENERGY = 100
MAX_ENERGY = 100
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
