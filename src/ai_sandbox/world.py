"""Minimal world state and movement for v0.1."""

from dataclasses import dataclass
from typing import ClassVar

from ai_sandbox.contracts import Action


@dataclass(frozen=True)
class Position:
    """Grid coordinates: x increases eastward and y increases southward."""

    x: int
    y: int


class World:
    """A fixed 25x25 grid whose boundary cells are valid positions."""

    SIZE: ClassVar[int] = 25

    def __init__(
        self,
        agent_position: Position,
        food_positions: set[Position] | frozenset[Position] = frozenset(),
    ) -> None:
        food = frozenset(food_positions)
        if not self._is_inside(agent_position):
            raise ValueError("agent position must be inside the grid")
        if any(not self._is_inside(position) for position in food):
            raise ValueError("all food positions must be inside the grid")
        if agent_position in food:
            raise ValueError("food must not overlap the agent at initialization")
        self._agent_position = agent_position
        self._food_positions = food

    @property
    def agent_position(self) -> Position:
        return self._agent_position

    @property
    def food_positions(self) -> frozenset[Position]:
        return self._food_positions

    def apply_action(self, action: Action) -> bool:
        """Apply one move, returning whether the agent actually moved."""
        match action:
            case Action.NORTH:
                dx, dy = 0, -1
            case Action.SOUTH:
                dx, dy = 0, 1
            case Action.EAST:
                dx, dy = 1, 0
            case Action.WEST:
                dx, dy = -1, 0
            case Action.STAY:
                return False
            case _:
                raise TypeError("action must be an Action member")

        destination = Position(
            self._agent_position.x + dx, self._agent_position.y + dy
        )
        if not self._is_inside(destination):
            return False
        self._agent_position = destination
        return True

    def _is_inside(self, position: Position) -> bool:
        return 0 <= position.x < self.SIZE and 0 <= position.y < self.SIZE
