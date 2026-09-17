"""A perception-aware, memoryless baseline controller.

Visible food is ranked by Manhattan distance, then row-major position (north
to south, west to east). The controller moves horizontally before vertically
when either axis would reduce the distance to the selected food.
"""

from random import Random

from ai_sandbox.contracts import Action, Cell, Observation


_CENTER = 2
_EXPLORATION_DIRECTIONS = (
    (Action.NORTH, -1, 0),
    (Action.SOUTH, 1, 0),
    (Action.EAST, 0, 1),
    (Action.WEST, 0, -1),
)


class RuleBasedController:
    """Move toward visible food, otherwise explore with an isolated RNG."""

    def __init__(self, seed: int) -> None:
        self._rng = Random(seed)

    def choose_action(self, observation: Observation) -> Action:
        """Choose an action using only the current observation."""
        food = [
            (row, column)
            for row, cells in enumerate(observation.cells)
            for column, cell in enumerate(cells)
            if cell is Cell.FOOD
        ]
        if food:
            row, column = min(
                food,
                key=lambda position: (
                    abs(position[0] - _CENTER) + abs(position[1] - _CENTER),
                    position[0],
                    position[1],
                ),
            )
            if column < _CENTER:
                return Action.WEST
            if column > _CENTER:
                return Action.EAST
            if row < _CENTER:
                return Action.NORTH
            if row > _CENTER:
                return Action.SOUTH
            return Action.STAY

        available_actions = tuple(
            action
            for action, row_offset, column_offset in _EXPLORATION_DIRECTIONS
            if observation.cells[_CENTER + row_offset][_CENTER + column_offset]
            is not Cell.OUT_OF_BOUNDS
        )
        return self._rng.choice(available_actions)
