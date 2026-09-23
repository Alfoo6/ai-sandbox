"""Shared data contracts for the v0.1 simulation boundary."""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class Action(Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    STAY = "stay"


class Cell(Enum):
    EMPTY = "empty"
    FOOD = "food"
    OUT_OF_BOUNDS = "out_of_bounds"


type CellRow = tuple[Cell, Cell, Cell, Cell, Cell]
type LocalCells = tuple[CellRow, CellRow, CellRow, CellRow, CellRow]


@dataclass(frozen=True)
class Observation:
    """Local cells in row-major order (north at top) and current energy.

    The agent is at cells[2][2]. Both the grid and its rows must be tuples
    so an observation cannot change after it is created.
    """

    cells: LocalCells
    energy: int

    def __post_init__(self) -> None:
        if not isinstance(self.cells, tuple):
            raise TypeError("cells must be a tuple of rows")
        if len(self.cells) != 5:
            raise ValueError("cells must contain exactly five rows")
        for row in self.cells:
            if not isinstance(row, tuple):
                raise TypeError("each row must be a tuple")
            if len(row) != 5:
                raise ValueError("each row must contain exactly five cells")
            if any(not isinstance(cell, Cell) for cell in row):
                raise TypeError("each cell must be a Cell member")


class Controller(Protocol):
    """Choose an action using only an observation."""

    def choose_action(self, observation: Observation) -> Action: ...
