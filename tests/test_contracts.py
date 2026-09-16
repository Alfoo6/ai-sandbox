"""Focused tests for the simulation's shared data contracts."""

from dataclasses import FrozenInstanceError
import unittest

from ai_sandbox.contracts import Action, Cell, CellRow, LocalCells, Observation


EMPTY_ROW: CellRow = (Cell.EMPTY,) * 5
EMPTY_CELLS: LocalCells = (EMPTY_ROW,) * 5


class ContractTests(unittest.TestCase):
    def test_actions(self) -> None:
        self.assertEqual(
            {action.name: action.value for action in Action},
            {
                "NORTH": "north",
                "SOUTH": "south",
                "EAST": "east",
                "WEST": "west",
                "STAY": "stay",
            },
        )

    def test_cell_types(self) -> None:
        self.assertEqual(
            {cell.name: cell.value for cell in Cell},
            {"EMPTY": "empty", "FOOD": "food", "OUT_OF_BOUNDS": "out_of_bounds"},
        )

    def test_observation_preserves_local_cells_and_energy(self) -> None:
        north_row: CellRow = (Cell.OUT_OF_BOUNDS,) * 5
        center_row: CellRow = (
            Cell.EMPTY, Cell.EMPTY, Cell.EMPTY, Cell.FOOD, Cell.EMPTY
        )
        cells: LocalCells = (
            north_row, EMPTY_ROW, center_row, EMPTY_ROW, EMPTY_ROW
        )
        observation = Observation(cells=cells, energy=73)

        self.assertEqual(observation.cells, cells)
        self.assertIs(observation.cells[0][2], Cell.OUT_OF_BOUNDS)
        self.assertIs(observation.cells[2][2], Cell.EMPTY)
        self.assertIs(observation.cells[2][3], Cell.FOOD)
        self.assertEqual(observation.energy, 73)

    def test_observation_is_immutable(self) -> None:
        observation = Observation(cells=EMPTY_CELLS, energy=100)

        with self.assertRaises(FrozenInstanceError):
            observation.energy = 50  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            observation.cells = EMPTY_CELLS  # type: ignore[misc]
        with self.assertRaises(TypeError):
            observation.cells[0] = EMPTY_ROW  # type: ignore[index]
        with self.assertRaises(TypeError):
            observation.cells[0][0] = Cell.FOOD  # type: ignore[index]

    def test_rejects_wrong_grid_dimensions(self) -> None:
        for size in (0, 4, 6):
            with self.subTest(rows=size), self.assertRaises(ValueError):
                Observation(cells=(EMPTY_ROW,) * size, energy=100)  # type: ignore[arg-type]
            with self.subTest(columns=size), self.assertRaises(ValueError):
                Observation(cells=((Cell.EMPTY,) * size,) * 5, energy=100)  # type: ignore[arg-type]

    def test_rejects_mutable_grids_and_rows(self) -> None:
        with self.assertRaises(TypeError):
            Observation(cells=list(EMPTY_CELLS), energy=100)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            Observation(cells=(list(EMPTY_ROW),) * 5, energy=100)  # type: ignore[arg-type]

    def test_rejects_non_cell_values(self) -> None:
        with self.assertRaises(TypeError):
            Observation(cells=(("empty",) * 5,) * 5, energy=100)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
